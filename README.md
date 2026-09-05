# Vision Paint

Draw on your webcam feed with your hands. A camera watches your hand, a
geometric classifier reads the pose, and a temporal filter turns that into
drawing commands — no training data, no model to fine-tune, every threshold
inspectable and adjustable.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
./scripts/fetch_model.sh          # one-time, ~7.5 MB hand landmark model
python run.py                      # 'h' shows the gesture list, 'q' quits
```

**macOS:** the app needs camera permission for whatever launches it (Terminal,
iTerm, your IDE). Grant it under System Settings → Privacy & Security → Camera,
then restart that app. Without it OpenCV reports `camera failed to properly
initialize` and the app exits with a message saying so.

## Gestures

| Gesture | Action |
| --- | --- |
| ☝️ Index finger | Draw |
| ✌️ Two fingers, held still | Change colour |
| ✊ Closed fist | Erase |
| 🤏 Pinch, held still | Select tool (brush / line / rectangle / circle) |
| 🖐️ Open palm, fingers spread | Clear screen *(hold 1.4 s)* |
| 👍 Thumbs up | Save drawing |
| 👎 Thumbs down | Delete last save *(hold 1.4 s)* |
| 🤞 Crossed fingers | Undo |
| ✋ Flat palm, fingers together | Pause / resume drawing |
| 👉 Point right | Next colour |
| 👈 Point left | Previous colour |
| 👆 Point up | Increase brush size |
| 👇 Point down | Decrease brush size |
| 🤟 Three fingers | Change brush style |
| 🖖 Four fingers | Change background |
| 👌 OK sign | Confirm selection |
| 🤙 Thumb + little finger | Open / close menu |
| ✌️ + swipe left/right | Change canvas |
| 🤏 + open/close the pinch | Zoom in / out |
| 🙏 Both palms together | Exit *(hold 1.4 s)* |

Keyboard shortcuts remain available: `h` help, `d` gesture score panel,
`l` landmarks, `z` undo, `c` clear, `s` save, `r` reset view, `q` quit.

### Three ambiguities in that list, and how they are resolved

The requested vocabulary contains gestures a camera cannot tell apart from
shape alone. Each is separated by a second signal rather than by luck:

1. **☝️ draw vs 👆 point up.** Both are an index finger raised. The thumb
   decides: tucked in is drawing, held out (an "L" shape) is a directional
   command. The same rule covers 👉 👈 👇, so all four directions work the same
   way and the direction itself comes from where the index finger points.
2. **🖐️ clear screen vs ✋ pause.** Both are five fingers up. Finger spread
   decides: fanned out clears, held together pauses. Clearing also needs a
   1.4-second hold, so a misread cannot wipe a drawing instantly.
3. **✌️ colour vs ✌️ + move, 🤏 tool vs 🤏 + move.** Movement decides, and it
   gets first refusal: if the hand swipes or the pinch opens, the moving
   command fires and the still command is suppressed for the rest of that
   gesture press. One hand movement never triggers two commands.

## How recognition works

```
camera ─▶ hand landmarks ─▶ geometric features ─▶ scored rules ─▶ voting window ─▶ event gate ─▶ canvas
         (MediaPipe +        (curl, spread,        (18 gestures,    (5 of 7 frames)  (cooldown +
          One Euro filter)     reach, direction)     0..1 each)                        dwell)
```

Four decisions carry most of the accuracy:

**Curl is measured across long baselines.** Finger flexion comes from the angle
between the proximal phalanx and the PIP-to-tip chord, not from summing
per-joint angles. The distal segments are only ~20 mm long, so per-joint angles
swing by 10° or more under the landmark noise MediaPipe actually produces —
enough to make the little finger read as curled on an open hand. Switching to
the chord form cut the 95th-percentile error on that measurement from 94° to
34° and took overall accuracy from 83% to 98%.

**Every measurement is normalised by hand size**, so the same thresholds hold
whether your hand fills the frame or sits in a corner, and for small and large
hands alike. Angles are rotation-invariant by construction, so a tilted wrist
does not change the reading.

**Rules score, they do not vote yes/no.** Each predicate returns 0..1 based on
how far the measurement sits from its threshold, and a gesture's confidence is
the geometric mean of its predicates blended with its weakest one. A plain mean
lets five satisfied predicates average away one badly violated one — which is
exactly how a pointing hand gets read as a drawing hand. Blending in the
minimum keeps near-misses ranked below clean matches while still degrading
smoothly rather than snapping to zero.

**Nothing fires on a single frame.** A gesture must win 5 of the last 7 frames
before it counts, one-shot commands go on cooldown so holding a pose does not
repeat it, and destructive commands need a visible 1.4-second hold. Deleting a
saved drawing moves the file to `saves/.trash/` rather than unlinking it.

### Measured accuracy

`tools/sweep.py` generates perturbed synthetic hands — imperfect curls, rotated
wrists, palms tilted away from the camera, either hand, landmark noise — and
scores the classifier over them.

| Condition | Correct | Rejected | Misfired |
| --- | --- | --- | --- |
| Per frame, moderate pose jitter | 98.4% | 0.4% | 1.1% |
| Per frame, heavy pose jitter | 92.0% | 4.2% | 3.8% |
| Held gesture through the stabiliser, heavy jitter | 99.1% | 0.7% | **0.1%** |

Rejections and misfires are counted separately on purpose: a rejection costs
you a repeated gesture, a misfire costs you an unwanted edit. The stabiliser
exists to turn the second kind into the first.

```bash
python tools/sweep.py --samples 400 --strength 0.5      # per-frame
python tools/sweep.py --stream --samples 120            # end to end
```

These numbers come from a synthetic hand model, not from footage of real hands.
They validate the geometry and catch regressions; they are not a substitute for
running it on your own hand, which is what calibration is for.

### Calibration

The defaults were fitted against a synthetic hand. Your thumb abducts a
particular amount and your fingers fan a particular width, and a few minutes of
calibration measures both:

```bash
python tools/calibrate.py        # guided capture, writes calibration.json
python tools/calibrate.py --show
python run.py --no-calibration   # ignore it and use the built-in defaults
```

It records eight reference poses and places each threshold on the boundary
between the two measured distributions, weighted by their spread so a tight
cluster is not dragged around by a loose one. `calibration.json` is picked up
automatically at startup; only keys that exist on `GestureConfig` are read from
it.

## Performance

At 1280×720 on an M-series Mac: ~13 ms for MediaPipe inference (worst case,
with the palm detector running every frame), ~8 ms for everything else —
features, dispatch, canvas render, composite and HUD, with 60 strokes on the
canvas. That is a ~47 fps ceiling against a 30 fps camera, so the pipeline is
not the bottleneck. Camera capture runs on its own thread so the main loop
never blocks on I/O.

## Layout

```
run.py                    entry point
vision_paint/
  config.py               every threshold, in one place
  camera.py               threaded capture
  hand_tracker.py         MediaPipe wrapper + One Euro smoothing
  filters.py              One Euro filter
  features.py             geometric features from landmarks
  gestures.py             scored classification rules
  stabilizer.py           voting window + event gate
  motion.py               swipe and pinch-rate detection
  canvas.py               vector strokes, styles, undo, zoom, save
  state.py                brush, menu and canvas state
  commands.py             gesture -> action bindings
  ui.py                   HUD, menu, help overlay
  app.py                  main loop
  synthetic.py            synthetic hand generator (tests and tools)
tools/
  sweep.py                accuracy sweep
  calibrate.py            guided threshold fitting
  headless_demo.py        render a frame with no camera
tests/                    157 tests, no camera required
```

## Design notes

**Strokes are vectors, not pixels.** Undo would otherwise mean holding dozens
of full-resolution frame snapshots (~150 MB at the configured depth); zoom
would magnify pixels instead of redrawing; and saving would be locked to
whatever resolution the camera happened to be running at. Committed strokes are
cached as a raster and only the stroke under your finger is redrawn each frame.

**Translucent brush styles blend rather than overwrite.** OpenCV draws by
overwriting, alpha channel included, so a marker stroke drawn naively punches a
semi-transparent hole through whatever it crosses. The marker and neon styles
paint into a coverage mask clipped to the stroke's bounding box and blend that
in.

**MediaPipe 1.x is not usable here.** It removed the legacy solutions API and
its macOS arm64 build crashes on the Tasks hand landmarker (`Check failed:
service_ Service is unavailable` from the Metal helper), on CPU delegate too.
`requirements.txt` pins 0.10.21.

## Testing

```bash
python -m pytest tests/ -q
```

157 tests, none of which need a camera: feature invariants (scale, rotation,
noise), classification of all 18 gestures on both hands at several distances,
the separations that matter most (draw vs point, fist vs pinch, clear vs
pause), stabiliser and dwell behaviour, calibration fitting, and end-to-end
tests that drive synthetic hands through the real pipeline and assert on what
lands on the canvas.
