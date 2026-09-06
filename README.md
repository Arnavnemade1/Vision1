# Vision Paint

A gesture-driven whiteboard. A camera watches your hand, a geometric classifier
reads the pose, and a temporal filter turns that into drawing commands — no
training data, no model to fine-tune, every threshold inspectable and
adjustable.

The board is larger than the window, and drawings are objects rather than
pixels: pinch one, move it aside, and draw another beside it.

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
| 🤏 Pinch **on a drawing**, then move | Pick that drawing up and move it |
| 🤏 Pinch **empty space**, then move | Pan the board |
| 🤏 Pinch, held still | Select tool (brush / line / rectangle / circle) |
| 🖐️ Open palm, fingers spread | Clear the board *(hold 1.4 s)* |
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
| ✌️ + swipe left/right | Change board |
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
3. **✌️ colour vs ✌️ + move, 🤏 tool vs 🤏 + grab / pan / zoom.** Movement
   decides, and it gets first refusal. A pinch commits to one of four readings
   within a fraction of a second and holds it until the hand opens, so once you
   have hold of a drawing, wobbling your fingers cannot turn the drag into a
   zoom. Pausing mid-pinch does *not* disqualify a grab — pinching, thinking,
   then dragging is how people actually pick things up.

## Moving drawings around

The board is twice the width and height of the window. Strokes that sit near
each other are treated as one drawing; strokes elsewhere on the board are a
different one. Pinch over a drawing and the group you would pick up is outlined
on screen before you commit to it; drag, and it comes with you. Pinch where
there is nothing and you pan the board instead.

A minimap in the corner shows where the window sits on the board and where the
drawings are, so nothing can be panned out of sight and lost. Moves go onto the
same undo stack as strokes, so 🤞 walks back through moves and marks alike.

Grouping is proximity-based, which is a blunt rule that happens to match how
people draw: the strokes of one doodle overlap or nearly touch, and a doodle
drawn elsewhere does not. `group_gap_ratio` in `config.py` sets how close is
close enough, as a fraction of window width rather than a pixel count, so it
behaves the same at any resolution.

## How recognition works

```
camera ─▶ hand landmarks ─▶ geometric features ─▶ scored rules ─▶ voting window ─▶ event gate ─▶ board
         (MediaPipe +        (curl, spread,        (18 gestures,    (weighted,       (cooldown +
          One Euro filter)     reach, direction,     0..1 each)       5 of 7 frames)   dwell)
                               exponentially
                               smoothed)
```

Six decisions carry most of the accuracy:

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

**The measurements are smoothed, not just the conclusions.** The voting window
rejects bad frames by throwing whole classifications away. Exponentially
smoothing the underlying measurements first (75 ms time constant, per hand,
reset when a hand leaves) means a noisy frame nudges the answer instead of
contradicting it. Under 5 mm of landmark noise this takes end-to-end accuracy
from 94.8% to 97.5% and cuts wrong commands from 2.9% to 1.9% — and because it
works upstream, the voting window can stay short, which is what keeps the
system feeling responsive. `--no-smoothing` on the sweep reproduces the A/B.

**A hand half out of frame is not trusted.** Landmarks MediaPipe has had to
extrapolate past the frame edge are guesses, so confidence is scaled by the
fraction of the hand comfortably inside the image before any threshold sees it.

**Nothing fires on a single frame.** A gesture must win 5 of the last 7 frames
before it counts, with votes weighted by confidence rather than counted — four
frames of a clean read should outrank five frames of a marginal one, and plain
counting gets that backwards. One-shot commands go on cooldown so holding a
pose does not repeat it, and destructive commands need a visible 1.4-second
hold. Deleting a saved drawing moves the file to `saves/.trash/` rather than
unlinking it.

### Measured accuracy

`tools/sweep.py` generates perturbed synthetic hands — imperfect curls, rotated
wrists, palms tilted away from the camera, either hand, landmark noise — and
scores the classifier over them.

| Condition | Correct | Rejected | Misfired |
| --- | --- | --- | --- |
| Per frame, moderate pose jitter | 98.4% | 0.4% | 1.1% |
| Per frame, heavy pose jitter | 92.0% | 4.2% | 3.8% |
| Held gesture through the stabiliser, heavy jitter | 99.3% | 0.6% | **0.2%** |
| Held gesture, 5 mm landmark noise, no feature smoothing | 94.8% | 2.3% | 2.9% |
| Held gesture, 5 mm landmark noise, feature smoothing on | 97.5% | 0.6% | 1.9% |

Rejections and misfires are counted separately on purpose: a rejection costs
you a repeated gesture, a misfire costs you an unwanted edit. The stabiliser
exists to turn the second kind into the first.

```bash
python tools/sweep.py --samples 400 --strength 0.5      # per-frame
python tools/sweep.py --stream --samples 120            # end to end
python tools/sweep.py --stream --no-smoothing           # A/B the smoother
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
with the palm detector running every frame), and for everything else —
features, dispatch, board render, composite and HUD — **0.5 ms** with a drawing
in one part of the board, rising to **4.3 ms** in the pathological case of ink
covering all four times the window area. Camera capture runs on its own thread
so the main loop never blocks on I/O.

The board being four times the window area could easily have cost four times
the frame budget. Three things stop it:

- **The viewport is a slice, not a resample.** At the default view the window
  is a whole-pixel window into the board raster, so cropping to it is a memory
  slice. Zoom and fractional pan fall back to a warp, which is output-driven
  and therefore still viewport-sized.
- **Live strokes never touch the board raster.** The stroke under your finger
  and any drawing being dragged are rasterised straight into view space, so
  drawing and dragging cost one small overlay redraw instead of copying a
  14 MB board every frame. A drag also lifts its strokes out of the cache, so
  moving a drawing never invalidates it.
- **Only the rectangle containing ink is blended,** through cv2 integer
  arithmetic rather than float32 numpy — bit-identical output at a little over
  half the time, and the blend was the single largest cost in the frame.

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
  canvas.py               board: vector strokes, groups, moves, undo, save
  state.py                brush, menu and board state
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

## Smoothness

Three separate smoothers, because they want different settings:

- **Landmarks** get a One Euro filter tuned for responsiveness — the pose
  recogniser needs to see a gesture form quickly.
- **The pen** gets its own, much gentler One Euro filter. Ink records every
  tremor the recogniser is happy to average away, so the drawing point is
  filtered harder than the hand that produced it.
- **Committed strokes** are rendered through two passes of Chaikin
  corner-cutting. Freehand input arrives as a polyline of hand samples and
  looks like one; corner-cutting both removes the jitter and quadruples the
  sample density, so a stroke draws as a curve rather than a chain of segments.

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
