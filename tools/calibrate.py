"""Fit the recognition thresholds to your hands.

The defaults were tuned against a synthetic hand model, which gets the shape of
each threshold right but cannot know how far your thumb actually abducts or how
wide your fingers fan. This walks through a handful of reference poses, records
what your hand really measures, and writes calibration.json with a boundary
placed between each pair of measured distributions.

    python tools/calibrate.py            # run the guided capture
    python tools/calibrate.py --show     # print the current calibration
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vision_paint.camera import Camera, CameraError
from vision_paint.config import CALIBRATION_PATH, MODEL_PATH, AppConfig
from vision_paint.features import HandFeatures, extract
from vision_paint.hand_tracker import HandTracker
from vision_paint.ui import _panel, _text

#: Poses to capture, in the order they are asked for.
PROMPTS: Sequence[Tuple[str, str]] = (
    ("open_palm", "Open hand, fingers spread wide"),
    ("palm_flat", "Flat palm facing the camera, fingers together"),
    ("fist", "Closed fist"),
    ("draw", "Index finger up, thumb tucked in"),
    ("point_up", "Index finger up, thumb sticking out"),
    ("pinch", "Pinch: thumb and index tips touching"),
    ("two_fingers", "Peace sign, fingers apart"),
    ("crossed_fingers", "Cross index and middle fingers"),
)

#: Each threshold is a boundary between one pose's measurement and another's.
#: (config key, feature, pose that should read low, pose that should read high)
BOUNDARIES: Sequence[Tuple[str, Callable[[HandFeatures], float], str, str]] = (
    ("finger_extended_max_curl", lambda f: max(f.curls.values()), "open_palm", "fist"),
    ("thumb_abduction_min", lambda f: f.thumb_abduction, "draw", "point_up"),
    ("pinch_close_max", lambda f: f.pinch, "pinch", "two_fingers"),
    ("fist_index_reach_max", lambda f: f.index_reach, "fist", "pinch"),
    ("palm_spread_flat_max", lambda f: f.spread, "palm_flat", "open_palm"),
    ("crossed_tip_max", lambda f: f.index_middle_tip_gap, "crossed_fingers", "two_fingers"),
)


def boundary(low: np.ndarray, high: np.ndarray) -> float:
    """Split point between two measured distributions.

    Weighted by each side's spread, so a tight cluster holds its ground against
    a loose one instead of the boundary landing halfway regardless.
    """
    ml, mh = float(np.median(low)), float(np.median(high))
    sl = max(float(np.std(low)), 1e-3)
    sh = max(float(np.std(high)), 1e-3)
    if mh <= ml:
        return (ml + mh) / 2.0
    return (ml * sh + mh * sl) / (sl + sh)


def capture(cfg: AppConfig, seconds: float, samples_needed: int) -> Dict[str, List[HandFeatures]]:
    recorded: Dict[str, List[HandFeatures]] = {}
    with Camera(cfg.camera) as cam, HandTracker(cfg.tracker, MODEL_PATH, cfg.camera.mirror) as tracker:
        window = "Calibration"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        for pose, instruction in PROMPTS:
            samples: List[HandFeatures] = []
            started = None
            while True:
                frame, stamp = cam.read()
                if frame is None:
                    continue
                frame = frame.copy()
                h, w = frame.shape[:2]
                hands = tracker.process(frame, stamp)
                hand = None
                if hands:
                    hand = extract(hands[0], cfg.gesture, w / h)

                if started is None and hand is not None:
                    started = time.time()
                elapsed = 0.0 if started is None else time.time() - started
                if hand is not None and started is not None and elapsed > 0.6:
                    samples.append(hand)

                _panel(frame, 0, 0, w, 96, 0.72)
                _text(frame, instruction, 20, 40, 0.8, (255, 255, 255), 2)
                status = (
                    "show your hand to start"
                    if hand is None and not samples
                    else f"hold still - {len(samples)}/{samples_needed}"
                )
                _text(frame, status, 20, 74, 0.6, (255, 190, 90))
                cv2.rectangle(frame, (0, 92), (int(w * min(1.0, len(samples) / samples_needed)), 96),
                              (120, 220, 140), -1)
                cv2.imshow(window, frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    cv2.destroyAllWindows()
                    raise KeyboardInterrupt
                if key == ord("s"):
                    break
                if len(samples) >= samples_needed or elapsed > seconds + 3:
                    break
            recorded[pose] = samples
            del instruction
        cv2.destroyAllWindows()
    return recorded


def fit(recorded: Dict[str, List[HandFeatures]], cfg: AppConfig) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key, getter, low_pose, high_pose in BOUNDARIES:
        low = recorded.get(low_pose) or []
        high = recorded.get(high_pose) or []
        if len(low) < 8 or len(high) < 8:
            print(f"  skipping {key}: not enough samples")
            continue
        lo = np.array([getter(f) for f in low])
        hi = np.array([getter(f) for f in high])
        value = boundary(lo, hi)
        out[key] = round(value, 4)
        print(f"  {key:28s} {value:7.3f}   ({low_pose} {np.median(lo):.2f} | "
              f"{high_pose} {np.median(hi):.2f})")

    # Derived partners, kept in the same relationship the defaults use.
    if "finger_extended_max_curl" in out:
        out["finger_folded_min_curl"] = round(out["finger_extended_max_curl"] * 1.8, 4)
    if "pinch_close_max" in out:
        out["pinch_open_min"] = round(out["pinch_close_max"] * 1.45, 4)
    if "palm_spread_flat_max" in out:
        out["palm_spread_wide_min"] = round(out["palm_spread_flat_max"] * 1.2, 4)
    del cfg
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--samples", type=int, default=45, help="frames to record per pose")
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--out", type=Path, default=CALIBRATION_PATH)
    ap.add_argument("--show", action="store_true", help="print the saved calibration and exit")
    args = ap.parse_args()

    if args.show:
        if not args.out.exists():
            print(f"no calibration at {args.out}")
            return 1
        print(args.out.read_text())
        return 0

    cfg = AppConfig()
    cfg.camera.index = args.camera
    print("Hold each pose still until its bar fills. 's' skips a pose, 'q' aborts.\n")
    try:
        recorded = capture(cfg, args.seconds, args.samples)
    except CameraError as e:
        print(f"\n{e}\n")
        return 2
    except KeyboardInterrupt:
        print("\naborted, nothing written")
        return 130

    print("\nfitted thresholds:")
    values = fit(recorded, cfg)
    if not values:
        print("nothing usable was captured; calibration not written")
        return 1

    payload = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "samples": {pose: len(s) for pose, s in recorded.items()},
        "gesture": values,
    }
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {args.out} - it is picked up automatically on the next run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
