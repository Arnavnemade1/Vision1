"""Accuracy sweep over perturbed synthetic hands.

Generates many jittered variants of each canonical pose -- imperfect curls,
rotated wrists, palms tilted away from the camera, either hand, landmark noise
-- and reports how often the classifier still returns the right gesture. Run it
after touching any threshold in config.py.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from copy import deepcopy
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from vision_paint.config import GestureConfig
from vision_paint.features import extract
from vision_paint.gestures import GestureClassifier
from vision_paint.synthetic import POSES, PoseSpec, build_frame


#: Mirroring a pose swaps these, because left and right are properties of the
#: image, not of the hand: a left hand pointing right mirrors a right hand
#: pointing left.
MIRRORED_NAME = {"point_left": "point_right", "point_right": "point_left"}


def jitter(spec: PoseSpec, name: str, rng: np.random.Generator,
           strength: float = 1.0) -> Tuple[PoseSpec, str]:
    s = deepcopy(spec)
    s.curls = {
        k: float(np.clip(v + rng.normal(0, 14 * strength), 0, 175)) for k, v in s.curls.items()
    }
    s.thumb_curl = float(np.clip(s.thumb_curl + rng.normal(0, 10 * strength), 0, 90))
    s.thumb_out = float(np.clip(s.thumb_out + rng.normal(0, 0.13 * strength), 0, 1))
    s.fan = float(max(0.0, s.fan + rng.normal(0, 0.28 * strength)))
    s.roll_deg = s.roll_deg + rng.normal(0, 16 * strength)
    s.pitch_deg = s.pitch_deg + rng.normal(0, 20 * strength)
    if rng.random() < 0.5:
        s.geometric_label = "Left" if s.geometric_label == "Right" else "Right"
        name = MIRRORED_NAME.get(name, name)
    return s, name


def run(samples: int, strength: float, noise: float, seed: int, verbose: bool,
        min_conf: float = 0.55) -> int:
    rng = np.random.default_rng(seed)
    cfg = GestureConfig()
    clf = GestureClassifier(cfg)

    hits: Dict[str, int] = defaultdict(int)
    rejects: Dict[str, int] = defaultdict(int)
    total: Dict[str, int] = defaultdict(int)
    confusions: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for name, spec in POSES.items():
        for _ in range(samples):
            s, expected = jitter(spec, name, rng, strength)
            frame = build_frame(s, noise=noise, rng=rng)
            result = clf.classify(extract(frame, cfg), min_confidence=min_conf)
            total[name] += 1
            if result.gesture.value == expected:
                hits[name] += 1
            elif result.gesture.value == "none":
                # A rejection costs the user a repeated gesture; a misfire costs
                # them an unwanted edit. They are not the same failure.
                rejects[name] += 1
            else:
                confusions[name][result.gesture.value] += 1

    order = sorted(POSES, key=lambda n: hits[n] / total[n])
    print(f"{'gesture':16s} {'correct':>8s} {'reject':>7s} {'misfire':>8s}   top confusions")
    for name in order:
        n = total[name]
        conf = sorted(confusions[name].items(), key=lambda kv: -kv[1])[:3]
        detail = ", ".join(f"{g}:{c}" for g, c in conf)
        print(f"{name:16s} {hits[name] / n:8.1%} {rejects[name] / n:7.1%} "
              f"{sum(confusions[name].values()) / n:8.1%}   {detail}")

    n_all = sum(total.values())
    overall = sum(hits.values()) / n_all
    misfire = sum(sum(c.values()) for c in confusions.values()) / n_all
    print(f"\noverall: {overall:.1%} correct, {sum(rejects.values()) / n_all:.1%} rejected, "
          f"{misfire:.1%} misfired over {n_all} samples "
          f"(jitter x{strength}, noise {noise * 1000:.0f} mm, min conf {min_conf})")
    if verbose:
        for name in order:
            print(name, dict(confusions[name]))
    return 0 if overall > 0.0 else 1


def run_stream(streams: int, strength: float, noise: float, seed: int, frames: int,
               min_conf: float) -> int:
    """End-to-end check: hold each gesture for `frames` frames and see what the
    stabiliser settles on. This is the number that matters in use -- the
    per-frame figures ignore the voting window that sits in front of the canvas.
    """
    from vision_paint.config import StabilizerConfig
    from vision_paint.stabilizer import GestureStabilizer

    rng = np.random.default_rng(seed)
    cfg = GestureConfig()
    clf = GestureClassifier(cfg)
    scfg = StabilizerConfig()

    settled: Dict[str, int] = defaultdict(int)
    wrong: Dict[str, int] = defaultdict(int)
    silent: Dict[str, int] = defaultdict(int)

    for name, spec in POSES.items():
        for _ in range(streams):
            stab = GestureStabilizer(scfg)
            base, expected = jitter(spec, name, rng, strength * 0.6)
            saw_right = False
            saw_wrong: List[str] = []
            t = 0.0
            for _f in range(frames):
                t += 1 / 30
                # Re-jitter lightly each frame: the hand is held, not frozen.
                s, _ = jitter(base, expected, rng, strength * 0.35)
                s.geometric_label = base.geometric_label
                result = clf.classify(extract(build_frame(s, noise=noise, rng=rng), cfg), min_conf)
                state = stab.update(result.gesture, result.confidence, t)
                if state.changed and state.active:
                    if state.gesture.value == expected:
                        saw_right = True
                    else:
                        saw_wrong.append(state.gesture.value)
            if saw_wrong:
                wrong[name] += 1
            elif saw_right:
                settled[name] += 1
            else:
                silent[name] += 1

    print(f"{'gesture':16s} {'settled':>8s} {'silent':>7s} {'wrong':>7s}")
    order = sorted(POSES, key=lambda n: settled[n])
    for name in order:
        n = streams
        print(f"{name:16s} {settled[name] / n:8.1%} {silent[name] / n:7.1%} {wrong[name] / n:7.1%}")
    total = len(POSES) * streams
    print(f"\nstabilised: {sum(settled.values()) / total:.1%} settled on the right gesture, "
          f"{sum(silent.values()) / total:.1%} never settled, "
          f"{sum(wrong.values()) / total:.1%} settled on something wrong "
          f"({frames}-frame holds, jitter x{strength}, noise {noise * 1000:.0f} mm)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=400)
    ap.add_argument("--strength", type=float, default=1.0, help="jitter multiplier")
    ap.add_argument("--noise", type=float, default=0.0025, help="landmark noise, metres")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--min-conf", type=float, default=0.55)
    ap.add_argument("--stream", action="store_true",
                    help="evaluate held gestures through the temporal stabiliser")
    ap.add_argument("--frames", type=int, default=20, help="frames per held gesture")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    if a.stream:
        return run_stream(a.samples, a.strength, a.noise, a.seed, a.frames, a.min_conf)
    return run(a.samples, a.strength, a.noise, a.seed, a.verbose, a.min_conf)


if __name__ == "__main__":
    raise SystemExit(main())
