"""Rule-based gesture classification.

Each gesture is a small set of geometric predicates. Instead of returning a
hard yes/no, every predicate returns a 0..1 score based on how far the measured
value sits from its threshold, and a gesture's confidence is the geometric mean
of its predicates. That gives three things a boolean rule set cannot:

  * a usable confidence number for the temporal stabiliser,
  * graceful behaviour near a threshold instead of frame-to-frame flicker,
  * a ranked runner-up list, which makes debugging misfires straightforward.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np

from .config import GestureConfig
from .features import HandFeatures


class Gesture(str, Enum):
    NONE = "none"
    DRAW = "draw"                      # index finger up, thumb tucked
    TWO_FINGERS = "two_fingers"        # peace sign
    FIST = "fist"
    PINCH = "pinch"
    OPEN_PALM = "open_palm"            # five fingers, spread wide
    PALM_FLAT = "palm_flat"            # five fingers, held together, facing camera
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    CROSSED_FINGERS = "crossed_fingers"
    POINT_RIGHT = "point_right"
    POINT_LEFT = "point_left"
    POINT_UP = "point_up"
    POINT_DOWN = "point_down"
    THREE_FINGERS = "three_fingers"
    FOUR_FINGERS = "four_fingers"
    OK_SIGN = "ok_sign"
    CALL_ME = "call_me"                # thumb + little finger
    PRAYER = "prayer"                  # both hands together (two-hand gesture)


#: Gestures that only make sense while a single hand is visible and still.
DIRECTIONAL = (
    Gesture.POINT_RIGHT,
    Gesture.POINT_LEFT,
    Gesture.POINT_UP,
    Gesture.POINT_DOWN,
)


@dataclass
class Classification:
    gesture: Gesture
    confidence: float
    ranking: List[Tuple[Gesture, float]]

    @property
    def runner_up(self) -> Optional[Tuple[Gesture, float]]:
        return self.ranking[1] if len(self.ranking) > 1 else None


def _ramp(value: float, one_at: float, zero_at: float) -> float:
    """Linear score: 1.0 at `one_at`, 0.0 at `zero_at`, clamped between."""
    if one_at == zero_at:
        return 1.0 if value == one_at else 0.0
    t = (value - zero_at) / (one_at - zero_at)
    return float(min(1.0, max(0.0, t)))


def _geo_mean(scores: List[float]) -> float:
    if not scores:
        return 0.0
    if min(scores) <= 0.0:
        return 0.0
    return float(np.exp(np.mean(np.log(scores))))


def _combine(scores: List[float]) -> float:
    """Confidence for a rule from its predicate scores.

    A plain geometric mean is too forgiving -- one badly violated predicate gets
    averaged away by five satisfied ones, which is exactly how a pointing pose
    gets mistaken for a drawing pose. Blending the mean with the weakest
    predicate keeps near-misses ranked below clean matches while still degrading
    smoothly instead of snapping to zero.
    """
    if not scores:
        return 0.0
    weakest = min(scores)
    if weakest <= 0.0:
        return 0.0
    return float(np.sqrt(_geo_mean(scores) * weakest))


class GestureClassifier:
    def __init__(self, cfg: GestureConfig):
        self.cfg = cfg

    # ---- primitive scores -------------------------------------------------

    def _ext(self, f: HandFeatures, name: str) -> float:
        """How confidently finger `name` is extended."""
        return _ramp(f.curls[name], self.cfg.finger_extended_max_curl, self.cfg.finger_folded_min_curl)

    def _fold(self, f: HandFeatures, name: str) -> float:
        return 1.0 - self._ext(f, name)

    def _thumb_out(self, f: HandFeatures) -> float:
        curl = _ramp(f.thumb_curl, self.cfg.thumb_extended_max_curl, self.cfg.thumb_extended_max_curl * 2.2)
        abduct = _ramp(f.thumb_abduction, self.cfg.thumb_abduction_min + 0.15, self.cfg.thumb_abduction_min - 0.11)
        return min(curl, abduct)

    def _thumb_in(self, f: HandFeatures) -> float:
        return 1.0 - self._thumb_out(f)

    def _pinched(self, f: HandFeatures) -> float:
        return _ramp(f.pinch, self.cfg.pinch_close_max * 0.6, self.cfg.pinch_open_min)

    def _direction_score(self, vec: np.ndarray, target: np.ndarray,
                         cone: Optional[float] = None) -> float:
        if not vec.any():
            return 0.0
        cos = float(np.clip(np.dot(vec / (np.linalg.norm(vec) + 1e-9), target), -1.0, 1.0))
        angle = float(np.degrees(np.arccos(cos)))
        return _ramp(angle, 0.0, cone if cone is not None else self.cfg.point_cone_half_angle)

    # ---- gesture rules ----------------------------------------------------

    def score_all(self, f: HandFeatures) -> List[Tuple[Gesture, float]]:
        c = self.cfg
        idx, mid, rng, pky = (self._ext(f, n) for n in ("index", "middle", "ring", "pinky"))
        nidx, nmid, nrng, npky = 1 - idx, 1 - mid, 1 - rng, 1 - pky
        t_out, t_in = self._thumb_out(f), self._thumb_in(f)
        pinched = self._pinched(f)
        open_fingers = 1.0 - pinched

        up = np.array([0.0, 1.0])
        down = np.array([0.0, -1.0])
        right = np.array([1.0, 0.0])
        left = np.array([-1.0, 0.0])

        scores: List[Tuple[Gesture, float]] = []

        def add(g: Gesture, parts: List[float]) -> None:
            scores.append((g, _combine(parts)))

        # Index only, thumb tucked -> drawing.
        add(Gesture.DRAW, [idx, nmid, nrng, npky, t_in, open_fingers])

        # Index + thumb out ("L" shape) -> a directional command; which one is
        # decided by where the index finger points. Keeping the thumb out is
        # what separates a deliberate "point" from the drawing pose.
        point_base = [idx, nmid, nrng, npky, t_out]
        add(Gesture.POINT_UP, point_base + [self._direction_score(f.index_dir, up)])
        add(Gesture.POINT_DOWN, point_base + [self._direction_score(f.index_dir, down)])
        add(Gesture.POINT_RIGHT, point_base + [self._direction_score(f.index_dir, right)])
        add(Gesture.POINT_LEFT, point_base + [self._direction_score(f.index_dir, left)])

        # Peace sign: index + middle up and clearly apart.
        apart = _ramp(f.index_middle_tip_gap, c.crossed_tip_max * 1.5, c.crossed_tip_max)
        add(Gesture.TWO_FINGERS, [idx, mid, nrng, npky, apart])

        # Crossed fingers: same two fingers, tips together or overlapping.
        together = _ramp(f.index_middle_tip_gap, c.crossed_tip_max * 0.6, c.crossed_tip_max * 1.5)
        add(Gesture.CROSSED_FINGERS, [idx, mid, nrng, npky, together])

        # Fist: everything curled tight, thumb wrapped in, index rolled into
        # the palm rather than held out in front of it.
        rolled_in = _ramp(f.index_reach, c.fist_index_reach_max - 0.11, c.fist_index_reach_max + 0.13)
        add(Gesture.FIST, [nidx, nmid, nrng, npky, t_in, rolled_in])

        # Pinch: thumb and index tips meeting out in front of the palm, with the
        # remaining fingers curled away.
        held_out = _ramp(f.index_reach, c.fist_index_reach_max + 0.13, c.fist_index_reach_max - 0.11)
        add(Gesture.PINCH, [pinched, nmid, nrng, npky, held_out])

        # OK: same thumb/index ring, but the other three stay up.
        ring_closed = _ramp(f.pinch, c.ok_ring_max * 0.6, c.ok_ring_max * 1.5)
        add(Gesture.OK_SIGN, [ring_closed, mid, rng, pky])

        # Thumbs up / down: fingers curled, thumb out and vertical.
        curled = [nidx, nmid, nrng, npky]
        cone = c.thumb_cone_half_angle
        add(Gesture.THUMBS_UP, curled + [t_out, self._direction_score(f.thumb_dir, up, cone)])
        add(Gesture.THUMBS_DOWN, curled + [t_out, self._direction_score(f.thumb_dir, down, cone)])

        # Three fingers: the ILY sign (thumb + index + little) or the plain
        # index/middle/ring trio - people reach for either, so accept both.
        ily = _combine([t_out, idx, nmid, nrng, pky])
        trio = _combine([idx, mid, rng, npky, t_in])
        scores.append((Gesture.THREE_FINGERS, max(ily, trio)))

        # Four fingers up, thumb folded across the palm.
        add(Gesture.FOUR_FINGERS, [idx, mid, rng, pky, t_in])

        # Call-me: thumb and little finger out, middle three curled.
        add(Gesture.CALL_ME, [t_out, nidx, nmid, nrng, pky])

        # Five fingers open. Spread separates "clear the screen" (fingers fanned
        # out) from "pause" (fingers held together like a stop sign).
        five = [idx, mid, rng, pky, t_out]
        wide = _ramp(f.spread, c.palm_spread_wide_min * 1.12, c.palm_spread_flat_max)
        flat = _ramp(f.spread, c.palm_spread_flat_max * 0.85, c.palm_spread_wide_min)
        facing = _ramp(f.palm_facing, 0.65, 0.05)
        add(Gesture.OPEN_PALM, five + [wide])
        add(Gesture.PALM_FLAT, five + [flat, max(facing, 0.35)])

        scores.sort(key=lambda kv: kv[1], reverse=True)
        return scores

    def classify(self, f: HandFeatures, min_confidence: float = 0.5) -> Classification:
        ranking = self.score_all(f)
        best, score = ranking[0]
        if score < min_confidence:
            return Classification(Gesture.NONE, score, ranking)
        return Classification(best, score, ranking)


def classify_two_hands(
    a: HandFeatures, b: HandFeatures, cfg: GestureConfig
) -> Tuple[Gesture, float]:
    """Palms pressed together — the exit gesture.

    Judged from image-space geometry: the two palms sit close relative to hand
    size, and both hands point roughly the same way (fingers up, edge-on to the
    camera) rather than one hand simply passing in front of the other.
    """
    gap = float(np.linalg.norm(a.palm_center - b.palm_center))
    scale = 0.5 * (a.hand_size + b.hand_size)
    # Palm centres are image-normalised while hand_size is metric, so compare
    # against the on-screen span of the index knuckles instead.
    span = max(
        float(np.linalg.norm(a.raw.landmarks[5, :2] - a.raw.landmarks[17, :2])),
        float(np.linalg.norm(b.raw.landmarks[5, :2] - b.raw.landmarks[17, :2])),
        1e-6,
    )
    closeness = _ramp(gap / span, 0.85, 2.1)
    aligned = _ramp(float(np.dot(a.hand_axis, b.hand_axis)), 0.85, 0.35)
    upright = _combine([_ramp(float(a.hand_axis[1]), 0.85, 0.2), _ramp(float(b.hand_axis[1]), 0.85, 0.2)])
    fingers_out = _combine(
        [
            _ramp(a.curls["middle"], cfg.finger_extended_max_curl, cfg.finger_folded_min_curl),
            _ramp(b.curls["middle"], cfg.finger_extended_max_curl, cfg.finger_folded_min_curl),
        ]
    )
    del scale
    return Gesture.PRAYER, _combine([closeness, aligned, upright, fingers_out])
