"""Movement analysis for the gestures that mean different things when moving.

Two gestures in the vocabulary are overloaded: two fingers held still changes
the colour but swiped sideways changes canvas, and a pinch held still selects a
tool but a pinch that opens or closes zooms. Both are resolved here, by
measuring how the hand is moving rather than only how it is shaped.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Tuple

import numpy as np

from .config import MotionConfig


@dataclass
class MotionState:
    speed: float = 0.0              # screen widths per second
    velocity: Tuple[float, float] = (0.0, 0.0)
    displacement: Tuple[float, float] = (0.0, 0.0)
    pinch_rate: float = 0.0         # aperture change per second
    pinch_delta: float = 0.0        # aperture change across the trail

    @property
    def still(self) -> bool:
        return self.speed < 1e-6


class MotionTracker:
    """Rolling trail of one hand's palm position and pinch aperture."""

    def __init__(self, cfg: MotionConfig):
        self.cfg = cfg
        self._trail: Deque[Tuple[float, np.ndarray, float]] = deque()
        self.state = MotionState()

    def reset(self) -> None:
        self._trail.clear()
        self.state = MotionState()

    def update(self, position: np.ndarray, pinch: float, now: float) -> MotionState:
        self._trail.append((now, np.asarray(position, dtype=np.float64).copy(), float(pinch)))
        cutoff = now - self.cfg.trail_seconds
        while len(self._trail) > 2 and self._trail[0][0] < cutoff:
            self._trail.popleft()

        if len(self._trail) < 2:
            self.state = MotionState()
            return self.state

        t0, p0, a0 = self._trail[0]
        t1, p1, a1 = self._trail[-1]
        dt = max(t1 - t0, 1e-6)
        d = p1 - p0
        velocity = d / dt
        self.state = MotionState(
            speed=float(np.linalg.norm(velocity)),
            velocity=(float(velocity[0]), float(velocity[1])),
            displacement=(float(d[0]), float(d[1])),
            pinch_rate=(a1 - a0) / dt,
            pinch_delta=a1 - a0,
        )
        return self.state

    def swipe(self) -> Optional[int]:
        """+1 for a rightward swipe, -1 for leftward, None if there isn't one."""
        s = self.state
        if s.speed < self.cfg.swipe_min_speed:
            return None
        dx, dy = s.displacement
        if abs(dx) < self.cfg.swipe_min_distance or abs(dx) < abs(dy) * 1.6:
            return None
        return 1 if dx > 0 else -1

    def consume(self) -> None:
        """Forget the trail so one movement cannot fire twice."""
        self._trail.clear()
        self.state = MotionState()

    @property
    def is_still(self) -> bool:
        return self.state.speed < self.cfg.static_max_speed
