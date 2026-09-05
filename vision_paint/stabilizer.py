"""Temporal filtering: per-frame guesses -> trustworthy commands.

Two stages sit between the classifier and the canvas:

  1. A voting window. A gesture becomes "stable" only once it wins a majority
     of the last N frames. A single bad frame -- a hand half-way between two
     poses, a dropped detection -- can never trigger anything.
  2. An event gate. Stable gestures that map to one-shot commands fire once,
     then go on cooldown, so holding a pose does not spam the command. Anything
     destructive additionally has to be held for a dwell period, which doubles
     as the user's chance to back out.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional, Tuple

from .config import StabilizerConfig
from .gestures import Gesture


@dataclass
class StableState:
    gesture: Gesture = Gesture.NONE
    confidence: float = 0.0
    since: float = 0.0          # when this gesture became stable
    held: float = 0.0           # seconds it has been stable
    changed: bool = False       # became stable on this frame

    @property
    def active(self) -> bool:
        return self.gesture is not Gesture.NONE


class GestureStabilizer:
    def __init__(self, cfg: StabilizerConfig):
        self.cfg = cfg
        self._window: Deque[Tuple[Gesture, float]] = deque(maxlen=cfg.history)
        self.state = StableState()

    def reset(self) -> None:
        self._window.clear()
        self.state = StableState()

    def update(self, gesture: Gesture, confidence: float, now: float) -> StableState:
        self._window.append((gesture, confidence))

        tally: Dict[Gesture, list] = {}
        for g, c in self._window:
            tally.setdefault(g, []).append(c)

        winner, votes, mean_conf = Gesture.NONE, 0, 0.0
        for g, confs in tally.items():
            if g is Gesture.NONE:
                continue
            n = len(confs)
            avg = sum(confs) / n
            if n > votes or (n == votes and avg > mean_conf):
                winner, votes, mean_conf = g, n, avg

        accepted = (
            winner is not Gesture.NONE
            and votes >= self.cfg.min_votes
            and mean_conf >= self.cfg.min_confidence
        )
        new = winner if accepted else Gesture.NONE

        if new is not self.state.gesture:
            self.state = StableState(new, mean_conf if accepted else 0.0, now, 0.0, True)
        else:
            self.state = StableState(
                new, mean_conf if accepted else 0.0, self.state.since,
                now - self.state.since, False
            )
        return self.state


@dataclass
class EventGate:
    """Turns a sustained stable gesture into single, rate-limited commands."""

    cfg: StabilizerConfig
    _last_fired: Dict[Gesture, float] = field(default_factory=dict)
    _armed: Dict[Gesture, bool] = field(default_factory=dict)

    def reset(self) -> None:
        self._last_fired.clear()
        self._armed.clear()

    def progress(self, state: StableState, dwell: float) -> float:
        """0..1 progress toward committing a dwell-gated gesture."""
        if not state.active or dwell <= 0:
            return 0.0
        return min(1.0, state.held / dwell)

    def try_fire(
        self,
        state: StableState,
        now: float,
        cooldown: Optional[float] = None,
        dwell: float = 0.0,
        repeat: bool = False,
    ) -> bool:
        """Should the command bound to `state.gesture` run on this frame?

        `repeat=True` lets a held gesture fire again every cooldown (used by the
        brush-size and colour steppers, where holding to scrub feels right);
        otherwise the gesture must be released and re-formed.
        """
        if not state.active:
            return False
        g = state.gesture
        if state.changed:
            self._armed[g] = True
        if state.held < dwell:
            return False
        if not repeat and not self._armed.get(g, False):
            return False

        cd = self.cfg.default_cooldown if cooldown is None else cooldown
        last = self._last_fired.get(g, -1e9)
        if now - last < cd:
            return False

        self._last_fired[g] = now
        if not repeat:
            self._armed[g] = False
        return True
