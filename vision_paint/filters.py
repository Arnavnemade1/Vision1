"""One Euro filter — low jitter when the hand is still, low lag when it moves."""

from __future__ import annotations

import math
from typing import Optional

import numpy as np


def _alpha(cutoff: float, dt: float) -> float:
    tau = 1.0 / (2.0 * math.pi * max(cutoff, 1e-6))
    return 1.0 / (1.0 + tau / max(dt, 1e-6))


class OneEuroFilter:
    """Vector-valued One Euro filter (Casiez et al., 2012)."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x_prev: Optional[np.ndarray] = None
        self._dx_prev: Optional[np.ndarray] = None
        self._t_prev: Optional[float] = None

    def reset(self) -> None:
        self._x_prev = self._dx_prev = self._t_prev = None

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if self._x_prev is None or self._t_prev is None:
            self._x_prev = x
            self._dx_prev = np.zeros_like(x)
            self._t_prev = t
            return x

        dt = t - self._t_prev
        if dt <= 0:
            return self._x_prev
        # Large gap (hand left the frame): restart rather than lerp across it.
        if dt > 0.5:
            self.reset()
            return self(x, t)

        dx = (x - self._x_prev) / dt
        a_d = _alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * self._dx_prev

        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        # Per-component adaptive alpha.
        a = 1.0 / (1.0 + (1.0 / (2.0 * np.pi * np.maximum(cutoff, 1e-6))) / dt)
        x_hat = a * x + (1 - a) * self._x_prev

        self._x_prev, self._dx_prev, self._t_prev = x_hat, dx_hat, t
        return x_hat
