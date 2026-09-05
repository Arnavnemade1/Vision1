"""Render a frame of the app without a camera.

Feeds scripted synthetic gestures through the real pipeline and writes the
composed frame -- canvas, cursor, HUD and all -- to a PNG. Useful for checking
the drawing and overlay code on a machine with no camera access, and as a quick
visual regression when the HUD changes.
"""

from __future__ import annotations

import argparse
import math
import sys
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vision_paint.canvas import CanvasSet
from vision_paint.commands import CommandDispatcher
from vision_paint.config import AppConfig
from vision_paint.features import extract
from vision_paint.state import AppState
from vision_paint.synthetic import POSES, build_frame
from vision_paint.ui import draw_cursor, draw_help, draw_hud, draw_landmarks

DT = 1 / 30
W, H = 1280, 720


def gradient_background() -> np.ndarray:
    y = np.linspace(0, 1, H)[:, None]
    x = np.linspace(0, 1, W)[None, :]
    r = (40 + 60 * y + 20 * x)
    g = (36 + 30 * y + 45 * x)
    b = (44 + 20 * y + 70 * x)
    return np.dstack([b, g, r]).astype(np.uint8)


class Runner:
    def __init__(self, cfg: AppConfig, save_dir: Path):
        self.cfg = cfg
        self.state = AppState(cfg=cfg, canvases=CanvasSet(W, H, cfg.canvas), save_dir=save_dir)
        self.dispatcher = CommandDispatcher(cfg, self.state)
        self.frame = gradient_background()
        self.dispatcher.last_camera_frame = self.frame
        self.t = 0.0
        self.hands = []

    def hold(self, pose: str, seconds: float, path=None, center=(0.5, 0.5)):
        frames = max(1, int(seconds / DT))
        spec = deepcopy(POSES[pose])
        for i in range(frames):
            self.t += DT
            f = i / max(1, frames - 1)
            at = path(f) if path else center
            hand = extract(build_frame(spec, center=at), self.cfg.gesture, W / H)
            self.hands = [hand]
            self.dispatcher.update(self.hands, self.t)
        return self

    def gap(self, seconds: float = 0.35):
        for _ in range(int(seconds / DT)):
            self.t += DT
            self.dispatcher.update([], self.t)
        self.hands = []
        return self

    def render(self, show_help: bool = False) -> np.ndarray:
        canvas = self.state.canvas
        composed = canvas.composite(canvas.background_frame(self.frame))
        if self.hands:
            draw_landmarks(composed, self.hands)
        draw_cursor(
            composed, self.dispatcher.cursor, self.state.color, self.state.brush_size,
            self.dispatcher.last_state.gesture.value == "draw",
        )
        draw_hud(
            composed, self.state,
            self.dispatcher.last_state.gesture.value if self.dispatcher.last_state.active else "",
            self.dispatcher.last_state.confidence, 29.4,
            self.dispatcher.dwell_progress, self.dispatcher.dwell_label,
            self.dispatcher.ranking,
        )
        if show_help:
            draw_help(composed)
        return composed


def arc(cx, cy, r, a0, a1):
    return lambda f: (cx + r * math.cos(math.radians(a0 + (a1 - a0) * f)),
                      cy + r * math.sin(math.radians(a0 + (a1 - a0) * f)))


def line(x0, y0, x1, y1):
    return lambda f: (x0 + (x1 - x0) * f, y0 + (y1 - y0) * f)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="scratch/demo.png")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--help-overlay", action="store_true")
    a = ap.parse_args()

    cfg = AppConfig()
    cfg.show_debug_panel = a.debug
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    run = Runner(cfg, out.parent)

    # A few strokes in different colours, sizes and styles.
    run.hold("draw", 1.0, path=line(0.14, 0.62, 0.40, 0.30)).gap()
    run.hold("point_right", 0.5).gap()          # next colour
    run.hold("point_up", 0.9).gap()             # bigger brush
    run.hold("draw", 1.2, path=arc(0.42, 0.50, 0.11, 190, 500)).gap()
    run.hold("three_fingers", 0.6).gap()        # marker style
    run.hold("point_right", 0.5).gap()
    run.hold("draw", 1.0, path=line(0.60, 0.28, 0.86, 0.62)).gap()
    run.hold("three_fingers", 0.6).gap()        # neon style
    run.hold("point_right", 0.5).gap()
    run.hold("point_up", 0.6).gap()
    run.hold("draw", 1.2, path=arc(0.72, 0.46, 0.10, 90, 430)).gap()
    run.hold("draw", 0.8, path=line(0.20, 0.78, 0.85, 0.78)).gap()

    # Leave the frame mid-gesture so the HUD has something live to show.
    run.hold("two_fingers", 0.5)

    frame = run.render(show_help=a.help_overlay)
    cv2.imwrite(str(out), frame)
    print(f"wrote {out}  ({len(run.state.canvas.strokes)} strokes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
