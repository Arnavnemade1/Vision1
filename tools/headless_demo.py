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
from vision_paint.ui import (draw_cursor, draw_group_highlight, draw_help, draw_hud,
                             draw_landmarks, draw_minimap, theme_for)

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

    def aim(self, pose: str, attribute: str, target):
        hand = extract(build_frame(POSES[pose], center=(0.5, 0.5)), self.cfg.gesture, W / H)
        off = np.asarray(getattr(hand, attribute)) - np.array([0.5, 0.5])
        return (target[0] - off[0], target[1] - off[1])

    def screen_of(self, group):
        canvas = self.state.canvas
        x0, y0, x1, y1 = group.bounds
        sx, sy = canvas.board_to_screen((x0 + x1) / 2, (y0 + y1) / 2)
        return (sx / canvas.width, sy / canvas.height)

    def grab_drag(self, frm, to, settle: float = 0.4, drag: float = 0.9):
        start = self.aim("pinch", "pinch_point", frm)
        end = self.aim("pinch", "pinch_point", to)
        self.hold("pinch", settle, center=start)
        self.hold("pinch", drag, path=line(start[0], start[1], end[0], end[1]))
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
        draw_group_highlight(
            composed, canvas,
            self.dispatcher.grabbed_group or self.dispatcher.hover_group,
            self.dispatcher.grabbed_group is not None,
        )
        if canvas.zoom != 1.0 or canvas.pan.any() or len(canvas.groups()) > 1:
            draw_minimap(composed, canvas)
        if self.hands:
            draw_landmarks(composed, self.hands, theme_for(canvas))
        draw_cursor(
            composed, self.dispatcher.cursor, self.state.color, self.state.brush_size,
            self.dispatcher.last_state.gesture.value == "draw",
        )
        draw_hud(
            composed, self.state,
            self.dispatcher.last_state.gesture.value if self.dispatcher.last_state.active else "",
            self.dispatcher.last_state.confidence, 29.4,
            self.dispatcher.dwell_progress, self.dispatcher.dwell_label,
            self.dispatcher.ranking, self.dispatcher.pinch_mode,
        )
        if show_help:
            draw_help(composed, theme_for(canvas))
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

    # Draw a first sketch near the middle of the board.
    run.hold("draw", 0.9, path=arc(0.46, 0.46, 0.10, 200, 520)).gap()
    run.hold("draw", 0.5, path=line(0.40, 0.62, 0.54, 0.62)).gap()

    # Pick it up and shove it aside to free the space -- the whole point of the
    # board being larger than the window.
    where = run.screen_of(run.state.canvas.groups()[0])
    run.grab_drag(where, (where[0] - 0.30, where[1] - 0.04)).gap()

    # Then draw something new where it used to be, in another colour and style.
    run.hold("point_right", 0.5).gap()
    run.hold("point_up", 0.7).gap()
    run.hold("three_fingers", 0.6).gap()          # marker
    run.hold("draw", 1.0, path=arc(0.56, 0.46, 0.11, 90, 430)).gap()
    run.hold("point_right", 0.5).gap()
    run.hold("three_fingers", 0.6).gap()          # neon
    run.hold("draw", 0.8, path=line(0.72, 0.34, 0.88, 0.60)).gap()

    # Leave the hand hovering a drawing so the grab highlight is on screen.
    target = run.screen_of(run.state.canvas.groups()[0])
    run.hold("pinch", 0.5, center=run.aim("pinch", "pinch_point", target))

    frame = run.render(show_help=a.help_overlay)
    cv2.imwrite(str(out), frame)
    print(f"wrote {out}  ({len(run.state.canvas.strokes)} strokes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
