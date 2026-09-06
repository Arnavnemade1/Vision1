"""Main loop: camera -> landmarks -> features -> gestures -> canvas -> screen."""

from __future__ import annotations

import argparse
import time
from collections import deque
from typing import Deque, List, Optional

import cv2
import numpy as np

from . import features as feat
from .camera import Camera, CameraError
from .canvas import CanvasSet
from .commands import CommandDispatcher
from .config import MODEL_PATH, SAVE_DIR, AppConfig
from .hand_tracker import HandTracker
from .state import AppState
from .ui import (draw_cursor, draw_group_highlight, draw_help, draw_hud,
                 draw_landmarks, draw_minimap, theme_for)


class VisionPaint:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.camera = Camera(cfg.camera)
        # The model is loaded inside run(), after the camera is open: loading it
        # first means a couple of seconds and a wall of MediaPipe logs before a
        # missing-camera message the user could have had immediately.
        self.tracker: Optional[HandTracker] = None
        self.state: Optional[AppState] = None
        self.dispatcher: Optional[CommandDispatcher] = None
        self.show_help = False
        self._fps: Deque[float] = deque(maxlen=30)

    def _init_state(self, width: int, height: int) -> None:
        canvases = CanvasSet(width, height, self.cfg.canvas)
        self.state = AppState(cfg=self.cfg, canvases=canvases, save_dir=SAVE_DIR)
        self.dispatcher = CommandDispatcher(self.cfg, self.state)
        self.state.notify("Press h for the gesture list", "good")

    def run(self) -> int:
        with self.camera:
            self.tracker = HandTracker(self.cfg.tracker, MODEL_PATH, mirrored=self.cfg.camera.mirror)
            with self.tracker:
                return self._loop()

    def _loop(self) -> int:
        assert self.tracker is not None
        width, height = self.camera.size
        self._init_state(width, height)
        assert self.state and self.dispatcher

        cv2.namedWindow(self.cfg.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.cfg.window_name, width, height)

        last_stamp = -1.0
        while self.state.running:
            frame, stamp = self.camera.read()
            if frame is None:
                time.sleep(0.005)
                continue
            if stamp == last_stamp:
                # No new frame yet; keep the UI responsive without redoing
                # inference on an image we have already processed.
                if self._handle_keys() is False:
                    break
                continue
            last_stamp = stamp

            started = time.perf_counter()
            h, w = frame.shape[:2]
            self.state.canvases.resize(w, h)
            aspect = w / h

            hands_raw = self.tracker.process(frame, stamp)
            hands = [
                feat.extract(hand, self.cfg.gesture, aspect, self.cfg.tracker.edge_margin)
                for hand in hands_raw
            ]

            self.dispatcher.last_camera_frame = frame
            self.dispatcher.update(hands, stamp)

            canvas = self.state.canvas
            background = canvas.background_frame(frame)
            composed = canvas.composite(background)

            draw_group_highlight(
                composed, canvas,
                self.dispatcher.grabbed_group or self.dispatcher.hover_group,
                self.dispatcher.grabbed_group is not None,
            )
            if canvas.zoom != 1.0 or canvas.pan.any() or len(canvas.groups()) > 1:
                draw_minimap(composed, canvas)
            if self.cfg.show_landmarks:
                draw_landmarks(composed, hands, theme_for(canvas))
            draw_cursor(
                composed, self.dispatcher.cursor, self.state.color,
                self.state.brush_size, self.dispatcher.last_state.gesture.value == "draw",
            )

            self._fps.append(time.perf_counter() - started)
            fps = 1.0 / max(np.mean(self._fps), 1e-6)
            draw_hud(
                composed, self.state,
                self.dispatcher.last_state.gesture.value if self.dispatcher.last_state.active else "",
                self.dispatcher.last_state.confidence, fps,
                self.dispatcher.dwell_progress, self.dispatcher.dwell_label,
                self.dispatcher.ranking, self.dispatcher.pinch_mode,
            )
            if self.show_help:
                draw_help(composed, theme_for(canvas))

            cv2.imshow(self.cfg.window_name, composed)
            if self._handle_keys() is False:
                break

        cv2.destroyAllWindows()
        return 0

    def _handle_keys(self) -> Optional[bool]:
        assert self.state
        key = cv2.waitKey(1) & 0xFF
        if key == 255:
            return None
        if key in (ord("q"), 27):
            return False
        if key == ord("h"):
            self.show_help = not self.show_help
        elif key == ord("l"):
            self.cfg.show_landmarks = not self.cfg.show_landmarks
        elif key == ord("d"):
            self.cfg.show_debug_panel = not self.cfg.show_debug_panel
        elif key == ord("c"):
            self.state.canvas.clear()
            self.state.notify("Screen cleared", "warn")
        elif key == ord("z"):
            what = self.state.canvas.undo()
            self.state.notify(f"Undo: {what}" if what else "Nothing to undo")
        elif key == ord("s"):
            path = self.state.canvas.save(self.state.save_dir, None)
            self.state.notify(f"Saved {path.name}", "good")
        elif key == ord("r"):
            self.state.canvas.reset_view()
            self.state.notify("View reset")
        elif key in (ord("+"), ord("=")):
            canvas = self.state.canvas
            canvas.set_zoom(canvas.zoom * 1.15)
            self.state.notify(f"Zoom {canvas.zoom:.2f}x")
        elif key in (ord("-"), ord("_")):
            canvas = self.state.canvas
            canvas.set_zoom(canvas.zoom / 1.15)
            self.state.notify(f"Zoom {canvas.zoom:.2f}x")
        return None


def build_config(args: argparse.Namespace) -> AppConfig:
    cfg = AppConfig()
    cfg.camera.index = args.camera
    cfg.camera.width, cfg.camera.height = args.width, args.height
    cfg.camera.mirror = not args.no_mirror
    cfg.tracker.max_hands = args.hands
    cfg.show_landmarks = not args.no_landmarks
    cfg.show_debug_panel = args.debug
    if not args.no_calibration:
        applied = cfg.load_calibration()
        if applied:
            print(f"calibration: applied {len(applied)} measured thresholds "
                  f"from {cfg.calibration}")
    return cfg


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Gesture-controlled whiteboard")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--hands", type=int, default=2)
    ap.add_argument("--no-mirror", action="store_true", help="do not flip the camera image")
    ap.add_argument("--no-landmarks", action="store_true")
    ap.add_argument("--debug", action="store_true", help="show the gesture score panel")
    ap.add_argument("--no-calibration", action="store_true",
                    help="ignore calibration.json and use the built-in thresholds")
    args = ap.parse_args(argv)

    try:
        return VisionPaint(build_config(args)).run()
    except CameraError as e:
        print(f"\n{e}\n")
        return 2
    except FileNotFoundError as e:
        print(f"\n{e}\n")
        return 3
    except KeyboardInterrupt:
        return 130
