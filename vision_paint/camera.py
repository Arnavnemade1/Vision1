"""Threaded webcam capture.

Grabbing frames on a background thread keeps the main loop from blocking on
camera I/O, which is what usually drags a MediaPipe pipeline below 30 FPS.
"""

from __future__ import annotations

import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np

from .config import CameraConfig


class CameraError(RuntimeError):
    pass


class Camera:
    def __init__(self, cfg: CameraConfig):
        self.cfg = cfg
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame: Optional[np.ndarray] = None
        self._stamp: float = 0.0
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def open(self) -> "Camera":
        # AVFoundation is the backend that actually honours resolution requests
        # on macOS; cv2.CAP_ANY falls back cleanly elsewhere.
        backend = cv2.CAP_AVFOUNDATION if hasattr(cv2, "CAP_AVFOUNDATION") else cv2.CAP_ANY
        cap = cv2.VideoCapture(self.cfg.index, backend)
        if not cap.isOpened():
            cap = cv2.VideoCapture(self.cfg.index)
        if not cap.isOpened():
            raise CameraError(
                f"Could not open camera index {self.cfg.index}. "
                "On macOS, grant camera access to your terminal in "
                "System Settings > Privacy & Security > Camera."
            )
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.height)
        cap.set(cv2.CAP_PROP_FPS, self.cfg.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._cap = cap

        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            raise CameraError("Camera opened but returned no frames.")
        self._frame, self._stamp = self._prepare(frame), time.time()

        self._running = True
        self._thread = threading.Thread(target=self._loop, name="camera", daemon=True)
        self._thread.start()
        return self

    def _prepare(self, frame: np.ndarray) -> np.ndarray:
        return cv2.flip(frame, 1) if self.cfg.mirror else frame

    def _loop(self) -> None:
        assert self._cap is not None
        while self._running:
            ok, frame = self._cap.read()
            if not ok or frame is None:
                time.sleep(0.005)
                continue
            prepared = self._prepare(frame)
            with self._lock:
                self._frame = prepared
                self._stamp = time.time()

    def read(self) -> Tuple[Optional[np.ndarray], float]:
        """Latest frame and its capture timestamp (seconds)."""
        with self._lock:
            if self._frame is None:
                return None, 0.0
            return self._frame, self._stamp

    @property
    def size(self) -> Tuple[int, int]:
        frame, _ = self.read()
        if frame is None:
            return self.cfg.width, self.cfg.height
        h, w = frame.shape[:2]
        return w, h

    def close(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "Camera":
        return self.open()

    def __exit__(self, *exc) -> None:
        self.close()
