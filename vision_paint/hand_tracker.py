"""MediaPipe HandLandmarker wrapper (Tasks API) with temporal smoothing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)

from .config import MODEL_PATH, TrackerConfig
from .filters import OneEuroFilter

# Landmark indices (MediaPipe hand topology).
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

FINGER_JOINTS = {
    "index": (INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP),
    "middle": (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP),
    "ring": (RING_MCP, RING_PIP, RING_DIP, RING_TIP),
    "pinky": (PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP),
}

CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)


@dataclass
class HandFrame:
    """One detected hand at one instant."""

    label: str                 # "Left" / "Right", as the user sees their own hand
    geometric_label: str       # handedness of the pixels as MediaPipe saw them
    score: float
    landmarks: np.ndarray      # (21, 3) normalised image coords, x/y in [0, 1]
    world: np.ndarray          # (21, 3) metric-ish coords, origin at hand centre
    timestamp: float

    @property
    def wrist(self) -> np.ndarray:
        return self.landmarks[WRIST]


class HandTracker:
    def __init__(self, cfg: TrackerConfig, model_path: Path = MODEL_PATH, mirrored: bool = True):
        if not model_path.exists():
            raise FileNotFoundError(
                f"Hand model not found at {model_path}. Run scripts/fetch_model.sh"
            )
        self.cfg = cfg
        self.mirrored = mirrored
        options = HandLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=str(model_path),
                # macOS builds of MediaPipe 1.x default to a Metal delegate whose
                # graph service is not wired up for Tasks; force CPU inference.
                delegate=BaseOptions.Delegate.CPU,
            ),
            running_mode=RunningMode.VIDEO,
            num_hands=cfg.max_hands,
            min_hand_detection_confidence=cfg.min_detection_confidence,
            min_hand_presence_confidence=cfg.min_presence_confidence,
            min_tracking_confidence=cfg.min_tracking_confidence,
        )
        self._landmarker = HandLandmarker.create_from_options(options)
        self._filters: Dict[str, OneEuroFilter] = {}
        self._last_ms = -1

    def _filter_for(self, label: str) -> OneEuroFilter:
        if label not in self._filters:
            self._filters[label] = OneEuroFilter(
                self.cfg.filter_min_cutoff, self.cfg.filter_beta, self.cfg.filter_d_cutoff
            )
        return self._filters[label]

    def process(self, bgr: np.ndarray, timestamp: float) -> List[HandFrame]:
        rgb = np.ascontiguousarray(bgr[:, :, ::-1])
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # detect_for_video demands strictly increasing integer milliseconds.
        ms = int(timestamp * 1000)
        if ms <= self._last_ms:
            ms = self._last_ms + 1
        self._last_ms = ms

        result = self._landmarker.detect_for_video(mp_image, ms)
        hands: List[HandFrame] = []
        seen = set()
        for i, lms in enumerate(result.hand_landmarks):
            cat = result.handedness[i][0]
            # MediaPipe labels assuming an unmirrored image; a selfie-view frame
            # is already flipped, so the label must be flipped back to match the
            # hand the user is actually holding up.
            geometric_label = cat.category_name
            label = geometric_label
            if self.mirrored:
                label = "Left" if label == "Right" else "Right"

            pts = np.array([[p.x, p.y, p.z] for p in lms], dtype=np.float64)
            world_lms = result.hand_world_landmarks[i]
            world = np.array([[p.x, p.y, p.z] for p in world_lms], dtype=np.float64)

            key = label if label not in seen else f"{label}#{i}"
            seen.add(key)
            pts = self._filter_for(key)(pts.reshape(-1), timestamp).reshape(21, 3)

            hands.append(
                HandFrame(
                    label=label,
                    geometric_label=geometric_label,
                    score=float(cat.score),
                    landmarks=pts,
                    world=world,
                    timestamp=timestamp,
                )
            )

        if not hands:
            for f in self._filters.values():
                f.reset()
        return hands

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> "HandTracker":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
