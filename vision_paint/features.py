"""Geometric features derived from hand landmarks.

Everything here is scale- and rotation-tolerant: joint angles instead of
"is the tip above the knuckle" tests, and every distance divided by the hand's
own size, so the same thresholds work near or far from the camera and with
small or large hands.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np

from .config import GestureConfig
from .hand_tracker import (
    FINGER_JOINTS,
    INDEX_MCP,
    INDEX_TIP,
    MIDDLE_MCP,
    MIDDLE_TIP,
    PINKY_MCP,
    PINKY_TIP,
    RING_TIP,
    THUMB_IP,
    THUMB_MCP,
    THUMB_TIP,
    WRIST,
    HandFrame,
)

FINGERS = ("index", "middle", "ring", "pinky")


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else np.zeros_like(v)


def angle_between(a: np.ndarray, b: np.ndarray) -> float:
    """Angle in degrees between two vectors."""
    ua, ub = _unit(a), _unit(b)
    if not ua.any() or not ub.any():
        return 0.0
    return float(np.degrees(np.arccos(np.clip(float(np.dot(ua, ub)), -1.0, 1.0))))


@dataclass
class HandFeatures:
    """Everything the gesture rules need, computed once per hand per frame."""

    label: str
    hand_size: float
    curls: Dict[str, float]            # degrees, 0 = straight
    extended: Dict[str, bool]
    thumb_curl: float
    thumb_extended: bool
    thumb_abduction: float             # thumb tip -> index MCP, hand-size units
    pinch: float                       # thumb tip -> index tip, hand-size units
    spread: float                      # index tip -> pinky tip, hand-size units
    index_reach: float                 # wrist -> index tip, hand-size units
    index_middle_tip_gap: float
    index_middle_mcp_gap: float
    palm_normal: np.ndarray            # world-space, points out of the palm
    palm_facing: float                 # 1 = palm at camera, -1 = back of hand
    hand_axis: np.ndarray              # image-space wrist -> middle MCP, y-up
    index_dir: np.ndarray              # image-space index MCP -> tip, y-up
    thumb_dir: np.ndarray              # image-space thumb MCP -> tip, y-up
    index_tip: np.ndarray              # image-space (x, y) in [0, 1]
    palm_center: np.ndarray
    pinch_point: np.ndarray            # midpoint of thumb and index tips
    extended_count: int = 0
    raw: HandFrame = field(repr=False, default=None)  # type: ignore[assignment]

    def is_extended(self, *names: str) -> bool:
        return all(self.extended[n] for n in names)

    def is_folded(self, *names: str) -> bool:
        return all(not self.extended[n] for n in names)


def _image_xy(landmarks: np.ndarray, idx: int, aspect: float) -> np.ndarray:
    """Aspect-corrected image point with y pointing up (so maths reads naturally)."""
    return np.array([landmarks[idx, 0] * aspect, 1.0 - landmarks[idx, 1]])


def extract(hand: HandFrame, cfg: GestureConfig, aspect: float = 16 / 9) -> HandFeatures:
    w = hand.world
    lm = hand.landmarks

    hand_size = float(np.linalg.norm(w[MIDDLE_MCP] - w[WRIST]))
    if hand_size < 1e-6:
        hand_size = 1e-6

    curls: Dict[str, float] = {}
    extended: Dict[str, bool] = {}
    for name, (mcp, pip, dip, tip) in FINGER_JOINTS.items():
        # Curl is measured across the longest available baselines -- proximal
        # phalanx against the PIP-to-tip chord -- rather than joint by joint.
        # The distal segments are only ~20 mm long, so per-joint angles swing by
        # 10 degrees or more under the landmark noise MediaPipe actually
        # produces, while this chord form barely moves. The 1.3 factor rescales
        # the chord angle back onto a full-curl scale (0 straight, ~165 fisted).
        curl = 1.3 * angle_between(w[pip] - w[mcp], w[tip] - w[pip])
        curls[name] = curl
        extended[name] = curl < cfg.finger_extended_max_curl

    thumb_curl = angle_between(w[THUMB_IP] - w[THUMB_MCP], w[THUMB_TIP] - w[THUMB_IP])
    # A tucked thumb folds onto the side of the index finger, so its tip sits
    # close to the index knuckle; an abducted thumb swings well clear of it.
    # Dividing by hand size keeps the threshold valid at any camera distance.
    thumb_abduction = float(np.linalg.norm(w[THUMB_TIP] - w[INDEX_MCP]) / hand_size)
    thumb_extended = (
        thumb_curl < cfg.thumb_extended_max_curl
        and thumb_abduction > cfg.thumb_abduction_min
    )

    pinch = float(np.linalg.norm(w[THUMB_TIP] - w[INDEX_TIP]) / hand_size)
    spread = float(np.linalg.norm(w[INDEX_TIP] - w[PINKY_TIP]) / hand_size)
    # How far the index tip sits from the wrist. A fist rolls it all the way in;
    # a pinch holds it out in front of the palm. This is what keeps "erase" and
    # "select tool" apart, since both curl the remaining fingers.
    index_reach = float(np.linalg.norm(w[INDEX_TIP] - w[WRIST]) / hand_size)
    im_tip_gap = float(np.linalg.norm(w[INDEX_TIP] - w[MIDDLE_TIP]) / hand_size)
    im_mcp_gap = float(np.linalg.norm(w[INDEX_MCP] - w[MIDDLE_MCP]) / hand_size)

    # Palm plane normal, oriented to point out through the palm for both hands.
    # The sign depends on the handedness of the *pixels*, not on which of the
    # user's hands it is: a selfie-mirrored frame swaps the two.
    n = _unit(np.cross(w[INDEX_MCP] - w[WRIST], w[PINKY_MCP] - w[WRIST]))
    if hand.geometric_label == "Right":
        n = -n
    # MediaPipe world z grows away from the camera, so a palm aimed at the lens
    # has a negative z component.
    palm_facing = float(-n[2])

    axis = _unit(_image_xy(lm, MIDDLE_MCP, aspect) - _image_xy(lm, WRIST, aspect))
    index_dir = _unit(_image_xy(lm, INDEX_TIP, aspect) - _image_xy(lm, INDEX_MCP, aspect))
    thumb_dir = _unit(_image_xy(lm, THUMB_TIP, aspect) - _image_xy(lm, THUMB_MCP, aspect))

    index_tip = lm[INDEX_TIP, :2].copy()
    palm_center = np.mean(lm[[WRIST, INDEX_MCP, PINKY_MCP], :2], axis=0)
    pinch_point = (lm[THUMB_TIP, :2] + lm[INDEX_TIP, :2]) / 2.0

    return HandFeatures(
        label=hand.label,
        hand_size=hand_size,
        curls=curls,
        extended=extended,
        thumb_curl=thumb_curl,
        thumb_extended=thumb_extended,
        thumb_abduction=thumb_abduction,
        pinch=pinch,
        spread=spread,
        index_reach=index_reach,
        index_middle_tip_gap=im_tip_gap,
        index_middle_mcp_gap=im_mcp_gap,
        palm_normal=n,
        palm_facing=palm_facing,
        hand_axis=axis,
        index_dir=index_dir,
        thumb_dir=thumb_dir,
        index_tip=index_tip,
        palm_center=palm_center,
        pinch_point=pinch_point,
        extended_count=sum(extended.values()),
        raw=hand,
    )
