"""Synthetic hand poses.

Builds anatomically plausible 21-point hands from a handful of parameters
(per-finger curl, finger fan, thumb position, whole-hand rotation). Used by the
test suite to exercise the classifier over hundreds of poses -- including noisy
and half-formed ones -- without needing a camera or recorded footage.

Coordinate convention matches MediaPipe world landmarks: x right, y down,
z away from the camera, metres, origin at the wrist.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import numpy as np

from .hand_tracker import HandFrame

# Segment lengths in metres, roughly an adult hand.
PALM_LEN = 0.090
FINGER_SEGMENTS: Dict[str, Sequence[float]] = {
    "index": (0.042, 0.026, 0.020),
    "middle": (0.046, 0.029, 0.021),
    "ring": (0.042, 0.027, 0.020),
    "pinky": (0.033, 0.020, 0.018),
}
# MCP offsets across the palm (x) for a geometric RIGHT hand seen palm-on:
# thumb side is -x, little finger side is +x.
MCP_X = {"index": -0.027, "middle": -0.009, "ring": 0.009, "pinky": 0.027}
MCP_Y = {"index": -0.086, "middle": -0.090, "ring": -0.085, "pinky": -0.075}
FAN_DEG = {"index": -7.0, "middle": -1.0, "ring": 5.0, "pinky": 12.0}


def _rot(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rodrigues rotation matrix."""
    k = axis / (np.linalg.norm(axis) + 1e-12)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(angle_rad) * K + (1 - math.cos(angle_rad)) * (K @ K)


@dataclass
class PoseSpec:
    """Parameters for one synthetic hand."""

    curls: Dict[str, float] = field(default_factory=dict)   # total curl degrees per finger
    thumb_curl: float = 8.0                                 # extra distal flexion, degrees
    thumb_out: float = 1.0                                  # 0 = tucked across palm, 1 = abducted
    fan: float = 1.0                                        # finger spread multiplier
    cross: float = 0.0                                      # index/middle convergence (0..1)
    pinch: float = 0.0                                      # 0 = thumb free, 1 = tip on index tip
    roll_deg: float = 0.0                                   # rotate hand in the image plane
    pitch_deg: float = 0.0                                  # tilt palm away from the camera
    geometric_label: str = "Right"
    label: Optional[str] = None                             # defaults to geometric_label

    def curl_of(self, finger: str) -> float:
        return self.curls.get(finger, 0.0)


def _finger_points(name: str, spec: PoseSpec) -> np.ndarray:
    mcp = np.array([MCP_X[name], MCP_Y[name], 0.0])
    fan_deg = FAN_DEG[name] * spec.fan
    if spec.cross:
        # Swing index and middle toward each other (and slightly past) so their
        # tips meet, the way crossed fingers actually sit.
        if name == "index":
            fan_deg += 9.0 * spec.cross
        elif name == "middle":
            fan_deg -= 4.5 * spec.cross
    fan = math.radians(fan_deg)
    direction = np.array([math.sin(fan), -math.cos(fan), 0.0])

    total = spec.curl_of(name)
    # Split the requested curl across the two measured joints, with a little
    # more bend proximally, the way a real finger folds.
    joint_angles = [math.radians(total * 0.55), math.radians(total * 0.45)]

    pts = [mcp]
    p = mcp
    d = direction
    lengths = FINGER_SEGMENTS[name]
    for i, L in enumerate(lengths):
        if i > 0:
            # Curl folds the finger toward the palm, i.e. toward +z (away from a
            # camera-facing palm) is wrong -- it must come toward the viewer's
            # side of the palm plane, which is -z.
            axis = np.array([-1.0, 0.0, 0.0])
            d = _rot(axis, joint_angles[i - 1]) @ d
        p = p + L * d
        pts.append(p)
    return np.array(pts)  # mcp, pip, dip, tip


THUMB_SEGMENTS = (0.046, 0.032, 0.026)  # metacarpal, proximal, distal
THUMB_CMC = np.array([-0.024, -0.018, 0.004])


def _reach_chain(root: np.ndarray, target: np.ndarray, lengths: Sequence[float], bow: float) -> np.ndarray:
    """Place a 3-link chain from `root` so its tip lands on (or reaches toward)
    `target`, bowing the first joint out of line by `bow` radians so the chain
    keeps a natural arch instead of collapsing to a straight stick.

    Returns the three joint positions after the root.
    """
    l1, l2, l3 = lengths
    d = target - root
    dist = float(np.linalg.norm(d))
    if dist < 1e-9:
        d = np.array([1.0, 0.0, 0.0])
        dist = 1e-9
    d = d / dist
    dist = min(dist, 0.99 * (l1 + l2 + l3))
    tip = root + dist * d

    perp = np.cross(d, np.array([0.0, 0.0, 1.0]))
    if np.linalg.norm(perp) < 1e-9:
        perp = np.array([1.0, 0.0, 0.0])
    perp = perp / np.linalg.norm(perp)

    mcp = root + l1 * (math.cos(bow) * d + math.sin(bow) * perp)
    rest = tip - mcp
    rest_len = float(np.linalg.norm(rest))
    if rest_len < 1e-9:
        return np.array([mcp, mcp, tip])
    u = rest / rest_len
    reach = min(rest_len, l2 + l3)
    cos_ip = float(np.clip((l2 ** 2 + reach ** 2 - l3 ** 2) / (2 * l2 * reach), -1.0, 1.0))
    theta = math.acos(cos_ip)
    perp2 = np.cross(u, np.array([0.0, 0.0, 1.0]))
    if np.linalg.norm(perp2) < 1e-9:
        perp2 = perp
    perp2 = perp2 / np.linalg.norm(perp2)
    ip = mcp + l2 * (math.cos(theta) * u + math.sin(theta) * perp2)
    return np.array([mcp, ip, mcp + reach * u])


def _thumb_points(spec: PoseSpec, index_mcp: np.ndarray, index_pip: np.ndarray,
                  index_tip: np.ndarray) -> np.ndarray:
    """Thumb chain, positioned by where its tip needs to end up.

    Driving the thumb from a target rather than from joint angles is what makes
    "tucked" behave like a real tucked thumb: the tip lands on the side of the
    index finger, inside the radius of its own IP joint, which is exactly the
    signal the classifier tests for.
    """
    cmc = THUMB_CMC.copy()
    out = float(np.clip(spec.thumb_out, 0.0, 1.0))

    # Fully abducted: swung out and up on the thumb side of the palm.
    a = math.radians(-52.0)
    free_dir = np.array([math.sin(a), -math.cos(a), -0.30])
    free_dir /= np.linalg.norm(free_dir)
    abducted = cmc + sum(THUMB_SEGMENTS) * 0.94 * free_dir

    # Fully tucked: tip folded across onto the side of the index finger.
    tucked = index_pip + np.array([0.010, 0.014, -0.006])

    target = tucked * (1 - out) + abducted * out
    if spec.pinch:
        pinch_target = index_tip + np.array([0.004, 0.006, -0.008])
        target = target * (1 - spec.pinch) + pinch_target * spec.pinch

    bow = math.radians(20.0 + 26.0 * (1 - out))
    mcp, ip, tip = _reach_chain(cmc, target, THUMB_SEGMENTS, bow)

    # Optional extra flexion of the distal segment on top of the reach solution.
    if spec.thumb_curl:
        seg = tip - ip
        seg = _rot(np.array([-1.0, 0.0, 0.0]), math.radians(spec.thumb_curl)) @ seg
        tip = ip + seg
    del index_mcp
    return np.array([cmc, mcp, ip, tip])


def build_world(spec: PoseSpec) -> np.ndarray:
    """21x3 world landmarks for the pose."""
    pts = np.zeros((21, 3))
    pts[0] = (0.0, 0.0, 0.0)  # wrist
    for i, name in enumerate(("index", "middle", "ring", "pinky")):
        pts[5 + i * 4 : 9 + i * 4] = _finger_points(name, spec)
    pts[1:5] = _thumb_points(spec, pts[5], pts[6], pts[8])

    # Mirroring x turns the pose into the other hand. In-plane rotations have
    # to be negated with it, otherwise a mirrored "thumb up" comes out tilted
    # instead of staying a thumbs-up.
    sign = 1.0
    if spec.geometric_label == "Left":
        pts[:, 0] *= -1.0
        sign = -1.0

    if spec.pitch_deg:
        pts = pts @ _rot(np.array([0.0, 1.0, 0.0]), math.radians(sign * spec.pitch_deg)).T
    if spec.roll_deg:
        pts = pts @ _rot(np.array([0.0, 0.0, 1.0]), math.radians(sign * spec.roll_deg)).T

    # MediaPipe centres world landmarks on the hand, not the wrist.
    return pts - pts.mean(axis=0)


def build_frame(
    spec: PoseSpec,
    center: Sequence[float] = (0.5, 0.5),
    scale: float = 2.6,
    aspect: float = 16 / 9,
    timestamp: float = 0.0,
    noise: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> HandFrame:
    """A HandFrame with matching world and (orthographically projected) image points."""
    world = build_world(spec)
    if noise:
        rng = rng or np.random.default_rng(0)
        world = world + rng.normal(0.0, noise, world.shape)

    # Project: world x/y (metres) -> normalised image coords. `scale` maps roughly
    # one hand length to a sensible fraction of the frame.
    img = np.zeros((21, 3))
    img[:, 0] = center[0] + (world[:, 0] * scale) / aspect
    img[:, 1] = center[1] + world[:, 1] * scale
    img[:, 2] = world[:, 2] * scale

    label = spec.label or spec.geometric_label
    return HandFrame(
        label=label,
        geometric_label=spec.geometric_label,
        score=0.99,
        landmarks=img,
        world=world,
        timestamp=timestamp,
    )


# ---- named poses for the gesture vocabulary -------------------------------

FOLDED = 165.0
STRAIGHT = 6.0


def _curls(index=STRAIGHT, middle=STRAIGHT, ring=STRAIGHT, pinky=STRAIGHT) -> Dict[str, float]:
    return {"index": index, "middle": middle, "ring": ring, "pinky": pinky}


POSES: Dict[str, PoseSpec] = {
    "draw": PoseSpec(_curls(middle=FOLDED, ring=FOLDED, pinky=FOLDED), thumb_curl=10, thumb_out=0.0),
    "point_up": PoseSpec(_curls(middle=FOLDED, ring=FOLDED, pinky=FOLDED), thumb_curl=6, thumb_out=1.0),
    "point_right": PoseSpec(
        _curls(middle=FOLDED, ring=FOLDED, pinky=FOLDED), thumb_curl=6, thumb_out=1.0, roll_deg=90
    ),
    "point_left": PoseSpec(
        _curls(middle=FOLDED, ring=FOLDED, pinky=FOLDED), thumb_curl=6, thumb_out=1.0, roll_deg=-90
    ),
    "point_down": PoseSpec(
        _curls(middle=FOLDED, ring=FOLDED, pinky=FOLDED), thumb_curl=6, thumb_out=1.0, roll_deg=180
    ),
    "two_fingers": PoseSpec(_curls(ring=FOLDED, pinky=FOLDED), thumb_curl=10, thumb_out=0.0, fan=2.4),
    "crossed_fingers": PoseSpec(
        _curls(ring=FOLDED, pinky=FOLDED), thumb_curl=10, thumb_out=0.0, fan=0.2, cross=1.0
    ),
    "fist": PoseSpec(_curls(FOLDED, FOLDED, FOLDED, FOLDED), thumb_curl=15, thumb_out=0.0),
    "thumbs_up": PoseSpec(_curls(FOLDED, FOLDED, FOLDED, FOLDED), thumb_curl=6, thumb_out=1.0, roll_deg=35),
    "thumbs_down": PoseSpec(_curls(FOLDED, FOLDED, FOLDED, FOLDED), thumb_curl=6, thumb_out=1.0, roll_deg=-145),
    "three_fingers": PoseSpec(_curls(middle=FOLDED, ring=FOLDED), thumb_curl=10, thumb_out=1.0),
    "four_fingers": PoseSpec(_curls(), thumb_curl=10, thumb_out=0.0, fan=1.2),
    "call_me": PoseSpec(_curls(FOLDED, FOLDED, FOLDED, STRAIGHT), thumb_curl=10, thumb_out=1.0),
    "pinch": PoseSpec(
        _curls(index=120, middle=FOLDED, ring=FOLDED, pinky=FOLDED), thumb_out=1.0, pinch=1.0
    ),
    "ok_sign": PoseSpec(_curls(index=125), thumb_out=1.0, pinch=1.0, fan=1.6),
    "open_palm": PoseSpec(_curls(), thumb_curl=10, thumb_out=1.0, fan=2.6),
    "palm_flat": PoseSpec(_curls(), thumb_curl=6, thumb_out=1.0, fan=0.35),
}
