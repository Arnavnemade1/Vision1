"""Tunable constants for the vision drawing system.

Everything a person might want to adjust for their camera, lighting or hand
size lives here so the recognition code stays free of magic numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "hand_landmarker.task"
SAVE_DIR = PROJECT_ROOT / "saves"

BGR = Tuple[int, int, int]


@dataclass
class CameraConfig:
    index: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30
    mirror: bool = True  # selfie view: what you see moves the way your hand moves


@dataclass
class TrackerConfig:
    max_hands: int = 2
    min_detection_confidence: float = 0.6
    min_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    # One Euro filter: lower min_cutoff = smoother but laggier.
    filter_min_cutoff: float = 1.2
    filter_beta: float = 0.06
    filter_d_cutoff: float = 1.0


@dataclass
class GestureConfig:
    """Thresholds for the geometric finger/pose tests.

    Angles are degrees; distances are normalised by hand size (wrist -> middle
    MCP) so they hold at any distance from the camera.
    """

    # Finger curl angle at the PIP joint. Small angle = straight finger.
    finger_extended_max_curl: float = 42.0
    finger_folded_min_curl: float = 75.0
    thumb_extended_max_curl: float = 38.0
    # Thumb tip must clear the index MCP by this much (hand-size units) to
    # count as "out" rather than tucked across the palm.
    thumb_abduction_min: float = 0.55
    # Pinch: index tip to thumb tip distance (hand-size units).
    pinch_close_max: float = 0.42
    pinch_open_min: float = 0.60
    # OK sign: thumb/index touching while the other three stay extended.
    ok_ring_max: float = 0.48
    # Wrist -> index-tip reach that separates a rolled-up fist from a pinch.
    fist_index_reach_max: float = 1.40
    # Spread between index and pinky tips separates "palm open wide" (clear)
    # from "palm held flat" (pause).
    palm_spread_wide_min: float = 1.25
    palm_spread_flat_max: float = 1.05
    # Crossed fingers: index/middle tips closer than their MCP joints.
    crossed_tip_max: float = 0.30
    # Pointing direction cone half-angle. The thumb gets a wider cone: nobody
    # holds a thumbs-up perfectly vertical.
    point_cone_half_angle: float = 40.0
    thumb_cone_half_angle: float = 58.0
    # Thumbs up/down: hand axis mostly vertical, fingers curled.
    thumb_vertical_min_abs_cos: float = 0.62


@dataclass
class StabilizerConfig:
    """Temporal filtering so a single noisy frame never fires a command."""

    history: int = 7            # frames kept in the voting window
    min_votes: int = 5          # votes needed inside the window to accept
    min_confidence: float = 0.55
    # Seconds a discrete command must wait before it can fire again.
    default_cooldown: float = 0.65
    # Seconds a destructive gesture must be held before it commits.
    default_dwell: float = 0.0
    destructive_dwell: float = 1.4


@dataclass
class MotionConfig:
    """Dynamic (moving) gesture detection."""

    trail_seconds: float = 0.55
    # Normalised-screen-width units per second.
    swipe_min_speed: float = 0.55
    swipe_min_distance: float = 0.16
    # Static gestures only fire when the hand is slower than this.
    static_max_speed: float = 0.28
    # Pinch-zoom: change in pinch aperture per second to register.
    zoom_min_delta: float = 0.05
    zoom_gain: float = 2.2


@dataclass
class CanvasConfig:
    num_canvases: int = 4
    brush_sizes: Tuple[int, ...] = (3, 6, 10, 16, 24, 36, 52)
    default_brush_index: int = 2
    eraser_scale: float = 3.0
    undo_depth: int = 40
    palette: Tuple[BGR, ...] = (
        (60, 60, 255),    # red
        (60, 200, 255),   # orange
        (60, 255, 255),   # yellow
        (90, 220, 90),    # green
        (255, 200, 60),   # cyan-blue
        (255, 120, 60),   # blue
        (220, 100, 220),  # violet
        (255, 255, 255),  # white
        (30, 30, 30),     # near-black
    )
    backgrounds: Tuple[Tuple[str, BGR], ...] = (
        ("camera", (0, 0, 0)),
        ("dark", (24, 24, 28)),
        ("light", (242, 242, 245)),
        ("slate", (60, 52, 44)),
        ("paper", (214, 226, 238)),
    )
    brush_styles: Tuple[str, ...] = ("solid", "marker", "neon", "dotted", "calligraphy")
    tools: Tuple[str, ...] = ("brush", "line", "rectangle", "circle")
    zoom_limits: Tuple[float, float] = (0.5, 4.0)


@dataclass
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    gesture: GestureConfig = field(default_factory=GestureConfig)
    stabilizer: StabilizerConfig = field(default_factory=StabilizerConfig)
    motion: MotionConfig = field(default_factory=MotionConfig)
    canvas: CanvasConfig = field(default_factory=CanvasConfig)
    show_landmarks: bool = True
    show_debug_panel: bool = False
    window_name: str = "Vision Paint"
