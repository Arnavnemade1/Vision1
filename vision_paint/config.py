"""Tunable constants for the vision drawing system.

Everything a person might want to adjust for their camera, lighting or hand
size lives here so the recognition code stays free of magic numbers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "hand_landmarker.task"
SAVE_DIR = PROJECT_ROOT / "saves"
CALIBRATION_PATH = PROJECT_ROOT / "calibration.json"

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
    # One Euro filter on the landmarks: lower min_cutoff = smoother but laggier.
    filter_min_cutoff: float = 1.2
    filter_beta: float = 0.06
    filter_d_cutoff: float = 1.0
    # A second, gentler filter on the drawing point alone. Ink shows every bit
    # of tremor the landmarks carry, so the pen wants more smoothing than the
    # pose recognition does -- and recognition wants the responsiveness back.
    draw_min_cutoff: float = 0.55
    draw_beta: float = 0.035
    # Exponential smoothing time constant for the scalar features that feed the
    # classifier, in seconds. Averages out single-frame landmark noise without
    # the lag a longer voting window would add.
    feature_smoothing: float = 0.075
    # Hands whose landmarks run past the frame edge are badly conditioned; their
    # confidence is scaled down rather than trusted at face value.
    edge_margin: float = 0.02


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
    # Two-handed zoom: the span between the two pinch points is compared with
    # the span when the gesture started, so zoom tracks the hands absolutely
    # rather than integrating a rate. Spans below this are too small for the
    # ratio to be stable.
    two_hand_zoom_min_span: float = 0.06
    # Grab-and-drag: how far a pinched hand must travel (fraction of screen
    # width) before the pinch is read as a drag rather than a held pose.
    grab_min_distance: float = 0.035
    # How long a pinch waits before it can settle into "held still" and cycle
    # the tool. Long enough that reaching for a drawing is never mistaken for it.
    grab_intent_seconds: float = 0.28


@dataclass
class CanvasConfig:
    num_canvases: int = 4
    # The board is bigger than the window, which is the whole point of being
    # able to shove a drawing aside: there is somewhere to shove it to.
    board_scale: float = 2.0
    # Strokes whose bounding boxes come this close are treated as one drawing.
    # Expressed as a fraction of the viewport width, because how far apart
    # people leave their strokes scales with how big the board looks to them,
    # not with a pixel count.
    group_gap_ratio: float = 0.060
    # How far from a drawing a pinch still counts as grabbing it, same units.
    grab_reach_ratio: float = 0.075
    # Minimum spacing between recorded stroke points, in board pixels.
    point_spacing: float = 2.5
    # Board rulings, in board pixels. 0 disables them.
    grid_spacing: int = 64
    grid_backgrounds: Tuple[str, ...] = ("light", "paper", "dark", "slate")
    default_background: int = 2   # start on the light whiteboard, not the camera
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
        ("light", (246, 246, 248)),
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
    calibration: Optional[str] = None   # path of the calibration actually applied

    def apply_calibration(self, data: Dict[str, Any]) -> list:
        """Overlay measured thresholds onto the gesture config.

        Only keys that exist on GestureConfig are taken, so an old or hand-edited
        calibration file can never inject unknown settings.
        """
        known = {f.name for f in fields(GestureConfig)}
        applied = []
        for key, value in (data.get("gesture") or {}).items():
            if key in known and isinstance(value, (int, float)):
                setattr(self.gesture, key, float(value))
                applied.append(key)
        return applied

    def load_calibration(self, path: Path = CALIBRATION_PATH) -> list:
        if not path.exists():
            return []
        applied = self.apply_calibration(json.loads(path.read_text()))
        if applied:
            self.calibration = str(path)
        return applied
