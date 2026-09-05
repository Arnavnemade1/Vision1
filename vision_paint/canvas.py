"""Drawing surfaces.

Strokes are kept as vectors, not pixels. That costs a little redraw time and
buys three things the gesture set needs: undo without holding 150 MB of frame
snapshots, zoom that stays sharp instead of magnifying pixels, and saving a
drawing at a different resolution than the camera happens to be running at.

Rendering is incremental -- committed strokes live in a cached raster and only
the stroke currently under the finger is redrawn each frame.
"""

from __future__ import annotations

import json
import math
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .config import BGR, CanvasConfig

Point = Tuple[float, float]


@dataclass
class Stroke:
    """One committed mark. Points are in canvas pixel coordinates."""

    tool: str
    color: BGR
    width: int
    style: str
    points: List[Point] = field(default_factory=list)
    erase: bool = False

    def bounds(self) -> Tuple[int, int, int, int]:
        xs = [p[0] for p in self.points] or [0]
        ys = [p[1] for p in self.points] or [0]
        pad = self.width + 8
        return (
            int(min(xs) - pad), int(min(ys) - pad),
            int(max(xs) + pad), int(max(ys) + pad),
        )


def _blend_shape(layer: np.ndarray, stroke: Stroke, color: BGR, alpha: float,
                 draw) -> None:
    """Alpha-blend a shape into a BGRA layer.

    cv2 draws by overwriting, including the alpha channel, so a translucent
    marker stroke would punch a semi-transparent hole through whatever it
    crosses. Painting into a coverage mask and blending keeps translucency
    behaving like ink. The mask is cut down to the stroke's bounding box so a
    wide neon stroke does not cost a full-frame buffer every frame.
    """
    x0, y0, x1, y1 = stroke.bounds()
    h, w = layer.shape[:2]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        return
    mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    draw(mask, (-x0, -y0))

    roi = layer[y0:y1, x0:x1]
    m = (mask.astype(np.float32) / 255.0 * alpha)[:, :, None]
    roi[:, :, :3] = (roi[:, :, :3].astype(np.float32) * (1 - m)
                     + np.array(color, dtype=np.float32) * m).astype(np.uint8)
    roi[:, :, 3] = np.maximum(roi[:, :, 3], (m[:, :, 0] * 255).astype(np.uint8))


def _polyline(mask: np.ndarray, points, offset, width: int) -> None:
    ox, oy = offset
    pts = [(int(x + ox), int(y + oy)) for x, y in points]
    if len(pts) == 1:
        cv2.circle(mask, pts[0], max(1, width // 2), 255, -1, cv2.LINE_AA)
        return
    for a, b in zip(pts, pts[1:]):
        cv2.line(mask, a, b, 255, width, cv2.LINE_AA)


def _draw_stroke(layer: np.ndarray, stroke: Stroke) -> None:
    """Rasterise one stroke into a BGRA layer, in place."""
    pts = stroke.points
    if not pts:
        return
    color = (0, 0, 0, 0) if stroke.erase else (*stroke.color, 255)
    width = max(1, int(stroke.width))

    def line(a: Point, b: Point, w: int, col) -> None:
        cv2.line(layer, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), col, w, cv2.LINE_AA)

    if stroke.tool == "line" and len(pts) >= 2:
        line(pts[0], pts[-1], width, color)
        return
    if stroke.tool == "rectangle" and len(pts) >= 2:
        p0, p1 = pts[0], pts[-1]
        cv2.rectangle(layer, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])), color, width, cv2.LINE_AA)
        return
    if stroke.tool == "circle" and len(pts) >= 2:
        p0, p1 = pts[0], pts[-1]
        radius = int(math.hypot(p1[0] - p0[0], p1[1] - p0[1]))
        cv2.circle(layer, (int(p0[0]), int(p0[1])), max(1, radius), color, width, cv2.LINE_AA)
        return

    style = "solid" if stroke.erase else stroke.style
    if style == "dotted":
        spacing = max(4, width * 2)
        travelled = 0.0
        for a, b in zip(pts, pts[1:]):
            seg = math.hypot(b[0] - a[0], b[1] - a[1])
            travelled += seg
            if travelled >= spacing:
                travelled = 0.0
                cv2.circle(layer, (int(b[0]), int(b[1])), max(1, width // 2), color, -1, cv2.LINE_AA)
        if len(pts) == 1:
            cv2.circle(layer, (int(pts[0][0]), int(pts[0][1])), max(1, width // 2), color, -1, cv2.LINE_AA)
        return

    if style == "neon" and not stroke.erase:
        # Three passes, each finished before the next starts: a soft halo, the
        # solid colour, then a bright core. Interleaving them per segment lets
        # each segment paint over the previous segment's core and the line
        # comes out looking dashed.
        _blend_shape(layer, stroke, stroke.color, 0.30,
                     lambda m, o: _polyline(m, pts, o, int(width * 2.2)))
        _blend_shape(layer, stroke, stroke.color, 1.0,
                     lambda m, o: _polyline(m, pts, o, width))
        _blend_shape(layer, stroke, (255, 255, 255), 0.85,
                     lambda m, o: _polyline(m, pts, o, max(1, int(width * 0.35))))
        return

    if style == "marker" and not stroke.erase:
        _blend_shape(layer, stroke, stroke.color, 0.55,
                     lambda m, o: _polyline(m, pts, o, int(width * 1.6)))
        return

    if style == "calligraphy" and not stroke.erase:
        # Nib angle fixed at 45 degrees: strokes across the nib come out broad,
        # strokes along it come out fine, like a chisel pen.
        nib = np.array([math.cos(math.radians(45)), math.sin(math.radians(45))])
        for a, b in zip(pts, pts[1:]):
            d = np.array([b[0] - a[0], b[1] - a[1]])
            n = np.linalg.norm(d)
            if n < 1e-6:
                continue
            across = abs(float(np.dot(d / n, np.array([-nib[1], nib[0]]))))
            w = max(1, int(width * (0.25 + 0.75 * across)))
            line(a, b, w, color)
        return

    if len(pts) == 1:
        cv2.circle(layer, (int(pts[0][0]), int(pts[0][1])), max(1, width // 2), color, -1, cv2.LINE_AA)
    for a, b in zip(pts, pts[1:]):
        line(a, b, width, color)


class Canvas:
    """A single drawing: an ordered list of strokes plus its view transform."""

    def __init__(self, width: int, height: int, cfg: CanvasConfig, name: str = "1"):
        self.cfg = cfg
        self.name = name
        self.width = width
        self.height = height
        self.strokes: List[Stroke] = []
        self.active: Optional[Stroke] = None
        self.background_index = 0
        self.zoom = 1.0
        self.pan = np.array([0.0, 0.0])
        self.last_saved: Optional[Path] = None
        self._base: Optional[np.ndarray] = None      # cached raster of committed strokes
        self._base_count = -1

    # ---- geometry ---------------------------------------------------------

    def screen_to_canvas(self, x: float, y: float) -> Point:
        cx, cy = self.width / 2, self.height / 2
        return (
            cx + (x - cx - self.pan[0]) / self.zoom,
            cy + (y - cy - self.pan[1]) / self.zoom,
        )

    def set_zoom(self, zoom: float) -> None:
        lo, hi = self.cfg.zoom_limits
        self.zoom = float(np.clip(zoom, lo, hi))

    # ---- editing ----------------------------------------------------------

    def begin_stroke(self, tool: str, color: BGR, width: int, style: str, erase: bool = False) -> Stroke:
        self.active = Stroke(tool=tool, color=color, width=width, style=style, erase=erase)
        return self.active

    def extend_stroke(self, point: Point) -> None:
        if self.active is None:
            return
        pts = self.active.points
        # Drop points the smoother has already collapsed onto each other; they
        # add nothing and cost redraw time.
        if pts and math.hypot(point[0] - pts[-1][0], point[1] - pts[-1][1]) < 1.0:
            return
        pts.append(point)

    def end_stroke(self) -> None:
        if self.active is not None and self.active.points:
            self.strokes.append(self.active)
        self.active = None

    def undo(self) -> bool:
        self.active = None
        if not self.strokes:
            return False
        self.strokes.pop()
        self._invalidate()
        return True

    def clear(self) -> bool:
        self.active = None
        if not self.strokes:
            return False
        self.strokes.clear()
        self._invalidate()
        return True

    def _invalidate(self) -> None:
        self._base = None
        self._base_count = -1

    @property
    def is_empty(self) -> bool:
        return not self.strokes and self.active is None

    # ---- rendering --------------------------------------------------------

    def _render_base(self) -> np.ndarray:
        keep = self.cfg.undo_depth
        if len(self.strokes) > keep:
            # Beyond the undo horizon a stroke can never be taken back, so it is
            # safe to fold it permanently into the cached raster.
            pass
        if self._base is None or self._base_count != len(self.strokes):
            layer = np.zeros((self.height, self.width, 4), dtype=np.uint8)
            for stroke in self.strokes:
                _draw_stroke(layer, stroke)
            self._base = layer
            self._base_count = len(self.strokes)
        return self._base

    def render(self) -> np.ndarray:
        """BGRA layer in canvas space, including the in-progress stroke."""
        base = self._render_base()
        if self.active is None or not self.active.points:
            return base
        layer = base.copy()
        _draw_stroke(layer, self.active)
        return layer

    def composite(self, background: np.ndarray) -> np.ndarray:
        """Draw the canvas over `background` (a BGR frame), honouring zoom/pan."""
        layer = self.render()
        if self.zoom != 1.0 or self.pan.any():
            cx, cy = self.width / 2, self.height / 2
            m = np.array(
                [[self.zoom, 0, cx - cx * self.zoom + self.pan[0]],
                 [0, self.zoom, cy - cy * self.zoom + self.pan[1]]],
                dtype=np.float32,
            )
            layer = cv2.warpAffine(
                layer, m, (self.width, self.height),
                flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0),
            )
        alpha = (layer[:, :, 3:4].astype(np.float32)) / 255.0
        return (background.astype(np.float32) * (1 - alpha) + layer[:, :, :3].astype(np.float32) * alpha).astype(np.uint8)

    def background_frame(self, camera_frame: np.ndarray) -> np.ndarray:
        name, color = self.cfg.backgrounds[self.background_index % len(self.cfg.backgrounds)]
        if name == "camera":
            return camera_frame
        return np.full_like(camera_frame, color)

    # ---- persistence ------------------------------------------------------

    def save(self, directory: Path, camera_frame: Optional[np.ndarray] = None) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        name, color = self.cfg.backgrounds[self.background_index % len(self.cfg.backgrounds)]
        if name == "camera" and camera_frame is not None:
            background = camera_frame
        else:
            background = np.full((self.height, self.width, 3), color, dtype=np.uint8)
        image = self.composite(background)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        path = directory / f"canvas{self.name}-{stamp}.png"
        cv2.imwrite(str(path), image)
        meta = path.with_suffix(".json")
        meta.write_text(
            json.dumps(
                {
                    "canvas": self.name,
                    "saved": stamp,
                    "strokes": [
                        {
                            "tool": s.tool, "color": list(s.color), "width": s.width,
                            "style": s.style, "erase": s.erase,
                            "points": [[round(x, 1), round(y, 1)] for x, y in s.points],
                        }
                        for s in self.strokes
                    ],
                },
                indent=1,
            )
        )
        self.last_saved = path
        return path

    def delete_last_save(self, directory: Path) -> Optional[Path]:
        """Move the most recent save for this canvas to the trash folder.

        Deliberately a move rather than an unlink: a misread gesture should
        never be the end of somebody's drawing.
        """
        target = self.last_saved
        if target is None or not target.exists():
            candidates = sorted(directory.glob(f"canvas{self.name}-*.png"))
            if not candidates:
                return None
            target = candidates[-1]
        trash = directory / ".trash"
        trash.mkdir(parents=True, exist_ok=True)
        moved = trash / target.name
        shutil.move(str(target), str(moved))
        meta = target.with_suffix(".json")
        if meta.exists():
            shutil.move(str(meta), str(trash / meta.name))
        if self.last_saved == target:
            self.last_saved = None
        return moved


class CanvasSet:
    """The stack of canvases the two-finger swipe moves between."""

    def __init__(self, width: int, height: int, cfg: CanvasConfig):
        self.cfg = cfg
        self.canvases = [Canvas(width, height, cfg, name=str(i + 1)) for i in range(cfg.num_canvases)]
        self.index = 0

    @property
    def current(self) -> Canvas:
        return self.canvases[self.index]

    def step(self, delta: int) -> Canvas:
        self.current.active = None
        self.index = (self.index + delta) % len(self.canvases)
        return self.current

    def resize(self, width: int, height: int) -> None:
        for canvas in self.canvases:
            if canvas.width != width or canvas.height != height:
                canvas.width, canvas.height = width, height
                canvas._invalidate()
