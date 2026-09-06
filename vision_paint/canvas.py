"""The whiteboard.

Strokes are kept as vectors, not pixels. That costs a little redraw time and
buys four things this interaction needs: undo without holding 150 MB of frame
snapshots, zoom that stays sharp instead of magnifying pixels, saving at a
resolution the camera never ran at, and -- the reason drawings can be picked up
and shoved aside -- the ability to move a mark after it has been made.

Three coordinate spaces are in play:

  screen   viewport pixels, what the user sees and where the hand is
  board    the whiteboard itself, larger than the viewport so there is room to
           push a drawing aside and start another beside it
  view     the board window currently on screen: pan offset plus zoom

Rendering is incremental. Committed strokes live in a cached board-sized
raster; the stroke under the finger and any group being dragged are drawn on
top each frame, so neither drawing nor dragging ever re-rasterises the board.
"""

from __future__ import annotations

import json
import math
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .config import BGR, CanvasConfig

Point = Tuple[float, float]

_STROKE_SEQ = [0]


def _next_id() -> int:
    _STROKE_SEQ[0] += 1
    return _STROKE_SEQ[0]


def _chaikin(points: Sequence[Point], iterations: int = 2) -> List[Point]:
    """Corner-cutting subdivision.

    Freehand input arrives as a polyline of hand positions, and a polyline
    looks like one: visible corners at every sample. Chaikin replaces each
    corner with two points a quarter and three quarters along, which both
    smooths the jitter and quadruples the sample density, so the result draws
    as a curve rather than a chain of segments. Two passes is the point where
    further smoothing stops being visible and only costs time.
    """
    pts = list(points)
    if len(pts) < 3:
        return pts
    for _ in range(iterations):
        if len(pts) < 3:
            break
        out: List[Point] = [pts[0]]
        for a, b in zip(pts, pts[1:]):
            out.append((a[0] * 0.75 + b[0] * 0.25, a[1] * 0.75 + b[1] * 0.25))
            out.append((a[0] * 0.25 + b[0] * 0.75, a[1] * 0.25 + b[1] * 0.75))
        out.append(pts[-1])
        pts = out
    return pts


@dataclass(eq=False)
class Stroke:
    """One mark on the board. Points are board coordinates.

    `offset` is what makes a stroke movable: dragging a drawing writes a
    translation here rather than rewriting every point, so a move is one vector
    per stroke to apply, to undo, and to redraw.
    """

    tool: str
    color: BGR
    width: int
    style: str
    points: List[Point] = field(default_factory=list)
    erase: bool = False
    offset: np.ndarray = field(default_factory=lambda: np.zeros(2))
    id: int = field(default_factory=_next_id)

    _smooth_cache: Optional[List[Point]] = field(default=None, repr=False, compare=False)
    _smooth_len: int = field(default=-1, repr=False, compare=False)

    @property
    def smooth(self) -> bool:
        return self.tool == "brush" and not self.erase

    def render_points(self) -> List[Point]:
        """Points as they should be drawn: smoothed, then translated."""
        if self.smooth:
            if self._smooth_cache is None or self._smooth_len != len(self.points):
                self._smooth_cache = _chaikin(self.points)
                self._smooth_len = len(self.points)
            pts = self._smooth_cache
        else:
            pts = self.points
        ox, oy = float(self.offset[0]), float(self.offset[1])
        if ox or oy:
            return [(x + ox, y + oy) for x, y in pts]
        return list(pts)

    def bounds(self, pad: int = 0) -> Tuple[int, int, int, int]:
        pts = self.points
        if not pts:
            return (0, 0, 0, 0)
        ox, oy = float(self.offset[0]), float(self.offset[1])
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        margin = self.width + 8 + pad
        if self.tool == "circle" and len(pts) >= 2:
            # A circle is stored as centre plus a rim point, so its extent is
            # the radius in every direction, not the span of the two points.
            r = math.hypot(pts[-1][0] - pts[0][0], pts[-1][1] - pts[0][1])
            return (
                int(pts[0][0] + ox - r - margin), int(pts[0][1] + oy - r - margin),
                int(pts[0][0] + ox + r + margin), int(pts[0][1] + oy + r + margin),
            )
        return (
            int(min(xs) + ox - margin), int(min(ys) + oy - margin),
            int(max(xs) + ox + margin), int(max(ys) + oy + margin),
        )


def _rects_touch(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int], gap: int) -> bool:
    return not (
        a[2] + gap < b[0] or b[2] + gap < a[0]
        or a[3] + gap < b[1] or b[3] + gap < a[1]
    )


def _union_bounds(boxes: Iterable[Tuple[int, int, int, int]]) -> Tuple[int, int, int, int]:
    boxes = list(boxes)
    return (
        min(b[0] for b in boxes), min(b[1] for b in boxes),
        max(b[2] for b in boxes), max(b[3] for b in boxes),
    )


@dataclass
class Group:
    """A cluster of strokes the user thinks of as one drawing."""

    strokes: List[Stroke]
    bounds: Tuple[int, int, int, int]

    def contains(self, x: float, y: float, pad: int = 0) -> bool:
        x0, y0, x1, y1 = self.bounds
        return x0 - pad <= x <= x1 + pad and y0 - pad <= y <= y1 + pad

    def distance_to(self, x: float, y: float) -> float:
        x0, y0, x1, y1 = self.bounds
        dx = max(x0 - x, 0.0, x - x1)
        dy = max(y0 - y, 0.0, y - y1)
        return math.hypot(dx, dy)


# ---- undo history ---------------------------------------------------------


class Op:
    """One reversible edit."""

    def undo(self, canvas: "Canvas") -> str:
        raise NotImplementedError


@dataclass
class AddStroke(Op):
    stroke: Stroke

    def undo(self, canvas: "Canvas") -> str:
        try:
            canvas.strokes.remove(self.stroke)
        except ValueError:
            return "nothing to undo"
        return "stroke removed"


@dataclass
class MoveStrokes(Op):
    strokes: List[Stroke]
    delta: np.ndarray

    def undo(self, canvas: "Canvas") -> str:
        for s in self.strokes:
            s.offset = s.offset - self.delta
        return "move undone"


@dataclass
class ClearBoard(Op):
    strokes: List[Stroke]

    def undo(self, canvas: "Canvas") -> str:
        canvas.strokes.extend(self.strokes)
        return f"{len(self.strokes)} strokes restored"


def _draw_stroke(layer: np.ndarray, stroke: Stroke, transform=None, scale: float = 1.0) -> None:
    """Rasterise one stroke into a BGRA layer, in place.

    With `transform` the stroke is drawn straight into view space, which is how
    the in-progress stroke and any dragged drawing reach the screen without
    copying the whole board raster every frame.
    """
    pts = stroke.render_points()
    if not pts:
        return
    if transform is not None:
        pts = [transform(x, y) for x, y in pts]
    color = (0, 0, 0, 0) if stroke.erase else (*stroke.color, 255)
    width = max(1, int(round(stroke.width * scale)))

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
            travelled += math.hypot(b[0] - a[0], b[1] - a[1])
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
        pad = int(width * 1.6)
        _blend_shape(layer, pts, pad, stroke.color, 0.30,
                     lambda m, o: _polyline(m, pts, o, int(width * 2.2)))
        _blend_shape(layer, pts, pad, stroke.color, 1.0,
                     lambda m, o: _polyline(m, pts, o, width))
        _blend_shape(layer, pts, pad, (255, 255, 255), 0.85,
                     lambda m, o: _polyline(m, pts, o, max(1, int(width * 0.35))))
        return

    if style == "marker" and not stroke.erase:
        _blend_shape(layer, pts, int(width * 1.2), stroke.color, 0.55,
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


def _blend_shape(layer: np.ndarray, pts: Sequence[Point], pad: int, color: BGR,
                 alpha: float, draw) -> None:
    """Alpha-blend a shape into a BGRA layer.

    cv2 draws by overwriting, including the alpha channel, so a translucent
    marker stroke would punch a semi-transparent hole through whatever it
    crosses. Painting into a coverage mask and blending keeps translucency
    behaving like ink. The mask is cut down to the shape's bounding box so a
    wide neon stroke does not cost a full-board buffer every frame.
    """
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x0, y0 = int(min(xs)) - pad, int(min(ys)) - pad
    x1, y1 = int(max(xs)) + pad, int(max(ys)) + pad
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


def _alpha_over(layer: np.ndarray, background: np.ndarray,
                region: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
    """Composite a BGRA layer over a BGR frame, returning a new frame.

    Only `region` is blended. Ink covers a small part of a whiteboard most of
    the time, and the float conversion is the expensive part, so restricting it
    to the rectangle that actually contains ink is worth far more than any
    cleverness inside the blend itself. Masking per pixel instead was measured
    and is four times slower: boolean fancy-indexing over a 720p array costs
    more than the arithmetic it avoids.

    Always returns a fresh array. The background may be the camera thread's
    live frame, and the caller draws its overlay onto whatever comes back.
    """
    out = background.copy()
    if region is None:
        x0, y0, x1, y1 = 0, 0, layer.shape[1], layer.shape[0]
    else:
        x0, y0, x1, y1 = region
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(layer.shape[1], x1), min(layer.shape[0], y1)
    if x1 <= x0 or y1 <= y0:
        return out

    patch = layer[y0:y1, x0:x1]
    under = out[y0:y1, x0:x1]
    alpha = patch[:, :, 3]
    # Integer blend through cv2 rather than float32 numpy. Bit-identical output,
    # measured at a little over half the time, and the blend is the single
    # largest cost in the frame.
    a3 = cv2.cvtColor(alpha, cv2.COLOR_GRAY2BGR)
    inv3 = cv2.cvtColor(cv2.bitwise_not(alpha), cv2.COLOR_GRAY2BGR)
    out[y0:y1, x0:x1] = cv2.add(
        cv2.multiply(under, inv3, scale=1 / 255.0),
        cv2.multiply(patch[:, :, :3], a3, scale=1 / 255.0),
    )
    return out


class Canvas:
    """One board: its strokes, its view window, and its undo history."""

    def __init__(self, width: int, height: int, cfg: CanvasConfig, name: str = "1"):
        self.cfg = cfg
        self.name = name
        self.width = width
        self.height = height
        self.board_w = int(width * cfg.board_scale)
        self.board_h = int(height * cfg.board_scale)
        self.group_gap = int(width * cfg.group_gap_ratio)
        self.grab_reach = width * cfg.grab_reach_ratio

        self.strokes: List[Stroke] = []
        self.active: Optional[Stroke] = None
        self.floating: List[Stroke] = []   # lifted out of the cache while dragging
        self.history: List[Op] = []

        self.background_index = cfg.default_background
        self.zoom = 1.0
        self.pan = np.array([0.0, 0.0])
        self.last_saved: Optional[Path] = None

        self._base: Optional[np.ndarray] = None
        self._base_key: Tuple = ()
        self._base_bounds: Optional[Tuple[int, int, int, int]] = None
        self._bg_cache: Optional[np.ndarray] = None
        self._bg_key: Tuple = ()
        self._groups: Optional[List[Group]] = None
        self._groups_key: Tuple = ()

    # ---- geometry ---------------------------------------------------------

    @property
    def board_center(self) -> Tuple[float, float]:
        return self.board_w / 2, self.board_h / 2

    def screen_to_board(self, x: float, y: float) -> Point:
        bcx, bcy = self.board_center
        return (
            bcx + (x - self.width / 2 - self.pan[0]) / self.zoom,
            bcy + (y - self.height / 2 - self.pan[1]) / self.zoom,
        )

    def board_to_screen(self, x: float, y: float) -> Point:
        bcx, bcy = self.board_center
        return (
            (x - bcx) * self.zoom + self.width / 2 + self.pan[0],
            (y - bcy) * self.zoom + self.height / 2 + self.pan[1],
        )

    # Kept so older call sites and saved sessions keep working.
    screen_to_canvas = screen_to_board

    def set_zoom(self, zoom: float) -> None:
        lo, hi = self.cfg.zoom_limits
        self.zoom = float(np.clip(zoom, lo, hi))
        self._clamp_pan()

    def pan_by(self, dx: float, dy: float) -> None:
        self.pan = self.pan + np.array([dx, dy], dtype=float)
        self._clamp_pan()

    def reset_view(self) -> None:
        self.zoom = 1.0
        self.pan = np.array([0.0, 0.0])

    def _clamp_pan(self) -> None:
        """Keep the viewport inside the board so panning cannot get lost."""
        slack_x = max(0.0, (self.board_w * self.zoom - self.width) / 2)
        slack_y = max(0.0, (self.board_h * self.zoom - self.height) / 2)
        self.pan = np.array([
            float(np.clip(self.pan[0], -slack_x, slack_x)),
            float(np.clip(self.pan[1], -slack_y, slack_y)),
        ])

    # ---- drawing ----------------------------------------------------------

    def begin_stroke(self, tool: str, color: BGR, width: int, style: str, erase: bool = False) -> Stroke:
        self.active = Stroke(tool=tool, color=color, width=width, style=style, erase=erase)
        return self.active

    def extend_stroke(self, point: Point) -> None:
        if self.active is None:
            return
        pts = self.active.points
        # Points closer together than this add nothing the smoother cannot
        # reconstruct, and every one of them costs a line segment per redraw.
        if pts and math.hypot(point[0] - pts[-1][0], point[1] - pts[-1][1]) < self.cfg.point_spacing:
            return
        pts.append(point)

    def end_stroke(self) -> None:
        if self.active is not None and self.active.points:
            self.strokes.append(self.active)
            self._push(AddStroke(self.active))
        self.active = None

    # ---- moving -----------------------------------------------------------

    def groups(self) -> List[Group]:
        """Cluster strokes into the drawings a person would call separate.

        Union-find over bounding boxes that come within `group_gap` of each
        other. It is a blunt rule, but it matches how people actually draw:
        the strokes of one doodle overlap or nearly touch, and a doodle drawn
        elsewhere on the board does not.
        """
        key = (len(self.strokes), tuple(s.id for s in self.strokes),
               tuple((float(s.offset[0]), float(s.offset[1])) for s in self.strokes),
               tuple(len(s.points) for s in self.strokes))
        if self._groups is not None and self._groups_key == key:
            return self._groups

        drawable = [s for s in self.strokes if not s.erase and s.points]
        parent = list(range(len(drawable)))

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(i: int, j: int) -> None:
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[rj] = ri

        boxes = [s.bounds() for s in drawable]
        gap = self.group_gap
        for i in range(len(drawable)):
            for j in range(i + 1, len(drawable)):
                if _rects_touch(boxes[i], boxes[j], gap):
                    union(i, j)

        buckets: Dict[int, List[int]] = {}
        for i in range(len(drawable)):
            buckets.setdefault(find(i), []).append(i)

        groups = [
            Group([drawable[i] for i in idxs], _union_bounds([boxes[i] for i in idxs]))
            for idxs in buckets.values()
        ]
        groups.sort(key=lambda g: (g.bounds[2] - g.bounds[0]) * (g.bounds[3] - g.bounds[1]))
        self._groups, self._groups_key = groups, key
        return groups

    def group_at(self, x: float, y: float, reach: Optional[float] = None) -> Optional[Group]:
        """The drawing under (or nearest to) a board point, if one is close."""
        reach = self.grab_reach if reach is None else reach
        best, best_d = None, reach
        for group in self.groups():
            d = group.distance_to(x, y)
            if d <= best_d:
                best, best_d = group, d
        return best

    def lift(self, group: Group) -> None:
        """Take a group out of the cached raster for the duration of a drag."""
        self.floating = list(group.strokes)
        self._invalidate_base()

    def drag_to(self, group: Group, delta: np.ndarray) -> None:
        for stroke in group.strokes:
            stroke.offset = stroke.offset + delta
        self._invalidate_groups()

    def drop(self, group: Group, total_delta: np.ndarray) -> None:
        self.floating = []
        if float(np.linalg.norm(total_delta)) > 0.5:
            self._push(MoveStrokes(list(group.strokes), np.array(total_delta, dtype=float)))
        self._invalidate()

    # ---- history ----------------------------------------------------------

    def _push(self, op: Op) -> None:
        self.history.append(op)
        if len(self.history) > self.cfg.undo_depth:
            self.history.pop(0)
        self._invalidate()

    def undo(self) -> Optional[str]:
        self.active = None
        self.floating = []
        if not self.history:
            return None
        what = self.history.pop().undo(self)
        self._invalidate()
        return what

    def clear(self) -> bool:
        self.active = None
        self.floating = []
        if not self.strokes:
            return False
        self._push(ClearBoard(list(self.strokes)))
        self.strokes.clear()
        self._invalidate()
        return True

    def _invalidate(self) -> None:
        self._invalidate_base()
        self._invalidate_groups()

    def _invalidate_base(self) -> None:
        self._base = None
        self._base_key = ()
        self._base_bounds = None

    def _invalidate_groups(self) -> None:
        self._groups = None
        self._groups_key = ()

    @property
    def is_empty(self) -> bool:
        return not self.strokes and self.active is None

    # ---- rendering --------------------------------------------------------

    def _render_base(self) -> np.ndarray:
        floating = {id(s) for s in self.floating}
        key = (
            len(self.strokes),
            tuple(s.id for s in self.strokes if id(s) not in floating),
            tuple(len(s.points) for s in self.strokes if id(s) not in floating),
            tuple((float(s.offset[0]), float(s.offset[1]))
                  for s in self.strokes if id(s) not in floating),
        )
        if self._base is None or self._base_key != key:
            layer = np.zeros((self.board_h, self.board_w, 4), dtype=np.uint8)
            boxes = []
            for stroke in self.strokes:
                if id(stroke) not in floating:
                    _draw_stroke(layer, stroke)
                    boxes.append(stroke.bounds())
            self._base = layer
            self._base_key = key
            self._base_bounds = _union_bounds(boxes) if boxes else None
        return self._base

    def _overlays(self) -> List[Stroke]:
        overlays = list(self.floating)
        if self.active is not None and self.active.points:
            overlays.append(self.active)
        return overlays

    def render(self) -> np.ndarray:
        """BGRA board layer at full board size, live strokes included.

        The whole board in one array, which is convenient to reason about and
        to assert against. Nothing on the frame path calls it: `composite`
        works viewport-sized throughout, precisely to avoid building this.
        """
        base = self._render_base()
        overlays = self._overlays()
        if not overlays:
            return base
        layer = base.copy()
        for stroke in overlays:
            _draw_stroke(layer, stroke)
        return layer

    def _view_matrix(self) -> np.ndarray:
        bcx, bcy = self.board_center
        return np.array(
            [[self.zoom, 0, self.width / 2 + self.pan[0] - bcx * self.zoom],
             [0, self.zoom, self.height / 2 + self.pan[1] - bcy * self.zoom]],
            dtype=np.float32,
        )

    def _viewport_layer(self, writable: bool) -> np.ndarray:
        """The committed board, cropped and scaled to the viewport.

        At the default view -- no zoom, whole-pixel pan -- the viewport is just
        a window into the board raster, so this is a memory slice rather than a
        resample. That is the common case and it costs nothing.
        """
        base = self._render_base()
        integral_pan = abs(self.pan - np.round(self.pan)).max() < 1e-6
        if abs(self.zoom - 1.0) < 1e-9 and integral_pan:
            bcx, bcy = self.board_center
            x0 = int(round(bcx - self.width / 2 - self.pan[0]))
            y0 = int(round(bcy - self.height / 2 - self.pan[1]))
            x0 = max(0, min(x0, self.board_w - self.width))
            y0 = max(0, min(y0, self.board_h - self.height))
            view = base[y0:y0 + self.height, x0:x0 + self.width]
            return view.copy() if writable else view
        return cv2.warpAffine(
            base, self._view_matrix(), (self.width, self.height),
            flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0),
        )

    def _dirty_region(self, overlays: Sequence[Stroke]) -> Optional[Tuple[int, int, int, int]]:
        """The screen rectangle that could contain ink, or None if nothing can."""
        boxes = [b for b in (self._base_bounds,) if b is not None]
        boxes.extend(s.bounds() for s in overlays)
        if not boxes:
            return None
        x0, y0, x1, y1 = _union_bounds(boxes)
        p0 = self.board_to_screen(x0, y0)
        p1 = self.board_to_screen(x1, y1)
        return (int(p0[0]) - 2, int(p0[1]) - 2, int(p1[0]) + 3, int(p1[1]) + 3)

    def composite(self, background: np.ndarray) -> np.ndarray:
        """Draw the visible part of the board over `background` (a BGR frame).

        The board is four times the area of the window, so nothing here touches
        it at full size: the committed raster is cropped or warped down to the
        viewport, live strokes are drawn straight into view space, and only the
        rectangle containing ink gets blended.
        """
        overlays = self._overlays()
        layer = self._viewport_layer(writable=bool(overlays))
        for stroke in overlays:
            _draw_stroke(layer, stroke, transform=self.board_to_screen, scale=self.zoom)
        return _alpha_over(layer, background, self._dirty_region(overlays))

    def background_frame(self, camera_frame: np.ndarray) -> np.ndarray:
        """The surface the ink sits on.

        A ruled board only changes when the view or the background does, so it
        is cached: redrawing forty anti-aliased grid lines every frame was pure
        waste. The camera background is passed straight through -- it is
        different every frame by definition.
        """
        name, color = self.cfg.backgrounds[self.background_index % len(self.cfg.backgrounds)]
        if name == "camera":
            return camera_frame
        key = (name, self.zoom, float(self.pan[0]), float(self.pan[1]),
               camera_frame.shape[:2])
        if self._bg_cache is None or self._bg_key != key:
            board = np.full_like(camera_frame, color)
            if self.cfg.grid_spacing > 0 and name in self.cfg.grid_backgrounds:
                _draw_grid(board, self, color)
            self._bg_cache, self._bg_key = board, key
        return self._bg_cache

    # ---- persistence ------------------------------------------------------

    def save(self, directory: Path, camera_frame: Optional[np.ndarray] = None) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        name, color = self.cfg.backgrounds[self.background_index % len(self.cfg.backgrounds)]
        if name == "camera" and camera_frame is not None:
            background = camera_frame
        else:
            background = np.full((self.height, self.width, 3), color, dtype=np.uint8)
            if self.cfg.grid_spacing > 0 and name in self.cfg.grid_backgrounds:
                _draw_grid(background, self, color)
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
                    "board": [self.board_w, self.board_h],
                    "strokes": [
                        {
                            "tool": s.tool, "color": list(s.color), "width": s.width,
                            "style": s.style, "erase": s.erase,
                            "offset": [round(float(s.offset[0]), 1), round(float(s.offset[1]), 1)],
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


def _draw_grid(frame: np.ndarray, canvas: Canvas, base: BGR) -> None:
    """Faint board rulings, drawn in view space so they pan and zoom with it."""
    h, w = frame.shape[:2]
    step = canvas.cfg.grid_spacing * canvas.zoom
    if step < 8:
        return
    tint = tuple(int(np.clip(c - 18 if sum(base) > 380 else c + 16, 0, 255)) for c in base)
    ox, oy = canvas.board_to_screen(0, 0)
    start_x = ox % step
    start_y = oy % step
    x = start_x
    while x < w:
        cv2.line(frame, (int(x), 0), (int(x), h), tint, 1, cv2.LINE_AA)
        x += step
    y = start_y
    while y < h:
        cv2.line(frame, (0, int(y)), (w, int(y)), tint, 1, cv2.LINE_AA)
        y += step


class CanvasSet:
    """The stack of boards the two-finger swipe moves between."""

    def __init__(self, width: int, height: int, cfg: CanvasConfig):
        self.cfg = cfg
        self.canvases = [Canvas(width, height, cfg, name=str(i + 1)) for i in range(cfg.num_canvases)]
        self.index = 0

    @property
    def current(self) -> Canvas:
        return self.canvases[self.index]

    def step(self, delta: int) -> Canvas:
        self.current.active = None
        self.current.floating = []
        self.index = (self.index + delta) % len(self.canvases)
        return self.current

    def resize(self, width: int, height: int) -> None:
        for canvas in self.canvases:
            if canvas.width != width or canvas.height != height:
                canvas.width, canvas.height = width, height
                canvas.board_w = int(width * self.cfg.board_scale)
                canvas.board_h = int(height * self.cfg.board_scale)
                canvas.group_gap = int(width * self.cfg.group_gap_ratio)
                canvas.grab_reach = width * self.cfg.grab_reach_ratio
                canvas._invalidate()
