"""Heads-up display: what the system thinks your hand is doing, and why."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional, Sequence

import cv2
import numpy as np

from .config import BGR, AppConfig
from .features import HandFeatures
from .hand_tracker import CONNECTIONS
from .state import MENU_ITEMS, AppState

FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE = (255, 255, 255)
DIM = (170, 170, 175)
PANEL = (28, 26, 30)
ACCENT = (255, 190, 90)
GOOD = (120, 220, 140)
WARN = (90, 150, 255)


@dataclass(frozen=True)
class Theme:
    """HUD colours for the board currently underneath them.

    The overlay used to assume a dark camera image. On a white whiteboard the
    same dark panels and light text turn into unreadable grey mush, so the
    chrome follows the board instead of fighting it.
    """

    panel: BGR = PANEL
    text: BGR = WHITE
    dim: BGR = DIM
    accent: BGR = ACCENT
    good: BGR = GOOD
    warn: BGR = WARN
    outline: BGR = (0, 0, 0)
    panel_alpha: float = 0.55


DARK_THEME = Theme()
LIGHT_THEME = Theme(
    panel=(252, 252, 253),
    text=(38, 36, 42),
    dim=(112, 110, 118),
    accent=(30, 120, 225),
    good=(60, 150, 70),
    warn=(40, 90, 210),
    outline=(255, 255, 255),
    panel_alpha=0.82,
)


def theme_for(canvas) -> Theme:
    name, color = canvas.cfg.backgrounds[canvas.background_index % len(canvas.cfg.backgrounds)]
    if name == "camera":
        return DARK_THEME
    return LIGHT_THEME if sum(color) > 380 else DARK_THEME


def _panel(img: np.ndarray, x: int, y: int, w: int, h: int, alpha: float = 0.55,
           theme: Theme = DARK_THEME) -> None:
    roi = img[max(0, y):y + h, max(0, x):x + w]
    if roi.size == 0:
        return
    overlay = np.full_like(roi, theme.panel)
    cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, roi)


def _text(img, s, x, y, scale=0.5, color=WHITE, thickness=1,
          outline: BGR = (0, 0, 0)) -> None:
    cv2.putText(img, s, (x, y), FONT, scale, outline, thickness + 2, cv2.LINE_AA)
    cv2.putText(img, s, (x, y), FONT, scale, color, thickness, cv2.LINE_AA)


def draw_landmarks(frame: np.ndarray, hands: Sequence[HandFeatures], theme: Theme = DARK_THEME) -> None:
    h, w = frame.shape[:2]
    bone = (150, 148, 156) if theme is LIGHT_THEME else (70, 70, 80)
    joint = (90, 88, 96) if theme is LIGHT_THEME else (200, 200, 205)
    for hand in hands:
        pts = hand.raw.landmarks
        for a, b in CONNECTIONS:
            pa = (int(pts[a, 0] * w), int(pts[a, 1] * h))
            pb = (int(pts[b, 0] * w), int(pts[b, 1] * h))
            cv2.line(frame, pa, pb, bone, 2, cv2.LINE_AA)
        for i, p in enumerate(pts):
            c = theme.accent if i in (4, 8, 12, 16, 20) else joint
            cv2.circle(frame, (int(p[0] * w), int(p[1] * h)), 3, c, -1, cv2.LINE_AA)


def draw_group_highlight(frame: np.ndarray, canvas, group, held: bool) -> None:
    """Outline the drawing a pinch is over, or has hold of.

    Without this, grabbing is guesswork: the grouping rule decides what counts
    as one drawing, and the user has no way to know where it drew the line
    until something moves.
    """
    if group is None:
        return
    theme = theme_for(canvas)
    x0, y0, x1, y1 = group.bounds
    p0 = canvas.board_to_screen(x0, y0)
    p1 = canvas.board_to_screen(x1, y1)
    a = (int(p0[0]), int(p0[1]))
    b = (int(p1[0]), int(p1[1]))
    color = theme.good if held else theme.accent
    thickness = 2 if held else 1

    cv2.rectangle(frame, a, b, color, thickness, cv2.LINE_AA)

    # Corner ticks read as "handle" far better than a plain box does.
    arm = max(10, min(26, abs(b[0] - a[0]) // 6))
    for cx, sx in ((a[0], 1), (b[0], -1)):
        for cy, sy in ((a[1], 1), (b[1], -1)):
            cv2.line(frame, (cx, cy), (cx + sx * arm, cy), color, thickness + 1, cv2.LINE_AA)
            cv2.line(frame, (cx, cy), (cx, cy + sy * arm), color, thickness + 1, cv2.LINE_AA)

    if held:
        _text(frame, f"holding {len(group.strokes)} strokes", a[0] + 6, a[1] - 9,
              0.45, theme.good, 1, theme.outline)


def draw_minimap(frame: np.ndarray, canvas) -> None:
    """Where the window is on the board, and where the drawings are.

    The board is bigger than the screen, so without this you can pan a drawing
    out of sight and have no idea which way to go to find it again.
    """
    theme = theme_for(canvas)
    h, w = frame.shape[:2]
    mw = 132
    mh = int(mw * canvas.board_h / max(canvas.board_w, 1))
    x, y = w - mw - 16, h - mh - 16
    _panel(frame, x - 6, y - 6, mw + 12, mh + 12, theme.panel_alpha * 0.8, theme)
    cv2.rectangle(frame, (x, y), (x + mw, y + mh), theme.dim, 1, cv2.LINE_AA)

    sx, sy = mw / canvas.board_w, mh / canvas.board_h
    for group in canvas.groups():
        gx0, gy0, gx1, gy1 = group.bounds
        cv2.rectangle(
            frame,
            (x + int(gx0 * sx), y + int(gy0 * sy)),
            (x + max(2, int(gx1 * sx)), y + max(2, int(gy1 * sy))),
            theme.dim, -1,
        )

    # The viewport, expressed back in board coordinates.
    vx0, vy0 = canvas.screen_to_board(0, 0)
    vx1, vy1 = canvas.screen_to_board(canvas.width, canvas.height)
    cv2.rectangle(
        frame,
        (x + int(vx0 * sx), y + int(vy0 * sy)),
        (x + int(vx1 * sx), y + int(vy1 * sy)),
        theme.accent, 1, cv2.LINE_AA,
    )


def draw_cursor(frame: np.ndarray, point: Optional[np.ndarray], color, size: int,
                active: bool) -> None:
    if point is None:
        return
    h, w = frame.shape[:2]
    c = (int(point[0] * w), int(point[1] * h))
    radius = max(6, size // 2)
    cv2.circle(frame, c, radius + 2, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.circle(frame, c, radius, color, 2 if not active else -1, cv2.LINE_AA)


def draw_hud(frame: np.ndarray, state: AppState, gesture_label: str, confidence: float,
             fps: float, dwell: float, dwell_label: str, ranking: List,
             pinch_mode: Optional[str] = None) -> None:
    cfg: AppConfig = state.cfg
    canvas = state.canvas
    theme = theme_for(canvas)
    h, w = frame.shape[:2]

    def text(s, x, y, scale=0.5, color=None, thickness=1):
        _text(frame, s, x, y, scale, color or theme.text, thickness, theme.outline)

    # Top bar -------------------------------------------------------------
    _panel(frame, 0, 0, w, 46, theme.panel_alpha, theme)
    text(f"{fps:4.1f} fps", 14, 29, 0.52, theme.dim)

    label = gesture_label.replace("_", " ") if gesture_label else "-"
    text(label.upper(), 108, 29, 0.62, theme.accent if gesture_label else theme.dim, 2)

    bar_x = 108 + 230
    cv2.rectangle(frame, (bar_x, 16), (bar_x + 120, 28), theme.dim, 1)
    cv2.rectangle(frame, (bar_x, 16), (bar_x + int(120 * confidence), 28), theme.good, -1)

    mode_label = {"grab": "MOVING", "pan": "PANNING"}.get(pinch_mode or "")
    if mode_label:
        text(mode_label, bar_x + 136, 29, 0.5, theme.good, 2)

    drawings = len(canvas.groups())
    info = (f"board {canvas.name}/{len(state.canvases.canvases)}   "
            f"{drawings} drawing{'s' if drawings != 1 else ''}   "
            f"{state.tool}   {state.style}   {state.brush_size}px")
    (tw, _), _ = cv2.getTextSize(info, FONT, 0.5, 1)
    text(info, w - 14 - tw, 29, 0.5, theme.dim)

    # Palette -------------------------------------------------------------
    palette = cfg.canvas.palette
    sw = 26
    px, py = 14, h - 44
    _panel(frame, px - 8, py - 8, sw * len(palette) + 16, sw + 16, theme.panel_alpha, theme)
    for i, color in enumerate(palette):
        x = px + i * sw
        cv2.rectangle(frame, (x, py), (x + sw - 6, py + sw - 6), color, -1)
        if i == state.color_index % len(palette):
            cv2.rectangle(frame, (x - 3, py - 3), (x + sw - 3, py + sw - 3), theme.text, 2)

    # Zoom / pause --------------------------------------------------------
    if canvas.zoom != 1.0:
        text(f"{canvas.zoom:.2f}x", w - 90, h - 22, 0.6, theme.accent, 2)
    if state.paused:
        _panel(frame, w // 2 - 90, 56, 180, 34, 0.85, theme)
        text("PAUSED", w // 2 - 48, 80, 0.7, theme.warn, 2)

    # Dwell ring for gestures that need holding ---------------------------
    if dwell > 0.01 and dwell_label:
        cx, cy = w // 2, h // 2
        cv2.circle(frame, (cx, cy), 46, theme.dim, 5, cv2.LINE_AA)
        cv2.ellipse(frame, (cx, cy), (46, 46), -90, 0, int(360 * dwell), theme.warn, 5, cv2.LINE_AA)
        (tw, _), _ = cv2.getTextSize(dwell_label, FONT, 0.5, 1)
        text(dwell_label, cx - tw // 2, cy + 74, 0.5, theme.warn)
        text("hold", cx - 20, cy + 6, 0.5, theme.dim)

    # Notices -------------------------------------------------------------
    now = time.time()
    notices = state.recent_notices()
    for i, notice in enumerate(reversed(notices[-3:])):
        alpha = max(0.0, 1.0 - (now - notice.at) / 2.2)
        color = {"good": theme.good, "warn": theme.warn}.get(notice.kind, theme.text)
        y = h - 74 - i * 26
        _panel(frame, 10, y - 18, 300, 24, theme.panel_alpha * alpha, theme)
        text(notice.text, 18, y, 0.5, color)

    if state.menu_open:
        _draw_menu(frame, state, theme)
    if cfg.show_debug_panel:
        _draw_debug(frame, ranking, theme)


def _draw_menu(frame: np.ndarray, state: AppState, theme: Theme) -> None:
    h, w = frame.shape[:2]
    mw, mh = 260, 30 * len(MENU_ITEMS) + 52
    x, y = w - mw - 20, 60
    _panel(frame, x, y, mw, mh, min(0.95, theme.panel_alpha + 0.2), theme)
    _text(frame, "MENU", x + 16, y + 28, 0.6, theme.accent, 2, theme.outline)
    _text(frame, "point up/down - OK to change", x + 16, y + 46, 0.38, theme.dim, 1, theme.outline)
    values = {
        "tool": state.tool,
        "color": f"#{state.color_index + 1}",
        "size": f"{state.brush_size}px",
        "style": state.style,
        "background": state.cfg.canvas.backgrounds[state.canvas.background_index][0],
        "canvas": state.canvas.name,
    }
    for i, (key, label) in enumerate(MENU_ITEMS):
        row_y = y + 74 + i * 30
        selected = i == state.menu_index
        if selected:
            cv2.rectangle(frame, (x + 8, row_y - 18), (x + mw - 8, row_y + 8), theme.dim, 1)
        _text(frame, label, x + 18, row_y, 0.5, theme.text if selected else theme.dim, 1, theme.outline)
        _text(frame, values[key], x + mw - 96, row_y, 0.5,
              theme.accent if selected else theme.dim, 1, theme.outline)


def _draw_debug(frame: np.ndarray, ranking: List, theme: Theme) -> None:
    if not ranking:
        return
    _panel(frame, 14, 56, 250, 26 * len(ranking) + 16, theme.panel_alpha, theme)
    for i, (gesture, score) in enumerate(ranking):
        y = 80 + i * 26
        _text(frame, f"{gesture.value:<16s}", 24, y, 0.45,
              theme.text if i == 0 else theme.dim, 1, theme.outline)
        cv2.rectangle(frame, (170, y - 10), (170 + int(80 * score), y),
                      theme.good if i == 0 else theme.dim, -1)


HELP_LINES = (
    ("index finger", "draw"),
    ("pinch a drawing + move", "pick it up, move it"),
    ("pinch empty space + move", "pan the board"),
    ("pinch, hold still", "select tool"),
    ("both hands pinched + apart", "zoom in / out"),
    ("two fingers, hold still", "next colour"),
    ("two fingers + swipe", "change board"),
    ("closed fist", "erase"),
    ("open palm, spread", "clear board (hold)"),
    ("flat palm, together", "pause / resume"),
    ("thumbs up", "save drawing"),
    ("thumbs down", "delete save (hold)"),
    ("crossed fingers", "undo"),
    ("point left / right", "prev / next colour"),
    ("point up / down", "brush size"),
    ("three fingers", "brush style"),
    ("four fingers", "background"),
    ("OK sign", "confirm"),
    ("thumb + little", "menu"),
    ("both palms together", "exit (hold)"),
)


def draw_help(frame: np.ndarray, theme: Theme = DARK_THEME) -> None:
    h, w = frame.shape[:2]
    pw, ph = 486, 24 * len(HELP_LINES) + 100
    x, y = (w - pw) // 2, (h - ph) // 2
    _panel(frame, x, y, pw, ph, min(0.96, theme.panel_alpha + 0.25), theme)
    _text(frame, "GESTURES", x + 20, y + 34, 0.7, theme.accent, 2, theme.outline)
    for i, (gesture, action) in enumerate(HELP_LINES):
        row = y + 60 + i * 24
        _text(frame, gesture, x + 20, row, 0.46, theme.text, 1, theme.outline)
        _text(frame, action, x + 262, row, 0.46, theme.dim, 1, theme.outline)
    _text(frame, "h close   q quit   z undo   c clear   s save",
          x + 20, y + ph - 36, 0.42, theme.dim, 1, theme.outline)
    _text(frame, "d debug   l landmarks   r reset view   +/- zoom",
          x + 20, y + ph - 16, 0.42, theme.dim, 1, theme.outline)
