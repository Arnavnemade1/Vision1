"""Heads-up display: what the system thinks your hand is doing, and why."""

from __future__ import annotations

import time
from typing import List, Optional, Sequence

import cv2
import numpy as np

from .config import AppConfig
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


def _panel(img: np.ndarray, x: int, y: int, w: int, h: int, alpha: float = 0.55) -> None:
    roi = img[max(0, y):y + h, max(0, x):x + w]
    if roi.size == 0:
        return
    overlay = np.full_like(roi, PANEL)
    cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, roi)


def _text(img, s, x, y, scale=0.5, color=WHITE, thickness=1) -> None:
    cv2.putText(img, s, (x, y), FONT, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, s, (x, y), FONT, scale, color, thickness, cv2.LINE_AA)


def draw_landmarks(frame: np.ndarray, hands: Sequence[HandFeatures]) -> None:
    h, w = frame.shape[:2]
    for hand in hands:
        pts = hand.raw.landmarks
        for a, b in CONNECTIONS:
            pa = (int(pts[a, 0] * w), int(pts[a, 1] * h))
            pb = (int(pts[b, 0] * w), int(pts[b, 1] * h))
            cv2.line(frame, pa, pb, (70, 70, 80), 2, cv2.LINE_AA)
        for i, p in enumerate(pts):
            c = ACCENT if i in (4, 8, 12, 16, 20) else (200, 200, 205)
            cv2.circle(frame, (int(p[0] * w), int(p[1] * h)), 3, c, -1, cv2.LINE_AA)


def draw_cursor(frame: np.ndarray, point: Optional[np.ndarray], color, size: int, active: bool) -> None:
    if point is None:
        return
    h, w = frame.shape[:2]
    c = (int(point[0] * w), int(point[1] * h))
    radius = max(6, size // 2)
    cv2.circle(frame, c, radius + 2, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.circle(frame, c, radius, color, 2 if not active else -1, cv2.LINE_AA)


def draw_hud(frame: np.ndarray, state: AppState, gesture_label: str, confidence: float,
             fps: float, dwell: float, dwell_label: str, ranking: List) -> None:
    cfg: AppConfig = state.cfg
    h, w = frame.shape[:2]

    # Top bar -------------------------------------------------------------
    _panel(frame, 0, 0, w, 46)
    _text(frame, f"{fps:4.1f} fps", 14, 29, 0.52, DIM)

    label = gesture_label.replace("_", " ") if gesture_label else "-"
    _text(frame, label.upper(), 108, 29, 0.62, ACCENT if gesture_label else DIM, 2)

    bar_x = 108 + 230
    cv2.rectangle(frame, (bar_x, 16), (bar_x + 120, 28), (60, 60, 66), -1)
    cv2.rectangle(frame, (bar_x, 16), (bar_x + int(120 * confidence), 28), GOOD, -1)

    right = w - 14
    info = f"canvas {state.canvas.name}/{len(state.canvases.canvases)}   {state.tool}   {state.style}   {state.brush_size}px"
    (tw, _), _ = cv2.getTextSize(info, FONT, 0.5, 1)
    _text(frame, info, right - tw, 29, 0.5, DIM)

    # Palette -------------------------------------------------------------
    palette = cfg.canvas.palette
    sw = 26
    px, py = 14, h - 44
    _panel(frame, px - 8, py - 8, sw * len(palette) + 16, sw + 16)
    for i, color in enumerate(palette):
        x = px + i * sw
        cv2.rectangle(frame, (x, py), (x + sw - 6, py + sw - 6), color, -1)
        if i == state.color_index % len(palette):
            cv2.rectangle(frame, (x - 3, py - 3), (x + sw - 3, py + sw - 3), WHITE, 2)

    # Zoom / pause --------------------------------------------------------
    if state.canvas.zoom != 1.0:
        _text(frame, f"{state.canvas.zoom:.2f}x", w - 90, h - 22, 0.6, ACCENT, 2)
    if state.paused:
        _panel(frame, w // 2 - 90, 56, 180, 34, 0.7)
        _text(frame, "PAUSED", w // 2 - 48, 80, 0.7, WARN, 2)

    # Dwell ring for gestures that need holding ---------------------------
    if dwell > 0.01 and dwell_label:
        cx, cy = w // 2, h // 2
        cv2.circle(frame, (cx, cy), 46, (60, 60, 66), 5, cv2.LINE_AA)
        cv2.ellipse(frame, (cx, cy), (46, 46), -90, 0, int(360 * dwell), WARN, 5, cv2.LINE_AA)
        (tw, _), _ = cv2.getTextSize(dwell_label, FONT, 0.5, 1)
        _text(frame, dwell_label, cx - tw // 2, cy + 74, 0.5, WARN)
        _text(frame, "hold", cx - 20, cy + 6, 0.5, DIM)

    # Notices -------------------------------------------------------------
    now = time.time()
    notices = state.recent_notices()
    for i, notice in enumerate(reversed(notices[-3:])):
        age = now - notice.at
        alpha = max(0.0, 1.0 - age / 2.2)
        color = {"good": GOOD, "warn": WARN}.get(notice.kind, WHITE)
        y = h - 74 - i * 26
        _panel(frame, 10, y - 18, 300, 24, 0.4 * alpha)
        _text(frame, notice.text, 18, y, 0.5, color)

    if state.menu_open:
        _draw_menu(frame, state)
    if cfg.show_debug_panel:
        _draw_debug(frame, ranking)


def _draw_menu(frame: np.ndarray, state: AppState) -> None:
    h, w = frame.shape[:2]
    mw, mh = 260, 30 * len(MENU_ITEMS) + 52
    x, y = w - mw - 20, 60
    _panel(frame, x, y, mw, mh, 0.78)
    _text(frame, "MENU", x + 16, y + 28, 0.6, ACCENT, 2)
    _text(frame, "point up/down - OK to change", x + 16, y + 46, 0.38, DIM)
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
            cv2.rectangle(frame, (x + 8, row_y - 18), (x + mw - 8, row_y + 8), (60, 58, 66), -1)
        _text(frame, label, x + 18, row_y, 0.5, WHITE if selected else DIM)
        _text(frame, values[key], x + mw - 96, row_y, 0.5, ACCENT if selected else DIM)


def _draw_debug(frame: np.ndarray, ranking: List) -> None:
    if not ranking:
        return
    _panel(frame, 14, 56, 250, 26 * len(ranking) + 16, 0.6)
    for i, (gesture, score) in enumerate(ranking):
        y = 80 + i * 26
        _text(frame, f"{gesture.value:<16s}", 24, y, 0.45, WHITE if i == 0 else DIM)
        cv2.rectangle(frame, (170, y - 10), (170 + int(80 * score), y), GOOD if i == 0 else (90, 90, 96), -1)


HELP_LINES = (
    ("index finger", "draw"),
    ("two fingers (still)", "next colour"),
    ("two fingers + swipe", "change canvas"),
    ("closed fist", "erase"),
    ("pinch (still)", "select tool"),
    ("pinch open/close", "zoom"),
    ("open palm, spread", "clear screen (hold)"),
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


def draw_help(frame: np.ndarray) -> None:
    h, w = frame.shape[:2]
    pw, ph = 430, 26 * len(HELP_LINES) + 92
    x, y = (w - pw) // 2, (h - ph) // 2
    _panel(frame, x, y, pw, ph, 0.85)
    _text(frame, "GESTURES", x + 20, y + 34, 0.7, ACCENT, 2)
    for i, (gesture, action) in enumerate(HELP_LINES):
        row = y + 62 + i * 26
        _text(frame, gesture, x + 20, row, 0.48, WHITE)
        _text(frame, action, x + 230, row, 0.48, DIM)
    _text(frame, "h close   q quit   d debug   l landmarks", x + 20, y + ph - 18, 0.42, DIM)
