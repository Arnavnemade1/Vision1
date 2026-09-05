"""Gesture -> action.

The binding table below is the contract between the recognition half of the
system and the drawing half. Two gestures are context-sensitive and resolved
against hand movement rather than shape alone:

  two fingers  held still -> next colour        swiped sideways -> change canvas
  pinch        held still -> next tool          opened/closed   -> zoom

In both cases the moving reading wins, and the still reading is suppressed for
the rest of that gesture press, so one hand movement never fires two commands.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .config import AppConfig
from .features import HandFeatures
from .gestures import Gesture, GestureClassifier, classify_two_hands
from .motion import MotionTracker
from .stabilizer import EventGate, GestureStabilizer, StableState
from .state import MENU_ITEMS, AppState


@dataclass(frozen=True)
class Binding:
    gesture: Gesture
    label: str
    cooldown: float = 0.65
    dwell: float = 0.0
    repeat: bool = False
    needs_still: bool = False
    destructive: bool = False


def build_bindings(cfg: AppConfig) -> Dict[Gesture, Binding]:
    dwell = cfg.stabilizer.destructive_dwell
    table = (
        Binding(Gesture.TWO_FINGERS, "Change colour", dwell=0.30, needs_still=True),
        Binding(Gesture.PINCH, "Select tool", dwell=0.35, needs_still=True),
        Binding(Gesture.OPEN_PALM, "Clear screen", dwell=dwell, destructive=True),
        Binding(Gesture.THUMBS_UP, "Save drawing", cooldown=1.2),
        Binding(Gesture.THUMBS_DOWN, "Delete drawing", dwell=dwell, destructive=True),
        Binding(Gesture.CROSSED_FINGERS, "Undo", cooldown=0.55),
        Binding(Gesture.PALM_FLAT, "Pause / resume", cooldown=0.9),
        Binding(Gesture.POINT_RIGHT, "Next colour", cooldown=0.42, repeat=True),
        Binding(Gesture.POINT_LEFT, "Previous colour", cooldown=0.42, repeat=True),
        Binding(Gesture.POINT_UP, "Bigger brush", cooldown=0.34, repeat=True),
        Binding(Gesture.POINT_DOWN, "Smaller brush", cooldown=0.34, repeat=True),
        Binding(Gesture.THREE_FINGERS, "Brush style", cooldown=0.7),
        Binding(Gesture.FOUR_FINGERS, "Background", cooldown=0.7),
        Binding(Gesture.OK_SIGN, "Confirm", cooldown=0.6),
        Binding(Gesture.CALL_ME, "Menu", cooldown=0.8),
        Binding(Gesture.PRAYER, "Exit", dwell=dwell, destructive=True),
    )
    return {b.gesture: b for b in table}


class CommandDispatcher:
    """Drives the canvas from a stream of detected hands."""

    def __init__(self, cfg: AppConfig, state: AppState):
        self.cfg = cfg
        self.state = state
        self.classifier = GestureClassifier(cfg.gesture)
        self.bindings = build_bindings(cfg)

        self.stabilizer = GestureStabilizer(cfg.stabilizer)
        self.gate = EventGate(cfg.stabilizer)
        self.motion = MotionTracker(cfg.motion)

        self.two_hand_stabilizer = GestureStabilizer(cfg.stabilizer)
        self.two_hand_gate = EventGate(cfg.stabilizer)

        self.primary_label: Optional[str] = None
        self.last_state = StableState()
        self.last_confidence = 0.0
        self.ranking: List = []
        self.cursor: Optional[np.ndarray] = None
        self.dwell_progress = 0.0
        self.dwell_label = ""
        self._dynamic_used = False   # a movement already claimed this press
        self._stroke_kind: Optional[str] = None   # "draw" | "erase" | None
        self._last_swipe = -1e9

    # ---- helpers ----------------------------------------------------------

    def _pick_primary(self, hands: List[HandFeatures]) -> Optional[HandFeatures]:
        if not hands:
            self.primary_label = None
            return None
        if self.primary_label:
            for h in hands:
                if h.label == self.primary_label:
                    return h
        best = max(hands, key=lambda h: h.raw.score)
        self.primary_label = best.label
        return best

    def _canvas_point(self, norm_xy: np.ndarray):
        canvas = self.state.canvas
        return canvas.screen_to_canvas(norm_xy[0] * canvas.width, norm_xy[1] * canvas.height)

    def _end_stroke(self) -> None:
        if self._stroke_kind is not None:
            self.state.canvas.end_stroke()
            self._stroke_kind = None

    # ---- main entry point -------------------------------------------------

    def update(self, hands: List[HandFeatures], now: float) -> None:
        st = self.state

        if len(hands) < 2:
            self.two_hand_stabilizer.update(Gesture.NONE, 0.0, now)

        if not hands:
            self._end_stroke()
            self.stabilizer.update(Gesture.NONE, 0.0, now)
            self.motion.reset()
            self.cursor = None
            self.dwell_progress = 0.0
            self.last_state = self.stabilizer.state
            return

        if len(hands) >= 2 and self._handle_two_hands(hands, now):
            return

        hand = self._pick_primary(hands)
        assert hand is not None

        result = self.classifier.classify(hand, self.cfg.stabilizer.min_confidence)
        self.ranking = result.ranking[:3]
        self.last_confidence = result.confidence

        state = self.stabilizer.update(result.gesture, result.confidence, now)
        self.last_state = state
        motion = self.motion.update(hand.palm_center, hand.pinch, now)

        if state.changed:
            self._dynamic_used = False

        self.cursor = hand.index_tip if state.gesture is not Gesture.FIST else hand.palm_center

        binding = self.bindings.get(state.gesture)
        # Only the destructive gestures get the hold ring. The short dwell on
        # the still-hand gestures is arbitration, not something to prompt about.
        show_dwell = bool(binding and binding.destructive)
        self.dwell_progress = self.gate.progress(state, binding.dwell) if show_dwell else 0.0
        self.dwell_label = binding.label if show_dwell else ""

        if state.gesture is Gesture.DRAW:
            self._handle_draw(hand)
            return
        if state.gesture is Gesture.FIST:
            self._handle_erase(hand)
            return
        self._end_stroke()

        if binding is None:
            return

        # Movement-sensitive gestures get first refusal on the frame.
        if state.gesture is Gesture.TWO_FINGERS and self._handle_canvas_swipe(now):
            return
        if state.gesture is Gesture.PINCH and self._handle_zoom(motion):
            return

        if binding.needs_still and (self._dynamic_used or not self.motion.is_still):
            return

        if self.gate.try_fire(state, now, binding.cooldown, binding.dwell, binding.repeat):
            self._run(state.gesture, binding)

    # ---- continuous gestures ---------------------------------------------

    def _handle_draw(self, hand: HandFeatures) -> None:
        st = self.state
        if st.paused or st.menu_open:
            self._end_stroke()
            return
        if self._stroke_kind != "draw":
            self._end_stroke()
            st.canvas.begin_stroke(st.tool, st.color, st.brush_size, st.style)
            self._stroke_kind = "draw"
        st.canvas.extend_stroke(self._canvas_point(hand.index_tip))

    def _handle_erase(self, hand: HandFeatures) -> None:
        st = self.state
        if st.paused or st.menu_open:
            self._end_stroke()
            return
        if self._stroke_kind != "erase":
            self._end_stroke()
            width = int(st.brush_size * self.cfg.canvas.eraser_scale)
            st.canvas.begin_stroke("brush", (0, 0, 0), width, "solid", erase=True)
            self._stroke_kind = "erase"
        st.canvas.extend_stroke(self._canvas_point(hand.palm_center))

    # ---- movement-sensitive gestures -------------------------------------

    def _handle_canvas_swipe(self, now: float) -> bool:
        direction = self.motion.swipe()
        if direction is None:
            return False
        # One continuous hand movement refills the trail as fast as it is
        # cleared, so the swipe needs its own cooldown as well.
        if now - self._last_swipe < self.cfg.stabilizer.default_cooldown:
            return True
        self._last_swipe = now
        self.motion.consume()
        self._dynamic_used = True
        self._end_stroke()
        canvas = self.state.canvases.step(direction)
        self.state.notify(f"Canvas {canvas.name} of {len(self.state.canvases.canvases)}", "good")
        return True

    def _handle_zoom(self, motion) -> bool:
        cfg = self.cfg.motion
        if abs(motion.pinch_rate) < cfg.zoom_min_delta:
            return False
        canvas = self.state.canvas
        canvas.set_zoom(canvas.zoom * (1.0 + motion.pinch_rate * cfg.zoom_gain * (1 / 30)))
        self._dynamic_used = True
        self.state.notify(f"Zoom {canvas.zoom:.2f}x")
        return True

    # ---- two hands --------------------------------------------------------

    def _handle_two_hands(self, hands: List[HandFeatures], now: float) -> bool:
        gesture, score = classify_two_hands(hands[0], hands[1], self.cfg.gesture)
        state = self.two_hand_stabilizer.update(
            gesture if score >= self.cfg.stabilizer.min_confidence else Gesture.NONE, score, now
        )
        if not state.active:
            return False

        binding = self.bindings[Gesture.PRAYER]
        self.last_state = state
        self.dwell_progress = self.two_hand_gate.progress(state, binding.dwell)
        self.dwell_label = binding.label
        self._end_stroke()
        if self.two_hand_gate.try_fire(state, now, binding.cooldown, binding.dwell):
            self.state.notify("Exiting", "warn")
            self.state.running = False
        return True

    # ---- one-shot commands ------------------------------------------------

    def _run(self, gesture: Gesture, binding: Binding) -> None:
        st = self.state
        cfg = self.cfg.canvas

        if st.menu_open and gesture not in (
            Gesture.CALL_ME, Gesture.OK_SIGN, Gesture.POINT_UP, Gesture.POINT_DOWN
        ):
            return

        if gesture is Gesture.CALL_ME:
            st.menu_open = not st.menu_open
            st.notify("Menu open" if st.menu_open else "Menu closed")
            return

        if st.menu_open and gesture in (Gesture.POINT_UP, Gesture.POINT_DOWN):
            step = -1 if gesture is Gesture.POINT_UP else 1
            st.menu_index = (st.menu_index + step) % len(MENU_ITEMS)
            st.notify(f"> {MENU_ITEMS[st.menu_index][1]}")
            return

        if gesture is Gesture.OK_SIGN:
            if st.menu_open:
                self._advance_menu_item()
            else:
                st.notify(f"Confirmed: {st.tool} / {st.style} / {st.brush_size}px", "good")
            return

        if gesture is Gesture.TWO_FINGERS:
            st.color_index = (st.color_index + 1) % len(cfg.palette)
            st.notify(f"Colour {st.color_index + 1}")
        elif gesture is Gesture.POINT_RIGHT:
            st.color_index = (st.color_index + 1) % len(cfg.palette)
            st.notify(f"Colour {st.color_index + 1}")
        elif gesture is Gesture.POINT_LEFT:
            st.color_index = (st.color_index - 1) % len(cfg.palette)
            st.notify(f"Colour {st.color_index + 1}")
        elif gesture is Gesture.POINT_UP:
            st.size_index = min(st.size_index + 1, len(cfg.brush_sizes) - 1)
            st.notify(f"Brush {st.brush_size}px")
        elif gesture is Gesture.POINT_DOWN:
            st.size_index = max(st.size_index - 1, 0)
            st.notify(f"Brush {st.brush_size}px")
        elif gesture is Gesture.PINCH:
            st.tool_index = (st.tool_index + 1) % len(cfg.tools)
            st.notify(f"Tool: {st.tool}")
        elif gesture is Gesture.THREE_FINGERS:
            st.style_index = (st.style_index + 1) % len(cfg.brush_styles)
            st.notify(f"Style: {st.style}")
        elif gesture is Gesture.FOUR_FINGERS:
            canvas = st.canvas
            canvas.background_index = (canvas.background_index + 1) % len(cfg.backgrounds)
            st.notify(f"Background: {cfg.backgrounds[canvas.background_index][0]}")
        elif gesture is Gesture.CROSSED_FINGERS:
            st.notify("Undo" if st.canvas.undo() else "Nothing to undo", "warn" if st.canvas.is_empty else "info")
        elif gesture is Gesture.PALM_FLAT:
            st.paused = not st.paused
            self._end_stroke()
            st.notify("Paused" if st.paused else "Resumed", "warn" if st.paused else "good")
        elif gesture is Gesture.OPEN_PALM:
            st.notify("Screen cleared" if st.canvas.clear() else "Already empty", "warn")
        elif gesture is Gesture.THUMBS_UP:
            path = st.canvas.save(st.save_dir, self.last_camera_frame)
            st.notify(f"Saved {path.name}", "good")
        elif gesture is Gesture.THUMBS_DOWN:
            moved = st.canvas.delete_last_save(st.save_dir)
            st.notify(f"Deleted {moved.name}" if moved else "No saved drawing", "warn")

        del binding

    # The app hands the current frame over each tick so "save" can bake in the
    # camera image when the background is the live view.
    last_camera_frame = None

    def _advance_menu_item(self) -> None:
        st = self.state
        cfg = self.cfg.canvas
        key = MENU_ITEMS[st.menu_index][0]
        if key == "tool":
            st.tool_index = (st.tool_index + 1) % len(cfg.tools)
            st.notify(f"Tool: {st.tool}", "good")
        elif key == "color":
            st.color_index = (st.color_index + 1) % len(cfg.palette)
            st.notify(f"Colour {st.color_index + 1}", "good")
        elif key == "size":
            st.size_index = (st.size_index + 1) % len(cfg.brush_sizes)
            st.notify(f"Brush {st.brush_size}px", "good")
        elif key == "style":
            st.style_index = (st.style_index + 1) % len(cfg.brush_styles)
            st.notify(f"Style: {st.style}", "good")
        elif key == "background":
            canvas = st.canvas
            canvas.background_index = (canvas.background_index + 1) % len(cfg.backgrounds)
            st.notify(f"Background: {cfg.backgrounds[canvas.background_index][0]}", "good")
        elif key == "canvas":
            canvas = st.canvases.step(1)
            st.notify(f"Canvas {canvas.name}", "good")
