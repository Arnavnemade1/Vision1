"""Gesture -> action.

The binding table below is the contract between the recognition half of the
system and the drawing half. Two gestures are context-sensitive and resolved
against what the hand is doing rather than its shape alone:

  two fingers  held still -> next colour   swiped sideways -> change board
  pinch        held still -> next tool     moved -> grab a drawing, or pan the
                                           board if the pinch caught nothing

Zoom is deliberately not on that list. It used to be -- a pinch that opened or
closed -- and sharing one gesture with grabbing made both unreliable, since
holding something still and changing the gap between your fingers are hard to
keep separate by accident. Zoom moved to a two-handed pinch, which is the
gesture everyone already knows from a touchscreen and which no single-handed
pose can be mistaken for. One hand pinching now means exactly one thing: hold
something.

A pinch commits to one reading within the first fraction of a second and keeps
it until the hand opens again, so once you have hold of a drawing, a wobble
cannot turn the drag into a pan.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .config import AppConfig
from .features import FeatureSmoother, HandFeatures
from .filters import OneEuroFilter
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
        Binding(Gesture.PINCH, "Grab / move", dwell=0.35, needs_still=True),
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
        self.smoother = FeatureSmoother(cfg.tracker.feature_smoothing)
        # The pen gets its own, heavier smoothing: ink records every tremor the
        # pose recogniser is happy to average away.
        self.pen = OneEuroFilter(cfg.tracker.draw_min_cutoff, cfg.tracker.draw_beta)

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
        self._last_now = 0.0

        # Pinch manipulation state, all reset on each new pinch press.
        self.pinch_mode: Optional[str] = None      # grab | pan
        self.hover_group = None                    # what a pinch would pick up
        self.grabbed_group = None                  # what a pinch is holding
        self._grab_anchor: Optional[np.ndarray] = None
        self._grab_last_board: Optional[np.ndarray] = None
        self._grab_total = np.zeros(2)
        self._zoom_span: Optional[float] = None    # two-hand zoom reference span
        self._zoom_base = 1.0

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
            self._zoom_span = None

        if not hands:
            self._end_stroke()
            self._release_grab()
            self.stabilizer.update(Gesture.NONE, 0.0, now)
            self.motion.reset()
            self.pen.reset()
            self._zoom_span = None
            self.cursor = None
            self.hover_group = None
            self.dwell_progress = 0.0
            self.last_state = self.stabilizer.state
            return

        if len(hands) >= 2 and self._handle_two_hands(hands, now):
            return

        hand = self._pick_primary(hands)
        assert hand is not None

        hand = self.smoother(hand, now, self.cfg.gesture)
        result = self.classifier.classify(hand, self.cfg.stabilizer.min_confidence)
        self.ranking = result.ranking[:3]
        self.last_confidence = result.confidence

        state = self.stabilizer.update(result.gesture, result.confidence, now)
        self.last_state = state
        motion = self.motion.update(hand.palm_center, hand.pinch, now)

        dt = max(now - self._last_now, 1e-3) if self._last_now else 1 / 30
        self._last_now = now
        # Arm before any handler can consume the frame, or a gesture whose first
        # stable frame goes to the pinch machine never becomes eligible to fire.
        self.gate.arm(state)

        if state.changed:
            self._dynamic_used = False
            self._release_grab()

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
        if state.gesture is Gesture.PINCH:
            if self._handle_pinch(hand, state):
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
            self.pen.reset()
            st.canvas.begin_stroke(st.tool, st.color, st.brush_size, st.style)
            self._stroke_kind = "draw"
        tip = self.pen(hand.index_tip, self._last_now)
        self.cursor = tip
        st.canvas.extend_stroke(self._canvas_point(tip))

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

    def _handle_pinch(self, hand: HandFeatures, state) -> bool:
        """Grab, pan or select -- decided once per pinch, then held.

        Returns True when the pinch has been consumed as a manipulation, which
        leaves the generic binding path (tool cycling) for the case where the
        hand pinched and then did nothing.
        """
        cfg = self.cfg.motion
        canvas = self.state.canvas
        point = hand.pinch_point
        board = np.array(canvas.screen_to_board(point[0] * canvas.width,
                                                point[1] * canvas.height))

        if self._grab_anchor is None:
            self._grab_anchor = np.array(point, dtype=float)
            self._grab_last_board = board

        travel = float(np.linalg.norm(np.array(point) - self._grab_anchor))

        if self.pinch_mode is None:
            # Keep tracking what the fingers are over, so the highlight follows
            # the hand and a grab picks up whatever is under it at the moment
            # the drag starts -- not whatever happened to be there when the
            # fingers first closed.
            self.hover_group = canvas.group_at(board[0], board[1])

            if travel >= cfg.grab_min_distance:
                if self.hover_group is not None:
                    self.pinch_mode = "grab"
                    self.grabbed_group = self.hover_group
                    self._grab_last_board = board
                    self._grab_total = np.zeros(2)
                    canvas.lift(self.grabbed_group)
                    self.state.notify(
                        f"Holding {len(self.grabbed_group.strokes)} strokes", "good"
                    )
                else:
                    self.pinch_mode = "pan"
            elif state.held < cfg.grab_intent_seconds:
                # Still deciding: hold the frame so a slow reach toward a
                # drawing is not mistaken for a deliberate still pose.
                return True

        if self.pinch_mode == "grab" and self.grabbed_group is not None:
            self._dynamic_used = True
            delta = board - self._grab_last_board
            self._grab_last_board = board
            self._grab_total = self._grab_total + delta
            canvas.drag_to(self.grabbed_group, delta)
            return True

        if self.pinch_mode == "pan":
            self._dynamic_used = True
            screen_delta = (np.array(point) - self._grab_anchor)
            self._grab_anchor = np.array(point, dtype=float)
            canvas.pan_by(screen_delta[0] * canvas.width, screen_delta[1] * canvas.height)
            return True

        return False

    def _release_grab(self) -> None:
        """Let go at the end of a pinch and record the move for undo."""
        canvas = self.state.canvas
        if self.pinch_mode == "grab" and self.grabbed_group is not None:
            canvas.drop(self.grabbed_group, self._grab_total)
            moved = float(np.linalg.norm(self._grab_total))
            if moved > 1.0:
                self.state.notify(f"Moved {len(self.grabbed_group.strokes)} strokes")
        else:
            canvas.floating = []
        self.pinch_mode = None
        self.grabbed_group = None
        self.hover_group = None
        self._grab_anchor = None
        self._grab_last_board = None
        self._grab_total = np.zeros(2)

    # ---- two hands --------------------------------------------------------

    def _handle_two_hands(self, hands: List[HandFeatures], now: float) -> bool:
        gesture, score = classify_two_hands(hands[0], hands[1], self.cfg.gesture)
        state = self.two_hand_stabilizer.update(
            gesture if score >= self.cfg.stabilizer.min_confidence else Gesture.NONE, score, now
        )
        if not state.active:
            self._zoom_span = None
            return False

        self.last_state = state
        self._end_stroke()
        self._release_grab()

        if state.gesture is Gesture.PINCH_SPREAD:
            self.dwell_progress = 0.0
            self.dwell_label = ""
            return self._handle_two_hand_zoom(hands, state)

        binding = self.bindings[Gesture.PRAYER]
        self.dwell_progress = self.two_hand_gate.progress(state, binding.dwell)
        self.dwell_label = binding.label
        self.two_hand_gate.arm(state)
        if self.two_hand_gate.try_fire(state, now, binding.cooldown, binding.dwell):
            self.state.notify("Exiting", "warn")
            self.state.running = False
        return True

    def _handle_two_hand_zoom(self, hands: List[HandFeatures], state) -> bool:
        """Pinch with both hands and move them apart or together.

        Zoom tracks the ratio of the current span to the span when the gesture
        started, the way a touchscreen does, rather than integrating a rate.
        That makes it absolute: the same hand positions always give the same
        zoom, so overshooting is corrected by moving back rather than by
        waiting for a drift to stop.
        """
        canvas = self.state.canvas
        aspect = canvas.width / max(canvas.height, 1)
        a, b = hands[0].pinch_point, hands[1].pinch_point
        span = float(np.hypot((a[0] - b[0]) * aspect, a[1] - b[1]))

        if state.changed or self._zoom_span is None:
            self._zoom_span = span
            self._zoom_base = canvas.zoom
            self.state.notify("Zooming", "good")
            return True

        if self._zoom_span < self.cfg.motion.two_hand_zoom_min_span:
            self._zoom_span = span
            return True

        canvas.set_zoom(self._zoom_base * (span / self._zoom_span))
        self.state.notify(f"Zoom {canvas.zoom:.2f}x")
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
            # Reached when a pinch has stayed put long enough to read as a pose
            # rather than a manipulation. Deliberately does NOT lock the pinch
            # into a mode: pinching, pausing, then dragging is a natural way to
            # pick something up, and locking here made that impossible. The
            # event gate already stops the tool firing twice in one press, and
            # re-anchoring means the pause itself is not counted as travel.
            st.tool_index = (st.tool_index + 1) % len(cfg.tools)
            st.notify(f"Tool: {st.tool}")
            self._grab_anchor = None
        elif gesture is Gesture.THREE_FINGERS:
            st.style_index = (st.style_index + 1) % len(cfg.brush_styles)
            st.notify(f"Style: {st.style}")
        elif gesture is Gesture.FOUR_FINGERS:
            canvas = st.canvas
            canvas.background_index = (canvas.background_index + 1) % len(cfg.backgrounds)
            st.notify(f"Background: {cfg.backgrounds[canvas.background_index][0]}")
        elif gesture is Gesture.CROSSED_FINGERS:
            self._release_grab()
            what = st.canvas.undo()
            st.notify(f"Undo: {what}" if what else "Nothing to undo",
                      "info" if what else "warn")
        elif gesture is Gesture.PALM_FLAT:
            st.paused = not st.paused
            self._end_stroke()
            st.notify("Paused" if st.paused else "Resumed", "warn" if st.paused else "good")
        elif gesture is Gesture.OPEN_PALM:
            self._release_grab()
            st.notify("Board cleared" if st.canvas.clear() else "Already empty", "warn")
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
