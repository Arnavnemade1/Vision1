"""End-to-end: synthetic hands in, canvas changes out.

These drive the real pipeline -- features, classifier, stabiliser, dispatcher,
canvas -- so they catch wiring mistakes that unit tests on either half miss.
"""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from vision_paint.canvas import CanvasSet
from vision_paint.commands import CommandDispatcher
from vision_paint.config import AppConfig
from vision_paint.features import extract
from vision_paint.state import AppState
from vision_paint.synthetic import POSES, build_frame

DT = 1 / 30


class Rig:
    """A dispatcher plus a clock, so tests read as 'hold this pose for a while'."""

    def __init__(self, tmp_path: Path):
        self.cfg = AppConfig()
        canvases = CanvasSet(640, 360, self.cfg.canvas)
        self.state = AppState(cfg=self.cfg, canvases=canvases, save_dir=tmp_path)
        self.dispatcher = CommandDispatcher(self.cfg, self.state)
        self.dispatcher.last_camera_frame = np.zeros((360, 640, 3), np.uint8)
        self.t = 0.0

    def hold(self, pose, seconds=0.5, center=(0.5, 0.5), move=(0.0, 0.0), pinch=None, spec=None):
        """Feed one pose for `seconds`, optionally sliding it across the frame."""
        frames = max(1, int(seconds / DT))
        base = spec if spec is not None else deepcopy(POSES[pose])
        for i in range(frames):
            self.t += DT
            f = i / max(1, frames - 1)
            at = (center[0] + move[0] * f, center[1] + move[1] * f)
            hand = extract(build_frame(base, center=at), self.cfg.gesture, 16 / 9)
            if pinch is not None:
                hand.pinch = pinch(f)
            self.dispatcher.update([hand], self.t)
        return self

    def blank(self, seconds=0.4):
        for _ in range(int(seconds / DT)):
            self.t += DT
            self.dispatcher.update([], self.t)
        return self

    def two_hands(self, seconds=2.0, gap=0.08):
        left = deepcopy(POSES["palm_flat"])
        right = deepcopy(POSES["palm_flat"])
        right.geometric_label = "Left"
        for _ in range(int(seconds / DT)):
            self.t += DT
            a = extract(build_frame(left, center=(0.5 - gap, 0.5)), self.cfg.gesture, 16 / 9)
            b = extract(build_frame(right, center=(0.5 + gap, 0.5)), self.cfg.gesture, 16 / 9)
            self.dispatcher.update([a, b], self.t)
        return self


@pytest.fixture
def rig(tmp_path):
    return Rig(tmp_path)


def test_index_finger_draws_a_stroke(rig):
    rig.hold("draw", 0.7, center=(0.35, 0.5), move=(0.25, 0.0)).blank()
    strokes = rig.state.canvas.strokes
    assert len(strokes) == 1
    assert len(strokes[0].points) > 3
    assert strokes[0].color == rig.state.color
    assert not strokes[0].erase


def test_a_new_gesture_ends_the_stroke(rig):
    rig.hold("draw", 0.6, move=(0.2, 0.0)).hold("palm_flat", 0.5).blank()
    assert len(rig.state.canvas.strokes) == 1
    assert rig.state.canvas.active is None


def test_fist_erases(rig):
    rig.hold("draw", 0.7, center=(0.3, 0.5), move=(0.3, 0.0)).blank()
    rig.hold("fist", 0.7, center=(0.3, 0.5), move=(0.3, 0.0)).blank()
    assert [s.erase for s in rig.state.canvas.strokes] == [False, True]


def test_two_fingers_held_still_changes_colour(rig):
    before = rig.state.color_index
    rig.hold("two_fingers", 0.9).blank()
    assert rig.state.color_index == (before + 1) % len(rig.cfg.canvas.palette)
    assert rig.state.canvases.index == 0


def test_two_fingers_swiped_sideways_changes_canvas_instead(rig):
    """The overloaded gesture: movement must win, and must not also recolour."""
    colour_before = rig.state.color_index
    rig.hold("two_fingers", 0.45, center=(0.3, 0.5), move=(0.4, 0.0)).blank()
    assert rig.state.canvases.index == 1
    assert rig.state.color_index == colour_before


def test_pinch_held_still_selects_the_next_tool(rig):
    rig.hold("pinch", 0.9, pinch=lambda f: 0.2).blank()
    assert rig.state.tool == rig.cfg.canvas.tools[1]
    assert rig.state.canvas.zoom == 1.0


def test_opening_the_pinch_zooms_instead_of_selecting(rig):
    rig.hold("pinch", 0.9, pinch=lambda f: 0.16 + 0.2 * f).blank()
    assert rig.state.canvas.zoom > 1.05
    assert rig.state.tool == rig.cfg.canvas.tools[0]


def test_closing_the_pinch_zooms_out(rig):
    rig.state.canvas.set_zoom(2.0)
    rig.hold("pinch", 0.9, pinch=lambda f: 0.36 - 0.2 * f).blank()
    assert rig.state.canvas.zoom < 1.95


def test_open_palm_clears_only_after_the_hold(rig):
    rig.hold("draw", 0.6, move=(0.2, 0.0)).blank()
    rig.hold("open_palm", rig.cfg.stabilizer.destructive_dwell * 0.5).blank()
    assert rig.state.canvas.strokes, "cleared before the dwell elapsed"
    rig.hold("open_palm", rig.cfg.stabilizer.destructive_dwell + 0.5).blank()
    assert not rig.state.canvas.strokes


def test_thumbs_up_saves_and_thumbs_down_bins_the_save(rig, tmp_path):
    rig.hold("draw", 0.6, move=(0.2, 0.0)).blank()
    rig.hold("thumbs_up", 0.6).blank()
    saved = list(tmp_path.glob("*.png"))
    assert len(saved) == 1

    rig.hold("thumbs_down", rig.cfg.stabilizer.destructive_dwell + 0.5).blank()
    assert not list(tmp_path.glob("*.png"))
    # Binned, not destroyed: a misread gesture must be recoverable.
    assert list((tmp_path / ".trash").glob("*.png"))


def test_crossed_fingers_undo_one_stroke_at_a_time(rig):
    for x in (0.3, 0.5):
        rig.hold("draw", 0.5, center=(x, 0.4), move=(0.1, 0.0)).blank()
    assert len(rig.state.canvas.strokes) == 2
    rig.hold("crossed_fingers", 0.5).blank()
    assert len(rig.state.canvas.strokes) == 1


def test_flat_palm_pauses_drawing_and_resumes_it(rig):
    rig.hold("palm_flat", 0.6).blank()
    assert rig.state.paused
    rig.hold("draw", 0.7, move=(0.2, 0.0)).blank()
    assert not rig.state.canvas.strokes, "drew while paused"
    rig.hold("palm_flat", 0.6).blank()
    assert not rig.state.paused
    rig.hold("draw", 0.7, move=(0.2, 0.0)).blank()
    assert rig.state.canvas.strokes


def test_pointing_steps_colour_and_brush_size(rig):
    rig.hold("point_right", 0.5).blank()
    assert rig.state.color_index == 1
    rig.hold("point_left", 0.5).blank()
    assert rig.state.color_index == 0

    small = rig.state.brush_size
    rig.hold("point_up", 0.5).blank()
    assert rig.state.brush_size > small
    rig.hold("point_down", 0.5).blank()
    assert rig.state.brush_size == small


def test_holding_a_pointing_gesture_repeats(rig):
    rig.hold("point_right", 2.0).blank()
    assert rig.state.color_index >= 3


def test_three_and_four_fingers_change_style_and_background(rig):
    rig.hold("three_fingers", 0.6).blank()
    assert rig.state.style == rig.cfg.canvas.brush_styles[1]
    rig.hold("four_fingers", 0.6).blank()
    assert rig.state.canvas.background_index == 1


def test_menu_opens_navigates_and_confirms(rig):
    rig.hold("call_me", 0.6).blank()
    assert rig.state.menu_open
    rig.hold("point_down", 0.5).blank()
    assert rig.state.menu_index == 1          # colour row
    before = rig.state.color_index
    rig.hold("ok_sign", 0.6).blank()
    assert rig.state.color_index == before + 1
    rig.hold("call_me", 0.6).blank()
    assert not rig.state.menu_open


def test_drawing_is_suppressed_while_the_menu_is_open(rig):
    rig.hold("call_me", 0.6).blank()
    rig.hold("draw", 0.7, move=(0.2, 0.0)).blank()
    assert not rig.state.canvas.strokes


def test_both_palms_together_exits_after_the_hold(rig):
    rig.two_hands(seconds=rig.cfg.stabilizer.destructive_dwell * 0.4)
    assert rig.state.running
    rig.two_hands(seconds=rig.cfg.stabilizer.destructive_dwell + 0.6)
    assert not rig.state.running


def test_each_canvas_keeps_its_own_drawing(rig):
    rig.hold("draw", 0.6, center=(0.3, 0.4), move=(0.2, 0.0)).blank()
    rig.hold("two_fingers", 0.45, center=(0.3, 0.5), move=(0.4, 0.0)).blank()
    assert rig.state.canvas.is_empty
    rig.hold("draw", 0.6, center=(0.3, 0.6), move=(0.2, 0.0)).blank()
    assert len(rig.state.canvas.strokes) == 1
    rig.hold("two_fingers", 0.45, center=(0.7, 0.5), move=(-0.4, 0.0)).blank()
    assert len(rig.state.canvas.strokes) == 1
