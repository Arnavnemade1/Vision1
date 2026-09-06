"""Board mechanics: grouping strokes into drawings, moving them, undoing that."""

import numpy as np
import pytest

from vision_paint.canvas import Canvas, CanvasSet, _chaikin
from vision_paint.config import CanvasConfig

CFG = CanvasConfig()


@pytest.fixture
def board():
    return Canvas(640, 360, CFG)


def scribble(canvas, x, y, length=60, color=(0, 0, 255), tool="brush"):
    canvas.begin_stroke(tool, color, 6, "solid")
    for i in range(0, length, 4):
        canvas.extend_stroke((x + i, y + (i % 12)))
    canvas.end_stroke()
    return canvas.strokes[-1]


def test_board_is_larger_than_the_viewport(board):
    assert board.board_w > board.width and board.board_h > board.height


def test_screen_and_board_coordinates_round_trip(board):
    board.pan_by(40, -25)
    board.set_zoom(1.8)
    for point in ((0, 0), (320, 180), (639, 359)):
        bx, by = board.screen_to_board(*point)
        sx, sy = board.board_to_screen(bx, by)
        assert (sx, sy) == pytest.approx(point, abs=1e-6)


def test_nearby_strokes_form_one_drawing(board):
    scribble(board, 300, 300)
    scribble(board, 320, 316)      # overlapping the first
    assert len(board.groups()) == 1
    assert len(board.groups()[0].strokes) == 2


def test_distant_strokes_stay_separate_drawings(board):
    scribble(board, 200, 200)
    scribble(board, 900, 600)
    assert len(board.groups()) == 2


def test_grabbing_picks_the_drawing_under_the_point(board):
    scribble(board, 200, 200)
    scribble(board, 900, 600)
    picked = board.group_at(210, 205)
    assert picked is not None
    assert picked.contains(210, 205, pad=board.grab_reach)
    assert board.group_at(600, 400) is None       # empty space between them


def test_moving_a_drawing_shifts_only_its_own_strokes(board):
    kept = scribble(board, 200, 200)
    moved_stroke = scribble(board, 900, 600)
    group = board.group_at(905, 605)
    delta = np.array([-260.0, 40.0])

    board.lift(group)
    board.drag_to(group, delta)
    board.drop(group, delta)

    assert moved_stroke.offset == pytest.approx(delta)
    assert kept.offset == pytest.approx(np.zeros(2))


def test_a_moved_drawing_can_be_grabbed_at_its_new_place(board):
    scribble(board, 900, 600)
    group = board.group_at(905, 605)
    delta = np.array([-300.0, 0.0])
    board.lift(group)
    board.drag_to(group, delta)
    board.drop(group, delta)

    assert board.group_at(905, 605) is None
    assert board.group_at(605, 605) is not None


def test_undo_reverses_a_move(board):
    stroke = scribble(board, 400, 300)
    group = board.group_at(405, 305)
    delta = np.array([120.0, -80.0])
    board.lift(group)
    board.drag_to(group, delta)
    board.drop(group, delta)
    assert stroke.offset == pytest.approx(delta)

    board.undo()
    assert stroke.offset == pytest.approx(np.zeros(2))
    assert stroke in board.strokes, "undoing a move must not delete the drawing"


def test_undo_walks_back_through_mixed_edits(board):
    a = scribble(board, 200, 200)
    b = scribble(board, 900, 600)
    group = board.group_at(905, 605)
    board.lift(group)
    board.drag_to(group, np.array([-10.0, 0.0]))
    board.drop(group, np.array([-10.0, 0.0]))

    board.undo()                       # the move
    assert b.offset == pytest.approx(np.zeros(2))
    board.undo()                       # stroke b
    assert b not in board.strokes and a in board.strokes
    board.undo()                       # stroke a
    assert board.is_empty
    assert board.undo() is None


def test_clear_is_undoable(board):
    scribble(board, 300, 300)
    scribble(board, 340, 320)
    assert board.clear()
    assert board.is_empty
    board.undo()
    assert len(board.strokes) == 2


def test_panning_stays_inside_the_board(board):
    board.pan_by(100000, 100000)
    slack_x = (board.board_w - board.width) / 2
    slack_y = (board.board_h - board.height) / 2
    assert board.pan[0] == pytest.approx(slack_x)
    assert board.pan[1] == pytest.approx(slack_y)


def test_dragging_does_not_rebuild_the_cached_raster(board):
    """A drag lifts its strokes out of the cache, so moving a drawing costs one
    overlay redraw rather than re-rasterising the whole board every frame."""
    scribble(board, 300, 300)
    scribble(board, 900, 600)
    board._render_base()
    group = board.group_at(905, 605)
    board.lift(group)
    base = board._render_base()

    for _ in range(10):
        board.drag_to(group, np.array([3.0, 1.0]))
        assert board._render_base() is base, "cache thrashed mid-drag"


def test_lifted_strokes_still_render(board):
    scribble(board, 300, 300)
    group = board.group_at(305, 305)
    board.lift(group)
    board.drag_to(group, np.array([50.0, 0.0]))
    painted = int((board.render()[:, :, 3] > 0).sum())
    assert painted > 0


def test_chaikin_smooths_without_running_away_from_the_input():
    corner = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0)]
    smoothed = _chaikin(corner)
    assert len(smoothed) > len(corner)
    xs = [p[0] for p in smoothed]
    ys = [p[1] for p in smoothed]
    assert min(xs) >= -1 and max(xs) <= 101
    assert min(ys) >= -1 and max(ys) <= 101
    # The sharp corner must actually be rounded off.
    assert not any(abs(x - 100) < 1 and abs(y) < 1 for x, y in smoothed)


def test_short_strokes_are_left_alone():
    assert _chaikin([(0.0, 0.0), (5.0, 5.0)]) == [(0.0, 0.0), (5.0, 5.0)]


def test_each_board_in_the_set_has_its_own_history():
    canvases = CanvasSet(640, 360, CFG)
    scribble(canvases.current, 300, 300)
    canvases.step(1)
    assert canvases.current.is_empty
    assert not canvases.current.history
    canvases.step(-1)
    assert len(canvases.current.strokes) == 1
