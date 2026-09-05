"""Voting window and event gate."""

import pytest

from vision_paint.config import StabilizerConfig
from vision_paint.gestures import Gesture
from vision_paint.stabilizer import EventGate, GestureStabilizer

CFG = StabilizerConfig()
DT = 1 / 30


def feed(stab, gesture, frames, conf=0.9, start=0.0):
    t = start
    state = None
    for _ in range(frames):
        t += DT
        state = stab.update(gesture, conf, t)
    return state, t


def test_gesture_needs_a_majority_before_it_counts():
    stab = GestureStabilizer(CFG)
    state, _ = feed(stab, Gesture.FIST, CFG.min_votes - 1)
    assert not state.active
    state, _ = feed(stab, Gesture.FIST, 1, start=CFG.min_votes * DT)
    assert state.gesture is Gesture.FIST


def test_one_bad_frame_cannot_change_the_answer():
    stab = GestureStabilizer(CFG)
    feed(stab, Gesture.DRAW, 10)
    state = stab.update(Gesture.FIST, 0.99, 11 * DT)
    assert state.gesture is Gesture.DRAW


def test_low_confidence_frames_never_settle():
    stab = GestureStabilizer(CFG)
    state, _ = feed(stab, Gesture.OK_SIGN, 15, conf=CFG.min_confidence - 0.1)
    assert not state.active


def test_event_fires_once_per_gesture_press():
    stab, gate = GestureStabilizer(CFG), EventGate(CFG)
    fires = 0
    t = 0.0
    for _ in range(60):
        t += DT
        fires += gate.try_fire(stab.update(Gesture.THUMBS_UP, 0.9, t), t)
    assert fires == 1


def test_releasing_and_reforming_fires_again():
    stab, gate = GestureStabilizer(CFG), EventGate(CFG)
    t, fires = 0.0, 0
    for gesture in (Gesture.THUMBS_UP, Gesture.NONE, Gesture.THUMBS_UP):
        for _ in range(40):
            t += DT
            fires += gate.try_fire(stab.update(gesture, 0.9, t), t)
    assert fires == 2


def test_repeat_mode_fires_on_a_cooldown_while_held():
    stab, gate = GestureStabilizer(CFG), EventGate(CFG)
    t, fires = 0.0, 0
    for _ in range(120):  # 4 seconds
        t += DT
        fires += gate.try_fire(
            stab.update(Gesture.POINT_UP, 0.9, t), t, cooldown=0.3, repeat=True
        )
    assert 8 <= fires <= 13


def test_destructive_gesture_waits_out_its_dwell():
    stab, gate = GestureStabilizer(CFG), EventGate(CFG)
    dwell = CFG.destructive_dwell
    t, fired_at = 0.0, None
    for _ in range(int(4 / DT)):
        t += DT
        if gate.try_fire(stab.update(Gesture.OPEN_PALM, 0.9, t), t, dwell=dwell):
            fired_at = t
            break
    assert fired_at is not None
    assert fired_at >= dwell


def test_letting_go_during_the_dwell_cancels_it():
    stab, gate = GestureStabilizer(CFG), EventGate(CFG)
    t = 0.0
    for _ in range(int((CFG.destructive_dwell * 0.6) / DT)):
        t += DT
        assert not gate.try_fire(
            stab.update(Gesture.OPEN_PALM, 0.9, t), t, dwell=CFG.destructive_dwell
        )
    for _ in range(10):
        t += DT
        stab.update(Gesture.NONE, 0.0, t)
    state = stab.update(Gesture.OPEN_PALM, 0.9, t)
    assert gate.progress(state, CFG.destructive_dwell) == pytest.approx(0.0)
