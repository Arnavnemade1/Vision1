"""Classifier behaviour over the whole gesture vocabulary."""

from copy import deepcopy

import numpy as np
import pytest

from vision_paint.config import GestureConfig, StabilizerConfig
from vision_paint.features import extract
from vision_paint.gestures import Gesture, GestureClassifier, classify_two_hands
from vision_paint.stabilizer import GestureStabilizer
from vision_paint.synthetic import POSES, PoseSpec, build_frame

CFG = GestureConfig()
CLF = GestureClassifier(CFG)
MIN_CONF = StabilizerConfig().min_confidence

MIRRORED_NAME = {"point_left": "point_right", "point_right": "point_left"}


def classify(spec: PoseSpec, **kw):
    return CLF.classify(extract(build_frame(spec, **kw), CFG), MIN_CONF)


@pytest.mark.parametrize("name", sorted(POSES))
def test_canonical_pose_is_recognised(name):
    result = classify(POSES[name])
    assert result.gesture.value == name, f"ranked {result.ranking[:3]}"
    assert result.confidence >= MIN_CONF


@pytest.mark.parametrize("name", sorted(POSES))
def test_canonical_pose_beats_runner_up(name):
    """Every gesture needs daylight between it and the next best reading,
    otherwise noise decides which command runs."""
    ranking = CLF.score_all(extract(build_frame(POSES[name]), CFG))
    assert ranking[0][1] - ranking[1][1] > 0.05, ranking[:3]


@pytest.mark.parametrize("name", sorted(POSES))
def test_recognised_on_the_left_hand_too(name):
    spec = deepcopy(POSES[name])
    spec.geometric_label = "Left"
    expected = MIRRORED_NAME.get(name, name)
    assert classify(spec).gesture.value == expected


@pytest.mark.parametrize("name", sorted(set(POSES) - {"point_left", "point_right", "point_up", "point_down", "thumbs_up", "thumbs_down"}))
@pytest.mark.parametrize("roll", [-25, 25])
def test_non_directional_gestures_survive_a_tilted_wrist(name, roll):
    spec = deepcopy(POSES[name])
    spec.roll_deg += roll
    assert classify(spec).gesture.value == name


@pytest.mark.parametrize("name", sorted(POSES))
@pytest.mark.parametrize("scale", [1.4, 4.2])
def test_recognised_near_and_far_from_the_camera(name, scale):
    assert classify(POSES[name], scale=scale).gesture.value == name


def test_drawing_and_pointing_up_are_distinguished_by_the_thumb():
    """These two share an index-up silhouette; only the thumb separates them,
    so the distinction has to be sharp rather than marginal."""
    draw = CLF.score_all(extract(build_frame(POSES["draw"]), CFG))
    point = CLF.score_all(extract(build_frame(POSES["point_up"]), CFG))
    assert dict(draw)[Gesture.DRAW] - dict(draw)[Gesture.POINT_UP] > 0.4
    assert dict(point)[Gesture.POINT_UP] - dict(point)[Gesture.DRAW] > 0.4


def test_fist_and_pinch_do_not_bleed_into_each_other():
    """Erase and select-tool must never be confused: both curl the fingers."""
    fist = dict(CLF.score_all(extract(build_frame(POSES["fist"]), CFG)))
    pinch = dict(CLF.score_all(extract(build_frame(POSES["pinch"]), CFG)))
    assert fist[Gesture.FIST] - fist[Gesture.PINCH] > 0.4
    assert pinch[Gesture.PINCH] - pinch[Gesture.FIST] > 0.4


def test_clear_screen_and_pause_are_distinguished_by_finger_spread():
    wide = dict(CLF.score_all(extract(build_frame(POSES["open_palm"]), CFG)))
    flat = dict(CLF.score_all(extract(build_frame(POSES["palm_flat"]), CFG)))
    assert wide[Gesture.OPEN_PALM] > wide[Gesture.PALM_FLAT]
    assert flat[Gesture.PALM_FLAT] > flat[Gesture.OPEN_PALM]


def test_nothing_is_recognised_from_a_meaningless_hand():
    rng = np.random.default_rng(11)
    spec = PoseSpec({"index": 90, "middle": 55, "ring": 100, "pinky": 70}, thumb_out=0.5)
    result = classify(spec, noise=0.004, rng=rng)
    assert result.gesture is Gesture.NONE


def test_prayer_needs_two_aligned_hands_close_together():
    left = extract(build_frame(POSES["palm_flat"], center=(0.46, 0.5)), CFG)
    right_spec = deepcopy(POSES["palm_flat"])
    right_spec.geometric_label = "Left"
    right = extract(build_frame(right_spec, center=(0.54, 0.5)), CFG)
    _, together = classify_two_hands(left, right, CFG)

    far = extract(build_frame(right_spec, center=(0.9, 0.5)), CFG)
    _, apart = classify_two_hands(left, far, CFG)

    assert together > 0.6
    assert apart < 0.3


def test_held_gesture_settles_under_continuous_jitter():
    """The end-to-end path: a hand held still but never perfectly still."""
    rng = np.random.default_rng(5)
    stab = GestureStabilizer(StabilizerConfig())
    settled = []
    t = 0.0
    for _ in range(20):
        t += 1 / 30
        spec = deepcopy(POSES["two_fingers"])
        spec.roll_deg += rng.normal(0, 6)
        spec.fan += rng.normal(0, 0.15)
        result = classify(spec, noise=0.002, rng=rng)
        state = stab.update(result.gesture, result.confidence, t)
        if state.changed and state.active:
            settled.append(state.gesture)
    assert settled == [Gesture.TWO_FINGERS]
