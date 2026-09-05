"""Threshold fitting and the calibration file contract."""

import json

import numpy as np
import pytest

from vision_paint.config import AppConfig, GestureConfig

from tools.calibrate import boundary


def test_boundary_lands_between_the_two_distributions():
    low = np.random.default_rng(0).normal(0.2, 0.03, 200)
    high = np.random.default_rng(1).normal(0.8, 0.03, 200)
    b = boundary(low, high)
    assert 0.2 < b < 0.8


def test_boundary_leans_away_from_the_noisier_class():
    """A tight cluster should keep its ground; the boundary moves toward the
    class whose samples are spread out."""
    tight = np.random.default_rng(2).normal(0.2, 0.01, 400)
    loose = np.random.default_rng(3).normal(0.8, 0.15, 400)
    assert boundary(tight, loose) < 0.5


def test_overlapping_classes_fall_back_to_the_midpoint():
    a = np.full(50, 0.5)
    b = np.full(50, 0.4)
    assert boundary(a, b) == pytest.approx(0.45)


def test_calibration_only_applies_known_gesture_settings():
    cfg = AppConfig()
    applied = cfg.apply_calibration(
        {"gesture": {"pinch_close_max": 0.5, "not_a_setting": 1, "point_cone_half_angle": "x"}}
    )
    assert applied == ["pinch_close_max"]
    assert cfg.gesture.pinch_close_max == 0.5
    assert cfg.gesture.point_cone_half_angle == GestureConfig().point_cone_half_angle


def test_missing_calibration_file_is_not_an_error(tmp_path):
    assert AppConfig().load_calibration(tmp_path / "nope.json") == []


def test_calibration_file_round_trips(tmp_path):
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps({"gesture": {"crossed_tip_max": 0.25}}))
    cfg = AppConfig()
    assert cfg.load_calibration(path) == ["crossed_tip_max"]
    assert cfg.gesture.crossed_tip_max == 0.25
    assert cfg.calibration == str(path)
