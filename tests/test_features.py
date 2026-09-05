"""Feature-extraction invariants."""

import numpy as np
import pytest

from vision_paint.config import GestureConfig
from vision_paint.features import extract
from vision_paint.synthetic import FOLDED, POSES, PoseSpec, _curls, build_frame

CFG = GestureConfig()


def feat(name_or_spec, **kw):
    spec = POSES[name_or_spec] if isinstance(name_or_spec, str) else name_or_spec
    return extract(build_frame(spec, **kw), CFG)


def test_curl_increases_with_flexion():
    prev = -1.0
    for curl in (0, 30, 60, 90, 120, 160):
        f = feat(PoseSpec(_curls(index=curl, middle=FOLDED, ring=FOLDED, pinky=FOLDED)))
        assert f.curls["index"] > prev
        prev = f.curls["index"]


def test_extension_thresholds_separate_open_from_fist():
    open_hand = feat("open_palm")
    fist = feat("fist")
    assert open_hand.extended_count == 4
    assert fist.extended_count == 0


@pytest.mark.parametrize("scale", [1.2, 2.6, 5.0])
def test_features_are_scale_invariant(scale):
    """Distances are normalised by hand size, so moving toward or away from the
    camera must not change any of them."""
    a = feat("pinch", scale=2.6)
    b = feat("pinch", scale=scale)
    assert a.pinch == pytest.approx(b.pinch, abs=1e-6)
    assert a.spread == pytest.approx(b.spread, abs=1e-6)
    assert a.index_reach == pytest.approx(b.index_reach, abs=1e-6)


def test_curls_are_rotation_invariant():
    upright = feat("draw")
    for roll in (-75, -30, 45, 120, 180):
        spec = PoseSpec(**{**POSES["draw"].__dict__, "roll_deg": roll})
        rotated = feat(spec)
        for finger in ("index", "middle", "ring", "pinky"):
            assert rotated.curls[finger] == pytest.approx(upright.curls[finger], abs=2.0)


def test_thumb_abduction_separates_tucked_from_out():
    tucked = feat("draw")
    out = feat("point_up")
    assert tucked.thumb_abduction < CFG.thumb_abduction_min < out.thumb_abduction


def test_index_reach_separates_fist_from_pinch():
    assert feat("fist").index_reach < CFG.fist_index_reach_max < feat("pinch").index_reach


def test_pointing_direction_follows_the_hand():
    directions = {
        "point_right": (1.0, 0.0),
        "point_left": (-1.0, 0.0),
        "point_up": (0.0, 1.0),
        "point_down": (0.0, -1.0),
    }
    for name, target in directions.items():
        f = feat(name)
        assert float(np.dot(f.index_dir, np.array(target))) > 0.75, name


def test_palm_normal_points_at_the_camera_for_both_hands():
    for label in ("Right", "Left"):
        spec = PoseSpec(**{**POSES["open_palm"].__dict__, "geometric_label": label})
        assert feat(spec).palm_facing > 0.5, label


def test_curl_survives_landmark_noise():
    """Short distal segments made the old per-joint curl unusable under noise;
    the chord form has to stay well inside the extended threshold."""
    rng = np.random.default_rng(0)
    values = [
        feat("open_palm", noise=0.0025, rng=rng).curls["pinky"] for _ in range(200)
    ]
    assert np.percentile(values, 95) < CFG.finger_extended_max_curl
