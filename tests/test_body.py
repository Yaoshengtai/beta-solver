"""The body model (step 2): attachment points, limb lengths, reach and joints."""
import math

import pytest

from betasolver.body import HIP, SHOULDER, make_body, two_link
from betasolver.model import AVA, BEN, CAL, PROFILES, VARIANTS


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def test_limb_lengths_match_plan():
    # step 2: Ava's arm about 0.55 m and leg 0.80 m; Ben's arm about 0.66 m and leg about 0.93 m
    a, b = make_body(AVA), make_body(BEN)
    assert abs(a.L["LH"] - 0.55) < 0.01 and a.L["LF"] == pytest.approx(0.80)
    assert abs(b.L["RH"] - 0.66) < 0.01 and abs(b.L["RF"] - 0.93) < 0.01


def test_arm_formula():
    c = CAL
    assert make_body(c).L["LH"] == pytest.approx((c.span - 0.259 * c.H) / 2 - 0.03 * c.H)


def test_anchors_are_mirrored_and_scaled():
    b = make_body(BEN)
    assert b.offsets["LH"] == pytest.approx((-SHOULDER[0] * BEN.H, SHOULDER[1] * BEN.H))
    assert b.offsets["RH"] == pytest.approx((SHOULDER[0] * BEN.H, SHOULDER[1] * BEN.H))
    assert b.offsets["LF"] == pytest.approx((-HIP[0] * BEN.H, HIP[1] * BEN.H))
    assert b.anchor("RF", (1.0, 2.0)) == pytest.approx((1.0 + HIP[0] * BEN.H, 2.0 + HIP[1] * BEN.H))


def test_segments_add_up_to_limb_length():
    for c in PROFILES.values():
        b = make_body(c)
        for limb, (u, l) in b.seg.items():
            assert u + l == pytest.approx(b.L[limb]) and u > 0 and l > 0


def test_reach_is_a_circle():
    b = make_body(AVA)
    torso = (1.5, 1.5)
    sx, sy = b.anchor("RH", torso)
    assert b.reaches("RH", torso, (sx, sy + b.L["RH"]))
    assert not b.reaches("RH", torso, (sx, sy + 1.05 * b.L["RH"]))  # step 10: 1.05 times is out of reach


def test_foot_limit_is_hip_plus_flex():
    torso = (1.5, 1.5)
    a, s = make_body(AVA), make_body(VARIANTS["A_stiff"])
    hip = torso[1] + HIP[1] * AVA.H
    assert a.foot_limit(torso) == pytest.approx(hip + 0.08 * AVA.H)
    assert s.foot_limit(torso) == pytest.approx(hip - 0.10 * AVA.H)
    # linear in the torso height
    assert a.foot_limit((0, 2.5)) - a.foot_limit((0, 1.5)) == pytest.approx(1.0)


def test_joints_keep_segment_lengths_and_bend_the_right_way():
    b = make_body(AVA)
    torso = (1.5, 1.42)
    ends = {"LH": (1.15, 2.30), "RH": (1.85, 2.05), "LF": (1.25, 0.75), "RF": (1.75, 0.95)}
    J = b.joints(torso, ends)
    for limb, j in J.items():
        root, mid = (j["shoulder"], j["elbow"]) if limb[1] == "H" else (j["hip"], j["knee"])
        assert dist(root, mid) == pytest.approx(b.seg[limb][0])
        assert dist(mid, j["end"]) == pytest.approx(b.seg[limb][1])
        assert j["end"] == ends[limb]
    assert J["LF"]["knee"][0] < J["LF"]["hip"][0] and J["RF"]["knee"][0] > J["RF"]["hip"][0]


def test_hanging_foot_and_missing_hand():
    b = make_body(AVA)
    torso = (1.5, 1.5)
    J = b.joints(torso, {"LH": (1.3, 2.2), "RH": (1.7, 2.2), "LF": (1.3, 0.9)})
    hip = b.anchor("RF", torso)
    assert J["RF"]["end"][1] < hip[1] - 0.9 * b.L["RF"]
    with pytest.raises(ValueError):
        b.joints(torso, {"LH": (1.3, 2.2)})


def test_two_link_straightens_when_out_of_reach():
    assert two_link((0, 0), (3, 0), 1, 1, lambda p: 0) == (1.0, 0.0)


def test_body_data_is_plain():
    d = make_body(AVA).to_dict()
    assert d["climber"] == "A" and d["d_wall"] == pytest.approx(0.24) and set(d["L"]) == {"LH", "RH", "LF", "RF"}
