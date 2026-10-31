import importlib.util
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("legend", ROOT / "site" / "legend.py")
legend = importlib.util.module_from_spec(spec)
sys.modules["legend"] = legend
spec.loader.exec_module(legend)

L = legend.legend()
TOL = 2e-3  # values are rounded to 3 decimals


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def test_all_eight_hold_types():
    names = [t["type"] for t in L["hold_types"]]
    assert names == ["jug", "crimp", "sloper", "pinch", "sidepull", "undercling", "foot_big", "foot_chip"]
    assert {t["type"] for t in L["hold_types"] if t["use"] == "foot"} == {"foot_big", "foot_chip"}


def test_wall_size():
    assert L["wall"] == {"w": 3.0, "h": 4.2}


def test_limb_lengths_match_plan():
    # step 2: Ava's arm is about 0.55 m and her leg 0.80 m; Ben's arm about 0.66 m
    assert abs(legend.limb_lengths(1.60, 1.60)["arm"] - 0.55) < 0.01
    assert legend.limb_lengths(1.60, 1.60)["leg"] == 0.80
    assert abs(legend.limb_lengths(1.85, 1.90)["arm"] - 0.66) < 0.01


def test_pose_segments_have_body_lengths():
    c = L["climber"]
    H, lens = c["H"], legend.limb_lengths(c["H"], c["span"])
    holds = {h["id"]: (h["x"], h["y"]) for h in L["scene"]["holds"]}
    frame = L["scene"]["frame"]
    for limb, j in frame["joints"].items():
        root, mid = (j["shoulder"], j["elbow"]) if limb[1] == "H" else (j["hip"], j["knee"])
        a = (legend.UPPER_ARM if limb[1] == "H" else legend.THIGH) * H
        total = lens["arm"] if limb[1] == "H" else lens["leg"]
        assert abs(dist(root, mid) - a) < TOL, limb
        assert abs(dist(mid, j["end"]) - (total - a)) < TOL, limb
        assert dist(j["end"], holds[frame["state"][limb]]) < TOL, limb


def test_knees_point_out():
    J = L["scene"]["frame"]["joints"]
    assert J["LF"]["knee"][0] < J["LF"]["hip"][0]
    assert J["RF"]["knee"][0] > J["RF"]["hip"][0]


def test_example_forces_balance_body_weight():
    f = L["scene"]["frame"]["forces"].values()
    assert abs(sum(v["f"][0] for v in f)) < TOL
    assert abs(sum(v["f"][1] for v in f) - 1.0) < TOL


def test_two_link_straightens_when_out_of_reach():
    mid = legend.two_link((0, 0), (3, 0), 1, 1, lambda p: 0)
    assert mid == (1.0, 0.0)
