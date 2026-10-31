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


def test_pose_segments_have_body_lengths():
    from betasolver.body import make_body
    from betasolver.model import PROFILES
    b = make_body(PROFILES[L["climber"]["key"]])
    holds = {h["id"]: (h["x"], h["y"]) for h in L["scene"]["holds"]}
    frame = L["scene"]["frame"]
    for limb, j in frame["joints"].items():
        root, mid = (j["shoulder"], j["elbow"]) if limb[1] == "H" else (j["hip"], j["knee"])
        a, rest = b.seg[limb]
        assert abs(dist(root, mid) - a) < TOL, limb
        assert abs(dist(mid, j["end"]) - rest) < TOL, limb
        assert dist(j["end"], holds[frame["state"][limb]]) < TOL, limb


def test_knees_point_out():
    J = L["scene"]["frame"]["joints"]
    assert J["LF"]["knee"][0] < J["LF"]["hip"][0]
    assert J["RF"]["knee"][0] > J["RF"]["hip"][0]


def test_example_forces_balance_body_weight():
    f = L["scene"]["frame"]["forces"].values()
    assert abs(sum(v["f"][0] for v in f)) < TOL
    assert abs(sum(v["f"][1] for v in f) - 1.0) < TOL


def test_body_figures_stand_on_the_ground():
    figs = legend.body_figures()
    assert [f["name"] for f in figs] == ["Ava", "Ben", "Cal", "Ava (injured)"]
    for f in figs:
        J = f["stand"]["joints"]
        assert abs(J["LF"]["end"][1]) < TOL and abs(J["RF"]["end"][1]) < TOL
        assert abs(f["reach"] - J["LH"]["end"][1]) < TOL
    ava, ben = figs[0], figs[1]
    assert ben["reach"] > ava["reach"]
