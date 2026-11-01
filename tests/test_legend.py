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


def test_strength_figure_matches_the_solver():
    from betasolver.body import make_body
    from betasolver.model import PROFILES
    from betasolver.stance import State, effort, solve

    S = legend.strength_figure()
    route, state = legend.strength_route(), State(**S["state"])
    assert [r["name"] for r in S["runs"]] == ["Ava", "Ben", "Cal"]
    for r in S["runs"]:
        b = make_body(PROFILES[r["climber"]])
        t, share, limiting = effort(b, route, state, legend.STRENGTH_FORCES)
        assert abs(r["t"] - t) < TOL and r["limiting"] == limiting
        assert all(abs(r["frame"]["util"][l] - share[l]) < TOL for l in share)
        assert abs(r["t_lp"] - solve(b, route, state, fixed=legend.STRENGTH_FORCES, upto="inplane")["t"]) < TOL
        assert r["t"] - TOL <= r["t_lp"] <= r["t"] / S["coverage"] + TOL
    by = {r["name"]: r for r in S["runs"]}
    # the three readings of t: Ava near the limit on the sloper, Ben on the crimp, Cal too weak
    assert by["Ava"]["band"] == "limit" and by["Ava"]["limiting"] == "RH"
    assert by["Ben"]["band"] == "hard" and by["Ben"]["limiting"] == "LH"
    assert by["Cal"]["band"] == "fail"


def test_strength_example_forces_are_physical():
    f = legend.STRENGTH_FORCES
    assert abs(sum(v["f"][0] for v in f.values())) < 1e-9
    assert abs(sum(v["f"][1] for v in f.values()) - 1.0) < 1e-9
    assert abs(sum(v.get("n", 0) for v in f.values()) - sum(v.get("m", 0) for v in f.values())) < 1e-9


def test_balance_figure_matches_the_solver():
    B = legend.balance_figure()
    hang, off, side = B["panels"]
    assert hang["ok"] and abs(hang["torso"][0] - hang["hold_x"]) < TOL
    assert hang["forces"]["LH"]["f"] == [0.0, 1.0] or max(abs(a - b) for a, b in zip(hang["forces"]["LH"]["f"], (0, 1))) < TOL
    assert not off["ok"] and abs(off["shift"] - 0.2) < TOL
    assert side["ok"] and abs(side["moment"] - side["torso"][0]) < TOL
    f = side["forces"].values()
    assert abs(sum(v["f"][0] for v in f)) < TOL and abs(sum(v["f"][1] for v in f) - 1) < TOL
    # the sidepull pulls the hand outward, the foot pushes back
    assert side["forces"]["RH"]["f"][0] > 0.1 and side["forces"]["RF"]["f"][0] < -0.1


def test_solver_beats_the_fixed_strength_example():
    for r in legend.balance_figure()["strength_again"]:
        assert r["t"] < r["t_fixed"]
        assert r["t"] - TOL <= r["t_lp"]


def test_strength_figure_torso_is_balanced():
    from betasolver.stance import State, moment
    S = legend.strength_figure()
    m = moment(legend.strength_route(), State(**S["state"]), legend.STRENGTH_FORCES)
    for r in S["runs"]:
        assert abs(r["frame"]["torso"][0] - m) < TOL


def test_tipping_figure_follows_d_over_h():
    T = legend.tipping_figure()
    low, high = T["views"]
    assert low["h"] > high["h"] and high["N"] > low["N"] and high["share"] > low["share"]
    for v in T["views"]:
        assert abs(v["N"] - T["d"] / v["h"]) < TOL and abs(v["M"] - v["N"]) < TOL
    assert all(abs(p["N"] - p["formula"]) <= 0.01 * p["formula"] for p in T["sweep"])
    for r in T["strength_again"]:
        assert r["t"] >= r["t_before"] - TOL and r["pull"] > 0


def test_barndoor_figure():
    D = legend.barndoor_figure()
    between, side = D["panels"]
    assert between["ok"] and abs(between["tau"]) <= between["cap"] + TOL and between["needed"] < TOL
    assert not side["ok"] and side["needed"] > side["cap"]
    assert min(between["feet_x"]) < between["hand_x"] < max(between["feet_x"])
    assert min(side["feet_x"]) > side["hand_x"]


def test_flex_figure():
    X = legend.flex_figure()
    ava, stiff = X["panels"]
    assert ava["ok"] and ava["foot"] <= ava["limit"] + TOL
    assert not stiff["ok"] and stiff["foot"] > stiff["best_limit"]


def test_result_cards_cover_every_outcome():
    cards = legend.result_cards()
    assert [c["key"] for c in cards] == ["holds", "strength", "reach", "balance", "unusable"]
    assert cards[0]["result"]["ok"] and cards[0]["result"]["t"] <= 1
    assert all(c["result"]["reason"] == c["key"] for c in cards[1:])
    assert cards[1]["result"]["t"] > 1
