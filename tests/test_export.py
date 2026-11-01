import json

import pytest

from betasolver.body import make_body
from betasolver.export import PHASES, SLACK, export_run, frames_of, smoothing_check
from betasolver.model import AVA, CAL, Hold, Route
from betasolver.search import solve
from betasolver.stance import assemble, satisfies, vector

SMALL = Route("small", "small", "", [Hold("S1", 1.30, 1.80, "jug"), Hold("S2", 1.70, 1.85, "crimp"),
                                     Hold("H3", 1.40, 2.25, "pinch"), Hold("T", 1.60, 2.55, "jug"),
                                     Hold("F1", 1.30, 0.70, "foot_big"), Hold("F2", 1.70, 0.75, "foot_big"),
                                     Hold("F3", 1.45, 1.10, "foot_chip"), Hold("F4", 1.70, 1.35, "foot_big")],
              ("S1", "S2"), "T")


@pytest.fixture(scope="module")
def beta():
    return solve(SMALL, AVA)


def test_smoothing_keeps_the_beta(beta):
    # step 10: before and after smoothing, the sequence of moves and the crux are the same
    run = export_run(SMALL, AVA, beta)
    assert [s["text"] for s in run["steps"]] == [m.describe(SMALL) for m, _, _ in beta.steps]
    assert run["crux"] == pytest.approx(beta.crux, abs=1e-3) and run["crux_step"] == beta.crux_step
    json.dumps(run)  # plain data only


def test_smoothed_frames_are_valid(beta):
    c = smoothing_check(SMALL, AVA, beta)
    assert c["valid"] and c["worst"] <= 1e-6


def test_four_frames_per_step(beta):
    run = export_run(SMALL, AVA, beta)
    for st in run["steps"]:
        assert [f["phase"] for f in st["frames"]] == list(PHASES)
        assert st["limb"] not in st["frames"][1]["forces"] and st["limb"] not in st["frames"][2]["forces"]
        assert set(st["frames"][0]["joints"]) == {"LH", "RH", "LF", "RF"}


def test_blends_between_frames_are_valid(beta):
    # what the player draws mid-slide: each slide's two smoothed frames, blended, meet that phase's constraints
    body = make_body(AVA)
    prev = None
    for move, res, _ in beta.steps:
        frames, prev = frames_of(body, SMALL, move, res, prev)
        slides = [(0, 1, move.state, None, None), (1, 2, move.state, move.limb, move.limb), (2, 3, move.new_state, None, None)]
        for a, b, state, nf, off in slides:
            if move.swing or (a == 0 and move.frm is None) or (b == 2 and move.to is None):
                continue
            P = assemble(body, SMALL, state, nf, off=off)
            za = vector(P.layout, frames[a]["torso"], frames[a]["forces"], frames[a]["t"])
            zb = vector(P.layout, frames[b]["torso"], frames[b]["forces"], frames[b]["t"])
            for u in (0.25, 0.5, 0.75):
                assert satisfies(P, (1 - u) * za + u * zb, tol=1e-5)


def test_cut_frames_are_filled():
    # a cut has no catch frame: the export copies the neighbouring frame so every step has four
    route = Route("cut", "cut", "", [Hold("S1", 1.30, 1.80, "jug"), Hold("S2", 1.70, 1.80, "jug"), Hold("T", 1.50, 2.20, "jug"),
                                     Hold("F1", 1.30, 0.75, "foot_big"), Hold("F2", 1.70, 0.80, "foot_big")], ("S1", "S2"), "T")
    from betasolver.moves import Move, evaluate
    from betasolver.stance import State
    body = make_body(AVA)
    m = Move(State("S1", "S2", "F1", "F2"), "LF", None)
    res = evaluate(body, route, m)
    frames, _ = frames_of(body, route, m, res, None)
    assert [f["phase"] for f in frames] == list(PHASES)
    assert frames[2]["torso"] == frames[1]["torso"]


def test_infeasible_run_exports_its_fail_record():
    route = Route("high", "high", "", [Hold("S1", 1.30, 1.80, "jug"), Hold("S2", 1.70, 1.80, "jug"), Hold("T", 1.50, 3.40, "jug"),
                                       Hold("F1", 1.35, 0.95, "foot_big"), Hold("F2", 1.65, 0.95, "foot_big")],
                  ("S1", "S2"), "T", allow_feet_on_handholds=False)
    run = export_run(route, CAL, solve(route, CAL))
    assert run["status"] == "infeasible" and run["steps"] == [] and run["fail"]["hold"] == "T"
