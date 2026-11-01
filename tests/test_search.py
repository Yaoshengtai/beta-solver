"""The search (step 5): states, edges, two-stage Dijkstra against an exhaustive reference, failure reports."""
import pytest

from betasolver.model import AVA, Hold, Route
from betasolver.moves import Move
from betasolver.search import Graph, all_states, foot_options, hand_options
from betasolver.stance import State

HOLDS = [Hold("S1", 1.30, 1.80, "jug"), Hold("S2", 1.70, 1.80, "jug"), Hold("T", 1.50, 2.30, "jug"),
         Hold("F1", 1.30, 0.70, "foot_big"), Hold("F2", 1.70, 0.75, "foot_big"), Hold("F3", 1.50, 1.20, "foot_big")]


def route(**kw):
    return Route("tiny", "tiny", "", HOLDS, ("S1", "S2"), "T", **kw)


def test_options_and_combinations():
    r = route()
    assert hand_options(r) == ["S1", "S2", "T"]
    assert foot_options(r) == ["S1", "S2", "T", "F1", "F2", "F3", None]
    assert foot_options(route(allow_feet_on_handholds=False)) == ["F1", "F2", "F3", None]
    assert len(list(all_states(r))) == 3 ** 2 * 7 ** 2


def test_start_states_hold_with_hands_on_the_start():
    g = Graph(route(), AVA)
    starts = g.start_states()
    assert starts and all(s.LH == "S1" and s.RH == "S2" and g.holds(s) for s in starts)
    assert State("S1", "S2", "F1", "F2") in starts


def test_goal_needs_both_hands_on_top_and_a_state_that_holds():
    g = Graph(route(), AVA)
    assert not g.is_goal(State("S1", "T", "F1", "F2"))
    assert g.is_goal(State("T", "T", "F3", None)) == g.holds(State("T", "T", "F3", None))


def test_edges_are_moves_that_work_with_their_cost():
    g = Graph(route(), AVA)
    s = State("S1", "S2", "F1", "F2")
    edges = g.edges(s)
    assert edges and all(res.ok and w["total"] >= res.difficulty for _, res, w in edges)
    n = g.checker.solves
    assert g.edges(s) is edges and g.checker.solves == n          # built once


def test_swing_move_into_a_state_that_does_not_hold_fails():
    # placing the first foot passes its catch check (the body still hangs flat), but the climber
    # must be able to hold the new state once the foot pushes: here the foot is far to one side
    holds = HOLDS + [Hold("F9", 2.40, 1.20, "foot_big")]
    r = Route("tiny", "tiny", "", holds, ("S1", "S2"), "T")
    g = Graph(r, AVA)
    res = g.checker.evaluate(Move(State("S1", "S2"), "RF", "F9"))
    assert Move(State("S1", "S2"), "RF", "F9").swing
    assert res.catch.ok and not res.end.ok
    assert not res.ok and res.failure["phase"] == "end"


# ------------------------------------------------------------ two-stage Dijkstra (unit 4.2)
from betasolver.model import BEN, CAL  # noqa: E402
from betasolver.search import EPS, pareto_reference, solve  # noqa: E402

SMALL = Route("small", "small", "", [Hold("S1", 1.30, 1.80, "jug"), Hold("S2", 1.70, 1.85, "crimp"),
                                     Hold("H3", 1.40, 2.25, "pinch"), Hold("T", 1.60, 2.55, "jug"),
                                     Hold("F1", 1.30, 0.70, "foot_big"), Hold("F2", 1.70, 0.75, "foot_big"),
                                     Hold("F3", 1.45, 1.10, "foot_chip"), Hold("F4", 1.70, 1.35, "foot_big")],
              ("S1", "S2"), "T")


def _three(top=2.65):
    return Route("three", "three", "", [Hold("S1", 1.30, 1.80, "jug"), Hold("S2", 1.70, 1.80, "jug"),
                                        Hold("H3", 1.50, 2.25, "jug"), Hold("T", 1.50, top, "jug"),
                                        Hold("F1", 1.35, 0.95, "foot_big"), Hold("F2", 1.65, 0.95, "foot_big")],
                 ("S1", "S2"), "T", allow_feet_on_handholds=False)


def test_three_move_route_gives_the_expected_beta():
    # step 10: the top is out of reach from the start, so one hand goes via H3
    r = _three()
    b = solve(r, AVA)
    assert b.status == "ok"
    assert [m.describe(r) for m, _, _ in b.steps] == ["Right hand → H3 (jug)", "Left hand → T (jug)", "Right hand → T (jug)"]


@pytest.mark.parametrize("climber", [AVA, BEN, CAL])
def test_two_stages_match_the_exhaustive_reference(climber):
    # step 10: on a small route, the (crux, total) of the two-stage search equals the best over all short paths
    b = solve(SMALL, climber)
    ref = pareto_reference(SMALL, climber, 8)
    assert b.status == "ok" and len(b.steps) <= 8
    path_crux = max(w["total"] for _, _, w in b.steps)
    assert path_crux == pytest.approx(ref[0], abs=EPS) and b.total == pytest.approx(ref[1], abs=1e-9)


def test_beta_is_consistent():
    b = solve(SMALL, AVA)
    costs = [w["total"] for _, _, w in b.steps]
    assert b.total == pytest.approx(sum(costs))
    assert max(costs) <= b.crux + EPS and costs[b.crux_step] == max(costs)
    labels = [lab for _, lab in b.trace["stage1"]]
    assert all(a <= c + 1e-12 for a, c in zip(labels, labels[1:]))      # bottleneck labels never decrease
    s = b.start
    for m, _, _ in b.steps:                                              # the steps chain from the start
        assert m.state == s
        s = m.new_state
    assert s.LH == s.RH == SMALL.finish


def test_infeasible_route_reports_where_it_got_stuck():
    # step 10: no solution gives "infeasible" with a fail record that says release or catch
    r = _three(top=3.30)
    b = solve(r, AVA)
    assert b.status == "infeasible" and b.steps == []
    assert b.fail["phase"] in ("release", "catch") and b.fail["reason"] and b.fail["limb"] in ("LH", "RH")
    assert b.fail["hold"] == "T" and b.fail["step"] >= 1


def test_search_limit():
    assert solve(SMALL, AVA, max_expansions=2).status == "limit"
