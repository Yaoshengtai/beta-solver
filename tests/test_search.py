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
