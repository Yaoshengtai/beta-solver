import pytest

from betasolver.body import make_body
from betasolver.model import AVA, Hold, Route
from betasolver.moves import Move, evaluate
from betasolver.stance import State, check


def route(holds):
    return Route("m", "m", "m", holds + [Hold("T", 1.5, 3.6, "jug")], (holds[0].id, holds[1].id), "T")


GOOD = [Hold("H1", 1.25, 2.05, "jug"), Hold("H2", 1.75, 2.00, "jug"), Hold("H3", 1.85, 2.45, "jug"),
        Hold("F1", 1.30, 0.90, "foot_big"), Hold("F2", 1.75, 1.00, "foot_big")]
LAYBACK = [Hold("H1", 1.29, 2.27, "sidepull", dir=0), Hold("H2", 2.02, 1.67, "jug"), Hold("H3", 1.41, 2.35, "jug"),
           Hold("F1", 1.10, 0.93, "foot_big"), Hold("F2", 1.43, 0.74, "foot_big")]
S = State("H1", "H2", "F1", "F2")


def test_move_kinds():
    assert Move(S, "RH", "H3").kind == "swap"
    assert Move(S, "RH", "H1").kind == "match"
    assert Move(S, "LF", None).kind == "cut"
    assert Move(State("H1", "H2", None, "F2"), "LF", "F1").kind == "place"
    assert Move(S, "RH", "H3").new_state == State("H1", "H3", "F1", "F2")
    with pytest.raises(ValueError):
        Move(S, "RH", None)       # a hand cannot hang
    with pytest.raises(ValueError):
        Move(S, "RH", "H2")       # nothing changes
    with pytest.raises(ValueError):
        Move(State("H1", "H2"), "LH", None)


def test_the_two_checks_are_state_checks_with_a_resting_limb():
    # section 4.1: release = check(sigma, no_force=k), catch = check(sigma', no_force=k)
    b, r = make_body(AVA), route(GOOD)
    m = Move(S, "RH", "H3")
    e = evaluate(b, r, m)
    assert e.t_release == pytest.approx(check(b, r, S, no_force="RH").t)
    assert e.t_catch == pytest.approx(check(b, r, m.new_state, no_force="RH").t)
    assert "RH" not in e.release.forces and "RH" not in e.catch.forces
    assert e.ok and e.difficulty == pytest.approx(max(e.t_release, e.t_catch))
    assert e.failure is None


def test_plain_checks_never_decide_a_move():
    # section 4.3: t(sigma) <= t_release and t(sigma') <= t_catch
    b, r = make_body(AVA), route(GOOD)
    e = evaluate(b, r, Move(S, "RH", "H3"))
    assert e.start.t <= e.t_release + 1e-6 and e.end.t <= e.t_catch + 1e-6


def test_release_can_fail_while_catch_passes():
    # section 4.1: both checks are needed; here the hand cannot stay on its old hold with no weight on it
    b, r = make_body(AVA), route(LAYBACK)
    e = evaluate(b, r, Move(S, "RH", "H3"))
    assert e.start.ok and not e.release.ok and e.catch.ok
    assert not e.ok and e.difficulty is None
    assert e.failure["phase"] == "release"


def test_cut_and_place_need_one_check_each():
    b, r = make_body(AVA), route(GOOD)
    cut = evaluate(b, r, Move(S, "LF", None))
    assert cut.catch is None and cut.release is not None and cut.ok == cut.release.ok
    place = evaluate(b, r, Move(State("H1", "H2", None, "F2"), "LF", "F1"))
    assert place.release is None and place.catch is not None and place.ok == place.catch.ok
    assert "describe" and Move(S, "LF", None).describe(r) == "Left foot off F1"
    assert Move(S, "RH", "H3").describe(r) == "Right hand → H3 (jug)"
