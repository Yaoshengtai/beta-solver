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


# ------------------------------------------------------------ convexity, checked numerically (unit 3.2)
from betasolver.compare import path_sweep  # noqa: E402
from betasolver.moves import path_check, segments  # noqa: E402
from betasolver.stance import assemble, satisfies, solve, vector  # noqa: E402


def test_swing_moves():
    # the last foot coming off, or the first going on, swings the body in or out
    assert Move(State("H1", "H2", "F1"), "LF", None).swing           # last foot off
    assert Move(State("H1", "H2", "F1"), "LF", "F2").swing           # the only foot changes hold
    assert Move(State("H1", "H2"), "LF", "F1").swing                 # first foot on
    assert not Move(S, "LF", "F2").swing                             # the other foot stays
    assert not Move(S, "RH", "H3").swing


def test_three_slides_stay_below_the_difficulty():
    # section 4.2: along x0 -> x1 -> x2 -> x3 every sampled position holds with t <= max(t_release, t_catch),
    # and the blended end solutions meet every constraint
    b, r = make_body(AVA), route(GOOD)
    m = Move(S, "RH", "H3")
    e = evaluate(b, r, m)
    assert [seg[0] for seg in segments(m, e)] == [1, 3, 5]
    pts = path_check(b, r, m, e, n=20)
    assert len(pts) == 63
    for p in pts:
        assert p["ok_mix"] and p["t_min"] is not None
        assert p["t_min"] <= e.difficulty + 1e-5 and p["t_mix"] <= e.difficulty + 1e-6
        assert p["t_min"] <= p["t_mix"] + 1e-5   # a fresh solve can only do better than the blend


def test_path_check_refuses_swing_and_failed_moves():
    b = make_body(AVA)
    with pytest.raises(ValueError):
        path_check(b, route(LAYBACK), Move(S, "RH", "H3"))
    holds = GOOD[:4] + [Hold("F2", 1.75, 1.00, "foot_big")]
    with pytest.raises(ValueError):
        path_check(b, route(holds), Move(State("H1", "H2", "F1"), "LF", "F2"))


def test_convexity_on_random_moves():
    # step 10: the same check on 100 random moves that work (swing moves set aside)
    S_ = path_sweep()
    assert S_["moves"] == 100 and S_["all_feasible"]
    assert S_["worst_min"] < 1e-5 and S_["worst_mix"] < 1e-5


def test_a_limb_in_the_air_neither_pushes_nor_reaches():
    b = make_body(AVA)
    far = route([Hold("H1", 1.25, 2.05, "jug"), Hold("H2", 2.9, 0.2, "jug"), Hold("H3", 1.85, 2.45, "jug"),
                 Hold("F1", 1.30, 0.90, "foot_big"), Hold("F2", 1.75, 1.00, "foot_big")])
    assert not solve(b, far, S, no_force="RH")["ok"]                 # resting on a hold out of reach
    assert solve(b, far, S, no_force="RH", off="RH")["ok"]           # in the air: no reach needed
    P = assemble(b, far, S, no_force="RH", off="RH")
    assert [o for o in P.cone_owners if o[0] == "reach"] == [("reach", "LH"), ("reach", "LF"), ("reach", "RF")]
    with pytest.raises(ValueError):
        assemble(b, far, S, off="RH")                                # in the air means no force too


def test_solution_vector_satisfies_its_program():
    b, r = make_body(AVA), route(GOOD)
    res = solve(b, r, S)
    P = assemble(b, r, S)
    z = vector(P.layout, res["torso"], res["forces"], res["t"])
    assert satisfies(P, z, tol=1e-5)
    z[P.layout[("t", None)]] = 0.5 * res["t"]                        # too little strength
    assert not satisfies(P, z, tol=1e-5)


# ------------------------------------------------------------ why two checks are enough (unit 3.3)
import math  # noqa: E402

from betasolver.moves import region  # noqa: E402


def _inside(poly, p):
    n = len(poly)
    return all((poly[(i + 1) % n][0] - poly[i][0]) * (p[1] - poly[i][1])
               - (poly[(i + 1) % n][1] - poly[i][1]) * (p[0] - poly[i][0]) >= -1e-9 for i in range(n))


def _holds(b, r, x):
    res = solve(b, r, S, no_force="RH", off="RH", torso=x, tidy=False)
    return res["ok"] and res["t"] <= 1 + 1e-9


def test_s3_boundary_is_traced_by_the_check():
    b, r = make_body(AVA), route(GOOD)
    e = evaluate(b, r, Move(S, "RH", "H3"))
    poly = region(b, r, S, "RH", e.release.torso)
    assert len(poly) == 72
    c = e.release.torso
    for p in poly[::9]:
        dx, dy = p[0] - c[0], p[1] - c[1]
        assert _holds(b, r, (c[0] + 0.97 * dx, c[1] + 0.97 * dy))       # just inside: holds
        assert not _holds(b, r, (c[0] + 1.03 * dx, c[1] + 1.03 * dy))   # just outside: does not


def test_release_and_catch_positions_share_s3():
    # section 4.3: x1 in S3 and D_A, x2 in S3 and D_B, and the straight slide between them stays in S3
    b, r = make_body(AVA), route(GOOD)
    e = evaluate(b, r, Move(S, "RH", "H3"))
    x1, x2 = e.release.torso, e.catch.torso
    for i in range(11):
        u = i / 10
        assert _holds(b, r, (x1[0] + u * (x2[0] - x1[0]), x1[1] + u * (x2[1] - x1[1])))
    for x, hid in ((x1, "H2"), (x2, "H3")):
        h, o = r.hold(hid), b.offsets["RH"]
        assert math.dist(x, (h.x - o[0], h.y - o[1])) <= b.L["RH"] + 1e-6


def test_cannot_let_go_means_s3_misses_d_a():
    b, r = make_body(AVA), route(LAYBACK)
    e = evaluate(b, r, Move(S, "RH", "H3"))
    poly = region(b, r, S, "RH", e.catch.torso)
    o, L = b.offsets["RH"], b.L["RH"]
    da = (r.hold("H2").x - o[0], r.hold("H2").y - o[1])
    db = (r.hold("H3").x - o[0], r.hold("H3").y - o[1])
    assert min(math.dist(p, da) for p in poly) > L          # S3 and D_A do not meet
    assert min(math.dist(p, db) for p in poly) < L          # S3 and D_B do
    assert region(b, r, S, "RH", (2.6, 0.3)) == []          # a start point outside S3 gives nothing
