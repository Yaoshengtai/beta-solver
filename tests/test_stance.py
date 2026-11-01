import pytest

from betasolver.body import make_body
from betasolver.model import AVA, PROFILES, Hold, Route
from betasolver.stance import CONSTRAINTS, REASONS, UNKNOWNS, State, check, layout, limb_sets, precheck, validate


def route(**kw):
    holds = [Hold("H1", 1.2, 1.6, "jug"), Hold("H2", 1.75, 1.55, "crimp"), Hold("H3", 1.5, 3.6, "jug"),
             Hold("F1", 1.25, 0.45, "foot_big"), Hold("F2", 1.85, 0.7, "foot_chip")]
    args = dict(id="t", title="t", lesson="t", holds=holds, start_hands=("H1", "H2"), finish="H3")
    args.update(kw)
    return Route(**args)


def test_state_needs_both_hands():
    with pytest.raises(ValueError):
        State("H1", None)
    s = State("H1", "H2", None, "F2")
    assert s.on() == ("LH", "RH", "RF")
    assert s.as_dict() == {"LH": "H1", "RH": "H2", "LF": None, "RF": "F2"}


def test_limb_sets_and_no_force():
    s = State("H1", "H2", "F1", "F2")
    assert limb_sets(s) == (("LH", "RH", "LF", "RF"), ("LH", "RH", "LF", "RF"))
    A, AF = limb_sets(s, no_force="RH")
    assert A == ("LH", "RH", "LF", "RF") and AF == ("LH", "LF", "RF")
    with pytest.raises(ValueError):
        limb_sets(State("H1", "H2", "F1"), no_force="RF")  # a hanging foot cannot be the resting limb


def test_unknowns_follow_section_3_0():
    assert [u["name"] for u in UNKNOWNS] == ["x", "f", "N", "M", "tau", "t"]
    assert [c["key"] for c in CONSTRAINTS] == ["reach", "direction", "strength", "inplane", "outofwall", "flex"]
    assert set(REASONS) == {"unusable", "reach", "strength", "balance"}


def test_layout_size_matches_plan():
    # section 3.7: about 17 variables for a four-limb state
    assert layout(("LH", "RH", "LF", "RF")).size == 17
    assert layout(("LH", "RH", "LF")).size == 14
    assert layout(("LH", "RH")).size == 11
    # the resting limb gets no force variables
    lay = layout(("LH", "LF", "RF"))
    assert ("f", "RH") not in lay.index and ("N", "RH") not in lay.index and ("f", "LH") in lay.index


def test_layout_slices_do_not_overlap():
    lay = layout(("LH", "RH", "LF", "RF"))
    cells = sorted(i for sl in lay.index.values() for i in range(sl.start, sl.stop))
    assert cells == list(range(lay.size))


def test_validate_rejects_illegal_states():
    b, r = make_body(AVA), route()
    with pytest.raises(ValueError):
        validate(b, r, State("H1", "F1"))           # a hand on a foothold
    with pytest.raises(ValueError):
        validate(b, r, State("H1", "X9"))           # unknown hold
    validate(b, r, State("H1", "H2", "H1", "F2"))   # a foot on a hand hold is fine...
    with pytest.raises(ValueError):                  # ...unless the route forbids it
        validate(b, route(allow_feet_on_handholds=False), State("H1", "H2", "H1", "F2"))


def test_injured_climber_rules_out_crimps_without_solving():
    # step 10: a state with a crimp is excluded for Ava (injured) before any LP
    r = route()
    res = check(make_body(PROFILES["D"]), r, State("H1", "H2", "F1", "F2"))
    assert not res.ok and res.reason == "unusable" and res.limb == "RH" and res.hold == "H2"
    # a foot on the same crimp is fine: it uses the foothold rules
    assert precheck(make_body(PROFILES["D"]), r, State("H1", "H1", "H2", "F2")) is None
    # a resting hand on a crimp is still in the state, so it still rules the state out
    assert precheck(make_body(PROFILES["D"]), r, State("H1", "H2", "F1", "F2"), no_force="RH").reason == "unusable"


def test_check_holds_a_good_state():
    res = check(make_body(AVA), route(), State("H1", "H2", "F1", "F2"))
    assert res.ok and 0 < res.t <= 1 and res.reason is None and res.limiting in ("LH", "RH", "LF", "RF")


# ------------------------------------------------------------ ① reach
import math  # noqa: E402

import numpy as np  # noqa: E402

from betasolver.stance import REACH_SIDES, centres, reach_rows, reaches_all, torso_region  # noqa: E402


def _region_contains(poly, p):
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        if (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) < -1e-9:
            return False
    return True


def test_reach_rows_are_the_inscribed_polygon():
    b, r, s = make_body(AVA), route(), State("H1", "H2", "F1", "F2")
    G, h, owner = reach_rows(b, r, s)
    assert G.shape == (4 * REACH_SIDES, 2) and len(owner) == 4 * REACH_SIDES
    c = centres(b, r, s)["LH"]
    L = b.L["LH"]
    rows = [i for i, o in enumerate(owner) if o == "LH"]
    # the polygon's vertices lie on the circle, its edge midpoints at L cos(pi/n)
    for i in rows:
        a = 2 * math.pi * rows.index(i) / REACH_SIDES
        mid = (c[0] - L * math.cos(math.pi / REACH_SIDES) * math.cos(a), c[1] - L * math.cos(math.pi / REACH_SIDES) * math.sin(a))
        assert G[i] @ mid == pytest.approx(h[i])
    vert = (c[0] + L * math.cos(math.pi / REACH_SIDES), c[1] + L * math.sin(math.pi / REACH_SIDES))
    assert np.all(G[rows] @ vert <= h[rows] + 1e-9)


def test_polygon_is_conservative():
    # every torso the polygon allows is truly within reach (never the other way round)
    b, r, s = make_body(AVA), route(), State("H1", "H2", "F1", "F2")
    G, h, _ = reach_rows(b, r, s)
    rng = np.random.default_rng(0)
    pts = rng.uniform([0.8, 0.2], [2.2, 1.8], size=(3000, 2))
    inside_poly = np.all(pts @ G.T <= h, axis=1)
    assert inside_poly.any()
    for p in pts[inside_poly]:
        assert all(reaches_all(b, r, s, p).values())


def test_region_matches_exact_reach():
    b, r, s = make_body(AVA), route(), State("H1", "H2", "F1", "F2")
    reg = torso_region(b, r, s, sides=256)
    rng = np.random.default_rng(1)
    for p in rng.uniform([0.8, 0.2], [2.2, 1.8], size=(2000, 2)):
        exact = all(reaches_all(b, r, s, p).values())
        if _region_contains(reg, p):
            assert exact
        elif exact:  # only a sliver along the boundary can differ
            d = min(abs(math.dist(p, centres(b, r, s)[l]) - b.L[l]) for l in s.on())
            assert d < 1e-3


def test_hand_at_105_percent_is_out_of_reach():
    # step 10: a hand hold at 1.05 times the arm length cannot be reached
    b = make_body(AVA)
    torso = (1.5, 1.0)
    sx, sy = b.anchor("LH", torso)
    far = Hold("H9", sx, sy + 1.05 * b.L["LH"], "jug")
    r = route(holds=[far, Hold("H2", 1.75, 1.55, "crimp"), Hold("H3", 1.5, 3.6, "jug")], start_hands=("H9", "H2"))
    assert reaches_all(b, r, State("H9", "H2"), torso)["LH"] is False


def test_unreachable_state_fails_with_reach():
    b, r = make_body(AVA), route()
    res = check(b, r, State("H3", "H2", "F1", "F2"))  # top hold and a low foot
    assert not res.ok and res.reason == "reach"


def test_precheck_pair_distance():
    b, r = make_body(AVA), route()
    assert precheck(b, r, State("H3", "H2", "F1", "F2")).reason == "reach"
    assert precheck(b, r, State("H1", "H2", "F1", "F2")) is None


def test_region_is_convex():
    reg = torso_region(make_body(AVA), route(), State("H1", "H2", "F1", "F2"))
    n = len(reg)
    turns = [(reg[(i + 1) % n][0] - reg[i][0]) * (reg[(i + 2) % n][1] - reg[i][1])
             - (reg[(i + 1) % n][1] - reg[i][1]) * (reg[(i + 2) % n][0] - reg[i][0]) for i in range(n)]
    assert all(t >= -1e-12 for t in turns)


# ------------------------------------------------------------ ② direction
from betasolver.model import HOLD_TYPES  # noqa: E402
from betasolver.stance import cone, cone_normals, direction_rows, in_cone  # noqa: E402


def _unit(deg, r=1.0):
    return (r * math.cos(math.radians(deg)), r * math.sin(math.radians(deg)))


@pytest.mark.parametrize("name", list(HOLD_TYPES))
def test_cone_edges_match_half_width(name):
    t = HOLD_TYPES[name]
    theta = 0.0 if t.dir is None else t.dir
    assert in_cone(_unit(theta), theta, t.alpha)
    assert in_cone(_unit(theta + t.alpha - 0.5), theta, t.alpha)
    assert in_cone(_unit(theta - t.alpha + 0.5), theta, t.alpha)
    assert not in_cone(_unit(theta + t.alpha + 0.5), theta, t.alpha)
    assert not in_cone(_unit(theta - t.alpha - 0.5), theta, t.alpha)


def test_normals_describe_the_same_cone():
    rng = np.random.default_rng(2)
    for theta, alpha in [(90, 70), (90, 15), (0, 35), (180, 35), (270, 40)]:
        nm, npl = cone_normals(theta, alpha)
        for f in rng.normal(size=(500, 2)):
            linear = nm[0] * f[0] + nm[1] * f[1] >= -1e-12 and npl[0] * f[0] + npl[1] * f[1] >= -1e-12
            assert linear == in_cone(f, theta, alpha, tol=1e-12)


def test_hanging_works_on_jug_and_sloper_only():
    up = (0.0, 1.0)
    assert in_cone(up, 90, HOLD_TYPES["jug"].alpha) and in_cone(up, 90, HOLD_TYPES["sloper"].alpha)
    assert not in_cone(up, 0, HOLD_TYPES["sidepull"].alpha)
    assert not in_cone(up, 270, HOLD_TYPES["undercling"].alpha)


def test_foot_on_a_hand_hold_uses_the_foothold_cone():
    r = route(holds=[Hold("H1", 1.2, 1.6, "jug"), Hold("H2", 1.75, 1.55, "undercling"), Hold("H3", 1.5, 3.6, "jug")])
    s = State("H1", "H2", "H2", None)
    assert cone(r, s, "RH") == (270.0, HOLD_TYPES["undercling"].alpha)
    assert cone(r, s, "LF") == (90.0, HOLD_TYPES["foot_big"].alpha)


def test_direction_rows_act_on_force_variables_only():
    r, s = route(), State("H1", "H2", "F1", "F2")
    A, AF = limb_sets(s, no_force="LF")
    lay = layout(AF)
    G, h, owner = direction_rows(r, s, AF, lay)
    assert G.shape == (2 * len(AF), lay.size) and owner.count("LF") == 0 and np.all(h == 0)
    used = np.flatnonzero(np.any(G != 0, axis=0))
    allowed = {i for l in AF for i in range(lay[("f", l)].start, lay[("f", l)].stop)}
    assert set(used) <= allowed
    # a straight-up force on every limb satisfies the rows (all these holds point up)
    z = np.zeros(lay.size)
    for l in AF:
        z[lay[("f", l)]] = (0.0, 0.5)
    assert np.all(G @ z <= 1e-12)
    z[lay[("f", "LH")]] = (0.5, 0.0)  # sideways on a jug is outside its 70 degrees
    assert np.any(G @ z > 0)


# ------------------------------------------------------------ ③ strength
from betasolver.model import BEN, CAL  # noqa: E402
from betasolver.stance import (  # noqa: E402
    EFFORT_BANDS, STRENGTH_AZIMUTHS, STRENGTH_ELEVATIONS, _corners, assemble, band, coverage, effort, solve,
    strength_directions, strength_limit, strength_rows,
)


def test_polyhedron_has_24_faces_of_unit_length():
    U = strength_directions()
    assert U.shape == (STRENGTH_AZIMUTHS * len(STRENGTH_ELEVATIONS), 3) == (24, 3)
    assert np.allclose(np.linalg.norm(U, axis=1), 1.0)
    assert np.all(U[:, 2] > 0)  # every face leans toward the wall side


def test_coverage_puts_the_polyhedron_inside_the_ball():
    c = coverage()
    corners = _corners(strength_directions()) * c
    assert np.max(np.linalg.norm(corners, axis=1)) == pytest.approx(1.0)
    # section 3.7: strength is understated by about 8% at most, never overstated
    assert 0.07 < 1 - c < 0.09


def test_polyhedron_is_between_two_balls():
    # every force the rows allow is within the true limit; every force within c times the limit is allowed
    U, c = strength_directions(), coverage()
    rng = np.random.default_rng(3)
    v = rng.normal(size=(20000, 3))
    v[:, 2] = np.abs(v[:, 2])
    v /= np.linalg.norm(v, axis=1)[:, None]
    v *= rng.uniform(0, 1.2, size=(len(v), 1))
    allowed = np.all(v @ U.T <= c + 1e-12, axis=1)
    r = np.linalg.norm(v, axis=1)
    assert np.all(r[allowed] <= 1 + 1e-9)
    assert np.all(allowed[r <= c])


def test_strength_limit_follows_the_grip():
    b = make_body(AVA)
    r = route(holds=[Hold("H1", 1.2, 1.6, "jug"), Hold("H2", 1.75, 1.55, "sidepull", dir=0),
                     Hold("H3", 1.5, 3.6, "jug"), Hold("F1", 1.25, 0.45, "foot_chip")])
    s = State("H1", "H2", "H1", "F1")
    assert strength_limit(b, r, s, "LH") == AVA.F["jug"]
    assert strength_limit(b, r, s, "RH") == AVA.F["pinch"]  # a sidepull uses the pinch limit
    assert strength_limit(b, r, s, "LF") == AVA.F["foot"]   # a foot on a jug uses the foot limit
    assert strength_limit(b, r, s, "RF") == AVA.F["foot"]


def test_strength_rows_touch_own_force_and_t_only():
    b, r, s = make_body(AVA), route(), State("H1", "H2", "F1", "F2")
    A, AF = limb_sets(s, no_force="RH")
    lay = layout(AF)
    G, h, owner = strength_rows(b, r, s, AF, lay)
    assert G.shape == (24 * len(AF), lay.size) and "RH" not in owner and np.all(h == 0)
    for limb in AF:
        rows = G[[i for i, o in enumerate(owner) if o == limb]]
        out = "N" if limb[1] == "H" else "M"
        cols = set(range(lay[("f", limb)].start, lay[("f", limb)].stop)) | {lay[(out, limb)].start, lay[("t", None)].start}
        assert set(np.flatnonzero(np.any(rows != 0, axis=0))) == cols
        assert np.allclose(rows[:, lay[("t", None)]], -coverage() * strength_limit(b, r, s, limb))


def test_effort_of_the_plan_example():
    # section 3.3: Ava, 0.45 on a crimp (limit 0.9) and 0.30 on a sloper (limit 0.35) -> t = 0.86, right hand
    b = make_body(AVA)
    r = route(holds=[Hold("H1", 1.2, 1.6, "crimp"), Hold("H2", 1.6, 1.65, "sloper"), Hold("H3", 1.5, 3.6, "jug")])
    s = State("H1", "H2")
    t, share, limiting = effort(b, r, s, {"LH": {"f": (0.0, 0.45)}, "RH": {"f": (0.0, 0.30)}})
    assert share["LH"] == pytest.approx(0.5) and share["RH"] == pytest.approx(0.30 / 0.35)
    assert t == pytest.approx(0.857, abs=1e-3) and limiting == "RH"
    # the pull toward the wall counts against the same limit
    t2, share2, _ = effort(b, r, s, {"LH": {"f": (0.0, 0.36), "n": 0.27}, "RH": {"f": (0.0, 0.30)}})
    assert share2["LH"] == pytest.approx(0.5)


def test_bands_read_t():
    assert [band(v)["key"] for v in (0.3, 0.5, 0.7, 0.9, 1.0, 1.2)] == ["easy", "easy", "hard", "limit", "limit", "fail"]
    assert EFFORT_BANDS[-1]["upto"] is None


def _two_jugs():
    holds = [Hold("H1", 1.25, 2.0, "jug"), Hold("H2", 1.75, 2.0, "jug"), Hold("H3", 1.5, 3.6, "jug"),
             Hold("F1", 1.25, 0.6, "foot_big"), Hold("F2", 1.75, 0.6, "foot_big")]
    return route(holds=holds)


def test_lp_with_fixed_forces_is_a_little_conservative():
    # step 10 preview: hanging evenly from two jugs, each hand takes 0.5, t = 0.5 / F_jug.
    # Balance arrives in unit 2.4; here the forces are fixed and the LP only measures them.
    r, s = _two_jugs(), State("H1", "H2")
    for climber in (AVA, BEN, CAL):
        b = make_body(climber)
        forces = {"LH": {"f": (0.0, 0.5)}, "RH": {"f": (0.0, 0.5)}}
        exact = 0.5 / climber.F["jug"]
        assert effort(b, r, s, forces)[0] == pytest.approx(exact)
        lp = solve(b, r, s, fixed=forces)
        assert lp["ok"] and exact - 1e-9 <= lp["t"] <= exact / coverage() + 1e-9


def test_lp_never_below_exact_effort():
    # the polyhedron sits between the balls of radius c F and F, so for the LP's own forces
    # exact share <= t_lp <= exact share / c
    rng = np.random.default_rng(4)
    b = make_body(AVA)
    done = 0
    for _ in range(60):
        dx, dy = rng.uniform(-0.25, 0.25), rng.uniform(-0.15, 0.15)
        holds = [Hold("H1", 1.25 + dx, 2.0 + dy, "crimp"), Hold("H2", 1.75 + dx, 2.0 - dy, "jug"),
                 Hold("H3", 1.5, 3.6, "jug"), Hold("F1", 1.2 + rng.uniform(-0.2, 0.2), 0.7, "foot_big"),
                 Hold("F2", 1.8 + rng.uniform(-0.2, 0.2), 0.8, "foot_chip")]
        lp = solve(b, route(holds=holds), State("H1", "H2", "F1", "F2"))
        if not lp["ok"]:
            continue
        done += 1
        assert lp["t_exact"] - 1e-7 <= lp["t"] <= lp["t_exact"] / coverage() + 1e-7
    assert done > 40


def test_lp_rejects_a_force_outside_the_cone():
    r, s = _two_jugs(), State("H1", "H2")
    lp = solve(make_body(AVA), r, s, fixed={"LH": {"f": (0.5, 0.0)}, "RH": {"f": (0.0, 0.5)}})  # sideways on a jug
    assert not lp["ok"]


def test_lp_size_matches_plan():
    # section 3.7: about 17 variables and 180 constraints for a four-limb state
    P = assemble(make_body(AVA), _two_jugs(), State("H1", "H2", "F1", "F2"))
    assert P.layout.size == 17 and P.G.shape == (len(P.h), 17) and P.A.shape == (6, 17)
    assert {o[0] for o in P.owners} == {"reach", "direction", "strength", "outofwall", "barndoor", "flex"}
    assert {o[0] for o in P.eq_owners} == {"inplane", "outofwall", "barndoor"}
    assert 150 <= len(P.h) + len(P.b) <= 185
    # a figure can ask for the model as it stood in an earlier section
    Q = assemble(make_body(AVA), _two_jugs(), State("H1", "H2", "F1", "F2"), upto="inplane")
    assert Q.A.shape == (3, 17) and {o[0] for o in Q.owners} == {"reach", "direction", "strength"}


# ------------------------------------------------------------ ④ in-plane balance
from betasolver.stance import balance_rows, moment  # noqa: E402


def _hang(x=1.5, y=2.3, kind="jug", **kw):
    return route(holds=[Hold("H1", x, y, kind, **kw), Hold("H3", 1.5, 3.6, "jug"),
                        Hold("F1", 1.5, 1.2, "foot_big")], start_hands=("H1", "H1"))


def test_one_hand_hang_puts_the_body_below_the_hand():
    # step 10: one hand on a jug, no feet: x_c equals the hand's x, the hand force is (0, 1)
    for climber in (AVA, BEN, CAL):
        b = make_body(climber)
        lp = solve(b, _hang(), State("H1", "H1"), no_force="RH")
        assert lp["ok"]
        assert lp["torso"][0] == pytest.approx(1.5, abs=1e-7)
        assert lp["forces"]["LH"]["f"] == pytest.approx((0.0, 1.0), abs=1e-7)
        assert lp["forces"]["LH"]["n"] == pytest.approx(0.0, abs=1e-7)
        assert lp["t_exact"] == pytest.approx(1 / climber.F["jug"])


def test_body_to_the_side_of_one_hand_cannot_balance():
    b, r, s = make_body(AVA), _hang(), State("H1", "H1")
    below = solve(b, r, s, no_force="RH")
    y = below["torso"][1]
    assert solve(b, r, s, no_force="RH", torso=(1.5, y))["ok"]
    for dx in (-0.1, -0.02, 0.02, 0.1):
        assert not solve(b, r, s, no_force="RH", torso=(1.5 + dx, y))["ok"]


def test_two_jugs_share_the_weight():
    # step 10: two jugs at the same height, no feet: half each, t = 0.5 / F_jug, body in the middle
    lp = solve(make_body(AVA), _two_jugs(), State("H1", "H2"))
    assert lp["ok"] and lp["torso"][0] == pytest.approx(1.5, abs=1e-7)
    assert lp["forces"]["LH"]["f"][1] == pytest.approx(0.5, abs=1e-7)
    assert lp["forces"]["RH"]["f"][1] == pytest.approx(0.5, abs=1e-7)
    assert lp["t_exact"] == pytest.approx(0.5 / AVA.F["jug"], abs=1e-6)


def test_sidepull_or_undercling_alone_cannot_balance():
    # step 10: one sidepull and no feet fails; so does an undercling, until a foot is added
    b = make_body(AVA)
    assert not solve(b, _hang(1.6, 2.2, "sidepull", dir=0), State("H1", "H1"))["ok"]
    assert not solve(b, _hang(1.5, 2.0, "undercling"), State("H1", "H1"))["ok"]
    assert solve(b, _hang(1.5, 2.0, "undercling"), State("H1", "H1", "F1"))["ok"]


def test_solution_satisfies_balance_about_any_point():
    # the moment is taken about the origin; with zero net force it is the same about the centre of mass
    b, r, s = make_body(AVA), _two_jugs(), State("H1", "H2", "F1", "F2")
    lp = solve(b, r, s)
    F = lp["forces"]
    assert sum(v["f"][0] for v in F.values()) == pytest.approx(0.0, abs=1e-7)
    assert sum(v["f"][1] for v in F.values()) == pytest.approx(1.0, abs=1e-7)
    assert moment(r, s, F) == pytest.approx(lp["torso"][0], abs=1e-7)
    xc, yc = lp["torso"]
    about_com = sum((r.hold(s.hold(l)).x - xc) * v["f"][1] - (r.hold(s.hold(l)).y - yc) * v["f"][0] for l, v in F.items())
    assert about_com == pytest.approx(0.0, abs=1e-7)


def test_balance_rows_ignore_the_resting_limb():
    r, s = _two_jugs(), State("H1", "H2", "F1", "F2")
    A_, AF = limb_sets(s, no_force="LF")
    lay = layout(AF)
    A, bvec, names = balance_rows(r, s, AF, lay)
    assert A.shape == (3, lay.size) and list(bvec) == [0.0, 1.0, 0.0] and len(names) == 3
    assert A[2, lay[("x", None)].start] == -1.0
    used = set(np.flatnonzero(np.any(A != 0, axis=0)))
    allowed = {i for l in AF for i in range(lay[("f", l)].start, lay[("f", l)].stop)} | {lay[("x", None)].start}
    assert used == allowed



# ------------------------------------------------------------ ⑤ no tipping
from betasolver.stance import tipping_rows, wall_distance  # noqa: E402


def _hand_and_foot(foot_y, hand_kind="jug"):
    holds = [Hold("H1", 1.5, 2.3, hand_kind), Hold("H3", 1.5, 3.6, "jug"), Hold("F1", 1.5, foot_y, "foot_big")]
    return route(holds=holds, start_hands=("H1", "H1"))


@pytest.mark.parametrize("climber", [AVA, BEN, CAL])
def test_pull_in_is_d_over_h(climber):
    # step 10: one hand and one foot, height difference h: the hand's pull-in N = d / h (within 1%)
    b = make_body(climber)
    for h in (0.6, 0.9, 1.2):
        r = _hand_and_foot(2.3 - h)
        lp = solve(b, r, State("H1", "H1", "F1"), no_force="RH")
        assert lp["ok"], h
        N, M = lp["forces"]["LH"]["n"], lp["forces"]["LF"]["m"]
        assert N == pytest.approx(b.d_wall / h, rel=0.01)
        assert M == pytest.approx(N, abs=1e-9)


def test_no_feet_means_no_pull_in():
    lp = solve(make_body(AVA), _two_jugs(), State("H1", "H2"))
    assert lp["ok"] and all(v["n"] == pytest.approx(0.0, abs=1e-9) for v in lp["forces"].values())


def test_a_resting_foot_does_not_hold_the_body_off_the_wall():
    b, r = make_body(AVA), _two_jugs()
    s = State("H1", "H2", "F1")
    assert wall_distance(b, limb_sets(s)[1]) == b.d_wall
    assert wall_distance(b, limb_sets(s, no_force="LF")[1]) == 0.0
    lp = solve(b, r, s, no_force="LF")
    assert lp["ok"] and all(v["n"] == pytest.approx(0.0, abs=1e-9) for v in lp["forces"].values())


def test_solution_does_not_tip():
    b, r, s = make_body(AVA), _two_jugs(), State("H1", "H2", "F1", "F2")
    lp = solve(b, r, s)
    F = lp["forces"]
    pull = sum(F[l]["n"] for l in ("LH", "RH"))
    push = sum(F[l]["m"] for l in ("LF", "RF"))
    assert pull == pytest.approx(push, abs=1e-9) and pull > 0
    y = {l: r.hold(s.hold(l)).y for l in s.on()}
    tip = sum(y[l] * F[l]["n"] for l in ("LH", "RH")) - sum(y[l] * F[l]["m"] for l in ("LF", "RF"))
    assert tip == pytest.approx(b.d_wall, abs=1e-9)


def test_pull_in_limit_of_each_hold():
    # N_k <= beta d_k . f_k: a sloper (beta 0.3) can hold a body in far less than a jug (beta 1.5)
    b = make_body(AVA)
    for kind in ("jug", "crimp", "sloper"):
        r = _hand_and_foot(1.4, kind)
        lp = solve(b, r, State("H1", "H1", "F1"), no_force="RH")
        if not lp["ok"]:
            continue
        f, N = lp["forces"]["LH"]["f"], lp["forces"]["LH"]["n"]
        assert N <= HOLD_TYPES[kind].beta * f[1] + 1e-9


def test_jug_to_sloper_makes_it_harder():
    # step 10: swap one hand from a jug to a sloper in the same state: t goes up or the state fails
    b = make_body(AVA)
    holds = lambda kind: [Hold("H1", 1.25, 2.0, "jug"), Hold("H2", 1.75, 2.0, kind), Hold("H3", 1.5, 3.6, "jug"),
                          Hold("F1", 1.3, 0.9, "foot_big"), Hold("F2", 1.7, 1.0, "foot_big")]
    s = State("H1", "H2", "F1", "F2")
    jug = solve(b, route(holds=holds("jug")), s)
    slo = solve(b, route(holds=holds("sloper")), s)
    assert jug["ok"]
    assert not slo["ok"] or slo["t"] > jug["t"] + 1e-6


def test_tipping_makes_the_hands_work_harder():
    b, r, s = make_body(AVA), _hand_and_foot(1.1), State("H1", "H1", "F1")
    before = solve(b, r, s, no_force="RH", upto="inplane")
    after = solve(b, r, s, no_force="RH")
    assert after["share"]["LH"] > before["share"]["LH"]


def test_tipping_rows_shape():
    b, r, s = make_body(AVA), _two_jugs(), State("H1", "H2", "F1", "F2")
    A_, AF = limb_sets(s)
    lay = layout(AF)
    A, bvec, names, G, h, owner = tipping_rows(b, r, s, AF, lay)
    assert A.shape == (2, lay.size) and list(bvec) == [0.0, b.d_wall] and len(names) == 2
    assert G.shape == (2, lay.size) and owner == ["LH", "RH"] and np.all(h == 0)


# ------------------------------------------------------------ ⑤ no barn-door
from dataclasses import replace as _replace  # noqa: E402

from betasolver.stance import barndoor_rows  # noqa: E402


def _one_hand_two_feet(f1x, f2x, kind="jug"):
    holds = [Hold("H1", 1.5, 2.2, kind), Hold("H3", 1.5, 3.6, "jug"),
             Hold("F1", f1x, 1.0, "foot_big"), Hold("F2", f2x, 1.05, "foot_big")]
    return route(holds=holds, start_hands=("H1", "H1"))


def test_barn_door_fails_with_both_feet_on_one_side(monkeypatch):
    # step 10: one hand, two feet, both feet right of the hand and r = 0: infeasible;
    # the hand between the feet: feasible
    monkeypatch.setitem(HOLD_TYPES, "jug", _replace(HOLD_TYPES["jug"], r=0.0))
    b, s = make_body(AVA), State("H1", "H1", "F1", "F2")
    assert not solve(b, _one_hand_two_feet(1.75, 1.95), s, no_force="RH")["ok"]
    assert solve(b, _one_hand_two_feet(1.25, 1.75), s, no_force="RH")["ok"]


def test_grip_torque_allows_a_small_offset():
    # with the jug's real r, feet slightly to one side are fine; far to one side still fails
    b, s = make_body(AVA), State("H1", "H1", "F1", "F2")
    near = solve(b, _one_hand_two_feet(1.5, 1.6), s, no_force="RH")
    assert near["ok"] and abs(near["forces"]["LH"]["tau"]) > 1e-6
    assert abs(near["forces"]["LH"]["tau"]) <= HOLD_TYPES["jug"].r * AVA.F["jug"] + 1e-9
    assert not solve(b, _one_hand_two_feet(1.95, 2.05), s, no_force="RH")["ok"]
    # without the barn-door rows the far case would pass
    assert solve(b, _one_hand_two_feet(1.95, 2.05), s, no_force="RH", upto="outofwall")["ok"]


def test_solution_does_not_swing_open():
    b, r, s = make_body(AVA), _two_jugs(), State("H1", "H2", "F1", "F2")
    lp = solve(b, r, s)
    F = lp["forces"]
    x = {l: r.hold(s.hold(l)).x for l in s.on()}
    total = sum(x[l] * F[l]["n"] + F[l]["tau"] for l in ("LH", "RH")) - sum(x[l] * F[l]["m"] for l in ("LF", "RF"))
    assert total == pytest.approx(0.0, abs=1e-9)


def test_barndoor_rows_shape():
    b, r, s = make_body(AVA), _two_jugs(), State("H1", "H2", "F1", "F2")
    A_, AF = limb_sets(s)
    lay = layout(AF)
    A, bvec, names, G, h, owner = barndoor_rows(b, r, s, AF, lay)
    assert A.shape == (1, lay.size) and list(bvec) == [0.0] and names == ["no barn-door"]
    assert G.shape == (4, lay.size) and owner == ["LH", "LH", "RH", "RH"]
    assert np.allclose(h, HOLD_TYPES["jug"].r * AVA.F["jug"])



# ------------------------------------------------------------ ⑥ flexibility
from betasolver.model import VARIANTS  # noqa: E402
from betasolver.stance import flex_rows  # noqa: E402


def _high_foot(y):
    holds = [Hold("H1", 1.3, 2.25, "jug"), Hold("H2", 1.75, 2.3, "jug"), Hold("H3", 1.5, 3.6, "jug"),
             Hold("F1", 1.85, y, "foot_big"), Hold("F2", 1.25, 0.75, "foot_big")]
    return route(holds=holds)


def test_flexible_climber_reaches_the_high_foot_stiff_one_does_not():
    # step 10: the same state is feasible with flex = +0.08 and infeasible with flex = -0.10
    r, s = _high_foot(1.45), State("H1", "H2", "F2", "F1")
    stiff = VARIANTS["A_stiff"]
    assert AVA.flex == 0.08 and stiff.flex == -0.10
    assert solve(make_body(AVA), r, s)["ok"]
    assert not solve(make_body(stiff), r, s)["ok"]
    # every other constraint can be met: flexibility alone decides it
    assert solve(make_body(stiff), r, s, upto="barndoor")["ok"]


def test_solution_keeps_feet_below_the_limit():
    for c in (AVA, BEN, CAL):
        b = make_body(c)
        r, s = _high_foot(1.25), State("H1", "H2", "F2", "F1")
        lp = solve(b, r, s)
        assert lp["ok"]
        for limb in ("LF", "RF"):
            assert r.hold(s.hold(limb)).y <= b.foot_limit(lp["torso"]) + 1e-9


def test_a_resting_foot_must_also_get_there():
    r, s = _high_foot(1.45), State("H1", "H2", "F2", "F1")
    stiff = make_body(VARIANTS["A_stiff"])
    assert not solve(stiff, r, s, no_force="RF")["ok"]


def test_flex_rows_shape():
    b, r, s = make_body(AVA), _high_foot(1.45), State("H1", "H2", "F2", "F1")
    lay = layout(limb_sets(s)[1])
    G, h, owner = flex_rows(b, r, s, lay)
    assert G.shape == (2, lay.size) and owner == ["LF", "RF"]
    assert np.all(G[:, lay[("x", None)].start + 1] == -1.0)
    assert h[1] == pytest.approx(b.foot_limit((0, 0)) - 1.45)



# ------------------------------------------------------------ the full check and its diagnosis
from betasolver.stance import diagnose, solve_exact  # noqa: E402

cp = pytest.importorskip("cvxpy")


def test_check_strength_failure_names_the_limb():
    holds = [Hold("H1", 1.3, 2.0, "crimp"), Hold("H2", 1.7, 2.0, "crimp"), Hold("H3", 1.5, 3.6, "jug")]
    res = check(make_body(CAL), route(holds=holds), State("H1", "H2"))
    assert not res.ok and res.reason == "strength" and res.t > 1
    assert res.limb in ("LH", "RH") and res.hold == ("H1" if res.limb == "LH" else "H2")


def test_check_balance_failure():
    res = check(make_body(AVA), _hang(1.6, 2.2, "sidepull", dir=0), State("H1", "H1"))
    assert not res.ok and res.reason == "balance" and res.limb is None


def test_check_reach_failure_from_flexibility():
    # every hold is within reach on its own, but not with the high foot below the stiff hip
    res = check(make_body(VARIANTS["A_stiff"]), _high_foot(1.45), State("H1", "H2", "F2", "F1"))
    assert not res.ok and res.reason == "reach" and res.limb == "RF" and res.hold == "F1"
    assert diagnose(make_body(VARIANTS["A_stiff"]), _high_foot(1.45), State("H1", "H2", "F2", "F1"))[0] == "reach"


def test_no_force_limb_carries_nothing_but_must_reach():
    # step 10: with no_force=k, limb k has no force; moving its hold out of reach makes the check fail
    b, r, s = make_body(AVA), _two_jugs(), State("H1", "H2", "F1", "F2")
    res = check(b, r, s, no_force="RF")
    assert res.ok and "RF" not in res.forces
    far = route(holds=[Hold("H1", 1.25, 2.0, "jug"), Hold("H2", 1.75, 2.0, "jug"), Hold("H3", 1.5, 3.6, "jug"),
                       Hold("F1", 1.25, 0.6, "foot_big"), Hold("F2", 2.9, 0.1, "foot_big")])
    bad = check(b, far, s, no_force="RF")
    assert not bad.ok and bad.reason == "reach"


def test_exact_is_never_above_the_lp():
    b, r, s = make_body(AVA), _two_jugs(), State("H1", "H2", "F1", "F2")
    lp, ex = solve(b, r, s), solve_exact(b, r, s)
    assert lp["ok"] and ex["ok"] and ex["t"] <= lp["t"] + 1e-6
    assert check(b, r, s, exact=True).ok


def test_exact_agrees_on_the_failure_examples():
    assert check(make_body(AVA), _hang(1.6, 2.2, "sidepull", dir=0), State("H1", "H1"), exact=True).reason == "balance"
    assert check(make_body(VARIANTS["A_stiff"]), _high_foot(1.45), State("H1", "H2", "F2", "F1"), exact=True).reason == "reach"



# ------------------------------------------------------------ the linear program against the exact one
from betasolver.compare import compare, summary  # noqa: E402


def test_lp_against_exact_on_500_random_states():
    # step 10: 500 random states; the LP's t* within 10% of the exact one, and the LP never
    # holds a state the exact problem rejects
    records = compare()  # n = 500 by default; same cache as the site build
    S = summary(records)
    assert S["n"] == 500 and S["both"] > 200
    assert S["lp_only"] == 0
    assert S["max_gap"] < 0.10 and S["min_gap"] >= -1e-6
    for r in records:
        if r["lp_ok"] and r["exact_ok"]:
            assert r["lp_t"] >= r["exact_t"] - 1e-6
