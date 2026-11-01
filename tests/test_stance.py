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


def test_remaining_constraints_not_built_yet():
    with pytest.raises(NotImplementedError):
        check(make_body(AVA), route(), State("H1", "H2", "F1", "F2"))


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
