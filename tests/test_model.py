import pytest

from betasolver.model import HOLD_TYPES, WALL_H, WALL_W, Hold, Route, hold_types_table

# Step 1 of the plan: type -> (dir, alpha, strength limit, beta, r, feet only)
PLAN = {
    "jug": (90, 70, "jug", 1.5, 0.05, False),
    "crimp": (90, 40, "crimp", 0.8, 0.02, False),
    "sloper": (90, 15, "sloper", 0.3, 0.0, False),
    "pinch": (90, 55, "pinch", 1.0, 0.04, False),
    "sidepull": (None, 35, "pinch", 0.8, 0.03, False),
    "undercling": (270, 40, "crimp", 0.8, 0.02, False),
    "foot_big": (90, 60, "foot", None, None, True),
    "foot_chip": (90, 35, "foot", None, None, True),
}


def test_hold_types_match_plan():
    assert list(HOLD_TYPES) == list(PLAN)
    for name, (d, alpha, grip, beta, r, feet) in PLAN.items():
        t = HOLD_TYPES[name]
        assert (t.dir, t.alpha, t.grip, t.beta, t.r, t.feet_only) == (d, alpha, grip, beta, r, feet), name


def test_every_hand_hold_is_a_convex_cone():
    # The direction sets must be convex cones: half-width below 90 degrees.
    assert all(0 < t.alpha < 90 for t in HOLD_TYPES.values())


def test_wall_size():
    assert (WALL_W, WALL_H) == (3.0, 4.2)


def test_table_is_plain_data_in_order():
    rows = hold_types_table()
    assert [r["name"] for r in rows] == list(PLAN)
    assert all(isinstance(r["note"], str) and r["note"] for r in rows)


def test_hold_defaults_from_type():
    h = Hold("H1", 1.0, 2.0, "undercling")
    assert h.dir == 270 and h.use == "both"
    assert Hold("F1", 1.0, 0.5, "foot_chip").use == "foot"


def test_sidepull_needs_left_or_right():
    with pytest.raises(ValueError):
        Hold("H1", 1.0, 2.0, "sidepull")
    with pytest.raises(ValueError):
        Hold("H1", 1.0, 2.0, "sidepull", dir=90)
    assert Hold("H1", 1.0, 2.0, "sidepull", dir=180).dir == 180


def test_footholds_are_feet_only():
    with pytest.raises(ValueError):
        Hold("F1", 1.0, 0.5, "foot_big", use="both")
    f = Hold("F1", 1.0, 0.5, "foot_big")
    assert f.usable_by("LF") and not f.usable_by("RH")


def test_unknown_type_rejected():
    with pytest.raises(ValueError):
        Hold("H1", 1.0, 2.0, "volume")


def test_foot_on_hand_hold_uses_foothold_rules():
    h = Hold("H1", 1.0, 2.0, "sloper")
    t, d = h.contact("LF")
    assert t.name == "foot_big" and d == 90
    t, d = h.contact("LH")
    assert t.name == "sloper"


def test_campus_route_keeps_feet_off_hand_holds():
    h = Hold("H1", 1.0, 2.0, "crimp")
    assert h.usable_by("RF", allow_feet_on_handholds=True)
    assert not h.usable_by("RF", allow_feet_on_handholds=False)


def _route(**kw):
    holds = [Hold("H1", 1.0, 1.6, "jug"), Hold("H2", 1.5, 1.6, "jug"),
             Hold("H3", 1.2, 3.5, "jug"), Hold("F1", 1.2, 0.5, "foot_big")]
    args = dict(id="t", title="t", lesson="t", holds=holds, start_hands=("H1", "H2"), finish="H3")
    args.update(kw)
    return Route(**args)


def test_route_roles_and_dict():
    r = _route()
    d = r.to_dict()
    roles = {h["id"]: h["role"] for h in d["holds"]}
    assert roles == {"H1": "start", "H2": "start", "H3": "finish", "F1": "normal"}
    assert d["wall"] == {"w": 3.0, "h": 4.2} and d["allow_feet_on_handholds"] is True


@pytest.mark.parametrize("kw", [
    {"finish": "F1"},                       # top must be a hand hold
    {"start_hands": ("H1", "X9")},          # unknown hold
    {"start_hands": ("H1",)},               # need two
    {"holds": [Hold("H1", 1.0, 1.6, "jug"), Hold("H1", 1.5, 1.6, "jug")]},  # duplicate id
    {"holds": [Hold("H1", 1.0, 1.6, "jug"), Hold("H2", 3.5, 1.6, "jug"), Hold("H3", 1.2, 3.5, "jug")]},  # off wall
])
def test_bad_routes_rejected(kw):
    with pytest.raises(ValueError):
        _route(**kw)


# ---------------------------------------------------------------- climbers
from betasolver.model import AVA, BEN, CAL, PROFILES, VARIANTS, Climber  # noqa: E402

# Step 1: key -> (name, H, span, jug, crimp, sloper, pinch, foot, flex)
PLAN_CLIMBERS = {
    "A": ("Ava", 1.60, 1.60, 1.2, 0.9, 0.35, 0.6, 1.5, 0.08),
    "B": ("Ben", 1.85, 1.90, 1.2, 0.6, 0.55, 0.6, 1.5, -0.10),
    "C": ("Cal", 1.70, 1.70, 0.8, 0.35, 0.25, 0.35, 1.5, 0.00),
    "D": ("Ava (injured)", 1.60, 1.60, 1.2, 0.0, 0.35, 0.6, 1.5, 0.08),
}


def test_profiles_match_plan():
    assert list(PROFILES) == list(PLAN_CLIMBERS)
    for key, (name, H, span, jug, crimp, sloper, pinch, foot, flex) in PLAN_CLIMBERS.items():
        c = PROFILES[key]
        assert (c.name, c.H, c.span, c.flex) == (name, H, span, flex), key
        assert c.F == {"jug": jug, "crimp": crimp, "sloper": sloper, "pinch": pinch, "foot": foot}, key
        assert c.w_cross == 0.15


def test_injured_ava_differs_from_ava_only_in_crimp():
    d = PROFILES["D"]
    assert {k for k in AVA.F if AVA.F[k] != d.F[k]} == {"crimp"}
    assert (d.H, d.span, d.flex, d.w_cross) == (AVA.H, AVA.span, AVA.flex, AVA.w_cross)


def test_strength_follows_hold_type():
    assert AVA.strength("sidepull") == AVA.F["pinch"]
    assert AVA.strength("undercling") == AVA.F["crimp"]
    assert AVA.strength("foot_chip") == AVA.F["foot"]
    d = PROFILES["D"]
    assert not d.can_use("crimp") and not d.can_use("undercling") and d.can_use("pinch")


def test_variants_change_one_attribute():
    expected = {"A_stiff": ("flex", -0.10), "A_cross0": ("w_cross", 0.0), "A_cross4": ("w_cross", 0.4)}
    assert set(VARIANTS) == set(expected) | {"C_grip"}
    for key, (attr, value) in expected.items():
        v = VARIANTS[key]
        changed = [a for a in ("H", "span", "F", "flex", "w_cross") if getattr(v, a) != getattr(AVA, a)]
        assert changed == [attr] and getattr(v, attr) == value, key
    assert VARIANTS["A_stiff"].flex == BEN.flex  # "Ava with Ben's hip flexibility"
    # "Cal with Ava's grip on jugs": only the jug strength changes
    g = VARIANTS["C_grip"]
    assert [k for k in g.F if g.F[k] != CAL.F[k]] == ["jug"] and g.F["jug"] == AVA.F["jug"]
    assert (g.H, g.span, g.flex, g.w_cross) == (CAL.H, CAL.span, CAL.flex, CAL.w_cross)


def test_variant_does_not_touch_the_original():
    AVA.variant("X", "x", F={"jug": 0.1})
    assert AVA.F["jug"] == 1.2


def test_wall_distance_scales_with_height():
    assert abs(AVA.d_wall - 0.24) < 1e-9 and abs(CAL.to_dict()["d_wall"] - 0.255) < 1e-9


def test_bad_profiles_rejected():
    with pytest.raises(ValueError):
        Climber("X", "x", 1.7, 1.7, {"jug": 1.0}, 0.0)
    with pytest.raises(ValueError):
        AVA.variant("X", "x", F={"jug": -0.1})
