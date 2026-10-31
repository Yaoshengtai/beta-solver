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
