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


def test_constraints_not_built_yet():
    with pytest.raises(NotImplementedError):
        check(make_body(AVA), route(), State("H1", "H2", "F1", "F2"))
