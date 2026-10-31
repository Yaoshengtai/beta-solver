"""The state check (step 3 of the plan).

With every limb on a fixed hold, can the climber hold still, and if so how
hard is the most loaded limb working? The answer is a small convex problem:
find a body position and contact forces that satisfy six constraints, with
the smallest effort t.

This module holds the framework: states, the set of unknowns, the result
type and the checks done before any solving. The six constraints are added
one by one in units 2.1 to 2.9.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .body import Body
from .model import FEET, HANDS, LIMBS, Route

LIMB_NAMES = {"LH": "left hand", "RH": "right hand", "LF": "left foot", "RF": "right foot"}


@dataclass(frozen=True)
class State:
    """Which hold each limb is on. Hands are always on a hold; a foot may hang (None)."""

    LH: str
    RH: str
    LF: str | None = None
    RF: str | None = None

    def __post_init__(self):
        if not self.LH or not self.RH:
            raise ValueError("both hands must be on a hold")

    def hold(self, limb: str) -> str | None:
        return getattr(self, limb)

    def on(self) -> tuple[str, ...]:
        """The limbs that are on a hold: the set A."""
        return tuple(l for l in LIMBS if self.hold(l) is not None)

    def as_dict(self) -> dict:
        return {l: self.hold(l) for l in LIMBS}


def limb_sets(state: State, no_force: str | None = None) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """A: limbs on a hold. A_F: those of them that carry force.

    With no_force=k, limb k must still reach its hold but carries nothing.
    This is how a move's release and catch checks are written (step 4).
    """
    A = state.on()
    if no_force is not None and no_force not in A:
        raise ValueError(f"no_force={no_force!r} is not on a hold in this state")
    AF = tuple(l for l in A if l != no_force)
    return A, AF


# ------------------------------------------------------------------ unknowns

# The unknowns of the state check, in the order of section 3.0. `per` says
# which limbs carry one: "body" (one for the whole body), "pushing" (every
# limb in A_F), "pushing hand" or "pushing foot".
UNKNOWNS = [
    {"name": "x", "tex": r"x = (x_c,\ y_c)", "size": 2, "per": "body",
     "meaning": "Position of the centre of mass on the wall: the body position.",
     "climbing": "On the same four holds the body can swing left and right, or move up and down."},
    {"name": "f", "tex": r"f_k", "size": 2, "per": "pushing",
     "meaning": "Force from the hold on limb k, in the plane of the wall (sideways, up).",
     "climbing": "Which way, and how hard, each hand and foot is loaded."},
    {"name": "N", "tex": r"N_k \ge 0", "size": 1, "per": "pushing hand",
     "meaning": "Pull of hand k toward the wall.",
     "climbing": "A hand has to hold you in, or you tip backwards."},
    {"name": "M", "tex": r"M_k \ge 0", "size": 1, "per": "pushing foot",
     "meaning": "Push of the wall on foot k, away from the wall.",
     "climbing": "Pressing the foot into the wall."},
    {"name": "tau", "tex": r"\tau_k", "size": 1, "per": "pushing hand",
     "meaning": "Twisting torque of hand k about a vertical axis.",
     "climbing": "Gripping hard to stop the body swinging open like a door."},
    {"name": "t", "tex": r"t", "size": 1, "per": "body",
     "meaning": "Effort: the largest share of its strength that any limb uses.",
     "climbing": "How tiring this position is. It is what the check minimises."},
]

CONSTRAINTS = [
    {"n": 1, "key": "reach", "name": "Reach", "section": "check-reach",
     "climbing": "Every hold is within reach of its arm or leg.", "limits": "body position"},
    {"n": 2, "key": "direction", "name": "Direction", "section": "check-direction",
     "climbing": "Each hold can only push you in certain directions.", "limits": "direction of the in-wall forces"},
    {"n": 3, "key": "strength", "name": "Strength", "section": "check-strength",
     "climbing": "No hand or foot goes over its strength.", "limits": "size of the forces; defines t"},
    {"n": 4, "key": "inplane", "name": "In-plane balance", "section": "check-inplane",
     "climbing": "No falling, no sliding sideways, no swinging.", "limits": "in-wall forces and body position"},
    {"n": 5, "key": "outofwall", "name": "Out-of-wall balance", "section": "check-tipping",
     "climbing": "No tipping backwards, no barn-door.", "limits": "hand pull, foot push, grip torque"},
    {"n": 6, "key": "flex", "name": "Flexibility", "section": "check-flex",
     "climbing": "A foot cannot go too far above the hip.", "limits": "body position"},
]


def _applies(per: str, limb: str) -> bool:
    return per == "pushing" or (per == "pushing hand" and limb in HANDS) or (per == "pushing foot" and limb in FEET)


@dataclass(frozen=True)
class Layout:
    """Where each unknown sits in the solver's vector of variables."""

    index: dict  # (name, limb or None) -> slice
    size: int

    def __getitem__(self, key) -> slice:
        return self.index[key]


def layout(AF: tuple[str, ...]) -> Layout:
    index, i = {}, 0
    for u in UNKNOWNS:
        owners = [None] if u["per"] == "body" else [l for l in AF if _applies(u["per"], l)]
        for limb in owners:
            index[(u["name"], limb)] = slice(i, i + u["size"])
            i += u["size"]
    return Layout(index, i)


# -------------------------------------------------------------------- result

REASONS = {
    "unusable": "a hold this climber cannot use at all",
    "reach": "cannot reach",
    "strength": "not strong enough",
    "balance": "direction or balance does not work",
}


@dataclass(frozen=True)
class Result:
    ok: bool
    t: float | None = None
    torso: tuple[float, float] | None = None
    forces: dict = field(default_factory=dict)
    limiting: str | None = None
    reason: str | None = None  # one of REASONS when not ok
    limb: str | None = None    # the limb the reason is about, if any
    hold: str | None = None

    def to_dict(self) -> dict:
        return {"ok": self.ok, "t": self.t, "torso": self.torso, "forces": self.forces,
                "limiting": self.limiting, "reason": self.reason, "limb": self.limb, "hold": self.hold}


# --------------------------------------------------------------------- check

def validate(body: Body, route: Route, state: State) -> None:
    """Raise ValueError if the state is not a legal state on this route."""
    for limb in state.on():
        hid = state.hold(limb)
        try:
            h = route.hold(hid)
        except KeyError:
            raise ValueError(f"{limb}: unknown hold {hid!r}") from None
        if not h.usable_by(limb, route.allow_feet_on_handholds):
            raise ValueError(f"{LIMB_NAMES[limb]} cannot use {hid} ({h.type}, use={h.use})")


def precheck(body: Body, route: Route, state: State, no_force: str | None = None) -> Result | None:
    """Cheap tests done before any solving. Returns a failed Result, or None to go on.

    A hold whose strength limit is zero for this climber (an injured finger on
    a crimp) is ruled out without solving anything (section 3.7).
    """
    A, _ = limb_sets(state, no_force)
    for limb in A:
        h = route.hold(state.hold(limb))
        kind, _ = h.contact(limb)
        if body.climber.strength(kind.name) <= 0:
            return Result(ok=False, reason="unusable", limb=limb, hold=h.id)
    return None


def check(body: Body, route: Route, state: State, no_force: str | None = None) -> Result:
    """Check one state: can the climber hold still here, and with what effort t?"""
    validate(body, route, state)
    A, AF = limb_sets(state, no_force)
    early = precheck(body, route, state, no_force)
    if early is not None:
        return early
    raise NotImplementedError("the six constraints are added in units 2.1 to 2.9")
