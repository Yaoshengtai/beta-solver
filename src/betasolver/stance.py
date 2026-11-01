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

import math
from dataclasses import dataclass, field

import numpy as np

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
     "meaning": "Centre of mass on the wall: the body position.",
     "climbing": "Sit low or high, lean left or right."},
    {"name": "f", "tex": r"f_k", "size": 2, "per": "pushing",
     "meaning": "In-wall force from the hold on limb k.",
     "climbing": "How each hand and foot is loaded."},
    {"name": "N", "tex": r"N_k \ge 0", "size": 1, "per": "pushing hand",
     "meaning": "Pull of hand k toward the wall.",
     "climbing": "Holding yourself in."},
    {"name": "M", "tex": r"M_k \ge 0", "size": 1, "per": "pushing foot",
     "meaning": "Push of the wall on foot k.",
     "climbing": "Pressing the foot in."},
    {"name": "tau", "tex": r"\tau_k", "size": 1, "per": "pushing hand",
     "meaning": "Twist of hand k about a vertical axis.",
     "climbing": "Gripping against a barn-door."},
    {"name": "t", "tex": r"t", "size": 1, "per": "body",
     "meaning": "Effort: the largest share of strength any limb uses.",
     "climbing": "How tiring the position is."},
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
    a crimp) is ruled out without solving anything. So is a pair of holds
    farther apart than the two limbs plus the distance between their
    attachment points (section 3.7).
    """
    A, _ = limb_sets(state, no_force)
    for limb in A:
        h = route.hold(state.hold(limb))
        kind, _ = h.contact(limb)
        if body.climber.strength(kind.name) <= 0:
            return Result(ok=False, reason="unusable", limb=limb, hold=h.id)
    pts = targets(body, route, state)
    for i, a in enumerate(A):
        for b in A[i + 1:]:
            gap = math.dist(pts[a], pts[b])
            span = body.L[a] + body.L[b] + math.dist(body.offsets[a], body.offsets[b])
            if gap > span:
                return Result(ok=False, reason="reach", limb=b, hold=state.hold(b))
    return None


# ------------------------------------------------------------ ① reach

REACH_SIDES = 16  # the disc of each limb is replaced by an inscribed regular 16-gon


def targets(body: Body, route: Route, state: State) -> dict:
    """Position of the hold each limb is on."""
    return {l: (route.hold(state.hold(l)).x, route.hold(state.hold(l)).y) for l in state.on()}


def centres(body: Body, route: Route, state: State) -> dict:
    """For each limb on a hold, the centre p_k - o_k of the disc of torso positions that reach it."""
    pts = targets(body, route, state)
    return {l: (pts[l][0] - body.offsets[l][0], pts[l][1] - body.offsets[l][1]) for l in pts}


def reaches_all(body: Body, route: Route, state: State, torso, tol: float = 1e-9) -> dict:
    """Exact form of ①: for each limb on a hold, is it within reach from this torso position?"""
    pts = targets(body, route, state)
    return {l: body.reaches(l, torso, pts[l], tol) for l in pts}


def reach_rows(body: Body, route: Route, state: State, sides: int = REACH_SIDES):
    """① as linear inequalities G x <= h in the torso position x = (x_c, y_c).

    For every limb k on a hold and every direction u_i of the polygon:
        u_i . (p_k - x - o_k) <= L_k cos(pi / sides)
    which is the inscribed regular polygon of the disc |p_k - x - o_k| <= L_k.
    """
    c = math.cos(math.pi / sides)
    rows, rhs, owner = [], [], []
    for limb, (cx, cy) in centres(body, route, state).items():
        for i in range(sides):
            a = 2 * math.pi * i / sides
            u = (math.cos(a), math.sin(a))
            rows.append((-u[0], -u[1]))
            rhs.append(body.L[limb] * c - (u[0] * cx + u[1] * cy))
            owner.append(limb)
    return np.array(rows, dtype=float), np.array(rhs, dtype=float), owner


def _clip(poly, a, b, c):
    """Keep the part of a convex polygon where a*x + b*y <= c."""
    out = []
    n = len(poly)
    for i in range(n):
        p, q = poly[i], poly[(i + 1) % n]
        fp, fq = a * p[0] + b * p[1] - c, a * q[0] + b * q[1] - c
        if fp <= 0:
            out.append(p)
        if fp * fq < 0:
            t = fp / (fp - fq)
            out.append((p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])))
    return out


def torso_region(body: Body, route: Route, state: State, sides: int = 128) -> list:
    """Every torso position from which all limbs reach their holds, as a polygon.

    This is the intersection of one disc per limb (centre p_k - o_k, radius
    L_k), each replaced by an inscribed polygon with `sides` sides. An empty
    list means no position reaches every hold. The intersection of discs is
    convex, which section 4.2 relies on.
    """
    G, h, _ = reach_rows(body, route, state, sides)
    cs = list(centres(body, route, state).values())
    big = max(body.L.values()) + 1.0
    x0, y0 = cs[0]
    poly = [(x0 - big, y0 - big), (x0 + big, y0 - big), (x0 + big, y0 + big), (x0 - big, y0 + big)]
    for (a, b), c in zip(G, h):
        poly = _clip(poly, a, b, c)
        if len(poly) < 3:
            return []
    return poly


def check(body: Body, route: Route, state: State, no_force: str | None = None) -> Result:
    """Check one state: can the climber hold still here, and with what effort t?"""
    validate(body, route, state)
    A, AF = limb_sets(state, no_force)
    early = precheck(body, route, state, no_force)
    if early is not None:
        return early
    if not torso_region(body, route, state, REACH_SIDES):
        return Result(ok=False, reason="reach")
    raise NotImplementedError("constraints 2 to 6 are added in units 2.2 to 2.9")
