"""The state check (step 3 of the plan).

With every limb on a fixed hold, can the climber hold still, and if so how
hard is the most loaded limb working? The answer is a small convex problem:
find a body position and contact forces that satisfy six constraints, with
the smallest effort t.

This module holds the framework (states, unknowns, result type, checks done
before any solving) and the constraints built so far: reach, direction,
strength, in-plane balance, no tipping, no barn-door, plus the linear
program that puts them together. Flexibility is added in unit 2.7.
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


# ------------------------------------------------------------ ② direction

def cone(route: Route, state: State, limb: str) -> tuple[float, float]:
    """Centre direction theta and half-width alpha (degrees) of limb k's cone K_k.

    A foot on a hand hold uses the foothold's cone (step 1).
    """
    kind, theta = route.hold(state.hold(limb)).contact(limb)
    return theta, float(kind.alpha)


def in_cone(f, theta: float, alpha: float, tol: float = 1e-9) -> bool:
    """Exact form of ②: d^T f >= |f| cos(alpha), with d the unit vector at angle theta."""
    t = math.radians(theta)
    d = (math.cos(t), math.sin(t))
    return d[0] * f[0] + d[1] * f[1] >= math.hypot(f[0], f[1]) * math.cos(math.radians(alpha)) - tol


def cone_normals(theta: float, alpha: float):
    """Inward normals of the two edges of the cone. In 2D the cone is exactly
    {f : n_minus . f >= 0 and n_plus . f >= 0} when alpha < 90 degrees."""
    lo, hi = math.radians(theta - alpha), math.radians(theta + alpha)
    return (-math.sin(lo), math.cos(lo)), (math.sin(hi), -math.cos(hi))


def direction_rows(route: Route, state: State, AF: tuple[str, ...], lay: Layout):
    """② as linear inequalities G z <= 0 on the solver's variable vector z.

    Two rows per limb that pushes: -n . f_k <= 0 for both edge normals n.
    """
    rows, owner = [], []
    for limb in AF:
        theta, alpha = cone(route, state, limb)
        sl = lay[("f", limb)]
        for n in cone_normals(theta, alpha):
            row = np.zeros(lay.size)
            row[sl] = (-n[0], -n[1])
            rows.append(row)
            owner.append(limb)
    G = np.array(rows, dtype=float).reshape(len(rows), lay.size)
    return G, np.zeros(len(rows)), owner


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


# ------------------------------------------------------------ ③ strength

# The ball |(f_k, N_k)| <= t F_k is replaced by a polyhedron: 8 directions in
# the wall plane times 3 elevations toward the wall, with every other layer
# turned by half a step. Each direction u gives one row u . (f_k, N_k) <= c t F_k.
# Alone, those rows describe a polyhedron slightly larger than the ball; the
# coverage factor c shrinks it until its farthest corner touches the ball, so
# the solver never credits a limb with more strength than it has.
STRENGTH_AZIMUTHS = 8
STRENGTH_ELEVATIONS = (5.0, 35.0, 70.0)

# Reading the effort t (section 3.3): upper end of each band and its name.
EFFORT_BANDS = [
    {"upto": 0.5, "key": "easy", "name": "easy"},
    {"upto": 0.8, "key": "hard", "name": "hard work"},
    {"upto": 1.0, "key": "limit", "name": "near the limit"},
    {"upto": None, "key": "fail", "name": "cannot hold: not strong enough"},
]


def strength_directions(azimuths: int = STRENGTH_AZIMUTHS, elevations=STRENGTH_ELEVATIONS) -> np.ndarray:
    """Unit vectors u = (in-wall x, in-wall y, toward/away from the wall) of the polyhedron's faces."""
    out = []
    for j, e in enumerate(elevations):
        e = math.radians(e)
        turn = math.pi / azimuths if j % 2 else 0.0
        for i in range(azimuths):
            a = 2 * math.pi * i / azimuths + turn
            out.append((math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)))
    return np.array(out)


def _corners(U: np.ndarray) -> np.ndarray:
    """Corners of {v : U v <= 1, v_z >= 0} (the out-of-wall part is never negative)."""
    A = np.vstack([U, [0.0, 0.0, -1.0]])
    b = np.r_[np.ones(len(U)), 0.0]
    pts = []
    n = len(A)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                M = A[[i, j, k]]
                if abs(np.linalg.det(M)) < 1e-9:
                    continue
                v = np.linalg.solve(M, b[[i, j, k]])
                if np.all(A @ v <= b + 1e-9):
                    pts.append(v)
    return np.array(pts)


_COVERAGE: dict = {}


def coverage(azimuths: int = STRENGTH_AZIMUTHS, elevations=STRENGTH_ELEVATIONS) -> float:
    """Coverage factor c: with rows U v <= c the polyhedron's farthest corner is exactly at |v| = 1."""
    key = (azimuths, tuple(elevations))
    if key not in _COVERAGE:
        U = strength_directions(azimuths, elevations)
        _COVERAGE[key] = 1.0 / float(np.max(np.linalg.norm(_corners(U), axis=1)))
    return _COVERAGE[key]


def strength_limit(body: Body, route: Route, state: State, limb: str) -> float:
    """F_k: the most force limb k can take on its hold. A foot on a hand hold uses the foot limit."""
    kind, _ = route.hold(state.hold(limb)).contact(limb)
    return body.climber.strength(kind.name)


def _normal(limb: str) -> str:
    """The out-of-wall unknown of a limb: the pull-in N for a hand, the push M for a foot."""
    return "N" if limb in HANDS else "M"


def strength_rows(body: Body, route: Route, state: State, AF: tuple[str, ...], lay: Layout):
    """③ as linear inequalities G z <= 0: u . (f_k, N_k or M_k) - c F_k t <= 0 for every face u."""
    U, c = strength_directions(), coverage()
    rows, owner = [], []
    for limb in AF:
        F = strength_limit(body, route, state, limb)
        sf, sn, st = lay[("f", limb)], lay[(_normal(limb), limb)], lay[("t", None)]
        for u in U:
            row = np.zeros(lay.size)
            row[sf] = u[:2]
            row[sn] = u[2]
            row[st] = -c * F
            rows.append(row)
            owner.append(limb)
    G = np.array(rows, dtype=float).reshape(len(rows), lay.size)
    return G, np.zeros(len(rows)), owner


def effort(body: Body, route: Route, state: State, forces: dict) -> tuple[float, dict, str | None]:
    """Exact form of ③ for given forces: each limb's share of its strength, and t, the largest share.

    forces: limb -> {"f": (fx, fy), "n": N} for a hand or {"f": ..., "m": M} for a foot.
    Returns (t, {limb: share}, limiting limb).
    """
    share = {}
    for limb, F in forces.items():
        out = F.get("n", 0.0) if limb in HANDS else F.get("m", 0.0)
        total = math.sqrt(F["f"][0] ** 2 + F["f"][1] ** 2 + out ** 2)
        share[limb] = total / strength_limit(body, route, state, limb)
    if not share:
        return 0.0, {}, None
    limiting = max(share, key=share.get)
    return share[limiting], share, limiting


def band(t: float) -> dict:
    """Which row of the effort table t falls in."""
    for b in EFFORT_BANDS:
        if b["upto"] is None or t <= b["upto"]:
            return b
    return EFFORT_BANDS[-1]


# ------------------------------------------------------------ ⑤ out-of-wall balance: no tipping

def wall_distance(body: Body, AF: tuple[str, ...]) -> float:
    """d: how far the centre of mass is from the wall.

    With no foot pushing, the body swings in under the hands and hangs flat
    against the wall, so d = 0 and every pull-in and push is zero (section 3.5).
    """
    return body.d_wall if any(l in FEET for l in AF) else 0.0


def tipping_rows(body: Body, route: Route, state: State, AF: tuple[str, ...], lay: Layout):
    """⑤ (a), (b) as equations and the pull-in limit as inequalities.

    The z axis points out of the wall; the centre of mass is at height y_c,
    distance d from the wall. Hands are pulled in by N_k, feet pushed out by M_k:
        (a)  sum M_k - sum N_k = 0                   (no moving away from the wall)
        (b)  sum p_ky N_k - sum p_ky M_k = d         (no tipping backwards)
        N_k <= beta_k d_k . f_k   for hands          (pull-in limit of the hold)
    Gravity's moment about the horizontal axis in the wall is d times the
    weight, which the hand pulls high up and the foot pushes low down must cancel.
    """
    d = wall_distance(body, AF)
    A = np.zeros((2, lay.size))
    for limb in AF:
        h = route.hold(state.hold(limb))
        if limb in HANDS:
            i = lay[("N", limb)].start
            A[0, i], A[1, i] = -1.0, h.y
        else:
            i = lay[("M", limb)].start
            A[0, i], A[1, i] = 1.0, -h.y
    rows, owner = [], []
    for limb in AF:
        if limb not in HANDS:
            continue
        kind, theta = route.hold(state.hold(limb)).contact(limb)
        t = math.radians(theta)
        row = np.zeros(lay.size)
        row[lay[("N", limb)].start] = 1.0
        sf = lay[("f", limb)]
        row[sf.start] = -kind.beta * math.cos(t)
        row[sf.start + 1] = -kind.beta * math.sin(t)
        rows.append(row)
        owner.append(limb)
    G = np.array(rows, dtype=float).reshape(len(rows), lay.size)
    return A, np.array([0.0, d]), ["push equals pull", "no tipping"], G, np.zeros(len(rows)), owner


# ------------------------------------------------------------ ⑤ out-of-wall balance: no barn-door

def barndoor_rows(body: Body, route: Route, state: State, AF: tuple[str, ...], lay: Layout):
    """⑤ (c) as an equation and the grip torque limit as inequalities.

        (c)  sum p_kx N_k - sum p_kx M_k + sum tau_k = 0     (no swinging open about a vertical axis)
        |tau_k| <= r_k F_k   for hands                       (how hard the hand can twist on the hold)
    With one hand and two feet, (a) and (c) put the hand at the push-weighted
    middle of the feet, give or take what the grip torque can make up. Both
    feet on one side of the hand and the body swings open like a door.
    """
    A = np.zeros((1, lay.size))
    rows, rhs, owner = [], [], []
    for limb in AF:
        h = route.hold(state.hold(limb))
        if limb in HANDS:
            A[0, lay[("N", limb)].start] = h.x
            it = lay[("tau", limb)].start
            A[0, it] = 1.0
            kind, _ = h.contact(limb)
            cap = kind.r * strength_limit(body, route, state, limb)
            for sign in (1.0, -1.0):
                row = np.zeros(lay.size)
                row[it] = sign
                rows.append(row)
                rhs.append(cap)
                owner.append(limb)
        else:
            A[0, lay[("M", limb)].start] = -h.x
    G = np.array(rows, dtype=float).reshape(len(rows), lay.size)
    return A, np.zeros(1), ["no barn-door"], G, np.array(rhs, dtype=float), owner


# ------------------------------------------------------- the linear program

def variable_bounds(lay: Layout) -> list:
    """N, M and t are never negative; the body position, forces and grip torques are free."""
    b = [(None, None)] * lay.size
    for (name, _), sl in lay.index.items():
        if name in ("N", "M", "t"):
            for i in range(sl.start, sl.stop):
                b[i] = (0.0, None)
    return b


# ------------------------------------------------------------ ④ in-plane balance

def balance_rows(route: Route, state: State, AF: tuple[str, ...], lay: Layout):
    """④ as linear equations A z = b, with gravity (0, -1) acting at the centre of mass x.

    Three rows, summing over the limbs that push:
        sum f_kx = 0                                (no sliding sideways)
        sum f_ky = 1                                (no falling)
        sum (p_kx f_ky - p_ky f_kx) - x_c = 0       (no swinging)
    Moments are taken about the wall origin, where the hold positions p_k are
    known numbers, so every row is linear in the unknowns. Taken about the
    centre of mass, the moment would contain (p_k - x) x f_k, a product of
    two unknowns.
    """
    A = np.zeros((3, lay.size))
    b = np.array([0.0, 1.0, 0.0])
    for limb in AF:
        h = route.hold(state.hold(limb))
        sf = lay[("f", limb)]
        A[0, sf.start] = 1.0
        A[1, sf.start + 1] = 1.0
        A[2, sf.start] = -h.y
        A[2, sf.start + 1] = h.x
    A[2, lay[("x", None)].start] = -1.0
    return A, b, ["force x", "force y", "moment"]


def moment(route: Route, state: State, forces: dict) -> float:
    """Total moment of the contact forces about the wall origin: sum p_kx f_ky - p_ky f_kx."""
    out = 0.0
    for limb, F in forces.items():
        h = route.hold(state.hold(limb))
        out += h.x * F["f"][1] - h.y * F["f"][0]
    return out


# ------------------------------------------------------- the linear program

@dataclass(frozen=True)
class Program:
    """The state check as a linear program: minimise t subject to G z <= h and A z = b."""

    layout: Layout
    G: np.ndarray
    h: np.ndarray
    owners: list  # (constraint, limb) of each row of G
    A: np.ndarray
    b: np.ndarray
    eq_owners: list  # (constraint, name) of each row of A


# The constraint families in the order they are built; `upto` stops after one of them.
BUILT = ("reach", "direction", "strength", "inplane", "outofwall", "barndoor")


def assemble(body: Body, route: Route, state: State, no_force: str | None = None, upto: str | None = None) -> Program:
    """The constraints built so far, as one linear program.

    upto: stop after this constraint family (one of BUILT), so a figure can
    show the model as it stood in its own section. None means all of them.
    """
    keep = BUILT if upto is None else BUILT[: BUILT.index(upto) + 1]
    A_, AF = limb_sets(state, no_force)
    lay = layout(AF)
    Gs, hs, owners = [], [], []
    Aeq, beq, eq_owners = [], [], []
    if "reach" in keep:
        Gr, hr, orr = reach_rows(body, route, state)
        G1 = np.zeros((len(Gr), lay.size))
        G1[:, lay[("x", None)]] = Gr
        Gs.append(G1); hs.append(hr); owners += [("reach", l) for l in orr]
    if "direction" in keep:
        G2, h2, o2 = direction_rows(route, state, AF, lay)
        Gs.append(G2); hs.append(h2); owners += [("direction", l) for l in o2]
    if "strength" in keep:
        G3, h3, o3 = strength_rows(body, route, state, AF, lay)
        Gs.append(G3); hs.append(h3); owners += [("strength", l) for l in o3]
    if "inplane" in keep:
        Ab, bb, ob = balance_rows(route, state, AF, lay)
        Aeq.append(Ab); beq.append(bb); eq_owners += [("inplane", n) for n in ob]
    if "outofwall" in keep:
        At, bt, ot, Gt, ht, gt = tipping_rows(body, route, state, AF, lay)
        Aeq.append(At); beq.append(bt); eq_owners += [("outofwall", n) for n in ot]
        Gs.append(Gt); hs.append(ht); owners += [("outofwall", l) for l in gt]
    if "barndoor" in keep:
        Ad, bd, od, Gd, hd, gd = barndoor_rows(body, route, state, AF, lay)
        Aeq.append(Ad); beq.append(bd); eq_owners += [("barndoor", n) for n in od]
        Gs.append(Gd); hs.append(hd); owners += [("barndoor", l) for l in gd]
    A = np.vstack(Aeq) if Aeq else np.zeros((0, lay.size))
    b = np.concatenate(beq) if beq else np.zeros(0)
    return Program(lay, np.vstack(Gs), np.concatenate(hs), owners, A, b, eq_owners)


def _tidy(P: Program, bounds: list, t_star: float, AF: tuple[str, ...]):
    """Second stage: keep t at its optimum and remove force that does nothing.

    The polyhedron has flat faces, so many force splits share the optimal t;
    some of them have two limbs squeezing sideways against each other, or a
    hand twisting, for no reason. Holding t <= t*, minimise
    sum |f_kx| + sum |tau_k| + sum N_k + sum M_k. This never changes t, only
    which of the equally good solutions is reported.
    """
    from scipy.optimize import linprog

    lay, n = P.layout, P.layout.size
    absolute = [lay[("f", l)].start for l in AF] + [sl.start for (name, _), sl in lay.index.items() if name == "tau"]
    k = len(absolute)
    G = np.hstack([P.G, np.zeros((len(P.G), k))])
    extra, rhs = [], []
    for i, col in enumerate(absolute):
        for sign in (1.0, -1.0):
            row = np.zeros(n + k)
            row[col], row[n + i] = sign, -1.0
            extra.append(row)
            rhs.append(0.0)
    row = np.zeros(n + k)
    row[lay[("t", None)].start] = 1.0
    extra.append(row)
    rhs.append(t_star * (1 + 1e-9) + 1e-12)
    G = np.vstack([G, np.array(extra)])
    h = np.concatenate([P.h, rhs])
    A = np.hstack([P.A, np.zeros((len(P.A), k))])
    c = np.zeros(n + k)
    c[n:] = 1.0
    for limb in AF:
        c[lay[(_normal(limb), limb)].start] = 1.0
    res = linprog(c, A_ub=G, b_ub=h, A_eq=A if len(A) else None, b_eq=P.b if len(A) else None,
                  bounds=bounds + [(0.0, None)] * k, method="highs")
    return res.x[:n] if res.status == 0 else None


def solve(body: Body, route: Route, state: State, no_force: str | None = None,
          fixed: dict | None = None, torso=None, tidy: bool = True, upto: str | None = None) -> dict:
    """Minimise t subject to the constraints built so far, with scipy's HiGHS.

    fixed: optional forces held at given values, in the format of `effort`.
    torso: optional body position (x_c, y_c) held fixed.
    tidy: run the second stage (`_tidy`) that removes useless squeezing.
    upto: use only the constraints up to this family (see `assemble`).
    Returns ok, t, the body position, every pushing limb's forces, each
    limb's exact share of its strength and the limiting limb.
    """
    from scipy.optimize import linprog

    P = assemble(body, route, state, no_force, upto)
    lay = P.layout
    bounds = variable_bounds(lay)
    for limb, F in (fixed or {}).items():
        sf = lay[("f", limb)]
        bounds[sf.start], bounds[sf.start + 1] = (F["f"][0],) * 2, (F["f"][1],) * 2
        v = F.get("n", 0.0) if limb in HANDS else F.get("m", 0.0)
        sn = lay[(_normal(limb), limb)]
        bounds[sn.start] = (v, v)
    if torso is not None:
        sx = lay[("x", None)]
        bounds[sx.start], bounds[sx.start + 1] = (torso[0],) * 2, (torso[1],) * 2
    c = np.zeros(lay.size)
    c[lay[("t", None)]] = 1.0
    eq = len(P.A) > 0
    res = linprog(c, A_ub=P.G, b_ub=P.h, A_eq=P.A if eq else None, b_eq=P.b if eq else None, bounds=bounds, method="highs")
    if res.status != 0:
        return {"ok": False, "status": res.status, "message": res.message}
    z = res.x
    AF = limb_sets(state, no_force)[1]
    if tidy:
        z2 = _tidy(P, bounds, float(z[lay[("t", None)]][0]), AF)
        if z2 is not None:
            z = z2
    forces = {}
    for limb in AF:
        f = z[lay[("f", limb)]]
        out = float(z[lay[(_normal(limb), limb)]][0])
        forces[limb] = {"f": (float(f[0]), float(f[1])), ("n" if limb in HANDS else "m"): out}
        if limb in HANDS:
            forces[limb]["tau"] = float(z[lay[("tau", limb)]][0])
    t_exact, share, limiting = effort(body, route, state, forces)
    return {"ok": True, "t": float(z[lay[("t", None)]][0]), "torso": tuple(float(v) for v in z[lay[("x", None)]]),
            "forces": forces, "share": share, "t_exact": t_exact, "limiting": limiting, "z": z, "layout": lay}


def check(body: Body, route: Route, state: State, no_force: str | None = None) -> Result:
    """Check one state: can the climber hold still here, and with what effort t?"""
    validate(body, route, state)
    A, AF = limb_sets(state, no_force)
    early = precheck(body, route, state, no_force)
    if early is not None:
        return early
    if not torso_region(body, route, state, REACH_SIDES):
        return Result(ok=False, reason="reach")
    raise NotImplementedError("flexibility is added in unit 2.7")
