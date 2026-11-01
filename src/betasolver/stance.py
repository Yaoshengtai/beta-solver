"""The state check (step 3 of the plan).

With every limb on a fixed hold, can the climber hold still, and if so how
hard is the most loaded limb working? The answer is a small convex problem:
find a body position and contact forces that satisfy six constraints, with
the smallest effort t.

This module holds the framework (states, unknowns, result type, checks done
before any solving), the six constraints, the second-order cone program that
puts them together (solved with Clarabel), and the check with its diagnosis
of failures.
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
    {"n": 4, "key": "inplane", "name": "In-plane balance", "section": "check-balance",
     "climbing": "No falling, no sliding sideways, no swinging.", "limits": "in-wall forces and body position"},
    {"n": 5, "key": "outofwall", "name": "Out-of-wall balance", "section": "check-balance",
     "climbing": "No tipping backwards, no barn-door.", "limits": "hand pull, foot push, grip torque"},
    {"n": 6, "key": "flex", "name": "Flexibility", "section": "check-reach",
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
    "unusable": "unusable hold",
    "reach": "cannot reach",
    "strength": "not strong enough",
    "balance": "direction or balance fails",
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


def reach_rows(body: Body, route: Route, state: State, sides: int):
    """The reach discs as polygons, for drawing only: G x <= h in the torso position x.

    For every limb k on a hold and every direction u_i of the polygon:
        u_i . (p_k - x - o_k) <= L_k cos(pi / sides)
    which is the inscribed regular polygon of the disc |p_k - x - o_k| <= L_k.
    The state check itself uses the discs (`reach_cones`).
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


def reach_cones(body: Body, route: Route, state: State, lay: Layout, skip: str | None = None):
    """① as second-order cones: (L_k, p_k - o_k - x) in the cone, one block per limb on a hold.

    In the solver's form a cone block is s = b - A z with s_0 >= |s_1..|; here
    A puts +1 on x_c and y_c and b = (L_k, p_k - o_k).
    """
    blocks, owner = [], []
    sx = lay[("x", None)].start
    for limb, (cx, cy) in centres(body, route, state).items():
        if limb == skip:
            continue
        A = np.zeros((3, lay.size))
        A[1, sx], A[2, sx + 1] = 1.0, 1.0
        blocks.append((A, np.array([body.L[limb], cx, cy])))
        owner.append(limb)
    return blocks, owner


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
    """Every torso position from which all limbs reach their holds, as a polygon, for drawing.

    This is the intersection of one disc per limb (centre p_k - o_k, radius
    L_k), each drawn as an inscribed polygon with `sides` sides. An empty
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

# Reading the effort t (section 3.3): upper end of each band and its name.
EFFORT_BANDS = [
    {"upto": 0.5, "key": "easy", "name": "easy"},
    {"upto": 0.8, "key": "hard", "name": "hard work"},
    {"upto": 1.0, "key": "limit", "name": "near the limit"},
    {"upto": None, "key": "fail", "name": "cannot hold: not strong enough"},
]


def strength_limit(body: Body, route: Route, state: State, limb: str) -> float:
    """F_k: the most force limb k can take on its hold. A foot on a hand hold uses the foot limit."""
    kind, _ = route.hold(state.hold(limb)).contact(limb)
    return body.climber.strength(kind.name)


def _normal(limb: str) -> str:
    """The out-of-wall unknown of a limb: the pull-in N for a hand, the push M for a foot."""
    return "N" if limb in HANDS else "M"


def strength_cones(body: Body, route: Route, state: State, AF: tuple[str, ...], lay: Layout):
    """③ as second-order cones: (t F_k, f_k, N_k or M_k) in the cone, one block per pushing limb.

    In the solver's form s = b - A z: A = -(F_k on t, 1 on f_kx, f_ky and the
    out-of-wall unknown), b = 0.
    """
    blocks, owner = [], []
    for limb in AF:
        A = np.zeros((4, lay.size))
        A[0, lay[("t", None)].start] = -strength_limit(body, route, state, limb)
        sf = lay[("f", limb)]
        A[1, sf.start], A[2, sf.start + 1] = -1.0, -1.0
        A[3, lay[(_normal(limb), limb)].start] = -1.0
        blocks.append((A, np.zeros(4)))
        owner.append(limb)
    return blocks, owner


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


# ------------------------------------------------------------ ⑥ flexibility

def flex_rows(body: Body, route: Route, state: State, lay: Layout, skip: str | None = None):
    """⑥ as linear inequalities on the body position: no foot higher than the hip plus flex H.

        p_ky <= y_c - 0.03 H + flex H,   i.e.   -y_c <= -p_ky + body.foot_limit((0, 0))
    for every foot on a hold, including a foot that only rests there (no_force):
    it still has to get there.
    """
    rows, rhs, owner = [], [], []
    top = body.foot_limit((0.0, 0.0))  # highest foot relative to y_c
    for limb in state.on():
        if limb not in FEET or limb == skip:
            continue
        row = np.zeros(lay.size)
        row[lay[("x", None)].start + 1] = -1.0
        rows.append(row)
        rhs.append(top - route.hold(state.hold(limb)).y)
        owner.append(limb)
    G = np.array(rows, dtype=float).reshape(len(rows), lay.size)
    return G, np.array(rhs, dtype=float), owner


# ------------------------------------------------------- sign of the unknowns

def sign_rows(lay: Layout):
    """N, M and t are never negative: -z_i <= 0. The body position, forces and grip torques are free."""
    cols = [i for (name, _), sl in lay.index.items() if name in ("N", "M", "t") for i in range(sl.start, sl.stop)]
    G = np.zeros((len(cols), lay.size))
    for r, i in enumerate(cols):
        G[r, i] = -1.0
    return G, np.zeros(len(cols))


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


# ------------------------------------------------------- the cone program

@dataclass
class Program:
    """The state check as a second-order cone program over z (and any extra columns):

        minimise c . z   subject to   A z = b,   G z <= h,   b_k - A_k z in a second-order cone
    """

    layout: Layout
    n: int
    A: np.ndarray
    b: np.ndarray
    eq_owners: list   # (constraint, name) of each row of A
    G: np.ndarray
    h: np.ndarray
    owners: list      # (constraint, limb) of each row of G
    cones: list       # (A_k, b_k) blocks
    cone_owners: list  # (constraint, limb) of each block

    def add_columns(self, k: int) -> "Program":
        pad = lambda M: np.hstack([M, np.zeros((len(M), k))])
        return Program(self.layout, self.n + k, pad(self.A), self.b, list(self.eq_owners), pad(self.G), self.h,
                       list(self.owners), [(pad(Ak), bk) for Ak, bk in self.cones], list(self.cone_owners))

    def add_eq(self, A, b, owners) -> None:
        self.A = np.vstack([self.A, np.atleast_2d(A)])
        self.b = np.concatenate([self.b, np.atleast_1d(b)])
        self.eq_owners += owners

    def add_ineq(self, G, h, owners) -> None:
        self.G = np.vstack([self.G, np.atleast_2d(G)])
        self.h = np.concatenate([self.h, np.atleast_1d(h)])
        self.owners += owners


def run(P: Program, c: np.ndarray):
    """Solve with Clarabel. Returns the solution z, or None if the program has no solution."""
    import clarabel
    from scipy import sparse

    blocks, rhs, cones = [], [], []
    if len(P.A):
        blocks.append(P.A); rhs.append(P.b); cones.append(clarabel.ZeroConeT(len(P.A)))
    if len(P.G):
        blocks.append(P.G); rhs.append(P.h); cones.append(clarabel.NonnegativeConeT(len(P.G)))
    for Ak, bk in P.cones:
        blocks.append(Ak); rhs.append(bk); cones.append(clarabel.SecondOrderConeT(len(Ak)))
    A = sparse.csc_matrix(np.vstack(blocks))
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    solver = clarabel.DefaultSolver(sparse.csc_matrix((P.n, P.n)), np.asarray(c, dtype=float), A,
                                    np.concatenate(rhs), cones, settings)
    sol = solver.solve()
    if str(sol.status) not in ("Solved", "AlmostSolved"):
        return None
    return np.array(sol.x)


def satisfies(P: Program, z: np.ndarray, tol: float = 1e-6) -> bool:
    """Does z meet every constraint of the program (within tol)?"""
    if len(P.A) and np.max(np.abs(P.A @ z - P.b)) > tol:
        return False
    if len(P.G) and np.max(P.G @ z - P.h) > tol:
        return False
    for Ak, bk in P.cones:
        sv = bk - Ak @ z
        if sv[0] < np.linalg.norm(sv[1:]) - tol:
            return False
    return True


def vector(lay: Layout, torso, forces: dict, t: float) -> np.ndarray:
    """Pack a body position, forces (in the format of `effort`, tau optional) and t into z."""
    z = np.zeros(lay.size)
    z[lay[("x", None)]] = torso
    z[lay[("t", None)]] = t
    for limb, F in forces.items():
        if ("f", limb) not in lay.index:
            continue
        z[lay[("f", limb)]] = F["f"]
        z[lay[(_normal(limb), limb)]] = F.get("n", 0.0) if limb in HANDS else F.get("m", 0.0)
        if limb in HANDS:
            z[lay[("tau", limb)]] = F.get("tau", 0.0)
    return z


# The constraint families in the order they are built; `upto` stops after one of them.
BUILT = ("reach", "direction", "strength", "inplane", "outofwall", "barndoor", "flex")


def assemble(body: Body, route: Route, state: State, no_force: str | None = None, upto: str | None = None,
             off: str | None = None) -> Program:
    """The state check as one cone program.

    upto: stop after this constraint family (one of BUILT), so a figure can
    show the model as it stood in its own section. None means all of them.
    off: a limb in the air (it must also be the no_force limb): it neither
    pushes nor has to reach anything. This is the three-limb problem of a
    move's travel phase (section 4.3).
    """
    if off is not None and off != no_force:
        raise ValueError("a limb in the air carries no force: pass the same limb as no_force")
    keep = BUILT if upto is None else BUILT[: BUILT.index(upto) + 1]
    A_, AF = limb_sets(state, no_force)
    lay = layout(AF)
    n = lay.size
    Gs, P = sign_rows(lay)
    P = Program(lay, n, np.zeros((0, n)), np.zeros(0), [], Gs, P, [("sign", None)] * len(Gs), [], [])
    if "reach" in keep:
        blocks, own = reach_cones(body, route, state, lay, skip=off)
        P.cones += blocks; P.cone_owners += [("reach", l) for l in own]
    if "direction" in keep:
        G, h, own = direction_rows(route, state, AF, lay)
        P.add_ineq(G, h, [("direction", l) for l in own])
    if "strength" in keep:
        blocks, own = strength_cones(body, route, state, AF, lay)
        P.cones += blocks; P.cone_owners += [("strength", l) for l in own]
    if "inplane" in keep:
        A, b, own = balance_rows(route, state, AF, lay)
        P.add_eq(A, b, [("inplane", o) for o in own])
    if "outofwall" in keep:
        A, b, own, G, h, gown = tipping_rows(body, route, state, AF, lay)
        P.add_eq(A, b, [("outofwall", o) for o in own])
        P.add_ineq(G, h, [("outofwall", l) for l in gown])
    if "barndoor" in keep:
        A, b, own, G, h, gown = barndoor_rows(body, route, state, AF, lay)
        P.add_eq(A, b, [("barndoor", o) for o in own])
        P.add_ineq(G, h, [("barndoor", l) for l in gown])
    if "flex" in keep:
        G, h, own = flex_rows(body, route, state, lay, skip=off)
        P.add_ineq(G, h, [("flex", l) for l in own])
    return P


def _fix(P: Program, cols, values, name) -> None:
    for i, v in zip(cols, values):
        row = np.zeros(P.n)
        row[i] = 1.0
        P.add_eq(row, v, [(name, None)])


def _tidy(P: Program, t_star: float, AF: tuple[str, ...]):
    """Second stage: keep t at its optimum and remove force that does nothing.

    Only the most loaded limb is pinned down by the optimal t; the others can
    often carry several force splits that are all as good, some of them with
    two limbs squeezing sideways against each other or a hand twisting for no
    reason. Holding t <= t*, minimise sum |f_kx| + sum |tau_k| + sum N_k + sum M_k.
    This never changes t, only which of the equally good solutions is reported.
    """
    lay, n = P.layout, P.n
    absolute = [lay[("f", l)].start for l in AF] + [sl.start for (name, _), sl in lay.index.items() if name == "tau"]
    k = len(absolute)
    Q = P.add_columns(k)
    for i, col in enumerate(absolute):
        for sign in (1.0, -1.0):
            row = np.zeros(n + k)
            row[col], row[n + i] = sign, -1.0
            Q.add_ineq(row, 0.0, [("tidy", None)])
        row = np.zeros(n + k)
        row[n + i] = -1.0
        Q.add_ineq(row, 0.0, [("tidy", None)])
    row = np.zeros(n + k)
    row[lay[("t", None)].start] = 1.0
    Q.add_ineq(row, t_star * (1 + 1e-7) + 1e-10, [("tidy", None)])
    c = np.zeros(n + k)
    c[n:] = 1.0
    for limb in AF:
        c[lay[(_normal(limb), limb)].start] = 1.0
    z = run(Q, c)
    return None if z is None else z[:n]


def solve(body: Body, route: Route, state: State, no_force: str | None = None,
          fixed: dict | None = None, torso=None, tidy: bool = True, upto: str | None = None,
          off: str | None = None) -> dict:
    """Minimise t subject to the constraints, with Clarabel.

    fixed: optional forces held at given values, in the format of `effort`.
    torso: optional body position (x_c, y_c) held fixed (to within 1e-7 m).
    tidy: run the second stage (`_tidy`) that removes useless squeezing.
    upto: use only the constraints up to this family (see `assemble`).
    off: a limb in the air, neither pushing nor reaching (see `assemble`).
    Returns ok, t, the body position, every pushing limb's forces, each
    limb's share of its strength and the limiting limb.
    """
    P = assemble(body, route, state, no_force, upto, off)
    lay = P.layout
    for limb, F in (fixed or {}).items():
        sf, sn = lay[("f", limb)], lay[(_normal(limb), limb)]
        v = F.get("n", 0.0) if limb in HANDS else F.get("m", 0.0)
        _fix(P, [sf.start, sf.start + 1, sn.start], [F["f"][0], F["f"][1], v], "fixed")
    if torso is not None:
        # held within 1e-7 m rather than exactly: a position on the edge of the
        # feasible region would otherwise be numerically degenerate
        sx = lay[("x", None)].start
        for i, v in enumerate(torso):
            for sign in (1.0, -1.0):
                row = np.zeros(P.n)
                row[sx + i] = sign
                P.add_ineq(row, sign * v + 1e-7, [("torso", None)])
    c = np.zeros(P.n)
    c[lay[("t", None)].start] = 1.0
    z = run(P, c)
    if z is None:
        return {"ok": False}
    AF = limb_sets(state, no_force)[1]
    t_star = float(z[lay[("t", None)].start])
    if tidy:
        z2 = _tidy(P, t_star, AF)
        if z2 is not None:
            z = z2
    forces = {}
    for limb in AF:
        f = z[lay[("f", limb)]]
        out = max(0.0, float(z[lay[(_normal(limb), limb)].start]))
        forces[limb] = {"f": (float(f[0]), float(f[1])), ("n" if limb in HANDS else "m"): out}
        if limb in HANDS:
            forces[limb]["tau"] = float(z[lay[("tau", limb)].start])
    _, share, limiting = effort(body, route, state, forces)
    return {"ok": True, "t": t_star, "torso": tuple(float(v) for v in z[lay[("x", None)]]),
            "forces": forces, "share": share, "limiting": limiting, "z": z, "layout": lay}


# --------------------------------------------------------------------- check

def _geometry_ok(body: Body, route: Route, state: State, skip: str | None = None,
                 flex: bool = True, skip_flex: str | None = None) -> bool:
    """Can some body position meet reach (1) and flexibility (6) alone?

    skip drops one limb's reach (and, for a foot, its flexibility row);
    flex=False drops every flexibility row; skip_flex drops only that foot's.
    """
    lay = layout(())
    pts = targets(body, route, state)
    top = body.foot_limit((0.0, 0.0))
    blocks, _ = reach_cones(body, route, state, lay, skip=skip)
    G, h = [], []
    for l in pts:
        if l in FEET and flex and l != skip and l != skip_flex:
            row = np.zeros(lay.size)
            row[lay[("x", None)].start + 1] = -1.0
            G.append(row)
            h.append(top - pts[l][1])
    P = Program(lay, lay.size, np.zeros((0, lay.size)), np.zeros(0), [],
                np.array(G).reshape(len(G), lay.size), np.array(h), [], blocks, [])
    return run(P, np.zeros(lay.size)) is not None


def diagnose(body: Body, route: Route, state: State) -> tuple[str, str | None]:
    """Why does a state have no solution? (section 3.7)

    Solve again with only reach and flexibility. If that has a solution, the
    holds can be reached, so the directions or the balance do not work.
    Otherwise the reason is reach, and the limb is found like this: if the
    holds are reachable once flexibility is ignored, it is the foot whose
    flexibility limit is to blame (the highest such foot); if not, it is a
    limb whose hold, if dropped, makes the rest reachable, the one farthest
    out of reach from where the other holds put the body.
    """
    if _geometry_ok(body, route, state):
        return "balance", None
    pts = targets(body, route, state)
    if _geometry_ok(body, route, state, flex=False):
        feet = [l for l in pts if l in FEET and _geometry_ok(body, route, state, skip_flex=l)]
        return ("reach", max(feet, key=lambda l: pts[l][1])) if feet else ("reach", None)
    cands = [l for l in pts if _geometry_ok(body, route, state, skip=l)]
    if not cands:
        return "reach", None

    def gap(l):
        others = [o for o in pts if o != l]
        cx = sum(pts[o][0] - body.offsets[o][0] for o in others) / len(others)
        cy = sum(pts[o][1] - body.offsets[o][1] for o in others) / len(others)
        ax, ay = cx + body.offsets[l][0], cy + body.offsets[l][1]
        return math.hypot(pts[l][0] - ax, pts[l][1] - ay) - body.L[l]
    return "reach", max(cands, key=gap)


def check(body: Body, route: Route, state: State, no_force: str | None = None) -> Result:
    """Check one state: can the climber hold still here, and with what effort t?

    Cheap prechecks first, then the cone program. With a solution, the state
    holds if t <= 1; above 1 the reason is strength and the limb is the most
    loaded one. With no solution, `diagnose` tells reach from balance.
    """
    validate(body, route, state)
    early = precheck(body, route, state, no_force)
    if early is not None:
        return early
    res = solve(body, route, state, no_force)
    if not res["ok"]:
        reason, limb = diagnose(body, route, state)
        return Result(ok=False, reason=reason, limb=limb, hold=state.hold(limb) if limb else None)
    t = res["t"]
    forces = {l: {k: (list(v) if isinstance(v, tuple) else v) for k, v in F.items()} for l, F in res["forces"].items()}
    lim = res["limiting"]
    if t > 1.0:
        return Result(ok=False, t=t, torso=res["torso"], forces=forces, limiting=lim,
                      reason="strength", limb=lim, hold=state.hold(lim))
    return Result(ok=True, t=t, torso=res["torso"], forces=forces, limiting=lim)
