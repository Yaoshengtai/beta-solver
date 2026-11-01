"""One move (step 4 of the plan).

A move changes the hold of one limb while the body slides continuously.
It is decided by two state checks:

    release check  = check(sigma,  no_force=k): the moving limb still touches
                     its old hold but carries nothing.
    catch check    = check(sigma', no_force=k): it has just touched the new
                     hold and carries nothing yet.

The move can be done statically when both have t <= 1, and its difficulty is
the larger of the two. Two more checks, of sigma and sigma' with every limb
pushing, give the start and end body positions x0 and x3 for the animation;
they never change whether the move works (section 4.3).
"""

from __future__ import annotations

from dataclasses import dataclass

from .body import Body
from .model import FEET, HANDS, Route
from .stance import LIMB_NAMES, Result, State, assemble, check, satisfies, solve, vector

KINDS = {
    "swap": "a limb goes from one hold to another",
    "cut": "a foot comes off its hold and hangs",
    "place": "a hanging foot goes onto a hold",
    "match": "a limb goes to a hold another limb is already on",
}


@dataclass(frozen=True)
class Move:
    state: State
    limb: str
    to: str | None  # the new hold, or None when a foot comes off

    @property
    def new_state(self) -> State:
        d = self.state.as_dict()
        d[self.limb] = self.to
        return State(**d)

    @property
    def frm(self) -> str | None:
        return self.state.hold(self.limb)

    @property
    def kind(self) -> str:
        if self.frm is None:
            return "place"
        if self.to is None:
            return "cut"
        others = [self.state.hold(l) for l in self.state.on() if l != self.limb]
        return "match" if self.to in others else "swap"

    def __post_init__(self):
        if self.limb in HANDS and self.to is None:
            raise ValueError("a hand cannot hang")
        if self.frm == self.to:
            raise ValueError("a move must change the hold")
        if self.frm is None and self.limb not in FEET:
            raise ValueError("only a foot can be placed from hanging")

    @property
    def swing(self) -> bool:
        """Does the move take off the last foot, or put on the first?

        With no foot pushing, the body hangs flat against the wall (d = 0);
        with one, it stands a distance d off it. So the moment the last foot
        unloads, or the first one loads, the body swings in or out: that part
        of the move is not static, and the straight-line argument of section
        4.2 does not cover it.
        """
        if self.limb not in FEET:
            return False
        others = [l for l in FEET if l != self.limb and self.state.hold(l) is not None]
        return not others

    def describe(self, route: Route) -> str:
        who = LIMB_NAMES[self.limb].capitalize()
        if self.kind == "cut":
            return f"{who} off {self.frm}"
        h = route.hold(self.to)
        return f"{who} → {self.to} ({h.type})"


@dataclass(frozen=True)
class MoveResult:
    move: Move
    start: Result
    release: Result | None  # None for a place: no old hold to let go of
    catch: Result | None    # None for a cut: no new hold to catch
    end: Result

    @property
    def t_release(self) -> float | None:
        return None if self.release is None else self.release.t

    @property
    def t_catch(self) -> float | None:
        return None if self.catch is None else self.catch.t

    @property
    def ok(self) -> bool:
        return all(r.ok for r in (self.release, self.catch) if r is not None)

    @property
    def difficulty(self) -> float | None:
        """max(t_release, t_catch) when the move works (section 4.3)."""
        if not self.ok:
            return None
        return max(t for t in (self.t_release, self.t_catch) if t is not None)

    @property
    def failure(self) -> dict | None:
        """Which check fails first and why, in the format of the exported `fail` field."""
        for phase, r in (("release", self.release), ("catch", self.catch)):
            if r is not None and not r.ok:
                return {"phase": phase, "reason": r.reason, "limb": r.limb, "hold": r.hold}
        return None


def evaluate(body: Body, route: Route, move: Move) -> MoveResult:
    """Run the checks of one move: start, release, catch and end (section 4.1)."""
    s0, s1 = move.state, move.new_state
    start = check(body, route, s0)
    release = check(body, route, s0, no_force=move.limb) if move.frm is not None else None
    catch = check(body, route, s1, no_force=move.limb) if move.to is not None else None
    end = check(body, route, s1)
    return MoveResult(move, start, release, catch, end)


# ------------------------------------------------------------ convexity, checked numerically (section 4.2)

def segments(move: Move, res: MoveResult):
    """The three straight slides of a move, each inside one convex region (section 4.3):

        1 unload   x0 -> x1   in state sigma, all limbs pushing
        3 travel   x1 -> x2   in the three-limb problem, the moving limb in the air
        5 load     x2 -> x3   in state sigma', all limbs pushing

    Each entry: (phase, state, no_force, off, start Result, end Result).
    """
    k, s0, s1 = move.limb, move.state, move.new_state
    out = []
    if res.release is not None:
        out.append((1, s0, None, None, res.start, res.release))
    if res.release is not None and res.catch is not None:
        out.append((3, s0, k, k, res.release, res.catch))
    if res.catch is not None:
        out.append((5, s1, None, None, res.catch, res.end))
    return out


def path_check(body: Body, route: Route, move: Move, res: MoveResult | None = None, n: int = 20) -> list:
    """Sample each slide of a working move at n + 1 points and check it two ways.

    t_min: the smallest effort with the body held at that point (a fresh solve);
    t_mix, ok_mix: the endpoint solutions blended linearly (body, forces and t
    together), and whether the blend meets every constraint of that phase.
    Both should stay at or below the move's difficulty.
    """
    res = res or evaluate(body, route, move)
    if not res.ok:
        raise ValueError("only a move that works has a path")
    if move.swing:
        raise ValueError("a swing move (last foot off or first foot on) is not a straight slide")
    out = []
    for phase, state, nf, off, a, b in segments(move, res):
        P = assemble(body, route, state, nf, off=off)
        za = vector(P.layout, a.torso, a.forces, a.t)
        zb = vector(P.layout, b.torso, b.forces, b.t)
        for i in range(n + 1):
            u = i / n
            z = (1 - u) * za + u * zb
            x = (float(z[P.layout[("x", None)]][0]), float(z[P.layout[("x", None)]][1]))
            fixed = solve(body, route, state, nf, torso=x, tidy=False, off=off)
            out.append({"phase": phase, "u": u, "torso": x,
                        "t_min": fixed["t"] if fixed["ok"] else None,
                        "t_mix": float(z[P.layout[("t", None)].start]), "ok_mix": satisfies(P, z, tol=1e-5)})
    return out


# ------------------------------------------------------------ why two checks are enough (section 4.3)

def region(body: Body, route: Route, state: State, limb: str, inside, rays: int = 72, reach: float = 1.5,
           tmax: float = 1.0, iters: int = 18) -> list:
    """S3: every body position where the other limbs hold the body with t <= tmax, `limb` in the air.

    It is convex (section 4.2), so it is star-shaped about any point inside
    it; its boundary is found by bisection along `rays` rays from `inside`.
    Returns the boundary as a polygon, or [] if `inside` is not inside.
    """
    import math

    def holds(x):
        r = solve(body, route, state, no_force=limb, off=limb, torso=x, tidy=False)
        return r["ok"] and r["t"] <= tmax + 1e-9

    if not holds(inside):
        return []
    poly = []
    for i in range(rays):
        a = 2 * math.pi * i / rays
        lo, hi = 0.0, reach
        for _ in range(iters):
            mid = (lo + hi) / 2
            if holds((inside[0] + mid * math.cos(a), inside[1] + mid * math.sin(a))):
                lo = mid
            else:
                hi = mid
        poly.append((inside[0] + lo * math.cos(a), inside[1] + lo * math.sin(a)))
    return poly


# ------------------------------------------------------------ cost, move types and candidates (section 4.4)

# Style costs in the units of t. w_cross comes from the climber profile.
COST = {
    "c0": 0.03,        # every move: prefer fewer moves
    "lam_d": 0.05,     # per metre the limb travels: prefer shorter moves
    "c_cut": 0.10,     # taking a foot off
    "cross_gap": 0.10,  # hands count as crossed when the left is this far right of the right (m)
}


def crossed(route: Route, state: State, gap: float = COST["cross_gap"]) -> bool:
    """Are the hands crossed: the left hand more than `gap` to the right of the right hand?"""
    return route.hold(state.LH).x - route.hold(state.RH).x > gap


def distance(route: Route, move: Move) -> float:
    """How far the limb travels; zero when a foot comes off or goes on from hanging."""
    if move.frm is None or move.to is None:
        return 0.0
    a, b = route.hold(move.frm), route.hold(move.to)
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5


def cost(route: Route, w_cross: float, res: MoveResult) -> dict | None:
    """The cost of a move that works, term by term (section 4.4); None for a move that fails.

        w = max(t_release, t_catch) + c0 + lam_d |p_B - p_A| + w_cross [enters a crossed state] + c_cut [cut]
    """
    if not res.ok:
        return None
    m = res.move
    enters = crossed(route, m.new_state) and not crossed(route, m.state)
    terms = {
        "difficulty": res.difficulty,
        "step": COST["c0"],
        "distance": COST["lam_d"] * distance(route, m),
        "cross": w_cross if enters else 0.0,
        "cut": COST["c_cut"] if m.kind == "cut" else 0.0,
    }
    terms["total"] = sum(terms.values())
    return terms


def targets_for(route: Route, state: State, limb: str) -> list:
    """Every hold the limb may move to under the rules: its own use, hands never hang, a hand and a
    foot share a hold only if feet may use it (`Hold.usable_by` and the route's flag)."""
    out = []
    for h in route.holds:
        if h.id != state.hold(limb) and h.usable_by(limb, route.allow_feet_on_handholds):
            out.append(h.id)
    if limb in FEET and state.hold(limb) is not None:
        out.append(None)  # a foot can always come off
    return out


class Checker:
    """State checks for one climber on one route, cached by (state, no_force).

    Each candidate move costs at most one new check: the catch check (the
    one that fails most often) first; the release check depends only on the
    state and the limb, so it is shared by every target of that limb.
    """

    def __init__(self, body: Body, route: Route):
        self.body, self.route = body, route
        self.cache: dict = {}
        self.solves = 0

    def check(self, state: State, no_force: str | None = None) -> Result:
        key = (state, no_force)
        if key not in self.cache:
            self.solves += 1
            self.cache[key] = check(self.body, self.route, state, no_force)
        return self.cache[key]

    def possible(self, move: Move) -> bool:
        """Cheap geometric filter before any full check: the new state passes the pairwise
        distance test, and some body position meets reach and flexibility for all its holds."""
        from .stance import _geometry_ok, precheck

        s1 = move.new_state
        early = precheck(self.body, self.route, s1)
        if early is not None and early.reason == "reach":
            return False
        return _geometry_ok(self.body, self.route, s1)

    def evaluate(self, move: Move) -> MoveResult:
        s0, s1, k = move.state, move.new_state, move.limb
        catch = self.check(s1, k) if move.to is not None else None
        if catch is not None and not catch.ok:
            return MoveResult(move, self.check(s0), None if move.frm is None else self.check(s0, k), catch, self.check(s1))
        release = self.check(s0, k) if move.frm is not None else None
        return MoveResult(move, self.check(s0), release, catch, self.check(s1))

    def candidates(self, state: State) -> list:
        """Every move from `state` under the rules, each tagged 'pruned' or with its MoveResult."""
        out = []
        for limb in state.on() + tuple(l for l in FEET if state.hold(l) is None):
            for to in targets_for(self.route, state, limb):
                move = Move(state, limb, to)
                if to is not None and not self.possible(move):
                    out.append((move, None))
                else:
                    out.append((move, self.evaluate(move)))
        return out
