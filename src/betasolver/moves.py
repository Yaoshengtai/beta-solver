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
from .stance import LIMB_NAMES, Result, State, check

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
