"""Finding the best beta (step 5 of the plan).

This module holds the state graph: states are the nodes, moves that work
are the edges, and edges are built lazily, only when a search reaches their
start state. Every state check goes through one cached `moves.Checker`.
The two-stage Dijkstra search is added in unit 4.2.
"""

from __future__ import annotations

import time
from collections import deque
from itertools import product

from .body import make_body
from .model import FEET, HANDS, Climber, Route
from .moves import Checker, cost
from .stance import State, _geometry_ok, precheck


class Graph:
    """The state graph of one climber on one route."""

    def __init__(self, route: Route, climber: Climber):
        self.route, self.climber = route, climber
        self.body = make_body(climber)
        self.checker = Checker(self.body, route)
        self._edges: dict = {}

    def holds(self, state: State) -> bool:
        """Can the climber hold this state at all (t <= 1)?"""
        return self.checker.check(state).ok

    def start_states(self) -> list:
        """Both hands on the start holds; the feet in every combination that holds, hanging included."""
        lh, rh = self.route.start_hands
        feet = foot_options(self.route)
        out = []
        for lf, rf in product(feet, feet):
            s = State(lh, rh, lf, rf)
            if self.holds(s):
                out.append(s)
        return out

    def is_goal(self, state: State) -> bool:
        return state.LH == self.route.finish and state.RH == self.route.finish and self.holds(state)

    def edges(self, state: State) -> list:
        """The moves that work from `state`, as (move, result, cost terms); computed once and kept."""
        if state not in self._edges:
            out = []
            for move, res in self.checker.candidates(state):
                if res is not None and res.ok:
                    out.append((move, res, cost(self.route, self.climber.w_cross, res)))
            self._edges[state] = out
        return self._edges[state]


def foot_options(route: Route) -> list:
    """Where a foot can be: any hold it may use, or hanging."""
    return [h.id for h in route.holds if h.usable_by("LF", route.allow_feet_on_handholds)] + [None]


def hand_options(route: Route) -> list:
    return [h.id for h in route.holds if h.usable_by("LH")]


def all_states(route: Route):
    """Every combination of holds the four limbs could be on."""
    hands, feet = hand_options(route), foot_options(route)
    for lh, rh, lf, rf in product(hands, hands, feet, feet):
        yield State(lh, rh, lf, rf)


def graph_stats(route: Route, climber: Climber) -> dict:
    """How the state graph shrinks, stage by stage, and what exploring all of it costs.

    states: every combination; after the prechecks (unusable holds, pairs of
    holds too far apart); after reach and flexibility alone; that can be held;
    reachable from a start state by moves that work; and of those, the ones
    from which the top can still be reached.
    """
    t0 = time.perf_counter()
    g = Graph(route, climber)
    body = g.body
    combos = list(all_states(route))
    pre = [s for s in combos if precheck(body, route, s) is None]
    geo = [s for s in pre if _geometry_ok(body, route, s)]
    held = [s for s in geo if g.holds(s)]
    t1 = time.perf_counter()
    # explore every state reachable from the starts by moves that work
    starts = g.start_states()
    seen, queue, parents = set(starts), deque(starts), {}
    n_cand = n_pruned = n_fail = n_ok = 0
    while queue:
        s = queue.popleft()
        for move, res in g.checker.candidates(s):
            n_cand += 1
            if res is None:
                n_pruned += 1
            elif not res.ok:
                n_fail += 1
            else:
                n_ok += 1
                t = move.new_state
                parents.setdefault(t, set()).add(s)
                if t not in seen:
                    seen.add(t)
                    queue.append(t)
    goals = [s for s in seen if g.is_goal(s)]
    useful, back = set(goals), deque(goals)
    while back:
        t = back.popleft()
        for s in parents.get(t, ()):
            if s not in useful:
                useful.add(s)
                back.append(s)
    t2 = time.perf_counter()
    return {
        "states": {"all": len(combos), "precheck": len(pre), "geometry": len(geo), "held": len(held),
                   "reachable": len(seen), "useful": len(useful)},
        "starts": len(starts), "goals": len(goals),
        "moves": {"candidates": n_cand, "pruned": n_pruned, "fail": n_fail, "ok": n_ok},
        "checks": g.checker.solves,
        "seconds": {"states": t1 - t0, "explore": t2 - t1},
    }
