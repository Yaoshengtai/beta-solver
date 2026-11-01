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


# ------------------------------------------------------------ two-stage Dijkstra (section 5)

import heapq  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402

MAX_EXPANSIONS = 200_000
EPS = 1e-6


@dataclass
class Beta:
    status: str                      # "ok", "infeasible" or "limit"
    crux: float | None = None        # B*: the smallest possible largest move cost
    total: float | None = None       # the smallest total cost among betas with that crux
    crux_step: int | None = None     # index of the hardest move
    steps: list = field(default_factory=list)   # (move, MoveResult, cost terms) in order
    start: State | None = None
    fail: dict | None = None         # for "infeasible": where and why the search got stuck
    stats: dict = field(default_factory=dict)
    trace: dict = field(default_factory=dict)   # expansion order of both stages, for the figure


def _path(parent: dict, goal: State) -> list:
    steps, s = [], goal
    while parent.get(s) is not None:
        prev, edge = parent[s]
        steps.append(edge)
        s = prev
    return list(reversed(steps)), s


def bottleneck(g: Graph, max_expansions: int = MAX_EXPANSIONS):
    """Stage 1: the smallest possible largest move cost from any start to any goal.

    A path's label is the largest edge cost on it; states are expanded in
    order of label. The first goal popped has the optimal label B*.
    Returns (B*, goal, parents, order, expansions) with B* None if no goal is reachable.
    """
    label, parent, order, seen = {}, {}, [], set()
    heap, n = [], 0
    for s in g.start_states():
        label[s], parent[s] = 0.0, None
        heapq.heappush(heap, (0.0, n, s)); n += 1
    expanded = 0
    while heap:
        lab, _, s = heapq.heappop(heap)
        if s in seen:
            continue
        seen.add(s)
        order.append((s, lab))
        if g.is_goal(s):
            return lab, s, parent, order, expanded
        expanded += 1
        if expanded > max_expansions:
            return "limit", None, parent, order, expanded
        for move, res, w in g.edges(s):
            t, new = move.new_state, max(lab, w["total"])
            if t not in seen and new < label.get(t, float("inf")) - 1e-12:
                label[t], parent[t] = new, (s, (move, res, w))
                heapq.heappush(heap, (new, n, t)); n += 1
    return None, None, parent, order, expanded


def cheapest(g: Graph, cap: float, max_expansions: int = MAX_EXPANSIONS):
    """Stage 2: ordinary Dijkstra on total cost, using only edges with cost <= cap."""
    dist, parent, order, seen = {}, {}, [], set()
    heap, n = [], 0
    for s in g.start_states():
        dist[s], parent[s] = 0.0, None
        heapq.heappush(heap, (0.0, n, s)); n += 1
    expanded = 0
    while heap:
        d, _, s = heapq.heappop(heap)
        if s in seen:
            continue
        seen.add(s)
        order.append((s, d))
        if g.is_goal(s):
            return d, s, parent, order, expanded
        expanded += 1
        if expanded > max_expansions:
            return "limit", None, parent, order, expanded
        for move, res, w in g.edges(s):
            if w["total"] > cap + EPS:
                continue
            t, nd = move.new_state, d + w["total"]
            if t not in seen and nd < dist.get(t, float("inf")) - 1e-12:
                dist[t], parent[t] = nd, (s, (move, res, w))
                heapq.heappush(heap, (nd, n, t)); n += 1
    return None, None, parent, order, expanded


def _stuck(g: Graph, order: list, parent: dict) -> dict:
    """Where an infeasible search got stuck: the highest state it reached (by its lower hand),
    and the first failing hand move from there toward a higher hold, with the step-3 diagnosis."""
    reached = [s for s, _ in order]
    y = lambda s: min(g.route.hold(s.LH).y, g.route.hold(s.RH).y)
    top = max(reached, key=lambda s: (y(s), max(g.route.hold(s.LH).y, g.route.hold(s.RH).y)))
    steps, _ = _path(parent, top)
    best = None
    for move, res in g.checker.candidates(top):
        if move.limb not in HANDS or move.to is None:
            continue
        h = g.route.hold(move.to)
        if h.y <= g.route.hold(top.hold(move.limb)).y:
            continue
        if res is None:
            f = {"phase": "catch", "reason": "reach", "limb": move.limb, "hold": move.to}
        elif not res.ok:
            f = dict(res.failure)
            f.setdefault("limb", move.limb)
            if f.get("limb") is None:
                f["limb"] = move.limb
            if f.get("hold") is None:
                f["hold"] = move.to
        else:
            continue
        key = (0 if move.to == g.route.finish else 1, h.y)
        if best is None or key < best[0]:
            best = (key, f)
    fail = best[1] if best else {"phase": None, "reason": None, "limb": None, "hold": None}
    return {"step": len(steps) + 1, "state": top.as_dict(), **fail}


def solve(route: Route, climber: Climber, max_expansions: int = MAX_EXPANSIONS) -> Beta:
    """The best beta: smallest crux first, then smallest total cost (section 5)."""
    t0 = time.perf_counter()
    g = Graph(route, climber)
    b_star, goal1, parent1, order1, exp1 = bottleneck(g, max_expansions)
    checks1 = g.checker.solves
    stats = {"expanded_1": exp1, "checks_1": checks1}
    if b_star == "limit":
        return Beta("limit", stats={**stats, "seconds": time.perf_counter() - t0})
    if b_star is None:
        beta = Beta("infeasible", fail=_stuck(g, order1, parent1))
        beta.stats = {**stats, "seconds": time.perf_counter() - t0}
        beta.trace = {"stage1": [(s.as_dict(), lab) for s, lab in order1]}
        return beta
    total, goal2, parent2, order2, exp2 = cheapest(g, b_star, max_expansions)
    steps, start = _path(parent2, goal2)
    costs = [w["total"] for _, _, w in steps]
    crux_step = max(range(len(costs)), key=lambda i: costs[i]) if costs else None
    stats.update({"expanded_2": exp2, "checks_2": g.checker.solves - checks1, "checks": g.checker.solves,
                  "seconds": time.perf_counter() - t0})
    return Beta("ok", crux=b_star, total=total, crux_step=crux_step, steps=steps, start=start, stats=stats,
                trace={"stage1": [(s.as_dict(), lab) for s, lab in order1],
                       "stage2": [(s.as_dict(), d) for s, d in order2]})


def pareto_reference(route: Route, climber: Climber, max_len: int) -> tuple | None:
    """An exhaustive reference for small routes: over every path of at most max_len moves, the
    lexicographically smallest (crux, total). For each state and length it keeps every
    (largest cost, total) pair that no other path beats in both; the lexicographic optimum is
    always among them. Independent of the Dijkstra code apart from the shared edges."""
    g = Graph(route, climber)
    front = {s: {(0.0, 0.0)} for s in g.start_states()}
    best = None

    def offer(s, pairs):
        nonlocal best
        if not g.is_goal(s):
            return
        for c, t in pairs:
            # cruxes within EPS count as equal (as in stage 2); then the smaller total wins
            if best is None or c < best[0] - EPS or (abs(c - best[0]) <= EPS and t < best[1]):
                best = (c, t)

    for s, pairs in front.items():
        offer(s, pairs)
    layer = front
    for _ in range(max_len):
        nxt: dict = {}
        for s, pairs in layer.items():
            for move, res, w in g.edges(s):
                t = move.new_state
                for c, tot in pairs:
                    nxt.setdefault(t, set()).add((max(c, w["total"]), tot + w["total"]))
        layer = {}
        for t, pairs in nxt.items():
            keep = {p for p in pairs if not any(q[0] <= p[0] and q[1] <= p[1] and q != p for q in pairs)}
            layer[t] = keep
            offer(t, keep)
        if not layer:
            break
    return best
