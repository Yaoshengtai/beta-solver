"""Cross-checking the state check (unit 2.9).

Random states on random holds, each solved twice: by `stance.solve`, which
builds the cone program by hand and calls Clarabel, and by
`reference.solve_reference`, the same problem written directly in cvxpy.
They should agree on every verdict and on t*.
"""

from __future__ import annotations

import time
from functools import lru_cache

import numpy as np

from .body import make_body
from .model import PROFILES, VARIANTS, Hold, Route
from .reference import solve_reference
from .stance import State, precheck, solve

HAND_TYPES = ("jug", "crimp", "sloper", "pinch", "sidepull", "undercling")
FOOT_TYPES = ("foot_big", "foot_chip")
CLIMBERS = ("A", "B", "C", "A_stiff")


def random_case(rng: np.random.Generator):
    """One random climber, route and state (with an occasional resting limb)."""
    key = CLIMBERS[rng.integers(len(CLIMBERS))]
    climber = PROFILES.get(key) or VARIANTS[key]
    holds = []

    def hand_hold(hid, x0, x1):
        t = HAND_TYPES[rng.integers(len(HAND_TYPES))]
        d = float(rng.choice([0.0, 180.0])) if t == "sidepull" else None
        holds.append(Hold(hid, round(float(rng.uniform(x0, x1)), 3), round(float(rng.uniform(1.7, 2.5)), 3), t, dir=d))
        return hid

    lh = hand_hold("H1", 1.0, 1.6)
    rh = lh if rng.random() < 0.1 else hand_hold("H2", 1.4, 2.0)
    feet = {}
    n_feet = int(rng.choice([0, 1, 2], p=[0.2, 0.3, 0.5]))
    for i, limb in enumerate(rng.permutation(["LF", "RF"])[:n_feet]):
        hid = f"F{i + 1}"
        t = FOOT_TYPES[rng.integers(2)]
        holds.append(Hold(hid, round(float(rng.uniform(1.0, 2.0)), 3), round(float(rng.uniform(0.5, 1.5)), 3), t))
        feet[str(limb)] = hid
    holds.append(Hold("T", 1.5, 3.6, "jug"))
    route = Route("random", "random", "", holds, (lh, rh), "T")
    state = State(lh, rh, feet.get("LF"), feet.get("RF"))
    no_force = None
    if rng.random() < 0.2:
        no_force = str(rng.choice(state.on()))
    return climber, route, state, no_force


@lru_cache(maxsize=4)
def compare(n: int = 300, seed: int = 7) -> tuple:
    """Check n random states (that pass the prechecks) both ways.

    Each record: climber key, ok and t from the solver, ok and t from the
    cvxpy reference, and the solver's time in milliseconds (one solve with
    the second stage, as the site uses it, and one without).
    """
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        climber, route, state, no_force = random_case(rng)
        body = make_body(climber)
        if precheck(body, route, state, no_force) is not None:
            continue
        t0 = time.perf_counter()
        ours = solve(body, route, state, no_force, tidy=False)
        t1 = time.perf_counter()
        solve(body, route, state, no_force)
        t2 = time.perf_counter()
        ref = solve_reference(body, route, state, no_force)
        t3 = time.perf_counter()
        out.append({"climber": climber.key, "ok": ours["ok"], "t": ours["t"] if ours["ok"] else None,
                    "ref_ok": ref["ok"], "ref_t": ref["t"] if ref["ok"] else None,
                    "ms": 1000 * (t1 - t0), "ms_tidy": 1000 * (t2 - t1), "ref_ms": 1000 * (t3 - t2)})
    return tuple(out)


def summary(records) -> dict:
    both = [r for r in records if r["ok"] and r["ref_ok"]]
    diffs = [abs(r["t"] - r["ref_t"]) for r in both]
    med = lambda k: float(np.median([r[k] for r in records]))
    return {
        "n": len(records),
        "both": len(both),
        "neither": sum(1 for r in records if not r["ok"] and not r["ref_ok"]),
        "disagree": sum(1 for r in records if r["ok"] != r["ref_ok"]),
        "max_diff": max(diffs) if diffs else 0.0,
        "held": sum(1 for r in both if r["t"] <= 1.0),
        "ms": med("ms"), "ms_tidy": med("ms_tidy"), "ref_ms": med("ref_ms"),
    }



# ------------------------------------------------------------ moves along straight lines (section 4.2)

def random_move(rng: np.random.Generator):
    """A random state (as in random_case, without a resting limb) and a move of one limb to a new hold nearby."""
    from .moves import Move

    climber, route, state, _ = random_case(rng)
    limb = str(rng.choice(state.on()))
    old = route.hold(state.hold(limb))
    t = ("foot_big" if limb in ("LF", "RF") else HAND_TYPES[rng.integers(len(HAND_TYPES))])
    d = float(rng.choice([0.0, 180.0])) if t == "sidepull" else None
    new = Hold("N", round(min(2.9, max(0.1, old.x + float(rng.uniform(-0.35, 0.35)))), 3),
               round(min(4.1, max(0.1, old.y + float(rng.uniform(-0.15, 0.45)))), 3), t, dir=d)
    route = Route(route.id, route.title, route.lesson, list(route.holds) + [new], route.start_hands, route.finish)
    return climber, route, Move(state, limb, "N")


@lru_cache(maxsize=4)
def path_sweep(n_moves: int = 100, seed: int = 11, samples: int = 10) -> dict:
    """Check the straight-line slides of n_moves random moves that work.

    Swing moves (last foot off or first foot on) are counted and left out:
    the body swings in or out there, which is not a straight slide.
    """
    from .moves import evaluate, path_check

    rng = np.random.default_rng(seed)
    done, points, worst_min, worst_mix, all_ok, tried, swings = 0, 0, -np.inf, -np.inf, True, 0, 0
    while done < n_moves:
        tried += 1
        climber, route, move = random_move(rng)
        body = make_body(climber)
        if precheck(body, route, move.state) is not None or precheck(body, route, move.new_state) is not None:
            continue
        res = evaluate(body, route, move)
        if not res.ok:
            continue
        if move.swing:
            swings += 1
            continue
        pts = path_check(body, route, move, res, n=samples)
        done += 1
        points += len(pts)
        for p in pts:
            if p["t_min"] is None:
                all_ok = False
                continue
            worst_min = max(worst_min, p["t_min"] - res.difficulty)
            worst_mix = max(worst_mix, p["t_mix"] - res.difficulty)
            all_ok = all_ok and p["ok_mix"]
    return {"moves": done, "tried": tried, "swing_moves": swings, "points": points, "all_feasible": all_ok,
            "worst_min": float(worst_min), "worst_mix": float(worst_mix)}
