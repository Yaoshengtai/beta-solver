"""The linear program against the exact cone program (section 3.7, unit 2.9).

Random states on random holds, each checked both ways. The linear program
replaces every disc and ball by a polygon or polyhedron inside it, so it can
only be more cautious: it should never hold a state the exact program
rejects, and its t* should stay within 10% of the exact one.
"""

from __future__ import annotations

import time
from functools import lru_cache

import numpy as np

from .body import make_body
from .model import PROFILES, VARIANTS, Hold, Route
from .stance import State, precheck, solve, solve_exact

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
def compare(n: int = 500, seed: int = 7) -> tuple:
    """Check n random states (that pass the prechecks) with both programs.

    Each record: climber key, lp_ok, lp_t, exact_ok, exact_t.
    """
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        climber, route, state, no_force = random_case(rng)
        body = make_body(climber)
        if precheck(body, route, state, no_force) is not None:
            continue
        t0 = time.perf_counter()
        lp = solve(body, route, state, no_force, tidy=False)
        t1 = time.perf_counter()
        ex = solve_exact(body, route, state, no_force)
        t2 = time.perf_counter()
        out.append({"climber": climber.key, "lp_ok": lp["ok"], "lp_t": lp["t"] if lp["ok"] else None,
                    "exact_ok": ex["ok"], "exact_t": ex["t"] if ex["ok"] else None,
                    "lp_ms": 1000 * (t1 - t0), "exact_ms": 1000 * (t2 - t1)})
    return tuple(out)


def summary(records) -> dict:
    both = [r for r in records if r["lp_ok"] and r["exact_ok"]]
    gaps = [(r["lp_t"] - r["exact_t"]) / r["exact_t"] for r in both if r["exact_t"] > 1e-9]
    return {
        "n": len(records),
        "both": len(both),
        "neither": sum(1 for r in records if not r["lp_ok"] and not r["exact_ok"]),
        "lp_only": sum(1 for r in records if r["lp_ok"] and not r["exact_ok"]),
        "exact_only": sum(1 for r in records if r["exact_ok"] and not r["lp_ok"]),
        "max_gap": max(gaps) if gaps else 0.0,
        "mean_gap": float(np.mean(gaps)) if gaps else 0.0,
        "min_gap": min(gaps) if gaps else 0.0,
        "lp_ms": float(np.median([r["lp_ms"] for r in records])) if records and "lp_ms" in records[0] else None,
        "exact_ms": float(np.median([r["exact_ms"] for r in records])) if records and "exact_ms" in records[0] else None,
    }
