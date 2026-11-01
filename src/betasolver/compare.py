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
