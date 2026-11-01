"""The height sweep (step 6 of the plan).

One route, one climber whose height changes: span = height, strength and
flexibility the average of Cal and Ava. Every length of the body scales with
the height (step 2), so the only thing that changes is size.

    rows     crux, moves and failure at 19 heights, 1.50 to 1.95 m
    H_star   the lowest height that gets up, found by bisection to 0.5 cm
             between neighbouring grid heights that fail and succeed;
             every switch on the grid is reported, in either direction
    runs     full betas at a few representative heights, for the page
"""

from __future__ import annotations

from .model import Climber, Route
from .search import solve

GRID = tuple(round(1.50 + 0.025 * i, 3) for i in range(19))
REPRESENTATIVE = (1.55, 1.65, 1.75, 1.85)
TOL = 0.005  # bisection precision (m)
F = {"jug": 1.0, "crimp": 0.625, "sloper": 0.30, "pinch": 0.475, "foot": 1.5}
FLEX = 0.04


def climber_at(H: float) -> Climber:
    """The sweep climber at height H: span H, the average of Cal's and Ava's strengths and flexibility."""
    cm = int(round(H * 1000))
    return Climber(f"S_{cm}", f"{H:.3f}".rstrip("0").rstrip(".") + " m", H, H, dict(F), FLEX,
                   blurb="The sweep climber: everything but size fixed.")


def _row(route: Route, H: float) -> dict:
    b = solve(route, climber_at(H))
    return {"H": H, "ok": b.status == "ok", "crux": b.crux, "n_steps": len(b.steps), "fail": b.fail}


def thresholds(route: Route, rows: list) -> list:
    """Bisect every switch between neighbouring grid heights. Each entry: the height where the
    outcome changes (to TOL), and whether it switches to success ("up") or to failure ("down")."""
    out = []
    for a, b in zip(rows, rows[1:]):
        if a["ok"] == b["ok"]:
            continue
        lo, hi = a["H"], b["H"]  # outcome at lo is a["ok"], at hi is b["ok"]
        while hi - lo > TOL:
            mid = (lo + hi) / 2
            if _row(route, mid)["ok"] == a["ok"]:
                lo = mid
            else:
                hi = mid
        out.append({"H": round((lo + hi) / 2, 4), "lo": round(lo, 4), "hi": round(hi, 4),
                    "direction": "up" if b["ok"] else "down"})
    return out


def sweep(route: Route, grid=GRID, log=None) -> dict:
    rows = []
    for H in grid:
        rows.append(_row(route, H))
        if log:
            r = rows[-1]
            log(f"  H={H:.3f}: {'ok' if r['ok'] else 'fail'} crux={r['crux'] and round(r['crux'], 3)}")
    sw = thresholds(route, rows)
    ups = [s for s in sw if s["direction"] == "up"]
    return {"rows": rows, "switches": sw, "H_star": ups[0]["H"] if len(sw) == 1 and ups else None}
