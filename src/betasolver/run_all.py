"""Solve every example and write site/results.json (steps 7 and 8 of the plan).

    python -m betasolver.run_all            # all examples
    python -m betasolver.run_all ex1 ex3    # only these (others kept from the old file)

The page build only reads results.json, so the searches run once, not on
every build. The file records a fingerprint of the route definitions and
the solver code; a test fails when it is out of date.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import sys
import time
from pathlib import Path

from .export import export_run
from .model import PROFILES, VARIANTS
from .routes import EXAMPLES, checks
from .search import solve
from .sweep import FLEX, GRID, F, climber_at, sweep

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "site" / "results.json"
SOURCES = ["routes.py", "model.py", "body.py", "stance.py", "moves.py", "search.py", "export.py", "sweep.py"]


def fingerprint() -> str:
    h = hashlib.sha256()
    here = Path(__file__).resolve().parent
    for name in SOURCES:
        h.update((here / name).read_bytes())
    return h.hexdigest()[:16]


def climber(key):
    """A profile, a variant, or a sweep climber S_<height in mm>."""
    if key.startswith("S_"):
        return climber_at(int(key[2:]) / 1000)
    return PROFILES.get(key) or VARIANTS[key]


def _r3(x):
    if isinstance(x, float):
        return round(x, 3)
    if isinstance(x, dict):
        return {k: _r3(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_r3(v) for v in x]
    return x


def run_example(ex) -> dict:
    runs, t0 = {}, time.perf_counter()
    for key in ex.climbers:
        c = climber(key)
        b = solve(ex.route, c)
        runs[key] = export_run(ex.route, c, b)
        print(f"  {ex.id} {c.name}: {b.status} crux={b.crux and round(b.crux, 3)} steps={len(b.steps)} "
              f"({b.stats.get('seconds', 0):.1f} s)", flush=True)
    sw = None
    if ex.sweep:
        print(f"  {ex.id} height sweep", flush=True)
        sw = sweep(ex.route, log=lambda m: print("  " + m, flush=True))
        sw = _r3({**sw, "grid": list(GRID), "F": F, "flex": FLEX})
    res = checks(ex, {**runs, "sweep": sw} if sw else runs)
    out = {
        "id": ex.id, "title": ex.title, "lesson": ex.lesson, "text": list(ex.text),
        "route": ex.route.to_dict(), "climbers": list(ex.climbers), "expect": ex.expect,
        "runs": [runs[k] for k in ex.climbers], "checks": res,
        "seconds": round(time.perf_counter() - t0, 1),
    }
    if sw:
        out["sweep"] = sw
    return out


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    old = json.loads(OUT.read_text()) if OUT.exists() else {"examples": []}
    keep = {e["id"]: e for e in old.get("examples", [])}
    out = []
    for ex in EXAMPLES:
        if argv and ex.id not in argv and ex.id in keep:
            out.append(keep[ex.id])
            continue
        print(ex.id, ex.title, flush=True)
        out.append(run_example(ex))
    data = {"meta": {"version": 1, "generated": dt.date.today().isoformat(), "fingerprint": fingerprint(),
                     "units": "m, body weights"}, "examples": out}
    OUT.write_text(json.dumps(data, separators=(",", ":")))
    bad = [(e["id"], c["text"]) for e in out for c in e["checks"] if not c["ok"]]
    for i, t in bad:
        print(f"FAILED {i}: {t}")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB), {len(out)} examples, {len(bad)} failed checks")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
