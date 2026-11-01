"""The examples: site/results.json must be current and every acceptance check must hold.

results.json is written by `python -m betasolver.run_all`; rerun it after changing the routes or the solver.
"""
import json
from pathlib import Path

import pytest

from betasolver.routes import EXAMPLES, HARD, checks
from betasolver.run_all import OUT, fingerprint

RESULTS = json.loads(Path(OUT).read_text()) if Path(OUT).exists() else None


def test_results_exist_and_are_current():
    assert RESULTS is not None, "run: python -m betasolver.run_all"
    assert RESULTS["meta"]["fingerprint"] == fingerprint(), "routes or solver changed: run python -m betasolver.run_all"
    assert [e["id"] for e in RESULTS["examples"]] == [e.id for e in EXAMPLES]


@pytest.mark.parametrize("ex", EXAMPLES, ids=lambda e: e.id)
def test_acceptance_checks(ex):
    stored = {e["id"]: e for e in RESULTS["examples"]}[ex.id]
    runs = {r["key"]: r for r in stored["runs"]}
    if ex.sweep:
        runs["sweep"] = stored["sweep"]
    for c in checks(ex, runs):                     # recomputed from the stored runs, not just read back
        assert c["ok"], f"{ex.id}: {c['text']}"
    statuses = {k: runs[k]["status"] for k in ex.climbers}
    assert "ok" in statuses.values() and any(s != "ok" for s in statuses.values()), "some pass, some fail"


@pytest.mark.parametrize("ex", EXAMPLES, ids=lambda e: e.id)
def test_runs_are_well_formed(ex):
    stored = {e["id"]: e for e in RESULTS["examples"]}[ex.id]
    for r in stored["runs"]:
        if r["status"] == "ok":
            assert r["crux"] >= HARD and r["steps"] and all(len(s["frames"]) == 4 for s in r["steps"])
            last = r["steps"][-1]["frames"][-1]["state"]
            assert last["LH"] == last["RH"] == ex.route.finish
        else:
            f = r["fail"]
            assert f["phase"] in ("release", "catch", "start", "end") and f["limb"] in ("LH", "RH") and f["hold"]
