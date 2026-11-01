"""Writes the test report shown in the site's appendix (unit 7.2):

    pytest --report site/test_report.json
"""

import ast
import datetime as dt
import json
import platform
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def pytest_addoption(parser):
    parser.addoption("--report", default=None, metavar="PATH", help="write a JSON test report for the site")


_STATE = {}  # filled while the session runs


def pytest_configure(config):
    _STATE.update(tests={}, t0=time.time())


def pytest_runtest_logreport(report):
    t = _STATE["tests"].setdefault(report.nodeid, {"outcome": "passed", "seconds": 0.0})
    t["seconds"] += report.duration
    if report.failed:
        t["outcome"] = "failed"
    elif report.skipped and t["outcome"] != "failed":
        t["outcome"] = "skipped"


def _about(path: Path) -> str:
    doc = ast.get_docstring(ast.parse(path.read_text(encoding="utf-8"))) or ""
    return doc.splitlines()[0] if doc else ""


def _name(nodeid: str) -> str:
    """test_two_hands_share[ex1] -> 'two hands share [ex1]'"""
    fn = nodeid.split("::")[-1]
    base, _, param = fn.partition("[")
    base = base[5:] if base.startswith("test_") else base
    return base.replace("_", " ") + (f" [{param}" if param else "")


def pytest_sessionfinish(session, exitstatus):
    out = session.config.getoption("--report")
    if not out:
        return
    files = {}
    for nodeid, t in _STATE["tests"].items():
        f = nodeid.split("::")[0]
        files.setdefault(f, []).append({"name": _name(nodeid), "outcome": t["outcome"], "seconds": round(t["seconds"], 3)})
    counts = {k: sum(t["outcome"] == k for t in _STATE["tests"].values()) for k in ("passed", "failed", "skipped")}
    report = {
        "generated": dt.date.today().isoformat(),
        "python": platform.python_version(),
        "pytest": pytest.__version__,
        "seconds": round(time.time() - _STATE["t0"], 1),
        "counts": counts,
        "files": [{"file": f, "about": _about(ROOT / f), "tests": ts} for f, ts in sorted(files.items())],
    }
    Path(out).write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
