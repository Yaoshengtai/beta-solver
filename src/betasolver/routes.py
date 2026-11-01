"""The examples (step 7 of the plan, redesigned): hard routes on which climbers' outcomes differ.

Each example is a route, the climbers who try it, what each should get
(pass or fail), and a short text. Texts may name numbers from the results
with placeholders that the page build fills in: {A.crux}, {B.steps},
{C.fail}, ... (climber key, then a field).

Routes are tuned by running the real search (run_all.py) and adjusting
hold positions; the acceptance checks below then freeze the outcome.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .model import Hold, Route

HARD = 0.6  # a passing climber's crux should be at least this: the routes are meant to be hard


@dataclass(frozen=True)
class Example:
    id: str
    title: str
    lesson: str
    route: Route
    climbers: tuple       # profile or variant keys, in display order
    expect: dict          # key -> "ok" or "fail"
    text: tuple = ()      # paragraphs, with {KEY.field} placeholders
    extra: tuple = field(default=())  # (description, function(runs) -> bool)


def checks(ex: Example, runs: dict) -> list:
    """The acceptance checks of one example, evaluated on its exported runs (keyed by climber key)."""
    out = []
    for k, want in ex.expect.items():
        r = runs[k]
        if want == "ok":
            out.append({"text": f"{r['label']} gets up", "ok": r["status"] == "ok"})
            out.append({"text": f"{r['label']}'s crux is at least {HARD}", "ok": r["status"] == "ok" and r["crux"] >= HARD})
        else:
            out.append({"text": f"{r['label']} does not get up", "ok": r["status"] != "ok"})
    for desc, fn in ex.extra:
        try:
            ok = bool(fn(runs))
        except Exception:
            ok = False
        out.append({"text": desc, "ok": ok})
    return out


def _route(rid, title, holds, start=("S1", "S2"), top="T", feet_on_hands=True):
    return Route(rid, title, "", [Hold(*h) if len(h) < 5 else Hold(h[0], h[1], h[2], h[3], dir=h[4]) for h in holds],
                 start, top, allow_feet_on_handholds=feet_on_hands)


def _first_hand_target(run):
    return next((st["to"] for st in run["steps"] if st["limb"] in ("LH", "RH")), None)


def _uses(run, hold):
    return any(st["to"] == hold for st in run["steps"])


EXAMPLES: list = [
    Example(
        "ex1", "The intermediate crimp",
        "A long gap with one small crimp on the way: reach it or work for it.",
        _route("ex1", "The intermediate crimp", [
            ("S1", 1.35, 1.45, "jug"), ("S2", 1.65, 1.45, "jug"), ("C", 1.25, 2.25, "crimp"),
            ("J1", 1.55, 2.60, "jug"), ("J2", 1.45, 3.05, "jug"), ("T", 1.55, 3.45, "jug"),
            ("F1", 1.30, 0.50, "foot_big"), ("F2", 1.70, 0.50, "foot_big"),
            ("F3", 1.25, 1.75, "foot_big"), ("F4", 1.75, 1.80, "foot_big")], feet_on_hands=False),
        ("B", "A", "C", "D"),
        {"B": "ok", "A": "ok", "C": "fail", "D": "fail"},
        (
            "Between the start jugs and the next jug J1 there is a long gap. The only thing in it is a small crimp, "
            "C, off to the side, and there are no footholds between the floor and the high ones.",
            "{B.label} is tall enough to reach J1 straight from the start, with the feet still low (crux {B.crux}). "
            "{A.label} is not: she pulls onto the crimp with both hands, steps her feet up to the high footholds, and "
            "reaches J1 from there. It works, but at her limit (crux {A.crux}).",
            "{C.label} gets a hand onto the crimp but is {C.fail}. With one hand on a crimp and both feet to one side, "
            "the body needs a little grip twist to stay square, and a weak grip on a small edge cannot give it: a barn-door.",
            "{D.label} cannot use the crimp at all ({D.fail}), and without it J1 is out of reach.",
        ),
        (
            ("Ava uses the crimp before she reaches J1", lambda R: R["A"]["status"] == "ok" and _first_hand_target(R["A"]) == "C"),
            ("Ben's first hand move goes straight to J1", lambda R: R["B"]["status"] == "ok" and _first_hand_target(R["B"]) == "J1"),
            ("Cal gets onto the crimp before failing", lambda R: R["C"]["fail"] and "C" in R["C"]["fail"]["state"].values()),
            ("Ava (injured) is stopped by the crimp itself", lambda R: R["D"]["fail"] and R["D"]["fail"]["reason"] == "unusable"),
        ),
    ),
]
