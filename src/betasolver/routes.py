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

HARD = 0.45  # a passing climber's crux should be at least this: the routes are meant to be hard


@dataclass(frozen=True)
class Example:
    id: str
    title: str            # the route's name
    lesson: str           # what it shows, in one line
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
        "ex1", "Halfway House",
        "A long gap with one small crimp halfway: reach past it, or hang on it.",
        _route("ex1", "Halfway House", [
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
    Example(
        "ex2", "Fork in the Road",
        "Small crimps straight up, or big jugs far out to the right.",
        _route("ex2", "Fork in the Road", [
            ("S1", 1.35, 1.45, "jug"), ("S2", 1.65, 1.45, "jug"),
            ("C1", 1.12, 2.05, "crimp"), ("C2", 1.12, 2.55, "crimp"), ("C3", 1.12, 3.05, "crimp"),
            ("J1", 2.40, 2.70, "jug"), ("J2", 2.10, 3.30, "jug"), ("T", 1.50, 3.50, "jug"),
            ("F1", 1.19, 0.50, "foot_big"), ("F2", 1.19, 1.05, "foot_big"), ("F3", 1.19, 1.55, "foot_big"),
            ("F4", 1.19, 2.00, "foot_big"), ("R1", 2.15, 1.00, "foot_big"), ("R2", 2.25, 1.70, "foot_big")],
            feet_on_hands=False),
        ("A", "B", "C", "D"),
        {"A": "ok", "B": "ok", "C": "fail", "D": "fail"},
        (
            "Two ways up. On the left, three small crimps in a line, with every foothold a few centimetres to their right. "
            "On the right, two big jugs, J1 and J2, so far apart that only a long reach links them.",
            "{A.label} takes the crimps (crux {A.crux}). With every foothold off to one side, a hand on a crimp has to "
            "twist a little to keep the body square, and her strong fingers can.",
            "{B.label} takes the jugs (crux {B.crux}), using a crimp only as a step while the other hand is on a jug, "
            "which does the twisting.",
            "{C.label} is {C.fail}. The same twist is too much for a weaker grip, and the body swings open.",
            "{D.label} is {D.fail}. She has Ava's body, so her failure also shows that the jugs alone are too far apart "
            "for a climber of Ava's size.",
        ),
        (
            ("Ava climbs on the crimps and never touches the jugs", lambda R: R["A"]["status"] == "ok"
             and {"C1", "C2", "C3"} <= {st["to"] for st in R["A"]["steps"]} and not ({"J1", "J2"} & {st["to"] for st in R["A"]["steps"]})),
            ("Ben uses the jug J1", lambda R: R["B"]["status"] == "ok" and "J1" in {st["to"] for st in R["B"]["steps"]}),
            ("Cal fails by a barn-door (direction or balance)", lambda R: R["C"]["fail"]["reason"] == "balance"),
            ("Ava (injured) is stopped by a crimp", lambda R: R["D"]["fail"]["reason"] == "unusable"),
        ),
    ),
    Example(
        "ex3", "Lean on Me",
        "A layback: hands pull sideways on a flake, feet push back, and nothing may twist open.",
        _route("ex3", "Lean on Me", [
            ("S1", 1.35, 1.45, "jug"), ("S2", 1.65, 1.45, "jug"),
            ("L1", 1.30, 1.90, "sidepull", 0.0), ("L2", 1.33, 2.35, "sidepull", 0.0),
            ("L3", 1.30, 2.80, "sidepull", 0.0), ("L4", 1.33, 3.25, "sidepull", 0.0), ("T", 1.55, 3.70, "jug"),
            ("F1", 1.36, 0.55, "foot_big"), ("F2", 1.41, 1.00, "foot_big"), ("F3", 1.36, 1.45, "foot_big"),
            ("F4", 1.41, 1.90, "foot_big"), ("F5", 1.36, 2.35, "foot_big")], feet_on_hands=False),
        ("A", "B", "C"),
        {"A": "ok", "B": "ok", "C": "fail"},
        (
            "A flake runs straight up: four sidepulls, L1 to L4, that pull the hands to the right, with the footholds "
            "almost directly below them, a few centimetres to the right. The hands pull, the feet push back, and every "
            "time a hand or foot moves, whatever stays on must also stop the body twisting open.",
            "{A.label} laybacks up it in {A.steps} moves (crux {A.crux}).",
            "{B.label} gets up too (crux {B.crux}), but needs {B.steps} moves, with more shuffling. He stands further "
            "from the wall, so his hands must pull in harder, and the same small offset of the feet needs more twist.",
            "{C.label} gets both hands onto L2 and then cannot move either foot: one foot pushing a few centimetres "
            "to the side of two weak sidepull grips swings the body open. With the feet stuck low, Cal is {C.fail}.",
        ),
        (
            ("Ben needs more moves than Ava", lambda R: len(R["B"]["steps"]) > len(R["A"]["steps"])),
            ("Cal is stuck with both hands on L2 and his feet low",
             lambda R: R["C"]["fail"]["state"]["LH"] == R["C"]["fail"]["state"]["RH"] == "L2"
             and all(R["C"]["fail"]["state"][f] in ("F1", "F2", None) for f in ("LF", "RF"))),
        ),
    ),
    Example(
        "ex4", "Monkey Bars",
        "Feet off: once the footholds run out, every hand move hangs the whole body on one arm.",
        _route("ex4", "Monkey Bars", [
            ("S1", 1.35, 1.45, "jug"), ("S2", 1.65, 1.45, "jug"),
            ("J1", 1.35, 1.95, "jug"), ("J2", 1.65, 2.45, "jug"), ("J3", 1.35, 2.95, "jug"),
            ("J4", 1.65, 3.45, "jug"), ("T", 1.50, 3.90, "jug"),
            ("F1", 1.35, 0.45, "foot_big"), ("F2", 1.65, 0.45, "foot_big"), ("F3", 1.50, 0.95, "foot_big")],
            feet_on_hands=False),
        ("A", "B", "C"),
        {"A": "ok", "B": "ok", "C": "fail"},
        (
            "Good jugs all the way, 0.5 m apart, but the footholds stop at knee height. Above J2 the feet can "
            "reach nothing and hang, and every time a hand lets go, the other hand holds the whole body.",
            "{A.label} climbs it in {A.steps} moves (crux {A.crux}). Her feet come off by J2; from then on she "
            "matches each jug before reaching for the next, and every reach needs t = 1/1.2 = 0.83.",
            "{B.label} has the same jug strength, so the same crux ({B.crux}), but his long arms skip J1: "
            "{B.steps} moves.",
            "{C.label} gets both hands onto J2 with his feet hanging, which is fine while two hands share the "
            "load. But moving a hand means hanging on the other one alone, which needs 1/0.8 = 1.25 of his jug "
            "strength. Cal is {C.fail}.",
        ),
        (
            ("Ben skips J1", lambda R: all(st["to"] != "J1" for st in R["B"]["steps"])),
            ("Ben needs fewer moves than Ava", lambda R: len(R["B"]["steps"]) < len(R["A"]["steps"])),
            ("Cal fails on strength with both feet off",
             lambda R: R["C"]["fail"]["reason"] == "strength"
             and R["C"]["fail"]["state"]["LF"] is None and R["C"]["fail"]["state"]["RF"] is None),
        ),
    ),
]
