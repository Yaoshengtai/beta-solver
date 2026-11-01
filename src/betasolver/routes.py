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
    sweep: bool = False   # also run the height sweep (step 6); the runs are then its representative heights


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


def _states(run):
    """Every state along a run's beta, the start included."""
    st = dict(run["start"])
    out = [dict(st)]
    for step in run["steps"]:
        st[step["limb"]] = step["to"]
        out.append(dict(st))
    return out


def _feet_off(run):
    """Does the climber ever hang with both feet off?"""
    return any(s["LF"] is None and s["RF"] is None for s in _states(run))


def _never_drops(xs, tol=0.005):
    """Is the sequence non-decreasing, up to tol?"""
    return all(b >= a - tol for a, b in zip(xs, xs[1:]))


EXAMPLES: list = [
    Example(
        "ex1", "Halfway House",
        "A long gap with one small crimp halfway: reach past it, or hang on it.",
        _route("ex1", "Halfway House", [
            ("S1", 1.35, 1.45, "jug"), ("S2", 1.65, 1.45, "jug"), ("C", 1.25, 2.25, "crimp"),
            ("J1", 1.55, 2.60, "jug"), ("J2", 1.45, 3.05, "jug"), ("T", 1.55, 3.45, "jug"),
            ("F1", 1.30, 0.50, "foot_big"), ("F2", 1.70, 0.50, "foot_big"),
            ("F3", 1.25, 1.75, "foot_big"), ("F4", 1.75, 1.80, "foot_big")]),
        ("B", "A", "C", "D"),
        {"B": "ok", "A": "ok", "C": "fail", "D": "fail"},
        (
            "Between the start jugs and the next jug J1 there is a long gap. The only thing in it is a small crimp, "
            "C, off to the side, and apart from the start jugs there are no footholds between the floor and the high ones.",
            "{B.label} is tall enough to reach J1 straight from the start, with the feet still low (crux {B.crux}). "
            "{A.label} is not: she pulls onto the crimp with both hands, steps her feet up onto a start jug and a high foothold, and "
            "reaches J1 from there (crux {A.crux}).",
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
            ("F4", 1.19, 2.00, "foot_big"), ("R1", 2.15, 1.00, "foot_big"), ("R2", 2.25, 1.70, "foot_big")]),
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
            ("F4", 1.41, 1.90, "foot_big"), ("F5", 1.36, 2.35, "foot_big")]),
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
        "ex4", "Knees Up",
        "One high foothold: flexible hips decide, not height.",
        _route("ex4", "Knees Up", [
            ("S1", 1.35, 1.45, "jug"), ("S2", 1.65, 1.45, "jug"),
            ("U1", 1.40, 2.15, "undercling"), ("U2", 1.62, 2.20, "undercling"),
            ("J1", 1.50, 3.05, "jug"), ("T", 1.50, 3.50, "jug"),
            ("F1", 1.35, 0.45, "foot_big"), ("F2", 1.65, 0.45, "foot_big"), ("F3", 1.30, 1.20, "foot_big")]),
        ("A", "B", "A_stiff", "B_flex"),
        {"A": "ok", "B": "fail", "A_stiff": "fail", "B_flex": "ok"},
        (
            "Above the start jugs there are only two underclings, and then J1, 0.85 m higher. The footholds are low, "
            "except F3, just 25 cm below the start jugs. To reach J1 the feet have to come up, first onto "
            "F3 and then onto the start jugs, while the hands pull up on the underclings.",
            "{A.label} starts with a foot on F3, a foot near hip height. Her flexible hips allow that. From there she "
            "steps onto the jugs and stands up into J1 ({A.steps} moves, crux {A.crux}).",
            "{B.label} is {B.fail}. He is much taller, but his stiff hips will not let a foot go anywhere near F3 "
            "or the start jugs, and with his feet that low even his reach falls short.",
            "{A_stiff.label} has Ava's body with Ben's hips. She is {A_stiff.fail}.",
            "{B_flex.label} has Ben's body with Ava's hips, and gets up ({B_flex.steps} moves, crux {B_flex.crux}). "
            "So it is the hips, not the height.",
        ),
        (
            ("Ava and Ben (flexible) start with a foot on F3",
             lambda R: all("F3" in (R[k]["start"]["LF"], R[k]["start"]["RF"]) for k in ("A", "B_flex"))),
            ("Both stiff climbers are stuck with both feet on the floor footholds",
             lambda R: all(R[k]["fail"]["state"][f] in ("F1", "F2") for k in ("B", "A_stiff") for f in ("LF", "RF"))),
        ),
    ),
    Example(
        "ex5", "Mind the Gap",
        "No footholds under the rail: bridge across with a long body, or let the feet go.",
        _route("ex5", "Mind the Gap", [
            ("S1", 0.30, 1.90, "jug"), ("S2", 0.60, 1.90, "jug"),
            ("R1", 1.05, 2.00, "jug"), ("R2", 1.52, 2.05, "jug"), ("R3", 1.98, 2.10, "jug"),
            ("R4", 2.45, 2.15, "jug"), ("T", 2.65, 2.70, "jug"),
            ("F1", 0.35, 0.50, "foot_big"), ("F2", 0.65, 0.95, "foot_big"),
            ("F3", 2.45, 1.05, "foot_big"), ("F4", 2.70, 1.50, "foot_big")]),
        ("A", "B", "C"),
        {"A": "ok", "B": "ok", "C": "fail"},
        (
            "A traverse to the right along a rail of good jugs, R1 to R4, about 0.47 m apart. The footholds are at "
            "both ends, and the wall under the rail is blank for almost 2 m.",
            "{B.label} bridges the gap ({B.steps} moves, crux {B.crux}): with a foot still on the start footholds he "
            "can already reach R2, and from there his right foot reaches F3 at the far end. He always has a foot on.",
            "{A.label} is too small for that. She lets both feet go and swings across on her arms, matching each jug "
            "({A.steps} moves, crux {A.crux}): every hand move hangs her whole weight on one jug, t = 1/1.2 = 0.83.",
            "{C.label} can do neither. He is too short to bridge like Ben, and hanging on one jug, like Ava, needs "
            "1/0.8 = 1.25 of his jug strength. The furthest he gets is a stretch from the start to R3 on one foot, and "
            "there he is {C.fail}.",
        ),
        (
            ("Ava hangs with both feet off at some point", lambda R: _feet_off(R["A"])),
            ("Ben always keeps a foot on", lambda R: not _feet_off(R["B"])),
            ("Ben's crux is lower than Ava's", lambda R: R["B"]["crux"] < R["A"]["crux"]),
        ),
    ),
    Example(
        "ex6", "The Stretch",
        "How tall do you need to be? One route, one climber, and only the height changes.",
        _route("ex6", "The Stretch", [
            ("S1", 1.35, 1.25, "jug"), ("S2", 1.65, 1.25, "jug"), ("J1", 1.50, 1.85, "jug"),
            ("J2", 1.50, 2.90, "jug"), ("T", 1.50, 3.30, "jug"),
            ("F1", 1.35, 0.40, "foot_chip"), ("F2", 1.65, 0.40, "foot_chip")]),
        ("S_1550", "S_1650", "S_1750", "S_1850"),
        {"S_1550": "fail", "S_1650": "fail", "S_1750": "ok", "S_1850": "ok"},
        (
            "Low start jugs, one more jug J1 above them, and then nothing: the next jug, J2, is 1.05 m higher, and "
            "the only footholds are two chips near the floor. Every climber on this page is the same person at a "
            "different height: arm span equal to height, grip and hips halfway between Cal and Ava, and every length "
            "of the body in proportion to the height (chapter 2).",
            "The chart solves the route at 19 heights from 1.50 m to 1.95 m. Below {sweep.hstar} nobody gets up, and "
            "always for the same reason: from J1, J2 is out of reach. From there up, everyone does. "
            "At 1.65 m the climber is {S_1650.fail}.",
            "Past the threshold, being taller does not make the route easier. The crux even creeps up, from "
            "{S_1750.crux} at 1.75 m to {S_1850.crux} at 1.85 m: a taller body stands further from the wall, so "
            "its hands must pull in harder. The tallest climbers save two moves, because they reach the top "
            "without stepping onto J1 first. Height buys reach, and on this route reach is all that matters.",
        ),
        (
            ("H* is between 1.60 and 1.85 m",
             lambda R: R["sweep"]["H_star"] is not None and 1.60 <= R["sweep"]["H_star"] <= 1.85),
            ("The grid switches once, from no beta to a beta",
             lambda R: len(R["sweep"]["switches"]) == 1 and R["sweep"]["switches"][0]["direction"] == "up"),
            ("Every height that fails is stopped by reach",
             lambda R: all(r["fail"]["reason"] == "reach" for r in R["sweep"]["rows"] if not r["ok"])),
            ("Past H*, the crux never drops with height",
             lambda R: _never_drops([r["crux"] for r in R["sweep"]["rows"] if r["ok"]])),
            ("The tallest climber needs fewer moves than the shortest that gets up",
             lambda R: R["sweep"]["rows"][-1]["n_steps"] < next(r["n_steps"] for r in R["sweep"]["rows"] if r["ok"])),
        ),
        sweep=True,
    ),
]
