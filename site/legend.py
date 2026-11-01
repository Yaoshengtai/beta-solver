"""Data for the figures in the #data and #body chapters.

The legend shows what every mark on the wall means. Its pose comes from the
body model (body.py); its forces and efforts are example values, not solver
output.
"""

from __future__ import annotations

from betasolver.body import make_body
from betasolver.model import AVA, HOLD_TYPES, PROFILES, WALL_H, WALL_W, Hold, Route

WALL = {"w": WALL_W, "h": WALL_H}

# type, which limbs may use it, default direction (deg, 90 = up); a sidepull is drawn facing right
HOLD_TYPES_SHORT = [
    {"type": t.name, "use": "foot" if t.feet_only else "both", "dir": 0 if t.dir is None else t.dir}
    for t in HOLD_TYPES.values()
]

def _round(obj):
    if isinstance(obj, float):
        return round(obj, 3)
    if isinstance(obj, (list, tuple)):
        return [_round(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _round(v) for k, v in obj.items()}
    return obj


def legend() -> dict:
    body = make_body(AVA)
    holds = [
        {"id": "H1", "x": 1.15, "y": 2.30, "type": "jug", "dir": 90, "role": "normal"},
        {"id": "H2", "x": 1.85, "y": 2.05, "type": "crimp", "dir": 90, "role": "normal"},
        {"id": "H3", "x": 1.55, "y": 2.75, "type": "sloper", "dir": 90, "role": "normal"},
        {"id": "H4", "x": 2.15, "y": 2.55, "type": "pinch", "dir": 90, "role": "normal"},
        {"id": "F1", "x": 1.25, "y": 0.75, "type": "foot_big", "dir": 90, "role": "normal"},
        {"id": "F2", "x": 1.75, "y": 0.95, "type": "foot_chip", "dir": 90, "role": "normal"},
        {"id": "F3", "x": 1.00, "y": 1.20, "type": "foot_big", "dir": 90, "role": "normal"},
    ]
    at = {h["id"]: (h["x"], h["y"]) for h in holds}
    state = {"LH": "H1", "RH": "H2", "LF": "F1", "RF": "F2"}
    torso = (1.50, 1.42)
    frame = {"torso": list(torso), "joints": body.joints(torso, {k: at[v] for k, v in state.items()})}
    frame["state"] = state
    # Example values only: in-wall forces sum to the body weight (0, 1).
    frame["forces"] = {
        "LH": {"f": [0.06, 0.32], "n": 0.21},
        "RH": {"f": [-0.10, 0.16], "n": 0.12},
        "LF": {"f": [0.07, 0.29], "m": 0.20},
        "RF": {"f": [-0.03, 0.23], "m": 0.13},
    }
    frame["util"] = {"LH": 0.32, "RH": 0.71, "LF": 0.19, "RF": 0.88}
    trail = [[1.42, 1.30], [1.47, 1.36], [1.50, 1.42]]

    markings = [
        {"id": "S1", "x": 0.25, "y": 0.25, "type": "jug", "dir": 90, "role": "start", "used": True},
        {"id": "T1", "x": 0.75, "y": 0.25, "type": "jug", "dir": 90, "role": "finish", "used": True},
        {"id": "U1", "x": 1.25, "y": 0.25, "type": "jug", "dir": 90, "role": "normal", "used": True},
        {"id": "N1", "x": 1.75, "y": 0.25, "type": "jug", "dir": 90, "role": "normal", "used": False},
    ]
    return _round({
        "wall": WALL,
        "hold_types": HOLD_TYPES_SHORT,
        "markings": markings,
        "climber": {"key": AVA.key, "H": AVA.H, "span": AVA.span, "d_wall": body.d_wall,
                    "head": list(body.head)},
        "scene": {"x0": 0.75, "y0": 0.45, "x1": 2.85, "y1": 3.05,
                  "holds": holds, "frame": frame, "trail": trail},
        "climbers": [{"key": c.key, "name": c.name} for c in PROFILES.values()],
        "force_scale": 0.3,
        "util_bands": [0.5, 0.8],
    })


def route_example() -> dict:
    """A small route that only illustrates the route format."""
    holds = [
        Hold("H1", 1.20, 1.60, "jug"),
        Hold("H2", 1.75, 1.55, "crimp"),
        Hold("H3", 1.05, 2.35, "pinch"),
        Hold("H4", 1.95, 2.55, "sidepull", dir=180),
        Hold("H5", 1.40, 3.05, "sloper"),
        Hold("H6", 1.60, 3.65, "jug"),
        Hold("F1", 1.25, 0.45, "foot_big"),
        Hold("F2", 1.85, 0.70, "foot_chip"),
        Hold("F3", 1.50, 1.10, "foot_big"),
    ]
    r = Route(
        id="format-demo",
        title="Format example",
        lesson="A small route that only shows how routes are written down.",
        holds=holds,
        start_hands=("H1", "H2"),
        finish="H6",
    )
    return _round(r.to_dict())


def body_figures() -> dict:
    """The four standard climbers' bodies, and a standing pose for each.

    Standing pose: legs straight and slightly apart with the feet on the
    ground, arms straight up at full length, so the hand height is the
    standing reach.
    """
    import math

    out = []
    for c in PROFILES.values():
        b = make_body(c)
        a = math.radians(8)  # legs spread a little
        yc = -b.offsets["LF"][1] + b.L["LF"] * math.cos(a)
        torso = (0.0, yc)
        ends = {}
        for limb in ("LH", "RH", "LF", "RF"):
            ax, ay = b.anchor(limb, torso)
            side = -1 if limb[0] == "L" else 1
            if limb[1] == "H":
                ends[limb] = (ax, ay + b.L[limb])
            else:
                ends[limb] = (ax + side * b.L[limb] * math.sin(a), ay - b.L[limb] * math.cos(a))
        out.append({
            **b.to_dict(),
            "name": c.name,
            "span": c.span,
            "stand": {"torso": list(torso), "joints": b.joints(torso, ends)},
            "reach": ends["LH"][1],
            "foot_limit": b.foot_limit(torso),
        })
    return _round(out)


def state_example() -> dict:
    """One state on the format example, and several body positions that all fit it.

    Each torso position is checked against the reach circles and the foot
    limit, so every pose drawn really is a body position the state allows.
    """
    from betasolver.model import Route
    from betasolver.stance import State

    r = route_example()
    holds = {h["id"]: h for h in r["holds"]}
    state = State("H1", "H2", "F1", "F2")
    body = make_body(AVA)
    ends = {l: (holds[h]["x"], holds[h]["y"]) for l, h in state.as_dict().items()}
    candidates = [(1.36, 0.84), (1.52, 0.96), (1.70, 0.76)]
    poses = []
    for torso in candidates:
        ok = all(body.reaches(l, torso, p) for l, p in ends.items())
        ok = ok and all(ends[f][1] <= body.foot_limit(torso) for f in ("LF", "RF"))
        if not ok:
            raise ValueError(f"example torso {torso} does not fit the state")
        poses.append({"torso": list(torso), "joints": body.joints(torso, ends)})
    return _round({"route": r["id"], "state": state.as_dict(), "holds": r["holds"],
                   "climber": {"key": AVA.key, "H": AVA.H, "head": list(body.head)}, "poses": poses})


def reach_figure() -> dict:
    """Data for the reach figure in #check: one state, its torso region, two starting positions."""
    from betasolver.model import Route
    from betasolver.stance import State, centres, reaches_all, torso_region

    r = route_example()
    route = Route(r["id"], r["title"], r["lesson"],
                  [Hold(h["id"], h["x"], h["y"], h["type"], h["dir"], h["use"]) for h in r["holds"]],
                  tuple(r["start_hands"]), r["finish"])
    state = State("H1", "H2", "F1", "F2")
    body = make_body(AVA)
    region = torso_region(body, route, state, sides=128)
    cx = sum(p[0] for p in region) / len(region)
    cy = sum(p[1] for p in region) / len(region)
    good, low = (cx, cy), (cx, cy - 0.42)
    if not all(reaches_all(body, route, state, good).values()):
        raise ValueError("the good example must reach every hold")
    if all(reaches_all(body, route, state, low).values()):
        raise ValueError("the low example must miss a hold")
    return _round({
        "state": state.as_dict(),
        "holds": r["holds"],
        "climber": {"key": AVA.key, "H": AVA.H, "head": list(body.head)},
        "body": body.to_dict(),
        "centres": {k: list(v) for k, v in centres(body, route, state).items()},
        "region": [list(p) for p in region],
        "presets": {"good": list(good), "low": list(low)},
        "view": {"x0": 0.55, "y0": 0.05, "x1": 2.45, "y1": 2.25},
    })


# ------------------------------------------------------------ ③ strength

# Example forces of section 3.3, the same for every climber. They are fixed
# inputs here, not solver output: until in-plane balance is added (unit 2.4)
# nothing decides how the weight is shared. They still sum to the body weight
# in the wall plane, the pull-ins equal the foot pushes, and every force lies
# in its hold's cone.
STRENGTH_FORCES = {
    "LH": {"f": (0.06, 0.40), "n": 0.197},
    "RH": {"f": (-0.03, 0.29), "n": 0.071},
    "LF": {"f": (-0.01, 0.16), "m": 0.130},
    "RF": {"f": (-0.02, 0.15), "m": 0.138},
}


def strength_route() -> Route:
    holds = [
        Hold("H1", 1.20, 2.10, "crimp"),
        Hold("H2", 1.75, 2.20, "sloper"),
        Hold("H3", 1.50, 3.60, "jug"),
        Hold("F1", 1.10, 0.80, "foot_big"),
        Hold("F2", 1.85, 0.90, "foot_big"),
    ]
    return Route("strength-demo", "Strength example", "A crimp, a sloper and two footholds.",
                 holds, ("H1", "H2"), "H3")


def strength_figure() -> dict:
    """Data for the strength figure in #check: one state, fixed forces, three climbers.

    For each climber: the strength limit of every contact, each limb's share
    of it, the effort t (exact), and t from the linear program with the forces
    held fixed, which can only be a little higher (the polyhedron sits inside
    the ball). Balance beyond the in-plane equations is left out: the fixed
    forces are an illustration, and the wall distance differs per climber.
    """
    from betasolver.stance import EFFORT_BANDS, State, band, coverage, effort, solve, strength_limit, torso_region

    route = strength_route()
    state = State("H1", "H2", "F1", "F2")
    ends = {l: (route.hold(h).x, route.hold(h).y) for l, h in state.as_dict().items()}
    runs = []
    for key in ("A", "B", "C"):
        c = PROFILES[key]
        body = make_body(c)
        region = torso_region(body, route, state)
        if not region:
            raise ValueError(f"{c.name} cannot reach the strength example")
        t, share, limiting = effort(body, route, state, STRENGTH_FORCES)
        lp = solve(body, route, state, fixed=STRENGTH_FORCES, upto="inplane")
        if not lp["ok"]:
            raise ValueError(f"the example forces break a constraint for {c.name}: {lp['message']}")
        # in-plane balance fixes x_c for these forces; y_c is free, so draw it mid-range
        torso = (lp["torso"][0], mid_y(region, lp["torso"][0]))
        runs.append({
            "climber": key, "name": c.name, "H": c.H, "head": list(body.head),
            "frame": {"torso": list(torso), "joints": body.joints(torso, ends), "state": state.as_dict(),
                      "forces": {l: {k: list(v) if isinstance(v, tuple) else v for k, v in F.items()}
                                 for l, F in STRENGTH_FORCES.items()},
                      "util": share},
            "limits": {l: strength_limit(body, route, state, l) for l in state.on()},
            "t": t, "t_lp": lp["t"], "limiting": limiting, "band": band(t)["key"],
        })
    return _round({
        "holds": [{**h, "role": "normal"} for h in route.to_dict()["holds"] if h["id"] != "H3"],
        "state": state.as_dict(),
        "runs": runs,
        "bands": EFFORT_BANDS,
        "coverage": coverage(),
        "force_scale": 0.25,
        "view": {"x0": 0.68, "y0": 0.4, "x1": 2.32, "y1": 2.42},
    })


def mid_y(region, x: float, frac: float = 0.5) -> float:
    """A point of the vertical extent of a convex polygon at a given x: 0 bottom, 0.5 middle, 1 top."""
    ys = []
    n = len(region)
    for i in range(n):
        (x1, y1), (x2, y2) = region[i], region[(i + 1) % n]
        if (x1 - x) * (x2 - x) <= 0 and x1 != x2:
            ys.append(y1 + (y2 - y1) * (x - x1) / (x2 - x1))
    if not ys:
        raise ValueError("x is outside the region")
    return float(min(ys) + frac * (max(ys) - min(ys)))


# ------------------------------------------------------------ ④ in-plane balance

def _route(rid, holds, start):
    return Route(rid, rid, "", holds, start, holds[-1].id)


def balance_figure() -> dict:
    """Data for the in-plane balance figure in #check: three panels, all checked by the solver.

    1. Hanging from one hand: the solver puts the centre of mass right below it.
    2. The same with the body held 0.2 m to the side: no forces balance it.
    3. A sidepull and a foot: the solver's forces, which close into a triangle with gravity.
    Also the strength example of unit 2.3, now with the solver choosing the forces.
    """
    from betasolver.stance import State, effort, moment, solve, torso_region

    UPTO = "inplane"  # the model as it stands in this section: constraints 1 to 4
    body = make_body(AVA)
    panels = []

    hang = _route("hang", [Hold("H1", 1.50, 2.30, "jug"), Hold("T", 1.50, 3.60, "jug")], ("H1", "H1"))
    s1 = State("H1", "H1")
    region = torso_region(body, hang, s1)
    r1 = solve(body, hang, s1, no_force="RH", upto=UPTO)
    if not r1["ok"]:
        raise ValueError("one-hand hang should hold")
    # hanging: draw the body low in its range, arms nearly straight
    torso1 = (r1["torso"][0], mid_y(region, r1["torso"][0], 0.03))
    shift = 0.20
    # same height if the arm still reaches, otherwise as low as it reaches
    torso2 = (torso1[0] + shift, max(torso1[1], mid_y(region, torso1[0] + shift, 0.03)))
    r2 = solve(body, hang, s1, no_force="RH", torso=torso2, upto=UPTO)
    if r2["ok"]:
        raise ValueError("a body to the side of a single hand must not balance")
    hold1 = hang.hold("H1")
    ends1 = {"LH": (hold1.x, hold1.y), "RH": (hold1.x, hold1.y)}
    for key, torso, res in (("hang", torso1, r1), ("off", torso2, None)):
        forces = res["forces"] if res else {"LH": {"f": (0.0, 1.0), "n": 0.0}}
        panels.append({
            "key": key, "ok": res is not None, "state": s1.as_dict(), "rest": "RH",
            "holds": [{**hang.to_dict()["holds"][0], "role": "normal"}],
            "torso": list(torso), "joints": body.joints(torso, ends1), "forces": forces,
            "share": res["share"] if res else None, "t": res["t_exact"] if res else None,
            "moment": moment(hang, s1, forces), "hold_x": hold1.x, "shift": torso[0] - hold1.x,
        })

    side = _route("side", [Hold("H1", 1.20, 2.05, "crimp"), Hold("H2", 1.90, 2.00, "sidepull", dir=0),
                           Hold("F1", 1.75, 1.00, "foot_big"), Hold("T", 1.50, 3.60, "jug")], ("H1", "H2"))
    s3 = State("H1", "H2", None, "F1")
    r3 = solve(body, side, s3, no_force="LH", upto=UPTO)
    if not r3["ok"]:
        raise ValueError("the sidepull example should hold")
    torso3 = (r3["torso"][0], mid_y(torso_region(body, side, s3), r3["torso"][0]))
    ends3 = {l: (side.hold(h).x, side.hold(h).y) for l, h in s3.as_dict().items() if h}
    panels.append({
        "key": "sidepull", "ok": True, "state": s3.as_dict(), "rest": "LH",
        "holds": [{**h, "role": "normal"} for h in side.to_dict()["holds"] if h["id"] != "T"],
        "torso": list(torso3), "joints": body.joints(torso3, ends3), "forces": r3["forces"],
        "share": r3["share"], "t": r3["t_exact"], "limiting": r3["limiting"],
        "moment": moment(side, s3, r3["forces"]),
    })

    # back to the strength example: same state, the solver now chooses the forces
    route, state = strength_route(), State("H1", "H2", "F1", "F2")
    again = []
    for key in ("A", "B", "C"):
        c = PROFILES[key]
        b = make_body(c)
        res = solve(b, route, state, upto=UPTO)
        again.append({"climber": key, "name": c.name, "t_fixed": effort(b, route, state, STRENGTH_FORCES)[0],
                      "t": res["t_exact"], "t_lp": res["t"], "share": res["share"], "limiting": res["limiting"]})

    return _round({
        "climber": {"key": AVA.key, "H": AVA.H, "head": list(body.head), "name": AVA.name},
        "panels": panels,
        "strength_again": again,
        "force_scale": 0.3,
        "view": {"x0": 1.0, "y0": 0.5, "x1": 2.15, "y1": 2.68},
    })


# ------------------------------------------------------------ ⑤ no tipping

def tipping_figure() -> dict:
    """Data for the no-tipping figure in #check.

    One hand on a jug, one foot below it, the other hand only touching
    (no_force). Two side views (a low foot and a high foot) and a sweep of the
    hand-foot height difference h, each solved by the state check, against
    the closed form N = d / h. Also the strength example once more, now with
    the hands pulling the body in.
    """
    from betasolver.stance import State, solve, torso_region

    body = make_body(AVA)
    hand = (1.50, 2.30)
    state = State("H1", "H1", "F1")

    def run(h):
        r = _route("tip", [Hold("H1", hand[0], hand[1], "jug"), Hold("F1", hand[0], hand[1] - h, "foot_big"),
                           Hold("T", 1.50, 3.60, "jug")], ("H1", "H1"))
        return r, solve(body, r, state, no_force="RH", upto="outofwall")

    views = []
    for h in (1.2, 0.6):
        r, res = run(h)
        if not res["ok"]:
            raise ValueError(f"tipping example h={h} should hold")
        torso = (res["torso"][0], mid_y(torso_region(body, r, state), res["torso"][0]))
        views.append({"h": h, "hand_y": hand[1], "foot_y": hand[1] - h, "torso": list(torso),
                      "N": res["forces"]["LH"]["n"], "M": res["forces"]["LF"]["m"],
                      "share": res["share"]["LH"], "formula": body.d_wall / h})
    sweep = []
    for i in range(12):
        h = round(0.4 + 0.1 * i, 2)
        r, res = run(h)
        if res["ok"]:
            sweep.append({"h": h, "N": res["forces"]["LH"]["n"], "formula": body.d_wall / h})
    err = max(abs(p["N"] - p["formula"]) / p["formula"] for p in sweep)

    route, s4 = strength_route(), State("H1", "H2", "F1", "F2")
    again = []
    for key in ("A", "B", "C"):
        c = PROFILES[key]
        b = make_body(c)
        before = solve(b, route, s4, upto="inplane")
        after = solve(b, route, s4, upto="outofwall")
        again.append({"climber": key, "name": c.name, "t_before": before["t_exact"], "t": after["t_exact"],
                      "pull": after["forces"]["LH"]["n"] + after["forces"]["RH"]["n"],
                      "share": after["share"], "limiting": after["limiting"]})

    return _round({
        "climber": {"key": AVA.key, "name": AVA.name, "H": AVA.H, "head": list(body.head),
                    "shoulder": body.offsets["LH"][1], "hip": body.offsets["LF"][1]},
        "d": body.d_wall,
        "views": views,
        "sweep": sweep,
        "max_error": err,
        "force_scale": 0.5,
        "strength_again": again,
    })


# ------------------------------------------------------------ ⑤ no barn-door

def _twist_needed(body, route, state, no_force):
    """Smallest grip torque that would hold the body, over every solution of the
    other constraints with t <= 1: min |sum p_x N - sum p_x M|."""
    import numpy as np
    from scipy.optimize import linprog

    from betasolver.model import HANDS
    from betasolver.stance import assemble, variable_bounds

    P = assemble(body, route, state, no_force, upto="outofwall")
    lay, n = P.layout, P.layout.size
    e = np.zeros(n)
    for limb in state.on():
        if limb == no_force:
            continue
        x = route.hold(state.hold(limb)).x
        if limb in HANDS:
            e[lay[("N", limb)].start] = x
        else:
            e[lay[("M", limb)].start] = -x
    G = np.hstack([P.G, np.zeros((len(P.G), 1))])
    extra = np.zeros((3, n + 1))
    extra[0, :n], extra[0, n] = e, -1.0
    extra[1, :n], extra[1, n] = -e, -1.0
    extra[2, lay[("t", None)].start] = 1.0
    G = np.vstack([G, extra])
    h = np.concatenate([P.h, [0.0, 0.0, 1.0]])
    A = np.hstack([P.A, np.zeros((len(P.A), 1))])
    c = np.zeros(n + 1)
    c[n] = 1.0
    res = linprog(c, A_ub=G, b_ub=h, A_eq=A, b_eq=P.b, bounds=variable_bounds(lay) + [(0.0, None)], method="highs")
    return float(res.x[n]) if res.status == 0 else None


def barndoor_figure() -> dict:
    """Data for the barn-door figure in #check: one hand, two feet, two foot placements.

    Left: the hand between the feet, solved by the state check. Right: both
    feet to the right of the hand. Every other constraint can still be met,
    but the twist the grip would need is more than it can give, so the state
    check fails.
    """
    from betasolver.stance import State, solve, strength_limit, torso_region

    body = make_body(AVA)
    state = State("H1", "H1", "F1", "F2")
    panels = []
    for key, feet in (("between", (1.22, 1.78)), ("side", (1.86, 2.14))):
        r = _route("door", [Hold("H1", 1.50, 2.20, "jug"), Hold("F1", feet[0], 1.00, "foot_big"),
                            Hold("F2", feet[1], 1.00, "foot_big"), Hold("T", 1.50, 3.60, "jug")], ("H1", "H1"))
        full = solve(body, r, state, no_force="RH")
        base = full if full["ok"] else solve(body, r, state, no_force="RH", upto="outofwall")
        if (key == "between") != full["ok"] or not base["ok"]:
            raise ValueError(f"barn-door example {key} does not behave as intended")
        torso = (base["torso"][0], mid_y(torso_region(body, r, state), base["torso"][0], 0.35))
        ends = {l: (r.hold(h).x, r.hold(h).y) for l, h in state.as_dict().items() if h}
        cap = HOLD_TYPES["jug"].r * strength_limit(body, r, state, "LH")
        panels.append({
            "key": key, "ok": full["ok"],
            "holds": [{**h, "role": "normal"} for h in r.to_dict()["holds"] if h["id"] != "T"],
            "state": state.as_dict(), "rest": "RH", "torso": list(torso), "joints": body.joints(torso, ends),
            "forces": base["forces"], "t": full["t_exact"] if full["ok"] else None,
            "tau": full["forces"]["LH"]["tau"] if full["ok"] else None,
            "needed": _twist_needed(body, r, state, "RH"), "cap": cap,
            "hand_x": 1.50, "feet_x": list(feet),
        })
    return _round({
        "climber": {"key": AVA.key, "name": AVA.name, "H": AVA.H, "head": list(body.head)},
        "r": HOLD_TYPES["jug"].r, "F": AVA.F["jug"],
        "panels": panels,
        "force_scale": 0.3,
        "view": {"x0": 0.95, "y0": 0.62, "x1": 2.5, "y1": 2.5},
    })


# ------------------------------------------------------------ ⑥ flexibility

def _y_range(region, x):
    lo, hi = mid_y(region, x, 0.0), mid_y(region, x, 1.0)
    return lo, hi


def flex_figure() -> dict:
    """Data for the flexibility figure in #check: one high foothold, Ava and a less flexible Ava.

    Flexible: the state check holds; the body is drawn in the middle of the
    heights that satisfy both reach and flexibility. Stiff: the state check
    fails; the body is drawn as tall as the other holds allow, and the foot
    is still above the limit.
    """
    from betasolver.model import VARIANTS
    from betasolver.stance import State, solve, torso_region

    r = _route("flex", [Hold("H1", 1.30, 2.25, "jug"), Hold("H2", 1.75, 2.30, "jug"), Hold("F1", 1.85, 1.45, "foot_big"),
                        Hold("F2", 1.25, 0.75, "foot_big"), Hold("T", 1.50, 3.60, "jug")], ("H1", "H2"))
    state = State("H1", "H2", "F2", "F1")
    ends = {l: (r.hold(h).x, r.hold(h).y) for l, h in state.as_dict().items()}
    high = r.hold("F1").y
    panels = []
    for c in (AVA, VARIANTS["A_stiff"]):
        body = make_body(c)
        full = solve(body, r, state)
        base = full if full["ok"] else solve(body, r, state, upto="barndoor")
        region = torso_region(body, r, state)
        x = base["torso"][0]
        lo, hi = _y_range(region, x)
        need = high - body.foot_limit((0.0, 0.0))  # lowest y_c that lets the foot reach its hold
        y = (max(lo, need) + hi) / 2 if full["ok"] else hi - 0.005
        torso = (x, y)
        panels.append({
            "climber": c.key, "name": c.name, "flex": c.flex, "H": c.H, "ok": full["ok"],
            "torso": [x, y], "joints": body.joints(torso, ends), "head": list(body.head),
            "hip": y + body.offsets["LF"][1], "limit": body.foot_limit(torso), "foot": high,
            "best_limit": body.foot_limit((x, hi)), "t": full["t_exact"] if full["ok"] else None,
        })
    return _round({
        "holds": [{**h, "role": "normal"} for h in r.to_dict()["holds"] if h["id"] != "T"],
        "state": state.as_dict(),
        "panels": panels,
        "view": {"x0": 0.85, "y0": 0.55, "x1": 2.25, "y1": 2.6},
    })


# ------------------------------------------------------------ the full problem: result cards

def result_cards() -> list:
    """A few preset states with the exact check's verdict: one that holds and one per failure reason."""
    from betasolver.model import VARIANTS
    from betasolver.stance import State, check

    cases = [
        ("holds", AVA, [Hold("H1", 1.25, 2.0, "jug"), Hold("H2", 1.75, 2.0, "jug"),
                        Hold("F1", 1.30, 0.90, "foot_big"), Hold("F2", 1.70, 0.95, "foot_big")],
         State("H1", "H2", "F1", "F2"), None,
         "Two jugs and two good footholds."),
        ("strength", CAL_, [Hold("H1", 1.30, 2.0, "crimp"), Hold("H2", 1.70, 2.0, "crimp")],
         State("H1", "H2"), None,
         "Hanging from two crimps with no feet."),
        ("reach", VARIANTS["A_stiff"], [Hold("H1", 1.30, 2.25, "jug"), Hold("H2", 1.75, 2.30, "jug"),
                                        Hold("F2", 1.25, 0.75, "foot_big"), Hold("F1", 1.85, 1.45, "foot_big")],
         State("H1", "H2", "F2", "F1"), None,
         "The high foothold of the flexibility figure."),
        ("balance", AVA, [Hold("H1", 1.60, 2.20, "sidepull", dir=0)],
         State("H1", "H1"), None,
         "Both hands on one sidepull, no feet."),
        ("unusable", PROFILES["D"], [Hold("H1", 1.25, 2.0, "jug"), Hold("H2", 1.75, 2.0, "crimp"),
                                     Hold("F1", 1.30, 0.90, "foot_big"), Hold("F2", 1.70, 0.95, "foot_big")],
         State("H1", "H2", "F1", "F2"), None,
         "A crimp, for a climber with an injured finger."),
    ]
    out = []
    for key, climber, holds, state, no_force, text in cases:
        r = _route("card", holds + [Hold("T", 1.5, 3.6, "jug")], (state.LH, state.RH))
        body = make_body(climber)
        res = check(body, r, state, no_force, exact=True)
        if (res.reason or "holds") != key:
            raise ValueError(f"card {key}: got {res.reason}")
        ends = {l: (r.hold(h).x, r.hold(h).y) for l, h in state.as_dict().items() if h}
        card = {"key": key, "text": text, "climber": climber.key, "name": climber.name, "head": list(body.head),
                "holds": [{**h, "role": "normal"} for h in r.to_dict()["holds"] if h["id"] != "T"],
                "state": state.as_dict(), "result": res.to_dict()}
        card["color"] = climber.key.split("_")[0]
        if res.torso is not None:
            # the body height is often free; draw it mid-way through the heights reach and flexibility allow
            from betasolver.stance import torso_region
            x = res.torso[0]
            lo, hi = _y_range(torso_region(body, r, state), x)
            feet = [r.hold(h).y for l, h in state.as_dict().items() if h and l in ("LF", "RF")]
            if feet:
                lo = max(lo, max(feet) - body.foot_limit((0.0, 0.0)))
            torso = (x, lo + (0.35 if feet else 0.06) * (hi - lo))  # hanging: arms nearly straight
            card["draw_torso"] = list(torso)
            card["joints"] = body.joints(torso, ends)
            card["share"] = None
        if res.limb:
            kind, _ = r.hold(res.hold).contact(res.limb)
            card["hold_type"] = kind.name
        if res.forces:
            from betasolver.stance import effort
            fx = {l: {k: (tuple(v) if isinstance(v, list) else v) for k, v in F.items()} for l, F in res.forces.items()}
            card["share"] = effort(body, r, state, fx)[1]
        out.append(card)
    return _round(out)


CAL_ = PROFILES["C"]



# ------------------------------------------------------------ the linear program against the exact one

def lp_figure() -> dict:
    """Data for the comparison in #check: 500 random states, LP against the exact cone program."""
    from betasolver.compare import compare, summary
    from betasolver.stance import REACH_SIDES, State, assemble, coverage

    records = compare()
    both = [r for r in records if r["lp_ok"] and r["exact_ok"]]
    pts = [[r["exact_t"], (r["lp_t"] - r["exact_t"]) / r["exact_t"]] for r in both if r["exact_t"] > 1e-9]
    P = assemble(make_body(AVA), _route("size", [Hold("H1", 1.25, 2.0, "jug"), Hold("H2", 1.75, 2.0, "jug"),
                                                  Hold("F1", 1.3, 0.9, "foot_big"), Hold("F2", 1.7, 0.95, "foot_big"),
                                                  Hold("T", 1.5, 3.6, "jug")], ("H1", "H2")),
                 State("H1", "H2", "F1", "F2"))
    import math
    return _round({
        "summary": summary(records),
        "points": pts,
        "reach_loss": 1 - math.cos(math.pi / REACH_SIDES),
        "strength_loss": 1 - coverage(),
        "size": {"variables": P.layout.size, "inequalities": len(P.h), "equalities": len(P.b)},
        "limit": 0.10,
    })
