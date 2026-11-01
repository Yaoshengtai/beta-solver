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
    the ball).
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
        torso = (float(sum(p[0] for p in region) / len(region)), float(sum(p[1] for p in region) / len(region)))
        t, share, limiting = effort(body, route, state, STRENGTH_FORCES)
        lp = solve(body, route, state, fixed=STRENGTH_FORCES)
        if not lp["ok"]:
            raise ValueError(f"the example forces break a constraint for {c.name}: {lp['message']}")
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
