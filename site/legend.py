"""Data for the legend figures in #appendix (unit 0.2).

The legend shows what every mark on the wall means. Its pose is illustrative:
it uses the body proportions from step 2 of the plan and a two-link inverse
kinematics, but its forces and efforts are example values, not solver output.
Once body.py exists (unit 1.3) the pose will come from there.
"""

from __future__ import annotations

import math

from betasolver.model import HOLD_TYPES, WALL_H, WALL_W, Hold, Route

WALL = {"w": WALL_W, "h": WALL_H}

# type, which limbs may use it, default direction (deg, 90 = up); a sidepull is drawn facing right
HOLD_TYPES_SHORT = [
    {"type": t.name, "use": "foot" if t.feet_only else "both", "dir": 0 if t.dir is None else t.dir}
    for t in HOLD_TYPES.values()
]

# Body proportions from step 2, as multiples of the height H.
SHOULDER = (0.13, 0.26)
HIP = (0.095, -0.03)
UPPER_ARM = 0.186
THIGH = 0.245
HEAD = (0.37, 0.065)  # centre height, radius
D_WALL = 0.15


def limb_lengths(H: float, span: float) -> dict:
    arm = (span - 0.259 * H) / 2 - 0.03 * H
    return {"arm": arm, "leg": 0.50 * H}


def two_link(root, end, a, b, prefer):
    """Middle joint of a two-link chain from root to end with segment lengths a, b.

    `prefer(point)` scores the two mirror solutions; the higher score wins.
    If the end is out of reach the chain is drawn straight toward it.
    """
    dx, dy = end[0] - root[0], end[1] - root[1]
    d = math.hypot(dx, dy)
    if d < 1e-9:
        return (root[0], root[1] - a)
    ux, uy = dx / d, dy / d
    if d >= a + b:
        return (root[0] + ux * a, root[1] + uy * a)
    c = max(-1.0, min(1.0, (a * a + d * d - b * b) / (2 * a * d)))
    th = math.acos(c)
    sols = []
    for s in (1, -1):
        cs, sn = math.cos(s * th), math.sin(s * th)
        sols.append((root[0] + a * (ux * cs - uy * sn), root[1] + a * (ux * sn + uy * cs)))
    return max(sols, key=prefer)


def pose(H: float, span: float, torso, ends: dict) -> dict:
    """Joint points for a torso position and the four limb end points."""
    L = limb_lengths(H, span)
    xc, yc = torso
    joints = {}
    for limb, end in ends.items():
        side = -1 if limb[0] == "L" else 1
        if limb[1] == "H":
            root = (xc + side * SHOULDER[0] * H, yc + SHOULDER[1] * H)
            a = UPPER_ARM * H
            b = L["arm"] - a
            # elbow: down first, then out
            mid = two_link(root, end, a, b, lambda p: -p[1] + 0.3 * side * p[0])
            joints[limb] = {"shoulder": root, "elbow": mid, "end": end}
        else:
            root = (xc + side * HIP[0] * H, yc + HIP[1] * H)
            a = THIGH * H
            b = L["leg"] - a
            # knee: out (frog legs)
            mid = two_link(root, end, a, b, lambda p: side * p[0])
            joints[limb] = {"hip": root, "knee": mid, "end": end}
    return {"torso": list(torso), "joints": joints}


def _round(obj):
    if isinstance(obj, float):
        return round(obj, 3)
    if isinstance(obj, (list, tuple)):
        return [_round(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _round(v) for k, v in obj.items()}
    return obj


def legend() -> dict:
    H, span = 1.60, 1.60
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
    frame = pose(H, span, torso, {k: at[v] for k, v in state.items()})
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
        "climber": {"key": "A", "H": H, "span": span, "d_wall": D_WALL * H,
                    "head": [HEAD[0] * H, HEAD[1] * H]},
        "scene": {"x0": 0.75, "y0": 0.45, "x1": 2.85, "y1": 3.05,
                  "holds": holds, "frame": frame, "trail": trail},
        "climbers": [{"key": "A", "name": "Ava"}, {"key": "B", "name": "Ben"},
                     {"key": "C", "name": "Cal"}, {"key": "D", "name": "Ava (injured)"}],
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
