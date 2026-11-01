"""Exporting a beta for the website (steps 5 and 8 of the plan).

Pose smoothing: the solver may put the body anywhere in the region of a
state where t is optimal, so consecutive frames can jump about. Each
keyframe of the beta is solved again with t at most its own optimum plus
SLACK, now minimising the L1 distance to the previous frame's body
position (the first frame aims at a natural height). This only changes how
the beta looks, never which beta it is.

Export: every step has four frames (start, release, catch, end) with the
state, body position, forces, each limb's share of its strength and the
joint points for drawing.
"""

from __future__ import annotations

import numpy as np

from .body import Body, make_body
from .model import FEET, HANDS, Climber, Route
from .moves import Move
from .search import Beta
from .stance import (LIMB_NAMES, State, _normal, assemble, effort, run, satisfies, torso_region)

SLACK = 0.05
PHASES = ("start", "release", "catch", "end")


def _y_mid(poly, x, frac):
    xs = [p[0] for p in poly]
    x = min(max(x, min(xs) + 1e-6), max(xs) - 1e-6)
    ys = []
    n = len(poly)
    for i in range(n):
        (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % n]
        if (x1 - x) * (x2 - x) <= 0 and x1 != x2:
            ys.append(y1 + (y2 - y1) * (x - x1) / (x2 - x1))
    return min(ys) + frac * (max(ys) - min(ys)) if ys else None


def natural_torso(body: Body, route: Route, state: State, x: float) -> tuple:
    """A natural-looking body position at a given x_c: a third of the way up the heights that reach
    and flexibility allow (the body hangs a little, arms nearly straight)."""
    poly = torso_region(body, route, state)
    lo, hi = _y_mid(poly, x, 0.0), _y_mid(poly, x, 1.0)
    feet = [route.hold(h).y for l, h in state.as_dict().items() if h and l in FEET]
    if feet:
        lo = max(lo, max(feet) - body.foot_limit((0.0, 0.0)))
    return (x, lo + 0.35 * (hi - lo))


def smooth_frame(body: Body, route: Route, state: State, no_force, t_star: float, target, slack: float = SLACK):
    """Re-solve one keyframe: t <= t_star + slack, body as close as possible (L1) to `target`.

    A tiny second term keeps the forces tidy (no needless squeezing or
    twisting). Returns (torso, forces, t) or None if the solve fails.
    """
    P = assemble(body, route, state, no_force).add_columns(2)
    lay, n = P.layout, P.n - 2
    sx, st = lay[("x", None)].start, lay[("t", None)].start
    for i in range(2):
        for sign in (1.0, -1.0):
            row = np.zeros(P.n)
            row[sx + i], row[n + i] = sign, -1.0
            P.add_ineq(row, sign * target[i], [("smooth", None)])
    row = np.zeros(P.n)
    row[st] = 1.0
    P.add_ineq(row, t_star + slack, [("smooth", None)])
    c = np.zeros(P.n)
    c[n], c[n + 1] = 1.0, 1.0
    eps = 1e-3
    for (name, limb), sl in lay.index.items():
        if name in ("N", "M"):
            c[sl.start] = eps
    z = run(P, c)
    if z is None:
        return None
    AF = [l for l in state.on() if l != no_force]
    forces = {}
    for limb in AF:
        f = z[lay[("f", limb)]]
        forces[limb] = {"f": (float(f[0]), float(f[1])), ("n" if limb in HANDS else "m"): max(0.0, float(z[lay[(_normal(limb), limb)].start]))}
        if limb in HANDS:
            forces[limb]["tau"] = float(z[lay[("tau", limb)].start])
    t_exact, share, limiting = effort(body, route, state, forces) if forces else (0.0, {}, None)
    return (float(z[sx]), float(z[sx + 1])), forces, t_exact, share, limiting


def frames_of(body: Body, route: Route, move: Move, res, prev):
    """The four keyframes of one step, smoothed, chaining from `prev` (a body position or None)."""
    out = []
    plan = [("start", move.state, None, res.start), ("release", move.state, move.limb, res.release),
            ("catch", move.new_state, move.limb, res.catch), ("end", move.new_state, None, res.end)]
    last = None
    for phase, state, nf, r in plan:
        if r is None:  # a cut has no catch, a place no release: copy the neighbouring frame
            out.append(None)
            continue
        target = prev if prev is not None else natural_torso(body, route, state, r.torso[0])
        sm = smooth_frame(body, route, state, nf, r.t, target)
        if sm is None:  # never expected; fall back to the solver's own frame
            torso, forces, t, share, lim = tuple(r.torso), {k: {kk: (tuple(vv) if isinstance(vv, list) else vv) for kk, vv in v.items()} for k, v in r.forces.items()}, r.t, None, r.limiting
            share = effort(body, route, state, forces)[1] if forces else {}
        else:
            torso, forces, t, share, lim = sm
        prev = torso
        out.append({"phase": phase, "state": state.as_dict(), "rest": nf, "torso": torso, "forces": forces,
                    "t": t, "t_check": r.t, "util": share, "limiting": lim})
    for i, f in enumerate(out):  # fill the missing frame from its neighbour
        if f is None:
            src = out[i - 1] if i > 0 and out[i - 1] is not None else out[i + 1]
            out[i] = {**src, "phase": PHASES[i]}
    return out, prev


def _joints(body: Body, route: Route, frame: dict) -> dict:
    ends = {l: (route.hold(h).x, route.hold(h).y) for l, h in frame["state"].items() if h}
    return body.joints(frame["torso"], ends)


def _r(obj, nd=3):
    if isinstance(obj, float):
        return round(obj, nd)
    if isinstance(obj, (list, tuple)):
        return [_r(v, nd) for v in obj]
    if isinstance(obj, dict):
        return {k: _r(v, nd) for k, v in obj.items()}
    return obj


def export_run(route: Route, climber: Climber, beta: Beta, label: str | None = None) -> dict:
    """One climber's beta on one route, in the format of step 8."""
    body = make_body(climber)
    run_ = {"climber": climber.key.split("_")[0], "key": climber.key, "label": label or climber.name,
            "status": beta.status, "crux": beta.crux, "total": beta.total, "crux_step": beta.crux_step,
            "fail": beta.fail, "start": beta.start.as_dict() if beta.start else None,
            "d_wall": body.d_wall, "head": list(body.head), "steps": [], "stats": beta.stats}
    prev = None
    for i, (move, res, w) in enumerate(beta.steps):
        frames, prev = frames_of(body, route, move, res, prev)
        for f in frames:
            f["joints"] = _joints(body, route, f)
        run_["steps"].append({
            "text": move.describe(route), "limb": move.limb, "kind": move.kind, "from": move.frm, "to": move.to,
            "swing": move.swing, "t_release": res.t_release, "t_catch": res.t_catch, "t_end": res.end.t,
            "difficulty": res.difficulty, "cost": w,
            "limiting": frames[1]["limiting"] if move.frm is not None else frames[2]["limiting"],
            "frames": frames,
        })
    return _r(run_)


def smoothing_check(route: Route, climber: Climber, beta: Beta) -> dict:
    """Smoothing must not change the beta: every smoothed frame is a valid solution of its own
    problem with t <= its optimum + SLACK. Returns the worst excess and whether all are valid."""
    from .stance import vector
    body = make_body(climber)
    worst, valid, prev = -np.inf, True, None
    for move, res, w in beta.steps:
        frames, prev = frames_of(body, route, move, res, prev)
        for f, (state, nf) in zip(frames, [(move.state, None), (move.state, move.limb), (move.new_state, move.limb), (move.new_state, None)]):
            if f["phase"] == "release" and move.frm is None or f["phase"] == "catch" and move.to is None:
                continue
            P = assemble(body, route, state, nf)
            z = vector(P.layout, f["torso"], f["forces"], f["t"])
            valid = valid and satisfies(P, z, tol=1e-5)
            worst = max(worst, f["t"] - (f["t_check"] + SLACK))
    return {"valid": valid, "worst": float(worst)}
