"""An independent formulation of the state check in cvxpy, to cross-check `stance.solve`.

`stance` builds the second-order cone program by hand, row by row, and
hands it to Clarabel. Here the same problem is written the obvious way in
cvxpy, straight from the formula of section 3.7, and solved with the same
solver. Agreement between the two checks the hand-built matrices.
"""

from __future__ import annotations

import math

import numpy as np

from .model import FEET, HANDS
from .stance import cone, cone_normals, limb_sets, strength_limit, wall_distance


def solve_reference(body, route, state, no_force=None) -> dict:
    import cvxpy as cp

    A, AF = limb_sets(state, no_force)
    if not AF:
        return {"ok": False}
    x = cp.Variable(2)
    t = cp.Variable(nonneg=True)
    f = {l: cp.Variable(2) for l in AF}
    out = {l: cp.Variable(nonneg=True) for l in AF}  # N for hands, M for feet
    tau = {l: cp.Variable() for l in AF if l in HANDS}
    p = {l: route.hold(state.hold(l)) for l in A}
    cons = []
    for l in A:  # 1 reach
        o = body.offsets[l]
        cons.append(cp.norm(np.array([p[l].x - o[0], p[l].y - o[1]]) - x) <= body.L[l])
    for l in AF:  # 2 direction, 3 strength
        theta, alpha = cone(route, state, l)
        for n in cone_normals(theta, alpha):
            cons.append(n[0] * f[l][0] + n[1] * f[l][1] >= 0)
        cons.append(cp.norm(cp.hstack([f[l], out[l]])) <= t * strength_limit(body, route, state, l))
    cons.append(sum(f[l][0] for l in AF) == 0)  # 4 in-plane balance
    cons.append(sum(f[l][1] for l in AF) == 1)
    cons.append(sum(p[l].x * f[l][1] - p[l].y * f[l][0] for l in AF) == x[0])
    d = wall_distance(body, AF)  # 5 out-of-wall balance
    hands, feet = [l for l in AF if l in HANDS], [l for l in AF if l in FEET]
    cons.append(sum(out[l] for l in feet) - sum(out[l] for l in hands) == 0)
    cons.append(sum(p[l].y * out[l] for l in hands) - sum(p[l].y * out[l] for l in feet) == d)
    cons.append(sum(p[l].x * out[l] for l in hands) - sum(p[l].x * out[l] for l in feet) + sum(tau[l] for l in hands) == 0)
    for l in hands:
        kind, theta = p[l].contact(l)
        dk = (math.cos(math.radians(theta)), math.sin(math.radians(theta)))
        cons.append(out[l] <= kind.beta * (dk[0] * f[l][0] + dk[1] * f[l][1]))
        cons.append(cp.abs(tau[l]) <= kind.r * strength_limit(body, route, state, l))
    top = body.foot_limit((0.0, 0.0))  # 6 flexibility
    for l in A:
        if l in FEET:
            cons.append(p[l].y <= x[1] + top)
    prob = cp.Problem(cp.Minimize(t), cons)
    try:
        prob.solve(solver=cp.CLARABEL)
    except cp.error.SolverError:
        return {"ok": False}
    if prob.status not in ("optimal", "optimal_inaccurate"):
        return {"ok": False}
    return {"ok": True, "t": float(t.value), "torso": (float(x.value[0]), float(x.value[1]))}
