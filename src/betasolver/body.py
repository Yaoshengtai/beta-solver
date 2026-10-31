"""The body model (step 2 of the plan).

The body is a torso that slides but never rotates, plus four limbs with a
fixed greatest length. The torso position x = (x_c, y_c) is also the centre
of mass. All proportions are multiples of the height H.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .model import FEET, HANDS, LIMBS, Climber

# Where the limbs attach, relative to the centre of mass, as multiples of H.
# The left side is negative x.
SHOULDER = (0.13, 0.26)
HIP = (0.095, -0.03)
HEAD = (0.37, 0.065)  # centre height above the centre of mass, radius
LEG = 0.50
GRIP_INSET = 0.03  # a hand grips with the middle of the palm, not the fingertips
SHOULDER_WIDTH_IN_SPAN = 0.259
# For drawing only: segment lengths of the upper arm and the thigh.
UPPER_ARM = 0.186
THIGH = 0.245


def arm_length(H: float, span: float) -> float:
    return (span - SHOULDER_WIDTH_IN_SPAN * H) / 2 - GRIP_INSET * H


def two_link(root, end, a, b, prefer):
    """Middle joint of a two-link chain from root to end, segment lengths a and b.

    `prefer(point)` scores the two mirror solutions and the higher one wins.
    If the end is out of reach the chain points straight at it.
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


@dataclass(frozen=True)
class Body:
    climber: Climber
    H: float
    offsets: dict  # limb -> (dx, dy) of its shoulder or hip from the centre of mass
    L: dict        # limb -> greatest length
    seg: dict      # limb -> (upper, lower) segment lengths, for drawing
    head: tuple    # (height above the centre of mass, radius)
    d_wall: float
    flex: float

    def anchor(self, limb: str, torso) -> tuple[float, float]:
        dx, dy = self.offsets[limb]
        return (torso[0] + dx, torso[1] + dy)

    def reaches(self, limb: str, torso, point, tol: float = 1e-9) -> bool:
        a = self.anchor(limb, torso)
        return math.hypot(point[0] - a[0], point[1] - a[1]) <= self.L[limb] + tol

    def foot_limit(self, torso) -> float:
        """Highest a foot can be: the hip height plus flex times H."""
        return torso[1] + HIP[1] * self.H + self.flex * self.H

    def joints(self, torso, ends: dict) -> dict:
        """Joint points for drawing, given the torso and each limb's end point.

        A foot that is not on a hold (end None) hangs straight down.
        Elbows bend down and out; knees bend out, as climbers do.
        """
        out = {}
        for limb in LIMBS:
            side = -1 if limb[0] == "L" else 1
            root = self.anchor(limb, torso)
            end = ends.get(limb)
            if end is None:
                if limb in HANDS:
                    raise ValueError("a hand is always on a hold")
                end = (root[0] + side * 0.05 * self.H, root[1] - 0.95 * self.L[limb])
            a, b = self.seg[limb]
            if limb in HANDS:
                mid = two_link(root, end, a, b, lambda p: -p[1] + 0.3 * side * p[0])
                out[limb] = {"shoulder": root, "elbow": mid, "end": tuple(end)}
            else:
                mid = two_link(root, end, a, b, lambda p: side * p[0])
                out[limb] = {"hip": root, "knee": mid, "end": tuple(end)}
        return out

    def to_dict(self) -> dict:
        return {
            "climber": self.climber.key,
            "H": self.H,
            "offsets": {k: list(v) for k, v in self.offsets.items()},
            "L": dict(self.L),
            "seg": {k: list(v) for k, v in self.seg.items()},
            "head": list(self.head),
            "d_wall": self.d_wall,
            "flex": self.flex,
        }


def make_body(climber: Climber) -> Body:
    H = climber.H
    arm, leg = arm_length(H, climber.span), LEG * H
    offsets, L, seg = {}, {}, {}
    for limb in LIMBS:
        side = -1 if limb[0] == "L" else 1
        if limb in HANDS:
            offsets[limb] = (side * SHOULDER[0] * H, SHOULDER[1] * H)
            L[limb] = arm
            seg[limb] = (UPPER_ARM * H, arm - UPPER_ARM * H)
        else:
            offsets[limb] = (side * HIP[0] * H, HIP[1] * H)
            L[limb] = leg
            seg[limb] = (THIGH * H, leg - THIGH * H)
    return Body(climber, H, offsets, L, seg, (HEAD[0] * H, HEAD[1] * H), climber.d_wall, climber.flex)


__all__ = ["Body", "make_body", "arm_length", "two_link", "FEET", "HANDS"]
