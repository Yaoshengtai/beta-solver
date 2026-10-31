"""Holds, routes and the wall (step 1 of the plan).

Lengths are in metres and forces in multiples of body weight (mg = 1). The
wall origin is its bottom-left corner, x to the right and y up. Angles are in
degrees, with 90 meaning straight up.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace

WALL_W = 3.0
WALL_H = 4.2

LIMBS = ("LH", "RH", "LF", "RF")
HANDS = ("LH", "RH")
FEET = ("LF", "RF")


@dataclass(frozen=True)
class HoldType:
    """What a kind of hold can do.

    dir     default centre of the directions the hold can push on a limb
    alpha   half-width of that range of directions
    grip    which of the climber's strength limits applies
    beta    pull-in factor: the hand's pull toward the wall is at most
            beta times the force along `dir` (hands only)
    r       grip torque arm in metres: how well the hand resists twisting
            (hands only)
    feet_only  True for footholds a hand cannot use
    """

    name: str
    dir: float | None
    alpha: float
    grip: str
    beta: float | None
    r: float | None
    feet_only: bool
    note: str


HOLD_TYPES: dict[str, HoldType] = {
    t.name: t
    for t in [
        HoldType("jug", 90, 70, "jug", 1.5, 0.05, False,
                 "Like a hook: it holds you in almost any direction."),
        HoldType("crimp", 90, 40, "crimp", 0.8, 0.02, False,
                 "A thin edge: pull down, within a moderate range."),
        HoldType("sloper", 90, 15, "sloper", 0.3, 0.0, False,
                 "Nothing to hook: the body must hang right below it."),
        HoldType("pinch", 90, 55, "pinch", 1.0, 0.04, False,
                 "Squeezed from both sides, so it holds a wide range."),
        HoldType("sidepull", None, 35, "pinch", 0.8, 0.03, False,
                 "Faces sideways: another limb has to push the other way."),
        HoldType("undercling", 270, 40, "crimp", 0.8, 0.02, False,
                 "Faces down: the hand pulls up, so a foot must push."),
        HoldType("foot_big", 90, 60, "foot", None, None, True,
                 "A big foothold to stand on."),
        HoldType("foot_chip", 90, 35, "foot", None, None, True,
                 "A small chip: the foot must push nearly straight down."),
    ]
}

# A foot on a hand hold is treated as if it stood on this type.
FOOT_ON_HANDHOLD = "foot_big"

USES = ("hand", "foot", "both")


@dataclass(frozen=True)
class Hold:
    id: str
    x: float
    y: float
    type: str
    dir: float | None = None
    use: str | None = None

    def __post_init__(self):
        if self.type not in HOLD_TYPES:
            raise ValueError(f"{self.id}: unknown hold type {self.type!r}")
        t = HOLD_TYPES[self.type]
        if self.dir is None:
            if t.dir is None:
                raise ValueError(f"{self.id}: a {self.type} needs an explicit dir (0 or 180)")
            object.__setattr__(self, "dir", float(t.dir))
        if self.type == "sidepull" and self.dir % 360 not in (0, 180):
            raise ValueError(f"{self.id}: a sidepull faces left or right (dir 0 or 180)")
        if self.use is None:
            object.__setattr__(self, "use", "foot" if t.feet_only else "both")
        if self.use not in USES:
            raise ValueError(f"{self.id}: use must be one of {USES}")
        if t.feet_only and self.use != "foot":
            raise ValueError(f"{self.id}: a {self.type} is a foothold only")

    @property
    def kind(self) -> HoldType:
        return HOLD_TYPES[self.type]

    def usable_by(self, limb: str, allow_feet_on_handholds: bool = True) -> bool:
        if limb in HANDS:
            return self.use in ("hand", "both")
        if self.use == "foot":
            return True
        return self.use == "both" and allow_feet_on_handholds

    def contact(self, limb: str) -> tuple[HoldType, float]:
        """The hold type and direction a limb actually uses on this hold.

        A foot on a hand hold uses the foothold direction and the foot's
        strength (step 1).
        """
        if limb in FEET and not self.kind.feet_only:
            t = HOLD_TYPES[FOOT_ON_HANDHOLD]
            return t, float(t.dir)
        return self.kind, float(self.dir)


@dataclass(frozen=True)
class Route:
    id: str
    title: str
    lesson: str
    holds: tuple[Hold, ...]
    start_hands: tuple[str, str]
    finish: str
    allow_feet_on_handholds: bool = True
    wall: tuple[float, float] = field(default=(WALL_W, WALL_H))

    def __post_init__(self):
        object.__setattr__(self, "holds", tuple(self.holds))
        object.__setattr__(self, "start_hands", tuple(self.start_hands))
        ids = [h.id for h in self.holds]
        if len(ids) != len(set(ids)):
            raise ValueError(f"{self.id}: hold ids must be unique")
        by_id = {h.id: h for h in self.holds}
        w, h = self.wall
        for hold in self.holds:
            if not (0 <= hold.x <= w and 0 <= hold.y <= h):
                raise ValueError(f"{self.id}: {hold.id} is off the wall")
        if len(self.start_hands) != 2:
            raise ValueError(f"{self.id}: two start holds are needed, one per hand")
        for hid in (*self.start_hands, self.finish):
            if hid not in by_id:
                raise ValueError(f"{self.id}: unknown hold {hid!r}")
            if not by_id[hid].usable_by("LH"):
                raise ValueError(f"{self.id}: {hid} must be a hand hold")

    def hold(self, hid: str) -> Hold:
        for h in self.holds:
            if h.id == hid:
                return h
        raise KeyError(hid)

    def role(self, hid: str) -> str:
        if hid in self.start_hands:
            return "start"
        if hid == self.finish:
            return "finish"
        return "normal"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "lesson": self.lesson,
            "wall": {"w": self.wall[0], "h": self.wall[1]},
            "holds": [{**asdict(h), "role": self.role(h.id)} for h in self.holds],
            "start_hands": list(self.start_hands),
            "finish": self.finish,
            "allow_feet_on_handholds": self.allow_feet_on_handholds,
        }


def hold_types_table() -> list[dict]:
    """The hold type table as plain data, in the order of step 1."""
    return [asdict(t) for t in HOLD_TYPES.values()]


# ---------------------------------------------------------------- climbers

GRIPS = ("jug", "crimp", "sloper", "pinch", "foot")

# Distance of the centre of mass from the wall, as a multiple of height (step 3.5).
D_WALL = 0.15


@dataclass(frozen=True)
class Climber:
    """A climber profile (step 1).

    H, span   height and arm span in metres
    F         most force one hand (or one foot) can take on each kind of hold,
              in body weights
    flex      how far a foot can go above the hip, as a multiple of H
    w_cross   style penalty for crossing the hands
    """

    key: str
    name: str
    H: float
    span: float
    F: dict
    flex: float
    w_cross: float = 0.15
    blurb: str = ""

    def __post_init__(self):
        if set(self.F) != set(GRIPS):
            raise ValueError(f"{self.key}: strengths must be given for {GRIPS}")
        if any(v < 0 for v in self.F.values()):
            raise ValueError(f"{self.key}: strengths cannot be negative")
        if not (1.0 < self.H < 2.5 and 1.0 < self.span < 2.6):
            raise ValueError(f"{self.key}: height and span look wrong")

    def strength(self, hold_type: str) -> float:
        """The limit that applies on a hold type, e.g. a sidepull uses the pinch limit."""
        return self.F[HOLD_TYPES[hold_type].grip]

    def can_use(self, hold_type: str) -> bool:
        """A hold whose strength limit is zero cannot be used at all."""
        return self.strength(hold_type) > 0

    @property
    def d_wall(self) -> float:
        return D_WALL * self.H

    def variant(self, key: str, name: str, blurb: str = "", **changes) -> "Climber":
        """The same climber with one attribute changed, as used in the examples."""
        if "F" in changes:
            changes["F"] = {**self.F, **changes["F"]}
        return replace(self, key=key, name=name, blurb=blurb or self.blurb, **changes)

    def to_dict(self) -> dict:
        return {**asdict(self), "d_wall": self.d_wall}


def _F(jug, crimp, sloper, pinch, foot):
    return {"jug": jug, "crimp": crimp, "sloper": sloper, "pinch": pinch, "foot": foot}


AVA = Climber("A", "Ava", 1.60, 1.60, _F(1.2, 0.9, 0.35, 0.6, 1.5), +0.08,
              blurb="Short, with strong fingers and flexible hips.")
BEN = Climber("B", "Ben", 1.85, 1.90, _F(1.2, 0.6, 0.55, 0.6, 1.5), -0.10,
              blurb="Tall with a long reach; weaker on crimps, better on slopers, stiff hips.")
CAL = Climber("C", "Cal", 1.70, 1.70, _F(0.8, 0.35, 0.25, 0.35, 1.5), 0.00,
              blurb="A beginner: average size, a weaker grip on everything.")
AVA_INJURED = AVA.variant("D", "Ava (injured)", F={"crimp": 0.0},
                          blurb="Ava with an injured finger: she cannot pull on crimps at all.")

PROFILES: dict[str, Climber] = {c.key: c for c in (AVA, BEN, CAL, AVA_INJURED)}

# Variants that change one attribute of a standard profile (step 7).
VARIANTS: dict[str, Climber] = {
    c.key: c
    for c in (
        AVA.variant("A_stiff", "Ava (less flexible)", flex=-0.10,
                    blurb="Ava with Ben's hip flexibility. Used in example 4."),
        AVA.variant("A_cross0", "Ava (crosses freely)", w_cross=0.0,
                    blurb="Ava with no dislike of crossing hands. Used in example 7."),
        AVA.variant("A_cross4", "Ava (avoids crossing)", w_cross=0.4,
                    blurb="Ava with a strong dislike of crossing hands. Used in example 7."),
    )
}
