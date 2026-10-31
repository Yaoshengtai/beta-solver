"""Holds, routes and the wall (step 1 of the plan).

Lengths are in metres and forces in multiples of body weight (mg = 1). The
wall origin is its bottom-left corner, x to the right and y up. Angles are in
degrees, with 90 meaning straight up.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

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
