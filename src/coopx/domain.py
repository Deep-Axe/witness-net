"""Observation semantics shared by perception, transport and fusion."""

from dataclasses import dataclass
from enum import IntEnum
import math

Vec3 = tuple[float, float, float]


class ObjectClass(IntEnum):
    UNKNOWN = 0
    PEDESTRIAN = 1
    VEHICLE = 2
    CYCLIST = 3


class PositionAnchor(IntEnum):
    SURFACE = 0
    CENTRE = 1
    GROUND = 2


def uint(value: int, bits: int, name: str) -> None:
    if type(value) is not int or not 0 <= value < 2**bits:
        raise ValueError(f"{name} must be an unsigned {bits}-bit integer")


def vector(value: Vec3, name: str, *, nonnegative: bool = False) -> None:
    if len(value) != 3 or not all(math.isfinite(v) for v in value):
        raise ValueError(f"{name} must have three finite components")
    if nonnegative and any(v < 0 for v in value):
        raise ValueError(f"{name} must have nonnegative components")


@dataclass(frozen=True, slots=True)
class Observation:
    """Metres/seconds in the envelope's spatial frame; replay time in µs.

    Dataset adapters establish axes explicitly. Legacy synthetic examples use
    CARLA x-forward/y-right/z-up; V2X-Real uses its calibrated site_map frame.

    source is configured identity, never authenticated by this object alone.
    object_id is a source-local uint32 allocated by the local tracker.
    sequence identifies this observation update, not each retransmission.
    position_std_m is per-axis one-sigma *estimated measurement* uncertainty.
    Surface position must not silently become an object centre downstream.
    """

    source: str
    object_id: int
    sequence: int
    captured_sim_us: int
    cls: ObjectClass
    xyz_m: Vec3
    anchor: PositionAnchor = PositionAnchor.SURFACE
    velocity_mps: Vec3 | None = None
    dimensions_m: Vec3 | None = None  # length, width, height
    yaw_deg: float | None = None  # spatial-frame yaw convention, [-180, 180]
    confidence: float | None = None  # detector score, not calibrated trust
    position_std_m: Vec3 | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("source must be nonempty")
        uint(self.object_id, 32, "object_id")
        uint(self.sequence, 32, "sequence")
        uint(self.captured_sim_us, 64, "captured_sim_us")
        if not isinstance(self.cls, ObjectClass):
            raise ValueError("cls must be an ObjectClass")
        if not isinstance(self.anchor, PositionAnchor):
            raise ValueError("anchor must be a PositionAnchor")
        vector(self.xyz_m, "xyz_m")
        for name in ("velocity_mps", "dimensions_m", "position_std_m"):
            value = getattr(self, name)
            if value is not None:
                vector(value, name, nonnegative=name != "velocity_mps")
        if self.dimensions_m is not None and any(v <= 0 for v in self.dimensions_m):
            raise ValueError("dimensions must be strictly positive")
        if self.yaw_deg is not None and not -180 <= self.yaw_deg <= 180:
            raise ValueError("yaw_deg must be finite and in [-180, 180]")
        if self.confidence is not None and not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be finite and in [0, 1]")

    @property
    def key(self) -> tuple[str, int]:
        return self.source, self.object_id

    def is_fresh(self, now_sim_us: int, ttl_us: int) -> bool:
        if ttl_us < 0:
            raise ValueError("ttl_us must be nonnegative")
        return 0 <= now_sim_us - self.captured_sim_us <= ttl_us
