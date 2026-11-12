"""Frozen formal robot identities shared by Generator, Runner, and Judge."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True)
class FormalRobotProfile:
    """One competition robot's stable formal-stage identity and topology."""

    role: str
    formal_mount_prim: str
    articulation_prim: str
    joint_count: int
    body_count: int
    asset_sha256: str | None = None
    gripper_closure_prims: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.role or self.role != self.role.strip():
            raise ValueError("robot role must be normalized non-empty text")
        for name in ("formal_mount_prim", "articulation_prim"):
            value = getattr(self, name)
            if not value.startswith("/") or value == "/" or value.endswith("/"):
                raise ValueError(f"{name} must be a non-root absolute Prim path")
        for name in ("joint_count", "body_count"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.asset_sha256 is not None and (
            len(self.asset_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.asset_sha256)
        ):
            raise ValueError("asset_sha256 must be lowercase SHA-256 hex")


_PROFILES = {
    "r1_pro": FormalRobotProfile(
        role="r1_pro",
        formal_mount_prim="/World/R1Pro/Asset",
        articulation_prim="/World/R1Pro/Asset/r1_pro_with_gripper/base_link",
        joint_count=28,
        body_count=36,
    ),
    "r1_lite": FormalRobotProfile(
        role="r1_lite",
        formal_mount_prim="/World/R1Lite",
        articulation_prim="/World/R1Lite",
        joint_count=25,
        body_count=34,
    ),
    "m20_piper": FormalRobotProfile(
        role="m20_piper",
        formal_mount_prim="/M20PiperVLA/M20_Piper",
        articulation_prim="/M20PiperVLA/M20_Piper/M20/base_link",
        joint_count=24,
        body_count=27,
        asset_sha256=(
            "e645194e0ecdceb3aa5e82747ec695483e646b23e85dde7c87334765ffaec5d7"
        ),
        gripper_closure_prims=(
            "/M20PiperVLA/M20_Piper/Piper/gripper_base",
        ),
    ),
}

FORMAL_ROBOT_PROFILES = MappingProxyType(_PROFILES)


def get_formal_robot_profile(role: str) -> FormalRobotProfile:
    """Return a frozen profile, rejecting free-form or unsupported roles."""

    try:
        return FORMAL_ROBOT_PROFILES[role]
    except (KeyError, TypeError) as error:
        raise ValueError(f"unsupported formal robot role: {role!r}") from error


__all__ = [
    "FORMAL_ROBOT_PROFILES",
    "FormalRobotProfile",
    "get_formal_robot_profile",
]
