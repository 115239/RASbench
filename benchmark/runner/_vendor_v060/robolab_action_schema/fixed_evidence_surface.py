"""Rubric-independent evidence discovery and validation for RoboLab contests.

This module deliberately has no dependency on ``pxr`` or Isaac Sim.  The Runner
adapter reads the composed formal Stage and exposes only the USD facts defined by
``FormalStageFactsSource``.  Everything after that boundary is deterministic and
can be tested without a simulator.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import h5py
import numpy as np

from .v4_hdf5_schema import (
    DATA_ROOT,
    FORBIDDEN_RUNTIME_PRIM_PREFIX,
    FORMAL_ROBOT_ARTICULATION_PRIMS,
    SchemaValidationError,
    articulation_feedback_path,
    articulation_state_path,
    contact_field_path,
    gripper_closure_path,
    rigid_root_pose_path,
    rigid_root_velocity_path,
    task_reference_pose_path,
    validate_formal_submission_structure,
)
from .robot_profiles import FORMAL_ROBOT_PROFILES

FIXED_EVIDENCE_SURFACE_SCHEMA = "robolab-embodied-fixed-evidence-surface/v0.3"
FORMAL_ROBOT_AXIS_COUNTS = {
    role: (profile.joint_count, profile.body_count)
    for role, profile in FORMAL_ROBOT_PROFILES.items()
}
REVOLUTE_LIMIT_TOLERANCE_RAD = 1e-4
PRISMATIC_LIMIT_TOLERANCE_M = 1e-5


@dataclass(frozen=True)
class EvidenceSurfaceBudget:
    """Temporary implementation guardrails against accidental resource blow-up.

    These limits are not competition performance limits and must not be reported
    as such.  The deployment may pass an explicitly measured budget.
    """

    max_prim_facts: int = 10_000
    max_contact_pairs: int = 65_536
    max_sum_contact_bodies: int = 1_000_000

    def __post_init__(self) -> None:
        for name in (
            "max_prim_facts",
            "max_contact_pairs",
            "max_sum_contact_bodies",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")


# Temporary safety limits pending performance measurement on the 50 server.
# They are deliberately named so they cannot be mistaken for competition limits.
DEFAULT_IMPLEMENTATION_SAFETY_BUDGET_V01 = EvidenceSurfaceBudget()


@dataclass(frozen=True)
class JointCoordinateEvidence:
    """Formal USD coordinate contract for one movable HDF5 J-axis joint."""

    joint_path: str
    joint_type: str
    body0_path: str | None
    body1_path: str | None
    axis: str
    local_pos0_m: tuple[float, float, float]
    local_rot0_wxyz: tuple[float, float, float, float]
    local_pos1_m: tuple[float, float, float]
    local_rot1_wxyz: tuple[float, float, float, float]
    lower_limit: float | None
    upper_limit: float | None
    unit: str


@dataclass(frozen=True)
class FormalPrimFacts:
    """Simulator-neutral facts for one Prim in the composed formal Stage.

    ``articulation_root_path`` is the owning articulation resolved from USD
    physics joints or the frozen robot registry.  It may be omitted when normal
    Prim ancestry is sufficient.  Explicit ownership is allowed to be
    non-ancestral: the formal Pro asset places ArticulationRootAPI on
    ``base_link`` while many of its 36 bodies are sibling Prims.

    ``collision_owner_path`` is the owning entity for a CollisionAPI Prim.  The
    Stage adapter derives it from USD physics ownership, never from a Rubric or
    a hand-authored evidence marker.  It may identify an articulation, an
    independent rigid body, or a standalone static collision root.  When it is
    omitted, physical ancestry is used and an otherwise standalone collider
    owns itself.
    """

    path: str
    has_rigid_body_api: bool = False
    has_collision_api: bool = False
    has_articulation_root_api: bool = False
    is_movable_joint: bool = False
    articulation_root_path: str | None = None
    collision_owner_path: str | None = None
    rigid_body_enabled: bool = True
    kinematic_enabled: bool = False
    joint_coordinate: JointCoordinateEvidence | None = None


@runtime_checkable
class FormalStageFactsSource(Protocol):
    """Boundary that the real USD/Isaac Runner adapter must implement."""

    def iter_prim_facts(self) -> Iterable[FormalPrimFacts]:
        """Yield facts from the fully composed, formally namespaced Stage."""


@dataclass(frozen=True)
class ArticulationEvidence:
    prim_path: str
    joint_paths: tuple[str, ...]
    joints: tuple[JointCoordinateEvidence, ...]
    body_paths: tuple[str, ...]


@dataclass(frozen=True)
class ContactPairEvidence:
    source_prim: str
    target_prim: str
    source_body_paths: tuple[str, ...]

    @property
    def contact_axis_size(self) -> int:
        return len(self.source_body_paths)


@dataclass(frozen=True)
class FixedEvidenceSurface:
    """Complete evidence plan derived only from the formal Stage."""

    robot_role: str
    robot_articulation_prim: str
    task_id: str | None
    formal_stage_digest: str | None
    rigid_objects: tuple[str, ...]
    articulations: tuple[ArticulationEvidence, ...]
    task_references: tuple[str, ...]
    static_contact_targets: tuple[str, ...]
    gripper_closure_prims: tuple[str, ...]
    contact_pairs: tuple[ContactPairEvidence, ...]

    def canonical_audit_dict(self) -> dict[str, object]:
        """Return a canonical, JSON-safe plan for CI logs and audit only."""

        return {
            "schema": FIXED_EVIDENCE_SURFACE_SCHEMA,
            "task_id": self.task_id,
            "artifact_role": self.robot_role,
            "formal_stage_digest": self.formal_stage_digest,
            "robot_articulation_prim": self.robot_articulation_prim,
            "rigid_objects": list(self.rigid_objects),
            "articulations": [
                {
                    "prim_path": articulation.prim_path,
                    "joint_paths": list(articulation.joint_paths),
                    "joints": [
                        {
                            "joint_path": joint.joint_path,
                            "joint_type": joint.joint_type,
                            "body0_path": joint.body0_path,
                            "body1_path": joint.body1_path,
                            "axis": joint.axis,
                            "local_pos0_m": list(joint.local_pos0_m),
                            "local_rot0_wxyz": list(joint.local_rot0_wxyz),
                            "local_pos1_m": list(joint.local_pos1_m),
                            "local_rot1_wxyz": list(joint.local_rot1_wxyz),
                            "lower_limit": joint.lower_limit,
                            "upper_limit": joint.upper_limit,
                            "unit": joint.unit,
                        }
                        for joint in articulation.joints
                    ],
                    "body_paths": list(articulation.body_paths),
                }
                for articulation in self.articulations
            ],
            "task_references": list(self.task_references),
            "static_contact_targets": list(self.static_contact_targets),
            "gripper_closure_prims": list(self.gripper_closure_prims),
            "contact_pairs": [
                {
                    "source": pair.source_prim,
                    "target": pair.target_prim,
                    "contact_body_paths": list(pair.source_body_paths),
                }
                for pair in self.contact_pairs
            ],
        }

    def audit_sha256(self) -> str:
        """Hash the canonical audit plan; this is not an HDF5 runtime manifest."""

        encoded = json.dumps(
            self.canonical_audit_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def required_dataset_paths(self) -> tuple[str, ...]:
        """Return deterministic paths required by this fixed evidence surface."""

        paths: list[str] = [
            f"{DATA_ROOT}/actions",
            f"{DATA_ROOT}/time/step_index",
            f"{DATA_ROOT}/time/action_time_s",
            f"{DATA_ROOT}/time/state_time_s",
            f"{DATA_ROOT}/tracked_pose/position",
            f"{DATA_ROOT}/tracked_pose/orientation",
            f"{DATA_ROOT}/tracked_pose/linear_velocity",
            f"{DATA_ROOT}/tracked_pose/angular_velocity",
        ]
        for prim_path in self.rigid_objects:
            paths.extend(
                (
                    rigid_root_pose_path(prim_path, initial=True),
                    rigid_root_velocity_path(prim_path, initial=True),
                    rigid_root_pose_path(prim_path),
                    rigid_root_velocity_path(prim_path),
                )
            )
        for articulation in self.articulations:
            for field in ("root_pose", "root_velocity", "joint_position", "joint_velocity"):
                paths.append(articulation_state_path(articulation.prim_path, field, initial=True))
                paths.append(articulation_state_path(articulation.prim_path, field))
            for field in (
                "joint_position",
                "joint_velocity",
                "body_pose",
                "body_velocity",
                "step_index",
                "sim_time_s",
            ):
                paths.append(articulation_feedback_path(articulation.prim_path, field))
        paths.extend(task_reference_pose_path(path) for path in self.task_references)
        # Reuse the existing v1 static-pose path for automatically discovered
        # static collision targets; no new HDF5 path or task-asset marker is
        # introduced by the fixed-evidence profile.
        paths.extend(
            task_reference_pose_path(path) for path in self.static_contact_targets
        )
        paths.extend(gripper_closure_path(path) for path in self.gripper_closure_prims)
        for pair in self.contact_pairs:
            for field in ("net_forces_w", "step_index", "sim_time_s"):
                paths.append(contact_field_path(pair.source_prim, pair.target_prim, field))
        return tuple(paths)


@dataclass(frozen=True)
class FixedEvidenceValidationResult:
    robot_role: str
    control_steps: int
    rigid_object_count: int
    articulation_count: int
    task_reference_count: int
    static_contact_target_count: int
    gripper_closure_count: int
    contact_pair_count: int


def _metadata_text_attribute(
    metadata: h5py.Group,
    name: str,
) -> str:
    if name not in metadata.attrs:
        raise SchemaValidationError(
            f"fixed-evidence metadata is missing attribute: {name}"
        )
    value = metadata.attrs[name]
    if isinstance(value, np.bytes_):
        value = bytes(value)
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except UnicodeDecodeError as error:
            raise SchemaValidationError(
                f"fixed-evidence metadata {name} is not UTF-8"
            ) from error
    if not isinstance(value, str) or not value or value != value.strip():
        raise SchemaValidationError(
            f"fixed-evidence metadata {name} must be normalized non-empty text"
        )
    return value


def _metadata_sha256(metadata: h5py.Group, name: str) -> str:
    value = _metadata_text_attribute(metadata, name)
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise SchemaValidationError(
            f"fixed-evidence metadata {name} must be lowercase SHA-256 hex"
        )
    return value


def validate_fixed_evidence_metadata(
    hdf5_file: h5py.File,
    *,
    expected_task_id: str | None = None,
    expected_role: str | None = None,
    expected_surface_sha256: str | None = None,
    expected_formal_stage_digest: str | None = None,
) -> dict[str, str]:
    """Validate formal identity and fixed-evidence provenance metadata."""

    role = validate_formal_submission_structure(
        hdf5_file,
        expected_task_id=expected_task_id,
        expected_role=expected_role,
    )
    metadata = hdf5_file.get("/metadata")
    if not isinstance(metadata, h5py.Group):  # covered by the structure check
        raise SchemaValidationError("HDF5 is missing /metadata")
    artifact_role = _metadata_text_attribute(metadata, "artifact_role")
    if artifact_role != role:
        raise SchemaValidationError(
            "metadata.artifact_role conflicts with formal articulation identity"
        )
    surface_schema = _metadata_text_attribute(
        metadata,
        "fixed_evidence_surface_schema",
    )
    if surface_schema != FIXED_EVIDENCE_SURFACE_SCHEMA:
        raise SchemaValidationError(
            "metadata.fixed_evidence_surface_schema must be "
            f"{FIXED_EVIDENCE_SURFACE_SCHEMA!r}"
        )
    surface_sha256 = _metadata_sha256(
        metadata,
        "fixed_evidence_surface_sha256",
    )
    formal_stage_digest = _metadata_sha256(metadata, "formal_stage_digest")
    exporter_version = _metadata_text_attribute(metadata, "exporter_version")
    if expected_surface_sha256 is not None and surface_sha256 != expected_surface_sha256:
        raise SchemaValidationError(
            "metadata.fixed_evidence_surface_sha256 does not match the official surface"
        )
    if (
        expected_formal_stage_digest is not None
        and formal_stage_digest != expected_formal_stage_digest
    ):
        raise SchemaValidationError(
            "metadata.formal_stage_digest does not match the official Stage"
        )
    return {
        "artifact_role": role,
        "fixed_evidence_surface_schema": surface_schema,
        "fixed_evidence_surface_sha256": surface_sha256,
        "formal_stage_digest": formal_stage_digest,
        "exporter_version": exporter_version,
    }


def _normal_prim_path(path: str) -> str:
    if not isinstance(path, str) or not path.startswith("/") or path == "/":
        raise SchemaValidationError("formal Stage Prim path must be non-root and absolute")
    if "//" in path or any(part in {".", ".."} for part in path.split("/")):
        raise SchemaValidationError(f"formal Stage Prim path must be normalized: {path!r}")
    if _is_under(path, FORBIDDEN_RUNTIME_PRIM_PREFIX):
        raise SchemaValidationError(
            "formal Stage inventory uses forbidden Runner runtime namespace: " + path
        )
    return path


def _finite_tuple(value: object, length: int, field: str) -> tuple[float, ...]:
    if not isinstance(value, (tuple, list)) or len(value) != length:
        raise SchemaValidationError(f"{field} must contain exactly {length} numbers")
    result: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float, np.integer, np.floating)):
            raise SchemaValidationError(f"{field} must contain only finite numbers")
        number = float(item)
        if not np.isfinite(number):
            raise SchemaValidationError(f"{field} must contain only finite numbers")
        result.append(number)
    return tuple(result)


def _canonical_quaternion(value: object, field: str) -> tuple[float, float, float, float]:
    raw = np.asarray(_finite_tuple(value, 4, field), dtype=np.float64)
    norm = float(np.linalg.norm(raw))
    if norm <= 1e-12:
        raise SchemaValidationError(f"{field} must not be a zero quaternion")
    raw /= norm
    first_nonzero = next((item for item in raw if abs(float(item)) > 1e-15), 0.0)
    if first_nonzero < 0.0:
        raw *= -1.0
    return tuple(float(item) for item in raw)


def _finite_limit(value: object, field: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise SchemaValidationError(f"{field} must be finite or null")
    number = float(value)
    if not np.isfinite(number):
        raise SchemaValidationError(f"{field} must be finite or null")
    return number


def _validate_joint_coordinate(
    value: JointCoordinateEvidence,
    *,
    expected_path: str,
) -> JointCoordinateEvidence:
    if not isinstance(value, JointCoordinateEvidence):
        raise SchemaValidationError(
            f"movable joint lacks a JointCoordinateEvidence contract: {expected_path}"
        )
    joint_path = _normal_prim_path(value.joint_path)
    if joint_path != expected_path:
        raise SchemaValidationError(
            f"joint coordinate path differs from Prim fact: {joint_path} != {expected_path}"
        )
    if value.joint_type not in {"revolute", "prismatic"}:
        raise SchemaValidationError(
            f"unsupported movable joint type for {joint_path}: {value.joint_type!r}"
        )
    expected_unit = "rad" if value.joint_type == "revolute" else "m"
    if value.unit != expected_unit:
        raise SchemaValidationError(
            f"joint unit must be {expected_unit!r} for {joint_path}, got {value.unit!r}"
        )
    if value.axis not in {"X", "Y", "Z"}:
        raise SchemaValidationError(f"joint axis must be X, Y, or Z: {joint_path}")
    body0 = _normal_prim_path(value.body0_path) if value.body0_path is not None else None
    body1 = _normal_prim_path(value.body1_path) if value.body1_path is not None else None
    if body0 is None and body1 is None:
        raise SchemaValidationError(f"joint body0/body1 cannot both be null: {joint_path}")
    lower = _finite_limit(value.lower_limit, f"{joint_path}.lower_limit")
    upper = _finite_limit(value.upper_limit, f"{joint_path}.upper_limit")
    if lower is not None and upper is not None and lower > upper:
        raise SchemaValidationError(
            f"joint lower_limit exceeds upper_limit: {joint_path}: {lower} > {upper}"
        )
    return JointCoordinateEvidence(
        joint_path=joint_path,
        joint_type=value.joint_type,
        body0_path=body0,
        body1_path=body1,
        axis=value.axis,
        local_pos0_m=_finite_tuple(value.local_pos0_m, 3, f"{joint_path}.local_pos0_m"),
        local_rot0_wxyz=_canonical_quaternion(
            value.local_rot0_wxyz, f"{joint_path}.local_rot0_wxyz"
        ),
        local_pos1_m=_finite_tuple(value.local_pos1_m, 3, f"{joint_path}.local_pos1_m"),
        local_rot1_wxyz=_canonical_quaternion(
            value.local_rot1_wxyz, f"{joint_path}.local_rot1_wxyz"
        ),
        lower_limit=lower,
        upper_limit=upper,
        unit=value.unit,
    )


def _is_under(path: str, root: str) -> bool:
    return path == root or path.startswith(root.rstrip("/") + "/")


def _inferred_articulation_owner(path: str, roots: tuple[str, ...]) -> str | None:
    owners = [root for root in roots if _is_under(path, root)]
    return max(owners, key=len) if owners else None


def _validate_fact(fact: FormalPrimFacts) -> FormalPrimFacts:
    path = _normal_prim_path(fact.path)
    owner = fact.articulation_root_path
    if owner is not None:
        owner = _normal_prim_path(owner)
    collision_owner = fact.collision_owner_path
    if collision_owner is not None:
        collision_owner = _normal_prim_path(collision_owner)
        if not fact.has_collision_api:
            raise SchemaValidationError(
                f"collision_owner_path requires CollisionAPI: {path}"
            )
    if fact.is_movable_joint and fact.has_rigid_body_api:
        raise SchemaValidationError(f"Prim cannot be both a movable joint and rigid body: {path}")
    if fact.kinematic_enabled and not fact.has_rigid_body_api:
        raise SchemaValidationError(f"kinematic flag requires RigidBodyAPI: {path}")
    if fact.has_articulation_root_api and owner not in (None, path):
        raise SchemaValidationError(
            f"articulation root cannot belong to a different articulation: {path} -> {owner}"
        )
    joint_coordinate = fact.joint_coordinate
    if fact.is_movable_joint:
        joint_coordinate = _validate_joint_coordinate(
            joint_coordinate, expected_path=path  # type: ignore[arg-type]
        )
    elif joint_coordinate is not None:
        raise SchemaValidationError(
            f"joint coordinate contract requires a movable joint Prim: {path}"
        )
    for field in (
        "has_rigid_body_api",
        "has_collision_api",
        "has_articulation_root_api",
        "is_movable_joint",
        "rigid_body_enabled",
        "kinematic_enabled",
    ):
        if not isinstance(getattr(fact, field), bool):
            raise SchemaValidationError(
                f"formal Stage fact {field} must be bool: {path}"
            )
    return FormalPrimFacts(
        path=path,
        has_rigid_body_api=fact.has_rigid_body_api,
        has_collision_api=fact.has_collision_api,
        has_articulation_root_api=fact.has_articulation_root_api,
        is_movable_joint=fact.is_movable_joint,
        articulation_root_path=owner,
        collision_owner_path=collision_owner,
        rigid_body_enabled=fact.rigid_body_enabled,
        kinematic_enabled=fact.kinematic_enabled,
        joint_coordinate=joint_coordinate,
    )


def build_fixed_evidence_surface(
    prim_facts: Iterable[FormalPrimFacts],
    *,
    expected_robot_role: str | None = None,
    task_id: str | None = None,
    formal_stage_digest: str | None = None,
    gripper_closure_prims: Iterable[str] = (),
    enforce_formal_robot_axis_counts: bool = True,
    budget: EvidenceSurfaceBudget = DEFAULT_IMPLEMENTATION_SAFETY_BUDGET_V01,
) -> FixedEvidenceSurface:
    """Purely derive the shared fixed evidence surface from Stage facts.

    Contact enumeration is independent of the Rubric and needs no semantic
    marker or task-asset overlay.  Let ``A`` be every articulation, ``D`` every
    standalone rigid body, and ``K`` every standalone static collision owning
    root derived from USD collision ownership.  The fixed surface contains
    every ordered pair ``source -> target`` with ``source in A union D``,
    ``target in A union D union K``, and ``source != target``.  Articulation
    sources use their complete standard body ordering for C; independent rigid
    sources use exactly one C entry containing the source Prim itself.

    Non-physical business regions cannot be inferred from raw USD physics
    without guessing, so fixed profile v0.3 deliberately emits no ``Q`` task
    references.  Each automatically discovered ``K`` nevertheless writes a
    fixed pose through the existing HDF5 v1 ``task_reference`` path so surface
    predicates retain a compatible static-pose source.
    """

    if not isinstance(budget, EvidenceSurfaceBudget):
        raise TypeError("budget must be an EvidenceSurfaceBudget")
    if task_id is not None and (
        not isinstance(task_id, str) or not task_id.strip() or task_id != task_id.strip()
    ):
        raise SchemaValidationError("task_id must be a normalized non-empty string")
    if formal_stage_digest is not None and (
        not isinstance(formal_stage_digest, str)
        or len(formal_stage_digest) != 64
        or any(character not in "0123456789abcdef" for character in formal_stage_digest)
    ):
        raise SchemaValidationError(
            "formal_stage_digest must be a lowercase SHA-256 hex digest"
        )
    if not isinstance(enforce_formal_robot_axis_counts, bool):
        raise TypeError("enforce_formal_robot_axis_counts must be bool")
    collected: list[FormalPrimFacts] = []
    for fact in prim_facts:
        collected.append(_validate_fact(fact))
        if len(collected) > budget.max_prim_facts:
            raise SchemaValidationError(
                "fixed evidence surface exceeds max_prim_facts budget: "
                f">{budget.max_prim_facts}"
            )
    facts = tuple(collected)
    if not facts:
        raise SchemaValidationError("formal Stage inventory is empty")
    by_path: dict[str, FormalPrimFacts] = {}
    for fact in facts:
        if fact.path in by_path:
            raise SchemaValidationError(f"duplicate Prim in formal Stage inventory: {fact.path}")
        by_path[fact.path] = fact

    roots = tuple(sorted(fact.path for fact in facts if fact.has_articulation_root_api))
    nested_roots = [
        (parent, child)
        for child in roots
        for parent in roots
        if parent != child and _is_under(child, parent)
    ]
    if nested_roots:
        raise SchemaValidationError(
            "formal Stage contains nested articulation roots with ambiguous ownership: "
            + ", ".join(f"{parent} -> {child}" for parent, child in nested_roots)
        )
    formal_matches = [
        (role, prim_path)
        for role, prim_path in FORMAL_ROBOT_ARTICULATION_PRIMS.items()
        if prim_path in roots
    ]
    if len(formal_matches) != 1:
        raise SchemaValidationError(
            "formal Stage must contain exactly one registered competition robot articulation; "
            f"found {[role for role, _path in formal_matches] or 'none'}"
        )
    robot_role, robot_prim = formal_matches[0]
    if expected_robot_role is not None:
        if expected_robot_role not in FORMAL_ROBOT_ARTICULATION_PRIMS:
            raise SchemaValidationError(f"unsupported expected robot role: {expected_robot_role!r}")
        if expected_robot_role != robot_role:
            raise SchemaValidationError(
                f"formal Stage robot role mismatch: {robot_role!r} != {expected_robot_role!r}"
            )

    # Explicit physics/registry ownership is authoritative and need not follow
    # path ancestry.  This is required by the formal Pro asset, whose unique
    # ArticulationRootAPI Prim is a sibling of many linked rigid bodies.
    direct_body_owners: dict[str, str | None] = {}
    for fact in facts:
        if fact.articulation_root_path is not None:
            if fact.articulation_root_path not in roots:
                raise SchemaValidationError(
                    "Prim refers to an articulation root absent from the inventory: "
                    f"{fact.path} -> {fact.articulation_root_path}"
                )
            direct_owner = fact.articulation_root_path
        else:
            direct_owner = _inferred_articulation_owner(fact.path, roots)
        if fact.has_rigid_body_api:
            direct_body_owners[fact.path] = direct_owner

    owned_body_paths = tuple(
        sorted(
            path for path, owner in direct_body_owners.items() if owner is not None
        )
    )

    def articulation_owner(fact: FormalPrimFacts) -> str | None:
        if fact.articulation_root_path is not None:
            return fact.articulation_root_path
        ancestry_owner = _inferred_articulation_owner(fact.path, roots)
        if ancestry_owner is not None:
            return ancestry_owner
        body_ancestor = _inferred_articulation_owner(fact.path, owned_body_paths)
        if body_ancestor is not None:
            return direct_body_owners[body_ancestor]
        return None

    bodies: dict[str, list[str]] = {root: [] for root in roots}
    joints: dict[str, list[str]] = {root: [] for root in roots}
    joint_contracts: dict[str, list[JointCoordinateEvidence]] = {
        root: [] for root in roots
    }
    standalone_rigid: list[FormalPrimFacts] = []
    for fact in facts:
        owner = articulation_owner(fact)
        if owner is not None and owner not in bodies:
            raise SchemaValidationError(
                f"Prim refers to an articulation root absent from the inventory: {fact.path} -> {owner}"
            )
        if fact.has_rigid_body_api:
            if owner is None:
                standalone_rigid.append(fact)
            else:
                bodies[owner].append(fact.path)
        if fact.is_movable_joint:
            if owner is None:
                raise SchemaValidationError(f"movable joint has no owning articulation: {fact.path}")
            if fact.joint_coordinate is None:
                raise SchemaValidationError(
                    f"movable joint lacks a formal coordinate contract: {fact.path}"
                )
            joints[owner].append(fact.path)
            joint_contracts[owner].append(fact.joint_coordinate)

    rigid_objects = tuple(sorted(fact.path for fact in standalone_rigid))
    rigid_object_set = set(rigid_objects)

    def rigid_owner(path: str) -> str | None:
        owner = _inferred_articulation_owner(path, rigid_objects)
        return owner

    static_collision_owners: set[str] = set()
    for fact in facts:
        if not fact.has_collision_api:
            continue
        physical_articulation = articulation_owner(fact)
        physical_rigid = rigid_owner(fact.path)
        physical_owner = physical_articulation or physical_rigid
        declared_owner = fact.collision_owner_path
        if declared_owner is not None and declared_owner not in by_path:
            raise SchemaValidationError(
                "collision owner is absent from the formal Stage inventory: "
                f"{fact.path} -> {declared_owner}"
            )
        if physical_owner is not None:
            if declared_owner is not None and declared_owner != physical_owner:
                raise SchemaValidationError(
                    "collision owner conflicts with rigid/articulation ownership: "
                    f"{fact.path} -> {declared_owner}, expected {physical_owner}"
                )
            continue

        owner = declared_owner or fact.path
        if not _is_under(fact.path, owner):
            raise SchemaValidationError(
                "standalone static collision owner must be the collider or its ancestor: "
                f"{fact.path} -> {owner}"
            )
        owner_fact = by_path[owner]
        if (
            owner_fact.has_articulation_root_api
            or owner_fact.has_rigid_body_api
            or owner in rigid_object_set
        ):
            raise SchemaValidationError(
                "standalone static collision owner cannot be an articulation or rigid body: "
                + owner
            )
        static_collision_owners.add(owner)

    articulations: list[ArticulationEvidence] = []
    for root in roots:
        ordered_bodies = tuple(sorted(set(bodies[root])))
        ordered_joints = tuple(sorted(joint_contracts[root], key=lambda item: item.joint_path))
        ordered_joint_paths = tuple(item.joint_path for item in ordered_joints)
        if ordered_joint_paths != tuple(sorted(set(joints[root]))):
            raise SchemaValidationError(
                f"formal joint contracts do not cover the complete J axis: {root}"
            )
        if not ordered_bodies:
            raise SchemaValidationError(
                f"articulation inventory has no rigid bodies; Stage adapter is incomplete: {root}"
            )
        articulations.append(
            ArticulationEvidence(
                prim_path=root,
                joint_paths=ordered_joint_paths,
                joints=ordered_joints,
                body_paths=ordered_bodies,
            )
        )
    articulations.sort(key=lambda item: item.prim_path)

    robot_articulation = next(
        item for item in articulations if item.prim_path == robot_prim
    )
    expected_joint_count, expected_body_count = FORMAL_ROBOT_AXIS_COUNTS[robot_role]
    if enforce_formal_robot_axis_counts and (
        len(robot_articulation.joint_paths), len(robot_articulation.body_paths)
    ) != (expected_joint_count, expected_body_count):
        raise SchemaValidationError(
            "formal robot topology does not match the frozen J/B baseline: "
            f"{robot_role} has J/B={len(robot_articulation.joint_paths)}/"
            f"{len(robot_articulation.body_paths)}, expected "
            f"{expected_joint_count}/{expected_body_count}"
        )

    # Q is intentionally empty in the raw-USD-only profile: arbitrary business
    # Xforms are not evidence.  K comes solely from static collision ownership.
    task_references: tuple[str, ...] = ()
    static_contact_targets = tuple(sorted(static_collision_owners))
    normalized_grippers = tuple(
        sorted({_normal_prim_path(path) for path in gripper_closure_prims})
    )
    unknown_grippers = [path for path in normalized_grippers if path not in by_path]
    if unknown_grippers:
        raise SchemaValidationError(
            "gripper closure registry refers to Prim absent from the formal Stage: "
            + ", ".join(unknown_grippers)
        )
    for path in normalized_grippers:
        owner = articulation_owner(by_path[path])
        if owner != robot_prim:
            raise SchemaValidationError(
                "gripper closure Prim must belong to the formal robot articulation: "
                f"{path} -> {owner or 'none'}"
            )
    articulation_body_paths = {
        articulation.prim_path: articulation.body_paths for articulation in articulations
    }
    source_paths = tuple(sorted(set(roots) | set(rigid_objects)))
    target_paths = tuple(
        sorted(set(roots) | set(rigid_objects) | set(static_contact_targets))
    )
    pairs: dict[tuple[str, str], ContactPairEvidence] = {}
    for source in source_paths:
        source_bodies = articulation_body_paths.get(source, (source,))
        for target in target_paths:
            if source == target:
                continue
            key = (source, target)
            pairs[key] = ContactPairEvidence(source, target, source_bodies)

    if len(pairs) > budget.max_contact_pairs:
        raise SchemaValidationError(
            "fixed evidence surface exceeds max_contact_pairs budget: "
            f"{len(pairs)} > {budget.max_contact_pairs}"
        )
    sum_contact_bodies = sum(pair.contact_axis_size for pair in pairs.values())
    if sum_contact_bodies > budget.max_sum_contact_bodies:
        raise SchemaValidationError(
            "fixed evidence surface exceeds max_sum_contact_bodies budget: "
            f"{sum_contact_bodies} > {budget.max_sum_contact_bodies}"
        )

    return FixedEvidenceSurface(
        robot_role=robot_role,
        robot_articulation_prim=robot_prim,
        task_id=task_id,
        formal_stage_digest=formal_stage_digest,
        rigid_objects=rigid_objects,
        articulations=tuple(articulations),
        task_references=task_references,
        static_contact_targets=static_contact_targets,
        gripper_closure_prims=normalized_grippers,
        contact_pairs=tuple(pairs[key] for key in sorted(pairs)),
    )


def discover_fixed_evidence_surface(
    stage_source: FormalStageFactsSource,
    *,
    expected_robot_role: str | None = None,
    task_id: str | None = None,
    formal_stage_digest: str | None = None,
    gripper_closure_prims: Iterable[str] = (),
    enforce_formal_robot_axis_counts: bool = True,
    budget: EvidenceSurfaceBudget = DEFAULT_IMPLEMENTATION_SAFETY_BUDGET_V01,
) -> FixedEvidenceSurface:
    """Build a plan through the simulator-neutral formal Stage adapter."""

    if not isinstance(stage_source, FormalStageFactsSource):
        raise TypeError("stage_source must implement iter_prim_facts()")
    return build_fixed_evidence_surface(
        stage_source.iter_prim_facts(),
        expected_robot_role=expected_robot_role,
        task_id=task_id,
        formal_stage_digest=formal_stage_digest,
        gripper_closure_prims=gripper_closure_prims,
        enforce_formal_robot_axis_counts=enforce_formal_robot_axis_counts,
        budget=budget,
    )


def _require_dataset(
    hdf5_file: h5py.File,
    path: str,
    shape: tuple[int, ...],
    dtype: np.dtype | type[np.generic],
) -> h5py.Dataset:
    node = hdf5_file.get(path)
    if not isinstance(node, h5py.Dataset):
        raise SchemaValidationError(f"fixed evidence dataset is missing: {path}")
    if tuple(node.shape) != shape:
        raise SchemaValidationError(
            f"fixed evidence dataset has invalid shape: {path} {tuple(node.shape)}, expected {shape}"
        )
    expected_dtype = np.dtype(dtype)
    if node.dtype != expected_dtype:
        raise SchemaValidationError(
            f"fixed evidence dataset has invalid dtype: {path} {node.dtype}, expected {expected_dtype}"
        )
    values = np.asarray(node[...])
    if values.dtype.kind == "f" and not np.isfinite(values).all():
        raise SchemaValidationError(f"fixed evidence dataset contains NaN or infinity: {path}")
    return node


def _validate_step_index(
    node: h5py.Dataset,
    control_steps: int,
    *,
    continuous: bool,
    require_every_control_step: bool = False,
) -> None:
    values = np.asarray(node[...], dtype=np.uint64)
    if continuous:
        expected = np.arange(control_steps, dtype=np.uint64)
        if not np.array_equal(values, expected):
            raise SchemaValidationError(f"step_index must be 0..T-1: {node.name}")
    elif values.size:
        if np.any(values[1:] < values[:-1]) or int(values[-1]) >= control_steps:
            raise SchemaValidationError(
                f"stream step_index must be nondecreasing and within control steps: {node.name}"
            )
        if require_every_control_step and not np.array_equal(
            np.unique(values), np.arange(control_steps, dtype=np.uint64)
        ):
            raise SchemaValidationError(
                "contact stream must contain at least one sample for every control step: "
                + node.name
            )


def _validate_strict_time(node: h5py.Dataset) -> None:
    values = np.asarray(node[...], dtype=np.float64)
    if values.size > 1 and np.any(values[1:] <= values[:-1]):
        raise SchemaValidationError(f"time values must be strictly increasing: {node.name}")


def _validate_quaternions(node: h5py.Dataset, *, offset: int) -> None:
    values = np.asarray(node[...], dtype=np.float64)
    quaternions = values[..., offset : offset + 4]
    if quaternions.shape[-1] != 4 or np.any(
        np.linalg.norm(quaternions, axis=-1) <= 1e-12
    ):
        raise SchemaValidationError(f"pose contains a zero or missing quaternion: {node.name}")


def _validate_joint_axis_contract(articulation: ArticulationEvidence) -> None:
    if len(articulation.joint_paths) != len(articulation.joints):
        raise SchemaValidationError(
            "joint_paths and joints must have the same length: "
            f"{articulation.prim_path}"
        )
    contract_paths = tuple(item.joint_path for item in articulation.joints)
    if contract_paths != articulation.joint_paths:
        raise SchemaValidationError(
            "joint_paths and joints must use the same formal J ordering: "
            f"{articulation.prim_path}"
        )
    if articulation.joint_paths != tuple(sorted(set(articulation.joint_paths))):
        raise SchemaValidationError(
            f"formal J axis must be sorted and unique: {articulation.prim_path}"
        )


def _validate_joint_position_limits(
    node: h5py.Dataset,
    articulation: ArticulationEvidence,
) -> None:
    values = np.asarray(node[...], dtype=np.float64)
    if values.shape[-1:] != (len(articulation.joints),):
        raise SchemaValidationError(
            f"joint position J axis differs from its contract: {node.name}"
        )
    for index, joint in enumerate(articulation.joints):
        column = values[..., index]
        tolerance = (
            REVOLUTE_LIMIT_TOLERANCE_RAD
            if joint.joint_type == "revolute"
            else PRISMATIC_LIMIT_TOLERANCE_M
        )
        observed_min = float(np.min(column))
        observed_max = float(np.max(column))
        if joint.lower_limit is not None and observed_min < joint.lower_limit - tolerance:
            raise SchemaValidationError(
                "JOINT_VALUE_OUT_OF_FORMAL_LIMITS: "
                f"articulation={articulation.prim_path}, joint={joint.joint_path}, "
                f"index={index}, observed=[{observed_min},{observed_max}], "
                f"formal=[{joint.lower_limit},{joint.upper_limit}], unit={joint.unit}"
            )
        if joint.upper_limit is not None and observed_max > joint.upper_limit + tolerance:
            raise SchemaValidationError(
                "JOINT_VALUE_OUT_OF_FORMAL_LIMITS: "
                f"articulation={articulation.prim_path}, joint={joint.joint_path}, "
                f"index={index}, observed=[{observed_min},{observed_max}], "
                f"formal=[{joint.lower_limit},{joint.upper_limit}], unit={joint.unit}"
            )


def _child_names(hdf5_file: h5py.File, path: str) -> set[str]:
    node = hdf5_file.get(path)
    if node is None:
        return set()
    if not isinstance(node, h5py.Group):
        raise SchemaValidationError(f"fixed evidence root must be a group: {path}")
    return set(node.keys())


def _validate_no_extra_identity_groups(
    hdf5_file: h5py.File, surface: FixedEvidenceSurface
) -> None:
    """Reject physical-evidence aliases outside the preflight surface.

    Optional fields inside a legitimate identity group remain allowed.  What
    is forbidden is a second encoded entity or contact-pair identity that did
    not exist in the fixed Stage-derived plan.
    """

    from .v4_hdf5_schema import contact_pair_id, entity_id

    expected_rigids = {entity_id(path) for path in surface.rigid_objects}
    expected_articulations = {
        entity_id(item.prim_path) for item in surface.articulations
    }
    expected_references = {
        entity_id(path)
        for path in (*surface.task_references, *surface.static_contact_targets)
    }
    expected_grippers = {
        entity_id(path) for path in surface.gripper_closure_prims
    }
    expected_pairs = {
        contact_pair_id(pair.source_prim, pair.target_prim)
        for pair in surface.contact_pairs
    }
    expected_by_root = {
        f"{DATA_ROOT}/initial_state/rigid_object": expected_rigids,
        f"{DATA_ROOT}/states/rigid_object": expected_rigids,
        f"{DATA_ROOT}/initial_state/articulation": expected_articulations,
        f"{DATA_ROOT}/states/articulation": expected_articulations,
        f"{DATA_ROOT}/feedback/articulation": expected_articulations,
        f"{DATA_ROOT}/task_reference": expected_references,
        f"{DATA_ROOT}/states/gripper": expected_grippers,
        f"{DATA_ROOT}/sensors/contact": expected_pairs,
    }
    for root, expected in expected_by_root.items():
        actual = _child_names(hdf5_file, root)
        extras = sorted(actual - expected)
        if extras:
            raise SchemaValidationError(
                f"fixed evidence root contains unexpected identity groups: {root} {extras}"
            )


def validate_contact_c_axis(
    hdf5_file: h5py.File,
    pair: ContactPairEvidence,
) -> int:
    """Validate that contact C contains every standard source body, in full."""

    expected_c = pair.contact_axis_size
    if expected_c < 1:
        raise SchemaValidationError(
            f"contact source has an empty standard body axis: {pair.source_prim}"
        )
    path = contact_field_path(pair.source_prim, pair.target_prim, "net_forces_w")
    node = hdf5_file.get(path)
    if not isinstance(node, h5py.Dataset):
        raise SchemaValidationError(f"fixed contact dataset is missing: {path}")
    if node.ndim != 3 or node.shape[0] < 1 or tuple(node.shape[1:]) != (expected_c, 3):
        raise SchemaValidationError(
            "contact C axis must contain the complete standard source body ordering: "
            f"{path} has {tuple(node.shape)}, expected [S,{expected_c},3]"
        )
    if node.dtype != np.dtype(np.float32):
        raise SchemaValidationError(
            f"fixed contact dataset has invalid dtype: {path} {node.dtype}, expected float32"
        )
    if not np.isfinite(np.asarray(node[...])).all():
        raise SchemaValidationError(f"fixed contact dataset contains NaN or infinity: {path}")
    return expected_c


def validate_fixed_evidence_surface(
    hdf5_file: h5py.File,
    surface: FixedEvidenceSurface,
) -> FixedEvidenceValidationResult:
    """Validate an HDF5 artifact without reading or knowing any Rule Package."""

    fixed_metadata = validate_fixed_evidence_metadata(
        hdf5_file,
        expected_task_id=surface.task_id,
        expected_role=surface.robot_role,
        expected_surface_sha256=surface.audit_sha256(),
        expected_formal_stage_digest=surface.formal_stage_digest,
    )
    role = fixed_metadata["artifact_role"]
    if surface.robot_articulation_prim != FORMAL_ROBOT_ARTICULATION_PRIMS[role]:
        raise SchemaValidationError("fixed evidence surface robot identity is internally inconsistent")
    _validate_no_extra_identity_groups(hdf5_file, surface)

    position = hdf5_file[f"{DATA_ROOT}/tracked_pose/position"]
    control_steps = int(position.shape[0])
    if control_steps < 1:
        raise SchemaValidationError("fixed evidence HDF5 must contain at least one control step")
    _require_dataset(
        hdf5_file,
        f"{DATA_ROOT}/tracked_pose/position",
        (control_steps, 3),
        np.float32,
    )
    tracked_orientation = _require_dataset(
        hdf5_file,
        f"{DATA_ROOT}/tracked_pose/orientation",
        (control_steps, 4),
        np.float32,
    )
    _require_dataset(
        hdf5_file,
        f"{DATA_ROOT}/tracked_pose/linear_velocity",
        (control_steps, 3),
        np.float32,
    )
    _require_dataset(
        hdf5_file,
        f"{DATA_ROOT}/tracked_pose/angular_velocity",
        (control_steps, 3),
        np.float32,
    )
    _validate_quaternions(tracked_orientation, offset=0)
    actions = hdf5_file.get(f"{DATA_ROOT}/actions")
    if not isinstance(actions, h5py.Dataset):
        raise SchemaValidationError(f"fixed evidence dataset is missing: {DATA_ROOT}/actions")
    if actions.ndim != 2 or actions.shape[0] != control_steps or actions.shape[1] < 1:
        raise SchemaValidationError(
            f"actions must have shape [T,A] with A>=1: {actions.name} {tuple(actions.shape)}"
        )
    if actions.dtype != np.dtype(np.float32) or not np.isfinite(np.asarray(actions[...])).all():
        raise SchemaValidationError("actions must be finite float32 values")

    step_index = _require_dataset(
        hdf5_file, f"{DATA_ROOT}/time/step_index", (control_steps,), np.uint64
    )
    action_time = _require_dataset(
        hdf5_file, f"{DATA_ROOT}/time/action_time_s", (control_steps,), np.float64
    )
    state_time = _require_dataset(
        hdf5_file, f"{DATA_ROOT}/time/state_time_s", (control_steps,), np.float64
    )
    _validate_step_index(step_index, control_steps, continuous=True)
    _validate_strict_time(action_time)
    _validate_strict_time(state_time)

    for prim_path in surface.rigid_objects:
        initial_pose = _require_dataset(
            hdf5_file, rigid_root_pose_path(prim_path, initial=True), (7,), np.float32
        )
        _require_dataset(
            hdf5_file, rigid_root_velocity_path(prim_path, initial=True), (6,), np.float32
        )
        state_pose = _require_dataset(
            hdf5_file, rigid_root_pose_path(prim_path), (control_steps, 7), np.float32
        )
        _require_dataset(
            hdf5_file, rigid_root_velocity_path(prim_path), (control_steps, 6), np.float32
        )
        _validate_quaternions(initial_pose, offset=3)
        _validate_quaternions(state_pose, offset=3)

    for articulation in surface.articulations:
        _validate_joint_axis_contract(articulation)
        joint_count = len(articulation.joint_paths)
        body_count = len(articulation.body_paths)
        initial_shapes = {
            "root_pose": (7,),
            "root_velocity": (6,),
            "joint_position": (joint_count,),
            "joint_velocity": (joint_count,),
        }
        state_shapes = {
            "root_pose": (control_steps, 7),
            "root_velocity": (control_steps, 6),
            "joint_position": (control_steps, joint_count),
            "joint_velocity": (control_steps, joint_count),
        }
        for field, shape in initial_shapes.items():
            node = _require_dataset(
                hdf5_file,
                articulation_state_path(articulation.prim_path, field, initial=True),
                shape,
                np.float32,
            )
            if field == "root_pose":
                _validate_quaternions(node, offset=3)
            elif field == "joint_position":
                _validate_joint_position_limits(node, articulation)
        for field, shape in state_shapes.items():
            node = _require_dataset(
                hdf5_file,
                articulation_state_path(articulation.prim_path, field),
                shape,
                np.float32,
            )
            if field == "root_pose":
                _validate_quaternions(node, offset=3)
            elif field == "joint_position":
                _validate_joint_position_limits(node, articulation)

        feedback_step_path = articulation_feedback_path(articulation.prim_path, "step_index")
        feedback_step = hdf5_file.get(feedback_step_path)
        if not isinstance(feedback_step, h5py.Dataset) or feedback_step.ndim != 1 or feedback_step.shape[0] < 1:
            raise SchemaValidationError(f"fixed evidence stream is missing or empty: {feedback_step_path}")
        sample_count = int(feedback_step.shape[0])
        feedback_step = _require_dataset(
            hdf5_file, feedback_step_path, (sample_count,), np.uint64
        )
        feedback_time = _require_dataset(
            hdf5_file,
            articulation_feedback_path(articulation.prim_path, "sim_time_s"),
            (sample_count,),
            np.float64,
        )
        feedback_joint_position = _require_dataset(
            hdf5_file,
            articulation_feedback_path(articulation.prim_path, "joint_position"),
            (sample_count, joint_count),
            np.float32,
        )
        _validate_joint_position_limits(feedback_joint_position, articulation)
        _require_dataset(
            hdf5_file,
            articulation_feedback_path(articulation.prim_path, "joint_velocity"),
            (sample_count, joint_count),
            np.float32,
        )
        body_pose = _require_dataset(
            hdf5_file,
            articulation_feedback_path(articulation.prim_path, "body_pose"),
            (sample_count, body_count, 7),
            np.float32,
        )
        _require_dataset(
            hdf5_file,
            articulation_feedback_path(articulation.prim_path, "body_velocity"),
            (sample_count, body_count, 6),
            np.float32,
        )
        _validate_quaternions(body_pose, offset=3)
        _validate_step_index(feedback_step, control_steps, continuous=False)
        _validate_strict_time(feedback_time)
        optional_target = hdf5_file.get(
            articulation_feedback_path(articulation.prim_path, "joint_position_target")
        )
        if optional_target is not None:
            target = _require_dataset(
                hdf5_file,
                articulation_feedback_path(
                    articulation.prim_path, "joint_position_target"
                ),
                (sample_count, joint_count),
                np.float32,
            )
            _validate_joint_position_limits(target, articulation)

    for prim_path in surface.task_references:
        reference_pose = _require_dataset(
            hdf5_file, task_reference_pose_path(prim_path), (7,), np.float32
        )
        _validate_quaternions(reference_pose, offset=3)

    for prim_path in surface.static_contact_targets:
        static_pose = _require_dataset(
            hdf5_file, task_reference_pose_path(prim_path), (7,), np.float32
        )
        _validate_quaternions(static_pose, offset=3)

    for prim_path in surface.gripper_closure_prims:
        closure = _require_dataset(
            hdf5_file,
            gripper_closure_path(prim_path),
            (control_steps,),
            np.float32,
        )
        values = np.asarray(closure[...], dtype=np.float32)
        if np.any(values < 0.0) or np.any(values > 1.0):
            raise SchemaValidationError(
                f"gripper closure must be within [0,1]: {closure.name}"
            )

    for pair in surface.contact_pairs:
        validate_contact_c_axis(hdf5_file, pair)
        force_path = contact_field_path(pair.source_prim, pair.target_prim, "net_forces_w")
        sample_count = int(hdf5_file[force_path].shape[0])
        pair_step = _require_dataset(
            hdf5_file,
            contact_field_path(pair.source_prim, pair.target_prim, "step_index"),
            (sample_count,),
            np.uint64,
        )
        pair_time = _require_dataset(
            hdf5_file,
            contact_field_path(pair.source_prim, pair.target_prim, "sim_time_s"),
            (sample_count,),
            np.float64,
        )
        _validate_step_index(
            pair_step,
            control_steps,
            continuous=False,
            require_every_control_step=True,
        )
        _validate_strict_time(pair_time)

    return FixedEvidenceValidationResult(
        robot_role=role,
        control_steps=control_steps,
        rigid_object_count=len(surface.rigid_objects),
        articulation_count=len(surface.articulations),
        task_reference_count=len(surface.task_references),
        static_contact_target_count=len(surface.static_contact_targets),
        gripper_closure_count=len(surface.gripper_closure_prims),
        contact_pair_count=len(surface.contact_pairs),
    )
