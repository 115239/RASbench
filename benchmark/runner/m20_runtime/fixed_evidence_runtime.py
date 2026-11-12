"""Runtime adapter for the embodied fixed evidence surface.

The adapter derives A/D/K from the read-only composed USD Stage and samples all
required state through PhysX tensor views.  Pair-wise contacts are read from a
RigidContactView, so the fixed directed contact matrix does not depend on a
Python contact-event callback.  The state axis keeps every enabled registered
rigid object; task tracked Prim paths bound the otherwise unbounded scene-contact
matrix without consulting a Rule Package or inventing scoring outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Iterable

import numpy as np

from robolab_action_schema import (
    EvidenceSurfaceBudget,
    FormalPrimFacts,
    FixedEvidenceSurface,
    JointCoordinateEvidence,
    discover_fixed_evidence_surface,
    get_formal_robot_profile,
)


RUNTIME_ENV_ROOT = "/World/envs/env_0"
RUNTIME_ROBOT_ROOT = f"{RUNTIME_ENV_ROOT}/robot"
RUNTIME_SCENE_ROOT = f"{RUNTIME_ENV_ROOT}/scene"
FORMAL_ROBOT_MOUNTS = {
    "r1_lite": "/World/R1Lite",
    "r1_pro": "/World/R1Pro/Asset",
    "m20_piper": "/M20PiperVLA/M20_Piper",
}

TASK_SCOPED_DISCOVERY_BUDGET = EvidenceSurfaceBudget(
    max_prim_facts=10_000,
    max_contact_pairs=1_000_000,
    max_sum_contact_bodies=10_000_000,
)


def _path(value: Any) -> str:
    return str(value).rstrip("/") or "/"


def _is_under(path: str, root: str) -> bool:
    return path == root or path.startswith(root.rstrip("/") + "/")


def _closest_owner(path: str, owners: Iterable[str]) -> str | None:
    matches = [owner for owner in owners if _is_under(path, owner)]
    return max(matches, key=len) if matches else None


def _joint_is_enabled(prim: Any, usd_physics: Any) -> bool:
    """Return whether a USD joint participates in the live physics graph."""

    if not prim.IsA(usd_physics.Joint):
        return False
    value = usd_physics.Joint(prim).GetJointEnabledAttr().Get()
    return True if value is None else bool(value)


def _paths_related(left: str, right: str) -> bool:
    return _is_under(left, right) or _is_under(right, left)


def task_scoped_evidence_surface(
    surface: FixedEvidenceSurface,
    evidence_prim_paths: Iterable[str],
) -> FixedEvidenceSurface:
    """Keep complete registered rigid state while bounding task contact evidence.

    The composed Stage may contain thousands of unrelated static colliders.  A
    task registering dozens of legitimate object candidates must not multiply
    those colliders into an all-to-all contact tensor.  Registered Prim paths
    select the relevant scene articulations and static collision owners; every
    enabled standalone rigid remains on the formal state axis.
    """

    requested = tuple(sorted({_path(value) for value in evidence_prim_paths}))
    if not requested:
        return surface
    relevant_articulations = {
        surface.robot_articulation_prim,
        *(
            articulation.prim_path
            for articulation in surface.articulations
            if any(_paths_related(articulation.prim_path, path) for path in requested)
        ),
    }
    relevant_static = tuple(
        target
        for target in surface.static_contact_targets
        if any(_paths_related(target, path) for path in requested)
    )
    source_prims = relevant_articulations | set(surface.rigid_objects)
    target_prims = source_prims | set(relevant_static)
    contact_pairs = tuple(
        pair
        for pair in surface.contact_pairs
        if pair.source_prim in source_prims and pair.target_prim in target_prims
    )
    if not contact_pairs:
        raise ValueError(
            "task-scoped fixed evidence resolved no contact pairs for registered Prim paths"
        )
    return replace(
        surface,
        static_contact_targets=relevant_static,
        contact_pairs=contact_pairs,
    )


def _xyzw_pose_to_wxyz(value: Any, *, origin: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape[-1] != 7:
        raise ValueError(f"PhysX pose must end in 7 values, got {array.shape}")
    result = np.empty_like(array, dtype=np.float32)
    result[..., :3] = array[..., :3] - origin
    result[..., 3] = array[..., 6]
    result[..., 4:7] = array[..., 3:6]
    norms = np.linalg.norm(result[..., 3:7], axis=-1, keepdims=True)
    if np.any(norms <= 1e-12) or not np.isfinite(norms).all():
        raise ValueError("PhysX returned an invalid quaternion")
    result[..., 3:7] /= norms
    return np.ascontiguousarray(result, dtype=np.float32)


def _numpy(value: Any, dtype: Any = np.float32) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    result = np.asarray(value, dtype=dtype)
    if result.dtype.kind == "f" and not np.isfinite(result).all():
        raise ValueError("physics evidence contains NaN or infinity")
    return np.ascontiguousarray(result)


class UsdFormalStageFactsSource:
    """Convert the composed single-env Stage into formal competition facts."""

    def __init__(
        self,
        stage: Any,
        *,
        robot_role: str,
        excluded_articulation_roots: Iterable[str] = (),
    ) -> None:
        from pxr import UsdGeom, UsdPhysics

        if robot_role not in FORMAL_ROBOT_MOUNTS:
            raise ValueError(f"unsupported formal robot role: {robot_role!r}")
        self.stage = stage
        self.robot_role = robot_role
        self.formal_robot_mount = FORMAL_ROBOT_MOUNTS[robot_role]
        self._UsdPhysics = UsdPhysics
        self._meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
        self.runtime_to_formal: dict[str, str] = {}
        self.formal_to_runtime: dict[str, str] = {}
        self.runtime_articulation_owner: dict[str, str] = {}
        self.runtime_rigid_owner: dict[str, str] = {}
        self.runtime_collision_owner: dict[str, str] = {}
        self._excluded_articulation_roots = {
            _path(value) for value in excluded_articulation_roots
        }
        self._facts = self._build()

    def formal_path(self, runtime_path: str) -> str:
        runtime_path = _path(runtime_path)
        if self.robot_role == "m20_piper":
            # This Runner composes the safety robot and task directly at their
            # frozen formal paths, so runtime-to-formal is an identity map.
            if not runtime_path.startswith("/") or runtime_path == "/":
                raise ValueError(f"invalid formal M20 Stage path: {runtime_path}")
            return runtime_path
        if _is_under(runtime_path, RUNTIME_ROBOT_ROOT):
            if (
                self.robot_role == "r1_lite"
                and runtime_path == f"{RUNTIME_ROBOT_ROOT}/base_link"
            ):
                return self.formal_robot_mount
            suffix = runtime_path[len(RUNTIME_ROBOT_ROOT) :]
            return self.formal_robot_mount + suffix
        if _is_under(runtime_path, RUNTIME_SCENE_ROOT):
            suffix = runtime_path[len(RUNTIME_SCENE_ROOT) :]
            return "/World" + suffix
        raise ValueError(f"physics Prim is outside the supported single-env Stage: {runtime_path}")

    def _register_path(self, runtime_path: str) -> str:
        formal = self.formal_path(runtime_path)
        prior = self.formal_to_runtime.get(formal)
        if prior is not None and prior != runtime_path:
            raise ValueError(f"formal path mapping is not one-to-one: {formal}: {prior}, {runtime_path}")
        self.runtime_to_formal[runtime_path] = formal
        self.formal_to_runtime[formal] = runtime_path
        return formal

    def _joint_targets(self, prim: Any) -> tuple[str, ...]:
        joint = self._UsdPhysics.Joint(prim)
        targets = []
        for relationship in (joint.GetBody0Rel(), joint.GetBody1Rel()):
            targets.extend(str(item) for item in relationship.GetTargets())
        return tuple(targets)

    @staticmethod
    def _quaternion_wxyz(value: Any) -> tuple[float, float, float, float]:
        if value is None:
            return (1.0, 0.0, 0.0, 0.0)
        real = float(value.GetReal())
        imag = value.GetImaginary()
        return (real, float(imag[0]), float(imag[1]), float(imag[2]))

    def _formal_relationship_target(self, relationship: Any, field: str) -> str | None:
        targets = tuple(str(item) for item in relationship.GetTargets())
        if len(targets) > 1:
            raise ValueError(f"{field} must have at most one target, got {targets}")
        return self._register_path(targets[0]) if targets else None

    @staticmethod
    def _normalized_limit(value: Any, scale: float) -> float | None:
        if value is None:
            return None
        result = float(value) * scale
        return result if np.isfinite(result) else None

    def _joint_coordinate(
        self,
        prim: Any,
        formal_path: str,
    ) -> JointCoordinateEvidence:
        UsdPhysics = self._UsdPhysics
        joint = UsdPhysics.Joint(prim)
        if prim.IsA(UsdPhysics.RevoluteJoint):
            typed_joint = UsdPhysics.RevoluteJoint(prim)
            joint_type = "revolute"
            unit = "rad"
            limit_scale = np.pi / 180.0
        elif prim.IsA(UsdPhysics.PrismaticJoint):
            typed_joint = UsdPhysics.PrismaticJoint(prim)
            joint_type = "prismatic"
            unit = "m"
            limit_scale = self._meters_per_unit
        else:
            raise ValueError(
                "JOINT_SIGNATURE_INVALID: unsupported movable joint type: "
                + formal_path
            )
        local_pos0 = joint.GetLocalPos0Attr().Get()
        local_pos1 = joint.GetLocalPos1Attr().Get()
        if local_pos0 is None:
            local_pos0 = (0.0, 0.0, 0.0)
        if local_pos1 is None:
            local_pos1 = (0.0, 0.0, 0.0)
        return JointCoordinateEvidence(
            joint_path=formal_path,
            joint_type=joint_type,
            body0_path=self._formal_relationship_target(
                joint.GetBody0Rel(), f"{formal_path}.body0"
            ),
            body1_path=self._formal_relationship_target(
                joint.GetBody1Rel(), f"{formal_path}.body1"
            ),
            axis=str(typed_joint.GetAxisAttr().Get() or "X").upper(),
            local_pos0_m=tuple(
                float(local_pos0[index]) * self._meters_per_unit for index in range(3)
            ),
            local_rot0_wxyz=self._quaternion_wxyz(joint.GetLocalRot0Attr().Get()),
            local_pos1_m=tuple(
                float(local_pos1[index]) * self._meters_per_unit for index in range(3)
            ),
            local_rot1_wxyz=self._quaternion_wxyz(joint.GetLocalRot1Attr().Get()),
            lower_limit=self._normalized_limit(
                typed_joint.GetLowerLimitAttr().Get(), limit_scale
            ),
            upper_limit=self._normalized_limit(
                typed_joint.GetUpperLimitAttr().Get(), limit_scale
            ),
            unit=unit,
        )

    def _build(self) -> tuple[FormalPrimFacts, ...]:
        UsdPhysics = self._UsdPhysics
        if self.robot_role == "m20_piper":
            prims = list(self.stage.Traverse())
        else:
            prims = [
                prim
                for prim in self.stage.Traverse()
                if _is_under(str(prim.GetPath()), RUNTIME_ROBOT_ROOT)
                or _is_under(str(prim.GetPath()), RUNTIME_SCENE_ROOT)
            ]
        candidate_roots = {
            str(prim.GetPath())
            for prim in prims
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI)
        }
        def rigid_body_is_enabled(prim: Any) -> bool:
            if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                return False
            value = UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Get()
            return True if value is None else bool(value)

        rigid_paths = {
            str(prim.GetPath())
            for prim in prims
            if rigid_body_is_enabled(prim)
        }

        joint_targets: dict[str, tuple[str, ...]] = {}
        for prim in prims:
            if _joint_is_enabled(prim, UsdPhysics):
                joint_targets[str(prim.GetPath())] = self._joint_targets(prim)

        body_owner: dict[str, str | None] = {}
        for path in rigid_paths:
            body_owner[path] = _closest_owner(path, candidate_roots)

        # Pro authors its articulation API on base_link while wheel/arm bodies
        # are siblings below r1_pro_with_gripper.  Resolve explicit physical
        # ownership through the USD joint graph; path ancestry alone is not a
        # valid articulation membership rule for this asset.
        adjacency: dict[str, set[str]] = {path: set() for path in rigid_paths}
        for targets in joint_targets.values():
            bodies = [path for path in targets if path in rigid_paths]
            if len(bodies) == 2:
                adjacency[bodies[0]].add(bodies[1])
                adjacency[bodies[1]].add(bodies[0])
        claimed_by_graph: dict[str, str] = {}
        for root in sorted(candidate_roots):
            if root not in rigid_paths:
                continue
            pending = [root]
            component: set[str] = set()
            while pending:
                body = pending.pop()
                if body in component:
                    continue
                component.add(body)
                pending.extend(adjacency[body] - component)
            for body in component:
                prior = claimed_by_graph.get(body)
                if prior is not None and prior != root:
                    raise ValueError(
                        f"rigid body belongs to multiple articulation graphs: {body}: {prior}, {root}"
                    )
                claimed_by_graph[body] = root
                body_owner[body] = root

        # Some imported assets author ArticulationRootAPI on inert placeholder
        # Prims (for example a microwave ``worldBody``) that own no rigid body.
        # Such a Prim cannot produce an Isaac/PhysX articulation view and must
        # not enter the formal evidence inventory.  Keep every root that owns
        # at least one real rigid body; malformed roots with physical bodies
        # remain visible to the strict downstream schema validation.
        roots = {
            owner
            for owner in body_owner.values()
            if owner is not None
        }
        joint_owner: dict[str, str | None] = {}
        for path, targets in joint_targets.items():
            candidates = {
                owner
                for target in targets
                for owner in [body_owner.get(target) or _closest_owner(target, roots)]
                if owner is not None
            }
            ancestry = _closest_owner(path, roots)
            if ancestry is not None:
                candidates.add(ancestry)
            if len(candidates) > 1:
                raise ValueError(f"joint has conflicting articulation ownership: {path}: {sorted(candidates)}")
            joint_owner[path] = next(iter(candidates), None)

        # The v0.6 fixed-evidence schema has one coordinate per joint and can
        # faithfully encode only revolute and prismatic joints.  Some authored
        # scene props (for example warehouse display vehicles) contain
        # spherical/D6 articulations that are unrelated to the formal robot.
        # Do not invent scalar coordinates for those articulations.  They stay
        # live in simulation/rendering, but their whole subtree is omitted from
        # the fixed evidence inventory because the protocol cannot represent
        # it faithfully.
        unsupported_roots = {
            joint_owner[str(prim.GetPath())]
            for prim in prims
            if str(prim.GetPath()) in joint_targets
            and not prim.IsA(UsdPhysics.FixedJoint)
            and not prim.IsA(UsdPhysics.RevoluteJoint)
            and not prim.IsA(UsdPhysics.PrismaticJoint)
            and joint_owner.get(str(prim.GetPath())) is not None
        }
        unsupported_roots.discard(None)
        excluded_roots = unsupported_roots | (
            self._excluded_articulation_roots & roots
        )
        self.unsupported_articulation_roots = tuple(sorted(excluded_roots))
        if excluded_roots:
            roots.difference_update(excluded_roots)
            for path, owner in tuple(joint_owner.items()):
                if owner in excluded_roots:
                    joint_owner[path] = None
            # A scene articulation can be excluded after its live PhysX
            # signature disagrees with the composed USD graph.  Clear every
            # cached body claim as well, including non-descendant sibling
            # bodies reached through a joint graph, so no fact can retain a
            # dangling owner that is absent from ``roots``.
            for path, owner in tuple(body_owner.items()):
                if owner in excluded_roots:
                    body_owner[path] = None

        self.runtime_articulation_owner.update(
            {
                path: owner
                for path, owner in body_owner.items()
                if owner is not None
                and not any(_is_under(path, root) for root in excluded_roots)
            }
        )
        self.runtime_rigid_owner.update(
            {
                path: path
                for path, owner in body_owner.items()
                if owner is None
                and not any(_is_under(path, root) for root in excluded_roots)
            }
        )

        rows: list[FormalPrimFacts] = []
        relevant_paths: set[str] = set()
        for prim in prims:
            path = str(prim.GetPath())
            if any(_is_under(path, root) for root in excluded_roots):
                continue
            rigid = path in rigid_paths
            collision = prim.HasAPI(UsdPhysics.CollisionAPI)
            articulation = path in roots
            joint = path in joint_targets
            if not (rigid or collision or articulation or joint):
                continue
            relevant_paths.add(path)
            formal = self._register_path(path)
            owner = body_owner.get(path)
            if owner is None and not rigid:
                owner = _closest_owner(path, roots)
            if joint:
                owner = joint_owner[path]
            movable_joint = (
                joint
                and not prim.IsA(UsdPhysics.FixedJoint)
                and owner is not None
            )
            physical_rigid_body = _closest_owner(path, rigid_paths)
            physical_articulation_owner = (
                body_owner.get(physical_rigid_body)
                if physical_rigid_body is not None
                else None
            )
            runtime_collision_owner = None
            if collision:
                runtime_collision_owner = (
                    physical_articulation_owner
                    or physical_rigid_body
                    or owner
                    or path
                )
                self.runtime_collision_owner[path] = runtime_collision_owner
            rigid_api = UsdPhysics.RigidBodyAPI(prim) if rigid else None
            rigid_enabled = True
            kinematic_enabled = False
            if rigid_api is not None:
                value = rigid_api.GetRigidBodyEnabledAttr().Get()
                rigid_enabled = True if value is None else bool(value)
                value = rigid_api.GetKinematicEnabledAttr().Get()
                kinematic_enabled = False if value is None else bool(value)
            rows.append(
                FormalPrimFacts(
                    path=formal,
                    has_rigid_body_api=rigid,
                    has_collision_api=collision,
                    has_articulation_root_api=articulation,
                    is_movable_joint=movable_joint,
                    articulation_root_path=(self._register_path(owner) if owner is not None else None),
                    collision_owner_path=(
                        self._register_path(runtime_collision_owner)
                        if runtime_collision_owner is not None
                        else None
                    ),
                    rigid_body_enabled=rigid_enabled,
                    kinematic_enabled=kinematic_enabled,
                    joint_coordinate=(
                        self._joint_coordinate(prim, formal)
                        if movable_joint
                        else None
                    ),
                )
            )
        return tuple(rows)

    def iter_prim_facts(self) -> Iterable[FormalPrimFacts]:
        return iter(self._facts)

    def articulation_owner(self, runtime_path: str) -> str | None:
        direct = self.runtime_articulation_owner.get(runtime_path)
        return direct or _closest_owner(runtime_path, set(self.runtime_articulation_owner.values()))

    def rigid_owner(self, runtime_path: str) -> str | None:
        return _closest_owner(runtime_path, self.runtime_rigid_owner)

    def collision_owner(self, runtime_path: str) -> str | None:
        return self.runtime_collision_owner.get(runtime_path)


@dataclass(slots=True)
class _ArticulationRuntime:
    formal_prim: str
    runtime_prim: str
    view: Any
    joint_paths: tuple[str, ...]
    joint_order: np.ndarray
    body_order: np.ndarray
    joint_slice: slice
    body_slice: slice
    joint_scale: np.ndarray
    joint_offset: np.ndarray
    joint_lower: np.ndarray
    joint_upper: np.ndarray
    joint_tolerance: np.ndarray
    joint_transform_proofs: tuple[str, ...]


def _limit_matches(runtime_value: float, formal_value: float | None, tolerance: float) -> bool:
    if formal_value is None:
        return not np.isfinite(runtime_value) or abs(runtime_value) >= 1e20
    return np.isfinite(runtime_value) and abs(runtime_value - formal_value) <= tolerance


def _joint_coordinate_transforms(
    formal_joints: tuple[JointCoordinateEvidence, ...],
    runtime_limits: np.ndarray,
    runtime_types: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    if runtime_limits.shape != (len(formal_joints), 2):
        raise ValueError(
            "JOINT_SIGNATURE_MISMATCH: PhysX DOF limits shape differs from formal J axis: "
            f"{runtime_limits.shape} != {(len(formal_joints), 2)}"
        )
    if runtime_types is not None and runtime_types.shape != (len(formal_joints),):
        raise ValueError(
            "JOINT_SIGNATURE_MISMATCH: PhysX DOF types shape differs from formal J axis: "
            f"{runtime_types.shape} != {(len(formal_joints),)}"
        )
    scales: list[float] = []
    offsets: list[float] = []
    lowers: list[float] = []
    uppers: list[float] = []
    tolerances: list[float] = []
    proofs: list[str] = []
    for index, joint in enumerate(formal_joints):
        if runtime_types is not None:
            expected_type = 0 if joint.joint_type == "revolute" else 1
            if int(runtime_types[index]) != expected_type:
                raise ValueError(
                    "JOINT_SIGNATURE_MISMATCH: "
                    f"joint={joint.joint_path}, formal_type={joint.joint_type}, "
                    f"physx_dof_type={int(runtime_types[index])}"
                )
        tolerance = 1e-4 if joint.joint_type == "revolute" else 1e-5
        runtime_lower = float(runtime_limits[index, 0])
        runtime_upper = float(runtime_limits[index, 1])
        identity = _limit_matches(runtime_lower, joint.lower_limit, tolerance) and _limit_matches(
            runtime_upper, joint.upper_limit, tolerance
        )
        reversed_limits = _limit_matches(
            -runtime_upper, joint.lower_limit, tolerance
        ) and _limit_matches(-runtime_lower, joint.upper_limit, tolerance)
        if identity:
            scale = 1.0
            proof = "physx_limits_identity_match"
        elif reversed_limits:
            scale = -1.0
            proof = "physx_limits_reverse_match"
        else:
            raise ValueError(
                "JOINT_COORDINATE_MAPPING_INVALID: "
                f"joint={joint.joint_path}, index={index}, "
                f"physx_limits=[{runtime_lower},{runtime_upper}], "
                f"formal_limits=[{joint.lower_limit},{joint.upper_limit}], unit={joint.unit}"
            )
        scales.append(scale)
        offsets.append(0.0)
        lowers.append(np.nan if joint.lower_limit is None else joint.lower_limit)
        uppers.append(np.nan if joint.upper_limit is None else joint.upper_limit)
        tolerances.append(tolerance)
        proofs.append(proof)
    return (
        np.asarray(scales, dtype=np.float32),
        np.asarray(offsets, dtype=np.float32),
        np.asarray(lowers, dtype=np.float32),
        np.asarray(uppers, dtype=np.float32),
        np.asarray(tolerances, dtype=np.float32),
        tuple(proofs),
    )


@dataclass(slots=True)
class _RigidRuntime:
    formal_prim: str
    runtime_prim: str
    view: Any


class FixedEvidenceRuntime:
    """Live sampler for one formally discovered evidence surface."""

    def __init__(
        self,
        env: Any | None,
        stage: Any,
        *,
        task_id: str,
        instruction: str,
        robot_role: str,
        formal_stage_digest: str,
        physics_dt: float | None = None,
        origin_xyz: Any | None = None,
        evidence_prim_paths: Iterable[str] = (),
    ) -> None:
        import omni.physics.tensors as physics_tensors
        from pxr import PhysxSchema

        self.env = env
        self.stage = stage
        self.task_id = str(task_id)
        self.instruction = str(instruction)
        if env is not None:
            self.origin = _numpy(env.scene.env_origins[0], np.float32)
            self.physics_dt = float(env.physics_dt)
        else:
            self.origin = _numpy(
                np.zeros(3, dtype=np.float32) if origin_xyz is None else origin_xyz,
                np.float32,
            ).reshape(3)
            if physics_dt is None or not np.isfinite(float(physics_dt)) or float(physics_dt) <= 0:
                raise ValueError("physics_dt must be positive when env is not provided")
            self.physics_dt = float(physics_dt)
        if robot_role not in FORMAL_ROBOT_MOUNTS:
            raise ValueError(f"unsupported formal robot role: {robot_role!r}")
        self.robot_role = robot_role
        self.evidence_prim_paths = tuple(
            sorted({_path(value) for value in evidence_prim_paths})
        )
        self.source = UsdFormalStageFactsSource(stage, robot_role=robot_role)
        digest = str(formal_stage_digest).lower()
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError("formal_stage_digest must be the frozen 64-character SHA-256")
        self.stage_sha256 = digest
        try:
            preliminary_surface = discover_fixed_evidence_surface(
                self.source,
                expected_robot_role=self.robot_role,
                task_id=self.task_id,
                formal_stage_digest=self.stage_sha256,
                enforce_formal_robot_axis_counts=True,
                budget=TASK_SCOPED_DISCOVERY_BUDGET,
            )
        except Exception as error:
            roots = sorted(
                item.path
                for item in self.source.iter_prim_facts()
                if item.has_articulation_root_api
            )
            raise ValueError(
                f"formal {self.robot_role} discovery failed; mapped articulation roots={roots}; "
                f"cause={type(error).__name__}: {error}"
            ) from error
        preliminary_robot = next(
            item
            for item in preliminary_surface.articulations
            if item.prim_path == preliminary_surface.robot_articulation_prim
        )
        if self.robot_role == "m20_piper":
            gripper_matches = list(
                get_formal_robot_profile(self.robot_role).gripper_closure_prims
            )
        else:
            gripper_matches = [
                path
                for path in preliminary_robot.body_paths
                if path.rsplit("/", 1)[-1] == "right_gripper_link"
            ]
            if len(gripper_matches) != 1:
                raise ValueError(
                    f"{self.robot_role} must expose exactly one right_gripper_link body, got {gripper_matches}"
                )

        # USD may advertise an articulation that PhysX does not expose with a
        # matching DOF/body signature.  Probe the live PhysX metadata before
        # freezing the surface.  A mismatch on the formal robot remains fatal;
        # mismatched scene props remain in the live simulation but are omitted
        # from the fixed surface, never represented by fabricated coordinates.
        self.sim_view = physics_tensors.create_simulation_view("torch")
        self.sim_view.set_subspace_roots("/")
        physx_incompatible_roots: list[str] = []
        for articulation in preliminary_surface.articulations:
            runtime_root = self.source.formal_to_runtime[articulation.prim_path]
            view = self.sim_view.create_articulation_view(runtime_root)
            failure: Exception | None = None
            if view._backend is None:
                failure = RuntimeError(f"PhysX articulation view is unavailable: {runtime_root}")
            else:
                try:
                    self._name_order(
                        articulation.joint_paths,
                        [str(name) for name in view.shared_metatype.dof_names],
                        "joint",
                        articulation.prim_path,
                    )
                    self._name_order(
                        articulation.body_paths,
                        [str(name) for name in view.shared_metatype.link_names],
                        "body",
                        articulation.prim_path,
                    )
                except ValueError as error:
                    failure = error
            if failure is None:
                continue
            if articulation.prim_path == preliminary_surface.robot_articulation_prim:
                raise ValueError(
                    f"formal robot PhysX signature mismatch: {articulation.prim_path}: {failure}"
                ) from failure
            physx_incompatible_roots.append(runtime_root)

        self.excluded_scene_articulation_roots = tuple(sorted(physx_incompatible_roots))
        if physx_incompatible_roots:
            self.source = UsdFormalStageFactsSource(
                stage,
                robot_role=robot_role,
                excluded_articulation_roots=physx_incompatible_roots,
            )
            preliminary_surface = discover_fixed_evidence_surface(
                self.source,
                expected_robot_role=self.robot_role,
                task_id=self.task_id,
                formal_stage_digest=self.stage_sha256,
                enforce_formal_robot_axis_counts=True,
                budget=TASK_SCOPED_DISCOVERY_BUDGET,
            )
            preliminary_robot = next(
                item
                for item in preliminary_surface.articulations
                if item.prim_path == preliminary_surface.robot_articulation_prim
            )
            if self.robot_role != "m20_piper":
                gripper_matches = [
                    path
                    for path in preliminary_robot.body_paths
                    if path.rsplit("/", 1)[-1] == "right_gripper_link"
                ]
                if len(gripper_matches) != 1:
                    raise ValueError(
                        f"{self.robot_role} must expose exactly one right_gripper_link body, "
                        f"got {gripper_matches}"
                    )
        self.surface: FixedEvidenceSurface = task_scoped_evidence_surface(
            discover_fixed_evidence_surface(
                self.source,
                expected_robot_role=self.robot_role,
                task_id=self.task_id,
                formal_stage_digest=self.stage_sha256,
                gripper_closure_prims=(gripper_matches[0],),
                enforce_formal_robot_axis_counts=True,
                budget=TASK_SCOPED_DISCOVERY_BUDGET,
            ),
            self.evidence_prim_paths,
        )

        self.articulations: list[_ArticulationRuntime] = []
        joint_cursor = 0
        body_cursor = 0
        for articulation in self.surface.articulations:
            runtime_root = self.source.formal_to_runtime[articulation.prim_path]
            view = self.sim_view.create_articulation_view(runtime_root)
            if view._backend is None:
                raise RuntimeError(f"PhysX articulation view is unavailable: {runtime_root}")
            dof_names = [str(name) for name in view.shared_metatype.dof_names]
            link_names = [str(name) for name in view.shared_metatype.link_names]
            joint_order = self._name_order(articulation.joint_paths, dof_names, "joint", articulation.prim_path)
            body_order = self._name_order(articulation.body_paths, link_names, "body", articulation.prim_path)
            raw_limits = _numpy(view.get_dof_limits(), np.float64)
            if raw_limits.ndim != 3 or raw_limits.shape[0] != 1 or raw_limits.shape[2] != 2:
                raise RuntimeError(
                    "JOINT_SIGNATURE_MISMATCH: PhysX DOF limits must have shape [1,J,2]: "
                    f"{articulation.prim_path}: {raw_limits.shape}"
                )
            raw_types = _numpy(view.get_dof_types(), np.uint8)
            if raw_types.ndim != 2 or raw_types.shape[0] != 1:
                raise RuntimeError(
                    "JOINT_SIGNATURE_MISMATCH: PhysX DOF types must have shape [1,J]: "
                    f"{articulation.prim_path}: {raw_types.shape}"
                )
            (
                joint_scale,
                joint_offset,
                joint_lower,
                joint_upper,
                joint_tolerance,
                joint_transform_proofs,
            ) = _joint_coordinate_transforms(
                articulation.joints,
                raw_limits[0][joint_order],
                raw_types[0][joint_order],
            )
            joint_slice = slice(joint_cursor, joint_cursor + len(joint_order))
            body_slice = slice(body_cursor, body_cursor + len(body_order))
            joint_cursor = joint_slice.stop
            body_cursor = body_slice.stop
            self.articulations.append(
                _ArticulationRuntime(
                    articulation.prim_path,
                    runtime_root,
                    view,
                    articulation.joint_paths,
                    joint_order,
                    body_order,
                    joint_slice,
                    body_slice,
                    joint_scale,
                    joint_offset,
                    joint_lower,
                    joint_upper,
                    joint_tolerance,
                    joint_transform_proofs,
                )
            )
        self.total_joints = joint_cursor
        self.total_bodies = body_cursor

        self.rigids: list[_RigidRuntime] = []
        for formal in self.surface.rigid_objects:
            runtime = self.source.formal_to_runtime[formal]
            view = self.sim_view.create_rigid_body_view(runtime)
            if view._backend is None:
                raise RuntimeError(f"PhysX rigid-body view is unavailable: {runtime}")
            self.rigids.append(_RigidRuntime(formal, runtime, view))

        self.static_runtime = [
            self.source.formal_to_runtime[formal]
            for formal in self.surface.static_contact_targets
        ]
        self.contact_slices: dict[tuple[str, str], slice] = {}
        contact_cursor = 0
        for pair in self.surface.contact_pairs:
            next_cursor = contact_cursor + pair.contact_axis_size
            self.contact_slices[(pair.source_prim, pair.target_prim)] = slice(contact_cursor, next_cursor)
            contact_cursor = next_cursor
        self.total_contact_channels = contact_cursor
        # Sensor axes follow the fixed surface's source C axes exactly: every
        # body of every A, then every D root.  Filter axes are the physical
        # bodies/colliders owned by every A/D/K target.  The backend returns
        # target-on-sensor force, which is the required target_on_source sign.
        pair_sources = {pair.source_prim for pair in self.surface.contact_pairs}
        pair_targets = {pair.target_prim for pair in self.surface.contact_pairs}
        self._contact_source_rows: dict[str, np.ndarray] = {}
        source_actor_paths: list[str] = []
        for articulation in self.surface.articulations:
            if articulation.prim_path not in pair_sources:
                continue
            rows = []
            for body in articulation.body_paths:
                rows.append(len(source_actor_paths))
                source_actor_paths.append(self.source.formal_to_runtime[body])
            self._contact_source_rows[articulation.prim_path] = np.asarray(rows, dtype=np.int64)
        for rigid in self.rigids:
            if rigid.formal_prim not in pair_sources:
                continue
            self._contact_source_rows[rigid.formal_prim] = np.asarray(
                [len(source_actor_paths)], dtype=np.int64
            )
            source_actor_paths.append(rigid.runtime_prim)

        self._contact_target_columns: dict[str, np.ndarray] = {}
        target_actor_paths: list[str] = []
        for articulation in self.surface.articulations:
            if articulation.prim_path not in pair_targets:
                continue
            columns = []
            for body in articulation.body_paths:
                columns.append(len(target_actor_paths))
                target_actor_paths.append(self.source.formal_to_runtime[body])
            self._contact_target_columns[articulation.prim_path] = np.asarray(columns, dtype=np.int64)
        for rigid in self.rigids:
            if rigid.formal_prim not in pair_targets:
                continue
            self._contact_target_columns[rigid.formal_prim] = np.asarray(
                [len(target_actor_paths)], dtype=np.int64
            )
            target_actor_paths.append(rigid.runtime_prim)
        for formal, runtime in zip(
            self.surface.static_contact_targets, self.static_runtime, strict=True
        ):
            if formal not in pair_targets:
                continue
            self._contact_target_columns[formal] = np.asarray(
                [len(target_actor_paths)], dtype=np.int64
            )
            target_actor_paths.append(runtime)

        expected_source_actors = sum(
            len(articulation.body_paths)
            for articulation in self.surface.articulations
            if articulation.prim_path in pair_sources
        ) + sum(rigid.formal_prim in pair_sources for rigid in self.rigids)
        if len(source_actor_paths) != expected_source_actors:
            raise RuntimeError("fixed contact sensor axis does not match scoped A/D source axes")
        if len(set(source_actor_paths)) != len(source_actor_paths):
            raise RuntimeError("fixed contact sensor paths are not unique")
        if len(set(target_actor_paths)) != len(target_actor_paths):
            raise RuntimeError("fixed contact filter paths are not unique")

        for actor_path in source_actor_paths:
            prim = stage.GetPrimAtPath(actor_path)
            if not prim or not prim.IsValid():
                raise RuntimeError(f"contact-report actor Prim disappeared: {actor_path}")
            api = PhysxSchema.PhysxContactReportAPI.Apply(prim)
            if hasattr(api, "CreateThresholdAttr"):
                api.CreateThresholdAttr().Set(0.0)
            else:
                api.CreatePhysxContactReportThresholdAttr().Set(0.0)

        # Explicit one-to-many construction preserves source/filter ordering;
        # wildcard resolution would make HDF axes backend-order dependent.
        filter_patterns = [list(target_actor_paths) for _ in source_actor_paths]
        self.contact_view = self.sim_view.create_rigid_contact_view(
            source_actor_paths,
            filter_patterns=filter_patterns,
        )
        if self.contact_view._backend is None:
            raise RuntimeError("PhysX rigid contact view is unavailable")
        if self.contact_view.sensor_count != len(source_actor_paths):
            raise RuntimeError(
                "PhysX contact sensor axis mismatch: "
                f"expected {len(source_actor_paths)}, got {self.contact_view.sensor_count}"
            )
        if self.contact_view.filter_count != len(target_actor_paths):
            raise RuntimeError(
                "PhysX contact filter axis mismatch: "
                f"expected {len(target_actor_paths)}, got {self.contact_view.filter_count}"
            )

    def _name_order(
        self,
        formal_paths: tuple[str, ...],
        runtime_names: list[str],
        kind: str,
        owner: str,
    ) -> np.ndarray:
        by_name: dict[str, list[int]] = {}
        for index, name in enumerate(runtime_names):
            by_name.setdefault(name, []).append(index)
        result = []
        for path in formal_paths:
            runtime_path = self.source.formal_to_runtime.get(path)
            if runtime_path is None:
                raise ValueError(f"{owner} {kind} path has no frozen runtime mapping: {path}")
            name = runtime_path.rsplit("/", 1)[-1]
            matches = by_name.get(name, [])
            if len(matches) != 1:
                raise ValueError(f"{owner} {kind} {name!r} maps to {len(matches)} PhysX axes")
            result.append(matches[0])
        if sorted(result) != list(range(len(runtime_names))):
            raise ValueError(f"{owner} formal {kind} ordering does not cover the complete PhysX axis")
        return np.asarray(result, dtype=np.int64)

    @staticmethod
    def _validate_joint_limits(item: _ArticulationRuntime, formal_q: np.ndarray) -> None:
        lower_mask = np.isfinite(item.joint_lower)
        upper_mask = np.isfinite(item.joint_upper)
        if np.any(
            formal_q[lower_mask]
            < item.joint_lower[lower_mask] - item.joint_tolerance[lower_mask]
        ) or np.any(
            formal_q[upper_mask]
            > item.joint_upper[upper_mask] + item.joint_tolerance[upper_mask]
        ):
            bad = np.flatnonzero(
                (lower_mask & (formal_q < item.joint_lower - item.joint_tolerance))
                | (upper_mask & (formal_q > item.joint_upper + item.joint_tolerance))
            )
            index = int(bad[0])
            raise RuntimeError(
                "JOINT_VALUE_OUT_OF_FORMAL_LIMITS: "
                f"articulation={item.formal_prim}, joint={item.joint_paths[index]}, "
                f"formal_index={index}, "
                f"value={float(formal_q[index])}, "
                f"formal=[{float(item.joint_lower[index])},{float(item.joint_upper[index])}]"
            )

    def _articulation_values(self) -> dict[str, np.ndarray]:
        roots_pose = []
        roots_velocity = []
        joints_position = np.zeros((self.total_joints,), dtype=np.float32)
        joints_velocity = np.zeros((self.total_joints,), dtype=np.float32)
        bodies_pose = np.zeros((self.total_bodies, 7), dtype=np.float32)
        bodies_velocity = np.zeros((self.total_bodies, 6), dtype=np.float32)
        for item in self.articulations:
            root_pose = _xyzw_pose_to_wxyz(_numpy(item.view.get_root_transforms())[0], origin=self.origin)
            root_velocity = _numpy(item.view.get_root_velocities())[0]
            runtime_joint_position = _numpy(item.view.get_dof_positions())[0][item.joint_order]
            runtime_joint_velocity = _numpy(item.view.get_dof_velocities())[0][item.joint_order]
            joint_position = (
                runtime_joint_position * item.joint_scale + item.joint_offset
            )
            joint_velocity = runtime_joint_velocity * item.joint_scale
            self._validate_joint_limits(item, joint_position)
            body_pose = _xyzw_pose_to_wxyz(_numpy(item.view.get_link_transforms())[0], origin=self.origin)
            body_velocity = _numpy(item.view.get_link_velocities())[0]
            roots_pose.append(root_pose)
            roots_velocity.append(root_velocity)
            joints_position[item.joint_slice] = joint_position
            joints_velocity[item.joint_slice] = joint_velocity
            bodies_pose[item.body_slice] = body_pose[item.body_order]
            bodies_velocity[item.body_slice] = body_velocity[item.body_order]
        return {
            "articulation_root_pose": np.asarray(roots_pose, dtype=np.float32),
            "articulation_root_velocity": np.asarray(roots_velocity, dtype=np.float32),
            "articulation_joint_position": joints_position,
            "articulation_joint_velocity": joints_velocity,
            "articulation_body_pose": bodies_pose,
            "articulation_body_velocity": bodies_velocity,
        }

    def _rigid_values(self) -> dict[str, np.ndarray]:
        poses = []
        velocities = []
        for item in self.rigids:
            poses.append(_xyzw_pose_to_wxyz(_numpy(item.view.get_transforms())[0], origin=self.origin))
            velocities.append(_numpy(item.view.get_velocities())[0])
        return {
            "rigid_root_pose": np.asarray(poses, dtype=np.float32),
            "rigid_root_velocity": np.asarray(velocities, dtype=np.float32),
        }

    def _static_pose(self) -> np.ndarray:
        from pxr import Usd, UsdGeom

        poses = []
        for path in self.static_runtime:
            prim = self.stage.GetPrimAtPath(path)
            matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            translation = matrix.ExtractTranslation()
            quaternion = matrix.ExtractRotationQuat()
            imaginary = quaternion.GetImaginary()
            pose = np.asarray(
                [
                    float(translation[0]) - float(self.origin[0]),
                    float(translation[1]) - float(self.origin[1]),
                    float(translation[2]) - float(self.origin[2]),
                    float(quaternion.GetReal()),
                    float(imaginary[0]),
                    float(imaginary[1]),
                    float(imaginary[2]),
                ],
                dtype=np.float32,
            )
            norm = np.linalg.norm(pose[3:])
            if norm <= 1e-12 or not np.isfinite(norm):
                raise ValueError(f"static collision owner has invalid pose: {path}")
            pose[3:] /= norm
            poses.append(pose)
        return np.asarray(poses, dtype=np.float32)

    def initial_sample(self) -> dict[str, np.ndarray]:
        articulation = self._articulation_values()
        articulation.pop("articulation_body_pose")
        articulation.pop("articulation_body_velocity")
        values = {**articulation, **self._rigid_values(), "static_pose": self._static_pose()}
        return {name: np.ascontiguousarray(value[None, ...], dtype=np.float32) for name, value in values.items()}

    def control_sample(self) -> dict[str, np.ndarray]:
        values = {**self._articulation_values(), **self._rigid_values()}
        joint_position = values["articulation_joint_position"]
        closure_values = []
        manifest = self.manifest()
        for closure in manifest["gripper_closure"]:
            indices = closure["joint_indices"]
            if closure.get("formula") == "m20_piper_gap_q7_minus_q8":
                gap = float(joint_position[indices[0]] - joint_position[indices[1]])
                closure_values.append(
                    np.clip(
                        1.0 - gap / float(closure["max_opening_m"]),
                        0.0,
                        1.0,
                    )
                )
            else:
                opening = float(np.mean(np.abs(joint_position[indices])))
                closure_values.append(
                    np.clip(
                        1.0 - opening / float(closure["max_opening_m"]),
                        0.0,
                        1.0,
                    )
                )
        values["gripper_closure"] = np.asarray(closure_values, dtype=np.float32)
        matrix = _numpy(
            self.contact_view.get_contact_force_matrix(self.physics_dt),
            np.float32,
        ).reshape(self.contact_view.sensor_count, self.contact_view.filter_count, 3)
        contact_force = np.zeros((self.total_contact_channels, 3), dtype=np.float32)
        for pair in self.surface.contact_pairs:
            rows = self._contact_source_rows[pair.source_prim]
            columns = self._contact_target_columns[pair.target_prim]
            pair_force = matrix[rows][:, columns].sum(axis=1, dtype=np.float32)
            pair_slice = self.contact_slices[(pair.source_prim, pair.target_prim)]
            if pair_force.shape != (pair_slice.stop - pair_slice.start, 3):
                raise RuntimeError(
                    f"fixed contact C axis mismatch: {pair.source_prim} -> {pair.target_prim}"
                )
            contact_force[pair_slice] = pair_force
        values["contact_net_forces_w"] = contact_force
        return {name: np.ascontiguousarray(value[None, ...], dtype=np.float32) for name, value in values.items()}

    def manifest(self) -> dict[str, Any]:
        articulations = []
        for item, surface_item in zip(self.articulations, self.surface.articulations, strict=True):
            articulations.append(
                {
                    "prim": surface_item.prim_path,
                    "joint_start": item.joint_slice.start,
                    "joint_stop": item.joint_slice.stop,
                    "body_start": item.body_slice.start,
                    "body_stop": item.body_slice.stop,
                    "joint_transforms": [
                        {
                            "joint_path": joint_path,
                            "runtime_index": int(item.joint_order[index]),
                            "formal_index": index,
                            "scale": float(item.joint_scale[index]),
                            "offset": float(item.joint_offset[index]),
                            "proof": item.joint_transform_proofs[index],
                        }
                        for index, joint_path in enumerate(item.joint_paths)
                    ],
                }
            )
        contacts = []
        for pair in self.surface.contact_pairs:
            offset = self.contact_slices[(pair.source_prim, pair.target_prim)]
            contacts.append(
                {
                    "source": pair.source_prim,
                    "target": pair.target_prim,
                    "start": offset.start,
                    "stop": offset.stop,
                }
            )
        robot_prim = self.surface.robot_articulation_prim
        robot = next(item for item in articulations if item["prim"] == robot_prim)
        robot_surface = next(item for item in self.surface.articulations if item.prim_path == robot_prim)
        closure_joint_names = (
            ("joint7", "joint8")
            if self.robot_role == "m20_piper"
            else ("right_gripper_finger_joint1", "right_gripper_finger_joint2")
        )
        closure_joint_paths = tuple(
            matches[0]
            for name in closure_joint_names
            for matches in [[path for path in robot_surface.joint_paths if path.rsplit("/", 1)[-1] == name]]
            if len(matches) == 1
        )
        if len(closure_joint_paths) != len(closure_joint_names):
            raise ValueError(
                f"{self.robot_role} gripper closure joints are missing or ambiguous: {closure_joint_paths}"
            )
        closure_indices = [robot["joint_start"] + robot_surface.joint_paths.index(path) for path in closure_joint_paths]
        return {
            "schema_name": "robolab-fixed-evidence-runtime-plan",
            "schema_version": 3,
            "task_id": self.task_id,
            "instruction": self.instruction,
            "coordinate_convention": "right_handed_z_up_world",
            "quaternion_order": "wxyz",
            "artifact_role": self.surface.robot_role,
            "surface": self.surface.canonical_audit_dict(),
            "surface_sha256": self.surface.audit_sha256(),
            "evidence_scope": {
                "mode": "task_tracked_objects",
                "requested_prims": list(self.evidence_prim_paths),
                "rigid_objects": list(self.surface.rigid_objects),
                "static_contact_targets": list(self.surface.static_contact_targets),
            },
            "articulations": articulations,
            "rigid_objects": list(self.surface.rigid_objects),
            "static_contact_targets": list(self.surface.static_contact_targets),
            "contacts": contacts,
            "gripper_closure": [
                {
                    "prim": self.surface.gripper_closure_prims[0],
                    "joint_indices": closure_indices,
                    "max_opening_m": 0.07 if self.robot_role == "m20_piper" else 0.05,
                    "formula": (
                        "m20_piper_gap_q7_minus_q8"
                        if self.robot_role == "m20_piper"
                        else "mean_absolute_finger_opening"
                    ),
                }
            ],
            "packed_counts": {
                "articulations": len(self.articulations),
                "rigid_objects": len(self.rigids),
                "static_contact_targets": len(self.static_runtime),
                "joints": self.total_joints,
                "bodies": self.total_bodies,
                "contact_channels": self.total_contact_channels,
                "contact_pairs": len(self.surface.contact_pairs),
            },
        }

    def close(self) -> None:
        self.contact_view = None
