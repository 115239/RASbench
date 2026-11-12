"""PhysX contact truth adapter for contact-based safety rules.

The M20 runner applies ``PhysxContactReportAPI`` to every enabled rigid body
before reset.  This adapter subscribes to that clean contact stream and keeps
collider identity, rigid-body ownership and per-step force evidence.
"""

from __future__ import annotations

from typing import Any


class SafetyContactTracker:
    def __init__(self, stage, *, robot_root_prim: str, physics_dt: float) -> None:
        import numpy as np
        from omni.physx import get_physx_simulation_interface
        from omni.physx.bindings._physx import ContactEventType
        from pxr import PhysicsSchemaTools, UsdPhysics

        self._np = np
        self._stage = stage
        self.robot_root_prim = str(robot_root_prim).rstrip("/")
        self.physics_dt = float(physics_dt)
        self._contact_lost = ContactEventType.CONTACT_LOST
        self._path_from_id = PhysicsSchemaTools.intToSdfPath
        self._rigid_body_api = UsdPhysics.RigidBodyAPI
        self._owner_cache: dict[str, str] = {}
        self._active: dict[tuple[str, str], dict[str, Any]] = {}
        self._step_impulses: dict[tuple[str, str], Any] = {}
        self._error: str | None = None
        self._adjacent_body_pairs = self._discover_adjacent_body_pairs(UsdPhysics)
        self._subscription = get_physx_simulation_interface().subscribe_contact_report_events(
            self._on_contact_reports
        )
        if self._subscription is None:
            raise RuntimeError("safety contact report subscription is unavailable")

    def _discover_adjacent_body_pairs(self, UsdPhysics) -> tuple[tuple[str, str], ...]:
        pairs = set()
        for prim in self._stage.Traverse():
            if not prim.IsA(UsdPhysics.Joint):
                continue
            joint = UsdPhysics.Joint(prim)
            body0 = joint.GetBody0Rel().GetTargets()
            body1 = joint.GetBody1Rel().GetTargets()
            if len(body0) != 1 or len(body1) != 1:
                continue
            owner0 = self._rigid_owner(str(body0[0]))
            owner1 = self._rigid_owner(str(body1[0]))
            if owner0 and owner1 and owner0 != owner1:
                pairs.add(tuple(sorted((owner0, owner1))))
        return tuple(sorted(pairs))

    def _rigid_owner(self, collider_path: str) -> str:
        cached = self._owner_cache.get(collider_path)
        if cached is not None:
            return cached
        prim = self._stage.GetPrimAtPath(collider_path)
        owner = ""
        while prim and prim.IsValid() and str(prim.GetPath()) != "/":
            if prim.HasAPI(self._rigid_body_api):
                owner = str(prim.GetPath())
                break
            prim = prim.GetParent()
        self._owner_cache[collider_path] = owner or collider_path
        return owner or collider_path

    def begin_step(self) -> None:
        self._step_impulses = {}

    def _on_contact_reports(self, headers, data) -> None:
        np = self._np
        try:
            for header in headers:
                collider0 = str(self._path_from_id(header.collider0))
                collider1 = str(self._path_from_id(header.collider1))
                key = tuple(sorted((collider0, collider1)))
                if header.type == self._contact_lost:
                    self._active.pop(key, None)
                    self._step_impulses.pop(key, None)
                    continue
                begin = int(header.contact_data_offset)
                end = begin + int(header.num_contact_data)
                if begin < 0 or end < begin or end > len(data):
                    raise ValueError("contact report data offsets are invalid")
                points = data[begin:end]
                impulse = sum(
                    (np.asarray(point.impulse, dtype=np.float64) for point in points),
                    np.zeros(3, dtype=np.float64),
                )
                touching = len(points) == 0 or any(
                    float(point.separation) <= 0.0
                    or float(np.linalg.norm(np.asarray(point.impulse, dtype=np.float64))) > 0.0
                    for point in points
                )
                if not touching:
                    self._active.pop(key, None)
                    continue
                record = {
                    "collider0": collider0,
                    "collider1": collider1,
                    "body0": self._rigid_owner(collider0),
                    "body1": self._rigid_owner(collider1),
                    "touching": True,
                    "points_world": [
                        np.asarray(point.position, dtype=np.float64).astype(float).tolist()
                        for point in points
                    ],
                }
                self._active[key] = record
                self._step_impulses[key] = self._step_impulses.get(
                    key, np.zeros(3, dtype=np.float64)
                ) + impulse
        except Exception as error:  # callback exceptions are not reliably propagated by Kit
            self._error = f"{type(error).__name__}: {error}"

    def end_step(self) -> None:
        # Contact callbacks are synchronous with the physics step.  The method
        # exists to keep the begin/step/end boundary explicit in the Runner.
        return None

    def snapshot(self) -> list[dict[str, Any]] | None:
        if self._error is not None:
            return None
        np = self._np
        result = []
        for key, record in sorted(self._active.items()):
            impulse = self._step_impulses.get(key, np.zeros(3, dtype=np.float64))
            force = impulse / self.physics_dt
            result.append({
                **record,
                "force_xyz_n": force.astype(float).tolist(),
                "force_norm_n": float(np.linalg.norm(force)),
            })
        return result

    @property
    def adjacent_body_pairs(self) -> tuple[tuple[str, str], ...]:
        return self._adjacent_body_pairs

    def close(self) -> None:
        self._subscription = None
