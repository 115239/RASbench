"""Same-run observations of semantic scene Prims and exact collision pairs.

This extension keeps semantic Xforms separate from the native physical inventory.
Moving geometry is transformed by live PhysX body poses, never stale USD poses.
Contact reports are filtered by collider identity, so leaves cannot touch the vase
on behalf of its bottle mesh and a chassis cannot touch on behalf of a finger.
"""
from __future__ import annotations

import json
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np


def array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def matrix_from_pose(pose):
    from pxr import Gf
    x, y, z, qx, qy, qz, qw = map(float, pose)
    result = Gf.Matrix4d(1)
    result.SetRotate(Gf.Quatd(qw, Gf.Vec3d(qx, qy, qz)))
    result.SetTranslateOnly(Gf.Vec3d(x, y, z))
    return np.asarray(result).T


def pose_from_matrix(matrix):
    from pxr import Gf
    rotation = matrix[:3, :3]
    # Remove authored scale before extracting a world orientation.
    u, _, vh = np.linalg.svd(rotation)
    rotation = u @ vh
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vh
    q = Gf.Matrix3d(rotation.T.tolist()).ExtractRotation().GetQuat()
    return np.asarray([*matrix[:3, 3], q.GetReal(), *q.GetImaginary()], dtype=np.float32)


def live_owner(stage, path):
    from pxr import UsdPhysics
    prim = stage.GetPrimAtPath(path)
    while prim and str(prim.GetPath()) != "/":
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            enabled = UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Get()
            if enabled is not False:
                return str(prim.GetPath())
        prim = prim.GetParent()
    return None


class ContactAccumulator:
    """Collider-exact impulses integrated across every observed physics step."""

    def __init__(self, contacts, collider_sets):
        self.contacts = contacts
        self.collider_sets = collider_sets
        self.pending = np.zeros((len(contacts), 3), dtype=np.float64)
        self.impulses = self.pending.copy()
        self.active_steps = np.zeros(len(contacts), dtype=np.uint64)
        self.active_pairs = {}
        self.steps = 0
        self.begun = False

    def begin_step(self):
        if self.begun:
            raise RuntimeError("rubric contact physics step was not closed")
        self.pending.fill(0)
        self.begun = True

    def report(self, collider0, collider1, impulse_on_0, *, lost=False, touching=True):
        # Retain observed contact state during settling: a resting body may
        # fall asleep before scored capture starts and emit no new report.
        # Only impulses inside a bracketed scoring step enter its force sum.
        key = (collider0, collider1)
        if lost:
            self.active_pairs.pop(key, None)
            self.active_pairs.pop((collider1, collider0), None)
            return
        hits = []
        for i, pair in enumerate(self.contacts):
            source = self.collider_sets[pair["source"]]
            target = self.collider_sets[pair["target"]]
            if collider0 in source and collider1 in target:
                if self.begun:
                    self.pending[i] += impulse_on_0
                hits.append(i)
            elif collider1 in source and collider0 in target:
                if self.begun:
                    self.pending[i] -= impulse_on_0
                hits.append(i)
        if touching and hits:
            self.active_pairs[key] = hits
        elif not touching:
            self.active_pairs.pop(key, None)
            self.active_pairs.pop((collider1, collider0), None)

    def end_step(self):
        if not self.begun:
            raise RuntimeError("rubric contact physics step was not started")
        self.impulses += self.pending
        for i in {i for indices in self.active_pairs.values() for i in indices}:
            self.active_steps[i] += 1
        self.steps += 1
        self.begun = False

    def consume(self, physics_dt):
        if self.begun or self.steps < 1:
            raise RuntimeError("rubric contacts have no completed physics observation")
        result = {"net_forces_w": (self.impulses / (self.steps * physics_dt)).astype(np.float32),
                  "impulse_w": self.impulses.astype(np.float32),
                  "active_fraction": (self.active_steps / self.steps).astype(np.float32),
                  "physics_steps": self.steps}
        self.impulses.fill(0)
        self.active_steps.fill(0)
        self.steps = 0
        return result

    def reset_interval(self):
        self.pending.fill(0)
        self.impulses.fill(0)
        self.active_steps.fill(0)
        self.steps = 0
        self.begun = False


class RubricEvidenceRuntime:
    def __init__(self, stage, plan, physics_dt, *, sim_view=None, subscribe=True):
        from pxr import Usd, UsdGeom, UsdPhysics
        self.stage, self.plan, self.physics_dt = stage, plan, float(physics_dt)
        if plan.get("schema") != "bboard-rubric-evidence-plan/v1":
            raise ValueError("unsupported rubric observation plan")
        if not np.isfinite(self.physics_dt) or self.physics_dt <= 0:
            raise ValueError("invalid rubric physics step")
        self.cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        self.views = {}
        self.sim_view = sim_view
        self.objects = []
        self.geometry = {}
        self.error = None
        self.subscription = None
        self.active = False
        for spec in plan["objects"]:
            path = spec["runtime_prim"]
            prim = stage.GetPrimAtPath(path)
            if not prim or not prim.IsValid():
                raise ValueError(f"RUBRIC_PRIM_MISSING: {path}")
            descendants = list(Usd.PrimRange(prim, Usd.TraverseInstanceProxies()))
            colliders = [p for p in descendants if p.HasAPI(UsdPhysics.CollisionAPI)
                         and UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get() is not False]
            if spec.get("collider_prims"):
                colliders = [stage.GetPrimAtPath(p) for p in spec["collider_prims"]]
                if any(not p or not p.HasAPI(UsdPhysics.CollisionAPI) or UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get() is False for p in colliders):
                    raise ValueError(f"RUBRIC_EXPLICIT_COLLIDER_BINDING_INVALID: {path}")
            anchor = spec.get("pose_anchor_prim") or live_owner(stage, path)
            if anchor is None:
                bodies = sorted({live_owner(stage, str(p.GetPath())) for p in colliders} - {None})
                if len(bodies) == 1:
                    anchor = bodies[0]
                # A group with several independent bodies has no single rigid
                # pose. Keep its actual Xform frame; sample each collider with
                # its own live owner for the union AABB below.
            if anchor and not stage.GetPrimAtPath(anchor):
                raise ValueError(f"RUBRIC_ANCHOR_MISSING: {anchor}")
            owner = live_owner(stage, anchor) if anchor else None
            pose_frame = spec.get("pose_frame_prim", path)
            if not stage.GetPrimAtPath(pose_frame):
                raise ValueError(f"RUBRIC_POSE_FRAME_MISSING: {pose_frame}")
            binding = self._binding(pose_frame, owner)
            collider_paths = [str(p.GetPath()) for p in colliders]
            for collider in colliders:
                collider_path = str(collider.GetPath())
                if collider_path in self.geometry:
                    continue
                geometry_binding = self._binding(collider_path, live_owner(stage, collider_path))
                analytic = None
                if collider.IsA(UsdGeom.Mesh):
                    points = np.asarray(UsdGeom.Mesh(collider).GetPointsAttr().Get(), dtype=np.float64)
                    representation = "collision_mesh_vertices"
                elif collider.IsA(UsdGeom.Cube):
                    radius = float(UsdGeom.Cube(collider).GetSizeAttr().Get()) / 2
                    points = np.asarray(list(product([-radius, radius], repeat=3)))
                    representation = "analytic_box"
                elif collider.GetTypeName() in {"Cylinder", "Sphere", "Capsule"}:
                    typed = getattr(UsdGeom, collider.GetTypeName())(collider)
                    radius = float(typed.GetRadiusAttr().Get())
                    height = float(typed.GetHeightAttr().Get()) if hasattr(typed, "GetHeightAttr") else 0.
                    axis = str(typed.GetAxisAttr().Get()) if hasattr(typed, "GetAxisAttr") else "Z"
                    analytic = {"type": collider.GetTypeName(), "radius": radius, "height": height,
                                "axis": {"X": 0, "Y": 1, "Z": 2}[axis]}
                    points = np.zeros((1, 3))
                    representation = "analytic_" + collider.GetTypeName().lower()
                else:
                    # Imported STL colliders can put CollisionAPI on an Xform.
                    # Read its actual descendant vertices, including invisible
                    # collision geometry and instance proxies.
                    parts = []
                    inverse = np.linalg.inv(self._world(collider_path))
                    for child in Usd.PrimRange(collider, Usd.TraverseInstanceProxies()):
                        if child.IsA(UsdGeom.Mesh):
                            vertices = np.asarray(UsdGeom.Mesh(child).GetPointsAttr().Get(), dtype=np.float64)
                            relative = inverse @ self._world(str(child.GetPath()))
                            parts.append(vertices @ relative[:3, :3].T + relative[:3, 3])
                    if not parts:
                        raise ValueError(f"RUBRIC_COLLISION_GEOMETRY_UNSUPPORTED: {collider_path}")
                    points = np.concatenate(parts)
                    representation = "collision_descendant_mesh_vertices"
                if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
                    raise ValueError(f"RUBRIC_COLLISION_GEOMETRY_INVALID: {collider_path}")
                self.geometry[collider_path] = {**geometry_binding, "points": points,
                                               "representation": representation, "analytic": analytic}
            self.objects.append({**spec, **binding, "colliders": collider_paths})
        identities = [o["prim"] for o in self.objects]
        if len(set(identities)) != len(identities):
            raise ValueError("duplicate rubric object identity")
        missing = set(plan["rubric_prim_references"]) - set(identities)
        if missing:
            raise ValueError(f"RUBRIC_OBJECT_COVERAGE_FAILED: {sorted(missing)}")
        self.collider_sets = {o["prim"]: set(o["colliders"]) for o in self.objects}
        for pair in plan["contacts"]:
            for role in ["source", "target"]:
                if not self.collider_sets.get(pair[role]):
                    raise ValueError(f"RUBRIC_CONTACT_GEOMETRY_MISSING: {pair[role]}")
        self.contacts = ContactAccumulator(plan["contacts"], self.collider_sets)
        if subscribe and plan["contacts"]:
            from omni.physx import get_physx_simulation_interface
            from pxr import PhysicsSchemaTools
            from omni.physx.bindings._physx import ContactEventType
            def callback(headers, data):
                try:
                    for header in headers:
                        a, b = [str(PhysicsSchemaTools.intToSdfPath(p)) for p in (header.collider0, header.collider1)]
                        lost = header.type == ContactEventType.CONTACT_LOST
                        points = data[int(header.contact_data_offset):int(header.contact_data_offset) + int(header.num_contact_data)]
                        impulse = sum((array(p.impulse) for p in points), np.zeros(3))
                        if not np.isfinite(impulse).all():
                            raise ValueError("nonfinite PhysX contact impulse")
                        touching = any(float(p.separation) <= 0 or np.linalg.norm(array(p.impulse)) > 0 for p in points)
                        self.contacts.report(a, b, impulse, lost=lost, touching=touching)
                except Exception as error:
                    self.error = error
            self.subscription = get_physx_simulation_interface().subscribe_contact_report_events(callback)
            if self.subscription is None:
                raise RuntimeError("RUBRIC_CONTACT_SUBSCRIPTION_UNAVAILABLE")

    def _world(self, path):
        return np.asarray(self.cache.GetLocalToWorldTransform(self.stage.GetPrimAtPath(path))).T.copy()

    def _binding(self, path, owner):
        local = self._world(path)
        if owner:
            if owner not in self.views:
                if self.sim_view is None:
                    import omni.physics.tensors as tensors
                    self.sim_view = tensors.create_simulation_view("torch")
                    self.sim_view.set_subspace_roots("/")
                view = self.sim_view.create_rigid_body_view(owner)
                if view._backend is None or view.count != 1:
                    raise ValueError(f"RUBRIC_BODY_VIEW_UNAVAILABLE: {owner}")
                self.views[owner] = view
            # Both matrices come from the same Stage, cancelling ancestor motion.
            authored_pose = pose_from_matrix(self._world(owner))
            rigid_world = matrix_from_pose(np.r_[authored_pose[:3], authored_pose[4:], authored_pose[3]])
            # PhysX actor transforms have no scale. Keep all authored scale in
            # the local geometry transform instead of accidentally cancelling it.
            local = np.linalg.inv(rigid_world) @ local
        return {"owner": owner, "local": local, "sample_prim": path}

    def manifest(self):
        objects = [{k: v for k, v in o.items() if k != "local"} |
                   {"owner_to_semantic_matrix": o["local"].tolist(),
                    "pose_source": "physx_body_with_fixed_local_transform" if o["owner"] else "live_static_usd_xform"}
                   for o in self.objects]
        return {**self.plan, "objects": objects, "physics_dt": self.physics_dt,
                "pose_order": "xyz_wxyz", "coordinate_frame": "world",
                "contact_force_convention": "target_on_source",
                "contact_sampling": "sum_reported_impulses_over_observed_physics_interval",
                "geometry": {p: {"owner": g["owner"], "representation": g["representation"],
                                  "vertex_count": len(g["points"])} for p, g in self.geometry.items()}}

    def _kinematics(self):
        self.cache.Clear()
        owners = {}
        for path, view in self.views.items():
            owners[path] = (matrix_from_pose(array(view.get_transforms())[0]),
                            array(view.get_velocities())[0])
        def transform(binding):
            if binding["owner"]:
                return owners[binding["owner"]][0] @ binding["local"]
            return self._world(binding["sample_prim"])
        bounds = {}
        for path, g in self.geometry.items():
            matrix = transform(g)
            if g["analytic"]:
                a = g["analytic"]
                linear = matrix[:3, :3]
                if a["type"] == "Sphere":
                    half = a["radius"] * np.linalg.norm(linear, axis=1)
                else:
                    axial = np.abs(linear[:, a["axis"]]) * a["height"] / 2
                    radial = linear if a["type"] == "Capsule" else np.delete(linear, a["axis"], axis=1)
                    half = axial + a["radius"] * np.linalg.norm(radial, axis=1)
                bounds[path] = np.stack([matrix[:3, 3] - half, matrix[:3, 3] + half])
            else:
                points = g["points"] @ matrix[:3, :3].T + matrix[:3, 3]
                bounds[path] = np.stack([points.min(axis=0), points.max(axis=0)])
        rows = []
        for obj in self.objects:
            matrix = transform(obj)
            velocity = np.zeros(6)
            if obj["owner"]:
                body_matrix, body_velocity = owners[obj["owner"]]
                velocity[3:] = body_velocity[3:]
                velocity[:3] = body_velocity[:3] + np.cross(velocity[3:], matrix[:3, 3] - body_matrix[:3, 3])
            row = {"root_pose": pose_from_matrix(matrix), "root_velocity": velocity.astype(np.float32)}
            if obj["colliders"]:
                pieces = np.asarray([bounds[p] for p in obj["colliders"]])
                row["world_aabb"] = np.stack([pieces[:, 0].min(axis=0), pieces[:, 1].max(axis=0)]).astype(np.float32)
                row["collider_world_aabb"] = pieces.astype(np.float32)
            rows.append(row)
        return rows

    def initial_sample(self):
        result = self._kinematics()
        self.contacts.reset_interval()
        self.active = True
        return result

    def begin_physics_step(self):
        if self.active:
            self.contacts.begin_step()

    def end_physics_step(self):
        if self.error:
            raise RuntimeError("RUBRIC_CONTACT_CALLBACK_FAILED") from self.error
        if self.active:
            self.contacts.end_step()

    def sample(self):
        if self.error:
            raise RuntimeError("RUBRIC_CONTACT_CALLBACK_FAILED") from self.error
        return self._kinematics(), self.contacts.consume(self.physics_dt)

    def close(self):
        self.active = False
        self.subscription = None
        self.views.clear()
