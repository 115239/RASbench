"""Retain frozen Q17 alternatives in the real PhysX recording surface."""
import os
from common import read, write


def bind_candidate_evidence(stage, owners, selected_owner, output):
    config = read(os.environ["THREE_SCENES_RUN_CONFIG"])
    profile = config.get("candidate_evidence_profile", "official")
    report = {"profile": profile, "source_scene_modified": False, "session_layer_only": True,
              "selected_dynamic_object": selected_owner, "objects": []}
    if profile == "official":
        write(output / "CANDIDATE_EVIDENCE_BINDING.json", report)
        return
    if profile != "kinematic_frozen" or config["task"]["question"] != "Q17":
        raise ValueError("kinematic_frozen evidence profile is only defined for Q17")
    if len(owners) != 47 or selected_owner in owners.values() or len(set(owners.values())) != 47:
        raise ValueError("Q17 frozen candidate identities differ from the reviewed 47 alternatives")
    from pxr import UsdGeom, UsdPhysics
    for object_id, path in sorted(owners.items()):
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            raise ValueError("Q17 candidate has no authored rigid body: " + path)
        body = UsdPhysics.RigidBodyAPI(prim)
        before = {"rigid_body_enabled": body.GetRigidBodyEnabledAttr().Get(),
                  "kinematic_enabled": body.GetKinematicEnabledAttr().Get()}
        if before["rigid_body_enabled"] is not False:
            raise ValueError("Q17 candidate was not frozen by the official runner: " + path)
        # A registered kinematic body remains fixed, as the official freeze
        # intends, while exposing measured root poses and paired contact forces.
        # No trajectories, contact values, or object aliases are manufactured.
        body.CreateKinematicEnabledAttr().Set(True)
        body.CreateRigidBodyEnabledAttr().Set(True)
        pose = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)
        report["objects"].append({"object_id": object_id, "prim": path, "before": before,
            "after": {"rigid_body_enabled": True, "kinematic_enabled": True},
            "initial_world_transform": [[float(v) for v in row] for row in pose]})
    report["recording_source"] = "LIVE_PHYSX_KINEMATIC_POSES_AND_CONTACT_MATRIX"
    report["physical_validation"] = "PENDING_EPISODE_AND_FROZEN_SCORER"
    write(output / "CANDIDATE_EVIDENCE_BINDING.json", report)
