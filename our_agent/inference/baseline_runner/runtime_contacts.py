"""Optional exact contact offsets from the existing selected-ONNX Nav2 wrapper."""
import os
from common import VLN, LOCOMOTION, binding, read, write, sha

def bind_contacts(stage, robot_path, output):
    from pxr import UsdPhysics, PhysxSchema
    cfg = read(os.environ["THREE_SCENES_RUN_CONFIG"])
    profile = cfg.get("robot_contact_profile", "official")
    source = VLN / "nav2_outbound/market_vln_nav2/isaac_nav2_runtime_wrapper.py"
    expected = "e3d92f124d1d4a327862bc4b27208fc499a13f1d3ecd33525c5691cf74dd9c7c"
    if profile == "navigation_bound" and sha(LOCOMOTION) != expected:
        raise ValueError("NAVIGATION_CONTACT_PROFILE_POLICY_MISMATCH")
    rows = []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if not path.startswith(robot_path + "/") or not prim.HasAPI(UsdPhysics.CollisionAPI): continue
        api = PhysxSchema.PhysxCollisionAPI(prim)
        before = {"contact": api.GetContactOffsetAttr().Get(), "rest": api.GetRestOffsetAttr().Get()}
        if profile == "navigation_bound":
            api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
            api.CreateContactOffsetAttr().Set(.05)
            api.CreateRestOffsetAttr().Set(.005)
        rows.append({"prim": path, "before": before,
                     "after": {"contact": api.GetContactOffsetAttr().Get(), "rest": api.GetRestOffsetAttr().Get()}})
    if not rows: raise ValueError("NO_ROBOT_COLLISION_SHAPES")
    write(output / "ROBOT_CONTACT_PROFILE.json", {"profile": profile, "policy": binding(LOCOMOTION),
          "reference_wrapper": binding(source), "robot_only": True, "source_usd_modified": False, "shapes": rows})
