"""Audit authored trigger roles without changing the scene or physics schema."""
def is_authored_trigger(attributes):
    return (attributes.get('market:interaction_type') == 'TRIGGER_ACTION'
            and attributes.get('market:collision_shape') == 'trigger_box'
            and attributes.get('market:asset_id') == 'procedural_trigger')

def audit_authored_triggers(stage, output):
    from pxr import Usd
    from common import write
    rows=[]
    for prim in Usd.PrimRange.Stage(stage):
        attrs={name:prim.GetAttribute(name).Get() for name in ['market:interaction_type','market:collision_shape','market:asset_id']}
        if not is_authored_trigger(attrs):continue
        schemas=list(prim.GetAppliedSchemas())
        rows.append({'prim':str(prim.GetPath()),'authored_role':attrs,'schemas':schemas,'physx_trigger': 'PhysxTriggerAPI' in schemas})
    write(output/'TRIGGER_SEMANTICS.json',{'scope':'read-only audit of authored trigger boxes','scene_or_physics_changes':False,'planner_ignores_only_actual_physx_triggers':True,'triggers':rows})
