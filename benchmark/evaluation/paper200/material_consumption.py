"""Read actual USD bindings, and join rendered pixels to the policy request."""
import hashlib
import numpy as np


def observe(c,*,stage,UsdShade,dt,images=None):
    active=c._visual_material_override
    if active is None or c.attack_boundary is None:return
    materials={}
    for path in active['affected_mesh_prims']:
        prim=stage.GetPrimAtPath(path)
        material,_=UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        materials[path]=str(material.GetPath()) if material and material.GetPrim().IsValid() else None
    record={'consumer':'usd_renderer_binding','receipt_status':'bindings_read',
        'physics_step':int(c.physics_step),'sim_time_s':c.physics_step*dt,
        'attack_id':active['attack_id'],'until_physics_step':active['until_step'],
        'original_materials':active['original_materials'],'actual_materials':materials,
        'override_material':str(active['material_path']),
        'texture_asset':active['texture_asset'],'texture_sha256':active['texture_sha256']}
    if images is not None:
        record.update(consumer='rendered_policy_observation',receipt_status='pixels_read',
            images={name:{'sha256':hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest(),
                          'shape':list(value.shape),'dtype':str(value.dtype)} for name,value in images.items()})
    c.attack_boundary.recorder.capture_consumption(record)
