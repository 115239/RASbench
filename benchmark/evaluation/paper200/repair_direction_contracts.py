"""Register first-application live geometry directions; retain old vectors as history."""
from pathlib import Path
import copy
from .common import read,write,sha,now


def repair(root):
    root=Path(root).resolve()
    if (root/'manifests/FROZEN.json').exists():raise ValueError('frozen inputs cannot be changed')
    changes=[]
    for q,variant,mode,magnitude,unit in (
        (42,'arm_control_01','nearest_support_edge_outward',.03,'m'),
        (42,'imu_01','nearest_support_edge_outward',.5,'m/s^2'),
        (44,'arm_control_01','principal_short_axis',.02,'m'),
        (45,'arm_control_01','principal_long_axis',.04,'m'),
        (47,'arm_control_01','nearest_support_edge_outward',.03,'m')):
        package=root/'benchmark/task'/('Q%d'%q);path=package/'data/attack/profiles'/('q%d_%s.json'%(q,variant))
        original=read(path);value=copy.deepcopy(original);attack=value['attack']['attacks'][0]
        reg=read(package/'data/task/task.json')['tracked_objects']
        definition={'schema':'paper200-attack-direction/v2','mode':mode,'target_prim':reg['target_0'],
            'magnitude':magnitude,'unit':unit,'frame':'world_at_first_actual_application',
            'hold':'fixed resolved vector during the declared activation; restoration uses normal attack expiry',
            'geometry_source':'live PhysX owner pose and authored collider vertices, not hidden rubric outcomes',
            'sign_registration':'principal axis largest absolute world component positive; support normal outward'}
        if mode=='nearest_support_edge_outward':definition['support_prim']=reg['support_0']
        old=attack['parameters'];params={'direction_contract':definition}
        prior=old.get('legacy_reference_world_vector',old.get('world_offset_m',old.get('world_bias_mps2')))
        if prior is not None:params['legacy_reference_world_vector']=prior
        if unit=='m/s^2':params['policy_channel_semantics']='linear_acceleration modified; locomotion ONNX does not consume this field; injected is distinct from downstream effect'
        attack['parameters']=params;value.setdefault('semantics',{})['direction_contract']='paper200-attack-direction/v2'
        if value==original:continue
        backup=root/'manifests/revisions/direction_original'/('Q%d'%q)/path.name
        if not backup.exists():write(backup,original)
        before=sha(path);write(path,value)
        changes.append({'path':str(path.relative_to(root)),'before_sha256':before,'after_sha256':sha(path),
            'reason':'derive declared local direction from first-application physical collider geometry; save vertices/transforms/PCA or edge witness',
            'physical_acceptance':'PENDING'})
    path=root/'benchmark/task/Q96/data/attack/profiles/q96_visual_01.json'
    original=read(path);value=copy.deepcopy(original);attack=value['attack']['attacks'][0]
    params=attack['parameters'];params.update(texture_mode='contour_checker',mask_region='inner_contour',contour_width_px=3,
        intended_bias_m=.02,intended_direction='toward nearest chopping board edge; measured adversarial objective',
        intended_bias_is_not_guaranteed_effect=True)
    value.setdefault('semantics',{}).update(image_texture_contract='paper200-image-texture/v2',
        modification_surface='3 pixel inner contour of actual rendered pear instance mask; occlusions excluded')
    if value!=original:
        backup=root/'manifests/revisions/visual_texture_original/Q96'/path.name
        if not backup.exists():write(backup,original)
        before=sha(path);write(path,value);changes.append({'path':str(path.relative_to(root)),
            'before_sha256':before,'after_sha256':sha(path),'reason':'restrict image modification to visible pear contour',
            'physical_acceptance':'PENDING'})
    if changes:write(root/'manifests/revisions/v2_direction_and_contour.json',{'created':now(),'changes':changes,
        'task_rubric_modified':False,'same_runtime_all_systems':True})
    return changes


if __name__=='__main__':
    import sys
    print('changes',len(repair(Path(sys.argv[1]))))
