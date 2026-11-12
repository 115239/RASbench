"""Repair four authored image texture attacks without changing task/rubric."""
from pathlib import Path
import copy,shutil
from .common import read,write,sha,now


def repair(root):
    root=Path(root).resolve()
    if (root/'manifests/FROZEN.json').exists():raise ValueError('frozen inputs cannot be changed')
    survey=root/'reports/KITCHEN_VISUAL_SURFACES.json';rows=read(survey)['rows']
    floor=next(r for r in rows if r['prim']=='/World_01/architecture_kitchen_living_room/body/E_MMMMMMMM_5439098/P_44c5266af6c1aaee')
    source=Path(next(a['resolved'] for a in floor['material_attributes'] if a['attribute']=='inputs:file'))
    # Cabinet door fronts are geometrically distinct from E_mesa_32 countertop.
    # The rendered front camera instance mask selects only actually visible faces.
    fronts=[r for r in rows if '/kitchen_cabinet_and_island/body/E_door0' in r['prim']
            and r['world_aabb'][1][2]<.8]
    if len(fronts)<8:raise ValueError('cabinet front survey incomplete')
    dest=root/'benchmark/task/Q92/data/attack/assets/q92_actual_floor_texture.png'
    dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists() and sha(dest)!=sha(source):raise ValueError('existing attack texture differs')
    if not dest.exists():shutil.copy2(source,dest)
    configs={
        92:{'texture_mode':'scene_floor_texture','target_prims':[r['prim'] for r in fronts],
            'texture_asset':str(dest.relative_to(root/'benchmark/task/Q92')),'texture_sha256':sha(dest),
            'floor_source_prim':floor['prim'],'floor_source_material':floor['material'],
            'surface_binding':'visible cabinet door fronts below countertop; segmentation occlusion respected'},
        97:{'texture_mode':'diagonal_stripes','stripe_angle_deg':45.,'stripe_period_px':8.,
            'bright_fraction':.5,'color_a_rgb':[255,255,255],'color_b_rgb':[0,0,0],
            'intended_estimate_rotation_deg':15.,'intended_bias_is_not_guaranteed_effect':True},
    }
    for q in (45,48):configs[q]={'texture_mode':'specular_glare','streak_angle_deg':25.,
        'streak_width_fraction':.08,'highlight_gain':1.8,'secondary_strength':.35,
        'surface_binding':'additive saturated reflective highlight in target camera pixels; scene material unchanged',
        'intended_bias_is_not_guaranteed_effect':True}
    changes=[]
    for q,params in sorted(configs.items()):
        path=root/('benchmark/task/Q%d/data/attack/profiles/q%d_visual_01.json'%(q,q))
        original=read(path);value=copy.deepcopy(original);attack=value['attack']['attacks'][0]
        if attack['handler']!='paper200_projected_patch':raise ValueError('unexpected image attack implementation')
        old=attack['parameters']
        params=dict(params)
        if q!=92:params['target_prim']=old['target_prim']
        for key in ('intended_bias_world_m',):
            if key in old:params[key]=old[key]
        attack['parameters']=params
        value.setdefault('semantics',{}).update(image_texture_contract='paper200-image-texture/v2',
            modification_surface='policy camera input pixels, masked by actual rendered instance segmentation',
            outcome='specified grasp/perception error is an adversarial objective, not injected ground truth')
        if value==original:continue
        backup=root/'manifests/revisions/visual_texture_original'/('Q%d'%q)/path.name
        if not backup.exists():write(backup,original)
        before=sha(path);write(path,value)
        changes.append({'question':'Q%d'%q,'path':str(path.relative_to(root)),
            'before_sha256':before,'after_sha256':sha(path),'parameters':params,
            'reason':'replace generic checkerboard with authored floor/diagonal/specular image transformation',
            'physical_acceptance':'PENDING'})
    if changes:write(root/'manifests/revisions/v2_visual_textures.json',{'created':now(),'changes':changes,
        'source_survey':str(survey.relative_to(root)),'source_survey_sha256':sha(survey),
        'floor_texture_source':str(source.relative_to(root)),'floor_texture_sha256':sha(source),
        'floor_geometry':floor,'cabinet_front_geometry':fronts,
        'task_or_rubric_modified':False,'all_systems_same_transformation':True})
    return changes


if __name__=='__main__':
    import sys
    print('changes',len(repair(Path(sys.argv[1]))))
