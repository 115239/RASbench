"""Explicitly separate interface diagnostics from original-trigger experiments."""
from pathlib import Path
import copy,sys
from .common import read,write,sha,now


def prepare(cfg):
    if cfg.get('purpose')!='probe':return
    root=Path(cfg['collection_root'])
    if (root/'manifests/FROZEN.json').exists():raise ValueError('probes cannot alter frozen collection')
    out=Path(cfg['output']);job=read(out/'job.json')
    if job['purpose']!='probe' or cfg.get('assistance'):raise ValueError('invalid diagnostic scope')
    path=out/'episode/resolved_attack_profile/all_attack_definitions.json';profile=read(path)
    selected=[a for a in profile['attack']['attacks'] if a.get('enabled') and a.get('effect_enabled',True)]
    if len(selected)!=1:raise ValueError('probe requires one selected handler')
    attack=selected[0];original=copy.deepcopy(attack)
    write(path.with_name('all_attack_definitions.original.json'),profile)
    original_sha=sha(path)
    attack['trigger']={'all':[{'type':'field','field':'paper200_probe.first_model_call','op':'truthy'}]}
    write(path,profile)
    setup_path=root/'manifests/handler_probe_setups.json';setup=None
    if setup_path.exists():setup=read(setup_path).get(job['question']+':'+job['variant'])
    route_change=None
    if setup and setup.get('spawn'):
        index=sys.argv.index('--route')+1;source=Path(sys.argv[index]).resolve();route=read(source)
        before=copy.deepcopy(route['spawn'])
        route['spawn']=setup['spawn']
        route['scene']['path']=str((source.parent/route['scene']['path']).resolve())
        diagnostic_route=out/'episode/handler_probe_route.json';write(diagnostic_route,route)
        sys.argv[index]=str(diagnostic_route)
        route_change={'source_route':str(source),'source_sha256':sha(source),'before_spawn':before,
                      'after_spawn':route['spawn'],'diagnostic_route':str(diagnostic_route),'reason':setup['reason']}
    write(out/'episode/HANDLER_PROBE.json',{'time':now(),'purpose':'probe','original_attack':original,
        'diagnostic_attack':attack,'original_runtime_profile_sha256':original_sha,
        'diagnostic_runtime_profile_sha256':sha(path),'only_trigger_overridden':route_change is None,
        'diagnostic_initial_pose_change':route_change,'main_package_modified':False,
        'interpretation':'real model/physics boundary diagnostic; NOT original-condition activation, NOT main or assisted experiment',
        'trigger':'first actual model request after normal initialization; payload parameters and duration unchanged'})


def configure_camera(c,cfg,runtime):
    """Optional rendered diagnostic view; never changes original experiments."""
    if cfg.get('purpose')!='probe':return
    from pxr import Usd,UsdGeom
    import numpy as np
    root=Path(cfg['collection_root']);job=read(Path(cfg['output'])/'job.json')
    setups=root/'manifests/handler_probe_setups.json'
    setup=read(setups).get(job['question']+':'+job['variant'],{}) if setups.exists() else {}
    view=setup.get('camera')
    if not view:return
    if (root/'manifests/FROZEN.json').exists():raise ValueError('diagnostic camera after freeze')
    path=Path(cfg['output'])/'episode/HANDLER_PROBE.json';evidence=read(path)
    target=evidence['original_attack']['parameters']['target_prim']
    prim=c.final_stage.GetPrimAtPath(target)
    bounds=UsdGeom.BBoxCache(Usd.TimeCode.Default(),['default','render']).ComputeWorldBound(prim).ComputeAlignedRange()
    center=np.asarray(bounds.GetMidpoint(),float);eye=center+np.asarray(view['offset_xyz_m'],float)
    previous=c.cameras[0].get_world_pose();original=c.update_cameras
    def update():
        original()
        c.cameras[0].set_world_pose(position=eye,orientation=runtime['look_at'](eye,center))
    c.update_cameras=update
    update()
    evidence.update(only_trigger_overridden=False,diagnostic_camera_change={
        'camera_index':0,'before_position':np.asarray(previous[0]).tolist(),
        'before_orientation':np.asarray(previous[1]).tolist(),'after_position':eye.tolist(),
        'look_at':center.tolist(),'target_prim':target,'reason':view['reason'],
        'actual_renderer_rgb_used':True,'original_package_and_performance_conditions_changed':False})
    write(path,evidence)
