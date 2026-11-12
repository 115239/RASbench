"""Real-scene, real-model visual boundary diagnostics; never formal episodes."""
import argparse,copy,os,json
from pathlib import Path
import numpy as np
from .common import read,write,sha,now,digest,PILOT_SEED


def prepare(cfg):
    if cfg.get('purpose')!='control_visual':return
    root=Path(cfg['collection_root']);out=Path(cfg['output'])
    if not (root/'STOP').exists() or (root/'manifests/FROZEN.json').exists():raise ValueError('control requires stopped development')
    path=out/'episode/resolved_attack_profile/all_attack_definitions.json';profile=read(path)
    selected=[a for a in profile['attack']['attacks'] if a.get('enabled') and a.get('effect_enabled',True)]
    if len(selected)!=1:raise ValueError('visual control needs one selected attack')
    attack=selected[0]
    material=(attack['stage']=='immediate_visual_material' and attack['handler']=='install_visual_material_override')
    if not material and (attack['stage']!='request_image' or attack['handler']!='paper200_projected_patch'):raise ValueError('unexpected visual stage')
    before=copy.deepcopy(attack);write(path.with_name('all_attack_definitions.original.json'),profile)
    if not material:
        attack['trigger']={'all':[{'type':'field','field':'paper200_probe.first_model_call','op':'truthy'}]}
    write(path,profile)
    # Bind the diagnostic expected contract to the actual private trigger. Never
    # relabel it as natural opportunity acceptance under the original contract.
    contract_path=out/'runner_judge_contract.json';contract=read(contract_path)
    write(out/'original_runner_judge_contract.json',contract)
    for item in contract['attacks']:
        for definition in item['profile']['attack']['attacks']:
            if definition['id']==attack['id']:definition['trigger']=copy.deepcopy(attack['trigger'])
    scope={'schema':'paper200-visual-control/v2','created':now(),'original_attack':before,
        'diagnostic_attack':attack,'trigger_overridden':not material,'camera_overridden':True,
        'visual_modality':'persistent_usd_material' if material else 'projected_image',
        'attack_parameters_changed':False,'duration_changed':False,
        'original_condition_opportunity_eligible':False,'formal_or_assisted_matrix_member':False,
        'source_original_contract_sha256':sha(out/'original_runner_judge_contract.json'),
        'scope':('real USD material binding and hold until diagnostic stop, ordinary renderer pixels to actual neural model; original duration is not shortened and natural expiry/120s hold is not claimed' if material else 'real USD render, masked image attack, actual neural model input/response; held robot commands for isolated visibility control')}
    contract['diagnostic_control']=scope;write(contract_path,contract)
    write(out/'episode/VISUAL_CONTROL.json',scope)


def configure_camera(c,cfg,runtime):
    if cfg.get('purpose')!='control_visual':return
    from pxr import Usd,UsdGeom
    out=Path(cfg['output']);scope=read(out/'episode/VISUAL_CONTROL.json')
    params=scope['original_attack']['parameters'];q=int(cfg['task']['question'][1:])
    if scope.get('visual_modality')=='persistent_usd_material':
        c.cameras[0].add_instance_id_segmentation_to_frame({'colorize':False})
    targets=params.get('target_prims') or [params['target_prim']]
    if q==92:
        # Front-facing door near the explicitly named chopping board/right
        # island. The full profile still covers all actual visible door fronts.
        targets=[p for p in targets if '/E_door012_17/' in p]
    if not targets:raise ValueError('diagnostic camera target missing')
    bbox=UsdGeom.BBoxCache(Usd.TimeCode.Default(),['default','render','proxy'])
    bounds=[bbox.ComputeWorldBound(c.final_stage.GetPrimAtPath(p)).ComputeAlignedRange() for p in targets]
    minimum=np.min([np.asarray(b.GetMin()) for b in bounds],axis=0)
    maximum=np.max([np.asarray(b.GetMax()) for b in bounds],axis=0);center=(minimum+maximum)/2
    offset=np.array([-.85,-.15,.10]) if q==92 else np.array([-.20,-.20,.25])
    eye=center+offset;prior=c.cameras[0].get_world_pose();original=c.update_cameras
    def update():
        original();c.cameras[0].set_world_pose(position=eye,orientation=runtime['look_at'](eye,center))
    c.update_cameras=update;update()
    scope['camera']={'source':'actual USD rendered target bounds','target_prims':targets,
        'world_bounds':[minimum.tolist(),maximum.tolist()],
        'previous_pose':[np.asarray(v).tolist() for v in prior],
        'eye':eye.tolist(),'look_at':center.tolist(),'synthetic_image':False,
        'image_source':'ordinary physical renderer and policy observation function'}
    write(out/'episode/VISUAL_CONTROL.json',scope)


def execute(backend,phase,epoch,carrying):
    if backend.cfg['purpose']!='control_visual':raise ValueError('wrong control scope')
    if getattr(backend,'_visual_control_done',False):
        return {'status':'DEVELOPER_CONTROL_ALREADY_CAPTURED','not_original_conditions':True,'carrying':carrying}
    backend._visual_control_done=True;backend.phase=phase;backend.epoch=epoch
    start=backend.now()
    for index in range(8):
        response=backend.infer(phase,epoch,carrying,phase_step=index)
        request_id=response[backend.r['VLA_REQUEST_ID_KEY']]
        scope=read(Path(backend.cfg['output'])/'episode/VISUAL_CONTROL.json')
        if scope.get('visual_modality')=='persistent_usd_material':
            frame=backend.c.cameras[0].get_current_frame()
            seg=frame.get('instance_id_segmentation',{})
            if 'data' not in seg:raise ValueError('material diagnostic instance visibility unavailable')
            ids=np.asarray(seg['data']);labels=seg.get('info',{}).get('idToLabels',{})
            meta={'request_id':int(request_id),'physics_step':int(backend.c.physics_step),
                  'target_prims':scope['original_attack']['parameters']['target_prims'],'idToLabels':labels,
                  'source':'actual rendered camera instance IDs; visibility evidence only, not an image attack'}
            path=Path(backend.cfg['output'])/'episode/material_visibility';path.mkdir(exist_ok=True)
            np.savez_compressed(path/('request_%06d.npz'%request_id),instance_ids=ids,
                                metadata_json=np.frombuffer(json.dumps(meta,sort_keys=True).encode(),dtype=np.uint8))
        # Real model outputs are retained, while this isolated diagnostic holds
        # the physical robot to keep camera geometry controlled for 4 seconds.
        backend.step(np.zeros(10,np.float32),request_id=request_id,source='developer_visual_control_hold')
    return {'status':'DEVELOPER_VISUAL_CONTROL_COMPLETE','simulation_seconds':backend.now()-start,
        'inference_calls':8,'carrying':carrying,'not_original_conditions':True,'formal_score':None}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path)
    p.add_argument('--question',required=True,type=int,choices=[45,48,92,96,97]);p.add_argument('--name',required=True)
    p.add_argument('--gpu',default=0,type=int);a=p.parse_args();root=a.root.resolve()
    if not (root/'STOP').exists() or (root/'manifests/FROZEN.json').exists():raise ValueError('control outside stopped development')
    q='Q%d'%a.question;out=root/'pilot/controls/visual'/q/a.name;out.mkdir(parents=True,exist_ok=False)
    job={'question':q,'system':'groot_agent_off','variant':'visual_01','seed':PILOT_SEED,
        'purpose':'control_visual','attempts':1,'owner':'developer:%s'%os.getpid(),'output':str(out),
        'run_id':digest(['control_visual',q,a.name,PILOT_SEED])[:24]}
    write(out/'job.json',job);write(out/'CONTROL_SCOPE.json',{'purpose':'real renderer and neural consumer diagnostic',
        'original_task_conditions':False,'formal_matrix_member':False,'attack_trigger_overridden':True,
        'attack_parameters_changed':False,'actual_renderer':True,'real_model':True,
        'controller_proposal_override':'zero nominal motion for visibility diagnostic'})
    from .launch import run_job
    try:
        result=run_job(root,job,a.gpu);write(out/'CONTROL_COLLECTION_RESULT.json',result)
        print('VISUAL_CONTROL',out,flush=True)
    except Exception as error:
        import traceback
        write(out/'FAILURE.json',{'error':str(error),'traceback':traceback.format_exc()});raise


if __name__=='__main__':main()
