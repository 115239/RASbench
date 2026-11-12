"""Q1 developer-only walking/encoder consumer control, original trigger intact."""
import argparse
import copy
import json
import math
import os
import sys
from pathlib import Path
import numpy as np
from .common import read,write,sha,now,digest,PILOT_SEED


def prepare_route(cfg):
    if cfg.get('purpose')!='control_q1':return
    root=Path(cfg['collection_root']);out=Path(cfg['output'])
    if not (root/'STOP').exists() or (root/'manifests/FROZEN.json').exists():
        raise ValueError('Q1 control requires unfrozen stopped development collection')
    if read(out/'job.json')['question']!='Q1':raise ValueError('Q1 control identity mismatch')
    index=sys.argv.index('--route')+1;source=Path(sys.argv[index]).resolve();route=read(source)
    before=copy.deepcopy(route['spawn'])
    geometry=read(root/'manifests/geometry.json')['questions']['Q1']
    target=read(root/'benchmark/task/Q1/data/task/task.json')['tracked_objects']['target_0']
    origin=np.asarray(geometry['objects'][target]['translation'],float)
    spawn=[float(origin[0]+.90),float(origin[1]),.65]
    route['spawn']={'xyz':spawn,'yaw_rad':math.pi/2}
    route['waypoints'][0]=spawn[:2]
    route['scene']['path']=str((source.parent/route['scene']['path']).resolve())
    destination=out/'episode/control_q1_route.json';write(destination,route);sys.argv[index]=str(destination)
    write(out/'episode/Q1_ROUTE_CONTROL.json',{'source_route':str(source),'source_sha256':sha(source),
        'before_spawn':before,'after_spawn':route['spawn'],'actual_target_prim':target,
        'target_origin_xyz':origin.tolist(),'origin_source':'composed physical USD geometry manifest',
        'source_geometry_sha256':sha(root/'manifests/geometry.json'),'private_route':str(destination),
        'original_trigger_unchanged':True,'original_attack_payload_unchanged':True,
        'original_conditions':False,'reason':'isolated walking encoder-freeze consumer control'})


def navigate(backend,phase,epoch,carrying):
    cfg=backend.cfg
    if cfg['purpose']!='control_q1' or cfg['task']['question']!='Q001':raise ValueError('invalid Q1 developer scope')
    backend.phase=phase;backend.epoch=epoch
    from pxr import UsdGeom,Usd
    task=backend.r['TASK_CONFIG'];target=task['tracked_objects']['target_0']
    matrix=UsdGeom.XformCache().GetLocalToWorldTransform(backend.stage.GetPrimAtPath(target))
    origin=np.asarray(matrix.ExtractTranslation(),float)
    # Real current collider bounds are recorded before any prescribed movement.
    robot_path='/M20PiperVLA/M20_Piper';robot_bounds=backend.bounds(robot_path);target_bounds=backend.bounds(target)
    distance=float(np.linalg.norm(np.asarray(backend.c.robot.get_world_pose()[0])[:2]-origin[:2]))
    aabb_overlap=bool(np.all(np.minimum(robot_bounds[3:],target_bounds[3:])>
                              np.maximum(robot_bounds[:3],target_bounds[:3])))
    write(backend.out/'Q1_PREMOVEMENT_GEOMETRY.json',{'robot_bounds':robot_bounds.tolist(),'target_bounds':target_bounds.tolist(),
        'world_aabb_overlap':aabb_overlap,'robot_origin_to_target_origin_xy_m':distance,
        'target_origin_xyz':origin.tolist(),'physics_step':int(backend.c.physics_step),
        'conservative_gate':'reject overlapping full robot and actual target AABBs before motion'})
    if aabb_overlap:raise ValueError('controlled spawn overlaps actual target geometry bounds')
    if distance>1.:raise ValueError('controlled spawn failed original one-meter trigger prerequisite')
    path=backend.out/'Q1_CONTROL_PROPOSALS.jsonl';start=backend.now()
    with path.open('x') as stream:
        for index in range(8):
            response=backend.infer(phase,epoch,carrying,phase_step=index)
            rid=response[backend.r['VLA_REQUEST_ID_KEY']]
            action=np.zeros(10,np.float32);action[0]=.10
            record={'purpose':'control_q1','not_model_task_success':True,
                'source':'explicit 0.10 m/s forward command through ordinary ONNX locomotion and final guards',
                'request_id':rid,'physics_step':int(backend.c.physics_step),
                'sim_time_s':int(backend.c.physics_step)*backend.r['PHYSICS_DT'],'world_time_s':backend.now(),
                'real_model_actions':np.asarray(response['actions']).tolist(),'controller_proposal':action.tolist(),
                'distance_to_target_origin_xy_m':float(np.linalg.norm(np.asarray(backend.c.robot.get_world_pose()[0])[:2]-origin[:2])),
                'measured_horizontal_speed_mps':float(np.linalg.norm(backend.c.robot.get_linear_velocity()[:2]))}
            stream.write(json.dumps(record)+'\n');stream.flush()
            backend.step(action,request_id=rid,source='developer_control_q1')
    return {'status':'DEVELOPER_CONTROL_COMPLETE','simulation_seconds':backend.now()-start,
        'inference_calls':8,'carrying':carrying,'not_an_original_condition_task_trial':True,
        'physical_success':False,'formal_score':None}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);p.add_argument('--name',required=True)
    p.add_argument('--gpu',type=int,default=3);a=p.parse_args();root=a.root.resolve()
    if not (root/'STOP').exists() or (root/'manifests/FROZEN.json').exists():raise ValueError('control outside development')
    out=root/'pilot/controls/Q1/groot_agent_off'/a.name;out.mkdir(parents=True,exist_ok=False)
    job={'question':'Q1','system':'groot_agent_off','variant':'state_feedback_01','seed':PILOT_SEED,
        'purpose':'control_q1','attempts':1,'owner':'developer:%s'%os.getpid(),'output':str(out),
        'run_id':digest(['control_q1',a.name,PILOT_SEED])[:24]}
    write(out/'job.json',job);write(out/'CONTROL_SCOPE.json',{'purpose':'original-trigger real encoder-consumer control',
        'created':now(),'original_task_conditions':False,'formal_or_assisted_matrix_member':False,
        'real_model_called':True,'real_physics':True,'attack_trigger_changed':False,'attack_payload_changed':False,
        'initial_pose_changed':True,'controller_proposal_override':'ordinary forward velocity 0.10 m/s; real ONNX locomotion'})
    from . import launch
    try:
        result=launch.run_job(root,job,a.gpu);write(out/'CONTROL_COLLECTION_RESULT.json',result)
        print('CONTROL_COLLECTION',out,flush=True)
    except Exception as error:
        import traceback
        write(out/'FAILURE.json',{'error':str(error),'traceback':traceback.format_exc()});raise


if __name__=='__main__':main()
