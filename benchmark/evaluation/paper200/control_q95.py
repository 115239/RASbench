"""Explicit developer control, real Q95 robot/model, original attack and guards.

The proposed arm motion is zero; a measured partial finger closure supplies the
real original opportunity. This validates the execution interface, never the
model's ability to pick the pear or an original-condition task score.
"""
import argparse
import os
from pathlib import Path
import time
import numpy as np
from .common import read,write,now,digest,PILOT_SEED


def manipulate(backend,phase,epoch,seconds):
    cfg=backend.cfg;root=Path(cfg['collection_root'])
    if cfg['purpose']!='control_q95' or cfg['task']['question']!='Q095' or (root/'manifests/FROZEN.json').exists():
        raise ValueError('Q95 controlled proposals require isolated unfrozen developer scope')
    backend.phase=phase;backend.epoch=epoch
    calls=0;start=backend.now();closing=None
    for index in range(16):
        response=backend.infer(phase,epoch,False,model_phase='ARM_APPROACH',phase_step=index)
        rid=response[backend.r['VLA_REQUEST_ID_KEY']]
        measured=float(backend.c._actual_gripper_closed_fraction())
        action=np.zeros(10,np.float32)
        if index==2:closing=min(.95,measured+.10)
        action[9]=measured if closing is None else closing
        before=backend.c.link_transform(backend.c.gripper_body)
        record={'event':'controlled_proposal','purpose':'control_q95','not_model_task_success':True,
            'source':'bounded physical finger closure at current arm pose; no object teleport or attack-trigger override',
            'request_id':rid,'physics_step':int(backend.c.physics_step),'sim_time_s':backend.now(),
            'model_actions':np.asarray(response['actions']).tolist(),'proposal':action.tolist(),
            'measured_gripper_close':measured,'gripper_world_position':np.asarray(before[0]).tolist()}
        from .execution_boundary import append
        append(backend,record)
        action=backend.prepare_ee_command(action,rid,'ARM_APPROACH')
        from .execution_boundary import apply_ordinary_ik
        apply_ordinary_ik(backend,action)
        backend.step(action,request_id=rid,source='developer_control_q95')
        append(backend,{'event':'controlled_physical_receipt','request_id':rid,
            'physics_step':int(backend.c.physics_step),'sim_time_s':backend.now(),
            'measured_gripper_close':float(backend.c._actual_gripper_closed_fraction()),
            'gripper_world_position':np.asarray(backend.c.link_transform(backend.c.gripper_body)[0]).tolist(),
            'assurance_receipt_ids':backend.c.policy_execution_events[-1]['assurance_receipt_ids']})
        calls+=1
    return {'status':'DEVELOPER_CONTROL_COMPLETE','simulation_seconds':backend.now()-start,
            'inference_calls':calls,'carrying':False,'physical_success':False,'grasp_success':False,
            'not_an_original_condition_task_trial':True}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--gpu',type=int,default=4);p.add_argument('--name',required=True)
    p.add_argument('--system',choices=['groot_agent_off','groot_agent_on','pi05_agent_off'],default='groot_agent_off')
    a=p.parse_args();root=a.root.resolve()
    if not (root/'STOP').exists() or (root/'manifests/FROZEN.json').exists():
        raise ValueError('developer control requires stopped, unfrozen collection')
    out=root/'pilot/controls/Q95'/a.system/a.name;out.mkdir(parents=True,exist_ok=False)
    job={'question':'Q95','system':a.system,'variant':'arm_control_01','seed':PILOT_SEED,
         'purpose':'control_q95','attempts':1,'owner':'developer:%s'%os.getpid(),'output':str(out),
         'run_id':digest(['control_q95',a.system,a.name,PILOT_SEED])[:24]}
    write(out/'job.json',job)
    write(out/'CONTROL_SCOPE.json',{'purpose':'developer-only execution-chain physical control',
        'original_task_conditions':False,'formal_or_assisted_matrix_member':False,
        'trigger_changed':False,'attack_payload_changed':False,'real_model_called':True,
        'real_physics':True,'proposal_override':'partial finger closure; zero nominal arm delta',
        'created':now()})
    from . import launch
    original_write=launch.write
    def control_config(path,value):
        if Path(path)==out/'config.json':
            value=dict(value,action_hz=10,development_control='control_q95')
        return original_write(path,value)
    launch.write=control_config
    try:
        result=launch.run_job(root,job,a.gpu)
        write(out/'CONTROL_COLLECTION_RESULT.json',result)
        print('CONTROL_COLLECTION',out,flush=True)
    except Exception as error:
        import traceback
        write(out/'FAILURE.json',{'error':str(error),'traceback':traceback.format_exc()})
        raise
    finally:launch.write=original_write


if __name__=='__main__':main()
