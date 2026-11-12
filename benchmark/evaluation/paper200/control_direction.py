"""Live geometry direction -> ordinary IK/guards/actuator diagnostic."""
import argparse,copy,os
from pathlib import Path
import numpy as np
from .common import read,write,sha,now,digest,PILOT_SEED


def prepare(cfg):
    if cfg.get('purpose')!='control_direction':return
    root=Path(cfg['collection_root']);out=Path(cfg['output'])
    if not (root/'STOP').exists() or (root/'manifests/FROZEN.json').exists():raise ValueError('control requires stopped development')
    path=out/'episode/resolved_attack_profile/all_attack_definitions.json';profile=read(path)
    selected=[a for a in profile['attack']['attacks'] if a.get('enabled') and a.get('effect_enabled',True)]
    if len(selected)!=1:raise ValueError('direction control needs one selected attack')
    attack=selected[0]
    if attack['stage']!='control_ee_target' or 'direction_contract' not in attack['parameters']:
        raise ValueError('direction diagnostic requires registered geometry direction')
    before=copy.deepcopy(attack);write(path.with_name('all_attack_definitions.original.json'),profile)
    attack['trigger']={'all':[{'type':'field','field':'paper200_probe.first_model_call','op':'truthy'}]}
    write(path,profile)
    contract_path=out/'runner_judge_contract.json';contract=read(contract_path)
    write(out/'original_runner_judge_contract.json',contract)
    for item in contract['attacks']:
        for definition in item['profile']['attack']['attacks']:
            if definition['id']==attack['id']:definition['trigger']=copy.deepcopy(attack['trigger'])
    scope={'schema':'paper200-direction-control/v2','created':now(),'original_attack':before,
        'diagnostic_attack':attack,'trigger_overridden':True,'camera_overridden':False,
        'attack_parameters_changed':False,'duration_changed':False,
        'original_condition_opportunity_eligible':False,'formal_or_assisted_matrix_member':False,
        'request_count':16,'action_hz':10,
        'source_original_contract_sha256':sha(out/'original_runner_judge_contract.json'),
        'reason':'isolate live physical direction resolution and hold/restoration from unreliable natural manipulation opportunity',
        'scope':'actual geometry witness, real neural model, nominal zero EE motion through original IK/clamp/Agent/actuator chain'}
    contract['diagnostic_control']=scope;write(contract_path,contract)
    write(out/'episode/DIRECTION_CONTROL.json',scope)


def execute(backend,phase,epoch,carrying):
    if backend.cfg['purpose']!='control_direction':raise ValueError('wrong control purpose')
    if getattr(backend,'_direction_control_done',False):
        return {'status':'DEVELOPER_CONTROL_ALREADY_CAPTURED','not_original_conditions':True,'carrying':carrying}
    backend._direction_control_done=True;backend.phase=phase;backend.epoch=epoch
    from .execution_boundary import append,apply_ordinary_ik
    start=backend.now()
    for index in range(16):
        response=backend.infer(phase,epoch,False,model_phase='ARM_APPROACH',phase_step=index)
        request_id=response[backend.r['VLA_REQUEST_ID_KEY']]
        action=np.zeros(10,np.float32);action[9]=float(backend.c._actual_gripper_closed_fraction())
        append(backend,{'event':'controlled_direction_proposal','purpose':'control_direction',
            'request_id':request_id,'physics_step':int(backend.c.physics_step),'sim_time_s':backend.now(),
            'nominal_action':action.tolist(),'model_actions':np.asarray(response['actions']).tolist(),
            'proposal_source':'zero nominal EE delta and measured gripper opening; no actuator guard bypass'})
        attacked=backend.prepare_ee_command(action,request_id,'ARM_APPROACH')
        apply_ordinary_ik(backend,attacked)
        backend.step(attacked,request_id=request_id,source='developer_control_direction')
        append(backend,{'event':'controlled_direction_physical_receipt','request_id':request_id,
            'physics_step':int(backend.c.physics_step),'sim_time_s':backend.now(),
            'gripper_world_position':np.asarray(backend.c.link_transform(backend.c.gripper_body)[0]).tolist(),
            'assurance_receipt_ids':backend.c.policy_execution_events[-1]['assurance_receipt_ids']})
    return {'status':'DEVELOPER_DIRECTION_CONTROL_COMPLETE','simulation_seconds':backend.now()-start,
        'inference_calls':16,'carrying':False,'not_original_conditions':True,'formal_score':None}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path)
    p.add_argument('--question',required=True,type=int,choices=[42,44,45,47]);p.add_argument('--name',required=True)
    p.add_argument('--gpu',default=0,type=int);a=p.parse_args();root=a.root.resolve()
    if not (root/'STOP').exists() or (root/'manifests/FROZEN.json').exists():raise ValueError('control outside stopped development')
    q='Q%d'%a.question;out=root/'pilot/controls/direction'/q/a.name;out.mkdir(parents=True,exist_ok=False)
    job={'question':q,'system':'groot_agent_off','variant':'arm_control_01','seed':PILOT_SEED,
        'purpose':'control_direction','attempts':1,'owner':'developer:%s'%os.getpid(),'output':str(out),
        'run_id':digest(['control_direction',q,a.name,PILOT_SEED])[:24]}
    write(out/'job.json',job);write(out/'CONTROL_SCOPE.json',{'purpose':'control_direction',
        'original_task_conditions':False,'formal_matrix_member':False,'attack_trigger_overridden':True,
        'attack_parameters_changed':False,'request_count':16,'action_hz':10,'real_model':True,
        'controller_proposal_override':'zero nominal EE delta; normal attack, IK, clamp, Agent and actuator chain'})
    from . import launch
    original_write=launch.write
    def configure(path,value):
        if Path(path)==out/'config.json':value=dict(value,action_hz=10,development_control='control_direction')
        return original_write(path,value)
    launch.write=configure
    try:
        result=launch.run_job(root,job,a.gpu);write(out/'CONTROL_COLLECTION_RESULT.json',result)
        print('DIRECTION_CONTROL',out,flush=True)
    except Exception as error:
        import traceback
        write(out/'FAILURE.json',{'error':str(error),'traceback':traceback.format_exc()});raise
    finally:launch.write=original_write


if __name__=='__main__':main()
