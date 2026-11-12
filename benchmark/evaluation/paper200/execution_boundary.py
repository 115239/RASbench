"""Nominal executable EE request -> attack -> ordinary IK -> existing guards."""
import json
import numpy as np


def closing_opportunity(*, nominal_close, measured_close, maximum_finger_opening, enabled):
    return bool(enabled and (nominal_close-measured_close)*2*maximum_finger_opening>.0001)


def append(backend,record):
    with (backend.out/'EE_COMMAND_CHAIN.jsonl').open('a') as stream:
        stream.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')


def cancel_at_stop(backend):
    """Cancel future attack bias without commanding motion after a safety stop.

    A stop is not a completed physical restoration. Keeping that distinction in
    evidence prevents a truncated attack from acquiring a fabricated restore.
    """
    tracker=getattr(backend,'_paper_world_target_offset',None)
    if tracker is None or not np.any(tracker.previous):return
    append(backend,{'event':'ee_offset_cancelled_at_stop',
        'physics_step':int(backend.c.physics_step),
        'sim_time_s':int(backend.c.physics_step)*backend.r['PHYSICS_DT'],
        'cancelled_future_bias_world_m':tracker.previous.tolist(),
        'physical_restoration_executed':False,
        'reason':'episode stopping; preserve ordinary stop and actuator guards'})
    tracker.previous[:]=0.


class WorldTargetOffset:
    """A held target bias is differenced before an incremental IK adapter.

    Integrating +2 cm on every request would move the target repeatedly. The
    integrated commanded target instead receives +2 cm once, holds that bias,
    then receives -2 cm on restoration. Physical tracking is measured separately.
    """
    def __init__(self):
        self.previous=np.zeros(3,np.float64)
        self.nominal_integral=np.zeros(3,np.float64)
        self.attacked_integral=np.zeros(3,np.float64)

    def adapt(self,nominal_world_delta,requested_world_offset):
        nominal=np.asarray(nominal_world_delta,np.float64)
        offset=np.asarray(requested_world_offset,np.float64)
        if nominal.shape!=(3,) or offset.shape!=(3,) or not np.isfinite([nominal,offset]).all():
            raise ValueError('world EE target offset must be finite XYZ metres')
        delta=nominal+offset-self.previous
        event='apply' if not np.any(self.previous) and np.any(offset) else (
            'restore' if np.any(self.previous) and not np.any(offset) else
            'hold' if np.any(offset) else 'inactive')
        self.nominal_integral+=nominal;self.attacked_integral+=delta
        self.previous=offset.copy()
        return delta,{'phase':event,'nominal_integrated_world_delta_m':self.nominal_integral.tolist(),
            'attacked_integrated_world_delta_m':self.attacked_integral.tolist(),
            'commanded_world_target_bias_m':(self.attacked_integral-self.nominal_integral).tolist()}


def prepare(backend,action,request_id,model_phase):
    c=backend.c;r=backend.r
    original=np.asarray(action,dtype=np.float32).copy()
    if original.shape!=(10,) or not np.isfinite(original).all():raise ValueError('invalid nominal EE command')
    enabled=model_phase not in ('LOCAL_ALIGN','SETTLE')
    measured=float(c._actual_gripper_closed_fraction())
    closing=closing_opportunity(nominal_close=float(original[9]),measured_close=measured,
        maximum_finger_opening=float(c.gripper_max_opening),enabled=enabled)
    step=int(c.physics_step);time_s=step*r['PHYSICS_DT']
    if closing and not getattr(backend,'_paper_gripper_close_started',False):
        backend._paper_gripper_close_started=True
        event={'type':'gripper_closing_started','physics_step':step,'sim_time_s':time_s,
               'request_id':request_id,'model_phase':model_phase,'measured_close':measured,
               'nominal_close':float(original[9]),'minimum_commanded_closure_m':.0001,
               'evidence':'nominal gripper command at enabled EE execution boundary'}
        backend._paper_physical_events.append(event)
        append(backend,{'event':'opportunity',**event})
    backend.observe_attacks()
    context={'physics_step':step,'time_s':time_s,'request_id':request_id,
             'mission_phase':backend.phase,'model_phase':model_phase,
             'boundary':'post_stage_adapter_pre_IK'}
    attacked=np.asarray(backend.policy.intercept('control_ee_target',original,context),np.float32)
    _,quat=c.link_transform(c.arm_body)
    rotation=r['quaternion_to_rotation_matrix'](quat)
    nominal_world=rotation@original[3:6];attacked_world=rotation@attacked[3:6]
    tracker=getattr(backend,'_paper_world_target_offset',None)
    if tracker is None:tracker=backend._paper_world_target_offset=WorldTargetOffset()
    world_offset=attacked_world-nominal_world
    adapted_world,hold_record=tracker.adapt(nominal_world,world_offset)
    adapted=attacked.copy();adapted[3:6]=rotation.T@adapted_world
    backend._paper_ee_request_id=request_id
    backend._paper_ee_command_id='ee:%s:%s'%(request_id,step)
    append(backend,{'event':'ee_boundary','request_id':request_id,'physics_step':step,
        'ee_command_id':backend._paper_ee_command_id,
        'sim_time_s':time_s,'nominal_action':original.tolist(),'attacked_action':attacked.tolist(),
        'arm_to_world_rotation':rotation.tolist(),'world_delta_before_m':nominal_world.tolist(),
        'world_delta_after_m':attacked_world.tolist(),
        'world_attack_offset_m':world_offset.tolist(),
        'incremental_IK_action':adapted.tolist(),'incremental_world_delta_m':adapted_world.tolist(),
        'target_bias_lifecycle':hold_record,
        'axis_order':['X','Y','Z'],'position_units':'m','model_phase':model_phase,
        'subsequent_chain':['damped_IK','joint_increment_clamp','Agent_guard','actuator_receipt'],
        'hold_semantics':'absolute world target bias; offset differences enter incremental IK; command target bias is not a measured physical displacement'})
    return adapted


def record_adapter(backend,request_id):
    c=backend.c
    if getattr(backend,'_paper_ee_request_id',None)!=request_id:return
    debug=getattr(c,'last_ik_debug',{})
    append(backend,{'event':'ik_and_ordinary_clamp','request_id':request_id,
        'ee_command_id':backend._paper_ee_command_id,
        'physics_step':int(c.physics_step),'sim_time_s':int(c.physics_step)*backend.r['PHYSICS_DT'],
        'ik':{k:np.asarray(v).tolist() for k,v in debug.items()},
        'joint_targets_after_ordinary_clamp':np.asarray(backend.arm).tolist(),
        'final_actuator_receipts_ref':'policy_execution_events.assurance_receipt_ids',
        'Agent_enabled':backend.cfg['backend']=='groot' and __import__('os').environ.get('SAFE_AGENT_MODE')=='on'})


def apply_ordinary_ik(backend,action):
    """One identical incremental IK/clamp/accepted-target chain for all systems.

    The IK adapter accumulates on piper_command_targets, so the ordinary clamp
    must feed its accepted target back. Otherwise a rejected increment remains
    hidden in the accumulator and can reappear in a later nominal zero action.
    """
    c=backend.c
    # The shared phase contract, not the backend/model identity, owns the one
    # deliberate exception: retracting while BASE_RETREAT is executing.
    state=np.asarray(backend.safe())
    if getattr(backend,'_paper_model_phase',None)!='BASE_RETREAT':
        if np.linalg.norm(state[6:9])>.03 or abs(state[11])>.05:
            from mission import Stop
            raise Stop('BASE_NOT_STABLE_FOR_MANIPULATION')
    raw=np.asarray(c.differential_ik_targets(action[3:9],float(action[9])),np.float32)
    if raw.shape!=(7,) or not np.isfinite(raw).all():raise ValueError('invalid ordinary IK target')
    measured=np.asarray(c.robot.get_joint_positions())[backend.arm_indices]
    accepted=raw.copy();accepted[:6]=measured+np.clip(raw[:6]-measured,-.05,.05)
    c.last_ik_debug.update(targets_before_ordinary_clamp=raw.copy(),
        measured_arm_joints=measured.copy(),ordinary_increment_limit_rad=np.asarray(.05),
        targets_after_ordinary_clamp=accepted.copy())
    backend.arm=accepted;c.piper_command_targets=accepted.copy()
    return accepted


def record_actuator_link(backend,request_id,receipt):
    """Join the EE transaction to the real post-guard signed motor receipt."""
    if receipt is None or getattr(backend,'_paper_ee_request_id',None)!=request_id:return
    append(backend,{'event':'actuator_receipt_link','request_id':request_id,
        'ee_command_id':backend._paper_ee_command_id,
        'physics_step':int(backend.c.physics_step),
        'sim_time_s':int(backend.c.physics_step)*backend.r['PHYSICS_DT'],
        'actuator_command_id':receipt['command_id'],
        'receipt_source':'actual assurance_bridge.latest_receipt after normal c.advance',
        'final_targets_source':'assurance receipt guarded/executed joint target arrays; no inference from requested action'})
