"""One attack boundary for the complete M20 and Piper actuator transaction."""
import copy
import numpy as np


def intercept(c,positions,velocities,r):
    if c.attack_boundary is None:return positions,velocities
    q=np.asarray(positions).copy();qd=np.asarray(velocities).copy()
    names=c.names
    before={'positions':{name:float(q[c.index[name]]) for name in names if np.isfinite(q[c.index[name]])},
            'wheel_velocities':{name:float(qd[c.index[name]]) for name in names if np.isfinite(qd[c.index[name]])}}
    after=c.attack_boundary.intercept_joint_control(copy.deepcopy(before),physics_step=c.physics_step,
                                                   command_id=getattr(c,'_paper_actuator_command_id',None),
                                                   time_s=c.physics_step*r['PHYSICS_DT'])
    for key in before:
        if set(after[key])!=set(before[key]):raise ValueError('joint attack changed controlled joint scope')
    for name,value in after['positions'].items():
        if not np.isfinite(value):raise ValueError('nonfinite attacked joint target')
        i=c.index[name];low=float(c.articulation['lower'][i]);high=float(c.articulation['upper'][i])
        margin=r['M20_REVOLUTE_TARGET_LIMIT_MARGIN_RAD']
        if name in r['LEG_JOINTS'] and high-low>2*margin:low+=margin;high-=margin
        q[i]=np.clip(value,low,high)
    for name,value in after['wheel_velocities'].items():
        if not np.isfinite(value):raise ValueError('nonfinite attacked wheel target')
        limit=r['VELOCITY_LIMIT_BY_NAME'][name];qd[c.index[name]]=np.clip(value,-limit,limit)
    c._paper_joint_transaction={'physics_step':int(c.physics_step),'sim_time_s':c.physics_step*r['PHYSICS_DT'],
        'command_id':getattr(c,'_paper_actuator_command_id',None),'physics_dt_s':r['PHYSICS_DT'],
        'original':before,'attacked':after,'ordinary_clamped':target_mapping(c,q,qd)}
    return q,qd


def target_mapping(c,positions,velocities):
    return {'positions':{n:float(positions[c.index[n]]) for n in c.names if np.isfinite(positions[c.index[n]])},
            'wheel_velocities':{n:float(velocities[c.index[n]]) for n in c.names if np.isfinite(velocities[c.index[n]])}}


def receipt(c,positions,velocities,r):
    pending=getattr(c,'_paper_joint_transaction',None)
    if pending is None or c.attack_boundary is None:return
    if pending['physics_step']!=c.physics_step:raise RuntimeError('stale complete joint transaction')
    c.attack_boundary.recorder.capture_consumption(dict(pending,
        consumer='articulation_controller.apply_action',receipt_status='apply_action_returned',
        consumed=target_mapping(c,positions,velocities),
        joint_names=list(c.names),joint_indices=[int(c.index[n]) for n in c.names],
        position_units={n:('m' if n in ('joint7','joint8') else 'rad') for n in pending['original']['positions']},
        velocity_units='rad/s',coordinate_frame='articulation_joint_local',
        Agent_mode=c.assurance_bridge.agent.cfg.mode if c.assurance_bridge else 'absent'))
    c._paper_joint_transaction=None
