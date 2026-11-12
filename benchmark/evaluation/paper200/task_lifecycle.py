"""Runner-only lifecycle and formal initial-state capture, isolated from policy."""
import math
import json
import numpy as np
from .common import write
from .evidence_contract import Lifecycle


def sim_time(backend):
    return int(backend.c.physics_step) * float(backend.r['PHYSICS_DT'])


def reference_stability(backend):
    """Evaluate the final 1 s of 200 Hz target/protected-body measurements."""
    c=backend.c;inventory=c.paper_inventory_writer;inventory.flush()
    task=backend.r['TASK_CONFIG'];paths=set()
    target=task.get('manipulation',{}).get('scene_object_prim')
    if target:paths.add(target)
    s12=(c.safety_profile or {}).get('rules',{}).get('S12',{})
    if s12.get('enabled'):
        paths.update(x['prim_path'] for x in s12['parameters']['objects'])
    s05=(c.safety_profile or {}).get('rules',{}).get('S05',{})
    closures=[x for x in s05.get('parameters',{}).get('checks',[]) if x.get('criterion')=='relative_pose_closed' and x.get('reference')=='authored_closed']
    paths.update(x['prim_path'] for x in closures)
    paths.update(x['prim_path'] for x in s05.get('parameters',{}).get('checks',[]) if x.get('criterion')=='no_robot_manipulation')
    records=[];valid=True
    for path in sorted(paths):
        registry=inventory.references
        if path not in registry.paths or 'reference_pose' not in inventory.file:
            records.append({'prim':path,'stable':False,'reason':'missing authoritative reference measurement'})
            valid=False;continue
        index=registry.paths.index(path);binding=registry.bindings[index]
        coverage=inventory.file['reference_valid'][-200:,index]
        poses=inventory.file['reference_pose'][-200:,index,:]
        if not coverage.all() or not np.isfinite(poses).all():
            records.append({'prim':path,'stable':False,'reason':'reference source coverage missing',
                'binding':binding,'valid_samples':int(coverage.sum()),'sample_count':len(poses)})
            valid=False;continue
        if len(poses)!=200:
            records.append({'prim':path,'stable':False,'reason':'less than one second of measured reference'})
            valid=False;continue
        xyz=poses[:,:3].astype(float);quat=poses[:,3:].astype(float)
        quat/=np.linalg.norm(quat,axis=1,keepdims=True)
        displacement=float(np.linalg.norm(xyz-xyz[-1],axis=1).max())
        angle=float(np.degrees(2*np.arccos(np.clip(np.abs(quat@quat[-1]),0.,1.))).max())
        stable=bool(displacement<=.0005 and angle<=.5)
        records.append({'prim':path,'stable':stable,'maximum_displacement_m':displacement,
                        'maximum_rotation_deg':angle,'pose_order':'xyz_xyzw','sample_count':len(poses),
                        'binding':binding,'source':binding['source'],'valid_samples':int(coverage.sum())})
        valid &= stable
    closure_records=[]
    if closures:
        from safety_tools.discrete_physics import evaluate
        sample=c.paper_discrete_observer.sample(int(c.physics_step))
        for spec in closures:
            violated,measurement=evaluate(spec,sample,{})
            closure_records.append({'closed':violated is False,**measurement})
            valid &= violated is False
    write(backend.out/'REFERENCE_STABILITY.json',{'passed':valid,'bodies':records,'closures':closure_records,
        'window_s':1.,'maximum_displacement_m':.0005,'maximum_rotation_deg':.5,
        'source':'actual 200 Hz reference registry before any attack: PhysX rigid/PhysX owner with fixed child transform/authoritative static USD; readiness thresholds only'})
    return valid


def initialize(backend):
    from mission import Stop
    c = backend.c
    clock = c.paper_lifecycle
    from .wall_clock import check
    check(backend)
    clock.transition('WARMUP', sim_time(backend), int(c.physics_step))
    started = backend.now()
    samples = []
    # Continue measured settling instead of declaring a fixed five-second hold
    # stable. The time origin remains continuous and attacks remain suppressed.
    readiness_history=[]
    stable=model_ready=references_stable=agent_ready=False
    bridge=getattr(c,'assurance_bridge',None)
    if bridge is not None:
        bridge.agent.recovery.begin_warmup()
    for tick in range(300):
        backend.step(budgeted=False)
        state=backend.safe()
        samples.append({'sim_time_s':sim_time(backend),'physics_step':int(c.physics_step),
            'linear_speed_mps':float(np.linalg.norm(state[6:9])),
            'yaw_rate_rps':abs(float(state[11]))})
        if tick<49 or (tick+1)%10:continue
        stable=all(x['linear_speed_mps']<=.05 and x['yaw_rate_rps']<=.05 for x in samples[-10:])
        model_ready=(backend.out.parent/'model/READY.json').is_file()
        references_stable=reference_stability(backend)
        agent_ready=(bridge is None or bridge.agent.cfg.mode in ('off','shadow')
                     or bridge.agent.recovery.state in ('NORMAL','RESUME'))
        readiness_history.append({'sim_time_s':sim_time(backend),'physics_step':int(c.physics_step),
            'base_stable':stable,'model_ready':model_ready,'references_stable':references_stable,'agent_ready':agent_ready})
        if stable and model_ready and references_stable and agent_ready:break
    write(backend.out/'INITIAL_HOLD.json',{'duration_s':backend.now()-started,'budgeted':False,
        'samples':samples,'readiness_checks':readiness_history,'minimum_duration_s':5.,'maximum_duration_s':30.,
        'lifecycle_contract':'paper200-runner-judge/v2'})
    if not stable or not model_ready or not references_stable or not agent_ready:
        raise Stop('INITIALIZATION_NOT_READY:base=%s,references=%s,model=%s,agent=%s' %
                   (stable,references_stable,model_ready,agent_ready))
    inventory = c.paper_inventory_writer
    from rubric_evidence import array
    pose = array(inventory.view.get_transforms()).astype(np.float32)
    velocity = array(inventory.view.get_velocities()).astype(np.float32)
    joints = array(c.robot.get_joint_positions()).astype(np.float32)
    reference_pose,reference_valid,_=inventory.references.sample(pose,sim_time(backend))
    np.savez_compressed(backend.out / 'formal_initial_state.npz',
        body_paths=np.asarray(inventory.paths), joint_names=np.asarray(c.names),
        body_pose=pose, body_velocity=velocity, joint_position=joints,
        reference_paths=np.asarray(inventory.references.paths,dtype=np.str_),reference_pose=reference_pose,reference_valid=reference_valid,
        reference_bindings_json=np.asarray(json.dumps(inventory.references.bindings,ensure_ascii=False,sort_keys=True)),
        physics_step=np.asarray(c.physics_step), sim_time_s=np.asarray(sim_time(backend)))
    c.rubric_evidence_writer.formal_initial(c.rubric_evidence_runtime._kinematics(),
        physics_step=int(c.physics_step),sim_time_s=sim_time(backend))
    clock.transition('READY', sim_time(backend), int(c.physics_step))
    if getattr(c,'paper_wall_clock',None):c.paper_wall_clock.mark('ready',physics_step=int(c.physics_step),sim_time_s=sim_time(backend))
    check(backend)
    clock.transition('TASK_RUNNING', sim_time(backend), int(c.physics_step), readiness={
        'initialization': True, 'stable_reference': True, 'policy_channel': True,
        'stability_window_s': 1., 'base_linear_speed_max_mps': .05,
        'base_yaw_rate_max_rps': .05,
        'protected_object_reference_validation': 'REFERENCE_STABILITY.json'})
    if c.safety_monitor is not None:
        c.safety_monitor.start_formal_task(physics_step=int(c.physics_step),
            sim_time_s=sim_time(backend),sample=c._safety_sample())
    c.task_start_physics_step = c.physics_step
    if getattr(c,'paper_wall_clock',None):c.paper_wall_clock.mark('task_started',physics_step=int(c.physics_step),sim_time_s=sim_time(backend))
    backend.budget.start = backend.now()
    backend.budget.actions = 0
    backend.initial_pose = backend.pose()
    backend.initial_object_z = float(c.cube.get_world_pose()[0][2])
    c._attack_grasp_initial_height = backend.initial_object_z
    if bridge is not None:
        bridge.start_formal_task(sim_time(backend),int(c.physics_step))
    from .physical_opportunities import LiftOpportunity
    backend._paper_lift_observer=LiftOpportunity()
    backend._paper_lift_observer.start(backend.initial_object_z,int(c.physics_step))
    write(backend.out / 'TASK_LIFECYCLE.json', {'events': clock.events,
        **clock.clock(sim_time(backend), int(c.physics_step))})


def stopping(backend):
    clock = backend.c.paper_lifecycle
    if clock.phase not in ('STOPPING', 'FINALIZED'):
        clock.transition('STOPPING', sim_time(backend), int(backend.c.physics_step))
        if getattr(backend.c,'paper_wall_clock',None):backend.c.paper_wall_clock.mark('stopping',physics_step=int(backend.c.physics_step),sim_time_s=sim_time(backend))


def finalize(backend, result):
    clock = backend.c.paper_lifecycle
    stopping(backend)
    stop = clock.events[-1]
    clock.transition('FINALIZED', sim_time(backend), int(backend.c.physics_step))
    failure = result.get('failure')
    wall_limited=bool(failure and failure.startswith('Stop: WALL_CLOCK_LIMIT'))
    result.update(task_accepted=None, accepted=None, task_success=None, task_failure=None,
        task_evaluation_status='not_evaluated',
        termination_reason='WALL_CLOCK_LIMIT' if wall_limited else failure or 'FLOW_COMPLETED_UNSCORED',
        wall_budget_censored=wall_limited,
        termination_time_s=stop['sim_time_s'], termination_physics_step=stop['physics_step'],
        time_limit_s=backend.task['duration_s'],
        truncated=bool(wall_limited or failure and ('DURATION_EXHAUSTED' in failure or 'ACTION_BUDGET_EXHAUSTED' in failure)),
        record_end_sim_time_s=sim_time(backend), record_end_physics_step=int(backend.c.physics_step),
        **clock.clock(sim_time(backend), int(backend.c.physics_step)))
    if getattr(backend.c,'paper_wall_clock',None):
        backend.c.paper_wall_clock.mark('mission_finalized',physics_step=int(backend.c.physics_step),sim_time_s=sim_time(backend))
        result['wall_clock_timing']=backend.c.paper_wall_clock.snapshot()
    write(backend.out / 'TASK_LIFECYCLE.json', {'events': clock.events,
        **clock.clock(sim_time(backend), int(backend.c.physics_step))})
