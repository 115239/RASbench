"""Baseline continuous control plus a direct-action pi05 backend."""
import os
from pathlib import Path
import numpy as np
from .common import read, write


def arm_jacobian(c,r):
    jac=np.asarray(c.robot._articulation_view.get_jacobians(),np.float64)
    offset=jac.shape[-1]-len(c.names)
    if offset not in (0,6):raise ValueError('unsupported Jacobian columns')
    cols=np.asarray([c.index['joint%d'%i] for i in range(1,7)])+offset
    row=c.gripper_body-(len(c.body_names)-jac.shape[1])
    result=jac[0,row][:,cols].copy()
    _,quat=c.link_transform(c.gripper_body)
    off=r['quaternion_to_rotation_matrix'](quat) @ r['TCP_OFFSET']
    skew=np.array([[0,-off[2],off[1]],[off[2],0,-off[0]],[-off[1],off[0],0]])
    result[:3]-=skew @ result[3:]
    _,quat=c.link_transform(c.arm_body)
    rot=r['quaternion_to_rotation_matrix'](quat).T
    result[:3]=rot @ result[:3];result[3:]=rot @ result[3:]
    return result.astype(np.float32)


def run(c,policy,runtime):
    from physical import Backend
    from mission import Mission, Stop
    cfg=read(os.environ['THREE_SCENES_RUN_CONFIG'])
    from paper200.handler_probe import configure_camera
    configure_camera(c,cfg,runtime)
    from paper200.control_visual import configure_camera as configure_visual_control_camera
    configure_visual_control_camera(c,cfg,runtime)
    from paper200.scene_adapters import bind_navigation_evidence
    bind_navigation_evidence(c,cfg)
    package=Path(os.environ['M20_QUESTION_PACKAGE_ROOT'])
    image_masks=any(a['handler']=='paper200_projected_patch'
        for p in (package/'data/attack/profiles').glob('*.json')
        for a in read(p).get('attack',{}).get('attacks',[]))
    if image_masks:
        c.cameras[0].add_instance_id_segmentation_to_frame({'colorize':False})
        c.paper_visual_evidence_dir=runtime['args'].output/'attack_evidence/visual_masks'
    from return_delivery.physics_clock import install_explicit_physics_clock
    install_explicit_physics_clock(c,runtime,runtime['args'].output)
    # Keep real geometry sampling enabled. Its coverage and precision must pass
    # the independent scoring-readiness gate, not merely collection validation.
    from paper200.physics_evidence import PhysicalInventoryWriter
    c.paper_task_config=runtime['TASK_CONFIG']
    c.paper_inventory_writer=PhysicalInventoryWriter(c,runtime['args'].output)
    from paper200.geometry_evidence import GeometryEvidence
    c.paper_geometry_writer=GeometryEvidence(c.safety_geometry_tracker,runtime['args'].output,runtime['PHYSICS_DT'])
    write(runtime['args'].output/'OFFLINE_SCORING.json',{
        'status':'NOT_SCORED','deferred_geometry_rules':[],
        'evidence':'200 Hz body poses/velocities/joints, collision geometry, and PhysX contact points'})

    class CollectionBackend(Backend):
        def event(self,event):
            if event.get('phase','').startswith('NAV_') and event.get('status')=='COMPLETE':
                self._paper_docking_completed=True
                # Observe the actual stopped state inside the formal task,
                # before the final stopping/parking tail. Same ordinary hold
                # and budget accounting for all navigation tasks and systems.
                previous_dt=self.dt
                try:
                    self.dt=.1
                    self.step(source='formal_docking_hold')
                finally:self.dt=previous_dt
                event=dict(event,end_time_s=self.now(),formal_docking_hold_s=.1)
            elif event.get('phase','').startswith('NAV_') and event.get('status')=='STARTED':
                self._paper_docking_completed=False
            return super().event(event)

        def barrier(self,epoch):
            from paper200.wall_clock import check
            check(self)
            # First real model request must be eligible at formal task time 0.
            # Later phase barriers retain the ordinary controller behavior.
            if self.request_count == 0 and self.c.paper_lifecycle.phase == 'TASK_RUNNING':
                self.epoch=epoch
                self.c.latest_unified_action[:]=0
                return
            return super().barrier(epoch)

        def initialize(self):
            from paper200.task_lifecycle import initialize
            return initialize(self)

        def stop(self,epoch):
            from paper200.task_lifecycle import stopping
            stopping(self)
            return super().stop(epoch)

        def infer(self,phase,epoch,carrying,model_phase='ARM_APPROACH',phase_step=0):
            from paper200.wall_clock import check
            check(self)
            self._paper_model_phase=model_phase
            if cfg['purpose'] in ('probe','control_visual','control_direction'):self._paper_probe_ready=True
            response=super().infer(phase,epoch,carrying,model_phase,phase_step)
            check(self)
            return response

        def plan(self,goal):
            from paper200.wall_clock import check
            check(self)
            entry=self.cfg['goals'].get(goal,{})
            if not entry or entry.get('unavailable'):
                raise Stop('GOAL_ANCHOR_UNAVAILABLE:'+str(goal))
            from all_scences.common import ContractError
            try:return super().plan(goal)
            except ContractError as error:
                # These are measured geometric planning outcomes; malformed
                # grids and other contract exceptions remain infrastructure errors.
                if str(error) in ('START_OCCUPIED','GOAL_OCCUPIED','NO_PATH'):
                    raise Stop('PLANNING_'+str(error)) from error
                raise

        def grasped(self):
            # This drives policy phase state, so use robot tactile feedback only.
            # Object lift/pose remains exclusively in the recorded judge evidence.
            from rubric_evidence import array
            forces=[np.linalg.norm(array(v.get_net_contact_forces(dt=self.r['PHYSICS_DT'])))
                    for v in self.c.scoring_contact_views]
            values={p.rsplit('/',1)[-1]:float(f) for p,f in zip(self.c.scoring_contact_body_paths,forces)}
            return values.get('link7',0.)>.05 and values.get('link8',0.)>.05

        def step(self,action=None,**kwargs):
            from paper200.wall_clock import check
            check(self)
            self._candidate_action=np.zeros(10,np.float32) if action is None else np.asarray(action,np.float32)
            if kwargs.get('request_id',-1)>=0:
                import json
                from paper200.execution_boundary import record_adapter
                record_adapter(self,kwargs['request_id'])
            before=len(self.c.policy_execution_events)
            try:
                result=super().step(action,**kwargs)
                check(self)
                return result
            finally:
                for event in self.c.policy_execution_events[before:]:
                    # Legacy 10-D ABI: this is the controller request, not the
                    # final 28-DOF targets. Exact targets are in signed receipts.
                    event['executed_action']=event['requested_action'].copy()
                    event['action_semantics']='controller_request_not_final_actuator_command'

        def observe_attacks(self):
            if self.c.paper_lifecycle.phase != 'TASK_RUNNING':
                return
            snapshot=self.c.attack_state_snapshot(self.initial_object_z)
            snapshot['simulation'].update(self.c.paper_lifecycle.clock(
                int(self.c.physics_step)*self.r['PHYSICS_DT'],int(self.c.physics_step)))
            snapshot['simulation']['lifecycle_phase']=self.c.paper_lifecycle.phase
            speed=float(np.linalg.norm(np.asarray(self.c.robot.get_linear_velocity())[:2]))
            snapshot['locomotion']['horizontal_speed_mps']=speed
            snapshot['locomotion']['walking']=self.phase.startswith('NAV_') and speed>.01
            from paper200.attack_runtime import augment_snapshot
            snapshot=augment_snapshot(self.c,snapshot)
            if cfg['purpose'] in ('probe','control_visual','control_direction'):snapshot['paper200_probe']={'first_model_call':getattr(self,'_paper_probe_ready',False)}
            snapshot['mission']={'phase':self.phase,'epoch':self.epoch}
            snapshot['mission']['return_destination_xy']=self.initial_pose[:2].tolist()
            snapshot['mission']['spawn_distance_m']=float(np.linalg.norm(self.pose()[:2]-self.initial_pose[:2]))
            candidate=getattr(self,'_candidate_action',np.zeros(10))
            from paper200.interaction_events import augment
            augment(self,snapshot,candidate)
            snapshot['command']={'yaw_abs':abs(float(candidate[2])),'yaw':float(candidate[2])}
            events=snapshot.setdefault('interaction',{}).setdefault('events',[])
            events.extend(self._paper_physical_events)
            if self.phase=='PUSH' and np.linalg.norm(candidate[3:6])>1e-6:events.append({'type':'push_started'})
            if self.phase.startswith('NAV_') and abs(candidate[2])>1e-6 and np.linalg.norm(candidate[:2])>1e-6:
                events.append({'type':'navigation_turn_started'})
            if self.phase=='NAV_RETURN':
                snapshot.setdefault('interaction',{}).setdefault('events',[]).append({'type':'return_started'})
            self.policy.observe_attack_state(snapshot,immediate_target={
                'immediate_robot_control':self.c,'immediate_gripper':self.c,'immediate_lidar':self.c,
                'immediate_imu':self.c,'immediate_visual_material':self.c})

        def navigate(self,phase,epoch,carrying):
            if cfg.get('purpose')=='control_direction':
                from paper200.control_direction import execute
                return execute(self,phase,epoch,carrying)
            if cfg.get('purpose')=='control_visual':
                from paper200.control_visual import execute
                return execute(self,phase,epoch,carrying)
            if cfg.get('purpose')=='control_q1':
                from paper200.control_q1 import navigate
                return navigate(self,phase,epoch,carrying)
            if cfg['backend']=='groot':return super().navigate(phase,epoch,carrying)
            # pi05 produces base actions directly; no hidden route follower.
            points=np.asarray(self.c.navigation_waypoints_xy)
            goal=points[-1] if phase!='NAV_RETURN' else self.initial_pose[:2]
            stable=0
            while True:
                response=self.infer(phase,epoch,carrying)
                action=np.asarray(response['actions'],np.float32)[0]
                self.step(action,request_id=response[runtime['VLA_REQUEST_ID_KEY']],source='pi05')
                state=self.c.state()
                at_goal=np.linalg.norm(state[:2]-goal)<=.25 and np.linalg.norm(state[6:9])<=.1 and abs(state[11])<=.1
                stable=stable+1 if at_goal else 0
                if stable*self.dt>=1.:
                    return {'status':'COMPLETE','carrying':carrying,'phase_transition_only':True,'formal_score':None}

        def manipulate(self,phase,epoch,seconds):
            if cfg.get('purpose')=='control_direction':
                from paper200.control_direction import execute
                return execute(self,phase,epoch,False)
            if cfg.get('purpose')=='control_visual':
                from paper200.control_visual import execute
                return execute(self,phase,epoch,False)
            if cfg.get('purpose')=='control_q95':
                from paper200.control_q95 import manipulate
                return manipulate(self,phase,epoch,seconds)
            if cfg['backend']=='groot' and not cfg.get('assistance'):return super().manipulate(phase,epoch,seconds)
            self.phase=phase
            start=self.now();calls=0
            while self.now()-start+self.dt<=seconds+1e-7:
                response=self.infer(phase,epoch,self.carrying,phase_step=calls)
                action=np.asarray(response['actions'],np.float32)[0]
                action=self.prepare_ee_command(action,response[runtime['VLA_REQUEST_ID_KEY']],phase)
                from paper200.execution_boundary import apply_ordinary_ik
                apply_ordinary_ik(self,action)
                self.step(action,request_id=response[runtime['VLA_REQUEST_ID_KEY']],source='auxiliary' if cfg.get('assistance') else 'pi05')
                calls+=1
            return {'status':'WINDOW_EXHAUSTED','inference_calls':calls,'carrying':self.grasped(),
                    'physical_success':False,'grasp_success':False}

        def prepare_ee_command(self,action,request_id,model_phase):
            from paper200.execution_boundary import prepare
            return prepare(self,action,request_id,model_phase)

    from paper200.evidence_contract import Lifecycle
    c.paper_lifecycle=Lifecycle()
    backend=CollectionBackend(c,policy,runtime,cfg)
    c.paper_backend=backend
    backend._paper_docking_completed=False
    backend._paper_physical_events=[]
    checks=(c.safety_profile or {}).get('rules',{}).get('S05',{}).get('parameters',{}).get('checks',[])
    if any(x['criterion'] in ('omnigraph_boolean_variable','usd_boolean_attribute','relative_pose_closed') for x in checks):
        from paper200.discrete_state import DiscreteObserver
        c.paper_discrete_observer=DiscreteObserver(c,backend.out,runtime['PHYSICS_DT'])
    from paper200.physical_opportunities import observe_backend
    c.paper_physical_observer=lambda:observe_backend(backend)
    from paper200.wall_clock import install as install_wall_clock
    install_wall_clock(backend)
    c.task_start_physics_step=None
    from paper200 import attack_runtime
    attack_runtime.CURRENT_COLLECTOR=c
    helper=None
    if cfg.get('assistance'):
        if cfg.get('purpose') not in ('assisted','assisted_pilot'):raise ValueError('auxiliary proposals forbidden outside assisted experiments')
        from paper200.assistance import ManipulationProposal
        helper=ManipulationProposal(c,runtime,cfg)
    auxiliary_calls=0
    def response_boundary(response,context=None):
        nonlocal auxiliary_calls
        import copy,json
        response=copy.deepcopy(response)
        detail=None
        if helper is not None:
            phase=backend.phase
            if phase.startswith('NAV_'):
                selection=response.get('selection',{})
                response['selection']={**selection,'decision':'ACCEPT',
                    'goal_id':'START_RETURN' if phase=='NAV_RETURN' else 'ASSIST_PUBLIC_DESTINATION' if phase=='NAV_DELIVERY' else 'ASSIST_PUBLIC_TARGET'}
                detail={'source':'auxiliary_public_target_navigation/v1','phase':phase,'original_selection':selection}
            else:
                action,detail=helper.action(phase)
                response['actions']=np.asarray([action],np.float32)
            if detail is not None:
                auxiliary_calls+=1
                detail.update(time_s=backend.now(),model_request=backend.request_count+1,
                    proposal=response['actions'].tolist(),assistance_manifest_sha256=cfg['assistance']['manifest_sha256'])
                with (runtime['args'].output/'AUXILIARY_PROPOSALS.jsonl').open('a') as log:
                    log.write(json.dumps(detail,ensure_ascii=False)+'\n')
        backend._candidate_action=np.asarray(response['actions'],np.float32)[0]
        backend.observe_attacks()
        if context is not None:context['base_yaw_rad']=float(backend.pose()[2])
        return response
    policy.__dict__['nominal_response_transform']=response_boundary
    try:
        result=Mission(backend,cfg['task'],cfg['pick_seconds'],cfg['place_seconds']).run()
        from paper200.task_lifecycle import finalize
        finalize(backend,result)
        failure=result.get('failure') or ''
        # Coding/inference exceptions must not be mislabeled as policy failures.
        infrastructure=bool((failure and not failure.startswith('Stop:')) or 'STOP_FAILED:' in failure or not result['task_started'])
        result.update(accepted=None,inference_count=backend.request_count,executed_actions=backend.budget.actions,
             frames=len(c.states),maximum_object_lift_m=backend.max_lift,
             final_base_xyz=np.asarray(c.robot.get_world_pose()[0]).tolist(),physical_collision=backend.collision,
             checks={'flow_completed':result['flow_completed']},model_bindings=cfg['model_bindings'],
             collection_infrastructure_error=infrastructure,task_score=None,backend=cfg['backend'])
        result['assistance']={'enabled':helper is not None,'proposal_calls':auxiliary_calls,
            'proposal_fraction':auxiliary_calls/max(1,backend.request_count),'formal_score':None}
        result['tactile_grasp_candidate']=result['physical_grasp_success']
        result['physical_grasp_success']=None
        result['physical_place_success']=None
        result['diagnostic_semantics']='tactile signals drive phase transitions; physical task success is not scored during collection'
        write(runtime['args'].output/'FULL_CHAIN_RESULT.json',result)
        write(runtime['args'].output/'vla_trace.json',c.vla_trace)
        return result
    finally:
        if getattr(c,'paper_discrete_observer',None):c.paper_discrete_observer.close()
        c.paper_inventory_writer.close()
        c.paper_geometry_writer.close()
        if getattr(c,'assurance_bridge',None):c.assurance_bridge.close()
        backend.events.close();backend.trace.close();backend.contact_log.close()
        from paper200.wall_clock import persist
        c.paper_wall_clock.mark('rollout_recording_closed',physics_step=int(c.physics_step))
        persist(backend,rollout_recording_closed=True)
