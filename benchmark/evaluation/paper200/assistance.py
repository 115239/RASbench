"""Explicit auxiliary-only scripts; all proposals cross normal attack/execution boundaries."""
import numpy as np
from .common import read,write,sha,now,digest,MAIN_SEED
from .queue import connect


def freeze_list(root):
    rows=read(root/'manifests/questions.json');selected=[];missing=[]
    geometry=read(root/'manifests/geometry.json')['questions']
    with connect(root/'queue.sqlite3') as db:
        pilots={r['question']:dict(r) for r in db.execute("SELECT * FROM runs WHERE purpose='pilot' AND system='groot_agent_off' AND variant='clean' AND status='collected'")}
    from pathlib import Path
    from .provenance import scene_revision_identity
    for row in rows:
        q=row['question']
        if q not in pilots:missing.append(q);continue
        identity_path=Path(pilots[q]['output'])/'input_identity.json'
        if not identity_path.exists():missing.append(q+' (legacy pilot lacks input identity)');continue
        identity=read(identity_path)
        package=root/'benchmark/task'/q
        stale=any(identity.get(key)!=sha(package/relative) for key,relative in [
            ('task_sha256','data/task/task.json'),('environment_sha256','data/env/environment.json'),
            ('rubric_sha256','data/task/rubric.json')])
        if int(q[1:])>150:stale |= identity.get('scene_revisions')!=scene_revision_identity(root,q)
        if stale:missing.append(q+' (revised input requires retest)');continue
        result_path=Path(pilots[q]['output'])/'episode/FULL_CHAIN_RESULT.json';result=read(result_path)
        if result.get('inference_count')==0 and result.get('stopped_stage')=='BOOT' and result.get('physical_collision'):
            missing.append(q+' (initial scene overlap precedes policy; repair and retest)');continue
        weak=not result.get('flow_completed')
        public_target_gap=None
        if row['task_type']=='navigation' and result.get('final_base_xyz'):
            target=read(package/'data/task/task.json')['tracked_objects']['target_0']
            bounds=geometry[q]['objects'][target]['aabb'];xy=np.asarray(result['final_base_xyz'][:2])
            public_target_gap=float(np.linalg.norm(np.maximum(0,np.maximum(np.asarray(bounds[0][:2])-xy,xy-np.asarray(bounds[1][:2])))))
            weak |= public_target_gap>1.
        if 'PICK' in row['stages']:weak |= result.get('maximum_object_lift_m',0.)<.05 or not result.get('tactile_grasp_candidate',result.get('physical_grasp_success'))
        if 'PLACE' in row['stages']:weak |= not result.get('physical_place_success')
        if any(s in ('PUSH','OPEN','BUTTON') for s in row['stages']):
            weak |= any(e.get('status')=='WINDOW_EXHAUSTED' for e in result.get('events',[]))
        if weak:selected.append({'question':q,'pilot_run_id':pilots[q]['run_id'],'pilot_result':str(result_path),
            'public_target_gap_m':public_target_gap,
            'pilot_result_sha256':sha(result_path),'reason':'development flow/lift-contact/place diagnostic incomplete; not a formal rubric score',
            'navigation':'public target navigation proposal after recording every original model response','manipulation':'bounded geometric proposal before attack boundary'})
    if missing:raise ValueError('clean capability pilots incomplete: '+','.join(missing))
    path=root/'manifests/assistance.json'
    if path.exists():raise ValueError('assistance list already exists; create a new version to revise it')
    manifest={'schema':'paper200-assistance/v1','created':now(),'frozen':True,'seed':MAIN_SEED,
        'main_experiments_affected':False,'selection_rule':'all 200 development clean groot_agent_off diagnostics, frozen before main',
        'questions':selected,'budget':'same per-question task budget as main; no teleportation or success-label overrides'}
    write(path,manifest)
    byq={r['question']:r for r in rows}
    with connect(root/'queue.sqlite3') as db:
        for item in selected:
            q=item['question']
            for system in ['groot_agent_on','groot_agent_off']:
                for variant in ['clean']+byq[q]['variants']:
                    rid=digest([root.name,q,system,variant,MAIN_SEED,'assisted'])[:24]
                    db.execute('INSERT OR IGNORE INTO runs(run_id,question,system,variant,seed,purpose) VALUES (?,?,?,?,?,?)',
                        (rid,q,system,variant,MAIN_SEED,'assisted'))
    return manifest


def prepare_pilots(root):
    """Select one clean Agent pair per represented auxiliary task type."""
    from .queue import add_assisted_pilot
    path=root/'manifests/assistance.json'
    manifest=read(path) if path.exists() else freeze_list(root)
    selected={x['question'] for x in manifest['questions']}
    rows=[r for r in read(root/'manifests/questions.json') if r['question'] in selected]
    preferred={'navigation':'Q151','grasp':'Q97','pick_place':'Q77','mobile_manipulation':'Q98','articulated_interaction':'Q29'}
    records=[]
    for kind in sorted({r['task_type'] for r in rows}):
        candidates=[r for r in rows if r['task_type']==kind]
        row=min(candidates,key=lambda r:(r['question']!=preferred.get(kind),r['duration_s'],int(r['question'][1:])))
        for system in ['groot_agent_on','groot_agent_off']:
            records.append({'question':row['question'],'task_type':kind,'system':system,
                            'run_id':add_assisted_pilot(root,row['question'],system)})
    result={'time':now(),'seed':72026001,'scope':'development pairs; excluded from formal auxiliary denominator','rows':records}
    write(root/'manifests/assisted_pilots.json',result)
    return result


class ManipulationProposal:
    def __init__(self,c,runtime,cfg):
        self.c=c;self.r=runtime;self.cfg=cfg;self.phase=None;self.state='approach';self.started=0.;self.origin=None;self.calls=0;self.base_positioned=False
        from pxr import Usd,UsdGeom
        target=c.final_stage.GetPrimAtPath(cfg['goals']['ASSIST_PUBLIC_TARGET']['prim'])
        bound=UsdGeom.BBoxCache(Usd.TimeCode.Default(),['default','render']).ComputeWorldBound(target).ComputeAlignedRange()
        position,quat=c.cube.get_world_pose()
        # Imported rigid-body origins need not lie inside their geometry.
        # Track the initial geometric center in the rigid body's local frame.
        self.center_local=runtime['quaternion_to_rotation_matrix'](quat).T@(np.asarray(bound.GetMidpoint(),float)-position)
    def action(self,phase):
        c=self.c;now=float(c.world.current_time)
        position,orientation=c.cube.get_world_pose()
        target=np.asarray(position,float)+self.r['quaternion_to_rotation_matrix'](orientation)@self.center_local
        if phase!=self.phase:
            self.phase=phase;self.state='approach';self.started=now;self.origin=target.copy();self.base_positioned=False
        position,quat=c.link_transform(c.gripper_body)
        tcp=np.asarray(position)+self.r['quaternion_to_rotation_matrix'](quat)@self.r['TCP_OFFSET']
        arm_position,arm_q=c.link_transform(c.arm_body);rotation=self.r['quaternion_to_rotation_matrix'](arm_q)
        if phase!='PLACE' and not self.base_positioned:
            # A target beyond arm reach requires motor-controlled approach.
            # Use the normal, attack-aware LiDAR controller; do not relocate
            # the robot or relax its collision protection/planning footprint.
            distance=float(np.linalg.norm((target-np.asarray(arm_position))[:2]))
            # Spend at most five seconds of the existing operation window on
            # positioning; a guarded/blocked approach must not consume every
            # grasp proposal or extend the task budget.
            if distance>.75 and now-self.started<5.:
                command,audit=c.lidar_navigation_command(target[:2])
                action=np.zeros(10,np.float32);action[:3]=np.clip(command,[-.08,-.08,-.20],[.08,.08,.20])
                self.calls+=1
                return action,{'source':'bounded_auxiliary_geometry_script/v2','phase':phase,'state':'position_base',
                    'uses_registered_scene_target':True,'proposal_index':self.calls,'arm_target_horizontal_distance_m':distance,
                    'target_world_xyz':target.tolist(),'tcp_world_xyz':tcp.tolist(),'lidar_guard':audit,'formal_score':None}
            self.base_positioned=True;self.started=now
        desired=target.copy();grip=0.
        if phase in ('PUSH','BUTTON','OPEN'):
            base=np.asarray(c.robot.get_world_pose()[0],float)
            direction=self.origin-base;direction[2]=0.
            if phase=='PUSH' and self.cfg.get('auxiliary_push_reference_prim'):
                from pxr import Usd,UsdGeom
                prim=c.final_stage.GetPrimAtPath(self.cfg['auxiliary_push_reference_prim'])
                bound=UsdGeom.BBoxCache(Usd.TimeCode.Default(),['default','render']).ComputeWorldBound(prim).ComputeAlignedRange()
                direction=np.asarray(bound.GetMidpoint(),float)-self.origin;direction[2]=0.
            direction/=max(float(np.linalg.norm(direction)),1e-8)
            if self.state=='approach':
                desired=self.origin-direction*.10
                if np.linalg.norm(tcp-desired)<.03:self.state='contact';self.started=now
            elif self.state=='contact':
                desired=self.origin;grip=float(phase=='OPEN')
                if np.linalg.norm(tcp-desired)<.025 and now-self.started>2.:
                    self.state='operate';self.started=now
            else:
                distance=min(.05 if phase=='PUSH' else .03 if phase=='BUTTON' else .12,(now-self.started)*.01)
                desired=self.origin+direction*distance*(-1. if phase=='OPEN' else 1.)
                grip=float(phase=='OPEN')
        elif phase=='PLACE':
            destination=self.cfg.get('auxiliary_destination_world_xyz')
            if destination is None:destination=self.origin
            desired=np.asarray(destination,float)+[0,0,.04];grip=1. if now-self.started<10. else 0.
        elif self.state=='approach':
            desired+=np.array([0,0,.10])
            if np.linalg.norm(tcp-desired)<.03:self.state='lower';self.started=now
        elif self.state=='lower':
            if np.linalg.norm(tcp-desired)<.025:self.state='close';self.started=now
        elif self.state=='close':
            grip=1.
            if now-self.started>=2.:self.state='lift';self.started=now
        else:
            desired=self.origin+[0,0,.06];grip=1.
        action=np.zeros(10,np.float32);action[3:6]=np.clip(rotation.T@(desired-tcp),-.012,.012);action[9]=grip
        self.calls+=1
        return action,{'source':'bounded_auxiliary_geometry_script/v2','phase':phase,'state':self.state,
                       'uses_registered_scene_target':True,'proposal_index':self.calls,'formal_score':None,
                       'target_world_xyz':target.tolist(),'tcp_world_xyz':tcp.tolist(),'desired_world_xyz':desired.tolist(),
                       'position_error_m':float(np.linalg.norm(desired-tcp))}
