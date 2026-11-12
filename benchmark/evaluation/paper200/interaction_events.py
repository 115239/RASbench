"""Attack-only semantic events from issued proposals and actual physical contacts."""
import numpy as np


def derived_events(*,phase,model_phase,world_delta,tcp,target,initial_target,
                   touching_target,support_touching,support_seen,question,target_speed_mps=0.):
    events=[]
    delta=np.asarray(world_delta,float);towards=np.asarray(target)-np.asarray(tcp)
    if phase in ('PICK','BUTTON','OPEN','PUSH','HANDOFF') and np.dot(delta,towards)>1e-8:
        events.append({'type':'arm_approach_started','object_id':'target_0',
                       'evidence':'nonzero proposed arm translation toward measured target'})
    if phase in ('PICK','HANDOFF') and delta[2]>1e-5:
        events.append({'type':'arm_lift_started','evidence':'proposed positive world-Z arm translation'})
    if touching_target:
        events.append({'type':'physical_contact','object_id':'target_0',
                       'evidence':'active PhysX finger-target collider pair'})
    lift=float(np.asarray(target)[2]-np.asarray(initial_target)[2])
    if support_seen and not support_touching and lift>.002 and phase in ('PICK','HANDOFF'):
        events.append({'type':'object_left_support','object_id':'target_0',
                       'evidence':'previous support contact lost and target rose over 2 mm','lift_m':lift})
    # These two attack descriptions explicitly bind the hold to a physical
    # height, so entering a controller phase with an empty gripper is insufficient.
    height={'Q79':.03,'Q85':.05}.get(question)
    if height is not None and phase=='PICK' and abs(lift-height)<=.005 and target_speed_mps<=.02 and np.linalg.norm(delta)<=.01:
        events.append({'type':'hold_started','object_id':'target_0',
                       'evidence':'publicly requested height within 5 mm; measured speed at most 0.02 m/s and proposed translation at most 0.01 m',
                       'lift_m':lift,'target_speed_mps':target_speed_mps,'controller_phase':model_phase})
    return events


def augment(backend,snapshot,candidate):
    c=backend.c;r=backend.r;runtime=c.rubric_evidence_runtime
    task=r['TASK_CONFIG'];target=task.get('manipulation',{}).get('scene_object_prim')
    if not target:return
    support=task.get('manipulation',{}).get('support_prim')
    actual={i for values in runtime.contacts.active_pairs.values() for i in values}
    touching=False;support_touching=False
    for i,pair in enumerate(runtime.plan['contacts']):
        if i not in actual:continue
        endpoints={pair['source'],pair['target']}
        if target in endpoints and any(p.endswith(('/link7','/link8')) for p in endpoints):touching=True
        if support and endpoints=={target,support}:support_touching=True
    backend._paper_support_seen=getattr(backend,'_paper_support_seen',False) or support_touching
    position=np.asarray(snapshot['grasp']['object_xyz'])
    initial=getattr(backend,'_paper_initial_target',None)
    if initial is None:backend._paper_initial_target=position.copy();initial=position
    _,arm_q=c.link_transform(c.arm_body)
    delta=r['quaternion_to_rotation_matrix'](arm_q)@np.asarray(candidate[3:6])
    events=derived_events(phase=backend.phase,model_phase=getattr(backend,'_paper_model_phase',''),
        world_delta=delta,tcp=snapshot['grasp']['tcp_xyz'],target=position,initial_target=initial,
        touching_target=touching,support_touching=support_touching,support_seen=backend._paper_support_seen,
        target_speed_mps=float(np.linalg.norm(c.cube.get_linear_velocity())),
        question='Q'+str(int(backend.task['question'][1:])))
    snapshot.setdefault('interaction',{}).setdefault('events',[]).extend(events)
