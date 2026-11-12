"""Repair a proven pre-policy initial overlap; never use attack outcomes."""
from pathlib import Path
import math,json
from .common import read,write,now,sha


def repair(root,question,attempt,retreat_m=.3,reference_question=None):
    if (root/'manifests/FROZEN.json').exists():raise ValueError('spawn repairs require a new unfrozen version')
    result=read(attempt/'episode/FULL_CHAIN_RESULT.json');job=read(attempt/'job.json')
    if job['purpose']!='pilot' or job['question']!=question:
        raise ValueError('spawn repair requires the matching development run')
    if job['variant']!='clean':
        from .chain_evidence import inspect
        chain=inspect(attempt/'episode',job)
        if chain['triggered'] or chain['effective']:raise ValueError('attack-contaminated initial condition')
    if result['inference_count'] or result['stopped_stage']!='BOOT' or not result.get('physical_collision'):
        raise ValueError('not a proven pre-policy spawn collision')
    # Use the first terminating collision; the stop command can report a later
    # contact while settling. BOOT + zero inference excludes policy actions.
    collision=result['physical_collision']
    failure=result.get('failure','')
    if failure.startswith('Stop: PHYSICAL_COLLISION:'):
        collision=json.JSONDecoder().raw_decode(failure.split('Stop: PHYSICAL_COLLISION:',1)[1])[0]
    if collision['time_s']>1.:raise ValueError('collision did not occur during initial settling')
    package=root/'benchmark/task'/question;path=package/'data/env/environment.json';env=read(path);before=read(path)
    yaw=env['spawn']['yaw_rad'];method='backwards_along_spawn_heading'
    task=read(package/'data/task/task.json');support=task.get('manipulation',{}).get('support_prim')
    geometry=read(root/'manifests/geometry.json')['questions'][question]
    bounds=geometry['objects'].get(support,{}).get('aabb')
    x,y=env['spawn']['xyz'][:2]
    reference=None
    if reference_question:
        from .queue import connect
        ref_package=root/'benchmark/task'/reference_question
        ref_env=read(ref_package/'data/env/environment.json');ref_task=read(ref_package/'data/task/task.json')
        if not support or ref_task.get('manipulation',{}).get('support_prim')!=support or ref_env['scene_usd']!=env['scene_usd']:
            raise ValueError('reference spawn must use the same scene and support surface')
        with connect(root/'queue.sqlite3') as db:
            source=db.execute("SELECT * FROM runs WHERE purpose='pilot' AND system='groot_agent_off' AND variant='clean' AND question=? AND status='collected'",(reference_question,)).fetchone()
        if source is None:raise ValueError('reference clean pilot not collected')
        ref_out=Path(source['output']);v=read(ref_out/'COLLECTION_VALIDATION.json')
        if v['inference_count']<1 or read(ref_out/'input_identity.json')['environment_sha256']!=sha(ref_package/'data/env/environment.json'):
            raise ValueError('reference pose lacks current physical policy-entry proof')
        env['spawn']['xyz']=list(ref_env['spawn']['xyz']);env['spawn']['yaw_rad']=ref_env['spawn']['yaw_rad']
        method='validated_same_support_initial_pose'
        reference={'question':reference_question,'run_id':source['run_id'],'validation_sha256':sha(ref_out/'COLLECTION_VALIDATION.json')}
    elif '/warehouse/' in geometry['scene'] and bounds and all(bounds[0][i]-.48<=v<=bounds[1][i]+.48 for i,v in enumerate([x,y])):
        # The original factory spawn can be INSIDE the packing-table footprint.
        # A small retreat cannot clear it. Put the base outside the nearest
        # support face with room for the initially extended arm; retest physics.
        lo,hi=bounds;clearance=1.05
        choices=[(lo[0]-clearance,y),(hi[0]+clearance,y),(x,lo[1]-clearance),(x,hi[1]+clearance)]
        px,py=min(choices,key=lambda p:(p[0]-x)**2+(p[1]-y)**2)
        target=task['manipulation']['target_world_aabb_center_xyz']
        env['spawn']['xyz'][:2]=[px,py]
        env['spawn']['yaw_rad']=math.atan2(target[1]-py,target[0]-px)
        method='nearest_support_face_with_1.05m_initial_arm_clearance'
    else:
        env['spawn']['xyz'][0]-=retreat_m*math.cos(yaw);env['spawn']['xyz'][1]-=retreat_m*math.sin(yaw)
    env['spawn']['source']='development_pre_policy_collision_repair'
    env['spawn']['validation']={'status':'REQUIRES_PHYSICAL_RETEST','prior_collision':collision}
    revision=root/'manifests/revisions'/('spawn_'+question+'_'+str(int(__import__('time').time())))
    write(revision/'before_environment.json',before);write(path,env)
    route=package/'data/env/route.json'
    if route.exists():
        old=read(route);write(revision/'before_route.json',old);old['spawn']=env['spawn'];write(route,old)
    record={'time':now(),'question':question,'evidence':str(attempt),'result_sha256':sha(attempt/'episode/FULL_CHAIN_RESULT.json'),
        'method':method,'retreat_m':retreat_m if method=='backwards_along_spawn_heading' else None,'reference':reference,
        'change':'clear recorded pre-policy robot-scene overlap; all resulting positions and headings retained in environment diff',
        'applies_to':'all three systems, clean and every attack','rubric_modified':False,'physical_retest_required':True}
    write(revision/'revision.json',record);return record


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--question',required=True);p.add_argument('--attempt',type=Path,required=True)
    a=p.parse_args();print(json.dumps(repair(a.root.resolve(),a.question,a.attempt.resolve()),ensure_ascii=False))
