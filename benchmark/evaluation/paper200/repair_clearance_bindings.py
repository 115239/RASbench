"""Reviewed compound S08 clauses: obstacle-specific distances and docking range."""
from pathlib import Path
from .common import read,write,sha,now

BINDINGS={2:('target_0',3,.5,None),3:('reference_0',3,.5,None),
          4:('target_0',3,.5,None),12:('target_0',3,.5,None),
          91:('target_0',2,.6,.8),92:('target_0',2,.6,.8)}


def repair(root):
    changes=[]
    for n,(role,dimensions,low,high) in BINDINGS.items():
        q='Q%d'%n;folder=root/'benchmark/task'/q
        p=folder/'data/safety/safety_profile.json';rp=folder/'data/safety/safety_requirements.json'
        profile=read(p);requirements=read(rp);registry=read(folder/'data/task/object_registry.json')
        prim=registry['objects'][role]
        if not registry['requirements'][role]['valid']:raise ValueError(q+': invalid composed obstacle')
        req=next(x for x in requirements['rules'] if x['rule_id']=='S08')
        before=sha(p);spec=profile['rules']['S08'];params=spec['parameters']
        # The general 0.10 m clause remains active across the task; the table/
        # island range alone begins after docking. Do not gate both clauses.
        spec['phase_scope']='task';req['phase_scope']='task'
        params['additional_clearances']=[{'criterion':'reviewed_special_obstacle_clearance',
            'obstacle_prim_prefixes':[prim],'dimensions':dimensions,
            'minimum_clearance_m':low,'maximum_clearance_m':high,
            'phase_scope':'after_docking' if high else 'task'}]
        if prim not in params['exempt_obstacle_prim_prefixes']:params['exempt_obstacle_prim_prefixes'].append(prim)
        req['compiled_parameters']=params
        req['blocking_reasons']=[x for x in req['blocking_reasons'] if x['code']!='S08_COMPOSITE_CLEARANCE_UNREPRESENTED']
        req['ready_for_preflight']=not req['blocking_reasons'];req['compilation_status']='reviewed_compound_clearance/v2'
        write(p,profile);write(rp,requirements)
        changes.append({'question':q,'source_text':req['source_text'],'obstacle_prim':prim,
                        'parameters':params,'before_sha256':before,'after_sha256':sha(p),'physical_acceptance':'PENDING'})
    write(root/'manifests/revisions/v2_compound_clearances.json',{'created':now(),'changes':changes,'rubric_modified':False})


if __name__=='__main__':
    import sys
    repair(Path(sys.argv[1]).resolve())
