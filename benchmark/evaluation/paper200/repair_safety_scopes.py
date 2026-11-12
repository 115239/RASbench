"""Compile reviewed whole-robot/base scopes from immutable v1 source requirements.

Plan every change before writing; reruns reconstruct the complete revision ledger,
including an interrupted earlier migration, without losing its source identity.
"""
from pathlib import Path
from .common import read,write,sha,now


def repair(root):
    source=Path(read(root/'manifests/repair_lineage.json')['source_root'])
    planned=[];changes=[]
    for n in range(1,201):
        q='Q%d'%n;relative=Path('benchmark/task')/q/'data/safety'
        p=root/relative/'safety_profile.json';rp=root/relative/'safety_requirements.json'
        original_p=source/relative/'safety_profile.json'
        original_req=read(source/relative/'safety_requirements.json')
        profile=read(p);requirements=read(rp);local=[]
        for origin in original_req['rules']:
            codes={x['code'] for x in origin.get('blocking_reasons',[])}
            req=next(x for x in requirements['rules'] if x['rule_id']==origin['rule_id'])
            resolved=None
            if 'S01_WHOLE_ROBOT_SCOPE_UNREPRESENTED' in codes:
                params=profile['rules']['S01']['parameters']
                ground=profile['rules']['S07']['parameters']
                params.update(source_scope='whole_robot',ground_contact_body_names=ground['wheel_body_names'],
                              ground_prim_prefixes=ground['ground_prim_prefixes'],
                              authorized_contact_body_names=list(params['source_body_names']))
                resolved='S01_WHOLE_ROBOT_SCOPE_UNREPRESENTED'
            elif 'S03_BASE_ONLY_SCOPE_UNREPRESENTED' in codes:
                params=profile['rules']['S03']['parameters']
                # Body binding exists even when the ground-contact rule is disabled.
                # The runner's M20 articulation root is explicitly base_link.
                base=profile['rules'].get('S07',{}).get('parameters',{}).get('base_body_name','base_link')
                runner=(root/'runtime/runner/m20_runtime/runners/m20_fourview_runner.py').read_text()
                if base!='base_link' or 'articulation_path = f"{robot_path}/M20/base_link"' not in runner:
                    raise ValueError(q+': unexpected robot articulation binding')
                for region in params['regions']:region['subject_body_names']=[base]
                resolved='S03_BASE_ONLY_SCOPE_UNREPRESENTED'
            if resolved:
                if req['source_text']!=origin['source_text']:
                    raise ValueError(q+': safety source text drift')
                req['compiled_parameters']=params
                req['blocking_reasons']=[x for x in req['blocking_reasons'] if x['code']!=resolved]
                req['ready_for_preflight']=not req['blocking_reasons']
                req['compilation_status']='explicit_scope_binding/v2'
                local.append({'rule_id':req['rule_id'],'source_text':req['source_text'],
                              'resolved_gap':resolved,'parameters':params})
        if local:
            profile['rule_set_version']='paper200-safety-repair/v2-candidate'
            planned.append((p,profile,rp,requirements))
            changes.append({'question':q,'source_sha256':sha(original_p),'rules':local,
                            'physical_acceptance':'PENDING','rubric_modified':False})
    for (p,profile,rp,requirements),record in zip(planned,changes):
        if read(p)!=profile:write(p,profile)
        if read(rp)!=requirements:write(rp,requirements)
        record['after_sha256']=sha(p)
    write(root/'manifests/revisions/v2_safety_scopes.json',{'created':now(),
        'source_root':str(source),'changes':changes,
        'history':'Full ledger reconstructed from immutable source after interrupted development migration.'})
    return sum(len(x['rules']) for x in changes)


if __name__=='__main__':
    import sys
    print('Compiled scope repairs:',repair(Path(sys.argv[1]).resolve()))
