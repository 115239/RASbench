"""Bind rubric witnesses to verified composed geometry without editing rubrics."""
from pathlib import Path
from .common import read,write,sha,now

def refine(root):
    geometry=read(root/'manifests/geometry.json')
    if geometry['issues']:raise ValueError('geometry references not resolved')
    revisions=[]
    robot='/M20PiperVLA/M20_Piper'
    for row in read(root/'manifests/questions.json'):
        p=root/'benchmark/task'/row['question']/'data/task/task.json';task=read(p);before=sha(p)
        plan=task['rubric_evidence'];objects=geometry['questions'][row['question']]['objects']
        paths=set(objects);support=task.get('manipulation',{}).get('support_prim')
        if support:paths.add(support)
        paths.update([robot,robot+'/Piper/link7',robot+'/Piper/link8',robot+'/Piper/gripper_base'])
        plan['objects']=[{'prim':x,'runtime_prim':x} for x in sorted(paths)]
        collidable=[p for p,v in objects.items() if v.get('colliders')]
        if support and support not in collidable:collidable.append(support)
        pairs=set()
        for source in [robot,robot+'/Piper/link7',robot+'/Piper/link8']:
            for target in collidable:pairs.add((source,target))
        for target in task['tracked_objects'].values():
            if target not in collidable:continue
            for other in collidable:
                if target!=other and not target.startswith(other+'/') and not other.startswith(target+'/'):pairs.add((target,other))
        plan['contacts']=[{'source':s,'target':t} for s,t in sorted(pairs)]
        write(p,task);revisions.append({'path':str(p.relative_to(root)),'before_sha256':before,'after_sha256':sha(p),'reason':'add 200 Hz robot/finger/object physical witnesses and contact pairs; rubric unchanged'})
    for name,old in [('formal_v072_export.py','expected_steps=len(record.actions)'),('m20_question_runner.py','expected_steps=fixed_validation.control_steps)')]:
        p=root/'runtime/runner/robolab_runner'/name;text=p.read_text(encoding='utf-8');before=sha(p)
        if old not in text:raise ValueError('export interface drift: '+name)
        p.write_text(text.replace(old,'expected_steps=None)'),encoding='utf-8')
        revisions.append({'path':str(p.relative_to(root)),'before_sha256':before,'after_sha256':sha(p),'reason':'rubric extension has independent 200 Hz clock; base protocol retains camera clock'})
    write(root/'manifests/revisions/evidence_binding.json',{'created':now(),'changes':revisions})
    return {'questions':len(revisions)-2}

if __name__=='__main__':
    import sys
    print(refine(Path(sys.argv[1]).resolve()))
