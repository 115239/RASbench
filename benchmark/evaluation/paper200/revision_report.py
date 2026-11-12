"""Rebuild reviewable private-snapshot deltas against preserved local sources."""
import difflib
from .common import SOURCE,BASE,AGENT,read,write,sha,now


def build(root):
    if (root/'manifests/FROZEN.json').exists():raise ValueError('cannot revise a frozen manifest')
    pairs=[(root/'benchmark/task',SOURCE/'task'),
           (root/'runtime/runner',BASE/'competition_runner/runner'),
           (root/'runtime/agent',AGENT),
           (root/'runtime/baseline_code',BASE/'baseline/code'),
           (root/'runtime/baseline_runner',AGENT/'integrations/safety_embodiment/baseline/code/runner')]
    records=[];patch=[];rubrics=[]
    for destination,source in pairs:
        for p in sorted(destination.rglob('*')):
            if not p.is_file() or p.is_symlink() or '__pycache__' in p.parts:continue
            if p.suffix not in ('.py','.json','.toml','.yaml','.yml','.md','.sh','.usd','.usda'):continue
            old=source/p.relative_to(destination);after=sha(p);before=sha(old) if old.is_file() else None
            if before==after:continue
            relative=str(p.relative_to(root))
            records.append({'file':relative,'source':str(old),'before_sha256':before,'after_sha256':after})
            if p.name=='rubric.json':rubrics.append(relative)
            try:
                old_text=old.read_text().splitlines(keepends=True) if old.is_file() else []
                new_text=p.read_text().splitlines(keepends=True)
                patch.extend(difflib.unified_diff(old_text,new_text,fromfile='source/'+relative,tofile='snapshot/'+relative))
            except UnicodeDecodeError:pass
    if rubrics:raise ValueError('unexpected rubric modifications: '+str(rubrics))
    result={'time':now(),'scope':'private text snapshots vs preserved original sources; binary/native dependencies have separate provenance',
            'files':records,'rubrics_modified':rubrics,'rubrics_checked':200}
    directory=root/'manifests/revisions'
    (directory/'development_changes.patch').write_text(''.join(patch))
    write(directory/'development_changes.json',result)
    return {'changed_files':len(records),'rubrics_modified':rubrics}
