"""Auditable recovery and release checks; no task-success scoring."""
from pathlib import Path
import json,os,signal,time,shutil
from .common import read,write,sha,now,PROJECT
from .queue import connect,report


def recover(root):
    recovered=[];active=[]
    with connect(root/'queue.sqlite3') as db:
        jobs=[dict(x) for x in db.execute("SELECT * FROM runs WHERE status='running'")]
    for job in jobs:
        out=Path(job['output']);info=read(out/'processes.json') if (out/'processes.json').exists() else {}
        owner=int(job['owner'].split(':')[0]);cmd=Path('/proc/%d/cmdline'%owner)
        if cmd.exists() and (str(root).encode() in cmd.read_bytes() or b'paper200' in cmd.read_bytes()):
            active.append(job['run_id']);continue
        owned=[]
        for role in ('runner','model'):
            pid=info.get(role);cmd=Path('/proc/%s/cmdline'%pid)
            if pid and cmd.exists() and str(out).encode() in cmd.read_bytes():
                if os.getpgid(pid)!=pid:raise ValueError('unexpected process group; refusing orphan termination')
                os.killpg(pid,signal.SIGTERM);owned.append(pid)
        deadline=time.monotonic()+20
        while any(Path('/proc/%d'%pid).exists() for pid in owned) and time.monotonic()<deadline:time.sleep(.2)
        for pid in owned:
            cmd=Path('/proc/%d/cmdline'%pid)
            if cmd.exists() and str(out).encode() in cmd.read_bytes():os.killpg(pid,signal.SIGKILL)
        status='pending' if job['attempts']<3 else 'infrastructure_failure'
        reason='ORPHAN_OWNER_EXIT; raw attempt preserved'
        with connect(root/'queue.sqlite3') as db:
            changed=db.execute("UPDATE runs SET status=?,reason=? WHERE run_id=? AND owner=? AND status='running'",(status,reason,job['run_id'],job['owner'])).rowcount
            if changed:db.execute("UPDATE attempts SET status='infrastructure_failure',reason=?,finished=? WHERE run_id=? AND attempt=?",(reason,time.time(),job['run_id'],job['attempts']))
        write(out/'RECOVERY.json',{'time':now(),'status':status,'terminated_owned_pids':owned,'reason':reason})
        recovered.append(job['run_id'])
    report(root)
    return {'recovered':recovered,'still_active':active}


def release_check(root,freeze=False):
    root=Path(root)
    if (root/'manifests/repair_lineage.json').exists() or (root/'benchmark/contracts/Q1.json').exists():
        from .release_v2 import release_check as check_v2
        return check_v2(root,freeze)
    from .provenance import package_identity,file_stat,verify_models,scene_revision_identity
    rows=read(root/'manifests/questions.json');byq={r['question']:r for r in rows};problems=[]
    with connect(root/'queue.sqlite3') as db:
        runs=[dict(x) for x in db.execute("SELECT * FROM runs WHERE purpose='pilot' AND status='collected'")]
    valid=[];stale=[]
    for run in runs:
        path=Path(run['output'])/'COLLECTION_VALIDATION.json'
        if not path.exists():problems.append('missing pilot validation '+run['run_id']);continue
        v=read(path)
        identity_path=Path(run['output'])/'input_identity.json'
        if not identity_path.exists():stale.append(run['run_id']);continue
        identity=read(identity_path)
        if int(run['question'][1:])>150 and identity.get('scene_revisions')!=scene_revision_identity(root,run['question']):
            stale.append(run['run_id']);continue
        package=root/'benchmark/task'/run['question']
        if any(identity.get(key)!=sha(package/relative) for key,relative in [
                ('task_sha256','data/task/task.json'),('rubric_sha256','data/task/rubric.json'),
                ('environment_sha256','data/env/environment.json')]):
            stale.append(run['run_id']);continue
        if v.get('physical_inventory_samples',0)>0:valid.append((run,v))
    covered_scenes={byq[r['question']]['scene'] for r,v in valid}
    covered_types={byq[r['question']]['task_type'] for r,v in valid}
    covered_systems={r['system'] for r,v in valid}
    from .paired_initial import check as check_pairs
    paired_system_questions=[]
    for q in sorted({r['question'] for r,v in valid},key=lambda x:int(x[1:])):
        candidates=[r for r,v in valid if r['question']==q]
        clean_systems={r['system'] for r in candidates if r['variant']=='clean'}
        if clean_systems!={'groot_agent_on','groot_agent_off','pi05_agent_off'}:continue
        if not all((Path(r['output'])/'initial_state.npz').exists() for r in candidates):continue
        if check_pairs(root,'pilot',questions={q})['passed']:paired_system_questions.append(q)
    if not paired_system_questions:problems.append('three-system paired initial-state development proof missing')
    for label,needed,observed in [('scenes',{r['scene'] for r in rows},covered_scenes),
                                  ('task_types',{r['task_type'] for r in rows},covered_types),
                                  ('systems',{'groot_agent_on','groot_agent_off','pi05_agent_off'},covered_systems)]:
        if needed-observed:problems.append('pilot coverage missing '+label+': '+str(sorted(needed-observed)))
    preflight=read(root/'reports/preflight.json')
    if preflight.get('passed')!=200:problems.append('full current preflight not passed')
    elif any(x.get('package_sha256')!=package_identity(root,x['question'])['sha256'] for x in preflight['results']):
        problems.append('preflight inputs changed or were not fingerprinted; rerun full preflight')
    if read(root/'manifests/geometry.json')['issues']:problems.append('unresolved scene geometry')
    bindings=root/'manifests/rubric_bindings.json'
    if not bindings.exists() or read(bindings).get('unresolved'):problems.append('rubric parameter bindings unfinished')
    if read(root/'manifests/revisions/attack_bindings.json')['remaining_issues']:problems.append('attack binding gaps')
    attacks=root/'reports/ATTACK_CHAIN_VALIDATION.json'
    if not attacks.exists() or not read(attacks).get('passed'):problems.append('attack-chain validation unfinished')
    auxiliary=root/'manifests/assistance.json'
    if not auxiliary.exists() or not read(auxiliary).get('frozen'):problems.append('assisted capability list not frozen')
    assisted_types=set();assisted_pairs=[]
    if auxiliary.exists() and read(auxiliary).get('frozen'):
        selected={x['question'] for x in read(auxiliary)['questions']}
        needed_types={byq[q]['task_type'] for q in selected}
        with connect(root/'queue.sqlite3') as db:
            aux=[dict(x) for x in db.execute("SELECT * FROM runs WHERE purpose='assisted_pilot' AND status='collected' AND variant='clean'")]
        by_question={}
        for run in aux:
            out=Path(run['output']);validation=out/'COLLECTION_VALIDATION.json'
            if run['question'] not in selected or not validation.exists():continue
            if read(out/'config.json')['task']['stages']!=byq[run['question']]['stages']:continue
            result=read(out/'episode/FULL_CHAIN_RESULT.json')
            # The auxiliary policy must actually have crossed the model and
            # execution boundaries, not merely loaded the helper configuration.
            proposed=out/'episode/AUXILIARY_PROPOSALS.jsonl'
            phases={json.loads(line)['phase'] for line in proposed.read_text().splitlines()} if proposed.exists() else set()
            pure_manipulation=not any(s.startswith('NAV_') for s in byq[run['question']]['stages'])
            if pure_manipulation and not phases.intersection(byq[run['question']]['stages']):continue
            if pure_manipulation and sha(out/'code/paper200/assistance.py')!=sha(PROJECT/'experiment/paper200/assistance.py'):continue
            if result.get('assistance',{}).get('proposal_calls',0)>0 and read(validation).get('model_wire_calls',0)>0:
                by_question.setdefault(run['question'],set()).add(run['system'])
        assisted_pairs=[q for q,systems in by_question.items() if systems=={'groot_agent_on','groot_agent_off'}]
        assisted_types={byq[q]['task_type'] for q in assisted_pairs}
        if needed_types-assisted_types:problems.append('paired assisted development coverage missing: '+str(sorted(needed_types-assisted_types)))
        if assisted_pairs:
            from .paired_initial import check
            if not check(root,'assisted_pilot',questions=set(assisted_pairs))['passed']:
                problems.append('assisted development paired initial states differ')
    native=root/'manifests/native_runtime.json'
    native_scenes=set()
    if native.exists() and read(native).get('enabled'):
        for run,v in valid:
            evidence=Path(run['output'])/'episode/NATIVE_RUNTIME.json'
            if evidence.exists() and read(evidence).get('verified'):
                native_scenes.add(byq[run['question']]['scene'])
        if len(native_scenes)<3:problems.append('isolated native dependency needs closed-loop evidence from at least three scenes')
    untested_revisions=[]
    for revision in (root/'manifests/revisions').glob('spawn_*/revision.json'):
        q=read(revision)['question']
        clean=[(r,v) for r,v in valid if r['question']==q and r['variant']=='clean' and r['system']=='groot_agent_off']
        if not clean or not any(v['inference_count']>0 for r,v in clean):untested_revisions.append(q)
    if untested_revisions:problems.append('spawn repairs await pre-policy-clear physical retest: '+','.join(sorted(set(untested_revisions))))
    initial_collisions=[]
    for run,v in valid:
        if run['system']!='groot_agent_off' or run['variant']!='clean':continue
        outcome=read(Path(run['output'])/'episode/FULL_CHAIN_RESULT.json')
        if outcome.get('inference_count')==0 and outcome.get('stopped_stage')=='BOOT' and outcome.get('physical_collision'):
            initial_collisions.append(run['question'])
    if initial_collisions:problems.append('clean development has unresolved pre-policy scene overlap: '+','.join(sorted(set(initial_collisions))))
    disk=shutil.disk_usage(root)
    if disk.free<100*1024**3:problems.append('less than 100 GiB disk space')
    result={'time':now(),'passed':not problems,'problems':problems,'pilot_full_evidence_count':len(valid),
            'covered_scenes':sorted(covered_scenes),'covered_types':sorted(covered_types),
            'covered_systems':sorted(covered_systems),'main_expected':2022,'score_status':'NOT_SCORED',
            'three_system_paired_development_questions':paired_system_questions,
            'isolated_native_closed_loop_scenes':sorted(native_scenes),
            'paired_assisted_development_questions':sorted(assisted_pairs),
            'paired_assisted_development_types':sorted(assisted_types),
            'stale_pilot_results_excluded':stale}
    if freeze:
        if problems:raise ValueError('release gate failed: '+json.dumps(problems))
        result['freeze']=freeze_version(root)
        write(root/'reports/MAIN_RELEASE_GATE.json',result)
    write(root/'reports/release_check.json',result)
    return result


def freeze_version(root,extra_files=(),prefreeze_check=None,allow_running_development=False):
    """Freeze only after the appropriate release gate has passed; never dispatch."""
    from .provenance import file_stat,verify_models
    if (root/'manifests/FROZEN.json').exists():raise ValueError('version already frozen')
    with connect(root/'queue.sqlite3') as db:
        query="SELECT COUNT(*) FROM runs WHERE status='running'"
        if allow_running_development:query+=" AND purpose IN ('main','assisted')"
        if db.execute(query).fetchone()[0]:
            raise ValueError('drain development workers before freezing')
    verify_models(root)
    if prefreeze_check:
        changed=prefreeze_check()
        if changed:raise ValueError('candidate changed during final model verification: '+json.dumps(changed))
    files={};links={}
    for directory in [root/'benchmark',root/'runtime',root/'manifests/public']:
        for path in sorted(directory.rglob('*')):
            if path.is_symlink():links[str(path.relative_to(root))]={'target':os.readlink(path),'resolved':str(path.resolve())}
            if path.is_file() and not path.is_symlink() and '__pycache__' not in path.parts and path.suffix!='.pyc':files[str(path.relative_to(root))]=sha(path)
    write(root/'manifests/symlinks_final.json',links)
    files['manifests/symlinks_final.json']=sha(root/'manifests/symlinks_final.json')
    code=root/'runtime/collection_code'
    if code.exists() and extra_files:
        archived=root/'runtime/collection_code_before_v2_freeze'
        if archived.exists():raise ValueError('previous collection-code archive already exists')
        code.rename(archived)
        for p in archived.rglob('*'):
            if p.is_file() and not p.is_symlink() and '__pycache__' not in p.parts:
                files[str(p.relative_to(root))]=sha(p)
        for relative in list(files):
            if relative.startswith('runtime/collection_code/'):del files[relative]
        for relative in list(links):
            if relative.startswith('runtime/collection_code/'):
                links[relative.replace('runtime/collection_code/','runtime/collection_code_before_v2_freeze/',1)]=links.pop(relative)
        write(root/'manifests/symlinks_final.json',links)
        files['manifests/symlinks_final.json']=sha(root/'manifests/symlinks_final.json')
    shutil.copytree(PROJECT/'experiment/paper200',code/'paper200',ignore=shutil.ignore_patterns('__pycache__'))
    for name in ['common.py','live_models.py']:shutil.copy2(PROJECT/'experiment'/name,code/name)
    for p in code.rglob('*.py'):files[str(p.relative_to(root))]=sha(p)
    for p in [root/'manifests/models.json',root/'manifests/questions.json',root/'manifests/protocol.json',
              root/'manifests/assistance.json',root/'manifests/geometry.json',root/'manifests/rubric_bindings.json',
              root/'manifests/model_files_verified.json',root/'manifests/pi05_transfer.json']:
        files[str(p.relative_to(root))]=sha(p)
    for directory in [root/'manifests/revisions',root/'manifests/environments']:
        for p in directory.rglob('*'):
            if p.is_file():files[str(p.relative_to(root))]=sha(p)
    for p in [root/'manifests/native_runtime.json',root/'manifests/handler_probes.json',root/'manifests/handler_probe_setups.json',root/'manifests/assisted_pilots.json']:
        if p.exists():files[str(p.relative_to(root))]=sha(p)
    for p in extra_files:
        p=Path(p)
        files[str(p.relative_to(root))]=sha(p)
    write(root/'manifests/files_final.json',files)
    write(root/'manifests/file_stats_final.json',{relative:file_stat(root/relative) for relative in files})
    identity={'time':now(),'files_manifest_sha256':sha(root/'manifests/files_final.json'),
              'stats_manifest_sha256':sha(root/'manifests/file_stats_final.json'),'main_seed':82026001,'main_count':2022}
    write(root/'manifests/FROZEN.json',identity)
    return identity
