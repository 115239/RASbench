"""Schema, index, semantic, provenance and recomputation validation."""
from pathlib import Path
import math
import json
from .final_results import read,digest,file_digest,record,SCHEMAS
from .final_analysis import analyze

class ResultInvalid(ValueError):pass

def require(condition,message):
    if not condition:raise ResultInvalid(message)

def within(root,path):
    root=Path(root).resolve();p=(root/path).resolve()
    require(p.is_relative_to(root),'RESULT_PATH_ESCAPE:'+str(path));return p

def validate(root,*,allow_synthetic=False,verify_evidence=False):
    from jsonschema import Draft202012Validator
    root=Path(root).resolve();candidate=Path(__file__).resolve().parents[4];schema_root=candidate/'schemas'
    validators={name:Draft202012Validator(read(schema_root/(name+'.schema.json'))) for name in SCHEMAS}
    indexed=set();episodes={};systems=[];tasks={};sources={};allruns=[]
    def load(p,kind):
        value=read(p)
        errors=list(validators[kind].iter_errors(value))
        require(not errors,'SCHEMA:'+str(p.relative_to(root))+':'+(';'.join('/'.join(map(str,e.path))+':'+e.message for e in errors[:3])))
        require(value.get('data_role')=='formal_test' or allow_synthetic and value.get('data_role')=='synthetic_example','NONFORMAL_RESULT_REJECTED')
        indexed.add(p);return value
    dataset=load(root/'dataset_info.json','dataset')
    for system_index in dataset['systems']:
        sroot=within(root,system_index['directory']);system=load(sroot/'system_info.json','system');sid=system['system_id']
        require(sid==system_index['system_id'] and sid not in tasks,'SYSTEM_INDEX_MISMATCH_OR_DUPLICATE');systems.append(system);tasks[sid]=[]
        for task_index in system['tasks']:
            troot=within(sroot,task_index['directory']);task=load(troot/'task_info.json','task');tasks[sid].append(task)
            require(task['task_id']==task_index['task_id'],'TASK_INDEX_MISMATCH')
            identity={k:task[k] for k in ('task_template_id','template_version','seed','generator_version')}
            require(task['instance_sha256']==digest({'identity':identity,'instance_config':task['instance_config']}),'INSTANCE_DIGEST_MISMATCH')
            require(task['config_sha256']==digest(task['config']),'CONFIG_DIGEST_MISMATCH')
            require(len(task['blocks'])==1 and task['blocks'][0]['repeat_index']==1,'ONE_BLOCK_REQUIRED')
            block=task['blocks'][0];registered=['clean']+[r['attack_id'] for r in task['config']['attack_registry']]
            require(len(set(registered))==len(registered),'DUPLICATE_REGISTERED_CONDITION')
            require(set(registered)=={c['condition_id'] for c in task['conditions']} and len(registered)==len(task['conditions']),'CONDITION_REGISTRY_MISMATCH')
            require(set(block['planned_conditions'])=={c['condition_id'] for c in task['conditions'] if c['planned']},'PLANNED_CONDITION_MISMATCH')
            seen=set()
            for run in task['runs']:
                require(run['condition_id'] in registered and run['condition_id'] not in seen,'DUPLICATE_OR_UNREGISTERED_RUN');seen.add(run['condition_id'])
                require(run['episode_id'] not in {r['episode_id'] for r in allruns},'DUPLICATE_EPISODE');allruns.append(run)
                require(run['block_id']==block['block_id'],'RUN_BLOCK_MISMATCH')
                if run['judge_status']!='complete':
                    require(run['result_file'] is None and not run['selected_for_analysis'],'PENDING_RESULT_SELECTED');continue
                require(run['result_file'] is not None,'COMPLETE_RUN_RESULT_MISSING')
                p=within(troot,run['result_file']);e=load(p,'episode');episodes[e['episode_id']]=e;sources[e['episode_id']]=p
                require(e['episode_id']==run['episode_id'] and e['system_id']==sid and e['system_version']==system['system_version'],'EPISODE_IDENTITY_MISMATCH')
                require(all(e[k]==task[k] for k in ('task_id','task_template_id','template_version','seed')),'EPISODE_TASK_MISMATCH')
                require(e['condition_id']==run['condition_id'] and e['block_id']==block['block_id'] and e['repeat_index']==1 and e['policy_seed']==block['policy_seed'],'EPISODE_BLOCK_MISMATCH')
                require(e['judge']['rules_sha256']==task['config_sha256'],'JUDGEMENT_RULES_MISMATCH')
                for scope_id,scope in e['scopes'].items():
                    require(scope_id in task['config']['scopes'],'UNREGISTERED_SCOPE')
                    if scope['status']!='available':
                        require(scope['task'] is None and scope['safety'] is None,'UNAVAILABLE_SCOPE_HAS_RESULT');continue
                    start,end=scope['start_time_s'],scope['end_time_s'];require(start is not None and end is not None and end>=start,'INVALID_SCOPE_WINDOW')
                    tr=scope['task'];definition=task['config']['scopes'][scope_id]
                    if tr is not None:
                        weights=definition['item_weights'];require(math.isclose(sum(weights.values()),definition['max_units']) and tr['max_units']==definition['max_units'],'FIXED_SCOPE_WEIGHTS_MISMATCH')
                        require(set(weights)=={x['item_id'] for x in tr['items']} and len(weights)==len(tr['items']),'TASK_ITEMS_MISMATCH')
                        awards=[x['awarded_units'] for x in tr['items']]
                        if any(x is None for x in awards):require(tr['awarded_units'] is None and tr['score_100'] is None,'UNKNOWN_SCORE_ZERO_IMPUTATION')
                        else:require(math.isclose(sum(awards),tr['awarded_units']) and math.isclose(100*sum(awards)/tr['max_units'],tr['score_100']),'TASK_SCORE_MISMATCH')
                        for item in tr['items']:
                            require(item['awarded_units'] in (None,0,weights[item['item_id']]),'NONBINARY_RUBRIC_AWARD')
                            if item['effective_verdict'] is None:require(item['awarded_units'] is None,'UNKNOWN_VERDICT_HAS_SCORE')
                    sr=scope['safety']
                    if sr is not None:
                        states=sr['per_type'];unknown=sorted(k for k,v in states.items() if v=='unknown');violated=sorted(k for k,v in states.items() if v=='violated');applicable=[v for v in states.values() if v!='not_applicable']
                        require(unknown==sr['unknown_types'] and violated==sr['violated_types'],'SAFETY_TYPE_SET_MISMATCH')
                        require(sr['any_violation'] is (True if violated else None if unknown or not applicable else False),'SAFETY_ANY_MISMATCH')
                        require(sr['violation_type_count']==(None if unknown or not applicable else len(violated)),'SAFETY_COUNT_MISMATCH')
                        for other in e['scopes'].values():
                            if other['status']=='available' and other.get('safety') and start<=other['start_time_s']<=other['end_time_s']<=end:
                                require(all(states.get(rid)!='clear' for rid in other['safety']['violated_types']),'CONTAINING_WINDOW_CLEARED_VIOLATION')
    for e in episodes.values():
        if e['matched_clean_episode_id'] in episodes:
            c=episodes[e['matched_clean_episode_id']]
            require(c['condition_id']=='clean' and all(c[k]==e[k] for k in ('system_id','system_version','task_id','block_id','policy_seed','repeat_index')),'INVALID_CLEAN_PAIR_REFERENCE')
    evidence_index=read(root/'evidence_index.json');indexed.add(root/'evidence_index.json')
    for e in episodes.values():
        sha=e['source']['evidence_sha256'];require(sha in evidence_index['artifacts'],'EVIDENCE_URI_UNRESOLVED')
        evidence=evidence_index['artifacts'][sha];require(e['source']['evidence_uri']==evidence['uri'],'EVIDENCE_URI_MISMATCH')
        if verify_evidence:
            if isinstance(evidence['source'],dict):
                import zipfile,hashlib,io
                location=evidence['source']
                with zipfile.ZipFile(location['archive_path']) as archive:
                    with archive.open(location['member']) as stream:
                        raw=stream.read()
                require(hashlib.sha256(raw).hexdigest()==sha,'EVIDENCE_BYTES_CHANGED');input_file=io.BytesIO(raw)
            else:
                require(file_digest(evidence['source'])==sha,'EVIDENCE_BYTES_CHANGED');input_file=evidence['source']
            import h5py
            with h5py.File(input_file,'r') as h:
                for scope in e['scopes'].values():
                    reference_groups=list((scope.get('safety') or {}).get('evidence_refs',{}).values())
                    reference_groups.extend(item.get('evidence_refs',[]) for item in (scope.get('task') or {}).get('items',[]))
                    for refs in reference_groups:
                        for ref in refs:
                            require(ref.startswith('sha256:'+sha+'#hdf5:'),'EVIDENCE_REFERENCE_SOURCE_MISMATCH')
                            path=ref.split('#hdf5:',1)[-1].split(':',1)[0];require(path in h,'EVIDENCE_PATH_MISSING:'+path)
    analysis=load(root/'analysis/analysis_results.json','analysis');coverage=load(root/'analysis/coverage.json','coverage');provenance=load(root/'analysis/provenance.json','provenance')
    expected,expected_coverage=analyze(dataset,systems,tasks,episodes)
    require(all(analysis.get(k)==v for k,v in expected.items()),'ANALYSIS_RECOMPUTATION_MISMATCH')
    require(all(coverage.get(k)==v for k,v in expected_coverage.items()),'COVERAGE_RECOMPUTATION_MISMATCH')
    require(all(p['analysis_id']==provenance['analysis_id'] and p['generated_at']==provenance['generated_at'] for p in (analysis,coverage)),'ANALYSIS_REVISION_MISMATCH')
    recorded=set()
    for row in provenance['inputs']+provenance['outputs']:
        p=within(root,row['path']);require(record(p,root)==row,'PROVENANCE_BYTES_MISMATCH:'+row['path']);recorded.add(p)
    require(recorded==indexed-{root/'analysis/provenance.json'},'PROVENANCE_INDEX_MISMATCH')
    require(provenance['analysis_id']=='sha256:'+digest({'inputs':provenance['inputs'],'analyzer_sources':provenance['analyzer_sources']}),'ANALYSIS_ID_MISMATCH')
    return {'passed':True,'systems':len(systems),'tasks':sum(map(len,tasks.values())),'episodes':len(episodes),'schema_checks':len(indexed),'evidence_bytes_verified':verify_evidence}
