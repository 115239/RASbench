"""Canonical final_result/4 export, separate from Q102-style audit archives."""
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
import re
import uuid

from .final_analysis import analyze
from .evidence_io import decode

SCHEMAS={'dataset':'final-dataset/2','system':'final-system/2','task':'final-task/2','episode':'final-episode-result/4',
         'analysis':'final-analysis/1','coverage':'final-coverage/1','provenance':'final-analysis-provenance/1'}
STAGES={'request_image':'observation','request_state':'proprioception','sensor_force_torque':'proprioception','sensor_wheel_encoder':'proprioception','immediate_lidar':'observation','immediate_imu':'proprioception','control_gripper':'action','immediate_robot_control':'action','immediate_gripper':'action','request_prompt':'instruction','request_rgb':'observation','immediate_visual_material':'observation','state_feedback':'proprioception','response_action':'action','control_base':'action','control_joint':'action','control_ee_target':'action','sensor_joint_encoder':'proprioception','sensor_imu':'proprioception','sensor_lidar':'observation','lidar':'observation'}
DEFAULT_SPEC={'score_scale':'0-100','rate_output_unit':'percent','difference_rate_unit':'percentage_points',
 'macro_order':['site_id','atomic_attack_id','task_template_id'],'within_cell_unit':'matched block, with seeds identified; repeat_index fixed to 1',
 'clean_baseline':'selected clean full scopes, deduplicated by episode_id',
 'cross_system_support':'intersect template, version, seed, repeat, condition, scope, rules and instance hashes independently per metric',
 'coverage_subsets':{'narrow':['instruction'],'expanded':['instruction','observation','proprioception']},
 'missing_policy':'metric-specific denominators; no zero imputation',
 'confidence_intervals':'not_estimated; downstream inference must preserve shared clean and seed/block clustering'}

def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)
def digest(value):return hashlib.sha256(canonical(value).encode()).hexdigest()
def file_digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def safe(value):
    value=re.sub(r'[^A-Za-z0-9._-]+','_',str(value))
    if not value or value in ('.','..'):raise ValueError('unsafe result identity')
    return value

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:f.write(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def read(path):return decode(Path(path).read_bytes(),str(path))
def record(path,root):return {'path':str(path.relative_to(root)),'size_bytes':path.stat().st_size,'sha256':file_digest(path)}

def code_sources():
    base=Path(__file__).resolve().parents[1]
    return [{'filename':str(p.relative_to(base)),'sha256':file_digest(p)} for p in sorted(base.rglob('*.py')) if not any(x in p.parts for x in ('vendor','tests','__pycache__'))]

def task_config(full,rubric,package,contract,metadata):
    groups=rubric['task_execution']['score_groups']
    items={g['item_id']:{'description':g['description'],'rule':canonical({'group_id':g['item_id'],'rubric':rubric,'package':package})} for g in groups}
    weights={g['item_id']:g['max_units'] for g in groups}
    expected=full['rules']['safety_contract_validation'].get('expected') or {}
    safety={rid:{'description':rid,'rule':canonical(rule)} for rid,rule in expected.get('rules',{}).items() if rule.get('enabled')}
    registry=full['rules'].get('attack_registry',[])
    if isinstance(registry,dict):registry=list(registry.values())
    components=[];attacks=[]
    for row in registry:
        aid=row['attack_id'];attack=row.get('contract') or {};duration=attack.get('duration_s')
        override=metadata.get('attack_taxonomy',{}).get(aid,{})
        components.append({'binding_id':aid,'atomic_attack_id':override.get('atomic_attack_id',row['handler']),
            'site_id':override.get('site_id',STAGES.get(row.get('stage'),row.get('stage','unclassified'))),
            'operator':canonical({'handler':row['handler'],'stage':row.get('stage'),'parameters':attack.get('parameters',{})}),
            'dose':override.get('dose',{'value':duration,'unit':'s' if duration is not None else 'unrecorded'}),
            'trigger_rule':canonical(attack.get('trigger')),'post_scope_id':'post_'+safe(aid)})
        attacks.append({'attack_id':'attack__'+aid,'component_bindings':[aid],'comparison_scope_id':'post_'+safe(aid),'execution_rule':canonical(attack)})
    scopes={sid:{'scope_rule_id':'runner-v0925-window/1:'+sid,'start_anchor':'capture_start' if sid=='episode' else 'task_start' if sid=='full' else sid.removeprefix('post_'),
         'end_rule':'capture_end' if sid=='episode' else 'task_termination','interval':'[start,end]','max_units':sum(weights.values()),'item_weights':weights,'applicable_safety_types':sorted(safety)} for sid in full['scopes']}
    return {'protocol_id':'paper200-final/4','pairing_method':'verified_formal_initial_state_and_model',
        'rubric_version':full['template_version'],'safety_version':(contract or {}).get('safety_profile',{}).get('rule_set_version',expected.get('profile_id','unrecorded')),
        'rubric_items':items,'safety_types':safety,'attack_components':components,'attack_registry':attacks,'scopes':scopes,'interaction_definitions':[],
        'rule_sources':{k:full['rules'].get(k) for k in ('rubric_sha256','runner_rubric_sha256','rubric_canonical_content_sha256','rule_package_sha256','rule_package_canonical_content_sha256','expected_safety_contract_digest','attack_registry_digest')},
        'taxonomy_version':'stage-and-handler/1; dose describes duration; full operator parameters retained'}

def scope_result(scope,full,scope_id):
    result={k:deepcopy(v) for k,v in scope.items() if k in ('status','start_time_s','end_time_s','start_physics_step','end_physics_step')}
    if scope['status']!='available':
        result.update(task=None,safety=None);return result
    task=deepcopy(scope.get('task'));safety=deepcopy(scope.get('safety'))
    if task is not None:
        facts=full['judge'].get('raw_facts',{});audits=full['judge'].get('item_audit',[])
        task['items']=[]
        for g in task.pop('group_scores',[]):
            refs=sorted({full['source']['hdf5_uri']+'#hdf5:'+r for a in audits for r in a.get('evidence',[])})
            effective=g.get('all_satisfied') if g.get('status') not in ('unknown','incomplete') else None
            task['items'].append({'item_id':g['item_id'],'raw_verdict':effective,'effective_verdict':effective,'awarded_units':g['awarded_units'],'evidence_refs':refs})
        task.setdefault('dependency_context',[])
        if task['status']=='incomplete':task.update(awarded_units=None,score_100=None)
        reasons=full['validation'].get('task_score_reasons',[])
        if not full['validation'].get('task_score_eligible',False) and any(x!='TASK_SCORING_INCOMPLETE' for x in reasons):
            task.update(status='incomplete',awarded_units=None,score_100=None)
            for item in task['items']:item.update(raw_verdict=None,effective_verdict=None,awarded_units=None)
    if safety is not None:
        proofs=safety.get('evidence_contract_by_type',{})
        safety['evidence_refs']={rid:[full['source']['hdf5_uri']+'#'+ref for ref in proof.get('raw_evidence_refs',[])] for rid,proof in proofs.items()}
        # A confirmed violation may coexist with unknown measurements. Any is true;
        # full count remains unavailable if another type is unknown.
        states=safety['per_type']
        eligibility=full['validation'].get('safety_eligible_by_rule_and_scope',{}).get(scope_id,{})
        for rid,state in list(states.items()):
            if state!='not_applicable' and not eligibility.get(rid,{}).get('eligible',False):states[rid]='unknown'
        safety['unknown_types']=sorted(k for k,v in states.items() if v=='unknown')
        safety['violated_types']=sorted(k for k,v in states.items() if v=='violated')
        active=[v for v in states.values() if v!='not_applicable']
        safety['any_violation']=True if safety['violated_types'] else None if safety['unknown_types'] or not active else False
        safety['violation_type_count']=None if safety['unknown_types'] or not active else len(safety['violated_types'])
        safety['status']='incomplete' if safety['unknown_types'] else 'ok' if active else 'not_applicable'
    result.update(task=task,safety=safety)
    return result

def termination(runtime):
    reason=str(runtime.get('termination_reason','')).upper()
    if runtime.get('truncated') or 'TIME_LIMIT' in reason or 'TIMEOUT' in reason:return 'time_limit'
    if 'SAFE_STOP' in reason:return 'safe_stop'
    if 'MANUAL' in reason:return 'manual_cancel'
    if 'INFRA' in reason or 'WALL_CLOCK' in reason:return 'infra_error'
    if 'ERROR' in reason or 'EXCEPTION' in reason:return 'system_error'
    if 'ABORT' in reason:return 'task_abort'
    if reason in ('FLOW_COMPLETED_UNSCORED','FLOW_COMPLETED','NORMAL_END'):return 'normal_end'
    return None  # Version 4 explicitly permits an unrecorded termination category.

def episode_result(full,task,judged_at,judge_version):
    runtime=full['runtime'];issues=list(full['validation']['issues']);opportunities=deepcopy(full['opportunities'])
    for row in opportunities.values():
        row['application_valid']=bool(row.get('effect_applied') and row.get('application_evidence_validation',{}).get('status')=='valid' and row.get('attack_contract_validation',{}).get('status')=='matched' and row.get('temporal_evidence_validation',{}).get('status')=='valid')
        if row.get('opportunity_evidence_validation',{}).get('status')!='valid':
            row['recorded_opportunity_reached']=row.get('opportunity_reached');row['opportunity_reached']=None
    if termination(runtime) is None:issues.append('TERMINATION_CATEGORY_UNRECORDED')
    result={k:deepcopy(full.get(k)) for k in ('episode_id','system_id','system_version','task_template_id','template_version','seed','block_id','policy_seed','condition_id','matched_clean_episode_id','checkpoints')}
    result.update(schema_version=SCHEMAS['episode'],data_role=full['data_role'],task_id=task['task_id'],repeat_index=1,
        judge={'judgement_id':str(uuid.uuid4()),'judge_version':judge_version,'judged_at':judged_at,'rules_sha256':task['config_sha256']},
        source={'kind':'recorded','evidence_uri':full['source']['hdf5_uri'],'evidence_sha256':full['source']['hdf5_sha256'],'note':'Persistent SHA-256 identity resolved by evidence_index.json; source input is read only.'},
        runtime={'start_time_s':runtime.get('start_time_s'),'end_time_s':runtime.get('recording_end_time_s'),'termination_reason':termination(runtime),'recorded':runtime},
        validation={'status':'ok' if not issues else 'incomplete','issues':issues,'original_status':full['validation']['status']},
        opportunities=opportunities,scopes={sid:scope_result(s,full,sid) for sid,s in full['scopes'].items()},
        eligibility={'task':full['validation'].get('task_score_eligible',False),'pairing':full['validation'].get('pairing_eligible',False),'attack_application':full['validation'].get('conditional_attack_effect_eligible',False)},
        legacy_identity={'source_episode_id':full.get('source_episode_id'),'repeat_index':full.get('repeat_index'),'task_id':full['task_id'],'run_id':full.get('run_id'),'attempt_id':full.get('attempt_id')},
        rule_provenance=deepcopy(full['rules']))
    return result

def export(root,manifest,records):
    from v2.tools.generate_episode_delivery import _load_rules,_resolve_input
    root=Path(root);dest=root/'final_result'
    if dest.exists():raise FileExistsError('final_result already exists; rejudgement requires a new output directory')
    dest.mkdir();metadata=manifest.get('result_metadata',{});created=datetime.now(timezone.utc).isoformat();sources=code_sources();version='runner-v0925-judge/1:'+digest(sources)
    entries=manifest['episodes'];systems={};tasks=defaultdict(dict);episodes={};evidence={};errors=[]
    for index,record_ in enumerate(records):
        if record_['status']!='ok':
            errors.append({'input_index':index,**record_});continue
        full=read(Path(record_['output_directory'])/'submission_result.json');entry=entries[index]
        try:
            if any(full.get(k) is None for k in ('seed','system_id','system_version','block_id','task_template_id')):raise ValueError('CANONICAL_RESULT_IDENTITY_INCOMPLETE')
            sid=full['system_id'];sv=full['system_version']
            if sid in systems and systems[sid]['system_version']!=sv:raise ValueError('MIXED_SYSTEM_VERSIONS')
            if sid not in systems:
                systems[sid]={'schema_version':SCHEMAS['system'],'data_role':full['data_role'],'system_id':sid,'display_name':sid,'system_version':sv,'submission':metadata.get('systems',{}).get(sid,{}).get('submission',{'kind':'recorded','reference':sv,'sha256':None}),'tasks':[]}
            base=Path(manifest.get('_manifest_directory','.'))
            rubric,package,info,_=_load_rules(entry,base);contract=None
            if entry.get('runner_judge_contract_path'):
                from .recorded_contract import load_trusted_contract
                contract=load_trusted_contract(_resolve_input(entry['runner_judge_contract_path'],base),entry['runner_judge_contract_sha256'])
            config=task_config(full,rubric,package,contract,metadata)
            params={'task':(contract or {}).get('task',{}),'object_registry':(contract or {}).get('object_registry',{}),'scene_id':full.get('scene_id'),'scene_digest':full['source'].get('scene_digest')}
            instance={'kind':'recorded','parameters':params,'note':'Frozen task and scene configuration; policy seed and attempt excluded.'}
            generator=(contract or {}).get('schema','paper200-recorded-contract/2')
            identity={k:full[k] for k in ('task_template_id','template_version','seed')};identity['generator_version']=generator
            instance_sha=digest({'identity':identity,'instance_config':instance});tid=safe(full['question_id']+'_seed'+str(full['seed'])+'_'+instance_sha[:12]);key=(tid,)
            task=tasks[sid].get(key)
            if task is None:
                cids=['clean']+[r['attack_id'] for r in config['attack_registry']]
                task={'schema_version':SCHEMAS['task'],'data_role':full['data_role'],'task_id':tid,**identity,'instance_config':instance,'instance_sha256':instance_sha,'config':config,'config_sha256':digest(config),
                     'conditions':[{'condition_id':cid,'applicability':'applicable','planned':False,'not_run_reason':'outside_selected_input_batch'} for cid in cids],
                     'blocks':[{'block_id':full['block_id'],'repeat_index':1,'policy_seed':full['policy_seed'],'planned_conditions':[]}],'runs':[]}
                tasks[sid][key]=task;systems[sid]['tasks'].append({'task_id':tid,'directory':tid})
            if task['config_sha256']!=digest(config):raise ValueError('MIXED_TASK_RULES')
            if task['blocks'][0]['block_id']!=full['block_id'] or task['blocks'][0]['policy_seed']!=full['policy_seed']:raise ValueError('MULTIPLE_BLOCKS_REQUIRE_EXPLICIT_SELECTION')
            cid=full['condition_id']
            if any(r['condition_id']==cid for r in task['runs']):raise ValueError('MULTIPLE_ATTEMPTS_REQUIRE_EXPLICIT_SELECTION')
            condition=next((c for c in task['conditions'] if c['condition_id']==cid),None)
            if condition is None:raise ValueError('UNREGISTERED_CONDITION')
            condition.update(planned=True,not_run_reason=None);task['blocks'][0]['planned_conditions'].append(cid)
            episode=episode_result(full,task,created,version);filename=safe(episode['episode_id'])+'__'+safe(cid)+'.json'
            epath=dest/safe(sid)/tid/filename;write(epath,episode);episodes[episode['episode_id']]=episode
            selected=entry.get('selected_for_analysis',True)
            task['runs'].append({'episode_id':episode['episode_id'],'block_id':full['block_id'],'condition_id':cid,'run_status':entry.get('run_status','failed' if episode['runtime']['termination_reason'] in ('system_error','infra_error') else 'finished'),
                'judge_status':'complete','result_file':filename,'selected_for_analysis':selected,'selection_reason':entry.get('selection_reason','fixed_input_selection; metric eligibility evaluated independently')})
            evidence[episode['source']['evidence_sha256']]={'uri':episode['source']['evidence_uri'],'sha256':episode['source']['evidence_sha256'],'source':entry.get('persistent_evidence_source',str(_resolve_input(entry['hdf5_path'],base))),
                    'contract_file':entry.get('runner_judge_contract_path'),'contract_byte_sha256':entry.get('runner_judge_contract_sha256'),
                    'archive_result':str((Path(record_['output_directory'])/'submission_result.json').relative_to(root))}
        except (ValueError,KeyError,TypeError) as exc:errors.append({'input_index':index,'episode_id':full.get('episode_id'),'error':str(exc)})
    # Explicit organizer task metadata can register tasks with no completed episode.
    for declaration in metadata.get('unrun_tasks',[]):
        system=deepcopy(declaration['system']);task=deepcopy(declaration['task']);sid=system['system_id']
        if sid in systems and system['system_version']!=systems[sid]['system_version']:raise ValueError('MIXED_SYSTEM_VERSIONS')
        systems.setdefault(sid,{**system,'tasks':[]})
        if (task['task_id'],) in tasks[sid]:raise ValueError('duplicate explicit unrun task')
        tasks[sid][(task['task_id'],)]=task;systems[sid]['tasks'].append({'task_id':task['task_id'],'directory':task['task_id']})
    for row in metadata.get('run_ledger',[]):
        sid=row['system_id'];matches=[t for t in tasks[sid].values() if t['task_id']==row.get('task_id') or t['task_template_id']==row.get('task_template_id') and t['seed']==row.get('seed')]
        if len(matches)!=1:raise ValueError('RUN_LEDGER_TASK_AMBIGUOUS')
        task=matches[0];condition=next(c for c in task['conditions'] if c['condition_id']==row['condition_id']);condition.update({k:row[k] for k in ('applicability','planned','not_run_reason') if k in row})
        if condition['planned'] and row['condition_id'] not in task['blocks'][0]['planned_conditions']:task['blocks'][0]['planned_conditions'].append(row['condition_id'])
        if row.get('run'):
            run=deepcopy(row['run'])
            if any(r['condition_id']==run['condition_id'] for r in task['runs']):raise ValueError('RUN_LEDGER_DUPLICATE_CONDITION')
            if run['judge_status']!='complete' and (run['result_file'] is not None or run['selected_for_analysis']):raise ValueError('PENDING_JUDGE_CANNOT_BE_SELECTED')
            task['runs'].append(run)
    dataset={'schema_version':SCHEMAS['dataset'],'data_role':metadata.get('data_role','formal_test'),'dataset_id':metadata.get('dataset_id','recorded_'+digest(sorted(evidence))[:16]),
         'description':metadata.get('description','Read-only scoring batch; raw Q102-style deliveries are archived outside final_result.'),'created_at':created,
         'systems':[{'system_id':sid,'directory':safe(sid)} for sid in sorted(systems)],'analysis_spec':deepcopy(DEFAULT_SPEC)}
    supplied=metadata.get('analysis_spec',{})
    if any(k!='coverage_subsets' and v!=DEFAULT_SPEC.get(k) for k,v in supplied.items()):raise ValueError('UNSUPPORTED_ANALYSIS_SPEC_CHANGE')
    dataset['analysis_spec'].update(supplied)
    write(dest/'dataset_info.json',dataset);write(dest/'evidence_index.json',{'schema_version':'final-evidence-index/1','artifacts':evidence})
    final_tasks={sid:list(rows.values()) for sid,rows in tasks.items()}
    for sid,system in systems.items():
        write(dest/safe(sid)/'system_info.json',system)
        for task in final_tasks[sid]:write(dest/safe(sid)/task['task_id']/'task_info.json',task)
    write(root/'canonical_export_report.json',{'exported_episodes':len(episodes),'errors':errors,'schema_version':'runner-v0925-export-report/1'})
    (dest/'README.md').write_text('按 dataset_info → system_info → task_info 的 runs 索引读取。仅 final_result 中所选修订参与统计。\n原始全量与摘要在外部归档；unknown 不补零。episode 身份不随重判变化，judgement_id 每次新建。\n证据 SHA-256 由 evidence_index.json 定位。repeat_index 规范化为 1，原值在 legacy_identity 保留。\n',encoding='utf-8')
    analysis,coverage=analyze(dataset,list(systems.values()),final_tasks,episodes)
    inputs=[record(p,dest) for p in sorted(dest.rglob('*.json'))]
    aid='sha256:'+digest({'inputs':inputs,'analyzer_sources':sources})
    base={'data_role':dataset['data_role'],'dataset_id':dataset['dataset_id'],'analysis_id':aid,'generated_at':created}
    write(dest/'analysis/analysis_results.json',{'schema_version':SCHEMAS['analysis'],**base,'note':'No missing-value imputation; successful export does not imply physical or attack validity.',**analysis})
    write(dest/'analysis/coverage.json',{'schema_version':SCHEMAS['coverage'],**base,**coverage})
    outputs=[record(dest/'analysis'/name,dest) for name in ('analysis_results.json','coverage.json')]
    write(dest/'analysis/provenance.json',{'schema_version':SCHEMAS['provenance'],**base,'analyzer_version':version,'analyzer_sources':sources,'path_base':'final_result/',
        'inputs':inputs,'outputs':outputs,'refresh_rule':'Regenerate all analysis outputs after any indexed input or analyzer change.','authority_rule':'Metadata and episode JSON are authoritative; analysis and archives are never episodes.'})
    (dest/'analysis/README.md').write_text('先查看 coverage，再查看各指标有效分母。跨系统仅比较每类指标共同支持；宏平均顺序为位点、原子攻击、模板。未估计置信区间。\n',encoding='utf-8')
    return {'directory':str(dest),'exported_episodes':len(episodes),'error_count':len(errors),'analysis_id':aid}
