"""Fail-closed v2 evidence-collection release gate.

A scoped physical control is not an original-task success. This module does not
score tasks, choose successful retries, remove STOP, or dispatch experiments.
The explicit acceptance manifest points to independently reviewed reports and
raw evidence; a boolean from a legacy/software report cannot satisfy the gate.
"""
from pathlib import Path
import json
import sqlite3
from .common import PROJECT, now, read, sha, write
from .evidence_contract import VERSION, canonical_digest

SCHEMA = 'paper200-release-acceptance/v2'
SYSTEMS = ('groot_agent_off', 'groot_agent_on', 'pi05_agent_off')


def _requirement(scope, assertions, sources, questions=(), systems=(), roles=()):
    return dict(scope=scope, assertions=assertions.split(), sources=sources.split(),
                questions=list(questions), systems=list(systems), roles=list(roles))


# Every assertion is a named, affirmative evidence review result. Its JSON
# pointer in the report is checked; raw evidence and producer sources are hashed
# separately. Software test reports and parent gate output are not admissible.
REQUIREMENTS = {
 'q84_prompt': _requirement('full_runner',
    'two_actual_requests task_times_0_and_1 independent_request_ids actual_model_consumption no_warmup_attack restored_normal_prompt',
    'collection/task_lifecycle.py collection/model_service.py', ['Q84'], SYSTEMS,
    ['hdf5','video','model_wire']),
 'q84_lift': _requirement('physical_control',
    'empty_gripper_no_trigger stationary_object_no_trigger grasp_detach_rise_once actual_contact_coverage',
    'collection/physical_opportunities.py', ['Q84'], roles=['physics_trace']),
 'q1_walking_feedback': _requirement('runner_control',
    'distance_at_most_1m walking_at_trigger named_joint_freeze measured_attacked_consumed actual_low_level_consumer',
    'runner/m20_runtime/runners/m20_fourview_runner.py collection/attack_proof.py', ['Q1'],
    roles=['hdf5','video','physics_trace','consumer_receipts']),
 'q95_offset': _requirement('runner_control',
    'actual_close_edge world_y_plus_002 nonaccumulating_hold command_restore_and_actuator_receipt same_command_chain defense_receipts',
    'collection/execution_boundary.py collection/rollout.py', ['Q95'], SYSTEMS,
    ['hdf5','video','model_wire','consumer_receipts']),
 'q28_state': _requirement('full_runner',
    'authoritative_state_source complete_state_coverage',
    'collection/discrete_state.py runner/m20_runtime/safety_tools/monitor.py', ['Q28'], roles=['hdf5','video']),
 'q28_short_pulse': _requirement('physical_control',
    'positive_5ms_pulse negative_control missing_sensor_unknown',
    'collection/discrete_state.py runner/m20_runtime/safety_tools/monitor.py', ['Q28'], roles=['physics_trace']),
 's04_controls': _requirement('physical_control',
    'warmup_micro_motion_no_drop actual_grasp_detach_loss_drop legal_place_no_drop missing_contact_unknown',
    'collection/physical_opportunities.py runner/m20_runtime/safety_tools/drop_state.py runner/m20_runtime/safety_tools/monitor.py',
    roles=['physics_trace']),
 's12_controls': _requirement('physical_control',
    'actual_bound_object formal_stable_reference translation_threshold_both_sides rotation_threshold_both_sides missing_pose_unknown',
    'runner/m20_runtime/safety_tools/monitor.py', ['Q77','Q84','Q89','Q95'], roles=['physics_trace']),
 'geometry_full_scene': _requirement('full_runner',
    'actual_collider_manifest per_step_geometry_ledger complete_applicable_coverage witness_and_bound_semantics',
    'runner/m20_runtime/safety_tools/physx_geometry.py runner/m20_runtime/safety_tools/geometry_verdict.py', ['Q1','Q14','Q27'], roles=['hdf5','video','geometry_ledger']),
 's03_full_scene': _requirement('full_runner',
    's03_enabled actual_region_and_robot_binding per_step_geometry_ledger complete_applicable_coverage',
    'runner/m20_runtime/safety_tools/physx_geometry.py runner/m20_runtime/safety_tools/geometry_verdict.py', roles=['hdf5','video','geometry_ledger']),
 'geometry_thresholds': _requirement('physical_control',
    's03_overlap_and_nonoverlap s08_threshold_both_sides conservative_bound_not_false_violation missing_geometry_unknown',
    'runner/m20_runtime/safety_tools/physx_geometry.py runner/m20_runtime/safety_tools/geometry_verdict.py', roles=['physics_trace']),
 'discrete_repairs_controls': _requirement('physical_control',
    'actual_revised_asset positive_state_change negative_state missing_state_unknown',
    'collection/discrete_state.py runner/m20_runtime/safety_tools/monitor.py', ['Q92','Q128','Q130','Q136'], roles=['physics_trace','asset_revision']),
 'discrete_repairs_full_runner': _requirement('full_runner',
    'actual_revised_asset authoritative_state_source complete_state_coverage',
    'collection/discrete_state.py runner/m20_runtime/safety_tools/monitor.py', ['Q92','Q128','Q130','Q136'], roles=['hdf5','video','asset_revision']),
 'handler_review': _requirement('modality_review',
    'all_selected_handlers_reviewed frozen_parameters_verified real_consumers_or_explicit_nonconsumption hold_and_restore_semantics scope_not_relabelled',
    'collection/attack_proof.py', roles=['review_matrix','consumer_receipts']),
 'judge_entrypoints': _requirement('judge_integration',
    'single_manifest_directory_archive_same source_unchanged applied_attack_preserved unreached_attack_preserved noop_rejected local_gap_and_scope_preserved',
    'judge/v2/safety/judge_result/episode_delivery.py', systems=SYSTEMS, roles=['hdf5','entrypoint_outputs']),
 'formal_pairing': _requirement('formal_pairing',
    'same_candidate clean_attack_each_system formal_origin_not_warmup actual_seed_match task_contract_match model_identity_match',
    'collection/task_lifecycle.py', ['Q84'], SYSTEMS,
    ['clean_hdf5','attack_hdf5','clean_formal_initial','attack_formal_initial']),
}


def source_path(root, name):
    prefix, relative = name.split('/', 1)
    if prefix == 'collection':
        return PROJECT/'experiment/paper200'/relative
    return root/'runtime'/prefix/relative


def resolve_local(root, name):
    p = Path(name)
    p = p if p.is_absolute() else root/p
    p = p.resolve()
    if root.resolve() not in p.parents:
        raise ValueError('evidence must reside in this v2 collection: '+str(p))
    return p


def pointer(document, path):
    if not isinstance(path,str) or not path.startswith('/'):
        raise ValueError('an explicit JSON pointer is required')
    value=document
    for part in path[1:].split('/'):
        part=part.replace('~1','/').replace('~0','~')
        value=value[int(part)] if isinstance(value,list) else value[part]
    return value


def artifact(root, entry, problems, label):
    try:
        path=resolve_local(root,entry['path'])
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError('missing/empty file')
        if sha(path)!=entry.get('sha256'):
            raise ValueError('content SHA256 differs')
        return path
    except (KeyError,ValueError,OSError) as error:
        problems.append(label+': '+str(error))
        return None


def check_sources(root, case, requirement, problems):
    supplied=case.get('sources',{})
    for name in requirement['sources']:
        entry=supplied.get(name)
        if not isinstance(entry,dict):
            problems.append('missing current/capture source binding: '+name);continue
        try:
            current=source_path(root,name)
            if sha(current)!=entry.get('sha256'):
                raise ValueError('candidate source changed')
            # Require the original run snapshot, or a contemporaneous producer
            # identity embedded in the hashed report. Retrospective current-file
            # hashes by themselves cannot establish the code used during a run.
            if entry.get('snapshot_path'):
                snapshot=resolve_local(root,entry['snapshot_path'])
                if sha(snapshot)!=entry['sha256']:raise ValueError('capture source differs')
            elif entry.get('report_sha256_pointer'):
                report=read(resolve_local(root,case['report']['path']))
                if pointer(report,entry['report_sha256_pointer'])!=entry['sha256']:
                    raise ValueError('report producer identity differs')
            else:raise ValueError('original snapshot or report producer identity missing')
        except (KeyError,ValueError,OSError) as error:
            problems.append(name+': '+str(error))


def compare_formal_states(clean, attack, tolerance=1e-5):
    """Compare measured formal-origin arrays, never warmup initial_state.npz."""
    import numpy as np
    from .paired_initial import compare_formal_arrays
    with np.load(clean,allow_pickle=False) as a, np.load(attack,allow_pickle=False) as b:
        required=('body_paths','joint_names','body_pose','body_velocity','joint_position','physics_step','sim_time_s')
        if any(k not in a or k not in b for k in required):return False
        return compare_formal_arrays(a,b,tolerance)['matched']


def current_attempt(root, attempt, question):
    """Final paired samples must use the complete candidate, not a warmup-era pilot."""
    problems=[]
    try:
        contract=read(attempt/'runner_judge_contract.json')
        expected=read(root/'benchmark/contracts'/(question+'.json'))
        if canonical_digest(contract)!=canonical_digest(expected):
            problems.append('final paired attempt contract differs from current candidate')
        inventory=read(root/'manifests/implementation_inventory.json')
        prelaunch=read(attempt/'PRELAUNCH_IDENTITY.json')
        if (prelaunch.get('status')!='CURRENT' or
            prelaunch.get('implementation_sha256')!=canonical_digest(expected['implementation']) or
            prelaunch.get('contract_sha256')!=canonical_digest(expected)):
            problems.append('final paired prelaunch identity does not bind candidate')
        for name,component in inventory['components'].items():
            if name=='judge/v2':continue  # Judge re-runs on its own captured source.
            if name=='native_entrypoints':continue  # Live content above + captured prelaunch identity.
            if name=='collection_shared':
                for relative,digest in component['files'].items():
                    source=PROJECT/'experiment'/relative
                    text=source.read_text(encoding='utf-8')
                    if relative=='common.py':
                        text=text.replace('ROOT = Path(__file__).resolve().parent','ROOT = Path('+repr(str(PROJECT/'experiment'))+')')
                    if sha(source)!=digest or (attempt/'code'/relative).read_text(encoding='utf-8')!=text:
                        problems.append('final paired shared-code snapshot differs: '+relative)
                continue
            if name=='baseline_configuration':
                base=attempt/'runtime_snapshot/baseline'
                actual={p.name:sha(p) for p in sorted(base.glob('*.json')) if p.is_file()}
            else:
                base=attempt/'code/paper200' if name=='collection' else attempt/'runtime_snapshot'/name
                if not base.is_dir():
                    problems.append('final paired attempt source snapshot missing: '+name);continue
                actual={str(p.relative_to(base)):sha(p) for p in sorted(base.rglob('*'))
                    if p.is_file() and '__pycache__' not in p.parts and '.git' not in p.parts
                    and (p.suffix=='.py' and p.parent==base if name=='collection' else p.suffix in ('.py','.json','.toml','.yaml','.yml','.sh'))}
            if actual!=component['files']:problems.append('final paired attempt candidate differs: '+name)
    except (OSError,ValueError,KeyError,TypeError) as error:
        problems.append('final paired snapshot verification failed: '+str(error))
    return problems


def check_judge_capture(root,case):
    problems=[]
    capture=artifact(root,case.get('judge_snapshot',{}),problems,'Judge source snapshot manifest')
    invocation=artifact(root,case.get('judge_invocations',{}),problems,'Judge CLI invocation record')
    if not capture or not invocation:return problems
    try:
        doc=read(capture);base=resolve_local(root,doc['snapshot'])
        for relative,digest in doc['files'].items():
            if sha(base/relative)!=digest:problems.append('Judge captured file changed: '+relative)
        current=root/'runtime/judge/v2'
        def source_files(folder):
            return {str(p.relative_to(folder)):sha(p) for p in sorted(folder.rglob('*'))
                    if p.is_file() and '__pycache__' not in p.parts and '.git' not in p.parts
                    and p.suffix in ('.py','.json','.toml','.yaml','.yml','.sh')}
        if source_files(base/'v2')!=source_files(current):problems.append('Judge complete candidate differs from executed snapshot')
        invocations=read(invocation)
        if set(invocations)!={'single','manifest','directory','archive'}:problems.append('Judge four actual invocations missing')
        for name,record in invocations.items():
            scripts=[Path(x) for x in record.get('argv',[]) if x.endswith('.py')]
            if not scripts or any(base not in p.resolve().parents for p in scripts):
                problems.append('Judge invocation not bound to captured code: '+name)
    except (OSError,ValueError,KeyError,TypeError) as error:
        problems.append('Judge capture verification failed: '+str(error))
    return problems


def check_case(root, case, spec):
    problems=[]
    if case.get('scope')!=spec['scope']:problems.append('scope mismatch; controls do not replace full Runner cases')
    if case.get('synthetic',False):problems.append('synthetic cases cannot establish physical release acceptance')
    if case.get('question') and case['question'] not in {'Q%d'%n for n in range(1,201)}:
        problems.append('invalid question identity')
    if case.get('system') and case['system'] not in SYSTEMS:problems.append('invalid system identity')
    report=artifact(root,case.get('report',{}),problems,'review report')
    if report:
        value=read(report)
        for assertion in spec['assertions']:
            try:
                if pointer(value,case.get('assertions',{})[assertion]) is not True:
                    raise ValueError('review assertion is not true')
            except (KeyError,ValueError,IndexError,TypeError) as error:
                problems.append(assertion+': '+str(error))
    roles={}
    for entry in case.get('evidence',[]):
        p=artifact(root,entry,problems,'raw evidence '+str(entry.get('role')))
        if p:roles.setdefault(entry.get('role'),[]).append(p)
    for role in spec['roles']:
        if not roles.get(role):problems.append('required raw evidence missing: '+role)
    check_sources(root,case,spec,problems)
    if case.get('scope')=='judge_integration':problems.extend(check_judge_capture(root,case))
    if case.get('scope') in ('full_runner','runner_control','formal_pairing'):
        attempts=case.get('attempts',[])
        if not attempts:problems.append('captured attempt identity missing')
        if case.get('scope')=='formal_pairing' and len(attempts)!=2:
            problems.append('formal pair must bind clean and attack attempt validations')
        pair_conditions=[]
        for entry in attempts:
            path=artifact(root,entry,problems,'attempt validation')
            if path:
                data=read(path)
                if case.get('scope')=='formal_pairing':
                    problems.extend(current_attempt(root,path.parent,case.get('question')))
                    kind='clean' if data.get('variant')=='clean' else 'attack'
                    pair_conditions.append(kind)
                    if not any(sha(p)==data.get('hdf5_sha256') for p in roles.get(kind+'_hdf5',[])):
                        problems.append('paired validation does not bind '+kind+' HDF5')
                    expected_state=(path.parent/'episode/formal_initial_state.npz').resolve()
                    if expected_state not in roles.get(kind+'_formal_initial',[]):
                        problems.append('paired formal state does not belong to '+kind+' attempt')
                if data.get('collection_status')!='VALID':problems.append('attempt is not collection VALID')
                if data.get('question')!=case.get('question') or data.get('system')!=case.get('system'):
                    problems.append('review/attempt question or system differs')
                if data.get('inference_count',0)<=0 or data.get('physical_inventory_samples',0)<=0:
                    problems.append('attempt lacks real model/physical samples')
                for role,key in (('hdf5','hdf5_sha256'),('video','mp4_sha256')):
                    paths=roles.get(role,[])
                    if paths and not any(sha(p)==data.get(key) for p in paths):
                        problems.append('validation does not bind supplied '+role)

    if case.get('scope')=='formal_pairing' and sorted(pair_conditions)!=['attack','clean']:
        problems.append('formal pairing requires one declared clean and one declared attack')
    if case.get('scope')=='formal_pairing' and roles.get('clean_formal_initial') and roles.get('attack_formal_initial'):
        a,b=roles['clean_formal_initial'],roles['attack_formal_initial']
        if len(a)!=1 or len(b)!=1:problems.append('formal pair must reference exactly two state files')
        elif any(p.name!='formal_initial_state.npz' for p in a+b):problems.append('warmup state is not admissible')
        elif not compare_formal_states(a[0],b[0]):problems.append('formal initial states differ')
    return problems


def check_acceptance(root, manifest):
    results=[]
    if manifest.get('schema')!=SCHEMA:
        return [dict(requirement=name,scope=spec['scope'],passed=False,
                     problems=['missing v2 acceptance manifest'],cases=[])
                for name,spec in REQUIREMENTS.items()]
    for name,spec in REQUIREMENTS.items():
        candidates=[c for c in manifest.get('cases',[]) if c.get('requirement')==name]
        checks=[dict(case_id=c.get('id'),question=c.get('question'),system=c.get('system'),
                     problems=check_case(root,c,spec)) for c in candidates]
        valid=[c for c in checks if not c['problems']]
        problems=[]
        if not valid:problems.append('no current qualified evidence case')
        needed_questions=spec['questions'] or [None]
        needed_systems=spec['systems'] or [None]
        for q in needed_questions:
            for s in needed_systems:
                if not any((q is None or c['question']==q) and (s is None or c['system']==s) for c in valid):
                    problems.append('missing required coverage: '+str(q or '*')+'/'+str(s or '*'))
        results.append(dict(requirement=name,scope=spec['scope'],passed=not problems,
                            problems=problems,cases=checks))
    return results


def check_contracts(root):
    problems=[]
    try:
        preflight=read(root/'reports/V2_CONTRACT_PREFLIGHT.json')
        inventory=read(root/'manifests/implementation_inventory.json')
        if preflight.get('schema')!=VERSION or preflight.get('passed') is not True or preflight.get('issues'):
            problems.append('current 200-question static preflight is not passed')
        rows=preflight.get('questions',[])
        if len(rows)!=200 or {x.get('question') for x in rows}!={'Q%d'%n for n in range(1,201)}:
            problems.append('preflight question set is not exactly Q1..Q200')
        if canonical_digest(rows)!=preflight.get('contract_set_sha256'):problems.append('preflight contract-set digest changed')
        from .repair_contracts import implementation_identity
        live_implementation=implementation_identity(root,write_inventory=False)
        for name,component in inventory.get('components',{}).items():
            if (canonical_digest(component.get('files',{}))!=component.get('sha256') or
                live_implementation['runtime_components'].get(name)!=component.get('sha256')):
                problems.append('implementation inventory stale: '+name)
        if set(inventory.get('components',{}))!=set(live_implementation['runtime_components']):
            problems.append('implementation component set incomplete')
        if live_implementation['inventory_sha256']!=canonical_digest(inventory):
            problems.append('implementation full inventory differs from live candidate')
        model_verified=read(root/'manifests/model_files_verified.json')
        from .provenance import file_stat
        if model_verified.get('passed') is not True:problems.append('model identity verification not passed')
        for p,ident in model_verified.get('files',{}).items():
            if file_stat(p)!={k:ident[k] for k in ('size','mtime_ns','device','inode')}:
                problems.append('verified model file changed: '+p)
        if not model_verified.get('files'):problems.append('empty model identity verification')
        geometry=read(root/'manifests/geometry.json').get('questions',{})
        model_inventory={p:{'sha256':v['sha256'],'role':v.get('role')} for p,v in model_verified.get('files',{}).items()}
        if model_inventory!=inventory.get('verified_model_files'):problems.append('model verification inventory differs from candidate')
        for row in rows:
            q=row['question'];contract=read(root/'benchmark/contracts'/(q+'.json'))
            if canonical_digest(contract)!=row.get('sha256') or row.get('static_binding_complete') is not True:
                problems.append(q+': contract digest/static binding mismatch')
            if contract.get('implementation',{}).get('inventory_sha256')!=canonical_digest(inventory):
                problems.append(q+': implementation inventory differs')
            impl=contract.get('implementation',{})
            if impl!=live_implementation:problems.append(q+': complete implementation identity differs')
            if contract.get('public_task_sha256')!=sha(root/'manifests/public'/q/'task.json'):
                problems.append(q+': public task contents changed')
            for key,file in (('model_bindings','models.json'),('pi05_transfer','pi05_transfer.json'),('native_runtime','native_runtime.json')):
                if impl.get(key)!=read(root/'manifests'/file):problems.append(q+': live '+key+' differs')
            envs={p.name:sha(p) for p in sorted((root/'manifests/environments').glob('*')) if p.is_file()}
            if impl.get('environments')!=envs:problems.append(q+': environment binding differs')
            if contract.get('geometry_binding',{}).get('compiled_inventory_sha256')!=canonical_digest(geometry.get(q,{})):
                problems.append(q+': compiled geometry inventory changed')
            package=root/'benchmark/task'/q
            actual={str(p.relative_to(package)):sha(p) for p in sorted(package.rglob('*'))
                    if p.is_file() and '__pycache__' not in p.parts}
            if actual!=contract.get('package_files'):problems.append(q+': package contents changed')
        return problems,preflight.get('contract_set_sha256'),canonical_digest(inventory)
    except (OSError,ValueError,KeyError,TypeError) as error:
        return problems+['contract verification failed: '+str(error)],None,None


def template():
    return {'schema':SCHEMA,'purpose':'evidence collection release; task scores remain deferred',
            'requirements':REQUIREMENTS,'cases':[],
            'case_format':{'id':'unique review case','requirement':'q84_prompt','scope':'full_runner',
                'question':'Q84','system':'groot_agent_off','synthetic':False,
                'report':{'path':'reports/review.json','sha256':'SHA256'},
                'assertions':{'two_actual_requests':'/assertions/two_actual_requests'},
                'evidence':[{'role':'hdf5','path':'pilot/.../episode/submission.hdf5','sha256':'SHA256'}],
                'attempts':[{'path':'pilot/.../COLLECTION_VALIDATION.json','sha256':'SHA256'}],
                'sources':{'collection/task_lifecycle.py':{'sha256':'SHA256','snapshot_path':'pilot/.../code/paper200/task_lifecycle.py'}}}}


EXPEDITED_REQUIRED = frozenset(('q84_lift','q1_walking_feedback','q95_offset',
    'q28_short_pulse','s04_controls','s12_controls','geometry_thresholds',
    'discrete_repairs_controls','handler_review'))
EXPEDITED_CORE = ('actual_neural_models_all_systems','clean_no_attack',
    'two_actual_prompt_requests','formal_initial_states_matched',
    'valid_hdf5_mp4_write','nonpolicy_source_delta_only')


def verify_development_snapshot(root, attempt):
    """Verify immutable development code against its own recorded execution identity."""
    problems=[]
    try:
        attempt=resolve_local(root,str(attempt))
        contract=read(attempt/'runner_judge_contract.json')
        launch=read(attempt/'PRELAUNCH_IDENTITY.json')
        if launch.get('purpose') in ('main','assisted') or launch.get('status')!='CURRENT':
            raise ValueError('not a captured development launch')
        if launch.get('contract_sha256')!=canonical_digest(contract):raise ValueError('captured contract changed')
        for folder,manifest in ((attempt/'code',read(attempt/'code/manifest.json')),
                                (attempt/'runtime_snapshot',read(attempt/'RUNTIME_SNAPSHOT.json')['files'])):
            if not manifest:raise ValueError('empty source snapshot')
            for relative,digest in manifest.items():
                path=(folder/relative).resolve()
                if folder.resolve() not in path.parents or sha(path)!=digest:
                    raise ValueError('development source snapshot changed: '+relative)
    except (OSError,ValueError,KeyError,TypeError) as error:problems.append(str(error))
    return problems


def expedited_policy(root, cases, implementation_digest):
    """Explicit collection-only authorization; incomplete acceptance stays incomplete."""
    path=root/'manifests/expedited_collection_policy.json'
    if not path.exists():return None,[],[]
    problems=[];references=[path];policy=read(path)
    if (policy.get('mode')!='expedited_collection' or
        policy.get('user_authorized') is not True or not policy.get('user_instruction')):
        problems.append('expedited collection requires recorded explicit user instruction')
    rows={x['requirement']:x for x in cases}
    for name in sorted(EXPEDITED_REQUIRED):
        if not rows.get(name,{}).get('passed'):problems.append(name+': mandatory collection evidence missing')
    deferred=policy.get('deferred_requirements',{})
    for name in REQUIREMENTS:
        if name not in EXPEDITED_REQUIRED and not rows.get(name,{}).get('passed') and not deferred.get(name):
            problems.append(name+': missing explicit deferral rationale')
    review=artifact(root,policy.get('core_review',{}),problems,'expedited core review')
    if review:
        references.append(review);doc=read(review)
        if set(doc.get('systems',[]))!=set(SYSTEMS):problems.append('core review must cover all three actual systems')
        if doc.get('current_implementation_inventory_sha256')!=implementation_digest:
            problems.append('core review does not bind current complete implementation')
        for name in EXPEDITED_CORE:
            if doc.get('assertions',{}).get(name) is not True:problems.append('core collection evidence missing: '+name)
        if not doc.get('evidence'):problems.append('core review raw evidence missing')
        for entry in doc.get('evidence',[]):
            item=artifact(root,entry,problems,'core evidence')
            if item:references.append(item)
        if doc.get('statistically_eligible') is not False:problems.append('expedited evidence must remain statistically ineligible')
    return policy,problems,references


def release_check(root, freeze=False):
    root=Path(root).resolve()
    if (root/'manifests/FROZEN.json').exists():
        if freeze:raise ValueError('version already frozen')
        from .provenance import verify_frozen
        verify_frozen(root)
        return read(root/'reports/MAIN_RELEASE_GATE.json')
    if freeze:
        # Separate preparation and final authorization: validate first; the
        # caller explicitly invokes freeze only after the gate is concrete.
        if (root/'manifests/FROZEN.json').exists():raise ValueError('version already frozen')
    issues,contract_digest,implementation_digest=check_contracts(root)
    acceptance=root/'manifests/release_acceptance_v2.json'
    cases=check_acceptance(root,read(acceptance) if acceptance.exists() else {})
    policy,policy_issues,policy_files=expedited_policy(root,cases,implementation_digest)
    expedited=policy is not None
    if expedited:issues.extend(policy_issues)
    else:issues.extend(x['requirement']+': '+ '; '.join(x['problems']) for x in cases if not x['passed'])
    running=[]
    queue=root/'queue.sqlite3'
    if queue.exists():
        with sqlite3.connect('file:'+str(queue)+'?mode=ro',uri=True) as db:
            columns=('run_id','purpose','output') if expedited else ('run_id','purpose')
            running=[dict(zip(columns,r)) for r in db.execute("SELECT "+','.join(columns)+" FROM runs WHERE status='running'")]
        if any(r['purpose'] in ('main','assisted') for r in running):issues.append('formal jobs still running')
        if freeze and running and not expedited:issues.append('drain development workers before freezing')
        if expedited:
            for row in running:
                issues.extend(row['run_id']+': '+error for error in verify_development_snapshot(root,row.get('output') or ''))
    else:issues.append('queue missing')
    source=root/'manifests/source_drain.json'
    if not source.exists():issues.append('source drain/parked-run identity missing')
    else:
        drained=read(source)
        if drained.get('drained') is not True:issues.append('source collection was not drained')
        original=Path(drained.get('source',''))/'queue.sqlite3'
        if not original.is_file():issues.append('source queue missing for read-only drain verification')
        else:
            with sqlite3.connect('file:'+str(original.resolve())+'?mode=ro',uri=True) as db:
                if db.execute("SELECT COUNT(*) FROM runs WHERE status='running'").fetchone()[0]:
                    issues.append('source collection still has running jobs')
    import shutil
    if shutil.disk_usage(root).free<100*1024**3:issues.append('less than 100 GiB free storage')
    result={'schema':'paper200-evidence-collection-release/v2','time':now(),
        'passed':not issues,'evidence_collection_ready':not issues,
        'real_scene_acceptance_passed':not issues and all(x['passed'] for x in cases),'statistically_eligible':False,
        'collection_release_passed':not issues,
        'release_mode':'expedited_collection' if expedited else 'full_acceptance',
        'all_acceptance_requirements_passed':all(x['passed'] for x in cases),
        'deferred_requirements':{x['requirement']:policy.get('deferred_requirements',{}).get(x['requirement'])
            for x in cases if not x['passed']} if expedited else {},
        'collection_complete':False,'score_status':'NOT_EVALUATED_DEFERRED_DEDICATED_SCORER',
        'scope':'certifies collection evidence pipeline; no task-success or final statistical score claim',
        'problems':issues,'requirements':cases,'running':running,
        'contract_set_sha256':contract_digest,'implementation_inventory_sha256':implementation_digest,
        'acceptance_manifest_sha256':sha(acceptance) if acceptance.exists() else None,
        'main_expected':2022,'main_seed':82026001,'dispatch_changed':False}
    write(root/'reports/release_check_v2.json',result)
    write(root/'reports/release_check.json',result)
    if not acceptance.exists():write(root/'reports/RELEASE_ACCEPTANCE_V2_TEMPLATE.json',template())
    if freeze:
        if issues:raise ValueError('v2 release gate failed: '+json.dumps(issues,ensure_ascii=False))
        from .operations import freeze_version
        extras=[acceptance,root/'reports/release_check_v2.json',root/'manifests/implementation_inventory.json',
                root/'reports/V2_CONTRACT_PREFLIGHT.json',root/'reports/RELEASE_CONTRACT_REVIEW.json',source,root/'manifests/repair_lineage.json']
        for case in read(acceptance).get('cases',[]):
            references=[case.get('report',{})]+case.get('evidence',[])+case.get('attempts',[])
            references.extend(case[k] for k in ('judge_snapshot','judge_invocations') if k in case)
            extras.extend(resolve_local(root,x['path']) for x in references if x.get('path'))
            extras.extend(resolve_local(root,x['snapshot_path']) for x in case.get('sources',{}).values() if x.get('snapshot_path'))
        extras=list(dict.fromkeys(extras+policy_files))
        result['freeze']=freeze_version(root,extra_files=extras,
            prefreeze_check=lambda: check_contracts(root)[0],allow_running_development=expedited)
        write(root/'reports/MAIN_RELEASE_GATE.json',result)
        write(root/'reports/SCORING_READINESS.json',result)
    return result


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('root',type=Path)
    p.add_argument('--freeze',action='store_true');a=p.parse_args()
    print(json.dumps(release_check(a.root,a.freeze),ensure_ascii=False,indent=2))
