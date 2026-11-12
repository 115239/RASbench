"""Register existing, bounded repair evidence; never manufacture a passing gate.

Derived review assertions preserve producer hashes and raw references. Missing
proof is null, old executable hashes remain old, and calibration/control scopes
are explicit. New physical runs are never launched by this helper.
"""
from pathlib import Path
import json
from .common import PROJECT,read,write,sha,now
from .release_v2 import SCHEMA,REQUIREMENTS,SYSTEMS,source_path


def build(root):
    root=Path(root).resolve();cases=[];cache={}
    reviews=root/'reports/release_evidence_v2';reviews.mkdir(exist_ok=True)
    def ref(path,role=None):
        path=Path(path);path=path if path.is_absolute() else root/path
        if not path.is_file():return None
        key=(str(path),path.stat().st_size,path.stat().st_mtime_ns)
        if key not in cache:cache[key]=sha(path)
        result={'path':str(path.relative_to(root)),'sha256':cache[key]}
        if role:result['role']=role
        return result
    def snapshots(attempt,names):
        result={}
        for name in names:
            group,relative=name.split('/',1)
            file=attempt/'code/paper200'/relative if group=='collection' else attempt/'runtime_snapshot'/group/relative
            manifest=attempt/'code/manifest.json' if group=='collection' else attempt/'RUNTIME_SNAPSHOT.json'
            if not file.exists() or not manifest.exists():continue
            content=read(manifest);content=content if group=='collection' else content['files']
            key='paper200/'+relative if group=='collection' else name
            original=content.get(key)
            if original and original==sha(file):result[name]={'sha256':original,'snapshot_path':str(file.relative_to(root))}
        return result
    def add(name,requirement,assertions,notes,evidence,producer=None,attempt=None,question=None,system=None,sources=None,observations=None):
        spec=REQUIREMENTS[requirement]
        assertions={key:assertions.get(key) for key in spec['assertions']}
        source_records=sources or {}
        if attempt:source_records.update(snapshots(attempt,spec['sources']))
        report={'schema':'paper200-scoped-evidence-review/v2','created':now(),
            'requirement':requirement,'scope':spec['scope'],'question':question,'system':system,
            'assertions':assertions,'limitations':notes,'observations':observations or {},
            'sources':source_records,'producer_reports':[ref(x) for x in (producer or []) if ref(x)],
            'source_hash_policy':'capture hashes copied from original manifests/reports, never replaced by current source hashes',
            'not_task_success':True,'not_formal_release':True}
        report_path=reviews/(name+'.json');write(report_path,report)
        sources_case={}
        for key,val in source_records.items():
            sources_case[key]=dict(val)
            if not val.get('snapshot_path'):
                sources_case[key]['report_sha256_pointer']='/sources/'+key.replace('~','~0').replace('/','~1')+'/sha256'
        evidence=[x for x in evidence if x]
        evidence.extend(dict(ref(x),role='producer_report') for x in (producer or []) if ref(x))
        entry={'id':name,'requirement':requirement,'scope':spec['scope'],'question':question,'system':system,
            'synthetic':False,'report':ref(report_path),
            'assertions':{key:'/assertions/'+key for key in spec['assertions']},
            'evidence':evidence,'sources':sources_case,'limitations':notes}
        if attempt:entry['attempts']=[ref(attempt/'COLLECTION_VALIDATION.json')]
        cases.append(entry)
    def original_sources(doc,names):
        records={}
        for name in names:
            path=str(source_path(root,name));digest=doc.get('source_hashes',{}).get(path)
            if digest:records[name]={'sha256':digest}
        return records
    def attempt_raw(attempt,model=False):
        evidence=[ref(attempt/'episode/submission.hdf5','hdf5'),ref(attempt/'episode/episode.mp4','video')]
        if model:
            evidence.extend(ref(p,'model_wire') for p in sorted((attempt/'model').glob('*.msgpack')))
        return evidence

    # Actual contact/physics controls. Q84 calibration cube is deliberately not
    # labelled as a Q84 apple run; it cannot fill question-specific acceptance.
    p=root/'pilot/controls/physical_controls_004/RESULT.json'
    if p.exists():
        doc=read(p);a=doc['assertions'];trace=Path(doc['trace'])
        add('s04_calibration_004','s04_controls',{
            'warmup_micro_motion_no_drop':a.get('initialization_micro_motion_no_drop') and a.get('production_monitor_warmup_formal_phase'),
            'actual_grasp_detach_loss_drop':a.get('true_loss_drop_detected'),
            'legal_place_no_drop':a.get('legal_place_no_drop'),'missing_contact_unknown':a.get('missing_contact_evidence_unknown')},
            ['Real native PhysX calibration cube/dynamic grippers, not original M20 task success.',
             'Missing-contact control deliberately marks observed coverage invalid; not a hardware sensor failure.'],
            [ref(trace,'physics_trace')],[p],sources=original_sources(doc,REQUIREMENTS['s04_controls']['sources']))
        initial=doc['case_results']['initialization_only'];lift=doc['case_results']['grasp_lift_drop']
        add('lift_calibration_004','q84_lift',{
            'stationary_object_no_trigger':initial['lift_events']==[],
            'grasp_detach_rise_once':len(lift['lift_events'])==1,
            'actual_contact_coverage':a.get('actual_contact_coverage')},
            ['Calibration target only; no Q84 target apple execution.',
             'No separate empty native gripper upward-motion control in this record.'],
            [ref(trace,'physics_trace')],[p],sources=original_sources(doc,REQUIREMENTS['q84_lift']['sources']))
        for q in ('Q84','Q95'):
            rows=[x for x in doc['protected_pose_controls'] if x['question']==q]
            checks={x['case']:x['status']==x['expected'] for x in rows}
            add('s12_'+q+'_004','s12_controls',{
                'actual_bound_object':len(rows)==4 and all(x.get('source_prim') for x in rows),
                'translation_threshold_both_sides':checks.get('below_translation') and checks.get('above_translation'),
                'rotation_threshold_both_sides':checks.get('below_rotation') and checks.get('above_rotation')},
                ['Actual referenced mesh with prescribed kinematic stable pose.',
                 'Formal lifecycle reference protocol and missing-pose control not exercised.',
                 'Does not stand in for Q77 or Q89 independent acceptance.'],
                [ref(trace,'physics_trace')],[p],question=q,
                sources=original_sources(doc,REQUIREMENTS['s12_controls']['sources']))

    # Q1 original trigger + actual named low-level network/actuator receipts.
    attempt=root/'pilot/controls/Q1/groot_agent_off/walking_encoder_001';p=attempt/'Q1_CONTROL_ACCEPTANCE.json'
    if p.exists():
        doc=read(p);state=doc['original_trigger']['evidence']['observed_snapshot'];receipts=doc['hold_consumers']
        confirmed=doc['passed'] is True and not doc['issues'] and len(receipts)==len(doc['hold_window'])
        add('q1_walking_encoder_001','q1_walking_feedback',{
            'distance_at_most_1m':doc['actual_trigger_distance_m']<=1,
            'walking_at_trigger':state['locomotion']['walking'] is True and state['locomotion']['horizontal_speed_mps']>.01,
            'named_joint_freeze':confirmed and len(doc['changed_encoder_steps'])>0,
            'measured_attacked_consumed':confirmed,
            'actual_low_level_consumer':confirmed and all(x.get('actual_output') and x.get('sdk_apply_returned') is True for x in receipts)},
            ['Changed initial pose and explicit movement proposal; original trigger/profile retained.',
             'S08 still has Cars/c_01 missing geometry. This case certifies only encoder execution evidence.'],
            attempt_raw(attempt)+[ref(attempt/'episode/PHYSICAL_TRACE.jsonl','physics_trace'),
                ref(attempt/'episode/attack_evidence/raw_payloads/sensor_joint_encoder.jsonl','consumer_receipts'),
                ref(attempt/'episode/attack_evidence/raw_payloads/consumer_receipts.jsonl','consumer_receipts')],
            [p],attempt,'Q1','groot_agent_off',observations={'review_source_evidence_sha256':doc['source_evidence_sha256']})

    for system in SYSTEMS:
        attempt=root/'pilot/controls/Q95'/system/'control_002';p=attempt/'Q95_CONTROL_ACCEPTANCE.json'
        if not p.exists():continue
        doc=read(p)
        chain=[json.loads(x) for x in (attempt/'episode/EE_COMMAND_CHAIN.jsonl').read_text().splitlines()]
        boundaries=[x for x in chain if x.get('event')=='ee_boundary']
        holds=[x for x in boundaries if x.get('target_bias_lifecycle',{}).get('phase')=='hold']
        restores=[x for x in boundaries if x.get('target_bias_lifecycle',{}).get('phase')=='restore']
        hold_ok=bool(holds) and all(max(abs(float(v)) for v in x['incremental_world_delta_m'])<1e-6 for x in holds)
        restore_ok=bool(restores) and all(abs(x['incremental_world_delta_m'][1]+.02)<1e-6 for x in restores)
        actual=doc['actual_actuator_receipts']
        restore_ids={x['request_id'] for x in restores}
        restore_ok=restore_ok and all(any(y['request_id']==rid for y in actual) for rid in restore_ids)
        add('q95_'+system+'_002','q95_offset',{
            'actual_close_edge':bool(doc['actual_closing_opportunities']) and doc['maximum_actual_closure_fraction_increment']>0,
            'world_y_plus_002':doc['maximum_world_offset_error_m']<1e-6,
            'nonaccumulating_hold':hold_ok and abs(doc['observed_hold_seconds']-.5)<1e-6,
            'command_restore_and_actuator_receipt':restore_ok,
            'same_command_chain':doc['explicit_ee_to_actuator_links']>0 and not doc['issues'],
            'defense_receipts':bool(actual) and all('guard_applied_target_changed' in x for x in actual)},
            ['Separate bounded closure proposal control, not original-condition task success.',
             'Restore means commanded target bias removed and actuator receipt retained; exact physical EE return is not asserted.',
             'This capture predates current neural-receipt/base-stability changes; current-source gate must decide stale status.'],
            attempt_raw(attempt,True)+[ref(attempt/'episode/EE_COMMAND_CHAIN.jsonl','consumer_receipts'),ref(attempt/'episode/assurance_v1/events.jsonl','consumer_receipts')],
            [p],attempt,'Q95',system,observations={'guard_modified_receipts':doc['post_guard_modified_attack_receipts'],'review_source_evidence_sha256':doc['source_sha256']})

    # Buttons: full-run coverage separate from isolated single-step pulse.
    attempt=root/'pilot/groot_agent_off/Q028/clean/seed_72026001/attempt_003'
    p=attempt/'episode/DISCRETE_STATE_BINDINGS.json'
    if p.exists():
        doc=read(p);summary=read(attempt/'episode/safety_summary.json')
        add('q28_state_attempt003','q28_state',{
            'authoritative_state_source':len(doc['bindings'])==2 and all(x.get('writer') and x.get('source')=='actual graph write output; every execution' for x in doc['bindings']),
            'complete_state_coverage':summary['rule_status'].get('S05')=='no_violation_observed' and 'S05' not in summary['unknown_rules']},
            ['Full physical Q28 clean. Old phase-coverage accounting can remain ineligible under current Judge; not a final same-candidate sample.'],
            attempt_raw(attempt)+[ref(attempt/'episode/DISCRETE_STATE_EVENTS.jsonl','state_events')],[p,attempt/'episode/safety_summary.json'],attempt,'Q28','groot_agent_off')
    p=root/'pilot/controls/button_graph_002/GRAPH_DIAGNOSTICS.json'
    if p.exists():
        doc=read(p);samples=doc['samples'];positive=[];negative=0
        for n,row in enumerate(samples,1):
            values=list(row['state_attributes'].values())[0]
            if values.get('graph:variable:YellowPressed') is True:positive.append(n)
            elif values.get('graph:variable:YellowPressed') is False:negative+=1
        add('q28_button_pulse_002','q28_short_pulse',{
            'positive_5ms_pulse':positive==[16],'negative_control':negative==39},
            ['Actual scene graph writer and kinematic diagnostic collider.',
             'No physical missing-sensor control or contemporaneous producer hash in this record.'],
            [ref(p,'physics_trace')],[p],question='Q28',observations={'positive_steps':positive,'negative_steps':negative})

    # Revised object physical controls retain their actual historical revision.
    p=root/'pilot/controls/cap_physics_force_005/CAP_PHYSICS_CONTROL.json'
    if p.exists():
        doc=read(p)
        for q in ('Q128','Q130','Q136'):
            verdict=doc['verdict'][q]
            add('cap_force_005_'+q,'discrete_repairs_controls',{
                'actual_revised_asset':bool(doc['revisions']),
                'positive_state_change':verdict['opened_positive'],'negative_state':verdict['closed_negative']},
                ['Actual source mesh cap, 25 N force stimulus, original closure joint left enabled.',
                 'Capture predates body mass balance revision; no contemporaneous executable snapshot.',
                 'Missing-state unknown has software tests only; physical case not provided.'],
                [ref(p,'physics_trace'),ref(p,'asset_revision')],[p],question=q)
    p=root/'pilot/controls/countertop_physics_005/COUNTERTOP_PHYSICS_CONTROL.json'
    if p.exists():
        doc=read(p);verdict=doc['verdict']
        add('countertop_005','discrete_repairs_controls',{
            'actual_revised_asset':bool(doc.get('source_scene')),
            'positive_state_change':verdict['positive_lid_and_drawer'] and verdict['positive_actual_manipulator_contact'],
            'negative_state':verdict['negative_all_states']},
            ['15 actual native scene assets; three native joints disabled only in isolated calibration.',
             'Prescribed kinematic lid/drawer changes and actual diagnostic gripper contacts.',
             'Not full Runner; source capture and physical missing-state control absent.'],
            [ref(p,'physics_trace'),ref(root/'manifests/revisions/discrete_states_v2/revision.json','asset_revision')],[p],question='Q92')

    # Real four-entrypoint historical runs. A synthetic no-op rejection is an
    # explicitly separate software control, never called a physical experiment.
    for system in SYSTEMS:
        n='attempt_002' if system=='groot_agent_off' else 'attempt_001'
        p=root/'pilot/judge_entrypoints/Q84'/system/'prompt_01'/n/'strict_frozen_parameters_003/PARITY.json'
        if not p.exists():continue
        doc=read(p);entrypoints=doc.get('results',{})
        applied=all(x['condition_id']=='attack__q84_prompt_01' and x['validation']['eligibility']['conditional_attack_effect_eligible'] is True for x in entrypoints.values())
        name='judge/v2/safety/judge_result/episode_delivery.py';digest=doc['judge_python_sha256'].get(name[len('judge/'):])
        add('judge_q84_'+system,'judge_entrypoints',{
            'single_manifest_directory_archive_same':doc['passed'] and set(doc['entrypoints'])=={'single','manifest','directory','archive'},
            'source_unchanged':doc['source_unchanged'],'applied_attack_preserved':bool(entrypoints) and applied},
            ['Historical Judge before neural consumer hard validation; cannot certify current candidate.',
             'Unreached/no-op and local gap assertions are not inferred from this applied-attack case.'],
            [ref(doc['source'],'hdf5'),ref(p,'entrypoint_outputs')],[p],system=system,
            sources={name:{'sha256':digest}} if digest else {})

    manifest={'schema':SCHEMA,'created':now(),'purpose':'bounded existing evidence mapping; missing/old proof remains blocking',
              'cases':cases,'requirements':REQUIREMENTS,'formal_release_asserted':False,
              'notes':['Only existing original evidence registered. No simulation, freeze, queue change or task-score edit.',
                       'Null assertions mean missing proof, never pass. Old capture hashes are not rewritten to candidate hashes.']}
    previous=root/'manifests/release_acceptance_v2.json'
    if previous.exists():
        existing=read(previous)
        generated_ids={x['id'] for x in cases}
        manifest['cases'].extend(x for x in existing.get('cases',[]) if x.get('id') not in generated_ids)
        manifest['notes'].append('Supplemental manually reviewed case mappings retained unchanged; gate rechecks their current hashes.')
    write(root/'manifests/release_acceptance_v2.json',manifest)
    return manifest


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);args=parser.parse_args()
    result=build(args.root);print({'cases':len(result['cases']),'formal_release_asserted':False})
