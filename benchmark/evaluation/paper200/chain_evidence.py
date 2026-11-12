"""Check activation and execution evidence without computing task scores."""
import json
from pathlib import Path
from collections import Counter
from .common import read,write,now
from .queue import connect


def physical_summary(value):
    if isinstance(value,dict):return {k:physical_summary(v) for k,v in value.items() if k!='attack_metadata'}
    if isinstance(value,list):return [physical_summary(v) for v in value]
    return value


def inspect(episode,job):
    profile=read(episode/'resolved_attack_profile/all_attack_definitions.json')
    selected='q%d_%s'%(int(job['question'][1:]),job['variant'])
    enabled={a['id'] for a in profile['attack']['attacks'] if a.get('enabled') and a.get('effect_enabled',True)}
    events=episode/'attack_evidence/unified_attack_audit.jsonl'
    initialized=[];triggered=set();effective=set();applications=Counter();handlers={};first_effect={}
    intercepted=set();changed_payloads=set()
    with events.open() as stream:
        for line in stream:
            item=json.loads(line);kind=item.get('event');aid=item.get('attack_id')
            if kind=='attack_runtime_initialized':initialized.append(item['attack_enabled'])
            if kind=='condition_triggered':triggered.add(aid)
            if kind=='effect_first_applied' and item.get('effect_applied'):
                evidence=item['evidence'];before=evidence.get('target_before_sha256');after=evidence.get('target_after_sha256')
                if before is not None and after is not None and before==after:
                    raise ValueError('attack effect marker claims an unchanged payload')
                effective.add(aid);handlers[aid]=evidence.get('executor');first_effect[aid]=item['time_s']
            if kind in ('payload_intercepted','immediate_executed'):applications[aid]+=1
            if kind=='payload_intercepted':
                intercepted.add(aid)
                if physical_summary(item['before'])!=physical_summary(item['after']):changed_payloads.add(aid)
    if effective & (intercepted-changed_payloads):
        raise ValueError('effect marker has no physical payload change; audit annotations excluded')
    if len(initialized)!=1:raise ValueError('attack runtime identity missing/duplicated')
    clean=job['variant']=='clean'
    if initialized[0]!=(not clean):raise ValueError('attack switch mismatch')
    if clean and (effective or applications):raise ValueError('clean episode contains applied attacks')
    if not clean and enabled!={selected}:raise ValueError('attack configuration is not exactly the selected variant')
    if effective-({selected} if not clean else set()):raise ValueError('unexpected attack actually applied')
    return {'configured':not clean,'configured_attack_ids':[] if clean else sorted(enabled),
            'triggered':not clean and selected in triggered,'effective':not clean and selected in effective,
            'all_condition_markers':sorted(triggered),'effect_handlers':handlers,'first_effect_time_s':first_effect,
            'applications':dict(applications),'effect_semantics':'input/command/scene modification; does not imply adversarial task success'}


def report(root):
    records=[];problems=[];covered=set();natural=set();probes=set()
    with connect(root/'queue.sqlite3') as db:
        rows=[dict(r) for r in db.execute("SELECT * FROM runs WHERE status='collected'")]
    for job in rows:
        try:
            evidence=inspect(Path(job['output'])/'episode',job)
            if evidence['effective']:
                covered.update(evidence['effect_handlers'].values())
                (probes if job['purpose']=='probe' else natural).update(evidence['effect_handlers'].values())
            records.append({k:job[k] for k in ['run_id','question','system','variant','purpose','output']}|evidence)
        except Exception as exc:problems.append({'run_id':job['run_id'],'error':str(exc)})
    required=set()
    for row in read(root/'manifests/questions.json'):
        for p in (root/'benchmark/task'/row['question']/'data/attack/profiles').glob('*.json'):
            required.update(a['handler'] for a in read(p).get('attack',{}).get('attacks',[]))
    result={'time':now(),'passed':not problems and not required-covered,'problems':problems,
        'required_handlers':sorted(required),'physically_exercised_handlers':sorted(covered),
        'original_trigger_handlers':sorted(natural),'controlled_trigger_probe_handlers':sorted(probes),
        'probe_interpretation':'physical boundary diagnostics with explicit trigger override; excluded from original-trigger coverage and all performance denominators',
        'missing_physical_handler_coverage':sorted(required-covered),'episodes':records,'score_status':'NOT_SCORED'}
    write(root/'reports/ATTACK_CHAIN_VALIDATION.json',result)
    return result
