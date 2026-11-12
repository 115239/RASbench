"""Shared Runner/Judge evidence semantics. No policy or Agent inputs live here."""
import hashlib
import json
import math

VERSION = 'paper200-runner-judge/v2'
IDENTITY_FIELDS = ('episode_id', 'run_id', 'attempt_id', 'task_id', 'system_id',
                   'system_version', 'condition_id', 'selected_attack_ids', 'seed',
                   'policy_seed', 'attack_seed', 'block_id', 'repeat_index', 'purpose')


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=False, allow_nan=False).encode('utf-8')).hexdigest()


def condition_id(selected_attack_ids):
    ids = list(selected_attack_ids)
    if any(not isinstance(x, str) or not x or '+' in x for x in ids) or len(set(ids)) != len(ids):
        raise ValueError('invalid selected attack IDs')
    return 'attack__' + '+'.join(sorted(ids)) if ids else 'clean'


def normalize_question(value):
    text = str(value)
    if not text.startswith('Q') or not text[1:].isdigit() or not 1 <= int(text[1:]) <= 200:
        raise ValueError('invalid paper200 question')
    return 'Q%d' % int(text[1:])


def scalar(value):
    if hasattr(value, 'tolist'):
        value = value.tolist()
    if isinstance(value, bytes):
        return value.decode('utf-8')
    return value


def json_dataset(group, key):
    if key not in group:
        return {}
    value = scalar(group[key][()])
    if isinstance(value, list):
        value = bytes(value).decode('utf-8')
    result = json.loads(value)
    if not isinstance(result, dict):
        raise ValueError('expected JSON object: ' + key)
    return result


def resolve_identity(hdf5, *, question_id, trusted=None):
    """Resolve declarations only. No attack-effect or pathname inference.

    A caller supplying a trusted manifest must already have verified its source
    hash. All recorded sources must agree; precedence never hides a conflict.
    Legacy extension variants are accepted only with matching question/provenance.
    Unknown seed/condition remains null, never zero/clean.
    """
    question = normalize_question(question_id)
    metadata = {k: scalar(v) for k, v in hdf5['metadata'].attrs.items()}
    candidates = [('hdf5:/metadata', metadata)]
    extension = hdf5.get('extensions/paper200_collection')
    if extension is not None:
        attrs = {k: scalar(v) for k, v in extension.attrs.items()}
        provenance = json_dataset(extension, 'provenance_json')
        for source in (attrs, provenance):
            if source.get('question') is not None and normalize_question(source['question']) != question:
                raise ValueError('PAPER200_QUESTION_SOURCE_MISMATCH')
        if attrs.get('question') is None or provenance.get('question') is None:
            raise ValueError('PAPER200_IDENTITY_QUESTION_MISSING')
        for name in ('system', 'variant', 'seed', 'purpose'):
            if name in attrs and name in provenance and attrs[name] != provenance[name]:
                raise ValueError('PAPER200_IDENTITY_SOURCE_MISMATCH:' + name)
        legacy = {**provenance, **attrs}
        mapped = {k: legacy[k] for k in IDENTITY_FIELDS if k in legacy}
        mapped['system_id'] = legacy.get('system')
        if legacy.get('variant') is not None:
            variant = legacy['variant']
            if not isinstance(variant, str) or not variant or '+' in variant:
                raise ValueError('invalid legacy variant')
            ids = [] if variant == 'clean' else ['q%d_%s' % (int(question[1:]), variant)]
            mapped.update(condition_id=condition_id(ids), selected_attack_ids=ids)
        candidates.append(('hdf5:/extensions/paper200_collection', mapped))
        standard = json_dataset(extension, 'identity_json')
        if standard:
            candidates.append(('hdf5:/extensions/paper200_collection/identity_json', standard))
    if trusted is not None:
        candidates.append(('trusted_manifest', trusted))
    result = {k: None for k in IDENTITY_FIELDS}
    sources = {}
    for source, values in candidates:
        for key in IDENTITY_FIELDS:
            value = values.get(key)
            if value is None:
                continue
            if isinstance(value,str) and key != 'selected_attack_ids':
                value=value.strip()
                if not value:continue
            if key == 'selected_attack_ids' and isinstance(value, str):
                value = json.loads(value)
            if result[key] is not None and result[key] != value:
                raise ValueError('IDENTITY_SOURCE_MISMATCH:' + key)
            result[key] = value
            sources.setdefault(key, []).append(source + ('@' if source=='hdf5:/metadata' else ':') + key)
    if result['selected_attack_ids'] is not None:
        declared = condition_id(result['selected_attack_ids'])
        if result['condition_id'] not in (None, declared):
            raise ValueError('CONDITION_SELECTION_MISMATCH')
        result['condition_id'] = declared
    for key in ('seed', 'policy_seed', 'attack_seed', 'repeat_index'):
        if result[key] is not None and (isinstance(result[key], bool) or not isinstance(result[key], int) or result[key] < 0):
            raise ValueError('invalid identity integer:' + key)
    result['identity_sources'] = {k:v[0] if len(v)==1 else v for k,v in sources.items()}
    return result


def resolve_termination(hdf5):
    """Read explicit outcomes; old flow/task_accepted placeholders are not scores."""
    extension = hdf5.get('extensions/paper200_collection')
    fields = ('termination_reason','termination_time_s','termination_physics_step',
              'time_limit_s','task_success','task_failure','truncated','task_started',
              'task_start_sim_time_s','task_start_physics_step',
              'record_end_sim_time_s','record_end_physics_step')
    result = {name:None for name in fields}
    sources = {}
    if extension is not None:
        recorded = json_dataset(extension,'result_json')
        for name in fields:
            if name in recorded:
                result[name] = recorded[name]
                sources[name] = 'hdf5:/extensions/paper200_collection/result_json:' + name
        if result['termination_reason'] is None and recorded.get('failure'):
            result['termination_reason'] = recorded['failure']
            sources['termination_reason'] = 'hdf5:/extensions/paper200_collection/result_json:failure'
        result['legacy_task_accepted'] = recorded.get('task_accepted')
        result['legacy_flow_completed'] = recorded.get('flow_completed')
    result['task_evaluation_status'] = ('not_evaluated' if result['task_success'] is None
                                      and result['task_failure'] is None else 'explicitly_recorded')
    result['field_sources'] = sources
    return result


def project_observation_gaps(events, summary, *, start_time_s, end_time_s, coverage_complete):
    """An unknown rule clears locally only with a complete enumerated gap ledger."""
    declared = set(summary.get('unknown_rules', []))
    counts = summary.get('unknown_interval_count', {})
    observed = {}
    projected = {}
    malformed = set()
    for index, event in enumerate(events):
        if event.get('kind') != 'observation_gap':continue
        rule = event.get('rule_id')
        if not rule:continue
        observed[rule] = observed.get(rule, 0) + 1
        a = event.get('start', {}).get('sim_time_s')
        b = event.get('end', {}).get('sim_time_s')
        if not all(isinstance(t, (int,float)) and not isinstance(t,bool) and math.isfinite(t) for t in (a,b)) or b < a:
            malformed.add(rule)
            continue
        # The resumed sample at b is observed; a gap still open at episode end
        # includes its final sample. Preserve these distinct endpoint semantics.
        end_closed = event.get('end_reason') != 'observation_resumed'
        if a > end_time_s or b < start_time_s or (b == start_time_s and not end_closed):continue
        projected.setdefault(rule, []).append({
            'source_event_id':event.get('source_event_id', 'm20_safety_v1:events_jsonl:%d'%index),
            'source':{'hdf5_path':'/extensions/m20_safety_v1/events_jsonl','row_index':index},
            'original_interval':{'start':event['start'], 'end':event['end']},
            'window_projection':{'start_time_s':max(a,start_time_s), 'end_time_s':min(b,end_time_s),
                                 'end_closed':end_closed if b <= end_time_s else True},
            'metrics':event.get('start',{}).get('metrics',{}),
            'end_reason':event.get('end_reason')})
    unknown = set(projected) | malformed
    for rule in declared | set(observed) | {k for k,v in counts.items() if v != 0}:
        count = counts.get(rule)
        ledger_complete = type(count) is int and count > 0 and count == observed.get(rule,0) and rule not in malformed
        if not coverage_complete or not ledger_complete:
            unknown.add(rule)
    return {'unknown_rules':sorted(unknown),'windows':projected,'malformed_rules':sorted(malformed)}


class Lifecycle:
    """Continuous physics clock with an explicit, once-only task origin."""
    NEXT = {'PREPARE': {'WARMUP', 'STOPPING'}, 'WARMUP': {'READY', 'STOPPING'},
            'READY': {'TASK_RUNNING', 'STOPPING'}, 'TASK_RUNNING': {'STOPPING'},
            'STOPPING': {'FINALIZED'}, 'FINALIZED': set()}

    def __init__(self):
        self.phase = 'PREPARE'
        self.task_start_sim_time_s = None
        self.task_start_physics_step = None
        self.events = []

    def transition(self, phase, sim_time_s, physics_step, *, readiness=None):
        if phase not in self.NEXT[self.phase]:
            raise ValueError('invalid lifecycle transition: ' + self.phase + ' -> ' + phase)
        if not math.isfinite(sim_time_s) or sim_time_s < 0 or type(physics_step) is not int or physics_step < 0:
            raise ValueError('invalid continuous physics clock')
        if self.events and (sim_time_s < self.events[-1]['sim_time_s'] or physics_step < self.events[-1]['physics_step']):
            raise ValueError('physics clock moved backwards')
        if phase == 'TASK_RUNNING':
            if not readiness or not all(readiness.get(k) is True for k in ('stable_reference', 'policy_channel', 'initialization')):
                raise ValueError('task start requires verified readiness')
            self.task_start_sim_time_s = sim_time_s
            self.task_start_physics_step = physics_step
        self.phase = phase
        event = {'phase': phase, 'sim_time_s': sim_time_s, 'physics_step': physics_step,
                 **self.clock(sim_time_s, physics_step)}
        if readiness is not None:
            event['readiness'] = readiness
        self.events.append(event)
        return event

    def clock(self, sim_time_s, physics_step):
        started = self.task_start_sim_time_s is not None
        return {'task_started': started, 'task_start_sim_time_s': self.task_start_sim_time_s,
                'task_start_physics_step': self.task_start_physics_step,
                'task_time_s': sim_time_s - self.task_start_sim_time_s if started else None,
                'task_physics_step': physics_step - self.task_start_physics_step if started else None}


def phase_applicability(events, summary, rule_id, start_time_s, end_time_s):
    """Project the declared phase ledger; absence is unknown, never clear.

    A post-step observation covers the first physics interval. Subsequent
    transitions apply at their recorded sample, including the scope endpoint.
    """
    rows=[e for e in events if e.get('kind')=='rule_applicability' and e.get('rule_id')==rule_id]
    if not rows:return None
    declared=summary.get('rule_applicability_event_count',{}).get(rule_id)
    if type(declared) is not int or declared!=len(rows):return None
    try:
        times=[float(row['point']['sim_time_s']) for row in rows]
        if not all(math.isfinite(t) for t in times) or any(b<=a for a,b in zip(times,times[1:])):return None
        origin=float(summary['start_sim_time_s'])
        end=float(summary['end_sim_time_s'])
        if start_time_s<origin or end_time_s>end or end_time_s<start_time_s:return None
        dt=float(summary.get('physics_dt') or
            (summary['end_sim_time_s']-origin)/(summary['end_physics_step']-summary['start_physics_step']))
        if rows[0]['point']['sim_time_s']>origin+dt+1e-8:return None
        if any(e.get('applicable') is not True and e.get('applicable') is not False and
               e.get('applicable') is not None for e in rows):return None
        state=None;values=[]
        for i,row in enumerate(rows):
            time=float(row['point']['sim_time_s'])
            if i==0:time=origin
            if time<=start_time_s:state=row['applicable']
            elif time<=end_time_s:values.append(row['applicable'])
        values.insert(0,state)
        if any(v is True for v in values):return True
        if any(v is None for v in values):return None
        return False
    except (KeyError,TypeError,ValueError,ZeroDivisionError):return None
