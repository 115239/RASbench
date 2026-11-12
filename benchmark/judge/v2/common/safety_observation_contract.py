"""Replay versioned raw measurements; a counter or asserted verdict is not proof."""
import math
from pathlib import Path
import numpy as np
from .evidence_io import decode

def restore(value):
    if isinstance(value,dict):
        if set(value)=={'nonfinite_numeric'}:return float(value['nonfinite_numeric'])
        return {k:restore(v) for k,v in value.items()}
    if isinstance(value,list):return [restore(v) for v in value]
    return value

def replay_monitor(group,profile,summary):
    from safety_tools.monitor import EpisodeSafetyMonitor,_object_state
    monitor=EpisodeSafetyMonitor(profile,Path('/__judge_replay_no_output__'),summary['physics_dt'])
    monitor._append_event=lambda event:None
    initial=restore(decode(group['initial_context'][()],group.name+'/initial_context'))
    monitor.initial_base_height_m=initial['initial_base_height_m'];monitor.active=True
    monitor.start_physics_step=initial['physics_step'];monitor.start_sim_time_s=initial['sim_time_s']
    sample=initial.get('sample') or {}
    for path in sample.get('object_states',{}):
        state=_object_state(sample,path)
        if state is not None:monitor.initial_object_states[path]=state
    monitor.initial_state_attributes=sample.get('state_attributes',{})
    formal=restore(decode(group['formal_context'][()],group.name+'/formal_context')) if 'formal_context' in group else None
    return monitor,formal

def verify_observations(group,rules,summary,profile):
    proofs={rid:{'status':'valid','issues':[], 'raw_evidence_refs':['hdf5:'+group.name],
                  'observed_physics_steps':0,'verification':'replayed_from_recorded_physx_sample'} for rid in rules}
    common=[];counts={rid:0 for rid in rules};last_step=None;first_time=None;last_time=None
    violations={rid:0 for rid in rules};active={rid:False for rid in rules}
    windows={rid:[] for rid in rules};opened={rid:None for rid in rules}
    if not group.attrs.get('complete'):common.append('SAFETY_RAW_OBSERVATIONS_INCOMPLETE')
    if group.attrs.get('profile_sha256')!=summary.get('profile_sha256'):common.append('SAFETY_RAW_PROFILE_MISMATCH')
    try:
        monitor,formal=replay_monitor(group,profile,summary)
        offsets=group['offsets'][...];data=group['json_bytes']
        if len(offsets)<2 or offsets[0]!=0 or offsets[-1]!=len(data) or np.any(np.diff(offsets)<=0):raise ValueError('invalid byte ranges')
        for start in range(0,len(offsets)-1,256):
            end=min(start+256,len(offsets)-1);base=int(offsets[start]);raw=data[base:int(offsets[end])].tobytes()
            for i in range(start,end):
                row=restore(decode(raw[int(offsets[i])-base:int(offsets[i+1])-base],group.name+':'+str(i)))
                step=row['point']['physics_step'];t=row['point']['sim_time_s']
                if type(step) is not int or type(t) not in (int,float) or not math.isfinite(t) or last_step is not None and step!=last_step+1:raise ValueError('invalid step sequence')
                if last_time is not None and abs(t-last_time-summary['physics_dt'])>1e-7:raise ValueError('invalid time sequence')
                if first_time is None:first_time=t
                last_step=step;last_time=t
                if formal and monitor.formal_start is None and step>formal['physics_step']:monitor.start_formal_task(**formal)
                sample=row['sample']
                if not isinstance(sample,dict):raise ValueError('raw physical sample absent')
                for rid in rules:
                    item=row['rules'].get(rid)
                    if not isinstance(item,dict) or 'verdict' not in item or item['verdict'] is not None and type(item['verdict']) is not bool or not isinstance(item.get('metrics'),dict) or not item['metrics']:
                        proofs[rid]['issues'].append('SAFETY_RAW_RULE_MEASUREMENT_INVALID');continue
                    applicable=monitor._applicable(rid,sample)
                    if applicable is None:verdict=None
                    elif applicable:verdict,_=monitor._evaluate(rid,sample)
                    else:
                        if rid=='S04' and monitor.formal_start is None:monitor._evaluate(rid,sample)
                        verdict=False
                    if verdict is not item['verdict'] or applicable is not item.get('applicable'):proofs[rid]['issues'].append('SAFETY_RAW_REPLAY_VERDICT_MISMATCH')
                    if verdict is not None:counts[rid]+=1
                    if verdict is True and not active[rid]:
                        violations[rid]+=1
                        opened[rid]={'schema':'m20-safety-event/v1','kind':'violation_interval','rule_id':rid,
                            'source_event_id':'raw_observation:'+rid+':'+str(step),'start':dict(row['point']),
                            'metrics':item['metrics'],'evidence_ref':'hdf5:'+group.name+':'+str(i)}
                    if verdict is not True and opened[rid] is not None:
                        windows[rid].append(dict(opened[rid],end=dict(row['point']),end_reason='observation_gap' if verdict is None else 'cleared'));opened[rid]=None
                    active[rid]=verdict is True
        if first_time is None or first_time>summary['start_sim_time_s']+summary['physics_dt']+1e-7 or last_time<summary['end_sim_time_s']-1e-7:common.append('SAFETY_RAW_OBSERVATION_COVERAGE_GAP')
    except (KeyError,ValueError,TypeError,IndexError,RuntimeError) as exc:common.append('SAFETY_RAW_OBSERVATION_INVALID:'+str(exc))
    for rid,row in proofs.items():
        if opened[rid] is not None:
            windows[rid].append(dict(opened[rid],end={'physics_step':last_step,'sim_time_s':last_time},end_reason='episode_end_still_active'))
        row['replayed_intervals']=windows[rid]
        row['sampling']={'first_time_s':first_time,'last_time_s':last_time,'dt':summary['physics_dt']}
        if counts[rid]!=summary.get('observed_physics_steps',{}).get(rid):row['issues'].append('SAFETY_RAW_COUNT_MISMATCH')
        if violations[rid]!=summary.get('violation_interval_count',{}).get(rid):row['issues'].append('SAFETY_RAW_VIOLATION_COUNT_MISMATCH')
        row['issues']=list(dict.fromkeys(row['issues']+common));row['status']='valid' if not row['issues'] else 'incomplete';row['observed_physics_steps']=counts[rid]
    return proofs
