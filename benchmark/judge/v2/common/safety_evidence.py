"""Bind safety verdicts to complete measurement streams, including old v2 runs."""
import json
import numpy as np
import h5py
from .evidence_io import decode

def _finite(ds):
    return all(np.isfinite(ds[i:i+1024]).all() for i in range(0, len(ds), 1024))

def prove_streams(hdf5, profile, summary):
    rules = {k:v for k,v in profile.get('rules', {}).items() if v.get('enabled')}
    result = {k: {'status': 'incomplete', 'issues': [], 'raw_evidence_refs': []} for k in rules}
    explicit = hdf5.get('/extensions/m20_safety_v1/observations')
    physical = hdf5.get('/extensions/paper200_physics')
    if isinstance(explicit, h5py.Group):
        from .safety_observation_contract import verify_observations
        return verify_observations(explicit, rules, summary, profile)
    # Historical files have no per-rule raw stream. Only certify rules whose
    # complete underlying quantities can be located, never by counts alone.
    valid = isinstance(physical, h5py.Group) and bool(physical.attrs.get('complete'))
    n = 0
    if valid:
        times = physical.get('time_s')
        if not isinstance(times, h5py.Dataset):
            valid = False
        else:
            t = times[...]; n = len(t)
            dt = summary.get('physics_dt') or (1.0 / physical.attrs.get('frequency_hz', 200))
            valid = n > 0 and np.isfinite(t).all() and np.allclose(np.diff(t), dt, rtol=0, atol=1e-7)
            valid = valid and t[0] <= summary['start_sim_time_s'] + dt + 1e-7 and t[-1] >= summary['end_sim_time_s'] - 1e-7
    for rid, row in result.items():
        refs = row['raw_evidence_refs']; issues = row['issues']
        if not valid:
            issues.append('SAFETY_PHYSICS_COVERAGE_UNAVAILABLE'); continue
        row['sampling']={'first_time_s':float(t[0]),'last_time_s':float(t[-1]),'dt':float(dt)}
        refs.append('hdf5:/extensions/paper200_physics/time_s')
        if rid in ('S05','S12'):
            from .legacy_pose_replay import replay
            try:result[rid]=dict(replay(hdf5,physical,rules[rid],rid,t),sampling=row['sampling'])
            except (ValueError,KeyError,TypeError,IndexError) as exc:issues.append('SAFETY_POSE_REPLAY_UNAVAILABLE:'+str(exc))
            continue
        required = {'S07': ['body_pose','body_velocity'], 'S09': ['body_velocity'],
                    'S10': ['body_velocity'], 'S11': ['joint_position','joint_velocity']}.get(rid)
        if required is None:
            # Remaining legacy rules require object/contact/geometry bindings,
            # not just a physics stream somewhere in the file.
            issues.append('SAFETY_RULE_RAW_BINDING_UNAVAILABLE'); continue
        for name in required:
            ds = physical.get(name)
            if not isinstance(ds,h5py.Dataset) or len(ds) != n or not _finite(ds):
                issues.append('SAFETY_RAW_STREAM_INVALID:' + name)
            else:
                refs.append('hdf5:' + ds.name)
        if rid == 'S11':
            issues.append('SAFETY_JOINT_EFFORT_AND_LIMIT_EVIDENCE_MISSING')
        if rid == 'S07':
            issues.append('SAFETY_SUPPORT_CONTACT_BINDING_UNAVAILABLE')
        if rid in ('S09','S10'):
            paths = decode(physical['body_paths_json'][()], physical['body_paths_json'].name)
            wanted = {x.get('body_name') for x in rules[rid]['parameters'].get('metrics', [])}
            if not wanted or any(sum(str(p).rsplit('/',1)[-1] == name for p in paths) != 1 for name in wanted):
                issues.append('SAFETY_BODY_BINDING_AMBIGUOUS')
            refs.append('hdf5:' + physical['body_paths_json'].name)
            if not issues:
                try:
                    if rules[rid].get('phase_scope','all_capture')!='all_capture':raise ValueError('phase-specific replay requires phase samples')
                    velocities=physical['body_velocity'][...].astype(float)
                    body_indices={str(p).rsplit('/',1)[-1]:i for i,p in enumerate(paths)}
                    mask=np.zeros(n,dtype=bool);maximum=0.0
                    for metric in rules[rid]['parameters']['metrics']:
                        index=body_indices[metric['body_name']];ref=metric.get('reference_body_name')
                        value=velocities[:,index,:3 if rid=='S09' else 6].copy()
                        if rid=='S10':value=value[:,3:]
                        if ref:
                            if sum(str(p).rsplit('/',1)[-1]==ref for p in paths)!=1:raise ValueError('reference body ambiguous')
                            j=body_indices[ref]
                            if rid=='S09':
                                poses=physical['body_pose'][...]
                                if not np.isfinite(poses).all():raise ValueError('nonfinite reference pose')
                                value-=velocities[:,j,:3]+np.cross(velocities[:,j,3:],poses[:,index,:3]-poses[:,j,:3])
                                refs.append('hdf5:'+physical['body_pose'].name)
                            else:value-=velocities[:,j,3:]
                        value=value[:,[{'x':0,'y':1,'z':2}[c] for c in metric['components']]]
                        speed=np.linalg.norm(value,axis=1);limit=metric['maximum_speed_mps' if rid=='S09' else 'maximum_angular_speed_radps']
                        maximum=max(maximum,float(speed.max()));mask|=speed>limit
                    intervals=[];starts=np.flatnonzero(mask & np.r_[True,~mask[:-1]])
                    for start in starts:
                        stop=int(start)+1
                        while stop<n and mask[stop]:stop+=1
                        end=float(t[stop]) if stop<n else float(t[-1])
                        intervals.append({'schema':'m20-safety-event/v1','kind':'violation_interval','rule_id':rid,
                            'source_event_id':'raw_physics:'+rid+':'+str(start),'start':{'sim_time_s':float(t[start])},
                            'end':{'sim_time_s':end},'end_reason':'cleared' if stop<n else 'episode_end_still_active',
                            'metrics':{'maximum_speed':maximum,'recomputed_from':list(refs),'row_start':int(start),'row_stop_exclusive':stop}})
                    row['replayed_intervals']=intervals;row['maximum_speed']=maximum;row['verification']='recomputed_from_physics_velocity'
                except (ValueError,KeyError,IndexError) as exc:issues.append('SAFETY_SPEED_REPLAY_UNAVAILABLE:'+str(exc))
        row['status'] = 'valid' if not issues else 'incomplete'
        row['observed_physics_steps'] = summary.get('observed_physics_steps',{}).get(rid,0)
    return result
