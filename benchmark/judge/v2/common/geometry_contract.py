"""Read-only clock/shape binding checks for the v2 geometry evidence ledger."""
import json
import numpy as np


def verify_geometry_ledger(hdf5,summary):
    root=hdf5.get('extensions/paper200_collection');issues=[]
    ledger=root.get('geometry_observations') if root is not None else None
    shapes=root.get('geometry_shapes') if root is not None else None
    if ledger is None:return {'passed':False,'issues':['GEOMETRY_OBSERVATION_LEDGER_MISSING']}
    if not ledger.attrs.get('complete'):issues.append('GEOMETRY_OBSERVATION_LEDGER_INCOMPLETE')
    try:
        steps=ledger['physics_step'][...];times=ledger['time_s'][...];offsets=ledger['offsets'][...]
        n=len(steps);dt=float(ledger.attrs['physics_dt'])
        if n==0 or dt<=0 or not np.isfinite(dt) or len(times)!=n or not np.all(np.isfinite(times)) or not np.array_equal(steps,np.arange(1,n+1)) or not np.allclose(times,steps*dt,atol=1e-7,rtol=0):
            issues.append('GEOMETRY_OBSERVATION_CLOCK_INVALID')
        if not n or times[0]>float(summary['start_sim_time_s'])+dt+1e-7 or times[-1]<float(summary['end_sim_time_s'])-1e-7:
            issues.append('GEOMETRY_OBSERVATION_COVERAGE_INCOMPLETE')
        if len(offsets)!=n+1 or offsets[0]!=0 or offsets[-1]!=len(ledger['json_bytes']) or np.any(np.diff(offsets)<=0):
            issues.append('GEOMETRY_OBSERVATION_BYTE_RANGES_INVALID')
        else:
            raw=ledger['json_bytes'][...].tobytes()
            for i,(a,b) in enumerate(zip(offsets,offsets[1:])):
                row=json.loads(raw[a:b])
                if row.get('physics_step')!=int(steps[i]) or abs(float(row.get('sim_time_s',float('nan')))-times[i])>1e-7 or not np.isfinite(row.get('sim_time_s',float('nan'))):
                    issues.append('GEOMETRY_OBSERVATION_ROW_CLOCK_MISMATCH');break
    except (KeyError,TypeError,ValueError,IndexError,OverflowError):issues.append('GEOMETRY_OBSERVATION_SCHEMA_INVALID')
    if shapes is None:issues.append('GEOMETRY_COLLISION_SHAPES_MISSING')
    else:
        if not shapes.attrs.get('complete'):issues.append('GEOMETRY_COLLISION_SHAPES_INCOMPLETE')
        try:
            records=json.loads(shapes['records_json'][...].tobytes());scope=json.loads(shapes['scope_manifest_json'][...].tobytes())
            offsets=shapes['part_offsets'][...];ranges=shapes['record_part_ranges'][...];vertices=shapes['vertices'][...]
            paths=[r['path'] for r in records]
            if not records or len(set(paths))!=len(paths) or vertices.ndim!=2 or vertices.shape[1]!=3 or not np.all(np.isfinite(vertices)) or len(offsets)==0 or offsets[0]!=0 or offsets[-1]!=len(vertices) or np.any(np.diff(offsets)<=0) or ranges.shape!=(len(records),2) or np.any(ranges<0) or np.any(ranges[:,1]<ranges[:,0]) or np.any(ranges[:,1]>len(offsets)-1):
                issues.append('GEOMETRY_COLLISION_SHAPES_INVALID')
            used=list(scope['robot_records'])+list(scope['S08_obstacles'])+[p for items in scope['S03_forbidden'].values() for p in items]
            path_set=set(paths)
            if any(p not in path_set for p in used):issues.append('GEOMETRY_COLLISION_SCOPE_BINDING_INVALID')
        except (KeyError,TypeError,ValueError,IndexError):issues.append('GEOMETRY_COLLISION_SHAPES_SCHEMA_INVALID')
    return {'passed':not issues,'issues':list(dict.fromkeys(issues)),
            'observation_path':'/extensions/paper200_collection/geometry_observations',
            'shape_path':'/extensions/paper200_collection/geometry_shapes'}
