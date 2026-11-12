"""Persist the measured geometry ledger and the actual final collision shapes."""
import json
import h5py
import numpy as np


def encode(value):
    def default(x):
        if isinstance(x,np.ndarray):return x.tolist()
        if isinstance(x,np.generic):return x.item()
        raise TypeError(type(x).__name__)
    return json.dumps(value,ensure_ascii=False,allow_nan=False,default=default,separators=(',',':')).encode()


class GeometryEvidence:
    def __init__(self,tracker,out,dt):
        self.tracker=tracker;self.dt=dt;self.pending=[];self.last_step=0
        self.file=h5py.File(out/'safety_geometry_observations.hdf5','x')
        self.file.attrs.update(schema='paper200-geometry-observations/v2',complete=False,
            physics_dt=dt,source='actual post-step PhysX poses and final composed collision shapes')
        self.file.create_dataset('json_bytes',shape=(0,),maxshape=(None,),dtype='u1',chunks=(1048576,),compression='gzip')
        self.file.create_dataset('offsets',data=np.asarray([0],np.int64),maxshape=(None,),chunks=(256,),compression='gzip')
        self.file.create_dataset('physics_step',shape=(0,),maxshape=(None,),dtype='i8',chunks=(256,),compression='gzip')
        self.file.create_dataset('time_s',shape=(0,),maxshape=(None,),dtype='f8',chunks=(256,),compression='gzip')
        if tracker is not None:self.export_shapes(out/'safety_geometry_shapes.hdf5')

    def export_shapes(self,path):
        with h5py.File(path,'x') as f:
            f.attrs.update(schema='paper200-final-collision-shapes/v2',complete=False,
                frame='record local geometry; owner_pose xyz+wxyz composed with owner_local_matrix',
                contact_offsets='distance is geometric collision-surface distance excluding contact/rest offsets')
            records=[];parts=[];offsets=[0];record_part_ranges=[]
            for record in self.tracker.records:
                params={k:v for k,v in record.parameters.items() if k not in ('convex_parts','distance_bound_hull_vertices')}
                records.append(dict(path=record.path,owner=record.owner,kind=record.kind,parameters=params,
                    complete=record.complete,reason=record.reason,owner_local_matrix=record.owner_local_matrix,
                    static_world_matrix=record.static_world_matrix,initial_bounds_xy=record.initial_bounds_xy))
                start=len(parts)
                for part in record.parameters.get('convex_parts',[]):
                    value=np.asarray(part,np.float64);parts.append(value);offsets.append(offsets[-1]+len(value))
                record_part_ranges.append([start,len(parts)])
            f.create_dataset('records_json',data=np.frombuffer(encode(records),np.uint8),compression='gzip')
            f.create_dataset('vertices',data=np.concatenate(parts) if parts else np.empty((0,3)),compression='gzip')
            f.create_dataset('part_offsets',data=np.asarray(offsets,np.int64),compression='gzip')
            f.create_dataset('record_part_ranges',data=np.asarray(record_part_ranges,np.int64),compression='gzip')
            f.create_dataset('scope_manifest_json',data=np.frombuffer(encode({
                'tracker':self.tracker.manifest(),'profile':self.tracker.profile,
                'robot_records':[r.path for r in self.tracker.robot_records],
                'S08_obstacles':[r.path for r in self.tracker._s08_obstacles],
                'S03_forbidden':{k:[r.path for r in v] for k,v in self.tracker._s03_forbidden.items()},
                'distance_verdict':'safe lower bound >= threshold certifies safety; only exact witness below threshold certifies violation'}),np.uint8),compression='gzip')
            f.attrs['complete']=True

    def observe(self,step,time_s,sample):
        if step!=self.last_step+1 or abs(time_s-step*self.dt)>1e-7:
            raise ValueError('geometry observation physics clock gap')
        row={k:sample.get(k) for k in ('lifecycle_phase','mission_phase','docking_completed','geometry_metrics')}
        row.update(physics_step=step,sim_time_s=time_s)
        self.pending.append((step,time_s,encode(row)));self.last_step=step
        if len(self.pending)>=256:self.flush()

    def flush(self):
        if not self.pending:return
        f=self.file;raw=b''.join(row[2] for row in self.pending);start=len(f['json_bytes']);n=len(self.pending)
        f['json_bytes'].resize(start+len(raw),axis=0);f['json_bytes'][start:]=np.frombuffer(raw,np.uint8)
        count=len(f['physics_step'])
        offsets=start+np.cumsum([len(row[2]) for row in self.pending])
        f['offsets'].resize(count+n+1,axis=0);f['offsets'][count+1:]=offsets
        for name,col in (('physics_step',0),('time_s',1)):
            f[name].resize(count+n,axis=0);f[name][count:]=[row[col] for row in self.pending]
        self.pending.clear();f.flush()

    def close(self):
        self.flush();self.file.attrs.update(complete=True,samples=self.last_step);self.file.close()


def validate_and_embed(collection,episode,physical_times):
    path=episode/'safety_geometry_observations.hdf5'
    if not path.exists():return False
    with h5py.File(path,'r') as source:
        n=len(source['time_s'])
        if not source.attrs.get('complete') or not np.array_equal(source['time_s'][...],physical_times):
            raise ValueError('geometry/rubric sample clock mismatch')
        if not np.array_equal(source['physics_step'][...],np.arange(1,n+1)):
            raise ValueError('geometry step coverage incomplete')
        offsets=source['offsets'][...]
        if len(offsets)!=n+1 or offsets[0]!=0 or offsets[-1]!=len(source['json_bytes']) or np.any(np.diff(offsets)<=0):
            raise ValueError('geometry ledger byte ranges invalid')
        target=collection.create_group('geometry_observations');target.attrs.update(dict(source.attrs))
        for name in source:source.copy(source[name],target,name=name)
    shapes=episode/'safety_geometry_shapes.hdf5'
    if shapes.exists():
        with h5py.File(shapes,'r') as source:
            if not source.attrs.get('complete'):raise ValueError('collision shape export incomplete')
            target=collection.create_group('geometry_shapes');target.attrs.update(dict(source.attrs))
            for name in source:source.copy(source[name],target,name=name)
    return True
