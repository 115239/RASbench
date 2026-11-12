"""Lossless block writes for 200 Hz physical observations."""
import numpy as np
from rubric_evidence_io import RubricEvidenceWriter as OriginalWriter

class RubricEvidenceWriter(OriginalWriter):
    block_size=256
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.pending={};self.written=0
    def _append(self,path,value,dtype):
        value=np.asarray(value,dtype=dtype)
        if value.dtype.kind=='f' and not np.isfinite(value).all():raise ValueError('nonfinite physical evidence: '+path)
        values=self.pending.setdefault(path,[])
        if values and (values[0].shape!=value.shape or values[0].dtype!=value.dtype):raise ValueError('physical evidence shape changed: '+path)
        values.append(value.copy())
    def flush_block(self):
        if not self.pending:return
        sizes={len(v) for v in self.pending.values()}
        if len(sizes)!=1:raise ValueError('physical evidence missing fields')
        count=sizes.pop()
        for path,values in self.pending.items():
            value=np.stack(values)
            if path not in self.file:
                parent,name=path.rsplit('/',1)
                self.file.require_group(parent).create_dataset(name,shape=(0,*value.shape[1:]),maxshape=(None,*value.shape[1:]),chunks=(self.block_size,*value.shape[1:]),dtype=value.dtype,compression='gzip',compression_opts=4)
            node=self.file[path]
            if node.shape[0]!=self.written:raise ValueError('physical evidence write gap')
            node.resize(self.written+count,axis=0);node[self.written:self.written+count]=value
        self.written+=count;self.pending.clear();self.file.flush()
    def step(self,*args,**kwargs):
        super().step(*args,**kwargs)
        if self.count%self.block_size==0:self.flush_block()
    def complete(self,expected_steps):
        self.flush_block()
        super().complete(expected_steps)

from rubric_evidence import RubricEvidenceRuntime as OriginalRuntime,array

class _SliceView:
    def __init__(self,parent,index):self.parent=parent;self.index=index
    def get_transforms(self):return self.parent.batch_poses[self.index:self.index+1]
    def get_velocities(self):return self.parent.batch_velocities[self.index:self.index+1]

class RubricEvidenceRuntime(OriginalRuntime):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.batch_view=None;self.physical_step=0;self.raw_contact_subscription=None;self.raw_contact_error=None
        # An affine-transformed mesh and its convex hull have identical extrema
        # on every axis. Reduce only the observation calculation, never PhysX
        # collision geometry; this avoids transforming 180k robot vertices/step.
        from scipy.spatial import ConvexHull,QhullError
        for g in self.geometry.values():
            points=g['points']
            if g['analytic'] or len(points)<32:continue
            unique=np.unique(points,axis=0)
            try:
                centered=unique-unique.mean(axis=0)
                _,singular,basis=np.linalg.svd(centered,full_matrices=False)
                rank=int(np.sum(singular>max(singular[0]*1e-10,1e-12)))
                if rank>=2:
                    hull=ConvexHull(centered@basis[:rank].T)
                    g['points']=unique[hull.vertices]
                else:g['points']=unique
            except QhullError:g['points']=unique
        if self.views:
            self.batch_view=self.sim_view.create_rigid_body_view(list(self.views))
            paths=list(self.batch_view.prim_paths)
            if set(paths)!=set(self.views):raise ValueError('batched physical body binding mismatch')
            self.views={p:_SliceView(self,i) for i,p in enumerate(paths)}
        import os,json
        from pathlib import Path
        from pxr import PhysicsSchemaTools
        from omni.physx import get_physx_simulation_interface
        out=Path(os.environ['THREE_SCENES_RUN_CONFIG']).parent/'episode'
        self.raw_contact_file=(out/'RUBRIC_CONTACT_POINTS.jsonl').open('x',encoding='utf-8',buffering=1024*1024)
        # Collider identity and semantic memberships are immutable for one
        # episode. Cache only those; never cache impulses, positions or events.
        raw_paths={};raw_membership={};raw_semantics_json={}
        self.raw_contact_timing={'calls':0,'wall_s':0.,'headers':0,'records':0}
        def record(headers,data):
            if not self.active:return
            lines=[]
            prefix=json.dumps({'physics_step':self.physical_step+1,
                'sim_time_s':(self.physical_step+1)*self.physics_dt},ensure_ascii=False,allow_nan=False)[:-1]
            for h in headers:
                identities=(h.collider0,h.collider1)
                if identities not in raw_paths:
                    raw_paths[identities]=tuple(str(PhysicsSchemaTools.intToSdfPath(p)) for p in identities)
                a,b=raw_paths[identities]
                if identities not in raw_membership:
                    relevant=[]
                    for pair in self.plan['contacts']:
                        s,t=self.collider_sets[pair['source']],self.collider_sets[pair['target']]
                        if (a in s and b in t) or (b in s and a in t):relevant.append(pair)
                    raw_membership[identities]=bool(relevant or any(p.startswith('/M20PiperVLA/M20_Piper/') for p in (a,b)))
                    # Preserve the same JSON key/order/whitespace and all pair
                    # duplicates; reuse only constant serialized metadata.
                    raw_semantics_json[identities]=(json.dumps({'collider0':a,'collider1':b},ensure_ascii=False,allow_nan=False)[1:-1],
                        json.dumps(relevant,ensure_ascii=False,allow_nan=False))
                if not raw_membership[identities]:continue
                points=[]
                for p in data[int(h.contact_data_offset):int(h.contact_data_offset)+int(h.num_contact_data)]:
                    points.append({'position_w':[float(p.position.x),float(p.position.y),float(p.position.z)],'normal_w':[float(p.normal.x),float(p.normal.y),float(p.normal.z)],
                        'impulse_w':[float(p.impulse.x),float(p.impulse.y),float(p.impulse.z)],'separation_m':float(p.separation)})
                # event_type stays before semantic_pairs, matching old output.
                base,semantic=raw_semantics_json[identities]
                separator=', "semantic_pairs": '
                lines.append(prefix+', '+base+', "event_type": '+json.dumps(str(h.type),ensure_ascii=False)
                    +separator+semantic+', "points": '+json.dumps(points,ensure_ascii=False,allow_nan=False)+'}\n')
                self.raw_contact_timing['records']+=1
                if len(lines)>=256:
                    self.raw_contact_file.write(''.join(lines));lines.clear()
            if lines:self.raw_contact_file.write(''.join(lines))
            # A completed PhysX callback remains visible to progress/audit
            # readers; no event data is buffered across callbacks.
            self.raw_contact_file.flush()
        def checked_record(headers,data):
            import time
            started=time.perf_counter()
            try:record(headers,data)
            except Exception as error:self.raw_contact_error=repr(error)
            finally:
                self.raw_contact_timing['calls']+=1
                self.raw_contact_timing['headers']+=len(headers)
                self.raw_contact_timing['wall_s']+=time.perf_counter()-started
        self.raw_contact_subscription=get_physx_simulation_interface().subscribe_contact_report_events(checked_record)
    def _kinematics(self):
        if self.batch_view is not None:
            self.batch_poses=array(self.batch_view.get_transforms())
            self.batch_velocities=array(self.batch_view.get_velocities())
        return super()._kinematics()
    def end_physics_step(self):
        if self.raw_contact_error:raise RuntimeError('contact evidence callback failed: '+self.raw_contact_error)
        super().end_physics_step()
        if self.active:self.physical_step+=1
    def close(self):
        self.raw_contact_subscription=None
        self.raw_contact_file.close()
        import json
        from pathlib import Path
        Path(self.raw_contact_file.name).with_name('RAW_CONTACT_TIMING.json').write_text(
            json.dumps(self.raw_contact_timing,indent=2)+'\n')
        super().close()

class PhysicalInventoryWriter:
    """Same-run rigid body and robot joint measurements, independent of scoring."""
    def __init__(self,c,output):
        import h5py,json
        from pxr import UsdPhysics
        self.c=c
        paths=[str(p.GetPath()) for p in c.final_stage.Traverse()
               if p.HasAPI(UsdPhysics.RigidBodyAPI) and UsdPhysics.RigidBodyAPI(p).GetRigidBodyEnabledAttr().Get() is not False]
        self.view=c.rubric_evidence_runtime.sim_view.create_rigid_body_view(paths)
        if self.view._backend is None:raise ValueError('physical inventory view unavailable')
        self.paths=list(self.view.prim_paths)
        self.file=h5py.File(output/'physics_inventory.hdf5','x')
        self.file.attrs.update(schema='paper200-physical-inventory/v1',complete=False,frequency_hz=200,pose_order='xyz_xyzw')
        self.file.create_dataset('body_paths_json',data=np.frombuffer(json.dumps(self.paths,ensure_ascii=False).encode(),np.uint8))
        self.file.create_dataset('joint_names_json',data=np.frombuffer(json.dumps(c.names,ensure_ascii=False).encode(),np.uint8))
        self.file.create_dataset('initial_body_pose',data=array(self.view.get_transforms()).astype(np.float32))
        self.file.create_dataset('initial_body_velocity',data=array(self.view.get_velocities()).astype(np.float32))
        self.file.create_dataset('initial_joint_position',data=array(c.robot.get_joint_positions()).astype(np.float32))
        from .reference_states import ReferenceStateRegistry
        self.references=ReferenceStateRegistry(c,self.paths,output)
        self.file.create_dataset('reference_paths_json',data=np.frombuffer(json.dumps(self.references.paths,ensure_ascii=False).encode(),np.uint8))
        self.file.create_dataset('reference_bindings_json',data=np.frombuffer(json.dumps(self.references.bindings,ensure_ascii=False).encode(),np.uint8))
        self.file.attrs.update(reference_pose_order='xyz_xyzw',reference_sample_axis='same time_s axis as body_pose',
            reference_missing_semantics='reference_valid=false and NaN pose; never a stable or zero-state substitute')
        self.pending={};self.count=0
    def step(self,time_s):
        sample={'time_s':np.asarray(time_s,np.float64),'body_pose':array(self.view.get_transforms()).astype(np.float32),
                'body_velocity':array(self.view.get_velocities()).astype(np.float32),
                'joint_position':array(self.c.robot.get_joint_positions()).astype(np.float32),
                'joint_velocity':array(self.c.robot.get_joint_velocities()).astype(np.float32)}
        if self.references.paths:
            poses,valid,_=self.references.sample(sample['body_pose'],time_s)
            sample.update(reference_pose=poses,reference_valid=valid)
        for k,v in sample.items():
            if k!='reference_pose' and not np.isfinite(v).all():raise ValueError('nonfinite physical inventory '+k)
            self.pending.setdefault(k,[]).append(v.copy())
        if len(self.pending['time_s'])>=256:self.flush()
    def flush(self):
        if not self.pending:return
        n=len(self.pending['time_s'])
        for k,values in self.pending.items():
            a=np.stack(values)
            if k not in self.file:self.file.create_dataset(k,shape=(0,*a.shape[1:]),maxshape=(None,*a.shape[1:]),chunks=(256,*a.shape[1:]),dtype=a.dtype,compression='gzip',compression_opts=4)
            d=self.file[k];d.resize(self.count+n,axis=0);d[self.count:]=a
        self.count+=n;self.pending.clear();self.file.flush()
    def close(self):
        self.flush();self.file.attrs.modify('complete',True);self.file.close();self.references.close()
