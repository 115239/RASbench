"""Three-valued physical evaluation of final-source contracts.

No image/visual bounding box substitutes for collision geometry. A missing
binding, missing sample, or unmeasured support remains UNKNOWN. Dependencies
constrain the evaluation time, not just the reported completion timestamp.
"""
import json,math,re
from pathlib import Path
from urllib.parse import quote
import h5py
import numpy as np
from .evidence_io import decode,records

BBOARD='/extensions/bboard_rubric_v04'
COLLECTION='/extensions/paper200_collection'
ROBOT='/M20PiperVLA/M20_Piper'
BASE=ROBOT+'/M20/base_link'
FINGERS=[ROBOT+'/Piper/link7',ROBOT+'/Piper/link8']
GRIPPER=ROBOT+'/Piper/gripper_base'

class MissingEvidence(Exception):pass

def inside_planes(shapes,planes,transform):
    """Entire actual convex parts inside a bound region, without AABB proxies."""
    rotation=np.asarray(transform)[:3,:3];origin=np.asarray(transform)[:3,3]
    if not shapes or not planes:return None
    for part in shapes:
        if not part.get('complete') or part.get('shape') is None:return None
        for plane in planes:
            n=np.asarray(plane[:3],float);world_n=rotation@n
            maximum=float(np.dot(part['shape'].support(world_n)-origin,world_n)+plane[3])
            if maximum>0:return False
    return True

def conjunction(*values):
    if any(v is False for v in values):return False
    return None if any(v is None for v in values) else True

def reduce_trace(values,times,steps,*,duration=0.,always=False,complete=True,dt=.005):
    """Return (verdict, index). Unknown samples break every continuous run."""
    if not len(values):return None,None
    complete = complete and all(steps[i]==steps[i-1]+1 and abs(times[i]-times[i-1]-dt)<=1e-7 for i in range(1,len(values)))
    if always:
        if any(v is False for v in values):return False,next(i for i,v in enumerate(values) if v is False)
        return (True,len(values)-1) if complete and all(v is True for v in values) else (None,None)
    start=None
    for i,v in enumerate(values):
        gap=i and (steps[i]!=steps[i-1]+1 or abs(times[i]-times[i-1]-dt)>1e-7)
        if v is not True:start=None;continue
        if start is None or gap:start=i
        if times[i]-times[start]+1e-10>=duration:return True,i
    return (False,None) if complete and all(v is not None for v in values) else (None,None)


def bound_verdict(measured,minimum,maximum):
    """Certified lower bounds prove separation, never proximity."""
    if not measured.get('complete') or measured.get('distance_m') is None:return None
    d=measured['distance_m'];lower='lower_bound' in measured.get('distance_kind','')
    if lower:
        if maximum is not None and d>maximum:return False
        if maximum is None and d>=minimum:return True
        return None
    # An exact pair is only an upper bound if not all pairs were examined.
    witness='upper_bound' in measured.get('distance_kind','')
    if d<minimum:return False
    if witness and minimum>0:return None
    return bool(maximum is None or d<=maximum)

class Evidence:
    def __init__(self,h5,runtime):
        self.h5=h5;self.b=h5.get(BBOARD)
        if self.b is None:raise MissingEvidence('RUBRIC_PHYSICS_EVIDENCE_MISSING')
        self.times=np.asarray(self.b['sim_time_s']);self.dt=float(decode(self.b['manifest_json'][()]).get('physics_dt',.005))
        self.steps=np.rint(self.times/self.dt).astype(np.int64)
        if not np.all(np.asarray(self.b['physics_steps'])==1):raise MissingEvidence('PER_PHYSICS_STEP_SAMPLES_REQUIRED')
        self.mask=(self.times>=runtime['task_start_time_s']-1e-9)&(self.times<=runtime['end_time_s']+1e-9)
        self.indices=np.flatnonzero(self.mask)
        self.complete=bool(self.b.attrs.get('complete')) and len(self.indices)>0
        if len(self.indices):
            t=self.times[self.indices];s=self.steps[self.indices]
            self.complete &= bool(np.all(np.diff(s)==1) and np.all(np.abs(np.diff(t)-self.dt)<1e-7) and t[0]<=runtime['task_start_time_s']+self.dt+1e-7 and t[-1]>=runtime['end_time_s']-self.dt-1e-7)
        self.cache={};self.contact_records=None;self.tracker=None;self.raw_samples=None
    def array(self,prim,name):
        path=BBOARD+'/objects/'+quote(prim,safe='')+'/'+name
        if path not in self.h5:
            phys=self.h5.get('/extensions/paper200_physics')
            if phys is not None and prim in decode(phys['body_paths_json'][()]) and name in ('root_pose','root_velocity'):
                key=('physical',prim,name)
                if key not in self.cache:
                    j=decode(phys['body_paths_json'][()]).index(prim)
                    if not np.array_equal(phys['time_s'][...],self.times):raise MissingEvidence('PHYSICS_CLOCK_MISMATCH')
                    raw=np.asarray(phys['body_pose' if name=='root_pose' else 'body_velocity'])[:,j]
                    self.cache[key]=raw[:,[0,1,2,6,3,4,5]] if name=='root_pose' else raw
                return self.cache[key]
            raise MissingEvidence('OBJECT_EVIDENCE_MISSING:'+path)
        if path not in self.cache:self.cache[path]=np.asarray(self.h5[path])
        return self.cache[path]
    def pose(self,prim,i):return self.array(prim,'root_pose')[i]
    def initial(self,prim,name='root_pose'):
        path=BBOARD+'/formal_initial/'+name
        # Export stores initial values per object group.
        for p in (BBOARD+'/formal_initial/objects/'+quote(prim,safe='')+'/'+name,
                  BBOARD+'/formal_initial/'+quote(prim,safe='')+'/'+name,
                  BBOARD+'/objects/'+quote(prim,safe='')+'/formal_initial/'+name):
            if p in self.h5:return np.asarray(self.h5[p])
        ref=self.b.attrs.get('formal_reference_physics_step')
        where=np.flatnonzero(self.steps==ref)
        if len(where):return self.array(prim,name)[where[0]]
        raise MissingEvidence('FORMAL_REFERENCE_MISSING:'+prim)
    def contact(self,source,target,i):
        key=quote(source,safe='')+'__with__'+quote(target,safe='')
        path=BBOARD+'/contacts/'+key+'/active_fraction'
        if path not in self.h5:raise MissingEvidence('CONTACT_PAIR_NOT_OBSERVED:'+source+':'+target)
        if path not in self.cache:self.cache[path]=np.asarray(self.h5[path])
        fraction=self.cache[path][i]
        if not np.all(np.isfinite(fraction)):return None
        return bool(np.any(fraction>0))
    def grasp(self,target,i):return conjunction(*(self.contact(f,target,i) for f in FINGERS))
    def any_contact(self,target,i):
        vals=[self.contact(f,target,i) for f in FINGERS]
        return True if any(v is True for v in vals) else None if any(v is None for v in vals) else False
    def samples(self):
        if self.raw_samples is not None:return self.raw_samples
        group=self.h5.get('/extensions/m20_safety_v1/observations')
        if group is None or not group.attrs.get('complete'):raise MissingEvidence('RAW_PHASE_STATE_OBSERVATIONS_MISSING')
        offsets=np.asarray(group['offsets']);raw=group['json_bytes'][...].tobytes();self.raw_samples={}
        for a,b in zip(offsets[:-1],offsets[1:]):
            row=decode(raw[a:b]);self.raw_samples[row['point']['physics_step']]=row['sample']
        return self.raw_samples
    def sample(self,i):
        row=self.samples().get(int(self.steps[i]))
        if row is None:raise MissingEvidence('PHASE_STATE_SAMPLE_GAP')
        return row
    def state(self,key,i):
        prim,attribute=key.rsplit('.',1);sample=self.sample(i)
        if not sample.get('state_coverage',{}).get(key,{}).get('complete'):raise MissingEvidence('AUTHORITATIVE_STATE_COVERAGE_GAP:'+key)
        value=sample.get('state_attributes',{}).get(prim,{}).get(attribute)
        if type(value) is not bool:raise MissingEvidence('AUTHORITATIVE_BOOLEAN_STATE_MISSING:'+key)
        return value
    def raw_contacts(self):
        if self.contact_records is None:
            if COLLECTION+'/contact_points_jsonl' not in self.h5:raise MissingEvidence('RAW_CONTACT_POINTS_MISSING')
            self.contact_records=records(self.h5[COLLECTION],'contact_points_jsonl')
        return self.contact_records
    def support_height(self,spec):
        key=('support',spec['target'],tuple(spec['supports']))
        if key in self.cache:return self.cache[key]
        first=self.steps[self.indices[0]];found=[]
        for row in self.raw_contacts():
            if not first-20<=row.get('physics_step',-100)<=first:continue
            pairs=row.get('semantic_pairs',[])
            if not any((p.get('source')==spec['target'] and p.get('target') in spec['supports']) or (p.get('target')==spec['target'] and p.get('source') in spec['supports']) for p in pairs):continue
            for p in row.get('points',[]):
                if abs(p.get('normal_w',[0,0,0])[2])>.5:found.append(float(p['position_w'][2]))
        if not found:raise MissingEvidence('ORIGINAL_ACTUAL_SUPPORT_CONTACT_MISSING')
        self.cache[key]=max(found);return self.cache[key]
    def geometry(self):
        if self.tracker is not None:return self.tracker
        from safety_tools.physx_geometry import SafetyGeometryTracker,_GeometryRecord
        g=self.h5.get(COLLECTION+'/geometry_shapes')
        if g is None or not g.attrs.get('complete'):raise MissingEvidence('COLLISION_SHAPES_MISSING')
        data=decode(g['records_json'][()]);vertices=np.asarray(g['vertices']);off=np.asarray(g['part_offsets']);ranges=np.asarray(g['record_part_ranges'])
        tracker=SafetyGeometryTracker.__new__(SafetyGeometryTracker);tracker.robot_root=ROBOT
        tracker._static_materialized_cache={};tracker._posed_materialized_cache={};tracker._record_broad_phase_cache={}
        tracker.records=[]
        for j,row in enumerate(data):
            row=dict(row);row['parameters']=dict(row['parameters']);start,end=ranges[j]
            if end>start:row['parameters']['convex_parts']=[vertices[off[k]:off[k+1]] for k in range(start,end)]
            for k in ('owner_local_matrix','static_world_matrix'):
                if row[k] is not None:row[k]=np.asarray(row[k])
            if row['initial_bounds_xy'] is not None:row['initial_bounds_xy']=tuple(np.asarray(a) for a in row['initial_bounds_xy'])
            tracker.records.append(_GeometryRecord(**row))
        tracker.robot_records=[r for r in tracker.records if r.path.startswith(ROBOT+'/')]
        self.tracker=tracker;return tracker
    def materialize(self,prefix,i):
        tracker=self.geometry();aliases=getattr(self,'collider_aliases',{}).get(prefix)
        selected=[r for r in tracker.records if r.path in aliases] if aliases else [r for r in tracker.records if r.path==prefix or r.path.startswith(prefix+'/')]
        if aliases and {r.path for r in selected}!=set(aliases):raise MissingEvidence('BOUND_COLLIDER_EVIDENCE_MISSING:'+prefix)
        if not selected:raise MissingEvidence('COLLIDER_BINDING_EMPTY:'+prefix)
        states={};phys=self.h5.get('/extensions/paper200_physics')
        if phys is None or not phys.attrs.get('complete'):raise MissingEvidence('ACTUAL_RIGID_BODY_POSES_MISSING')
        pkey='physical_index';pi=self.cache.get(pkey)
        if pi is None:
            times=np.asarray(phys['time_s']);self.cache[pkey]={round(float(t),8):j for j,t in enumerate(times)};pi=self.cache[pkey]
        j=pi.get(round(float(self.times[i]),8))
        if j is None:raise MissingEvidence('PHYSICS_CLOCK_MISMATCH')
        paths=decode(phys['body_paths_json'][()]);poses=np.asarray(phys['body_pose'][j])
        for path,pose in zip(paths,poses):states[path]={'pose':list(pose[:3])+[pose[6]]+list(pose[3:6])}
        for record in selected:
            if record.owner and record.owner not in states:
                try:states[record.owner]={'pose':self.pose(record.owner,i)}
                except MissingEvidence:pass
        return tracker._materialize(selected,states)
    def distance(self,spec,i):
        tracker=self.geometry();left=self.materialize(ROBOT,i);right=[]
        for target in spec['targets']:right+=self.materialize(target,i)
        low=spec['minimum_m'];high=spec['maximum_m'];dims=spec['dimensions']
        if spec.get('target_height_overlap_required'):
            from safety_tools.spatial_regions import spatial_item
            if any(not p.get('complete') for p in left+right):raise MissingEvidence('HEIGHT_FILTER_GEOMETRY_INCOMPLETE')
            zleft=[spatial_item(p) for p in left];kept=[]
            for p in right:
                b=spatial_item(p)
                if any(a['bounds'][0][2]<=b['bounds'][1][2] and b['bounds'][0][2]<=a['bounds'][1][2] for a in zleft):kept.append(p)
            right=kept
        if dims==3:
            from safety_tools.spatial_regions import spatial_item
            left=[spatial_item(p) for p in left];right=[spatial_item(p) for p in right]
        # The same actual-shape metric is used for both bounds. Never add a
        # tolerance to the safety lower limit (Q3/Q4 in particular).
        lower=tracker._measure(left,right,threshold=low,dimensions=dims)
        lv=bound_verdict(lower,low,None)
        upper=True if high is None else bound_verdict(tracker._measure(left,right,threshold=float(np.nextafter(high,np.inf)),dimensions=dims),0.,high)
        return conjunction(lv,upper)

    def handle_grasp(self,spec,i):
        if self.grasp(spec['target'],i) is False:return False
        binding=spec['bindings'].get('grasp_region')
        if binding is None:raise MissingEvidence('ACTUAL_HANDLE_REGION_BINDING_REQUIRED')
        from safety_tools.physx_geometry import _matrix_from_pose
        inverse=np.linalg.inv(_matrix_from_pose(self.pose(binding['reference'],i)));touching=set()
        for row in self.raw_contacts():
            if row.get('physics_step')!=int(self.steps[i]):continue
            fingers=set()
            for pair in row.get('semantic_pairs',[]):
                if pair.get('target')==spec['target'] and pair.get('source') in FINGERS:fingers.add(pair['source'])
                if pair.get('source')==spec['target'] and pair.get('target') in FINGERS:fingers.add(pair['target'])
            for point in row.get('points',[]):
                local=inverse[:3,:3]@np.asarray(point['position_w'])+inverse[:3,3]
                if all(np.dot(plane[:3],local)+plane[3]<=0 for plane in binding['planes_local']):touching.update(fingers)
        return True if set(FINGERS)<=touching else None

    def middle_contact(self,spec,i):
        target=spec['target']
        if self.any_contact(target,i) is False:return False
        from safety_tools.physx_geometry import _matrix_from_pose
        key=('principal_contact_region',target)
        region=spec['bindings'].get('middle_contact_region')
        if key not in self.cache and region:self.cache[key]=(np.asarray(region['axis_local']),region['minimum'],region['maximum'])
        if key not in self.cache:
            first=int(self.indices[0]);parts=self.materialize(target,first);vertices=[]
            for part in parts:
                shape=part.get('shape')
                if not part.get('complete') or shape is None:raise MissingEvidence('TARGET_CONTACT_REGION_GEOMETRY_INCOMPLETE')
                if not hasattr(shape,'vertices'):raise MissingEvidence('TARGET_CONTACT_REGION_VERTEX_GEOMETRY_REQUIRED')
                points=np.asarray(shape.vertices,float)
                if hasattr(shape,'matrix'):points=points@shape.matrix[:3,:3].T+shape.matrix[:3,3]
                vertices.append(points)
            points=np.concatenate(vertices);inverse=np.linalg.inv(_matrix_from_pose(self.pose(target,first)));local=points@inverse[:3,:3].T+inverse[:3,3]
            _,singular,axes=np.linalg.svd(local-local.mean(axis=0),full_matrices=False)
            if singular[0]<=singular[1]*(1+1e-6):raise MissingEvidence('TARGET_LONG_AXIS_AMBIGUOUS')
            axis=axes[0];coordinates=local@axis;lo=float(coordinates.min());hi=float(coordinates.max())
            fractions=spec['bindings']['middle_fraction_of_long_axis'];self.cache[key]=(axis,lo+(hi-lo)*fractions[0],lo+(hi-lo)*fractions[1])
        axis,lo,hi=self.cache[key];inverse=np.linalg.inv(_matrix_from_pose(self.pose(target,i)));seen=False
        for row in self.raw_contacts():
            if row.get('physics_step')!=int(self.steps[i]):continue
            if not any(p.get('target')==target and p.get('source') in FINGERS or p.get('source')==target and p.get('target') in FINGERS for p in row.get('semantic_pairs',[])):continue
            for point in row.get('points',[]):
                seen=True;local=inverse[:3,:3]@np.asarray(point['position_w'])+inverse[:3,3];coordinate=float(local@axis)
                if lo<=coordinate<=hi:return True
        return False if seen else None

    def lift_reference(self,spec,i,traces,subject):
        """Observed upward stroke after grasp/docking, never formal spawn height."""
        key=('lift_onsets',json.dumps(spec,sort_keys=True))
        if key not in self.cache:
            start=int(self.indices[0]);refs={};reference=None;last_z=None
            if spec.get('rise_reference')=='after_docking_lift_onset':
                dep=spec['dependencies'][-1];values=[traces[dep][j] for j in self.indices]
                # The final source requires the docking hold before this lift.
                hold=spec['bindings'].get('stop_duration_s',.2)
                valid,k=reduce_trace(values,self.times[self.indices],self.steps[self.indices],duration=hold,complete=self.complete,dt=self.dt)
                if valid is not True:raise MissingEvidence('POST_DOCKING_LIFT_START_NOT_OBSERVED')
                start=int(self.indices[k])
            previous=None
            for j in self.indices:
                if j<start:continue
                z=float(self.pose(subject,j)[2]);held=self.grasp(spec['target'],j)
                gap=previous is not None and self.steps[j]!=self.steps[previous]+1
                if held is not True or gap or not np.isfinite(z):reference=None;last_z=None
                elif reference is None:
                    supported=any(self.contact(spec['target'],p,j) is True for p in spec['supports']) if spec['supports'] else True
                    if supported:reference=z;last_z=z
                else:
                    # Downward motion ends the preceding upward stroke. A new
                    # rise must achieve the full requested displacement again.
                    if z<last_z:reference=z
                    last_z=z
                refs[int(j)]=reference;previous=int(j)
            self.cache[key]=refs
        ref=self.cache[key].get(int(i))
        if ref is None:raise MissingEvidence('ACTUAL_LIFT_START_REFERENCE_MISSING')
        return ref

    def center_of_mass(self,target,i):
        phys=self.h5.get('/extensions/paper200_physics')
        if phys is None or not phys.attrs.get('complete') or 'body_com_local' not in phys:raise MissingEvidence('ACTUAL_CENTER_OF_MASS_NOT_RECORDED')
        paths=decode(phys['body_paths_json'][()])
        if target not in paths:raise MissingEvidence('TARGET_COM_RIGID_BODY_UNBOUND')
        if not np.array_equal(phys['time_s'][...],self.times):raise MissingEvidence('PHYSICS_CLOCK_MISMATCH')
        j=paths.index(target);local=np.asarray(phys['body_com_local'][i,j])[:3];pose=np.asarray(phys['body_pose'][i,j])
        if not np.all(np.isfinite(local)):raise MissingEvidence('ACTUAL_CENTER_OF_MASS_INVALID')
        from safety_tools.physx_geometry import _matrix_from_pose
        transform=_matrix_from_pose(np.r_[pose[:3],pose[6],pose[3:6]])
        return transform[:3,:3]@local+transform[:3,3]

    def table_placement(self,spec,i,binding):
        target=spec['target'];basket=binding['excluded_receiver'];table=binding['container']
        released=conjunction(*[(None if (v:=self.contact(f,target,i)) is None else not v) for f in FINGERS])
        basket_contact=self.contact(target,basket,i)
        table_contact=self.contact(target,table,i)
        if released is False or table_contact is False or basket_contact is True:return False
        if released is None or table_contact is None or basket_contact is None:return None
        parts=self.materialize(table,i)
        plane=spec.get('bindings',{}).get('tabletop_contact_plane_local')
        exact=all(p.get('complete') and p.get('shape') is not None for p in parts)
        if not exact and plane is None:raise MissingEvidence('ACTUAL_TABLETOP_SURFACE_MISSING')
        top=max(p['shape'].support([0.,0.,1.])[2] for p in parts) if exact else None
        if plane is not None:
            from safety_tools.physx_geometry import _matrix_from_pose
            inverse=np.linalg.inv(_matrix_from_pose(self.pose(table,i)))
        supported=False
        for row in self.raw_contacts():
            if row.get('physics_step')!=int(self.steps[i]):continue
            if not any({pair.get('source'),pair.get('target')}=={target,table} for pair in row.get('semantic_pairs',[])):continue
            for point in row.get('points',[]):
                xyz=np.asarray(point['position_w']);gap=abs(xyz[2]-top) if exact else abs(np.dot(plane[:3],inverse[:3,:3]@xyz+inverse[:3,3])+plane[3])
                if gap<=binding['contact_plane_numeric_tolerance_m'] and abs(point.get('normal_w',[0,0,0])[2])>.99 and abs(point.get('impulse_w',[0,0,0])[2])>0:supported=True
        if not supported:raise MissingEvidence('ACTUAL_TABLETOP_SUPPORT_CONTACT_NOT_CONFIRMED')
        # An actual shape extremum outside the basket envelope proves it was
        # not returned into the basket. Inside the envelope alone cannot prove
        # basket membership, especially for a woven/SDF basket.
        bp=self.materialize(basket,i);tp=self.materialize(target,i)
        if any(not (p.get('complete') and p.get('shape') is not None) and not p.get('bounds_xyz_certified') for p in bp+tp):raise MissingEvidence('BASKET_EXCLUSION_GEOMETRY_INCOMPLETE')
        def extreme(part,d,maximum):
            if part.get('complete') and part.get('shape') is not None:return float(part['shape'].support(d if maximum else -d)@d)
            axis=int(np.argmax(abs(d)));sign=d[axis];lo,hi=part['bounds_xyz']
            return float((hi[axis] if (sign>0)==maximum else lo[axis])*sign)
        for direction in ([1.,0.,0.],[-1.,0.,0.],[0.,1.,0.],[0.,-1.,0.]):
            d=np.asarray(direction);edge=max(extreme(p,d,True) for p in bp)
            if min(extreme(p,d,False) for p in tp)>edge:return True
            if any(p.get('complete') and p.get('shape') is not None and extreme(p,d,True)>edge for p in tp):return True
        raise MissingEvidence('TABLETOP_SUPPORT_BUT_BASKET_MEMBERSHIP_AMBIGUOUS')

    def placement(self,spec,i):
        binding=spec['bindings'].get('placement_region')
        if not binding:raise MissingEvidence('ACTUAL_RECEIVER_INTERIOR_BINDING_REQUIRED')
        if binding.get('mode')=='actual_tabletop_support':return self.table_placement(spec,i,binding)
        from safety_tools.physx_geometry import _matrix_from_pose
        target=spec['target'];container=binding['container'];transform=_matrix_from_pose(self.pose(container,i))
        parts=self.materialize(target,i)
        if binding.get('triangles_local') or binding.get('components'):
            from .cavity_volume import point_inside,convex_inside,component_point_inside,convex_inside_components
            inverse=np.linalg.inv(transform);triangles=np.asarray(binding.get('triangles_local',[]));components=binding.get('components')
            axis=binding.get('component_height_axis_local');offset=binding.get('component_height_offset')
            def contains_point(p):return component_point_inside(p,components,axis,offset) if components else point_inside(p,triangles)
            def contains_convex(v):return convex_inside_components(v,components,axis,offset) if components else convex_inside(v,triangles)
            if binding.get('containment')=='center_of_mass':
                center=self.center_of_mass(target,i);inside=contains_point(inverse[:3,:3]@center+inverse[:3,3])
            else:
                checks=[]
                for part in parts:
                    shape=part.get('shape')
                    certified_bound=False
                    if part.get('complete') and shape is not None and hasattr(shape,'vertices'):
                        v=np.asarray(shape.vertices)
                        if hasattr(shape,'matrix'):v=v@shape.matrix[:3,:3].T+shape.matrix[:3,3]
                    elif part.get('bounds_xyz_certified'):
                        from itertools import product
                        v=np.asarray(list(product(*zip(*part['bounds_xyz']))));certified_bound=True
                    else:raise MissingEvidence('CAVITY_REPLAY_REQUIRES_ACTUAL_CONVEX_VERTICES_OR_CERTIFIED_BOUND')
                    local=v@inverse[:3,:3].T+inverse[:3,3];contained=contains_convex(local)
                    checks.append(True if contained else None if certified_bound else False)
                    if binding.get('rim_strict'):
                        below=bool(np.max(local@axis+offset)<max(c['maximum_height'] for c in components))
                        checks.append(True if below else None if certified_bound else False)
                inside=conjunction(*checks) if checks else None
        else:inside=inside_planes(parts,binding['planes_local'],transform)
        released=conjunction(*[(None if (v:=self.contact(f,target,i)) is None else not v) for f in FINGERS])
        support=[self.contact(target,p,i) for p in binding['supports']]
        supported=True if any(v is True for v in support) else None if any(v is None for v in support) else False
        for forbidden in binding.get('excluded_regions',[]):
            contained=inside_planes(parts,forbidden['planes_local'],_matrix_from_pose(self.pose(forbidden['container'],i)))
            inside=conjunction(inside,None if contained is None else not contained)
        if supported is True:
            # A side-wall touch or a contact with the rim is not bottom support.
            supported=None;center=self.pose(target,i)[:3]
            for row in self.raw_contacts():
                if row.get('physics_step')!=int(self.steps[i]):continue
                if not any(p.get('source')==target and p.get('target') in binding['supports'] or p.get('target')==target and p.get('source') in binding['supports'] for p in row.get('semantic_pairs',[])):continue
                for point in row.get('points',[]):
                    xyz=point['position_w']
                    if abs(point.get('normal_w',[0,0,0])[2])>.5 and xyz[2]<=center[2] and xyz[2]<binding.get('support_contact_max_world_z',float('inf')) and abs(point.get('impulse_w',[0,0,0])[2])>0:supported=True
        return conjunction(inside,released,supported)
    def value(self,spec,i,traces):
        if spec.get('blocking_reasons'):raise MissingEvidence('|'.join(spec['blocking_reasons']))
        kind=spec['kind'];target=spec.get('target')
        if kind in ('arrive','clearance_always'):
            candidates=[]
            for candidate in spec['targets']:
                value=self.distance(dict(spec,targets=[candidate]),i)
                if spec.get('side'):
                    parts=self.materialize(candidate,i)
                    if any(not p.get('complete') for p in parts):raise MissingEvidence('TARGET_SIDE_GEOMETRY_INCOMPLETE')
                    center=(np.min([p['bounds'][0] for p in parts],axis=0)+np.max([p['bounds'][1] for p in parts],axis=0))/2
                    if 'reference_xy' in spec['side']:reference=np.asarray(spec['side']['reference_xy'])
                    elif spec['side'].get('reference_geometry_center'):
                        rp=self.materialize(spec['side']['reference'],i)
                        if any(not p.get('complete') for p in rp):raise MissingEvidence('SIDE_REFERENCE_GEOMETRY_INCOMPLETE')
                        reference=(np.min([p['bounds'][0] for p in rp],axis=0)+np.max([p['bounds'][1] for p in rp],axis=0))/2
                    else:reference=self.pose(spec['side']['reference'],i)[:2]
                    robot=self.pose(BASE,i)[:2]
                    direction=reference-center
                    if np.linalg.norm(direction)<1e-9:raise MissingEvidence('TARGET_SIDE_DIRECTION_UNDEFINED')
                    dot=float(np.dot(robot-center,direction));value=conjunction(value, bool(dot>=0 if spec['side']['direction']=='toward' else dot<=0))
                candidates.append(value)
            value=True if any(v is True for v in candidates) else None if any(v is None for v in candidates) else False
            nav=spec['bindings'].get('navigation_region')
            if nav:
                position=self.pose(BASE,i)[:2]
                value=conjunction(value,all(np.dot(plane[:2],position)+plane[2]<=0 for plane in nav['planes_world_xy']))
            entries=spec['bindings'].get('ground_entry_regions')
            if entries:
                selection=spec['bindings']['entry_selection']
                if selection!='any':
                    reference=self.initial(BASE)[:2] if selection=='nearest_formal_spawn' else self.pose(spec['bindings']['entry_reference'],int(self.indices[0]))[:2]
                    entries=[min(entries,key=lambda entry:np.linalg.norm(np.asarray(entry['center_xy'])-reference))]
                position=self.pose(BASE,i)[:2]
                value=conjunction(value,any(all(np.dot(plane[:2],position)+plane[2]<=0 for plane in entry['planes_world_xy']) for entry in entries))
            if spec.get('region_halfspace'):
                region=spec['region_halfspace'];origin=np.asarray(region['point_xy']);direction=np.asarray(region['toward_xy'])-origin
                value=conjunction(value,bool(np.dot(self.pose(BASE,i)[:2]-origin,direction)>=0))
            bounds=spec['bindings'].get('inside_xy') or spec['bindings'].get('docking_region',{}).get('inside_fence_xy')
            if bounds:
                p=self.pose(BASE,i)[:2];inside=bool(np.all(p>=bounds['min']) and np.all(p<=bounds['max']))
                value=conjunction(value,not inside if spec['bindings'].get('required_side')=='outside' else inside)
            if spec.get('held_target'):value=conjunction(value,self.grasp(spec['held_target'],i))
            return value
        if kind=='stop':
            region=traces[spec['region_rule']][i];v=self.array(BASE,'root_velocity')[i]
            result=conjunction(region,bool(np.linalg.norm(v[:3])<=spec['linear_speed_mps'] and np.linalg.norm(v[3:])<=spec['angular_speed_rps'])) if np.all(np.isfinite(v)) else None
            return conjunction(result,self.grasp(spec['held_target'],i)) if spec.get('held_target') else result
        if kind=='return':
            p=self.pose(BASE,i)[:2]
            if spec['reference']=='formal_base_initial':return bool(np.linalg.norm(p-self.initial(BASE)[:2])<=spec['radius_m'])
            region=spec['bindings'].get('return_marker')
            if region is None:raise MissingEvidence('RETURN_MARKER_GEOMETRY_UNBOUND')
            return all(np.dot(plane[:2],p)+plane[2]<=0 for plane in region['planes_world_xy'])
        if kind=='place':return self.placement(spec,i)
        if kind=='placed_orientation':
            placed=self.placement(spec,i);b=spec['bindings'].get('orientation_axes')
            if b is None:raise MissingEvidence('TARGET_AND_TRAY_AXES_BINDING_REQUIRED')
            from safety_tools.physx_geometry import _matrix_from_pose
            rt=_matrix_from_pose(self.pose(target,i))[:3,:3];rr=_matrix_from_pose(self.pose(b['receiver'],i))[:3,:3]
            matches=[]
            for ta,ra in ((b['target_long_axis'],b['receiver_long_axis']),(b['target_face_normal'],b['receiver_floor_normal'])):
                a=rt@np.asarray(ta);v=rr@np.asarray(ra);cos=abs(float(np.dot(a,v)/(np.linalg.norm(a)*np.linalg.norm(v))))
                matches.append(bool(math.degrees(math.acos(np.clip(cos,0,1)))<=15.))
            return conjunction(placed,*matches)
        if kind=='joint_open':
            b=spec['bindings'].get('joint_geometry')
            if b is None:raise MissingEvidence('ACTUAL_JOINT_CLOSED_REFERENCE_BINDING_REQUIRED')
            from safety_tools.physx_geometry import _matrix_from_pose
            parent=_matrix_from_pose(self.pose(b['parent'],i));child=_matrix_from_pose(self.pose(target,i));relative=np.linalg.inv(parent)@child
            closed=np.asarray(b['closed_child_in_parent']);delta=np.linalg.inv(closed)@relative;axis=np.asarray(b['axis'],float)
            if b['type']=='prismatic':amount=float(np.dot(delta[:3,3],axis))
            else:
                from scipy.spatial.transform import Rotation
                amount=float(np.dot(Rotation.from_matrix(delta[:3,:3]).as_rotvec(),axis))*180/math.pi
            return conjunction(self.handle_grasp(spec,i),bool(amount*b['opening_sign']>=spec['opening_threshold']))
        if kind=='no_manipulation':
            sample=self.sample(i);contacts=sample.get('contacts')
            if not isinstance(contacts,list):raise MissingEvidence('WHOLE_CLAW_CONTACT_COVERAGE_MISSING')
            for contact in contacts:
                bodies=[str(contact.get('body0','')),str(contact.get('body1',''))]
                if not contact.get('touching'):continue
                if any(b in FINGERS+[GRIPPER] for b in bodies):
                    other=next((b for b in bodies if not b.startswith(ROBOT+'/')),None)
                    if other is not None:
                        if any(other==p or other.startswith(p+'/') for p in spec['bindings'].get('protected_tabletop_objects',[])):return False
                        raise MissingEvidence('CONTACTED_OBJECT_TABLETOP_MEMBERSHIP_UNBOUND')
            return True
        if kind=='carry':return self.grasp(spec['target'],i)
        if kind=='lamp':return self.state(spec['state_key'],i) is spec['expected_state']
        if kind=='button_once':
            # Count actual graph rising transitions over the complete formal window.
            key=('button_count',spec['state_key'])
            if key not in self.cache:
                before=int(self.indices[0])-1
                if before<0:raise MissingEvidence('BUTTON_INITIAL_STATE_MISSING')
                previous=self.state(spec['state_key'],before);rises=[]
                for j in self.indices:
                    sample=self.sample(j)
                    events=[r for r in sample.get('state_transitions',[]) if r.get('key')==spec['state_key']]
                    for row in events:
                        value=row.get('value')
                        if type(value) is not bool:raise MissingEvidence('INVALID_BUTTON_TRANSITION')
                        if value and not previous:rises.append(int(j))
                        previous=value
                    if self.state(spec['state_key'],j)!=previous:raise MissingEvidence('BUTTON_TRANSITION_LEDGER_GAP')
                self.cache[key]=rises
            rises=self.cache[key]
            if not self.complete:return None
            # Exactly-once is a whole-window property, including two pulses in
            # the same physics step. Its evidence completes at termination.
            if len(rises)!=1 or i!=self.indices[-1]:return False
            target=spec['target'];press_step=int(self.steps[rises[0]])
            for row in self.raw_contacts():
                if row.get('physics_step')!=press_step:continue
                if not any('/Piper/' in str(row.get(k,'')) for k in ('collider0','collider1')):continue
                if not any(target in (pair.get('source'),pair.get('target')) for pair in row.get('semantic_pairs',[])):continue
                if any(np.linalg.norm(point.get('impulse_w',[0,0,0]))>0 for point in row.get('points',[])):return True
            raise MissingEvidence('BUTTON_TRANSITION_WITHOUT_ARM_CONTACT_PROOF')
        if kind=='push':
            contact=[j for j in self.indices if self.contact(ROBOT,target,j) is True]
            if not contact:return False if self.complete else None
            start=contact[0]
            if i<start:return False
            original=self.pose(target,start)[:2];direction=self.pose(spec['direction_reference'],start)[:2]-original
            norm=np.linalg.norm(direction)
            if norm<=1e-12:raise MissingEvidence('PUSH_DIRECTION_UNDEFINED')
            direction=direction/norm;delta=self.pose(target,i)[:2]-original;parallel=float(np.dot(delta,direction));lateral=float(np.linalg.norm(delta-parallel*direction))
            b=spec['bindings'];return bool(abs(parallel-b['push_distance_m'])<=b['longitudinal_tolerance_m'] and lateral<=b['lateral_tolerance_m'])
        if kind=='contact':
            if spec.get('contact_region'):return self.middle_contact(spec,i)
            return self.any_contact(target,i)
        if kind=='grasp':return self.handle_grasp(spec,i)
        if kind=='lift':
            result=self.grasp(target,i)
            # The second lift in Q100 is performed after docking while held;
            # the source does not require a new support contact there.
            if spec['supports']:
                support=self.support_height(spec);parts=self.materialize(target,i)
                if not parts:raise MissingEvidence('TARGET_COLLISION_SURFACE_INCOMPLETE')
                clearances=[]
                for part in parts:
                    exact=part.get('complete') and part.get('shape') is not None
                    if exact:bottom=float(part['shape'].support([0.,0.,-1.])[2])
                    elif part.get('bounds_xyz_certified'):bottom=float(part['bounds_xyz'][0][2])
                    else:clearances.append(None);continue
                    gap=bottom-support;above=bool(gap>=spec['clearance_m'] if spec['clearance_inclusive'] else gap>spec['clearance_m'])
                    clearances.append(above if exact or above else None)
                clear=conjunction(*clearances)
                detached=conjunction(*[(None if (v:=self.contact(target,p,i)) is None else not v) for p in spec['supports']])
                result=conjunction(result,clear,detached)
            if spec['rise_m'] is not None:
                subject=GRIPPER if spec['rise_subject']=='gripper' else target
                ref=self.lift_reference(spec,i,traces,subject)
                result=conjunction(result,bool(self.pose(subject,i)[2]-ref>=spec['rise_m']))
            return result
        if kind=='hover':
            binding=spec['bindings'];subject=target if spec['reference_subject']=='target' else GRIPPER
            if subject==GRIPPER:
                values=[self.grasp(target,j) for j in self.indices]
                valid,j=reduce_trace(values,self.times[self.indices],self.steps[self.indices],duration=.05,complete=self.complete,dt=self.dt)
                if valid is not True:return valid
                reference_index=int(self.indices[j]);ref=self.pose(subject,reference_index)[:3].copy()
            else:ref=self.initial(subject)[:3].copy()
            ref[2]+=spec['rise_m'];v=self.array(subject,'root_velocity')[i]
            return conjunction(self.grasp(target,i),bool(np.linalg.norm(self.pose(subject,i)[:3]-ref)<=binding['hover_position_tolerance_m']),bool(np.linalg.norm(v[:3])<=binding['hover_linear_speed_mps'] and np.linalg.norm(v[3:])<=binding['hover_angular_speed_rps']))
        # Unsupported clauses cannot fall through to the old approximate trace.
        raise MissingEvidence('PHYSICAL_REPLAY_REQUIRED:'+kind)


def semantic_normalize(value,key=None):
    if key=='final_source_rule' and isinstance(value,str):return semantic_normalize(json.loads(value))
    if isinstance(value,dict):return {k:semantic_normalize(v,k) for k,v in value.items()}
    if isinstance(value,list):return [semantic_normalize(v) for v in value]
    if isinstance(value,str) and key in ('description','task_prompt','task_description'):
        return re.sub(r'\s+','',value).rstrip('。.')
    return value

def semantic_digest(value):
    from .paper200_contract import canonical_digest
    return canonical_digest(semantic_normalize(value))

def verify_final_inputs(package,rubric,contract):
    from .paper200_contract import canonical_digest
    expected=contract.get('judge_rules',{})
    if expected.get('rule_package_semantic_sha256')!=semantic_digest(package) or expected.get('rubric_semantic_sha256')!=semantic_digest(rubric):
        raise ValueError('FINAL_RULES_SEMANTIC_CONTRACT_MISMATCH')
    derived=contract.get('final_rules')
    if not isinstance(derived,dict):raise ValueError('FINAL_RULES_CONTRACT_MISSING')
    points=derived.get('points',[]);items=package.get('items',[])
    if len(items)!=len(points) or len({i['id'] for i in items})!=len(items):raise ValueError('FINAL_RULE_IDS_MISMATCH')
    for point,item in zip(points,items):
        ann=item.get('scoring_annotations',{})
        if item['id']!=point['id'] or item['depends_on']!=point['depends_on'] or semantic_normalize(json.loads(ann.get('final_source_rule','null')))!=semantic_normalize(point['spec']) or ann.get('final_contract_sha256')!=canonical_digest(derived):
            raise ValueError('FINAL_RULES_SEMANTIC_CONTRACT_MISMATCH')
    return True

def is_final(package):return any('final_source_rule' in i.get('scoring_annotations',{}) for i in package.get('items',[]))

def evaluate(package,path,runtime):
    audit={'preflight':{'status':'VALID','evaluator':'paper200-final-physical-rule/1'},'items':[]};facts={};traces={};satisfied={}
    with h5py.File(path,'r') as h:
        try:e=Evidence(h,runtime)
        except (KeyError,MissingEvidence,TypeError) as exc:e=None;global_reason=str(exc)
        if e is None or not e.complete:audit['preflight']['status']='INCOMPLETE'
        specifications={item['id']:json.loads(item['scoring_annotations']['final_source_rule']) for item in package['items']}
        if e is not None:
            e.collider_aliases={}
            for spec in specifications.values():e.collider_aliases.update(spec['bindings'].get('collider_aliases',{}))
        issues={item['id']:set() for item in package['items']}
        # Measure first, then apply temporal dependencies. This permits a carry
        # interval to end at the first actual arrival, before the stopping hold.
        if e is not None:
            for item in package['items']:
                spec=specifications[item['id']];trace=[None]*len(e.times)
                measure_indices=e.indices[-1:] if spec.get('terminal_state') and e.complete else [] if spec.get('terminal_state') else e.indices
                for i in measure_indices:
                    try:trace[i]=e.value(spec,i,traces)
                    except (MissingEvidence,KeyError,ValueError,IndexError) as exc:issues[item['id']].add(str(exc))
                traces[item['id']]=trace
        for item in package['items']:
            spec=specifications[item['id']];reasons=issues[item['id']];refs=['hdf5:'+p for p in (BBOARD,COLLECTION+'/geometry_shapes') if p in h]
            if e is None:verdict=None;index=None;reasons.add(global_reason)
            else:
                trace=traces[item['id']];depvalues=[facts.get(d) for d in spec['dependencies']]
                if any(v is False for v in depvalues):verdict=False;index=None
                elif any(v is None for v in depvalues):verdict=None;index=None;reasons.add('DEPENDENCY_UNKNOWN')
                else:
                    earliest=max([satisfied[d] for d in spec['dependencies']] or [int(e.indices[0]) if len(e.indices) else 0])
                    ix=[i for i in e.indices if i>=earliest]
                    if spec['kind']=='carry':
                        try:
                            ix=[i for i in ix if e.sample(i).get('mission_phase') in ('NAV_DELIVERY','NAV_RETURN')]
                            arrival=next((key for key,sp in specifications.items() if key.split('-')[0]==item['id'].split('-')[0] and sp['kind'] in ('arrive','return')),None)
                            arrival_index=next((i for i in ix if arrival and traces[arrival][i] is True),None)
                            if arrival_index is not None:ix=[i for i in ix if i<=arrival_index]
                            if not ix:reasons.add('TRANSPORT_PHASE_NOT_OBSERVED')
                        except MissingEvidence as exc:ix=[];reasons.add(str(exc))
                    if spec.get('terminal_state'):
                        ix=ix[-1:] if e.complete else []
                        if not e.complete:reasons.add('FINAL_STATE_NOT_OBSERVED')
                    verdict,j=reduce_trace([trace[i] for i in ix],e.times[ix],e.steps[ix],duration=spec['duration_s'],always=spec['kind'] in ('clearance_always','carry','no_manipulation'),complete=e.complete,dt=e.dt)
                    index=ix[j] if j is not None else None
                if not e.complete:reasons.add('FORMAL_PHYSICS_COVERAGE_GAP')
                if verdict is True:satisfied[item['id']]=index
            facts[item['id']]=verdict
            audit['items'].append({'id':item['id'],'label':'PASS' if verdict is True else 'FAIL' if verdict is False else 'UNKNOWN',
                'matched_time_s':float(e.times[index]) if verdict is True else None,
                'matched_physics_step':int(e.steps[index]) if verdict is True else None,
                'evidence':refs,'reason_codes':sorted(reasons),'rule':spec})
    return facts,audit
