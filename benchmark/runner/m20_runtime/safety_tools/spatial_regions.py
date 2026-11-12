"""Height-aware forbidden volumes without merging disconnected colliders."""
import numpy as np

class ExtrudedShape:
    """Convex hull of the original shape and its XY projection at a Z plane."""
    def __init__(self,shape,z):self.shape=shape;self.z=float(z)
    def support(self,direction):
        d=np.asarray(direction,float);original=np.asarray(self.shape.support(d),float)
        xy=d.copy();xy[2]=0.;plane=np.asarray(self.shape.support(xy),float).copy();plane[2]=self.z
        return plane if plane@d>original@d else original

def spatial_item(item,shape=None):
    shape=item['shape'] if shape is None else shape
    if not item.get('complete') or shape is None:
        lower,upper=item['bounds_xyz'] if item.get('bounds_xyz_certified') else item.get('bounds',(np.full(2,-np.inf),np.full(2,np.inf)))
        # Retain certified XY bounds; an unknown height cannot erase a valid
        # horizontal safe-distance certificate. Near pairs remain unknown.
        lower=np.r_[lower,-np.inf] if len(lower)==2 else lower
        upper=np.r_[upper,np.inf] if len(upper)==2 else upper
        return dict(item,bounds=(lower,upper))
    axes=np.eye(3)
    return dict(item,shape=shape,bounds=(np.array([shape.support(-v)[i] for i,v in enumerate(axes)]),
                                      np.array([shape.support(v)[i] for i,v in enumerate(axes)])))

def _spatial_parts(tracker,record,states,sample_cache):
    # Static records belong to this frozen tracker; moving records are reused
    # only within one sample. Both S03 and S08 consume the same exact bounds.
    cache=tracker.__dict__.setdefault('_height_spatial_static_cache',{}) if hasattr(record,'owner_local_matrix') and record.owner_local_matrix is None else sample_cache
    key=id(record)
    if key not in cache:
        items=[spatial_item(x) for x in tracker._materialize([record],states)]
        cache[key]=(items,np.asarray([x['bounds'][0][2] for x in items]),
                    np.asarray([x['bounds'][1][2] for x in items]),
                    np.asarray([bool(x['complete'] or x.get('bounds_xyz_certified')) for x in items],dtype=bool))
    return cache[key]

def measure_forbidden(tracker,subjects,forbidden,states,spec):
    left=[spatial_item(x) for x in tracker._materialize(subjects,states)]
    sample_cache={}
    right=[x for record in forbidden for x in _spatial_parts(tracker,record,states,sample_cache)[0]]
    if not left or not right:return {'complete':False,'distance_m':None,'reason':'forbidden_volume_binding_empty'}
    if spec['region_geometry']=='underbody_volumes_xyz':
        z=float(spec['region_floor_z_m'])
    else:
        # Exact finite representation of upward-extruded prisms for this query.
        # The ceiling is above every subject; it cannot omit an intersection.
        if any((not x['complete'] and not x.get('bounds_xyz_certified')) or not np.isfinite(x['bounds'][1][2]) for x in left):
            return {'complete':False,'distance_m':None,'reason':'subject_volume_unavailable'}
        z=max(x['bounds'][1][2] for x in left)+1.0
    extruded=[]
    for x in right:
        # Vertical extrusion preserves XY extrema and adds only the Z plane.
        # Narrow-phase support still uses the exact original shape.
        lower,upper=(np.array(v,copy=True) for v in x['bounds'])
        lower[2]=min(lower[2],z);upper[2]=max(upper[2],z)
        shape=ExtrudedShape(x['shape'],z) if x['complete'] and x['shape'] is not None else x['shape']
        extruded.append(dict(x,shape=shape,bounds=(lower,upper)))
    right=extruded
    result=tracker._measure(left,right,threshold=float(spec['minimum_clearance_m']),dimensions=3)
    return dict(result,region_geometry=spec['region_geometry'],height_aware=True,disconnected_parts_preserved=True)

def measure_height_filtered(tracker,subjects,obstacles,states,threshold):
    # Only broad-phase-near records are materialized. Height is tested per
    # collision part, not on the enclosing stair/platform mesh union.
    if not subjects or not obstacles:return {'complete':False,'distance_m':None,'reason':'height_filtered_scope_empty'}
    results=[];excluded=0;pruned=[]
    sample_cache={}
    arrays = None
    if hasattr(tracker, '_right_bounds_arrays'):
        arrays, missing = tracker._right_bounds_arrays(obstacles, states)
        missing = set(missing)
        missing_indices = np.asarray([i for i, record in enumerate(obstacles) if record.path in missing], dtype=np.int64)
    for subject in subjects:
        bound=tracker._record_bounds_xy(subject,states)
        near=[]
        if arrays is not None:
            # The same per-collider lower bounds as the scalar loop, with
            # static geometry cached and dynamic poses refreshed this sample.
            # Missing bounds always stay in the narrow phase.
            gaps = np.zeros(len(obstacles)) if bound is None else np.linalg.norm(
                np.maximum(0., np.maximum(bound[0]-arrays['hi'], arrays['lo']-bound[1])), axis=1)
            gaps[missing_indices] = 0.
            mask = gaps < threshold
            near = [obstacles[i] for i in np.flatnonzero(mask)]
            if np.any(~mask):pruned.append(float(np.min(gaps[~mask])))
        else:
            for obstacle in obstacles:
                other=tracker._record_bounds_xy(obstacle,states)
                gap=0. if bound is None or other is None else float(np.linalg.norm(np.maximum(0.,np.maximum(bound[0]-other[1],other[0]-bound[1]))))
                if gap<threshold:near.append(obstacle)
                else:pruned.append(gap)
        if not near:continue
        left=[spatial_item(x) for x in tracker._materialize([subject],states)]
        right=[_spatial_parts(tracker,record,states,sample_cache) for record in near]
        if not left or not any(len(items) for items,_,_,_ in right):return {'complete':False,'distance_m':None,'reason':'height_filtered_materialization_empty'}
        for a in left:
            kept=[]
            for items,lower,upper,certified in right:
                separated=certified & ((a['bounds'][1][2]<lower-1e-8)|(upper<a['bounds'][0][2]-1e-8)) if (a['complete'] or a.get('bounds_xyz_certified')) else np.zeros(len(items),dtype=bool)
                excluded+=int(np.count_nonzero(separated))
                for index in np.flatnonzero(~separated):
                    b=items[index]
                    kept.append(dict(b,bounds=(b['bounds'][0][:2],b['bounds'][1][:2])))
            if kept:
                row=tracker._measure([dict(a,bounds=(a['bounds'][0][:2],a['bounds'][1][:2]))],kept,threshold=threshold)
                if row.get('complete') and row.get('distance_m',threshold)<threshold and 'lower_bound' not in row.get('distance_kind',''):
                    return dict(row,height_overlap_required=True,height_excluded_pairs=excluded)
                results.append(row)
    if any(not x.get('complete') for x in results):return {'complete':False,'distance_m':None,'reason':'height_filtered_geometry_incomplete'}
    distance=min([threshold]+pruned+[x['distance_m'] for x in results])
    return {'complete':True,'distance_m':distance,'distance_kind':'height_filtered_collision_lower_bound',
            'height_overlap_required':True,'height_excluded_pairs':excluded,'threshold_m':threshold}
