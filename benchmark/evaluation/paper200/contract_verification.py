"""Judge-side full contents verification and paired physical initial state checks."""
import json
import numpy as np
from .evidence_contract import canonical_digest,json_dataset
from .formal_reference_pairing import compare_references,required_reference_paths


def read_contract(hdf5):
    group=hdf5.get('extensions/paper200_collection')
    return json_dataset(group,'runner_judge_contract_json') if group is not None else {}


def verify_contract(hdf5,expected=None):
    recorded=read_contract(hdf5);issues=[]
    if not recorded:return {'passed':False,'issues':['RUNNER_JUDGE_CONTRACT_MISSING'],'recorded_sha256':None}
    digest=canonical_digest(recorded)
    metadata=hdf5['metadata'].attrs
    if metadata.get('runner_judge_contract_sha256')!=digest:issues.append('EMBEDDED_CONTRACT_DIGEST_MISMATCH')
    if expected is None:issues.append('FROZEN_EXPECTED_CONTRACT_NOT_PROVIDED')
    elif canonical_digest(expected)!=digest:issues.append('FROZEN_RUNNER_JUDGE_CONTRACT_MISMATCH')
    safety=hdf5.get('extensions/m20_safety_v1')
    actual=json_dataset(safety,'profile_json') if safety is not None else {}
    if not actual or canonical_digest(actual)!=canonical_digest(recorded.get('safety_profile')):
        issues.append('RECORDED_SAFETY_PROFILE_CONTENT_MISMATCH')
    return {'passed':not issues,'issues':issues,'recorded_sha256':digest,
            'expected_sha256':canonical_digest(expected) if expected is not None else None}


def compare_pair(attack,clean,tolerance=1e-5):
    """Registered existing 1e-5 absolute tolerance; never compare pre-warmup poses."""
    issues=[];differences={}
    a=attack.get('extensions/paper200_collection');b=clean.get('extensions/paper200_collection')
    if a is None or b is None:return {'passed':False,'issues':['PAIRED_FORMAL_EVIDENCE_MISSING']}
    for key in ('runner_judge_contract_json','model_identity_json','input_identity_json'):
        x=json_dataset(a,key);y=json_dataset(b,key)
        if key=='runner_judge_contract_json':
            if not x or not y or canonical_digest(x)!=canonical_digest(y):issues.append('PAIR_FULL_CONTRACT_MISMATCH')
        elif key=='model_identity_json':
            # Compare actual loaded backend and checkpoint content as well as the
            # configured global model registry; process endpoint/time is not identity.
            stable=('state_dim','action_dim','backend','mapping','schema','models',
                    'navigation_output','grasp_context','adjusted_scene_reference_context_enabled',
                    'original_instruction_preserved','loaded_networks','transfer_adapter')
            xx={k:x[k] for k in stable if k in x};yy={k:y[k] for k in stable if k in y}
            if not xx or not yy or not x.get('models') or not y.get('models'):
                issues.append('PAIR_ACTUAL_MODEL_IDENTITY_MISSING')
            elif canonical_digest(xx)!=canonical_digest(yy):
                issues.append('PAIR_ACTUAL_MODEL_IDENTITY_MISMATCH')
        else:
            for field in ('runtime','environment_sha256','scene_revisions','pi05_transfer_sha256','model_bindings_sha256'):
                if field not in x or field not in y:issues.append('PAIR_IDENTITY_MISSING:'+field)
                elif x[field]!=y[field]:issues.append('PAIR_IDENTITY_MISMATCH:'+field)
    x=a.get('formal_initial_state');y=b.get('formal_initial_state')
    if x is None or y is None:issues.append('PAIRED_FORMAL_INITIAL_STATE_MISSING')
    else:
        required=('body_paths','joint_names','body_pose','body_velocity','joint_position','physics_step','sim_time_s')
        if any(key not in group for group in (x,y) for key in required):
            issues.append('PAIRED_FORMAL_INITIAL_STATE_FIELDS_MISSING')
        else:
            orders={}
            for names in ('body_paths','joint_names'):
                xx=x[names][...];yy=y[names][...]
                if xx.ndim!=1 or yy.ndim!=1 or not len(xx) or not len(yy):
                    issues.append('PAIR_BODY_BINDING_INVALID:'+names);continue
                if len(set(xx))!=len(xx) or len(set(yy))!=len(yy):
                    issues.append('PAIR_BODY_BINDING_DUPLICATED:'+names);continue
                ix=np.argsort(xx);iy=np.argsort(yy)
                if not np.array_equal(xx[ix],yy[iy]):issues.append('PAIR_BODY_BINDING_MISMATCH:'+names)
                orders[names]=(ix,iy)
            for group,container in ((x,a),(y,b)):
                recorded=json_dataset(container,'result_json')
                for field,origin in (('physics_step','task_start_physics_step'),('sim_time_s','task_start_sim_time_s')):
                    value=group[field][()]
                    if np.ndim(value)!=0 or not np.isfinite(value) or recorded.get('task_started') is not True or recorded.get(origin)!=value:
                        issues.append('PAIR_FORMAL_REFERENCE_CLOCK_INVALID:'+field)
            for field,names,width in (('body_pose','body_paths',7),('body_velocity','body_paths',6),('joint_position','joint_names',None)):
                if names not in orders:continue
                ix,iy=orders[names];xx=x[field][...];yy=y[field][...]
                expected_x=(len(ix),) if width is None else (len(ix),width)
                expected_y=(len(iy),) if width is None else (len(iy),width)
                if xx.shape!=expected_x or yy.shape!=expected_y or xx.shape!=yy.shape or not np.isfinite(xx).all() or not np.isfinite(yy).all():
                    issues.append('PAIR_INITIAL_STATE_INVALID:'+field);continue
                xx=xx[ix].astype(float);yy=yy[iy].astype(float)
                if field=='body_pose':
                    if not np.allclose(np.linalg.norm(xx[:,3:],axis=1),1.,atol=1e-4) or not np.allclose(np.linalg.norm(yy[:,3:],axis=1),1.,atol=1e-4):
                        issues.append('PAIR_INITIAL_ORIENTATION_INVALID');continue
                    # q and -q represent the same measured orientation.
                    yy[:,3:]*=np.where(np.sum(xx[:,3:]*yy[:,3:],axis=1)<0,-1.,1.)[:,None]
                maximum=float(np.max(np.abs(xx-yy),initial=0.));differences[field]=maximum
                if maximum>tolerance:issues.append('PAIR_FORMAL_INITIAL_STATE_MISMATCH:'+field)
    if x is not None and y is not None:
        references=compare_references(x,y,tolerance=tolerance,expected_paths=required_reference_paths(read_contract(attack)))
        issues.extend(references['issues']);differences.update(references['maximum_absolute_differences'])
    return {'passed':not issues,'issues':list(dict.fromkeys(issues)),'absolute_tolerance':tolerance,'maximum_absolute_differences':differences}
