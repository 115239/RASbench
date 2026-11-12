"""Formal-reference pairing is statistical qualification, not collection success.

Only attempts inside this release are candidates. Parked prior-release episodes
cannot silently supply a missing clean, even if their visible poses happen to agree.
"""
from collections import defaultdict
from pathlib import Path
import json,os,subprocess,sys
import numpy as np
from .queue import connect
from .common import write,read,now,SYSTEMS,NATIVE
from .evidence_contract import canonical_digest
from .formal_reference_pairing import compare_references,required_reference_paths,FIELDS


def compare_formal_arrays(a,b,tolerance=1e-5):
    issues=[];differences={};orders={}
    for name in ('body_paths','joint_names'):
        x=np.asarray(a[name]).astype(str);y=np.asarray(b[name]).astype(str)
        if x.ndim!=1 or y.ndim!=1 or not len(x) or not len(y) or len(set(x))!=len(x) or len(set(y))!=len(y):
            issues.append('FORMAL_BINDING_INVALID:'+name);continue
        ix=np.argsort(x);iy=np.argsort(y)
        if not np.array_equal(x[ix],y[iy]):issues.append('FORMAL_BINDING_MISMATCH:'+name)
        orders[name]=(ix,iy)
    for field,names,width in (('body_pose','body_paths',7),('body_velocity','body_paths',6),('joint_position','joint_names',None)):
        if names not in orders:continue
        ix,iy=orders[names];x=np.asarray(a[field]);y=np.asarray(b[field])
        sx=(len(ix),) if width is None else (len(ix),width);sy=(len(iy),) if width is None else (len(iy),width)
        if x.shape!=sx or y.shape!=sy or x.shape!=y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
            issues.append('FORMAL_STATE_INVALID:'+field);continue
        x=x[ix].astype(float);y=y[iy].astype(float)
        if field=='body_pose':
            if not np.allclose(np.linalg.norm(x[:,3:],axis=1),1.,atol=1e-4,rtol=0) or not np.allclose(np.linalg.norm(y[:,3:],axis=1),1.,atol=1e-4,rtol=0):
                issues.append('FORMAL_ORIENTATION_INVALID');continue
            y[:,3:]*=np.where(np.sum(x[:,3:]*y[:,3:],axis=1)<0,-1.,1.)[:,None]
        maximum=float(np.max(np.abs(x-y),initial=0.));differences[field]=maximum
        if maximum>tolerance:issues.append('FORMAL_STATE_MISMATCH:'+field)
    references=compare_references(a,b,tolerance=tolerance)
    issues.extend(references['issues']);differences.update(references['maximum_absolute_differences'])
    return {'matched':not issues,'issues':issues,'maximum_absolute_differences':differences,'absolute_tolerance':tolerance}


def load_formal(root,job):
    out=Path(job['output']).resolve();root=Path(root).resolve()
    if root not in out.parents:raise ValueError('PARKED_OR_EXTERNAL_RELEASE_NOT_PAIR_CANDIDATE')
    identity=read(out/'identity.json');recorded_job=read(out/'job.json')
    if identity.get('schema')!='paper200-runner-judge/v2':raise ValueError('FORMAL_IDENTITY_SCHEMA_UNSUPPORTED')
    for key in ('question','system','variant','seed','purpose'):
        if recorded_job.get(key)!=job.get(key):raise ValueError('QUEUE_ATTEMPT_IDENTITY_MISMATCH:'+key)
    for key,job_key in (('system_id','system'),('seed','seed'),('purpose','purpose')):
        if identity.get(key)!=job.get(job_key):raise ValueError('FORMAL_IDENTITY_MISMATCH:'+key)
    for key in ('seed','policy_seed','repeat_index'):
        if type(identity.get(key)) is not int or identity[key]<0:raise ValueError('FORMAL_IDENTITY_UNKNOWN:'+key)
    if identity.get('condition_id')!=('clean' if job['variant']=='clean' else 'attack__q%d_%s'%(int(job['question'][1:]),job['variant'])):
        raise ValueError('FORMAL_CONDITION_MISMATCH')
    result=read(out/'episode/result.json')
    if result.get('task_started') is not True:raise ValueError('FORMAL_TASK_NOT_STARTED')
    # Deliberately no fallback to out/initial_state.npz: historical exports there
    # may contain warmup poses and lack the measured formal-origin clock.
    with np.load(out/'episode/formal_initial_state.npz',allow_pickle=False) as file:
        required=('body_paths','joint_names','body_pose','body_velocity','joint_position','physics_step','sim_time_s')+FIELDS
        if any(key not in file for key in FIELDS):raise ValueError('FORMAL_AUTHORITATIVE_REFERENCE_FIELDS_MISSING')
        if any(key not in file for key in required):raise ValueError('FORMAL_INITIAL_SCHEMA_INCOMPLETE')
        state={k:file[k].copy() for k in required}
    for key,origin in (('physics_step','task_start_physics_step'),('sim_time_s','task_start_sim_time_s')):
        value=state[key]
        if value.ndim!=0 or not np.isfinite(value) or value<0 or (key=='physics_step' and value.dtype.kind not in 'iu') or result.get(origin)!=value.item():raise ValueError('FORMAL_REFERENCE_CLOCK_MISMATCH:'+key)
    contract=read(out/'runner_judge_contract.json')
    references=compare_references(state,state,expected_paths=required_reference_paths(contract))
    if not references['matched']:raise ValueError(','.join(references['issues']))
    self_check=compare_formal_arrays(state,state)
    if not self_check['matched']:raise ValueError(','.join(self_check['issues']))
    return {'job':job,'out':out,'identity':identity,'state':state,'input':read(out/'input_identity.json'),
            'contract':contract,
            'clock':{k:state[k].item() for k in ('physics_step','sim_time_s')}}


def physical_context(a,b):
    issues=[]
    for key in ('task_id','seed','policy_seed','repeat_index','block_id','purpose','system_version'):
        x=a['identity'].get(key);y=b['identity'].get(key)
        if x is None or y is None or x!=y:issues.append('PAIR_CONTEXT_IDENTITY_MISMATCH:'+key)
    for key in ('task_sha256','environment_sha256','scene_revisions','public_task_sha256'):
        if key not in a['input'] or key not in b['input'] or a['input'][key]!=b['input'][key]:issues.append('PAIR_PHYSICAL_CONTEXT_MISMATCH:'+key)
    if canonical_digest(a['contract'])!=canonical_digest(b['contract']):issues.append('PAIR_FULL_CONTRACT_MISMATCH')
    # The selected backend/weights can differ across systems. Shared physical
    # Runner sources must agree; changes to either require a fresh comparison.
    def physical_sources(value):
        files=value.get('runtime',{}).get('files',{})
        return {k:v for k,v in files.items() if k.startswith(('runtime/runner/','runtime/baseline_runner/'))}
    left=physical_sources(a['input']);right=physical_sources(b['input'])
    if not left or not right or left!=right:issues.append('PAIR_PHYSICAL_RUNTIME_MISMATCH')
    return issues


def _compare_hdf5_in_process(a,b,tolerance):
    import h5py
    from .contract_verification import compare_pair
    from .evidence_contract import resolve_identity
    issues=[]
    with h5py.File(Path(a)/'episode/submission.hdf5','r') as x,h5py.File(Path(b)/'episode/submission.hdf5','r') as y:
        result=compare_pair(x,y,tolerance=tolerance)
        for path,file in ((Path(a),x),(Path(b),y)):
            job=read(path/'job.json');sidecar=read(path/'identity.json')
            identity=resolve_identity(file,question_id=job['question'])
            for field in ('system_id','system_version','task_id','seed','policy_seed','purpose','repeat_index','block_id','condition_id'):
                if identity.get(field)!=sidecar.get(field):issues.append('PAIR_HDF5_SIDECAR_IDENTITY_MISMATCH:'+field)
            group=file.get('extensions/paper200_collection/formal_initial_state')
            if group is None:issues.append('PAIR_HDF5_FORMAL_REFERENCE_MISSING');continue
            with np.load(path/'episode/formal_initial_state.npz',allow_pickle=False) as state:
                matched=compare_formal_arrays(state,group,tolerance=0.)
                issues.extend('HDF5_SIDECAR_'+reason for reason in matched['issues'])
                for field in ('physics_step','sim_time_s'):
                    if field not in group or state[field].item()!=group[field][()].item():issues.append('PAIR_HDF5_SIDECAR_CLOCK_MISMATCH:'+field)
    result['issues']=list(dict.fromkeys(result['issues']+issues));result['passed']=not result['issues']
    return result


def compare_hdf5(a,b,tolerance):
    try:
        import h5py
    except ImportError:
        # The collection supervisor uses Python 3.12 without h5py; the pinned
        # Isaac Python 3.11 environment owns the HDF5 ABI. Do not install or
        # import that ABI into the supervisor interpreter.
        env=dict(os.environ);env['PYTHONPATH']=str(Path(__file__).resolve().parents[1])
        call=subprocess.run([str(NATIVE/'isaac_python.sh'),'-m','paper200.paired_initial','--compare-hdf5',str(a),str(b),str(tolerance)],
            env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=120)
        if call.returncode:raise ValueError('PAIR_HDF5_INSPECTION_FAILED:'+call.stderr[-1000:])
        result=json.loads(call.stdout);result['inspection_runtime']='pinned_isaac_python_subprocess';return result
    result=_compare_hdf5_in_process(a,b,tolerance);result['inspection_runtime']='current_h5py_compatible_interpreter';return result


def compare_records(a,b,kind,tolerance):
    measured=compare_formal_arrays(a['state'],b['state'],tolerance)
    issues=physical_context(a,b)+measured['issues'];same_system=None
    if kind=='same_system_clean_attack':
        # Strong existing comparison verifies actual loaded models, all runtime
        # and transfer bindings, full contract, measured origins and HDF5 arrays.
        same_system=compare_hdf5(a['out'],b['out'],tolerance)
        issues.extend(same_system['issues'])
    return {'question':a['job']['question'],'seed':a['job']['seed'],'purpose':a['job']['purpose'],
        'kind':kind,'reference':a['job']['run_id'],'run_id':b['job']['run_id'],
        'systems':[a['job']['system'],b['job']['system']],'matched':not issues,'issues':list(dict.fromkeys(issues)),
        'maximum_absolute_differences':measured['maximum_absolute_differences'],
        'formal_reference_clocks':[a['clock'],b['clock']],'same_system_hdf5_validation':same_system,
        'reference_files':[str(a['out']/'episode/formal_initial_state.npz'),str(b['out']/'episode/formal_initial_state.npz')]}


def evaluate(root,rows,purpose='main',tolerance=1e-5,questions=None):
    if questions:rows=[r for r in rows if r['question'] in questions]
    records=[];issues=[];pending=[];loaded=[]
    for job in rows:
        if job.get('status')!='collected':continue
        try:loaded.append(load_formal(root,job))
        except (ValueError,KeyError,OSError,TypeError,subprocess.SubprocessError) as exc:
            issue={'run_id':job['run_id'],'reason':str(exc)}
            (pending if str(exc)=='PARKED_OR_EXTERNAL_RELEASE_NOT_PAIR_CANDIDATE' else issues).append(issue)
    by_system=defaultdict(list);by_condition=defaultdict(list)
    for row in loaded:
        j=row['job'];by_system[(j['question'],j['seed'],j['purpose'],j['system'])].append(row)
        by_condition[(j['question'],j['seed'],j['purpose'],j['variant'])].append(row)
    for key,group in by_system.items():
        clean=[r for r in group if r['job']['variant']=='clean']
        attacks=[r for r in group if r['job']['variant']!='clean']
        for attack in attacks:
            if len(clean)!=1:
                pending.append({'run_id':attack['job']['run_id'],'reason':'CLEAN_PAIR_NOT_IN_CURRENT_RELEASE' if not clean else 'CLEAN_PAIR_AMBIGUOUS','needed':{'question':key[0],'seed':key[1],'purpose':key[2],'system':key[3],'variant':'clean'}});continue
            try:records.append(compare_records(clean[0],attack,'same_system_clean_attack',tolerance))
            except (ValueError,KeyError,OSError,TypeError,subprocess.SubprocessError) as exc:issues.append({'run_id':attack['job']['run_id'],'reason':'SAME_SYSTEM_PAIR_EVIDENCE_INVALID:'+str(exc)})
    expected_systems=SYSTEMS[:2] if purpose in ('assisted','assisted_pilot') else SYSTEMS
    for key,group in by_condition.items():
        group.sort(key=lambda r:r['job']['system'])
        missing=sorted(set(expected_systems)-{r['job']['system'] for r in group})
        if missing:pending.append({'question':key[0],'seed':key[1],'purpose':key[2],'variant':key[3],'reason':'CROSS_SYSTEM_PAIR_PENDING','missing_systems':missing})
        for other in group[1:]:records.append(compare_records(group[0],other,'cross_system_physical_reference',tolerance))
    issues.extend(r for r in records if not r['matched'])
    uncollected=sum(r.get('status')!='collected' for r in rows)
    if uncollected:pending.append({'reason':'CURRENT_RELEASE_COLLECTION_PENDING','runs':uncollected})
    if not rows:pending.append({'reason':'NO_CURRENT_RELEASE_PAIR_CANDIDATES'})
    ready=bool(records) and not issues and not pending
    return {'schema':'paper200-formal-pairing/v2','time':now(),'purpose':purpose,'compared_pairs':len(records),
        'observed_runs':sum(r.get('status')=='collected' for r in rows),'qualified_formal_references':len(loaded),
        'passed':ready,'pairing_ready':ready,'pairing_pending':pending,'collection_completeness_independent':True,
        'absolute_tolerance':tolerance,'issues':issues,'pairs':records,'question_filter':sorted(questions) if questions else None,
        'reference_policy':'episode/formal_initial_state.npz only; no warmup fallback; no parked release as clean',
        'score_status':'NOT_SCORED'}


def check(root,purpose='main',tolerance=1e-5,questions=None):
    root=Path(root)
    with connect(root/'queue.sqlite3') as db:
        rows=[dict(x) for x in db.execute('SELECT * FROM runs WHERE purpose=?',(purpose,))]
    result=evaluate(root,rows,purpose,tolerance,questions)
    suffix='_'+'_'.join(sorted(questions)) if questions else ''
    write(root/'reports'/('PAIRED_INITIAL_'+purpose+suffix+'.json'),result);return result


if __name__=='__main__':
    if len(sys.argv)==5 and sys.argv[1]=='--compare-hdf5':
        print(json.dumps(_compare_hdf5_in_process(sys.argv[2],sys.argv[3],float(sys.argv[4])),allow_nan=False))
    else:raise SystemExit('internal entrypoint: --compare-hdf5 ATTEMPT ATTEMPT TOLERANCE')
