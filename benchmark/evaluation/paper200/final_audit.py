"""Final artifact acceptance; existing decode evidence is bound by file hashes."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from .common import read,write,sha,now,SYSTEMS,MAIN_SEED
from .queue import connect
from .paired_initial import check as check_initial
from .provenance import verify_frozen


def matrix_issues(root,jobs,purpose):
    from .resume_repaired import expected_matrix
    expected=expected_matrix(root,purpose)
    actual=[tuple(j[k] for k in ['question','system','variant','seed','purpose']) for j in jobs]
    issues=[]
    if set(actual)!=expected or len(actual)!=len(expected):
        issues.append({'category':'matrix_identity','expected':len(expected),'actual':len(actual),
                       'missing':sorted(expected-set(actual)),'unexpected':sorted(set(actual)-expected)})
    return issues


def audit(root,purpose='main'):
    issues=[]
    try:verify_frozen(root)
    except Exception as error:issues.append({'category':'frozen_inputs','error':str(error)})
    with connect(root/'queue.sqlite3') as db:
        jobs=[dict(j) for j in db.execute('SELECT * FROM runs WHERE purpose=?',(purpose,))]
    issues.extend(matrix_issues(root,jobs,purpose))
    from .resume_repaired import scope, expected_matrix
    collection_scope=scope(root)
    def inspect(job):
        problems=[]
        if job['status']!='collected':return [{'run_id':job['run_id'],'category':'uncollected','status':job['status']}]
        out=Path(job['output'])
        try:
            v=read(out/'COLLECTION_VALIDATION.json');identity=read(out/'input_identity.json')
            frozen=read(root/'manifests/FROZEN.json')
            if identity.get('frozen_version')!=frozen:raise ValueError('attempt was not launched from this frozen version')
            if v.get('collection_status')!='VALID':raise ValueError('invalid collection record')
            for key in ['hdf5','mp4']:
                path=Path(v[key])
                if path.parent!=out/'episode' or sha(path)!=v[key+'_sha256']:
                    raise ValueError(key+' missing, moved, or changed since successful validation')
            initial='formal_initial_state.npz' if collection_scope['collection_scope']=='repaired_remaining' else 'initial_state.npz'
            if not (out/initial).exists():raise ValueError('initial physical state missing: '+initial)
            if not read(out/'PROCESS_CLEANUP.json').get('completed'):raise ValueError('owned processes not cleaned up')
        except Exception as error:problems.append({'run_id':job['run_id'],'category':'artifact','error':str(error)})
        return problems
    with ThreadPoolExecutor(max_workers=4) as pool:
        for problems in pool.map(inspect,jobs):issues.extend(problems)
    pairs=check_initial(root,purpose) if jobs else {'passed':purpose=='assisted','compared_pairs':0}
    # A partial remainder can lack its parked counterpart. Pairing eligibility is
    # separate from artifact completeness; old initial states cannot fill the gap.
    if not pairs['passed'] and collection_scope['collection_scope']!='repaired_remaining':
        issues.append({'category':'paired_initial_states','report':'PAIRED_INITIAL_'+purpose+'.json'})
    result={'time':now(),'purpose':purpose,'passed':not issues,'expected':len(expected_matrix(root,purpose)),
            'collection_scope':collection_scope['collection_scope'],
            'whole_paper_complete':False,'legacy_audit_required':collection_scope['legacy_audit_required'],
            'paired_statistically_eligible':pairs.get('pairing_ready',pairs['passed']),
            'pairing_pending':pairs.get('pairing_pending',[]),'pairing_issues':pairs.get('issues',[]),
            'pairing_report':'PAIRED_INITIAL_'+purpose+'.json',
            'collected':sum(j['status']=='collected' for j in jobs),'issues':issues,
            'evidence':'artifact hashes, prior HDF5/video/clock validation and process cleanup checked; pairing eligibility reported separately; this does not certify whole-paper completion',
            'score_status':'NOT_SCORED'}
    write(root/'reports'/('FINAL_AUDIT_'+purpose+'.json'),result)
    return result
