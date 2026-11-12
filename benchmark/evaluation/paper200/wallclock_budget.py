"""One 600-second budget from queue claim through validation, not per subprocess."""
import functools,signal,time,os
from pathlib import Path
from .common import write,read,now
LIMIT_S=600
# R4 measured reserve: successful stop-to-attempt completion max 86.725 s;
# timed-out cooperative stop lag max 24.414 s; round their sum up to 120 s.
# Includes physical stopping, export, collection validation and owned cleanup.
# Shared by all three systems. Hard claim-to-completion limit remains 600 s.
EXPORT_RESERVE_S=120

class WallClockLimit(RuntimeError):
    pass



def cleanup_timed_out_attempt(out):
    """SIGALRM can interrupt run_job's finally; finish only its private groups."""
    from .operational_wallclock import process_identity,still_same
    manifest=out/'processes.json'
    if not manifest.exists():return
    info=read(manifest);owned=[]
    if info.get('launcher')!=os.getpid():raise RuntimeError('timeout cleanup launcher ownership mismatch')
    if (out/'RESIDENT_WORKER.json').exists() and not (out/'RESIDENT_EPISODE_RESULT.json').exists():
        from .resident_pool import abort
        abort(out,reason='hard 600s attempt deadline')
    for role in ('runner','model'):
        if not info.get(role):continue
        identity=process_identity(int(info[role]))
        if identity and str(out) in identity['command'] and identity['pgrp']==identity['pid']:
            try:os.killpg(identity['pgrp'],signal.SIGTERM);owned.append(identity)
            except ProcessLookupError:pass
    until=time.monotonic()+5
    while time.monotonic()<until and any(still_same(identity) for identity in owned):time.sleep(.05)
    for identity in owned:
        if still_same(identity):
            try:os.killpg(identity['pgrp'],signal.SIGKILL)
            except ProcessLookupError:pass
    write(out/'TIMEOUT_PRIVATE_CLEANUP.json',{'time':now(),'private_groups':owned,'shared_model_daemons_excluded':True,
        'scope':'bounded administrative cleanup after hard deadline; no task work budget extension'})


def enforce_wallclock(function):
    @functools.wraps(function)
    def wrapped(root,job,*args,**kwargs):
        from .queue import connect
        with connect(root/'queue.sqlite3') as db:
            row=db.execute('SELECT started FROM attempts WHERE run_id=? AND attempt=?',(job['run_id'],job['attempts'])).fetchone()
        if row is None:raise ValueError('wallclock budget requires a claimed attempt start')
        started=float(row[0]);deadline=started+LIMIT_S;out=Path(job['output'])
        job['_wall_clock_started_unix_s']=started;job['_wall_clock_deadline_unix_s']=deadline
        initial={'schema':'paper200-wallclock-budget/v1','started_unix_s':started,'deadline_unix_s':deadline,
                 'total_budget_s':LIMIT_S,'export_reserve_s':EXPORT_RESERVE_S,
                 'scope':'queue claim, episode preparation, model attachment/loading, scene reset/loading, physics, export and collection validation',
                 'shared_service_startup':'reported separately; only pre-existing ready services are outside attempt budget'}
        write(out/'WALLCLOCK_BUDGET.json',initial)
        previous=signal.getsignal(signal.SIGALRM)
        def expired(signum,frame):
            write(out/'WALLCLOCK_TIMEOUT.json',{**initial,'time':now(),'elapsed_s':time.time()-started,
                'reason':'WALLCLOCK_LIMIT_600S','automatic_retry':False,'partial_evidence_retained':True})
            raise WallClockLimit('WALLCLOCK_LIMIT_600S')
        signal.signal(signal.SIGALRM,expired)
        remaining=deadline-time.time()
        if remaining<=0:
            signal.signal(signal.SIGALRM,previous)
            raise WallClockLimit('WALLCLOCK_LIMIT_600S')
        signal.setitimer(signal.ITIMER_REAL,remaining)
        try:
            result=function(root,job,*args,**kwargs)
            if time.time()>deadline:raise WallClockLimit('WALLCLOCK_LIMIT_600S')
            timing={**initial,'finished_unix_s':time.time(),'elapsed_s':time.time()-started,'within_budget':True}
            write(out/'WALLCLOCK_RESULT.json',timing)
            if isinstance(result,dict):result['wallclock']=timing
            return result
        except WallClockLimit:
            signal.setitimer(signal.ITIMER_REAL,0)
            cleanup_timed_out_attempt(out)
            raise
        finally:
            signal.setitimer(signal.ITIMER_REAL,0)
            signal.signal(signal.SIGALRM,previous)
    return wrapped
