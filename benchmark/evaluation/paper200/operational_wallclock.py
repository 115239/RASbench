"""External timeout supervision of an immutable collection. Raw attempts are retained."""
import argparse,json,os,signal,time,sqlite3
from pathlib import Path
from .common import now,read,write


def process_identity(pid):
    try:
        base=Path('/proc')/str(pid);raw=(base/'stat').read_text();fields=raw[raw.rfind(')')+2:].split()
        return {'pid':pid,'start_ticks':fields[19],'pgrp':int(fields[2]),
                'command':(base/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')}
    except (FileNotFoundError,ProcessLookupError):return None


def running(root):
    with sqlite3.connect('file:'+str(root/'queue.sqlite3')+'?mode=ro',uri=True) as db:
        db.row_factory=sqlite3.Row
        return [dict(x) for x in db.execute('SELECT r.*,a.started FROM runs r JOIN attempts a ON r.run_id=a.run_id AND r.attempts=a.attempt WHERE r.status="running" AND r.purpose IN ("main","assisted")')]


def capture(root,job,limit):
    out=Path(job['output']).resolve();out.relative_to(root.resolve())
    info=read(out/'processes.json');owner_pid=int(job['owner'].split(':')[0])
    if info.get('launcher')!=owner_pid:raise ValueError('launcher does not match queue owner')
    identities={}
    for name in ('launcher','model','runner'):
        if not info.get(name):continue
        identity=process_identity(int(info[name]))
        if identity is None:continue
        command=identity['command']
        if name=='launcher':
            if str(root) not in command or job['run_id'] not in command:raise ValueError('launcher command identity mismatch')
        elif str(out) not in command:raise ValueError('child command does not bind this attempt')
        identities[name]=identity
    return {'schema':'paper200-operational-timeout/v1','requested':now(),'requested_epoch':time.time(),
            'run_id':job['run_id'],'attempt':job['attempts'],'owner':job['owner'],'output':str(out),
            'started_epoch':job['started'],'wall_limit_s':limit,'elapsed_at_request_s':time.time()-job['started'],
            'identities':identities,'signals':[],'finalized':False,'collection_validity':'not asserted',
            'task_outcome':'not inferred','reason':'OPERATIONAL_WALLCLOCK_LIMIT_600S'}


def still_same(identity):
    live=process_identity(identity['pid'])
    return live is not None and live['start_ticks']==identity['start_ticks'] and live['command']==identity['command']


def send(record,name,signum,group=False):
    identity=record['identities'].get(name)
    if not identity or not still_same(identity):return
    if group:
        if identity['pgrp']!=identity['pid']:raise ValueError('child is not its own recorded process group')
        os.killpg(identity['pgrp'],signum)
    else:os.kill(identity['pid'],signum)
    record['signals'].append({'name':name,'signal':signum,'group':group,'time':now(),'identity':identity})


def finish_timeout(root,record):
    out=Path(record['output'])
    with sqlite3.connect(str(root/'queue.sqlite3'),timeout=60) as db:
        db.row_factory=sqlite3.Row;db.execute('BEGIN IMMEDIATE')
        row=db.execute('SELECT * FROM runs WHERE run_id=?',(record['run_id'],)).fetchone()
        if row is None or row['attempts']!=record['attempt'] or row['owner']!=record['owner']:
            raise ValueError('attempt ownership changed; refusing timeout reclassification')
        record['previous_queue_result']={k:row[k] for k in ('status','reason','result')}
        if row['status']=='collected':
            record.update(finalized=True,finished=now(),elapsed_s=time.time()-record['started_epoch'],
                          raw_collection_validation_preserved=True,legacy_collection_preserved=True,
                          wall_limit_exceeded=True,scope='600s new-attempt hard budget applies to v3; completed v2 artifact remains collected')
            write(out/'WALLCLOCK_TIMEOUT.json',record)
            return
        result={'error':'OPERATIONAL_WALLCLOCK_LIMIT_600S','timeout_report':str(out/'WALLCLOCK_TIMEOUT.json'),
                'partial_evidence_retained':True,'eligible_under_600s_budget':False,
                'automatic_retry':False,'prior_result':row['result']}
        db.execute('UPDATE runs SET status="infrastructure_failure",reason=?,result=?,heartbeat=? WHERE run_id=?',
                   ('OPERATIONAL_WALLCLOCK_LIMIT_600S',json.dumps(result),time.time(),record['run_id']))
        db.execute('UPDATE attempts SET status="infrastructure_failure",reason=?,finished=COALESCE(finished,?) WHERE run_id=? AND attempt=?',
                   ('OPERATIONAL_WALLCLOCK_LIMIT_600S',time.time(),record['run_id'],record['attempt']))
    record.update(finalized=True,finished=now(),elapsed_s=time.time()-record['started_epoch'],
                  raw_collection_validation_preserved=(out/'COLLECTION_VALIDATION.json').exists(),automatic_retry=False)
    write(out/'WALLCLOCK_TIMEOUT.json',record)


def run(root,limit=600,grace=45):
    root=Path(root).resolve()
    if not (root/'STOP').exists():raise ValueError('STOP must prevent new immutable-version dispatch')
    report=root/'reports/OPERATIONAL_600S_WATCHDOG.json'
    records={r['run_id']:r for r in read(report).get('records',[])} if report.exists() else {}
    while True:
        jobs=running(root);now_epoch=time.time()
        for job in jobs:
            if now_epoch-job['started']<limit or job['run_id'] in records:continue
            record=capture(root,job,limit);records[job['run_id']]=record
            out=Path(record['output']);write(out/'WALLCLOCK_TIMEOUT.json',record)
            # The existing physical backend checks this marker and calls normal
            # mission stopping/export. We never change frozen runner code.
            (out/'STOP').write_text('Operational total wallclock budget exceeded; please stop through normal collector export.\n')
        for record in records.values():
            if record['finalized']:continue
            elapsed=now_epoch-record['requested_epoch'];launcher=record['identities'].get('launcher')
            if elapsed>=grace and not any(s['name']=='launcher' for s in record['signals']):send(record,'launcher',signal.SIGTERM)
            if elapsed>=grace+35:
                for name in ('runner','model'):
                    if not any(s['name']==name and s['signal']==signal.SIGTERM for s in record['signals']):send(record,name,signal.SIGTERM,True)
            if elapsed>=grace+50:
                for name in ('runner','model'):send(record,name,signal.SIGKILL,True)
                send(record,'launcher',signal.SIGTERM)
            alive=[name for name,identity in record['identities'].items() if still_same(identity)]
            if not alive:finish_timeout(root,record)
            else:write(Path(record['output'])/'WALLCLOCK_TIMEOUT.json',record)
        write(report,{'time':now(),'pid':os.getpid(),'limit_s':limit,'grace_s':grace,'running':len(jobs),
                      'records':list(records.values()),'status':'DRAINING' if jobs or any(not r['finalized'] for r in records.values()) else 'DRAINED',
                      'source_code_changed':False,'new_dispatch_stopped':True})
        if not jobs and all(r['finalized'] for r in records.values()):return
        time.sleep(2)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--limit',type=int,default=600);p.add_argument('--grace',type=int,default=45);a=p.parse_args();run(a.root,a.limit,a.grace)
