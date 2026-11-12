"""SQLite claims are atomic; an abandoned running job needs explicit recovery."""
from pathlib import Path
import json
import os
import sqlite3
import time
from .common import SYSTEMS, MAIN_SEED, PILOT_SEED, digest, write


def connect(path):
    db = sqlite3.connect(str(path), timeout=60)
    db.row_factory = sqlite3.Row
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('PRAGMA busy_timeout=60000')
    return db


def initialize(root, questions, version):
    if (root/'manifests/source_drain.json').exists():
        raise ValueError('repaired remainder requires gated resume_repaired preparation, not full initialization')
    with connect(root/'queue.sqlite3') as db:
        db.executescript('''
        CREATE TABLE IF NOT EXISTS runs (
          run_id TEXT PRIMARY KEY, question TEXT NOT NULL, system TEXT NOT NULL,
          variant TEXT NOT NULL, seed INTEGER NOT NULL, purpose TEXT NOT NULL,
          status TEXT NOT NULL DEFAULT 'pending', attempts INTEGER NOT NULL DEFAULT 0,
          owner TEXT, heartbeat REAL, output TEXT, reason TEXT, result TEXT,
          UNIQUE(question,system,variant,seed,purpose));
        CREATE TABLE IF NOT EXISTS attempts (
          run_id TEXT NOT NULL, attempt INTEGER NOT NULL, output TEXT NOT NULL,
          started REAL NOT NULL, finished REAL, status TEXT NOT NULL, reason TEXT,
          PRIMARY KEY(run_id,attempt));
        CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY,value TEXT NOT NULL);
        ''')
        db.execute('INSERT OR IGNORE INTO meta VALUES (?,?)', ('version', version))
        if db.execute('SELECT value FROM meta WHERE key=?', ('version',)).fetchone()[0] != version:
            raise ValueError('queue version mismatch')
        for q in questions:
            for system in SYSTEMS:
                for variant in ['clean'] + q['variants']:
                    identity = [version, q['question'], system, variant, MAIN_SEED, 'main']
                    db.execute('INSERT OR IGNORE INTO runs(run_id,question,system,variant,seed,purpose) VALUES (?,?,?,?,?,?)',
                               (digest(identity)[:24], q['question'], system, variant, MAIN_SEED, 'main'))


def add_pilot(root, q, system, variant):
    with connect(root/'queue.sqlite3') as db:
        version = db.execute('SELECT value FROM meta WHERE key=?', ('version',)).fetchone()[0]
        rid = digest([version,q,system,variant,PILOT_SEED,'pilot'])[:24]
        db.execute('INSERT OR IGNORE INTO runs(run_id,question,system,variant,seed,purpose) VALUES (?,?,?,?,?,?)',
                   (rid,q,system,variant,PILOT_SEED,'pilot'))
    return rid


def add_probe(root,q,system,variant):
    if (root/'manifests/FROZEN.json').exists():raise ValueError('diagnostic probes precede freezing')
    if variant=='clean':raise ValueError('handler probe requires an attack variant')
    with connect(root/'queue.sqlite3') as db:
        version=db.execute('SELECT value FROM meta WHERE key=?',('version',)).fetchone()[0]
        rid=digest([version,q,system,variant,PILOT_SEED,'probe'])[:24]
        db.execute('INSERT OR IGNORE INTO runs(run_id,question,system,variant,seed,purpose) VALUES (?,?,?,?,?,?)',
                   (rid,q,system,variant,PILOT_SEED,'probe'))
    return rid


def add_assisted_pilot(root,q,system,variant='clean'):
    from .common import read
    manifest=read(root/'manifests/assistance.json')
    if not manifest.get('frozen') or q not in {x['question'] for x in manifest['questions']}:
        raise ValueError('assisted development run must use the frozen capability list')
    if system not in ('groot_agent_on','groot_agent_off'):raise ValueError('assistance is only defined for GR00T')
    with connect(root/'queue.sqlite3') as db:
        version=db.execute('SELECT value FROM meta WHERE key=?',('version',)).fetchone()[0]
        rid=digest([version,q,system,variant,PILOT_SEED,'assisted_pilot'])[:24]
        db.execute('INSERT OR IGNORE INTO runs(run_id,question,system,variant,seed,purpose) VALUES (?,?,?,?,?,?)',
                   (rid,q,system,variant,PILOT_SEED,'assisted_pilot'))
    return rid


def claim(root, owner, *, run_id=None, system=None, purpose='main'):
    with connect(root/'queue.sqlite3') as db:
        db.execute('BEGIN IMMEDIATE')
        query = "SELECT * FROM runs WHERE status='pending' AND purpose=?"
        args = [purpose]
        if run_id:
            query += ' AND run_id=?'; args.append(run_id)
        if system:
            query += ' AND system=?'; args.append(system)
        row = db.execute(query+' ORDER BY CAST(substr(question,2) AS INTEGER),variant,system LIMIT 1', args).fetchone()
        if row is None:
            return None
        job = dict(row)
        attempt = row['attempts']+1
        base=root/'pilot/handler_probes' if purpose=='probe' else root/'pilot/assisted' if purpose=='assisted_pilot' else root/purpose
        path = base/row['system']/('Q%03d' % int(row['question'][1:]))/row['variant']/('seed_%s' % row['seed'])/('attempt_%03d' % attempt)
        path.mkdir(parents=True, exist_ok=False)
        db.execute("UPDATE runs SET status='running',attempts=?,owner=?,heartbeat=?,output=? WHERE run_id=?",
                   (attempt,owner,time.time(),str(path),row['run_id']))
        db.execute('INSERT INTO attempts VALUES (?,?,?,?,?,?,?)',
                   (row['run_id'],attempt,str(path),time.time(),None,'running',None))
        job.update(attempts=attempt,output=str(path),owner=owner,status='running')
        write(path/'job.json',job)
        return job


def finish(root, job, result, error=None):
    deterministic_wallcap = error is not None and ('WALLCLOCK_LIMIT_600S' in error or 'OPERATIONAL_WALLCLOCK_LIMIT_600S' in error)
    status = 'collected' if error is None else ('pending' if job['attempts'] < 3 and not deterministic_wallcap else 'infrastructure_failure')
    with connect(root/'queue.sqlite3') as db:
        changed = db.execute('UPDATE runs SET status=?,reason=?,result=?,heartbeat=? WHERE run_id=? AND owner=? AND status=?',
                             (status,error,json.dumps(result),time.time(),job['run_id'],job['owner'],'running')).rowcount
        if changed != 1:
            raise RuntimeError('lost queue ownership')
        db.execute('UPDATE attempts SET finished=?,status=?,reason=? WHERE run_id=? AND attempt=?',
                   (time.time(),'collected' if error is None else 'infrastructure_failure',error,job['run_id'],job['attempts']))


def report(root):
    with connect(root/'queue.sqlite3') as db:
        groups = [dict(r) for r in db.execute('SELECT purpose,system,status,COUNT(*) AS count FROM runs GROUP BY purpose,system,status')]
        missing = [dict(r) for r in db.execute("SELECT run_id,question,system,variant,status,attempts,reason FROM runs WHERE status!='collected'")]
        attempts = db.execute('SELECT COUNT(*) FROM attempts').fetchone()[0]
        main_rows = [dict(r) for r in db.execute("SELECT * FROM runs WHERE purpose='main'")]
    main_total = sum(r['count'] for r in groups if r['purpose']=='main')
    main_valid = sum(r['count'] for r in groups if r['purpose']=='main' and r['status']=='collected')
    from .resume_repaired import scope, expected_matrix, key
    planned = scope(root)
    identities = [key(row) for row in main_rows]
    matrix_matches = len(identities) == len(set(identities)) and set(identities) == expected_matrix(root, 'main')
    remaining_complete = matrix_matches and main_valid == planned['main_expected']
    result = {**planned, 'main_enqueued': main_total, 'main_collected': main_valid, 'attempts': attempts,
              'groups': groups, 'uncollected': missing, 'matrix_matches': matrix_matches,
              'remaining_collection_complete': remaining_complete,
              'collection_complete': remaining_complete and not planned['legacy_audit_required'],
              'whole_paper_complete': False,
              'legacy_audit_status': 'REQUIRED_AFTER_REMAINING_COLLECTION' if planned['legacy_audit_required'] else 'NOT_APPLICABLE',
              'score_status': 'NOT_SCORED', 'task_success_rate': None}
    write(root/'reports/coverage.json',result)
    return result


def retry(root, run_id, reason):
    """Explicit repair retry, preserving the full attempt counter/history."""
    with connect(root/'queue.sqlite3') as db:
        db.execute('BEGIN IMMEDIATE')
        row=db.execute('SELECT * FROM runs WHERE run_id=?',(run_id,)).fetchone()
        if row is None:raise ValueError('unknown run')
        if row['status'] in ('running','collected'):raise ValueError('cannot retry active or collected run')
        if row['output']:
            for p in Path(row['output']).glob('processes.json'):
                info=json.loads(p.read_text())
                for key in ('launcher','model','runner'):
                    pid=info.get(key)
                    if pid and Path('/proc/%s/cmdline'%pid).exists():
                        cmd=Path('/proc/%s/cmdline'%pid).read_bytes()
                        if str(root).encode() in cmd:raise ValueError('owned process still alive: '+str(pid))
        db.execute("UPDATE runs SET status='pending',reason=? WHERE run_id=?",('REPAIR_RETRY:'+reason,run_id))
    return {'run_id':run_id,'status':'pending','reason':reason}


def retest_changed_package(root,run_id,reason):
    """Supersede a development result only when its recorded package changed."""
    from .common import read,sha,now
    if (root/'manifests/FROZEN.json').exists():raise ValueError('frozen version cannot be revised')
    with connect(root/'queue.sqlite3') as db:
        db.execute('BEGIN IMMEDIATE')
        row=db.execute('SELECT * FROM runs WHERE run_id=?',(run_id,)).fetchone()
        if row is None or row['purpose']!='pilot' or row['status']!='collected':
            raise ValueError('requires a completed development pilot')
        out=Path(row['output']);identity=read(out/'input_identity.json')
        files={'task_sha256':'data/task/task.json','rubric_sha256':'data/task/rubric.json',
               'environment_sha256':'data/env/environment.json'}
        changes={k:{'before':identity[k],'after':sha(root/'benchmark/task'/row['question']/p)}
                 for k,p in files.items() if sha(root/'benchmark/task'/row['question']/p)!=identity[k]}
        if not changes:raise ValueError('no recorded package changes')
        if not read(out/'PROCESS_CLEANUP.json').get('completed'):raise ValueError('attempt processes not cleaned up')
        write(out/'SUPERSEDED.json',{'time':now(),'reason':reason,'changes':changes,
            'original_collection_status':'collected','raw_evidence_retained':True})
        db.execute("UPDATE runs SET status='pending',reason=? WHERE run_id=?",('REVISED_PACKAGE_RETEST:'+reason,run_id))
    return {'run_id':run_id,'status':'pending','changes':changes}


def retest_changed_code(root,run_id,reason):
    from .common import read,sha,now,PROJECT
    if (root/'manifests/FROZEN.json').exists():raise ValueError('frozen version cannot be revised')
    with connect(root/'queue.sqlite3') as db:
        db.execute('BEGIN IMMEDIATE')
        row=db.execute('SELECT * FROM runs WHERE run_id=?',(run_id,)).fetchone()
        if row is None or row['purpose'] not in ('pilot','probe','assisted_pilot') or row['status']!='collected':
            raise ValueError('requires a completed development pilot')
        out=Path(row['output']);old=read(out/'code/manifest.json');changes={}
        for p in (PROJECT/'experiment/paper200').glob('*.py'):
            key='paper200/'+p.name;current=sha(p)
            if old.get(key)!=current:changes[key]={'before':old.get(key),'after':current}
        if not changes:raise ValueError('no collection code changes')
        if not read(out/'PROCESS_CLEANUP.json').get('completed'):raise ValueError('attempt still owns processes')
        write(out/'SUPERSEDED.json',{'time':now(),'reason':reason,'code_changes':changes,'raw_evidence_retained':True})
        db.execute("UPDATE runs SET status='pending',reason=? WHERE run_id=?",('REVISED_CODE_RETEST:'+reason,run_id))
    return {'run_id':run_id,'status':'pending','changed_files':len(changes)}
