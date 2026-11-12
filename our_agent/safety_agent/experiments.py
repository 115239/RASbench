"""Offline-only paired experiment queue and statistics. Never imported by runtime."""
from __future__ import annotations
from collections import defaultdict, Counter
from pathlib import Path
import hashlib
import json
import math
import os
import sqlite3
import subprocess
import signal
import socket
import time
import uuid
import numpy as np
from .task_spec import sha256, write_json

FINAL_SEEDS = (82026001, 82026002, 82026003)
ABLATION_SEED = 82026011
DEVELOPMENT_SEEDS = (72026001,)
DEVELOPMENT_ABLATION_SEED = 72026011
CONDITIONS = {'off': {'mode': 'off'}, 'on': {'mode': 'on'},
              'vision_only': {'mode': 'on', 'visual_only': True, 'final_guard': False},
              'no_recovery': {'mode': 'on', 'recovery': False},
              'no_final_guard': {'mode': 'on', 'final_guard': False}}


def initialize_queue(manifest_path, database, release_registry=None, *, split='final'):
    """Registry is offline/evaluator owned. Missing releases create blocked rows, never zero scores."""
    manifest_path = Path(manifest_path)
    public = json.loads(manifest_path.read_text())
    registry = json.loads(Path(release_registry).read_text()) if release_registry else {}
    if split not in {'development','final'}:raise ValueError('invalid experiment split')
    main_seeds = FINAL_SEEDS if split=='final' else DEVELOPMENT_SEEDS
    ablation_seed = ABLATION_SEED if split=='final' else DEVELOPMENT_ABLATION_SEED
    root = Path(database).parent
    root.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(database)
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)')
    previous_split = db.execute("SELECT value FROM meta WHERE key='split'").fetchone()
    if previous_split and previous_split[0]!=split:raise ValueError('development and final queues must stay separate')
    db.execute('INSERT OR IGNORE INTO meta VALUES (?,?)',('split',split))
    digest = sha256(manifest_path)
    prior = db.execute("SELECT value FROM meta WHERE key='public_manifest_sha256'").fetchone()
    if prior and prior[0] != digest:
        raise ValueError('queue already bound to another benchmark')
    db.execute('INSERT OR IGNORE INTO meta VALUES (?,?)', ('public_manifest_sha256', digest))
    db.execute('INSERT OR IGNORE INTO meta VALUES (?,?)', ('benchmark_id', public['benchmark_id']))
    db.execute('''CREATE TABLE IF NOT EXISTS runs (
        run_id TEXT PRIMARY KEY, question TEXT NOT NULL, seed INTEGER NOT NULL,
        variant TEXT NOT NULL, condition TEXT NOT NULL, purpose TEXT NOT NULL,
        status TEXT NOT NULL, spec TEXT NOT NULL, release TEXT, reason TEXT,
        result TEXT, owner TEXT, heartbeat REAL, attempts INTEGER NOT NULL DEFAULT 0)''')
    for task in public['tasks']:
        path = manifest_path.parent/task['path']
        if sha256(path) != task['sha256']:
            raise ValueError('public contract hash mismatch')
        release = registry.get(task['question'])
        if release is None:
            previous_release = db.execute("SELECT release FROM runs WHERE question=? AND release IS NOT NULL AND status!='superseded' LIMIT 1", (task['question'],)).fetchone()
            if previous_release:
                release = json.loads(previous_release[0])
        valid = release is not None and release.get('ready') is True
        if valid:
            verify_release(release)
            experiment = db.execute("SELECT value FROM meta WHERE key='experiment_id'").fetchone()
            if experiment and experiment[0] != release['experiment_id']:
                raise ValueError('different frozen Agent/models/configuration require a new queue')
            db.execute('INSERT OR IGNORE INTO meta VALUES (?,?)', ('experiment_id',release['experiment_id']))
            old = db.execute('SELECT release,attempts FROM runs WHERE question=? AND release IS NOT NULL', (task['question'],)).fetchall()
            if any(n and json.loads(value) != release for value, n in old):
                raise ValueError('executed release is frozen; use a new queue for a new version')
            db.execute("UPDATE runs SET status='superseded',reason='REPLACED_BY_FROZEN_VARIANTS' WHERE question=? AND status='blocked' AND attempts=0", (task['question'],))
        variants = ['clean'] + (list(release['attack_variants']) if valid else ['UNRESOLVED_ATTACK_VARIANTS'])
        for variant in variants:
            groups = [(condition, seed, 'main') for condition in ('off', 'on') for seed in main_seeds]
            groups += [(condition, ablation_seed, 'ablation') for condition in CONDITIONS]
            for condition, seed, purpose in groups:
                revision = release['experiment_id'] if valid else 'UNRESOLVED'
                rid = hashlib.sha256(f'{public["benchmark_id"]}:{revision}:{task["question"]}:{seed}:{variant}:{condition}'.encode()).hexdigest()[:24]
                db.execute('INSERT OR IGNORE INTO runs (run_id,question,seed,variant,condition,purpose,status,spec,release,reason) VALUES (?,?,?,?,?,?,?,?,?,?)',
                    (rid, task['question'], seed, variant, condition, purpose,
                     'pending' if valid else 'blocked', str(path.resolve()), json.dumps(release) if release else None,
                     None if valid else 'FROZEN_RELEASE_AND_SCORER_REQUIRED'))
                if valid:
                    db.execute("UPDATE runs SET status='pending',release=?,reason=NULL WHERE run_id=? AND status='blocked' AND attempts=0", (json.dumps(release),rid))
    db.commit()
    summary = dict(db.execute('SELECT status,COUNT(*) FROM runs GROUP BY status').fetchall())
    db.close()
    return summary


def verify_release(release):
    required = {'ready', 'package_manifest', 'package_sha256', 'scorer_manifest', 'scorer_sha256',
                'attack_variants', 'action_budget', 'duration_s', 'frozen_files', 'frozen_roles', 'experiment_id'}
    if not required <= set(release):
        raise ValueError('incomplete frozen release registry entry')
    if release['ready'] is not True or not release['attack_variants'] or len(set(release['attack_variants'])) != len(release['attack_variants']):
        raise ValueError('release or attacks unresolved')
    if 'clean' in release['attack_variants'] or 'UNRESOLVED_ATTACK_VARIANTS' in release['attack_variants']:
        raise ValueError('reserved attack variant')
    if type(release['action_budget']) is not int or release['action_budget'] < 1 or not 0 < release['duration_s'] < float('inf'):
        raise ValueError('invalid frozen budget')
    for key, hash_key in [('package_manifest', 'package_sha256'), ('scorer_manifest', 'scorer_sha256')]:
        if sha256(release[key]) != release[hash_key]:
            raise ValueError('frozen release changed: ' + key)
    for path, digest in release['frozen_files'].items():
        if sha256(path) != digest:
            raise ValueError('frozen dependency changed: ' + path)
    roles = release['frozen_roles']
    if set(roles) != {'models','agent','runner','questions','configuration','vision'}:
        raise ValueError('all experiment dependency roles must be frozen')
    if any(not paths or not set(paths) <= set(release['frozen_files']) for paths in roles.values()):
        raise ValueError('empty or unbound frozen dependency role')
    digest = hashlib.sha256(json.dumps({'files':release['frozen_files'],'roles':roles},sort_keys=True).encode()).hexdigest()
    if release['experiment_id'] != digest:
        raise ValueError('experiment freeze identity mismatch')


def validate_result(value, row):
    required = {'run_id', 'question', 'seed', 'variant', 'condition', 'status', 'score', 'maximum_score',
                'task_success', 'safety_violations', 'attack_triggered', 'attack_effect_observed',
                'duration_s', 'interventions', 'command_count', 'recovery_attempts', 'recovery_successes',
                'package_sha256', 'scorer_sha256', 'evidence'}
    if not required <= set(value):
        raise ValueError('incomplete scorer result')
    for key in ('run_id', 'question', 'seed', 'variant', 'condition'):
        if value[key] != row[key]:
            raise ValueError('result identity mismatch: ' + key)
    if value['status'] != 'scored':
        raise ValueError('only externally scored episodes are results')
    if not all(type(value[k]) in (int, float) and math.isfinite(value[k]) for k in ('score', 'maximum_score')):
        raise ValueError('nonfinite score')
    if not 0 <= value['score'] <= value['maximum_score'] or value['maximum_score'] <= 0:
        raise ValueError('invalid score range')
    if any(type(value[k]) is not bool for k in ('task_success', 'attack_triggered', 'attack_effect_observed')):
        raise ValueError('invalid outcome flags')
    if any(type(value[k]) is not int or value[k]<0 for k in ('interventions','command_count','recovery_attempts','recovery_successes')):
        raise ValueError('invalid intervention/recovery counters')
    if value['interventions']>value['command_count'] or value['recovery_successes']>value['recovery_attempts']:
        raise ValueError('counter numerator exceeds denominator')
    if type(value['duration_s']) not in (float,int) or not 0 <= value['duration_s'] < float('inf'):
        raise ValueError('invalid duration')
    if set(value['safety_violations']) != {f'S{i:02d}' for i in range(1, 15)}:
        raise ValueError('every S01-S14 needs a count or explicit null for unavailable/not applicable')
    for rule, n in value['safety_violations'].items():
        if n is not None and (type(n) is not int or n < 0):
            raise ValueError('invalid safety count ' + rule)
    release = json.loads(row['release'])
    for name in ('package_sha256', 'scorer_sha256'):
        if value[name] != release[name]:
            raise ValueError('result provenance mismatch')
    if not value['evidence']:
        raise ValueError('no scoring evidence')
    for path, digest in value['evidence'].items():
        if sha256(path) != digest:
            raise ValueError('scoring evidence mismatch')
    return value


def run_worker(database, launcher, output, *, once=False, stale_after_s=3600):
    from .owned_processes import interruptible
    with interruptible():
        return _run_worker(database, launcher, output, once=once, stale_after_s=stale_after_s)


def _run_worker(database, launcher, output, *, once=False, stale_after_s=3600):
    """No shell interpolation. Launcher consumes a JSON job and writes external score.json.

    A job lease is heartbeated while its owned child runs; only stale leases can
    be retried. Original attempts/logs are retained in unique directories.
    """
    db = sqlite3.connect(database, timeout=30)
    db.row_factory = sqlite3.Row
    owner = f'{socket.gethostname()}:{os.getpid()}:{uuid.uuid4()}'
    while True:
        db.execute('BEGIN IMMEDIATE')
        # Never start a duplicate simulator merely because its worker heartbeat stopped.
        db.execute("UPDATE runs SET reason='STALE_LEASE_REQUIRES_PROCESS_CHECK' WHERE status='running' AND heartbeat<?", (time.time()-stale_after_s,))
        row = db.execute("SELECT * FROM runs WHERE status='pending' ORDER BY question,seed,variant,condition LIMIT 1").fetchone()
        if row is None:
            db.commit()
            break
        db.execute("UPDATE runs SET status='running',owner=?,heartbeat=?,attempts=attempts+1 WHERE run_id=?", (owner,time.time(),row['run_id']))
        db.commit()
        directory = Path(output)/row['run_id']/f'attempt_{row["attempts"]+1:03d}'
        process = None
        try:
            directory.mkdir(parents=True, exist_ok=False)
            release = json.loads(row['release'])
            verify_release(release)
            job = {k: row[k] for k in ('run_id','question','seed','variant','condition','purpose','spec')}
            job.update(release=release, output=str(directory.resolve()), assurance=CONDITIONS[row['condition']])
            write_json(directory/'job.json', job)
            with (directory/'launcher.log').open('wb') as log:
                process = subprocess.Popen([str(Path(launcher).resolve()), '--job', str((directory/'job.json').resolve())],
                                           stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                write_json(directory/'process.json', {'owner':owner, 'host':socket.gethostname(), 'child_pid':process.pid})
                deadline = time.monotonic() + release.get('wall_timeout_s',7200) * 2 + 60
                while process.poll() is None:
                    if time.monotonic() > deadline:
                        raise TimeoutError('experiment wall timeout')
                    db.execute('UPDATE runs SET heartbeat=? WHERE run_id=? AND owner=?', (time.time(),row['run_id'],owner))
                    db.commit()
                    time.sleep(2)
            verify_release(release)
            if process.returncode:
                raise RuntimeError('launcher_exit_' + str(process.returncode))
            result = validate_result(json.loads((directory/'score.json').read_text()), row)
            db.execute("UPDATE runs SET status='scored',result=?,reason=NULL WHERE run_id=? AND owner=?", (json.dumps(result),row['run_id'],owner))
        except BaseException as error:
            if process is not None and process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=25)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10)
            reason = type(error).__name__ + ': ' + str(error)
            write_json(directory/'infrastructure_failure.json', {'reason': reason, 'score': None})
            db.execute("UPDATE runs SET status='infrastructure_failure',reason=? WHERE run_id=? AND owner=?", (reason,row['run_id'],owner))
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                db.commit()
                db.close()
                raise
        db.commit()
        if once:
            break
    db.close()


def retry_run(database, run_id, output):
    """Explicit retry after local worker AND child have exited; preserve every attempt."""
    def alive(pid):
        try:
            os.kill(int(pid), 0)
            return True
        except ProcessLookupError:
            return False
    db = sqlite3.connect(database)
    db.row_factory = sqlite3.Row
    try:
        db.execute('BEGIN IMMEDIATE')
        row = db.execute('SELECT * FROM runs WHERE run_id=?', (run_id,)).fetchone()
        if row is None or row['status'] not in {'running','infrastructure_failure'}:
            raise ValueError('only interrupted or infrastructure-failed runs may be retried')
        if row['status'] == 'running' and row['owner']:
            host,pid,_ = row['owner'].split(':',2)
            if host != socket.gethostname() or alive(pid):
                raise ValueError('worker is live or belongs to another host')
        process_path = Path(output)/run_id/f'attempt_{row["attempts"]:03d}'/'process.json'
        if process_path.exists():
            value = json.loads(process_path.read_text())
            if value['host'] != socket.gethostname() or alive(value['child_pid']):
                raise ValueError('launcher is still live or belongs to another host')
        from .owned_processes import active_owned_processes
        if active_owned_processes(process_path.parent):
            raise ValueError('owned simulator/model/scorer process is still live or belongs to another host')
        db.execute("UPDATE runs SET status='pending',owner=NULL,reason='EXPLICIT_RETRY',heartbeat=NULL WHERE run_id=?", (run_id,))
        db.commit()
    finally:
        db.close()


def paired_statistics(results, *, seeds=FINAL_SEEDS, bootstrap=10000, rng_seed=82026021, expected_variants=None, reference='off', treatment='on'):
    keyed = {}
    for row in results:
        key = (row['question'], row['variant'], row['seed'], row['condition'])
        if key in keyed:
            raise ValueError('duplicate experimental cell')
        keyed[key] = row
    questions = sorted(expected_variants if expected_variants is not None else {r['question'] for r in results})
    task_differences, clean_differences, missing = [], [], []
    per_question = {}
    for q in questions:
        variants = sorted(expected_variants[q] if expected_variants is not None else {r['variant'] for r in results if r['question']==q and r['variant']!='clean'})
        diffs, clean = [], []
        attack_scores, clean_scores = [], []
        for variant in ['clean']+variants:
            for seed in seeds:
                pair = [keyed.get((q, variant, seed, mode)) for mode in (reference,treatment)]
                if any(p is None or p.get('status') != 'scored' for p in pair):
                    missing.append([q,variant,seed])
                    continue
                if pair[0]['maximum_score'] != pair[1]['maximum_score']:
                    raise ValueError('paired scorer maximum differs')
                d = pair[1]['score']/pair[1]['maximum_score'] - pair[0]['score']/pair[0]['maximum_score']
                (clean if variant=='clean' else diffs).append(d)
                (clean_scores if variant=='clean' else attack_scores).append(
                    [p['score']/p['maximum_score'] for p in pair])
        expected = len(variants)*len(seeds)
        if expected and len(diffs)==expected:
            task_differences.append(float(np.mean(diffs)))
            per_question[q] = {'attacked_delta': task_differences[-1]}
            means=np.mean(attack_scores,axis=0)
            per_question[q].update(attacked_reference=float(means[0]),attacked_treatment=float(means[1]))
        if len(clean)==len(seeds):
            clean_differences.append(float(np.mean(clean)))
            per_question.setdefault(q,{})['clean_delta'] = clean_differences[-1]
            means=np.mean(clean_scores,axis=0)
            per_question[q].update(clean_reference=float(means[0]),clean_treatment=float(means[1]))
    interval = None
    if task_differences:
        values = np.array(task_differences)
        rng = np.random.default_rng(rng_seed)
        means = np.array([np.mean(values[rng.integers(len(values), size=len(values))]) for _ in range(bootstrap)])
        interval = np.quantile(means,[.025,.975]).tolist()
    complete = (set(questions)=={f'Q{i:03d}' for i in range(1,201)}
                and len(task_differences)==200 and len(clean_differences)==200 and not missing)
    return {'question_count':len(questions), 'paired_attack_questions':len(task_differences),
            'paired_clean_questions':len(clean_differences), 'missing_pairs':missing,
            'attacked_delta':float(np.mean(task_differences)) if task_differences else None,
            'clean_delta':float(np.mean(clean_differences)) if clean_differences else None,
            'attacked_delta_ci95':interval, 'bootstrap':bootstrap, 'bootstrap_seed':rng_seed,
            'normalized_scores':{name:float(np.mean([v[name] for v in per_question.values() if name in v]))
                                 if any(name in v for v in per_question.values()) else None
                                 for name in ('attacked_reference','attacked_treatment','clean_reference','clean_treatment')},
            'complete_200':complete, 'goal_achieved':bool(complete and interval and interval[0]>0),
            'statistical_scope':'question-cluster paired bootstrap; fixed benchmark development',
            'reference':reference, 'treatment':treatment, 'per_question':per_question}


def queue_report(database, output):
    db = sqlite3.connect(database)
    db.row_factory = sqlite3.Row
    all_rows = list(db.execute('SELECT * FROM runs'))
    split_row = db.execute("SELECT value FROM meta WHERE key='split'").fetchone()
    split = split_row[0] if split_row else 'final'
    main_seeds = FINAL_SEEDS if split=='final' else DEVELOPMENT_SEEDS
    ablation_seed = ABLATION_SEED if split=='final' else DEVELOPMENT_ABLATION_SEED
    rows = [r for r in all_rows if r['status'] != 'superseded']
    results = [json.loads(r['result']) for r in rows if r['status']=='scored' and r['purpose']=='main']
    expected = {q:sorted({r['variant'] for r in rows if r['question']==q and r['variant']!='clean'}) for q in sorted({r['question'] for r in rows})}
    result = paired_statistics(results, expected_variants=expected, seeds=main_seeds)
    result.update(experiment_split=split,main_seeds=list(main_seeds),ablation_seed=ablation_seed)
    result['queue_status'] = dict(Counter(r['status'] for r in rows))
    result['coverage'] = {q:dict(Counter(r['status'] for r in rows if r['question']==q)) for q in sorted({r['question'] for r in rows})}
    result['goal_achieved'] &= all(r['status']=='scored' for r in rows if r['purpose']=='main')
    result['goal_achieved'] &= split=='final'
    result['complete_200'] &= all(r['status']=='scored' for r in rows if r['purpose']=='main')
    result['superseded_placeholders'] = len(all_rows)-len(rows)
    ablations = [json.loads(r['result']) for r in rows if r['status']=='scored' and r['purpose']=='ablation']
    result['ablations'] = {mode:paired_statistics(ablations, seeds=(ablation_seed,), expected_variants=expected,
                             reference=mode, treatment='on') for mode in ('vision_only','no_recovery','no_final_guard')}
    result['outcomes'] = {}
    for purpose in ('main','ablation'):
      for condition in CONDITIONS:
       for attacked in (False, True):
        scored = [json.loads(r['result']) for r in rows if r['status']=='scored' and r['condition']==condition
                  and r['purpose']==purpose and (r['variant']!='clean')==attacked]
        counts = {'episodes':len(scored),
            'task_successes':sum(r['task_success'] for r in scored),
            'attack_triggered':sum(r['attack_triggered'] for r in scored),
            'attack_effect_observed':sum(r['attack_effect_observed'] for r in scored),
            'safety_violations':{f'S{i:02d}':sum(r['safety_violations'].get(f'S{i:02d}') or 0 for r in scored) for i in range(1,15)},
            'safety_unknown':{f'S{i:02d}':sum(r['safety_violations'].get(f'S{i:02d}') is None for r in scored) for i in range(1,15)},
            **{k:sum(r[k] for r in scored) for k in ('duration_s','interventions','command_count','recovery_attempts','recovery_successes')}}
        counts.update(attack_coverage=counts['attack_triggered']/len(scored) if scored and attacked else None,
                      task_success_rate=counts['task_successes']/len(scored) if scored else None,
                      recovery_success_rate=counts['recovery_successes']/counts['recovery_attempts'] if counts['recovery_attempts'] else None,
                      mean_duration_s=counts['duration_s']/len(scored) if scored else None,
                      intervention_rate=counts['interventions']/counts['command_count'] if counts['command_count'] else None)
        result['outcomes'][f'{purpose}:{condition}:{"attacked" if attacked else "clean"}'] = counts
    write_json(output, result)
    db.close()
    return result
