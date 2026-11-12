"""Prepare only the drained remainder; releasing STOP is a separate final action.

No historical artifacts are accepted, edited or rescored by this module.
"""
import argparse
import fcntl
import json
import os
import sqlite3
from pathlib import Path
from .common import SYSTEMS, MAIN_SEED, digest, now, read, sha, write

IDENTITY = ('question', 'system', 'variant', 'seed', 'purpose')


def key(row):
    return tuple(row[field] for field in IDENTITY)


def full_matrix(root, purpose):
    questions = read(root/'manifests/questions.json')
    selected = ({row['question'] for row in read(root/'manifests/assistance.json')['questions']}
                if purpose == 'assisted' else {row['question'] for row in questions})
    systems = SYSTEMS[:2] if purpose == 'assisted' else SYSTEMS
    return {(q['question'], s, v, MAIN_SEED, purpose) for q in questions if q['question'] in selected
            for s in systems for v in ['clean'] + q['variants']}


def drain_plan(root):
    """Validate the complete partition without consulting or changing live queues."""
    path = root/'manifests/source_drain.json'
    drain = read(path)
    if drain.get('drained') is not True:
        raise ValueError('source drain is not certified')
    parked, remaining = drain['parked'], drain['remaining']
    combined = parked + remaining
    identities = [key(row) for row in combined]
    run_ids = [row['run_id'] for row in combined]
    if len(set(identities)) != len(identities) or len(set(run_ids)) != len(run_ids):
        raise ValueError('source drain has duplicate identities or run IDs')
    if any(row['status'] != 'collected' for row in parked):
        raise ValueError('parked source rows must be collected')
    if any(row['status'] in ('collected', 'running') for row in remaining):
        raise ValueError('remaining source rows must be drained and uncollected')
    expected = full_matrix(root, 'main') | full_matrix(root, 'assisted')
    if set(identities) != expected:
        raise ValueError('source drain does not exactly partition main and assisted matrices')
    return drain


def expected_matrix(root, purpose):
    if (root/'manifests/source_drain.json').exists():
        return {key(row) for row in drain_plan(root)['remaining'] if row['purpose'] == purpose}
    return full_matrix(root, purpose)


def scope(root):
    full = len(full_matrix(root, 'main'))
    path = root/'manifests/source_drain.json'
    if not path.exists():
        return {'collection_scope': 'full_matrix', 'main_planned': full,
                'main_parked': 0, 'main_expected': full, 'legacy_audit_required': False}
    drain = drain_plan(root)
    return {'collection_scope': 'repaired_remaining', 'main_planned': full,
            'main_parked': sum(row['purpose'] == 'main' for row in drain['parked']),
            'main_expected': sum(row['purpose'] == 'main' for row in drain['remaining']),
            'legacy_audit_required': bool(drain['parked']), 'source_root': drain['source'],
            'source_drain_sha256': sha(path)}



def collection_release_allowed(readiness):
    if readiness.get('passed') is not True:
        return False
    if readiness.get('real_scene_acceptance_passed') is True:
        return True
    return (readiness.get('release_mode') == 'expedited_collection'
            and readiness.get('collection_release_passed') is True
            and readiness.get('statistically_eligible') is False
            and isinstance(readiness.get('deferred_requirements'), dict))


def verify_running_development(root, running, readiness):
    if not running:
        return
    if readiness.get('release_mode') != 'expedited_collection' or not collection_release_allowed(readiness):
        raise ValueError('target development or formal work is still running')
    from .release_v2 import verify_development_snapshot
    for row in running:
        if row['purpose'] not in ('pilot','probe','assisted_pilot'):
            raise ValueError('running formal or unknown-purpose work prevents repaired preparation')
        problems=verify_development_snapshot(root,row.get('output') or '')
        if problems:
            raise ValueError('running development snapshot is not immutable: '+str(problems))
        launch=read(Path(row['output'])/'PRELAUNCH_IDENTITY.json')
        if any(launch.get(k)!=row[k] for k in ('question','system','purpose')):
            raise ValueError('running development snapshot does not match queue identity')


def verify_release(root):
    from .provenance import verify_frozen
    frozen = verify_frozen(root)
    readiness = read(root/'reports/SCORING_READINESS.json')
    if not collection_release_allowed(readiness):
        raise ValueError('real-scene Runner/Judge acceptance or explicit collection release has not passed')
    gate = read(root/'reports/MAIN_RELEASE_GATE.json')
    if gate.get('passed') is not True or gate.get('freeze') != frozen or gate.get('release_mode') != readiness.get('release_mode'):
        raise ValueError('release gate is missing or not bound to this frozen version')
    return {'frozen_sha256': sha(root/'manifests/FROZEN.json'),
            'readiness_sha256': sha(root/'reports/SCORING_READINESS.json'),
            'release_gate_sha256': sha(root/'reports/MAIN_RELEASE_GATE.json')}


def verify_source(root, drain):
    """Check every immutable ancestor snapshot against its actual read-only queue."""
    source = Path(drain['source']).resolve()
    snapshots = drain.get('sources')
    if snapshots is None:
        snapshots = [{'source_root': str(source), 'rows': drain['parked'] + drain['remaining']}]
    roots = [str(Path(item['source_root']).resolve()) for item in snapshots]
    if len(roots) != len(set(roots)) or str(source) not in roots:
        raise ValueError('source chain has duplicate or missing source roots')
    saved_by_source = {}
    for snapshot in snapshots:
        ancestor = Path(snapshot['source_root']).resolve()
        if ancestor == root.resolve() or not (ancestor/'STOP').is_file():
            raise ValueError('drained source must be a different stopped version')
        with sqlite3.connect('file:' + str(ancestor/'queue.sqlite3') + '?mode=ro', uri=True) as db:
            db.row_factory = sqlite3.Row
            db.execute('BEGIN')
            rows = [dict(row) for row in db.execute('SELECT * FROM runs WHERE purpose IN ("main", "assisted")')]
            running = db.execute('SELECT COUNT(*) FROM runs WHERE status="running"').fetchone()[0]
        if running:
            raise ValueError('source still has running work')
        saved = {row['run_id']: row for row in snapshot['rows']}
        if len(saved) != len(snapshot['rows']) or {row['run_id']: row for row in rows} != saved:
            raise ValueError('live source queue changed after the frozen drain snapshot')
        if ('source_freeze_sha256' in snapshot
                and snapshot['source_freeze_sha256'] != sha(ancestor/'manifests/FROZEN.json')):
            raise ValueError('source chain frozen identity changed')
        saved_by_source[str(ancestor)] = saved
    for row in drain['parked'] + drain['remaining']:
        origin = str(Path(row.get('source_root', str(source))).resolve())
        actual = saved_by_source.get(origin, {}).get(row['run_id'])
        stripped = {k:v for k,v in row.items() if k != 'source_root'}
        if actual != stripped:
            raise ValueError('flat source partition is not backed by its actual source snapshot')
    if any(Path(row.get('source_root', str(source))).resolve() != source for row in drain['remaining']):
        raise ValueError('remaining work must come only from the immediate source version')
    lineage = read(root/'manifests/repair_lineage.json')
    if Path(lineage['source_root']).resolve() != source or lineage['source_freeze_sha256'] != sha(source/'manifests/FROZEN.json'):
        raise ValueError('source lineage changed')
    if ('source_drain_sha256' in lineage
            and lineage['source_drain_sha256'] != sha(source/'manifests/source_drain.json')):
        raise ValueError('ancestor source drain changed')


def prepared_rows(root, drain, version):
    rows = []
    for original in drain['remaining']:
        identity = key(original)
        rid = digest([version] + list(identity))[:24]
        if rid == original['run_id']:
            raise ValueError('repaired version must have distinct run IDs')
        rows.append(dict(zip(('run_id',) + IDENTITY, (rid,) + identity),
                         source_run_id=original['run_id'], source_status=original['status'],
                         source_attempts=original['attempts'], source_output=original['output']))
        if drain.get('sources') is not None or 'source_root' in original:
            rows[-1]['source_root'] = original.get('source_root', drain['source'])
    return rows


def prepare(root, activate=False):
    """Idempotent, gated preparation. Only explicit activate atomically removes STOP."""
    from .queue import connect
    root = Path(root).resolve()
    with (root/'resume_repaired.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not (root/'STOP').is_file():
            raise ValueError('repair target must retain STOP until the final activation')
        certificates = verify_release(root)
        drain = drain_plan(root)
        verify_source(root, drain)
        with connect(root/'queue.sqlite3') as db:
            db.execute('BEGIN IMMEDIATE')
            running=[dict(row) for row in db.execute('SELECT * FROM runs WHERE status="running"')]
            if running:
                verify_running_development(root,running,read(root/'reports/SCORING_READINESS.json'))
            version = db.execute('SELECT value FROM meta WHERE key="version"').fetchone()[0]
            if version != read(root/'manifests/protocol.json')['version']:
                raise ValueError('queue and protocol versions differ')
            rows = prepared_rows(root, drain, version)
            expected = {key(row): row for row in rows}
            existing = [dict(row) for row in db.execute('SELECT * FROM runs WHERE purpose IN ("main", "assisted")')]
            for row in existing:
                if key(row) not in expected or row['run_id'] != expected[key(row)]['run_id']:
                    raise ValueError('target contains an unexpected or parked formal identity')
            db.execute('CREATE TABLE IF NOT EXISTS resume_lineage (run_id TEXT PRIMARY KEY, source_root TEXT NOT NULL, source_run_id TEXT NOT NULL, source_drain_sha256 TEXT NOT NULL)')
            drain_sha = sha(root/'manifests/source_drain.json')
            for row in rows:
                db.execute('INSERT OR IGNORE INTO runs(run_id,question,system,variant,seed,purpose) VALUES (?,?,?,?,?,?)',
                           (row['run_id'],) + key(row))
                lineage = (row['run_id'], row.get('source_root', drain['source']), row['source_run_id'], drain_sha)
                previous = db.execute('SELECT * FROM resume_lineage WHERE run_id=?', (row['run_id'],)).fetchone()
                if previous is not None and tuple(previous) != lineage:
                    raise ValueError('existing repaired run lineage differs')
                db.execute('INSERT OR IGNORE INTO resume_lineage VALUES (?,?,?,?)', lineage)
            # File writes precede commit: a failure leaves STOP and rolls back queue mutations.
            manifest = {'schema': 'paper200-repaired-remainder/v1', 'version': version,
                        'source_root': drain['source'], 'source_drain_sha256': drain_sha,
                        'parked_source_run_ids': [r['run_id'] for r in drain['parked']],
                        'runs': rows, 'certificates': certificates,
                        'old_records_accepted': False, 'post_remaining_stage': 'LEGACY_AUDIT_REQUIRED'}
            path = root/'manifests/resume_repaired.json'
            if path.exists() and read(path) != manifest:
                raise ValueError('previous repaired remainder manifest differs')
            if not path.exists():
                write(path, manifest)
        verify_source(root, drain)
        if verify_release(root) != certificates:
            raise ValueError('release certificates changed during preparation')
        result = {'time': now(), 'status': 'PREPARED_STOPPED', 'main': sum(r['purpose'] == 'main' for r in rows),
                  'assisted': sum(r['purpose'] == 'assisted' for r in rows), 'parked': len(drain['parked']),
                  'automatic_dispatch': False, 'score_status': 'NOT_SCORED'}
        if activate:
            # No command starts workers. Rename is the final, atomic dispatch-enabling action.
            archive = root/'reports/STOP_BEFORE_REPAIRED_RESUME'
            if archive.exists():
                raise ValueError('activation already recorded; inspect rather than overwrite prior STOP')
            result.update(status='ACTIVATED_PENDING_SCHEDULER', activated=now())
            write(root/'reports/REPAIRED_RESUME.json', result)
            os.rename(str(root/'STOP'), str(archive))
        else:
            write(root/'reports/REPAIRED_RESUME.json', result)
        return result



def verify_prepared(root):
    """Read-only admission check for a prepared remainder, including resumed workers."""
    from .queue import connect
    root = Path(root)
    certificates = verify_release(root)
    drain = drain_plan(root)
    verify_source(root, drain)
    manifest = read(root/'manifests/resume_repaired.json')
    with connect(root/'queue.sqlite3') as db:
        version = db.execute('SELECT value FROM meta WHERE key="version"').fetchone()[0]
        expected = prepared_rows(root, drain, version)
        actual = [dict(row) for row in db.execute('SELECT * FROM runs WHERE purpose IN ("main", "assisted")')]
        if {(row['run_id'], key(row)) for row in actual} != {(row['run_id'], key(row)) for row in expected}:
            raise ValueError('prepared remainder queue identity changed')
        actual_lineage = {tuple(row) for row in db.execute('SELECT * FROM resume_lineage')}
    drain_sha = sha(root/'manifests/source_drain.json')
    expected_lineage = {(row['run_id'], row.get('source_root', drain['source']), row['source_run_id'], drain_sha) for row in expected}
    if actual_lineage != expected_lineage:
        raise ValueError('prepared remainder lineage changed')
    if (manifest['version'] != version or manifest['source_root'] != drain['source']
            or manifest['source_drain_sha256'] != drain_sha or manifest['runs'] != expected
            or manifest['certificates'] != certificates
            or manifest['parked_source_run_ids'] != [r['run_id'] for r in drain['parked']]
            or manifest.get('old_records_accepted') is not False):
        raise ValueError('prepared remainder manifest changed')
    return manifest


def request_legacy_audit(root):
    """Produce a pending ledger after both remaining matrices pass artifact audit."""
    drain = drain_plan(root)
    for purpose in ('main', 'assisted'):
        audit = read(root/'reports'/('FINAL_AUDIT_' + purpose + '.json'))
        if audit.get('passed') is not True or audit.get('collection_scope') != 'repaired_remaining':
            raise ValueError('all repaired remainder artifact audits must pass before historical review')
    result = {'time': now(), 'status': 'LEGACY_AUDIT_REQUIRED', 'passed': False,
              'source_root': drain['source'], 'source_drain_sha256': sha(root/'manifests/source_drain.json'),
              'whole_paper_complete': False, 'score_status': 'NOT_SCORED',
              'records': [dict({field: row[field] for field in ('run_id',) + IDENTITY + ('output',)},
                               source_root=row.get('source_root', drain['source']))
                          for row in drain['parked']],
              'required_action': 'Audit each parked source record against repaired contracts and formal initial pairing; rerun only affected identities with preserved lineage. No historical files changed.'}
    write(root/'reports/LEGACY_AUDIT_REQUIRED.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--activate', action='store_true', help='After all gates, atomically archive STOP; does not start workers')
    args = parser.parse_args()
    print(json.dumps(prepare(args.root, args.activate), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
