"""Create an isolated repair candidate. This never dispatches or mutates v1."""
import argparse
import os
import shutil
import sqlite3
import subprocess
from pathlib import Path
from .common import now, read, sha, write


def prepare(source, target, judge_source, repair_spec):
    source, target = Path(source).resolve(), Path(target).resolve()
    if not (source / 'STOP').is_file():
        raise ValueError('source must have a dispatch stop marker')
    if target.exists():
        raise FileExistsError(target)
    target.mkdir(parents=True)
    (target / 'STOP').write_text('Repair candidate: dispatch prohibited until real-scene acceptance.\n')
    write(target / 'REPAIR_IN_PROGRESS.json', {'created': now(), 'source': str(source),
        'spec': str(repair_spec), 'spec_sha256': sha(repair_spec),
        'status': 'DEVELOPMENT_NOT_RELEASED'})
    # Reflinks are independent files, never hard links into the frozen source.
    for name in ('runtime', 'benchmark'):
        subprocess.run(['cp', '-a', '--reflink=auto', str(source / name), str(target / name)], check=True)
    link = target / 'runtime/baseline/code'
    if not link.is_symlink():
        raise ValueError('unexpected baseline code binding')
    link.unlink()
    link.symlink_to(target / 'runtime/baseline_code')
    shutil.copytree(judge_source, target / 'runtime/judge/v2',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    # Old release certificates remain ancestry evidence, never v2 certificates.
    shutil.copytree(source / 'manifests', target / 'ancestry/v1_manifests')
    manifests = target / 'manifests'
    manifests.mkdir()
    retained = ('models.json', 'model_files_verified.json', 'native_runtime.json',
                'pi05_transfer.json', 'questions.json', 'assistance.json',
                'geometry.json', 'rubric_bindings.json')
    for name in retained:
        value = read(source / 'manifests' / name)
        write(manifests / name, value)
    for name in ('public', 'revisions', 'environments'):
        shutil.copytree(source / 'manifests' / name, manifests / name)
    protocol = read(source / 'manifests/protocol.json')
    protocol.update(version=target.name, status='REPAIR_CANDIDATE', ancestor=str(source),
                    collection_contract='paper200-runner-judge/v2', scoring='NOT_SCORED')
    write(manifests / 'protocol.json', protocol)
    write(target / 'reports/SCORING_READINESS.json', {
        'passed': False, 'status': 'REPAIR_IN_PROGRESS',
        'collection_complete': False, 'evidence_qualified': False,
        'statistically_eligible': False,
        'required': ['C1', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6',
                     'J1', 'J2', 'J3', 'J4', 'J5', 'J6', 'J7', 'J8', 'G1']})
    write(target / 'manifests/repair_lineage.json', {
        'created': now(), 'source_root': str(source),
        'source_freeze_sha256': sha(source / 'manifests/FROZEN.json'),
        'judge_source': str(Path(judge_source).resolve()),
        'copy_method': 'independent reflink or ordinary copy; no hard links',
        'pending_matrix': 'created only after source drains; completed v1 parked',
        'old_new_pairing': 'ineligible until matching contracts and formal initial state verified'})


def record_drain(source, target):
    """Snapshot queue consistently; refuse to call a live source drained."""
    source, target = Path(source).resolve(), Path(target).resolve()
    if not (source / 'STOP').is_file():
        raise ValueError('source dispatch stop marker missing')
    with sqlite3.connect('file:' + str(source / 'queue.sqlite3') + '?mode=ro', uri=True) as db:
        db.row_factory = sqlite3.Row
        rows = [dict(row) for row in db.execute('SELECT * FROM runs WHERE purpose IN ("main", "assisted")')]
    running = [row for row in rows if row['status'] == 'running']
    if running:
        return {'drained': False, 'running': len(running)}
    parked = [row for row in rows if row['status'] == 'collected']
    remaining = [row for row in rows if row['status'] != 'collected']
    path = target / 'manifests/source_drain.json'
    if path.exists():
        previous = read(path)
        if previous['parked'] != parked or previous['remaining'] != remaining:
            raise ValueError('source queue changed after frozen drain snapshot')
        return previous
    result = {'drained': True, 'observed': now(), 'source': str(source),
              'parked': parked, 'remaining': remaining, 'automatic_dispatch': False}
    write(path, result)
    return {'drained': True, 'parked': len(parked), 'remaining': len(remaining)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'drain'])
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--target', type=Path, required=True)
    parser.add_argument('--judge-source', type=Path)
    parser.add_argument('--repair-spec', type=Path)
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare(args.source, args.target, args.judge_source, args.repair_spec)
    else:
        print(record_drain(args.source, args.target))
