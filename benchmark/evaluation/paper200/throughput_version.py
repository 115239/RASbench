"""Create a stopped independent throughput candidate and certify drained ancestry.

Runtime/configuration files are independent copies. Large benchmark assets are
shared by a declared read-only symlink; callers must copy them before editing.
This module never enqueues a formal run or removes STOP.
"""
import argparse
import json
import os
import shutil
import sqlite3
from pathlib import Path
from .common import now, read, sha, write


def source_snapshot(root):
    root = Path(root).resolve()
    if not (root/'STOP').is_file():
        raise ValueError('source must retain STOP')
    with sqlite3.connect('file:' + str(root/'queue.sqlite3') + '?mode=ro', uri=True) as db:
        db.row_factory = sqlite3.Row
        db.execute('BEGIN')
        rows = [dict(row) for row in db.execute('SELECT * FROM runs WHERE purpose IN ("main", "assisted") ORDER BY run_id')]
        running = db.execute('SELECT COUNT(*) FROM runs WHERE status="running"').fetchone()[0]
    return {'source_root': str(root), 'source_freeze_sha256': sha(root/'manifests/FROZEN.json'),
            'rows': rows, 'running': running}


def _copy_tree(source, target):
    shutil.copytree(source, target, symlinks=True,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.git'))


def prepare(source, target):
    source, target = Path(source).resolve(), Path(target).absolute()
    if not (source/'STOP').is_file():
        raise ValueError('source dispatch stop marker missing')
    if target.exists():
        raise FileExistsError('refusing to alter existing target: ' + str(target))
    if not (source/'manifests/FROZEN.json').is_file():
        raise ValueError('source frozen identity missing')
    target.mkdir(parents=True)
    (target/'STOP').write_text('Throughput candidate: dispatch stopped; source drain and release pending.\n')
    write(target/'REPAIR_IN_PROGRESS.json', {'created': now(), 'source': str(source),
          'status': 'THROUGHPUT_DEVELOPMENT_NOT_RELEASED', 'automatic_dispatch': False})
    _copy_tree(source/'runtime', target/'runtime')
    (target/'benchmark').mkdir()
    for entry in (source/'benchmark').iterdir():
        dest = target/'benchmark'/entry.name
        if entry.name == 'assets':
            dest.symlink_to(entry.resolve(), target_is_directory=True)
        elif entry.is_dir():
            _copy_tree(entry, dest)
        else:
            shutil.copy2(entry, dest, follow_symlinks=False)
    # Rewrite only symlinks into copied runtime/task files, leaving weights and
    # the explicitly shared asset tree pointed at their true original sources.
    retargeted = []
    for base in ('runtime', 'benchmark/task'):
        for directory, dirs, files in os.walk(str(target/base), followlinks=False):
            for name in dirs + files:
                path = Path(directory)/name
                if not path.is_symlink():
                    continue
                original = os.readlink(path)
                old_path = Path(original)
                if old_path.is_absolute() and str(old_path).startswith(str(source)+'/'):
                    suffix = old_path.relative_to(source)
                    if str(suffix).startswith(('runtime/', 'benchmark/task/')):
                        replacement = target/suffix
                        path.unlink()
                        path.symlink_to(replacement)
                        retargeted.append({'path': str(path.relative_to(target)), 'source': original, 'target': str(replacement)})
    ancestry = target/'ancestry'/source.name
    _copy_tree(source/'manifests', ancestry/'manifests')
    _copy_tree(source/'manifests', target/'manifests')
    # These are source release certificates/derived inventories, not target ones.
    excluded = ('FROZEN.json', 'files_final.json', 'file_stats_final.json', 'symlinks_final.json',
                'implementation_inventory.json', 'source_drain.json', 'resume_repaired.json',
                'repair_lineage.json', 'vision_service.json', 'release_acceptance_v2.json',
                'expedited_collection_policy.json')
    for name in excluded:
        (target/'manifests'/name).unlink(missing_ok=True)
    protocol = read(target/'manifests/protocol.json')
    protocol.update(version=target.name, created=now(), ancestor=str(source),
                    status='THROUGHPUT_CANDIDATE', readiness='DEVELOPMENT_REQUIRED',
                    score_status='NOT_SCORED', scoring='NOT_SCORED')
    write(target/'manifests/protocol.json', protocol)
    write(target/'manifests/repair_lineage.json', {
        'created': now(), 'source_root': str(source),
        'source_freeze_sha256': sha(source/'manifests/FROZEN.json'),
        'source_drain_sha256': sha(source/'manifests/source_drain.json'),
        'copy_method': 'ordinary independent runtime/task copies; explicitly read-only shared asset symlink',
        'shared_read_only_assets': [{'target': str(target/'benchmark/assets'), 'source': str((source/'benchmark/assets').resolve()),
                                    'edit_policy': 'copy to a target-owned file before changing any asset; never edit through symlink'}],
        'retargeted_symlinks': retargeted,
        'pending_matrix': 'created only after source drains; all collected ancestors parked for later review',
        'old_new_pairing': 'ineligible until matching contracts and formal initial state verified'})
    from .repair_contracts import relocate
    relocate(target)
    from .queue import initialize
    initialize(target, [], target.name)
    write(target/'reports/SCORING_READINESS.json', {'passed': False, 'status': 'THROUGHPUT_DEVELOPMENT',
          'real_scene_acceptance_passed': False, 'collection_release_passed': False,
          'collection_complete': False, 'statistically_eligible': False})
    write(target/'reports/MAIN_RELEASE_GATE.json', {'passed': False, 'status': 'NOT_RELEASED'})
    result = {'created': now(), 'source': str(source), 'target': str(target),
              'status': 'STOPPED_EMPTY_CANDIDATE', 'formal_runs_enqueued': 0,
              'source_drain_recorded': False, 'shared_assets_read_only': True}
    write(target/'reports/THROUGHPUT_VERSION_PREPARATION.json', result)
    return result


def record_drain(source, target):
    """Snapshot every source read-only; never certify a still-running source."""
    from .resume_repaired import drain_plan, verify_source, key
    source, target = Path(source).resolve(), Path(target).resolve()
    if not (target/'STOP').is_file():
        raise ValueError('target must retain STOP')
    lineage = read(target/'manifests/repair_lineage.json')
    if Path(lineage['source_root']).resolve() != source:
        raise ValueError('target ancestry differs from requested source')
    current = source_snapshot(source)
    if current['running']:
        return {'drained': False, 'running': current['running'], 'source': str(source)}
    previous = drain_plan(source)
    verify_source(source, previous)
    if {key(row) for row in current['rows']} != {key(row) for row in previous['remaining']}:
        raise ValueError('current source queue differs from inherited remainder')
    snapshots = previous.get('sources')
    if snapshots is None:
        snapshots = [source_snapshot(previous['source'])]
    snapshots = list(snapshots) + [current]
    parked = [dict(row, source_root=row.get('source_root', previous['source'])) for row in previous['parked']]
    parked += [dict(row, source_root=str(source)) for row in current['rows'] if row['status'] == 'collected']
    remaining = [dict(row, source_root=str(source)) for row in current['rows'] if row['status'] != 'collected']
    result = {'schema': 'paper200-source-drain/v2', 'drained': True, 'observed': now(),
              'source': str(source), 'sources': snapshots, 'parked': parked, 'remaining': remaining,
              'automatic_dispatch': False, 'old_records_accepted': False,
              'parked_by_source': {s['source_root']: sum(r['source_root'] == s['source_root'] for r in parked) for s in snapshots}}
    path = target/'manifests/source_drain.json'
    if path.exists():
        saved = read(path)
        if {k:v for k,v in saved.items() if k!='observed'} != {k:v for k,v in result.items() if k!='observed'}:
            raise ValueError('source chain changed after frozen drain snapshot')
        return saved
    # Validate in memory before persisting a certificate.
    from .resume_repaired import full_matrix
    flat = parked + remaining
    if len({key(row) for row in flat}) != len(flat) or {key(row) for row in flat} != full_matrix(target,'main') | full_matrix(target,'assisted'):
        raise ValueError('source chain does not partition the full experiment')
    verify_source(target, result)
    write(path, result)
    write(target/'reports/SOURCE_CHAIN_DRAIN.json', {'time': now(), 'drained': True,
          'parked_by_source': result['parked_by_source'],
          'remaining_main': sum(row['purpose']=='main' for row in remaining),
          'remaining_assisted': sum(row['purpose']=='assisted' for row in remaining),
          'automatic_dispatch': False})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'drain'])
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--target', required=True, type=Path)
    args = parser.parse_args()
    result = prepare(args.source, args.target) if args.action == 'prepare' else record_drain(args.source, args.target)
    print(json.dumps({k:v for k,v in result.items() if k not in ('parked','remaining','sources')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
