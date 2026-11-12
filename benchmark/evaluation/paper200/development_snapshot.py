"""Freeze executable source per repair pilot while the next candidate evolves."""
from pathlib import Path
import shutil
from .common import write,sha,now


def snapshot(root,out):
    target=out/'runtime_snapshot';target.mkdir(exist_ok=False)
    for name in ('runner','baseline_runner','agent','baseline_code','openpi'):
        source=root/'runtime'/name
        if source.exists():shutil.copytree(source,target/name,symlinks=True,
            ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    baseline=target/'baseline';baseline.mkdir()
    for name in ('config.json','ROBOT_MANIFEST.json'):shutil.copy2(root/'runtime/baseline'/name,baseline/name)
    for name in ('models','assets'):(baseline/name).symlink_to((root/'runtime/baseline'/name).resolve(),target_is_directory=True)
    (baseline/'code').symlink_to(target/'baseline_code',target_is_directory=True)
    files={str(p.relative_to(target)):sha(p) for p in sorted(target.rglob('*.py')) if not p.is_symlink()}
    write(out/'RUNTIME_SNAPSHOT.json',{'created':now(),'runtime_root':str(target),'files':files,
        'reason':'repair pilot source isolation; model weights and physical assets remain explicitly shared read-only inputs'})
    return target
