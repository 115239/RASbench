"""Frozen input identities and inexpensive per-episode mutation guards."""
from pathlib import Path
import shutil,os
from .common import read,write,sha,digest,now,BASE,PROJECT


def file_stat(path):
    s=Path(path).stat()
    return {'size':s.st_size,'mtime_ns':s.st_mtime_ns,'device':s.st_dev,'inode':s.st_ino}


def package_identity(root,question):
    package=root/'benchmark/task'/question
    files={str(p.relative_to(package)):sha(p) for p in sorted(package.rglob('*'))
           if p.is_file() and '__pycache__' not in p.parts}
    return {'files':files,'sha256':digest(files)}


def runtime_identity(root):
    files={str(p.relative_to(root)):sha(p) for p in sorted((root/'runtime').rglob('*'))
        if p.is_file() and not p.is_symlink() and p.suffix in ('.py','.json','.toml','.yaml','.yml','.sh')
        and '__pycache__' not in p.parts and 'collection_code' not in p.parts}
    return {'files':files,'sha256':digest(files)}


def scene_revision_identity(root,question):
    """Record asset-level revisions that are outside the per-question package."""
    files=[]
    if int(question[1:])>150:files.append(root/'manifests/revisions/material_dependencies.json')
    if int(question[1:])<=50:files.append(root/'manifests/revisions/fence_scene.json')
    return {str(p.relative_to(root)):sha(p) for p in files if p.exists()}


def snapshot_external_code(root):
    source=Path(os.environ['RASBENCH_OPENPI_SOURCE'])
    target=root/'runtime/openpi'
    if not target.exists():
        for relative in ['src','packages/openpi-client/src']:
            shutil.copytree(source/relative,target/relative,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        write(target/'SOURCE.json',{'source':str(source),'copied':now(),'network_weights':'unchanged original pi05_base'})


def verify_models(root):
    bindings=read(root/'manifests/models.json');records={};issues=[]
    for label,entry in bindings.items():
        base=Path(entry['path'])
        for relative,expected in entry['files'].items():
            p=base/relative if base.is_dir() else base
            actual=sha(p)
            if actual!=expected:issues.append({'path':str(p),'expected':expected,'actual':actual})
            records[str(p)]={'sha256':actual,**file_stat(p),'role':label}
    extras=[('groot_backbone',BASE/'baseline/models/nvidia/Cosmos-Reason2-2B'),
            ('vision',PROJECT/'mc/safe_agent_vision_models'),
            ('pi_tokenizer',Path(os.environ['RASBENCH_PI_TOKENIZER']))]
    for role,base in extras:
        paths=[base] if base.is_file() else sorted(p for p in base.rglob('*') if p.is_file() and '.cache' not in p.parts)
        for p in paths:records[str(p)]={'sha256':sha(p),**file_stat(p),'role':role}
    result={'time':now(),'passed':not issues,'issues':issues,'files':records}
    write(root/'manifests/model_files_verified.json',result)
    if issues:raise ValueError('model identity changed since preparation')
    return result


def verify_frozen(root):
    frozen=read(root/'manifests/FROZEN.json')
    if sha(root/'manifests/files_final.json')!=frozen['files_manifest_sha256']:
        raise ValueError('frozen file manifest changed')
    if sha(root/'manifests/file_stats_final.json')!=frozen['stats_manifest_sha256']:
        raise ValueError('frozen stat manifest changed')
    for relative,expected in read(root/'manifests/file_stats_final.json').items():
        if file_stat(root/relative)!=expected:raise ValueError('frozen input changed: '+relative)
    for relative,expected in read(root/'manifests/symlinks_final.json').items():
        path=root/relative
        if not path.is_symlink() or os.readlink(path)!=expected['target'] or str(path.resolve())!=expected['resolved']:
            raise ValueError('frozen symbolic binding changed: '+relative)
    for path,expected in read(root/'manifests/model_files_verified.json')['files'].items():
        if file_stat(path)!={k:expected[k] for k in ('size','mtime_ns','device','inode')}:
            raise ValueError('frozen model/dependency changed: '+path)
    return frozen
