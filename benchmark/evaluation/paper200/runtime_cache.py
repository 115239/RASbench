"""Private copy-on-write Kit caches; retain experiment evidence and logs."""
import fcntl,os,shutil,subprocess
from pathlib import Path
from .common import read,write,now


def prepare_worker_cache(root,worker_id,identity,template_scene=None):
    """One exclusive writable cache per v3 worker, seeded once from a base.

    ext4 has no reflink here. Never silently copy the 20 GB texture template
    per episode, and never hardlink writable cache files back into evidence.
    """
    from .common import digest
    root=Path(root).resolve()
    if root.name=='collection_20260924_v2':raise ValueError('v3 worker cache cannot modify frozen v2')
    if not worker_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in worker_id):
        raise ValueError('invalid worker cache identifier')
    key=digest(identity);directory=root/'.cache/resident_workers'/worker_id
    directory.mkdir(parents=True,exist_ok=True)
    with (directory/'prepare.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        ready=directory/'READY.json'
        if ready.exists():
            if read(ready)['identity_sha256']!=key:raise ValueError('resident cache source identity changed')
            return directory
        for name in ('cache','config','tmp','kit'):(directory/name).mkdir(exist_ok=True)
        seeded=[]
        if template_scene:
            source=root/'.cache/texture_templates'/template_scene/'cache'
            if (source.parent/'READY.json').exists():
                destination=directory/'cache/ov/texturecache';destination.parent.mkdir(parents=True,exist_ok=True)
                if destination.exists():raise ValueError('partial worker texture seed requires inspection')
                subprocess.run(['cp','-a',str(source),str(destination)],check=True)
                seeded.append({'read_only_source':str(source),'private_worker_copy':str(destination)})
        write(ready,{'time':now(),'worker_id':worker_id,'identity_sha256':key,'identity':identity,
            'seeded_once':seeded,'shared_writable_cache':False,'per_episode_bytes_copied':0,
            'isolation':'exclusive resident worker, sequential episodes; no writable hardlinks or shared cache mutation'})
    return directory


def prepare(root,out,scene):
    if os.environ.get('PAPER200_RESIDENT_ACTIVE')=='1':
        write(out/'CACHE_SEED.json',{'time':now(),'mode':'exclusive_resident_worker_cache',
            'cache':os.environ['XDG_CACHE_HOME'],'per_episode_bytes_copied':0,'shared_writable_cache':False})
        return
    template=root/'.cache/kit_templates'/scene
    destination=out/'episode/.kit/cache'
    if (template/'READY.json').exists():
        destination.parent.mkdir(parents=True,exist_ok=True)
        subprocess.run(['cp','-a','--reflink=auto',str(template/'cache'),str(destination)],check=True)
        write(out/'CACHE_SEED.json',{'template':str(template),'private_cache':str(destination),'time':now()})
    texture=root/'.cache/texture_templates'/scene
    destination=out/'private_runtime/cache/ov/texturecache'
    if (texture/'READY.json').exists() and not destination.exists():
        destination.parent.mkdir(parents=True,exist_ok=True)
        subprocess.run(['cp','-a','--reflink=auto',str(texture/'cache'),str(destination)],check=True)
        write(out/'TEXTURE_CACHE_SEED.json',{'template':str(texture),'private_cache':str(destination),
            'time':now(),'shared_writable_cache':False,'source':read(texture/'READY.json')})


def seed_textures(root,out,scene):
    source=out/'private_runtime/cache/ov/texturecache'
    if not source.is_dir():return
    if not read(out/'PROCESS_CLEANUP.json').get('completed') or not (out/'COLLECTION_VALIDATION.json').exists():
        raise ValueError('texture template requires a completed, validated source')
    from .provenance import scene_revision_identity
    job=read(out/'job.json');identity=read(out/'input_identity.json')
    if identity.get('scene_revisions',{})!=scene_revision_identity(root,job['question']):return
    template=root/'.cache/texture_templates'/scene;template.parent.mkdir(parents=True,exist_ok=True)
    with (template.parent/(scene+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if (template/'READY.json').exists():return
        if source.is_symlink() or any(p.is_symlink() for p in source.rglob('*')):raise ValueError('linked texture cache')
        template.mkdir(exist_ok=True)
        if (template/'cache').exists():raise ValueError('incomplete texture template needs inspection')
        subprocess.run(['cp','-a','--reflink=auto',str(source),str(template/'cache')],check=True)
        write(template/'READY.json',{'source_attempt':str(out),'time':now(),'scene_revisions':identity.get('scene_revisions'),
            'semantics':'renderer-generated source-texture acceleration only; each episode receives a private copy'})


def finish_kit(root,out,scene):
    out=out.resolve();out.relative_to(root.resolve())
    if not read(out/'PROCESS_CLEANUP.json').get('completed'):raise ValueError('cannot archive cache before process cleanup')
    if not (out/'COLLECTION_VALIDATION.json').exists():return None
    cache=out/'episode/.kit/cache'
    if not cache.is_dir():return None
    if cache.is_symlink() or any(p.is_symlink() for p in cache.rglob('*')):
        raise ValueError('unexpected symlink in private generated cache')
    template=root/'.cache/kit_templates'/scene;template.parent.mkdir(parents=True,exist_ok=True)
    with (template.parent/(scene+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if not (template/'READY.json').exists():
            template.mkdir(exist_ok=True)
            if (template/'cache').exists():raise ValueError('incomplete template needs inspection')
            subprocess.run(['cp','-a','--reflink=auto',str(cache),str(template/'cache')],check=True)
            write(template/'READY.json',{'source_attempt':str(out),'time':now(),'disposable_generated_cache':True})
    manifest=[{'path':str(p.relative_to(cache)),'bytes':p.stat().st_size} for p in cache.rglob('*') if p.is_file()]
    write(out/'CACHE_DISPOSAL.json',{'time':now(),'exact_path':str(cache),'files':manifest,
        'logical_bytes':sum(f['bytes'] for f in manifest),'template':str(template),
        'preserved':'all HDF5, videos, raw model/attack/physical records, Kit logs and settings'})
    shutil.rmtree(cache)
    (out/'SIZE_MEASUREMENT.json').unlink(missing_ok=True)
    return {'attempt':str(out),'logical_bytes':sum(f['bytes'] for f in manifest)}


def finish(root,out,scene):
    kit=finish_kit(root,out,scene)
    cache=out/'private_runtime/cache'
    if not cache.is_dir():return kit
    if not read(out/'PROCESS_CLEANUP.json').get('completed') or not (out/'COLLECTION_VALIDATION.json').exists():
        return kit
    info=read(out/'processes.json')
    for key in ['runner','model']:
        cmd=Path('/proc/%s/cmdline'%info.get(key))
        if cmd.exists() and str(out).encode() in cmd.read_bytes():raise ValueError('cache still owned by live process')
    if cache.is_symlink() or any(p.is_symlink() for p in cache.rglob('*')):
        raise ValueError('unexpected symlink in generated XDG cache')
    allowed={'ov','cuda','jax','warp','torch','matplotlib'}
    if {p.name for p in cache.iterdir()}-allowed:raise ValueError('unknown private cache content')
    if (out/'input_identity.json').exists():seed_textures(root,out,scene)
    manifest=[{'path':str(p.relative_to(cache)),'bytes':p.stat().st_size} for p in cache.rglob('*') if p.is_file()]
    amount=sum(x['bytes'] for x in manifest)
    write(out/'XDG_CACHE_DISPOSAL.json',{'time':now(),'exact_path':str(cache),'files':manifest,
        'logical_bytes':amount,'reason':'disposable generated renderer texture and framework compilation caches',
        'preserved':'all raw evidence, models, HDF5, MP4, configurations, failed traces and logs'})
    shutil.rmtree(cache)
    (out/'SIZE_MEASUREMENT.json').unlink(missing_ok=True)
    return {'attempt':str(out),'logical_bytes':amount+(kit or {}).get('logical_bytes',0)}


def completed(root):
    from .queue import connect
    rows={q['question']:q for q in read(root/'manifests/questions.json')};result=[]
    with connect(root/'queue.sqlite3') as db:
        jobs=[dict(j) for j in db.execute("SELECT * FROM runs WHERE status='collected'")]
    for job in jobs:
        out=Path(job['output'])
        info=read(out/'processes.json')
        if any(Path('/proc/%s/cmdline'%info.get(k)).exists() and str(out).encode() in Path('/proc/%s/cmdline'%info[k]).read_bytes()
               for k in ['launcher','runner','model']):continue
        value=finish(root,out,rows[job['question']]['scene'])
        if value:result.append(value)
    write(root/'reports/CACHE_DISPOSAL.json',{'time':now(),'attempts':result});return result
