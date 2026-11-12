"""Setup-time provenance utilities, never used in the control loop."""
from pathlib import Path
import json
from .task_spec import sha256, write_json


def fetch_pinned_qwen_mirror(root):
    """Alternate transport only; trust the pinned official HF LFS SHA256, not the mirror branch."""
    import concurrent.futures
    import urllib.request
    root=Path(root)
    model=json.loads((root/'models.lock.json').read_text())['models']['qwen']
    for name, meta in model['files'].items():
        if not name.endswith('.safetensors'): continue
        if not meta['sha256']: raise ValueError('official LFS hash required for alternate transport')
        target=root/'qwen'/name
        if target.exists() and sha256(target)==meta['sha256']: continue
        directory=root/'.mirror_parts'/name
        directory.mkdir(parents=True,exist_ok=True)
        chunk=64*1024**2;size=meta['bytes']
        ranges=[(i,min(i+chunk,size)-1) for i in range(0,size,chunk)]
        def download(pair):
            start,end=pair;part=directory/f'{start:012d}.part'
            if part.exists() and part.stat().st_size==end-start+1: return part
            request=urllib.request.Request('https://modelscope.cn/models/Qwen/Qwen3-VL-4B-Instruct/resolve/master/'+name,
                                            headers={'Range':f'bytes={start}-{end}'})
            opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(request,timeout=120) as response:
                if response.status!=206 or response.headers.get('Content-Range')!=f'bytes {start}-{end}/{size}':
                    raise ValueError('mirror range/size mismatch')
                with part.with_suffix('.partial').open('wb') as out:
                    while True:
                        data=response.read(4*1024**2)
                        if not data:break
                        out.write(data)
            if part.with_suffix('.partial').stat().st_size!=end-start+1:raise ValueError('short mirror part')
            part.with_suffix('.partial').replace(part)
            return part
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            parts=list(pool.map(download,ranges))
        complete=target.with_suffix('.mirror-complete')
        with complete.open('wb') as out:
            for part in parts:
                with part.open('rb') as src:
                    for block in iter(lambda:src.read(8*1024**2),b''):out.write(block)
        if sha256(complete)!=meta['sha256']:raise ValueError('mirror does not match pinned official Qwen revision')
        complete.replace(target)
        write_json(directory/'VERIFIED.json',{'revision':model['revision'],'official_sha256':meta['sha256'],'bytes':size})


def freeze(paths, output):
    files = {}
    for name in paths:
        path = Path(name).resolve(strict=True)
        candidates = sorted(path.rglob('*')) if path.is_dir() else [path]
        for f in candidates:
            if not f.is_file() or any(p.startswith(('.venv','.git','__pycache__','.cache')) for p in f.parts):
                continue
            if f.name.startswith(('optimizer','rng_state','scheduler','training_args')):
                continue
            files[str(f)] = {'sha256':sha256(f), 'bytes':f.stat().st_size}
    result = {'schema':'safe-agent-dependency-freeze/v1', 'files':files}
    write_json(output,result)
    return result


def verify_local_vision(qwen, sam):
    lock_path=Path(qwen).parent/'models.lock.json'
    lock=json.loads(lock_path.read_text())
    for name,path in [('qwen',Path(qwen)),('sam',Path(sam))]:
        value=lock['models'][name]
        if value['status']!='VERIFIED':raise ValueError('vision model not fully verified: '+name)
        for filename,meta in value['files'].items():
            if sha256(path/filename)!=meta['sha256']:raise ValueError('vision model changed: '+filename)
    return {'models_lock_sha256':sha256(lock_path),
            'revisions':{k:v['revision'] for k,v in lock['models'].items()}}


def prepare_vision(root):
    from huggingface_hub import HfApi, snapshot_download
    root = Path(root)
    root.mkdir(parents=True,exist_ok=True)
    lock_path = root/'models.lock.json'
    lock = json.loads(lock_path.read_text()) if lock_path.exists() else {'schema':'safe-agent-vision-models/v1','models':{}}
    for name, repo in [('qwen','Qwen/Qwen3-VL-4B-Instruct'),('sam','facebook/sam2.1-hiera-tiny')]:
        if name not in lock['models']:
            info = HfApi().model_info(repo,files_metadata=True,timeout=30)
            lock['models'][name] = {'repo_id':repo,'revision':info.sha,'status':'PINNED',
                                   'files':{f.rfilename:{'bytes':f.size,'sha256':getattr(f.lfs,'sha256',None) if f.lfs else None}
                                            for f in info.siblings if f.rfilename.endswith(('.json','.safetensors','.txt','.model'))}}
            write_json(lock_path,lock)
        model = lock['models'][name]
        local = root/name
        missing=[name for name,meta in model['files'].items() if not (local/name).is_file()
                 or (meta['bytes'] is not None and (local/name).stat().st_size!=meta['bytes'])]
        if missing:
            snapshot_download(repo,revision=model['revision'],local_dir=local,allow_patterns=missing,max_workers=4)
        for filename, metadata in model['files'].items():
            path = local/filename
            if not path.is_file() or (metadata['bytes'] is not None and path.stat().st_size != metadata['bytes']):
                raise ValueError('incomplete model file: '+filename)
            digest = sha256(path)
            if metadata['sha256'] and digest != metadata['sha256']:
                raise ValueError('model integrity mismatch: '+filename)
            metadata['sha256'] = digest
        model.update(status='VERIFIED',path=str(local.resolve()))
        write_json(lock_path,lock)
    return lock
