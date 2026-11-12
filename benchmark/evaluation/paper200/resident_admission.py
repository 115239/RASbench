"""Charge shared model/Kit daemons once and reserve only new private work."""
from pathlib import Path
from .common import read
PROXY_PEAK_MIB=1024
KIT_PEAK_MIB=18000


def live_record(path,kind):
    try:
        value=read(path);pid=int(value['pid']);base=Path('/proc')/str(pid)
        raw=(base/'stat').read_text();token=raw[raw.rfind(')')+2:].split()[19]
        if value.get('process_start_token') and value['process_start_token']!=token:return None
        command=(base/'cmdline').read_bytes().decode(errors='replace')
        if kind=='model' and 'paper200.resident_model_service' not in command:return None
        if kind=='kit' and 'paper200.resident_scene' not in command:return None
        if value.get('status') in ('retired','exited'):return None
        return value
    except (OSError,KeyError,TypeError,ValueError):return None


def reservations(root,memories,gpus):
    from .scheduler import descendants
    uuids={g['index']:g['uuid'] for g in gpus};plan=read(root/'manifests/resident_services.json')
    result=[]
    for kind,paths in [('model',(root/'services/models').glob('*/process.json')),('kit',(root/'.resident').glob('*/WORKER.json'))]:
        for path in paths:
            item=live_record(path,kind)
            if item is None:continue
            gpu=int(item['gpu']);pids=descendants(item['pid']);used=sum(m for p,u,m in memories if p in pids and u==uuids[gpu])
            if kind=='model':
                cfg=plan.get('backends',{}).get(item['backend'],{});peak=cfg.get('reserved_peak_mib',24000 if item['backend']=='pi05' else 22000)
            else:peak=item.get('reserved_peak_mib',KIT_PEAK_MIB)
            result.append({'gpu':gpu,'pid':item['pid'],'role':'resident_'+kind,'predicted_peak_mib':max(peak,int(used*1.1)+512),
                           'current_mib':used,'worker_id':item.get('worker_id'),'scene':item.get('scene'),
                           'status':item.get('status'),'active_output':item.get('active_output'),'source':str(path)})
    return result


def choose(gpus,active,services,scene,ordinary_peak,compatible_worker_ids=None):
    from .scheduler import admission
    reserved_ids={a.get('resident_reuse_worker_id') for a in active}
    choices=[]
    for gpu in gpus:
        per_gpu=[x for x in active+services if x['gpu']==gpu['index']]
        idle=[s for s in services if s.get('role')=='resident_kit' and s['gpu']==gpu['index'] and s.get('scene')==scene and s.get('status')=='idle' and s.get('worker_id') not in reserved_ids and (compatible_worker_ids is None or s.get('worker_id') in compatible_worker_ids)]
        candidate=min(idle,key=lambda s:s['worker_id']) if idle else None
        peak=PROXY_PEAK_MIB if candidate else max(KIT_PEAK_MIB,ordinary_peak)+PROXY_PEAK_MIB
        headroom=admission(gpu['total_mib']-gpu['used_mib'],peak,per_gpu)
        if headroom>=0:choices.append((int(candidate is not None),headroom,gpu['index'],peak,candidate['worker_id'] if candidate else None))
    if not choices:return None,None,None
    _,_,gpu,peak,worker=max(choices)
    return gpu,peak,worker
