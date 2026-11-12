"""Start a versioned local vision service for v2 development acceptance."""
from pathlib import Path
import argparse,subprocess,socket,shutil,os
from .common import read,write,sha,now,PROJECT


def provision(root,gpu):
    path=root/'manifests/vision_service.json'
    previous=None;attempt=1
    if path.exists():
        previous=read(path)
        state=__import__('json').loads(subprocess.check_output(['docker','inspect','--format','{{json .State}}',previous['services'][0]['container']],encoding='utf-8'))
        if state['Running']:return previous
        attempt=previous.get('attempt',1)+1
        write(root/('reports/VISION_SERVICE_FAILED_%03d.json'%(attempt-1)),{'configuration':previous,'state':state})
    source_root=Path(read(root/'manifests/repair_lineage.json')['source_root'])
    image=read(source_root/'manifests/vision_service.json')['services'][0]['image']
    frozen=root/'runtime/vision_service_source'
    if not frozen.exists():shutil.copytree(root/'runtime/agent',frozen,symlinks=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    source_hashes={str(p.relative_to(frozen)):sha(p) for p in frozen.rglob('*.py')}
    with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
    cpus=sorted(os.sched_getaffinity(0));assigned=cpus[gpu*32:(gpu+1)*32] or cpus[:32]
    name=root.name+'-repair-vision-%d'%attempt;out=root/('pilot/vision_service_%d'%attempt);out.mkdir()
    command=['docker','run','-d','--name',name,'--gpus','device=%d'%gpu,'--network','host',
        '--user','1000:1000','--cpuset-cpus',','.join(map(str,assigned)),
        '-v',str(frozen)+':/agent:ro','-v',str(out)+':/output:rw',
        '-v',str(PROJECT/'mc/safe_agent_vision_models')+':/vision-models:ro',
        '-e','PYTHONPATH=/agent','-e','HF_HUB_OFFLINE=1','-e','TRANSFORMERS_OFFLINE=1',
        '-e','OMP_NUM_THREADS=4','-e','OPENBLAS_NUM_THREADS=4',
        '--entrypoint','/opt/groot-env/bin/python',image,'-m','safety_agent.vision_service',
        '--qwen','/vision-models/qwen','--sam','/vision-models/sam','--port',str(port),'--device','cuda:0','--output','/output/service']
    identifier=subprocess.check_output(command,encoding='utf-8').strip()
    pid=int(subprocess.check_output(['docker','inspect','--format','{{.State.Pid}}',identifier],encoding='utf-8'))
    item={'endpoint':'http://127.0.0.1:%d'%port,'container':name,'gpu':gpu,'image':image,
        'host_pid':pid,'reserved_peak_mib':14000,'created':now(),'status':'STARTING',
        'frozen_source':str(frozen),'source_sha256':source_hashes,'model_source':'shared verified vision weights read-only'}
    result={'services':[item],'endpoint':item['endpoint'],'attempt':attempt};write(path,result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--gpu',type=int,default=5)
    a=p.parse_args();print(provision(a.root.resolve(),a.gpu)['endpoint'])
