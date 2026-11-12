"""Real frozen-network replay of preserved wire inputs, without new physics."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import msgpack
from .common import NATIVE,PROJECT,read,write
from .launch import runner_environment,stop_group,interpreter_environment


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--case',choices=['groot_vla','groot_nav','pi05'],required=True)
    p.add_argument('--gpu',type=int,default=4);p.add_argument('--name',default='neural_receipts_001')
    a=p.parse_args();root=a.root.resolve();out=root/'pilot/controls'/a.name/a.case
    out.mkdir(parents=True,exist_ok=False)
    source=(root/'pilot/groot_agent_off/Q001/state_feedback_01/seed_72026001/attempt_002' if a.case=='groot_nav'
        else root/'pilot/controls/Q95'/('pi05_agent_off' if a.case=='pi05' else 'groot_agent_off')/'control_002')
    cfg=read(source/'config.json');runtime=Path(cfg['runtime_snapshot']);write(out/'config.json',cfg)
    env=runner_environment(root,runtime);env['CUDA_VISIBLE_DEVICES']=str(a.gpu)
    if a.case=='pi05':
        python=os.environ['RASBENCH_OPENPI_PYTHON']
        env['PYTHONPATH']=':'.join(map(str,[PROJECT/'experiment',runtime/'openpi/src',runtime/'openpi/packages/openpi-client/src']))
        env.update(XLA_PYTHON_CLIENT_PREALLOCATE='false',XLA_PYTHON_CLIENT_MEM_FRACTION='.30')
    else:
        env=interpreter_environment(env,'model')
        python=str(NATIVE/'python.sh')
        env['PYTHONPATH']=':'.join(map(str,[PROJECT/'experiment',runtime/'baseline_runner',runtime/'baseline_code/Isaac-GR00T',runtime/'baseline_code/pylibs']))+':'+env['PYTHONPATH']
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    with (out/'model.log').open('w') as log:
        process=subprocess.Popen([python,'-m','paper200.model_service','--config',str(out/'config.json'),
            '--out',str(out/'model'),'--port',str(port)],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            deadline=time.monotonic()+300
            while not (out/'model/READY.json').exists():
                if process.poll() is not None:raise RuntimeError('real model load failed; see retained log')
                if time.monotonic()>deadline:raise RuntimeError('real model load timeout')
                time.sleep(.5)
            async def replay():
                import sys
                sys.path.insert(0,str(runtime/'baseline_code/pylibs'))
                import websockets.asyncio.client as ws
                receipts=[]
                async with ws.connect('ws://127.0.0.1:%s'%port,max_size=64*1024**2,ping_interval=None) as client:
                    await client.recv()
                    for number in range(1,4 if a.case=='groot_nav' else 2):
                        wire=(source/'model'/('input_%06d.msgpack'%number)).read_bytes()
                        await client.send(wire);response=msgpack.unpackb(await client.recv(),raw=False)
                        receipts.append(response['_paper_neural_receipt'])
                return receipts
            receipts=asyncio.run(replay())
            if a.case=='groot_nav':
                passed=([r['forward_executed'] for r in receipts]==[False,False,True]
                    and receipts[-1]['consumed_request_ids']['front']==[1,2,3]
                    and receipts[-1]['consumed_request_ids']['state']==[])
            else:
                passed=receipts[0]['forward_executed'] and receipts[0]['consumed_request_ids']['front']==[1]
            write(out/'NEURAL_RECEIPT_CHECK.json',{'passed':passed,'case':a.case,'receipts':receipts,
                'wire_source_attempt':str(source),'scope':'actual model loaded and actual inference on replayed real observations; no new physical trial',
                'ready':read(out/'model/READY.json')})
            print(a.case,'NEURAL_RECEIPT_CHECK',passed,flush=True)
        finally:stop_group(process)


if __name__=='__main__':main()
