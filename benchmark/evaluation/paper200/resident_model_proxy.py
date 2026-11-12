"""Per-run small proxy preserving the Runner's original WebSocket protocol."""
import argparse,asyncio,json,sys,time
from pathlib import Path
from .common import read,write

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--port',type=int,required=True);p.add_argument('--resident-uri',required=True);a=p.parse_args();cfg=read(a.config)
    runtime=Path(cfg.get('runtime_snapshot',str(Path(cfg['collection_root'])/'runtime')))
    sys.path.insert(0,str(runtime/'baseline_code/pylibs'));sys.path.insert(0,str(runtime/'baseline_code'))
    from openpi_client import msgpack_numpy
    import websockets.asyncio.client as client
    import websockets.asyncio.server as server
    async def run():
        async with client.connect(a.resident_uri,max_size=64*1024**2,ping_interval=None) as remote:
            await remote.send(json.dumps({'op':'open_session','config':cfg,'out':str(a.out.resolve())}))
            metadata=await remote.recv();decoded=msgpack_numpy.unpackb(metadata);connected=False
            async def forward(local):
                nonlocal connected
                if connected:await local.close(code=1013,reason='one Runner connection per session');return
                connected=True
                try:
                    await local.send(metadata)
                    count=0
                    async for wire in local:
                        started=time.monotonic();await remote.send(wire);response=await remote.recv();elapsed=time.monotonic()-started;count+=1
                        with (a.out/'RESIDENT_RPC_METRICS.jsonl').open('a') as stream:stream.write(json.dumps({'request_id':count,'roundtrip_wall_s':elapsed,'request_bytes':len(wire),'response_bytes':len(response)})+'\n')
                        await local.send(response)
                finally:connected=False
            async with server.serve(forward,'127.0.0.1',a.port,compression=None,max_size=64*1024**2,ping_interval=None):
                write(a.out/'READY.json',decoded);await asyncio.Future()
    asyncio.run(run())
if __name__=='__main__':main()
