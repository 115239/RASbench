"""Real model services, with per-episode RNG/history and raw outputs retained."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import sys
import time
import traceback
import hashlib
import numpy as np
from .common import read, write


def source_binding(module, expected):
    """Attest the imported file, not merely the configured import directory."""
    actual=Path(module.__file__).resolve();expected=Path(expected).resolve()
    if actual!=expected:
        raise ValueError('MODEL_SOURCE_SNAPSHOT_MISMATCH: %s != %s'%(actual,expected))
    return {'module':module.__name__,'actual_file':str(actual),'expected_file':str(expected),
            'sha256':hashlib.sha256(actual.read_bytes()).hexdigest(),'snapshot_path_matched':True}


class NeuralAudit:
    """Observer-only proof of neural calls and the wire samples they consumed."""
    fields={'front':'observation/m20_camera_0_rgb','rear':'observation/m20_camera_1_rgb',
            'wrist':'observation/piper_wrist_rgb','state':'observation/state','prompt':'prompt'}

    def __init__(self,backend,out):
        self.backend=backend;self.out=out;self.history={};self.current=None;self.forward=None

    @staticmethod
    def value_identity(value):
        if isinstance(value,str):payload=value.encode('utf-8');dtype='utf8';shape=[]
        else:
            array=np.ascontiguousarray(value);payload=array.tobytes();dtype=str(array.dtype);shape=list(array.shape)
        return {'value_sha256':hashlib.sha256(payload).hexdigest(),'dtype':dtype,'shape':shape}

    def begin(self,obs,call_id,wire):
        self.current={'call_id':int(call_id),'mission':dict(obs['mission']),
            'input_wire_sha256':hashlib.sha256(wire).hexdigest(),
            'sources':{name:{'request_id':int(call_id),'source_field':key,**self.value_identity(obs[key])}
                       for name,key in self.fields.items()}}
        self.history[call_id]=self.current;self.forward=None

    def record_direct(self,raw_path,modalities,state_indices,**details):
        sources={name:([self.current['sources'][name]] if name in modalities else []) for name in self.fields}
        self.forward={'sources':sources,'state_indices':list(state_indices),'raw_path':raw_path,**details}

    def record_navigation(self,history,raw_path,network_batches):
        sources={name:[] for name in self.fields}
        for time_s,image,_instruction in history:
            identity=self.value_identity(image);mission=self.current['mission']
            matches=[row for row in self.history.values()
                if row['mission']['time_s']==time_s and row['mission']['phase']==mission['phase']
                and row['mission']['epoch']==mission['epoch']
                and row['sources']['front']['value_sha256']==identity['value_sha256']]
            if len(matches)!=1:raise ValueError('AMBIGUOUS_NEURAL_NAVIGATION_FRAME_SOURCE')
            sources['front'].append(matches[0]['sources']['front'])
        sources['prompt']=[self.current['sources']['prompt']]
        self.forward={'sources':sources,'state_indices':[],'raw_path':raw_path,
                      'network_batches':int(network_batches),
                      'navigation_state_source':'public mission scene/phase/carrying encoding; 25-D robot state is not consumed'}

    def finish(self,response):
        used=self.forward;raw_path=Path(used['raw_path']) if used else None
        row={'schema':'paper200-neural-consumption/v1','call_id':self.current['call_id'],
             'request_id_namespace':'one ordered WebSocket inference request stream; must be checked against Runner request_id',
             'backend':self.backend,'phase':self.current['mission']['phase'],
             'service_decision':response.get('selection',{}).get('decision','ACTION_RETURNED'),
             'forward_executed':used is not None,
             'consumed_sources':used['sources'] if used else {name:[] for name in self.fields},
             'state_indices':used['state_indices'] if used else [],
             'input_wire_sha256':self.current['input_wire_sha256'],
             'raw_output_path':str(raw_path) if raw_path else None,
             'raw_output_sha256':hashlib.sha256(raw_path.read_bytes()).hexdigest() if raw_path else None,
             'collector_only':True}
        row['consumed_request_ids']={name:[item['request_id'] for item in values] for name,values in row['consumed_sources'].items()}
        if used:
            row.update({k:v for k,v in used.items() if k not in ('sources','state_indices','raw_path')})
        write(self.out/('neural_receipt_%06d.json'%row['call_id']),row)
        return row


def flatten_raw_arrays(raw):
    result={}
    def visit(value,key):
        if isinstance(value,dict):
            for child,item in value.items():visit(item,key+'/'+str(child) if key else str(child))
        else:
            if hasattr(value,'detach'):value=value.detach().cpu().numpy()
            array=np.asarray(value)
            if array.dtype.kind=='O':raise ValueError('unsupported object in neural raw output')
            result[key]=array
    visit(raw,'')
    return result


class PiPolicy:
    def __init__(self,cfg,out):
        # The existing transfer adapter loads original Orbax weights. Both modes
        # share a single network rather than loading a second 12-GB checkpoint.
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
        from live_models import Pi05
        from common import MODELS
        actual_weights=(MODELS/'pi05_base').resolve()
        declared_weights=Path(cfg['model_bindings']['pi05']['path']).resolve()
        if actual_weights!=declared_weights:
            raise ValueError('PI05_LOADED_WEIGHT_PATH_DIFFERS_FROM_DECLARED_IDENTITY')
        self.cfg,self.out=cfg,out
        self.mode='vln' if cfg['task']['stages'][0].startswith('NAV_') else 'vla'
        self.model=Pi05(self.mode,cfg['experiment_seed'])
        self.metadata={'state_dim':25,'action_dim':10,'backend':'pi05','mapping':'documented_M20_transfer_v1',
                       'models':cfg['model_bindings'],'transfer_adapter':cfg.get('transfer_adapter'),
                       'actual_pi05_weight_root':str(actual_weights),
                       'weight_identity_path_matched':True}

    def infer(self,obs):
        # Mixed mobile-manipulation switches normalization, retaining weights.
        mode='vln' if obs['mission']['phase'].startswith('NAV_') else 'vla'
        if mode!=self.mode:
            from openpi.policies.policy import Policy
            from openpi import transforms
            from openpi.models import tokenizer
            from openpi.shared import normalize
            from common import MODELS
            stats=normalize.load(MODELS/'pi05_base/assets'/('trossen_mobile' if mode=='vln' else 'trossen'))
            # Policy stores the network in _model; rebuild transforms only.
            model=self.model.policy._model
            self.model.policy=Policy(model,transforms=[transforms.Normalize(stats,use_quantiles=True),
                transforms.ResizeImages(224,224),transforms.TokenizePrompt(tokenizer.PaligemmaTokenizer(200),discrete_state_input=True),
                transforms.PadStatesAndActions(32)],output_transforms=[transforms.Unnormalize(stats,use_quantiles=True)])
            self.model.stats=stats;self.model.mode=mode;self.mode=mode
        response,raw,native=self.model.infer({**obs,'time_s':obs['mission']['time_s']})
        if 'joint_delta' in response:
            jacobian=np.asarray(obs['observation/arm_jacobian'],np.float32)
            if jacobian.shape!=(6,6) or not np.isfinite(jacobian).all():raise ValueError('invalid public arm Jacobian')
            delta=jacobian @ np.asarray(response.pop('joint_delta'),np.float32)
            response['actions'][0,3:6]=np.clip(delta[:3],-.02,.02)
            response['actions'][0,6:9]=np.clip(delta[3:],-.08,.08)
        number=self.model.calls
        np.savez_compressed(self.out/('native_%06d.npz'%number),**raw,**{'input_'+k:v for k,v in native.items()})
        response.update(phase=obs['mission']['phase'],epoch=obs['mission']['epoch'],
                        sample_time_s=obs['mission']['time_s'],models=self.cfg['model_bindings'])
        return response


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--port',type=int,required=True)
    a=p.parse_args();cfg=read(a.config);a.out.mkdir(parents=True,exist_ok=True)
    audit=NeuralAudit(cfg['backend'],a.out)
    if cfg['backend']=='groot':
        import torch
        torch.manual_seed(cfg['experiment_seed']);torch.cuda.manual_seed_all(cfg['experiment_seed'])
        runtime_root=Path(cfg.get('runtime_snapshot',str(Path(cfg['collection_root'])/'runtime')))
        sys.path.insert(0,str(runtime_root/'baseline_runner'))
        from model_server import Router
        import model_server
        model_sources={'router':source_binding(model_server,runtime_root/'baseline_runner/model_server.py')}
        from gr00t.model.gr00t_n1d7 import gr00t_n1d7, processing_gr00t_n1d7
        cosmos=os.environ['PAPER_BASELINE_ROOT']+'/models/nvidia/Cosmos-Reason2-2B'
        old_backbone=gr00t_n1d7.get_backbone_cls
        def backbone(config):
            if str(config.model_name).endswith('Cosmos-Reason2-2B'):config.model_name=cosmos
            return old_backbone(config)
        gr00t_n1d7.get_backbone_cls=backbone
        old_processor=processing_gr00t_n1d7.build_processor
        processing_gr00t_n1d7.build_processor=lambda name,kw:old_processor(cosmos if str(name).endswith('Cosmos-Reason2-2B') else name,kw)
        model=Router(cfg,a.out)
        if 'grasp' in model.models:
            import contestant_adapter
            model_sources['contestant_adapter']=source_binding(contestant_adapter,
                runtime_root/'baseline_runner/contestant_adapter.py')
            actor=model.models['grasp'];get_action=actor.policy.get_action
            def recorded_grasp(*args,**kwargs):
                result=get_action(*args,**kwargs)
                raw_path=a.out/('grasp_raw_%06d.npz'%audit.current['call_id'])
                np.savez_compressed(raw_path,**flatten_raw_arrays(result[0]))
                audit.record_direct(raw_path,('front','rear','wrist','state','prompt'),range(25),
                    actual_model_prompt_sha256=getattr(actor,'last_model_prompt_sha256',None))
                return result
            actor.policy.get_action=recorded_grasp
        model.metadata['imported_model_sources']=model_sources
        model.metadata['source_runtime_root']=str(runtime_root.resolve())
        torch.manual_seed(cfg['experiment_seed']);torch.cuda.manual_seed_all(cfg['experiment_seed'])
        # Capture raw semantic logits, not only the decoder's chosen goal.
        nav=model.models.get('unified')
        if nav:
            score=nav.score
            def recorded_score(*args,**kwargs):
                result=score(*args,**kwargs)
                raw_path=a.out/('nav_raw_%06d.npz'%(model.calls+1))
                np.savez_compressed(raw_path,goals=result[0],routes=result[1])
                audit.record_navigation(list(model.history),raw_path,len(kwargs.get('seeds',(20260921,20260922,20260923))))
                return result
            nav.score=recorded_score
    else:
        model=PiPolicy(cfg,a.out)
    from openpi_client import msgpack_numpy
    import websockets.asyncio.server as ws
    count=0
    async def handle(connection):
        nonlocal count
        await connection.send(msgpack_numpy.packb(model.metadata))
        async for data in connection:
            count+=1;obs=msgpack_numpy.unpackb(data);started=time.monotonic()
            (a.out/('input_%06d.msgpack'%count)).write_bytes(data)
            try:
                audit.begin(obs,count,data)
                result=model.infer(obs)
                if cfg['backend']=='pi05':
                    audit.record_direct(a.out/('native_%06d.npz'%model.model.calls),
                        ('front','wrist','state','prompt'),[12,13,14,15,16,17,24],
                        state_adapter='fixed joint sign map and Aloha gripper transform; other native channels use normalization mean')
                result['_paper_neural_receipt']=audit.finish(result)
                # Lossless wire payload includes model-consumed RGB/state/prompt.
                encoded=msgpack_numpy.packb(result)
                (a.out/('output_%06d.msgpack'%count)).write_bytes(encoded)
                write(a.out/('call_%06d.json'%count),{'sequence':count,'latency_s':time.monotonic()-started,
                      'phase':obs.get('mission'),'prompt':obs.get('prompt'),'backend':cfg['backend']})
                await connection.send(encoded)
            except Exception:
                write(a.out/'ERROR.json',{'traceback':traceback.format_exc(),'request':count})
                await connection.close(code=1011,reason='inference failed; see retained ERROR.json')
                return
    async def serve():
        async with ws.serve(handle,'127.0.0.1',a.port,compression=None,max_size=64*1024**2,ping_interval=None):
            write(a.out/'READY.json',model.metadata)
            await asyncio.Future()
    asyncio.run(serve())


if __name__=='__main__':main()
