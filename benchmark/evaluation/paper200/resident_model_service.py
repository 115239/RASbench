"""One frozen weight bank per process, isolated ordered inference sessions.

Local-only transport. Inference is serialized; mutable adapters and all host/Torch
RNG state are restored per session. The original Runner wire stays unchanged.
"""
import argparse,asyncio,copy,hashlib,json,os,random,sys,time,traceback,uuid
from collections import deque
from pathlib import Path
import numpy as np
from .common import read,write
from .model_service import NeuralAudit,PiPolicy,flatten_raw_arrays,source_binding


def source_manifest(runtime,backend):
    runtime=Path(runtime)
    bases=['baseline_runner','baseline_code/grasp','baseline_code/navigation','baseline_code/Isaac-GR00T'] if backend=='groot' else ['openpi/src','openpi/packages/openpi-client/src']
    result={}
    for base in bases:
        for path in sorted((runtime/base).rglob('*.py')):
            result[str(path.relative_to(runtime))]=hashlib.sha256(path.read_bytes()).hexdigest()
    if not result:raise ValueError('RESIDENT_SOURCE_MANIFEST_EMPTY')
    return result


def snapshot_adapter(obj):
    # Functions close over the single loaded actor/route; swap their data, never
    # clone bound closures or a network. Unexpected mutable types fail closed.
    import types
    result={}
    for key,value in vars(obj).items():
        if key in ('policy','config','phase_route') or callable(value) or isinstance(value,types.ModuleType):continue
        if not isinstance(value,(str,int,float,bool,type(None),dict,list,tuple,np.ndarray)):
            raise TypeError('UNSUPPORTED_SESSION_ADAPTER_STATE: '+key+' '+type(value).__name__)
        result[key]=copy.deepcopy(value)
    return result


def restore_adapter(obj,state):
    current=snapshot_adapter(obj)
    for key in current:
        if key not in state:delattr(obj,key)
    for key,value in state.items():setattr(obj,key,copy.deepcopy(value))


class HostRNG:
    def __init__(self,seed,torch_enabled):
        self.torch_enabled=torch_enabled;self.python=random.Random(seed).getstate();self.numpy=np.random.RandomState(seed).get_state()
        if torch_enabled:
            import torch
            torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            self.cpu=torch.get_rng_state();self.cuda=torch.cuda.get_rng_state_all()
    def activate(self):
        random.setstate(self.python);np.random.set_state(self.numpy)
        if self.torch_enabled:
            import torch
            torch.set_rng_state(self.cpu);torch.cuda.set_rng_state_all(self.cuda)
    def capture(self):
        self.python=random.getstate();self.numpy=np.random.get_state()
        if self.torch_enabled:
            import torch
            self.cpu=torch.get_rng_state();self.cuda=torch.cuda.get_rng_state_all()


class ResidentBank:
    def __init__(self,cfg,out):
        self.cfg=cfg;self.out=Path(out);self.backend=cfg['backend'];self.active=None;self.sessions={}
        self.runtime=Path(cfg.get('runtime_snapshot',str(Path(cfg['collection_root'])/'runtime'))).resolve()
        self.sources=source_manifest(self.runtime,self.backend);self.instance_id=str(uuid.uuid4())
        self.total_calls=0;self.inference_wall_s=0.;self.started=time.monotonic()
        self.load_count=1
        self.code_identity={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).with_name('model_service.py'),Path(__file__).parents[1]/'live_models.py']}
        if cfg.get('resident_code_identity',self.code_identity)!=self.code_identity:raise ValueError('RESIDENT_BOOTSTRAP_CODE_DIFFERS')
        if self.backend=='groot':self._load_groot()
        else:self._load_pi()
        self.source_digest=hashlib.sha256(json.dumps(self.sources,sort_keys=True).encode()).hexdigest()
    def _load_groot(self):
        import torch
        sys.path.insert(0,str(self.runtime/'baseline_runner'))
        import model_server
        self.Router=model_server.Router
        self.loaded_sources={'router':source_binding(model_server,self.runtime/'baseline_runner/model_server.py')}
        from gr00t.model.gr00t_n1d7 import gr00t_n1d7,processing_gr00t_n1d7
        cosmos=os.environ['PAPER_BASELINE_ROOT']+'/models/nvidia/Cosmos-Reason2-2B'
        backbone=gr00t_n1d7.get_backbone_cls;processor=processing_gr00t_n1d7.build_processor
        def local_backbone(config):
            if str(config.model_name).endswith('Cosmos-Reason2-2B'):config.model_name=cosmos
            return backbone(config)
        gr00t_n1d7.get_backbone_cls=local_backbone
        processing_gr00t_n1d7.build_processor=lambda name,kw:processor(cosmos if str(name).endswith('Cosmos-Reason2-2B') else name,kw)
        config=copy.deepcopy(self.cfg);config['task']['stages']=['NAV_OUTBOUND','PICK']
        if not config.get('nav_model'):raise ValueError('RESIDENT_REQUIRES_CURRENT_UNIFIED_NAV_ROUTER')
        self.template=self.Router(config,self.out/'bootstrap')
        import contestant_adapter
        self.loaded_sources['contestant_adapter']=source_binding(contestant_adapter,self.runtime/'baseline_runner/contestant_adapter.py')
        self.actor=self.template.models['grasp'];self.route=self.actor.phase_route
        self.initial_actor=snapshot_adapter(self.actor);self.initial_route=snapshot_adapter(self.route) if self.route else None
        get_action=self.actor.policy.get_action
        def recorded_grasp(*args,**kwargs):
            session=self.active
            if session is None:raise RuntimeError('NEURAL_CALL_OUTSIDE_RESIDENT_SESSION')
            result=get_action(*args,**kwargs);path=session.out/('grasp_raw_%06d.npz'%session.count)
            np.savez_compressed(path,**flatten_raw_arrays(result[0]))
            session.audit.record_direct(path,('front','rear','wrist','state','prompt'),range(25),actual_model_prompt_sha256=getattr(self.actor,'last_model_prompt_sha256',None))
            return result
        self.actor.policy.get_action=recorded_grasp
        nav=self.template.models['unified'];score=nav.score
        def recorded_nav(*args,**kwargs):
            session=self.active
            if session is None:raise RuntimeError('NEURAL_CALL_OUTSIDE_RESIDENT_SESSION')
            result=score(*args,**kwargs);path=session.out/('nav_raw_%06d.npz'%session.count)
            np.savez_compressed(path,goals=result[0],routes=result[1])
            session.audit.record_navigation(list(session.model.history),path,len(kwargs.get('seeds',(20260921,20260922,20260923))))
            return result
        nav.score=recorded_nav
    def _load_pi(self):
        self.template=PiPolicy(self.cfg,self.out/'bootstrap')
        self.pi_policies={};self.pi_stats={}
        from openpi.policies.policy import Policy
        from openpi import transforms
        from openpi.models import tokenizer
        from openpi.shared import normalize
        from common import MODELS
        network=self.template.model.policy._model;sample=self.template.model.policy._sample_actions
        for mode,asset in [('vln','trossen_mobile'),('vla','trossen')]:
            stats=normalize.load(MODELS/'pi05_base/assets'/asset)
            policy=Policy(network,transforms=[transforms.Normalize(stats,use_quantiles=True),transforms.ResizeImages(224,224),transforms.TokenizePrompt(tokenizer.PaligemmaTokenizer(200),discrete_state_input=True),transforms.PadStatesAndActions(32)],output_transforms=[transforms.Unnormalize(stats,use_quantiles=True)])
            # Reuse one compiled sampling callable, not one JIT cache per session.
            policy._sample_actions=sample;self.pi_policies[mode]=policy;self.pi_stats[mode]=stats
    def open(self,cfg,out):
        if cfg.get('resident_code_identity',self.code_identity)!=self.code_identity:raise ValueError('RESIDENT_SESSION_CODE_DIFFERS')
        if cfg['backend']!=self.backend or cfg['model_bindings']!=self.cfg['model_bindings']:
            raise ValueError('RESIDENT_MODEL_BINDINGS_DIFFER')
        for key in ('nav_model','grasp_model','nav_out','nav_return'):
            if self.backend=='groot' and cfg.get(key)!=self.cfg.get(key):raise ValueError('RESIDENT_CHECKPOINT_CONFIG_DIFFERS: '+key)
        if Path(cfg['collection_root']).resolve()!=Path(self.cfg['collection_root']).resolve():raise ValueError('RESIDENT_COLLECTION_DIFFERS')
        runtime=Path(cfg.get('runtime_snapshot',str(Path(cfg['collection_root'])/'runtime'))).resolve()
        if source_manifest(runtime,self.backend)!=self.sources:raise ValueError('RESIDENT_RUNTIME_SOURCE_DIFFERS')
        out=Path(out).resolve();allowed=(Path(cfg['output']).resolve()/'model')
        if out!=allowed:raise ValueError('RESIDENT_OUTPUT_MUST_BE_SESSION_MODEL_DIRECTORY')
        if any(session.out==out for session in self.sessions.values()):raise ValueError('RESIDENT_OUTPUT_ALREADY_ACTIVE')
        if (out/'input_000001.msgpack').exists():raise ValueError('RESIDENT_REFUSES_EXISTING_RAW_SESSION')
        out.mkdir(parents=True,exist_ok=True);session=Session(self,cfg,out);self.sessions[session.id]=session
        return session


class Session:
    def __init__(self,bank,cfg,out):
        self.bank=bank;self.cfg=copy.deepcopy(cfg);self.out=out;self.id=str(uuid.uuid4());self.count=0
        self.audit=NeuralAudit(bank.backend,out);self.rng=HostRNG(int(cfg['experiment_seed']),bank.backend=='groot')
        if bank.backend=='groot':
            self.model=bank.Router.__new__(bank.Router);self.model.cfg=self.cfg;self.model.output=out
            self.model.models=bank.template.models;self.model.history=deque(maxlen=3);self.model.key=None;self.model.calls=0
            self.actor_state=copy.deepcopy(bank.initial_actor);self.route_state=copy.deepcopy(bank.initial_route)
        else:
            self.model=copy.copy(bank.template);self.model.cfg=self.cfg;self.model.out=out;self.model.model=copy.copy(bank.template.model)
            self.model.model.seed=int(cfg['experiment_seed']);self.model.model.calls=0
            self.pi_policies={mode:copy.copy(policy) for mode,policy in bank.pi_policies.items()}
        self.metadata=copy.deepcopy(bank.template.metadata);self.metadata['models']=cfg['model_bindings']
        self.metadata['resident_session']={'schema':'paper200-resident-session/v1','session_id':self.id,'daemon_instance_id':bank.instance_id,'serialized_inference':True,'weight_bank_load_count':bank.load_count,'source_runtime_root':str(bank.runtime),'requested_runtime_root':cfg.get('runtime_snapshot'),'runtime_source_sha256':bank.source_digest,'source_manifests_equal':True,'independent_seed':int(cfg['experiment_seed'])}
        if bank.backend=='groot':self.metadata['imported_model_sources']=bank.loaded_sources
        write(out/'RESIDENT_SESSION.json',self.metadata['resident_session'])
    def infer(self,data):
        from openpi_client import msgpack_numpy
        bank=self.bank
        if bank.active is not None:raise RuntimeError('RESIDENT_INFERENCE_NOT_SERIALIZED')
        obs=msgpack_numpy.unpackb(data);self.count+=1;started=time.monotonic();bank.active=self
        (self.out/('input_%06d.msgpack'%self.count)).write_bytes(data)
        self.rng.activate()
        if bank.backend=='groot':
            restore_adapter(bank.actor,self.actor_state)
            if bank.route:restore_adapter(bank.route,self.route_state)
        else:
            mode='vln' if obs['mission']['phase'].startswith('NAV_') else 'vla'
            self.model.mode=mode;self.model.model.mode=mode;self.model.model.policy=self.pi_policies[mode];self.model.model.stats=bank.pi_stats[mode]
        try:
            self.audit.begin(obs,self.count,data);response=self.model.infer(obs)
            if bank.backend=='pi05':self.audit.record_direct(self.out/('native_%06d.npz'%self.model.model.calls),('front','wrist','state','prompt'),[12,13,14,15,16,17,24],state_adapter='fixed joint sign map and Aloha gripper transform; other native channels use normalization mean')
            response['_paper_neural_receipt']=self.audit.finish(response)
            encoded=msgpack_numpy.packb(response);(self.out/('output_%06d.msgpack'%self.count)).write_bytes(encoded)
            write(self.out/('call_%06d.json'%self.count),{'sequence':self.count,'latency_s':time.monotonic()-started,'phase':obs.get('mission'),'prompt':obs.get('prompt'),'backend':bank.backend,'resident_session_id':self.id})
            return encoded
        finally:
            self.rng.capture();bank.total_calls+=1;bank.inference_wall_s+=time.monotonic()-started
            write(bank.out/'METRICS.json',{'sessions':len(bank.sessions),'total_calls':bank.total_calls,'inference_wall_s':bank.inference_wall_s,'uptime_wall_s':time.monotonic()-bank.started,'weight_bank_load_count':bank.load_count})
            if bank.backend=='groot':
                self.actor_state=snapshot_adapter(bank.actor)
                if bank.route:self.route_state=snapshot_adapter(bank.route)
            bank.active=None


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--port',type=int,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True);bank=ResidentBank(read(a.config),a.out)
    from openpi_client import msgpack_numpy
    import websockets.asyncio.server as ws
    async def handle(connection):
        session=None
        try:
            request=json.loads(await connection.recv())
            if request.get('op')!='open_session':raise ValueError('RESIDENT_FIRST_MESSAGE_MUST_OPEN_SESSION')
            session=bank.open(request['config'],request['out']);await connection.send(msgpack_numpy.packb(session.metadata))
            async for data in connection:
                if not isinstance(data,bytes):raise ValueError('RESIDENT_INFERENCE_MUST_BE_BINARY_WIRE')
                # No await inside inference: one GPU call owns all mutable state.
                await connection.send(session.infer(data))
        except Exception:
            if session:write(session.out/'ERROR.json',{'traceback':traceback.format_exc(),'request':session.count})
            else:write(a.out/('REGISTRATION_ERROR_'+str(time.time_ns())+'.json'),{'traceback':traceback.format_exc()})
            await connection.close(code=1011,reason='resident inference failed; retained error report')
        finally:
            if session:
                write(session.out/'RESIDENT_SESSION_CLOSED.json',{'session_id':session.id,'inference_requests':session.count});bank.sessions.pop(session.id,None)
    async def serve():
        async with ws.serve(handle,'127.0.0.1',a.port,compression=None,max_size=64*1024**2,ping_interval=None):
            write(a.out/'READY.json',{'backend':bank.backend,'daemon_instance_id':bank.instance_id,'runtime_source_sha256':bank.source_digest,'model_bindings':bank.cfg['model_bindings'],'weight_bank_load_count':1,'runtime':str(bank.runtime)})
            await asyncio.Future()
    asyncio.run(serve())
if __name__=='__main__':main()
