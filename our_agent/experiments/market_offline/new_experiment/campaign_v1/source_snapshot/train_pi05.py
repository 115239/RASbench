import os,sys,pathlib,json,time,dataclasses,argparse,functools
R=pathlib.Path(__file__).resolve().parents[2];SRC=R/'restored/openpi'
sys.path[:0]=[str(SRC/'src'),str(SRC/'packages/openpi-client/src'),str(SRC/'scripts')]
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false');os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION','0.19');os.environ.setdefault('WANDB_MODE','disabled')
from openpi.training import checkpoints
import jax,jax.numpy as jnp,flax.nnx as nnx,numpy as np,optax,sentencepiece
from openpi.training import config as tc,optimizer,weight_loaders,sharding
from openpi.models import pi0_config,model as model_module
from openpi.shared import nnx_utils
import train as upstream
import phase_loss;phase_loss.install()
from common_data import Data
p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=20261111);p.add_argument('--steps',type=int,default=20);p.add_argument('--plan-steps',type=int,default=2000);p.add_argument('--lr',type=float,default=1e-5);p.add_argument('--batch',type=int,default=4);p.add_argument('--output',type=pathlib.Path,required=True);p.add_argument('--eval-params',type=pathlib.Path);p.add_argument('--eval-split',default='validation');p.add_argument('--eval-limit',type=int,default=0);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False)
assert len(jax.devices())==1 and jax.devices()[0].platform=='gpu'
ds=Data(a.seed);sp=sentencepiece.SentencePieceProcessor(model_proto=(R/'assets/paligemma_tokenizer.model').read_bytes());tokens_max=768
cfg=tc.TrainConfig(name='rebuttal_pi05',exp_name=str(a.seed),model=pi0_config.Pi0Config(pi05=True,action_dim=32,action_horizon=15,max_token_len=tokens_max,paligemma_variant='gemma_2b_lora',action_expert_variant='gemma_300m_lora'),batch_size=a.batch,num_workers=0,num_train_steps=a.plan_steps,seed=a.seed,lr_schedule=optimizer.CosineDecaySchedule(warmup_steps=100,peak_lr=a.lr,decay_steps=a.plan_steps,decay_lr=0.0),optimizer=optimizer.AdamW(clip_gradient_norm=1,b2=0.999,weight_decay=0.0),ema_decay=None,freeze_filter=nnx.Not(nnx.Any(nnx_utils.PathRegex('.*lora.*'),nnx_utils.PathRegex('(action_.*|time_mlp_.*|state_proj)/.*'))),weight_loader=weight_loaders.CheckpointWeightLoader(str(R/'assets/pi05_base/params')),wandb_enabled=False)
mesh=sharding.make_mesh(1);rng=jax.random.key(a.seed)
print('INITIALIZING',flush=True)
with sharding.set_mesh(mesh):state,_=upstream.init_train_state(cfg,rng,mesh,resume=False)
jax.block_until_ready(state);print('INITIALIZED',flush=True)
(a.output/'config.json').write_text(json.dumps(dict(seed=a.seed,steps=a.steps,plan_steps=a.plan_steps,lr=a.lr,batch=a.batch,normalization='train-only minmax, constants masked',trainable_parameters=sum(x.size for x in jax.tree.leaves(state.params.filter(cfg.trainable_filter))),initialization='pi05_base; fresh LoRA and action projection adaptation',split=json.loads((R/'new_experiment/DATA_SPLIT.json').read_text())),indent=2))
def norm(x,key):
 st=ds.stats[key];low=np.array(st['min']);span=np.array(st['max'])-low
 return np.where(span<1e-6,0,(x-low)/np.maximum(span,1e-6)*2-1).astype(np.float32)
def encode(r):
 s=norm(r['state'],'state');bins=np.digitize(s,np.linspace(-1,1,257)[:-1])-1
 prompt=r['prompt'].strip().replace('_',' ').replace('\n',' ');full=f"Task: {prompt}, State: {' '.join(map(str,bins))};\nAction: "
 t=sp.encode(full,add_bos=True);assert len(t)<=tokens_max,('TOKEN_OVERFLOW',len(t))
 images={target:r['images'][src].astype(np.float32)[None]/127.5-1 for src,target in [('image','base_0_rgb'),('wrist_image','left_wrist_0_rgb'),('rear_image','right_wrist_0_rgb')]}
 obs=model_module.Observation.from_dict(dict(image=images,image_mask={k:np.ones(1,bool) for k in images},state=np.pad(s,(0,7))[None],tokenized_prompt=np.pad(t,(0,tokens_max-len(t))).astype(np.int32)[None],tokenized_prompt_mask=(np.arange(tokens_max)<len(t))[None]))
 n=len(r['actions']);act=np.zeros((1,15,32),np.float32);act[0,:n,:10]=norm(r['actions'],'actions');w=np.zeros_like(act);w[0,:n,:10]=1;w[0,:n,9]=3
 return jax.tree.map(jnp.asarray,(obs,act,w))
def sample(i):return encode(ds.sample(i))
if a.eval_params:
 import flax.traverse_util as ft
 with np.load(a.eval_params,allow_pickle=False) as z:
  expected=ft.flatten_dict(state.params.filter(cfg.trainable_filter).to_pure_dict(),sep='/')
  assert set(z.files)==set(expected)
  for k in z.files:assert z[k].shape==expected[k].shape
  vals=ft.unflatten_dict({tuple(k.split('/')):jnp.asarray(z[k]) for k in z.files})
  train_state=state.params.filter(cfg.trainable_filter);train_state.replace_by_pure_dict(vals)
  m=nnx.merge(state.model_def,state.params);nnx.update(m,train_state);m.eval()
 loaded_params=nnx.state(m)
 @jax.jit
 def infer(params,key,obs):
  mm=nnx.merge(state.model_def,params);mm.eval();return mm.sample_actions(key,obs,num_steps=10)
 def predict(r,i):
  obs,_,_=encode(r);action=np.asarray(infer(loaded_params,jax.random.key(20261111+i),obs))[0,:,:10]
  st=ds.stats['actions'];low=np.array(st['min']);span=np.array(st['max'])-low
  return np.where(span<1e-6,low,(action+1)/2*span+low)
 from offline_eval import evaluate
 evaluate(predict,a.eval_split,a.output/'evaluation',a.eval_limit)
 sys.exit(0)
@jax.jit
def grad(params,key,batch):
 m=nnx.merge(state.model_def,params);m.train()
 def loss(m):
  err=m.compute_loss_per_dim(key,batch[0],batch[1],train=True);return (err*batch[2]).sum()/jnp.maximum(batch[2].sum(),1)
 return nnx.value_and_grad(loss,argnums=nnx.DiffState(0,cfg.trainable_filter))(m)
@functools.partial(jax.jit, donate_argnums=(0,1))
def update(st,g):
 params=st.params.filter(cfg.trainable_filter);updates,opt=st.tx.update(g,st.opt_state,params);new=optax.apply_updates(params,updates);m=nnx.merge(st.model_def,st.params);nnx.update(m,new)
 return dataclasses.replace(st,step=st.step+1,params=nnx.state(m),opt_state=opt)
t0=time.monotonic()
with (a.output/'metrics.jsonl').open('w',buffering=1) as out:
 for step in range(a.steps):
  gs=None;loss=0
  for j in range(a.batch):
   l,g=grad(state.params,jax.random.fold_in(rng,step*a.batch+j),sample(step*a.batch+j));gs=g if gs is None else jax.tree.map(lambda x,y:x+y,gs,g);loss+=float(l)/a.batch
  gs=jax.tree.map(lambda x:x/a.batch,gs);state=update(state,gs);jax.block_until_ready(state)
  assert np.isfinite(loss)
  row=dict(step=step+1,loss=loss,elapsed_s=time.monotonic()-t0);out.write(json.dumps(row)+'\n');print(row,flush=True)
# Save trainable values only, explicitly requiring the frozen base and recipe for reload.
import flax.traverse_util as ft
flat=ft.flatten_dict(state.params.filter(cfg.trainable_filter).to_pure_dict(),sep='/')
np.savez(a.output/'trainable_params.npz',**{k:np.asarray(jax.device_get(v)) for k,v in flat.items()})
(a.output/'STATUS.json').write_text(json.dumps(dict(status='TRAINING_SEGMENT_COMPLETE_NOT_EVALUATED',steps=a.steps,optimizer_saved=False,checkpoint_kind='base plus trainable parameters; inference only'),indent=2))
