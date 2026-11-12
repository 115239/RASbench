import pathlib,json,numpy as np,time
from common_data import R,Reader

def samples(split,limit=0):
 reader=Reader(split);idx=0
 for e,entry in enumerate(reader.entries):
  a=reader.arrays(e);valid=np.flatnonzero(a['train_mask'])
  for pos in np.linspace(0,len(valid)-1,4).astype(int):
   step=int(valid[pos]);yield idx,entry,step,reader.read(e,step);idx+=1
   if limit and idx>=limit:return

def evaluate(predict,split,out,limit=0):
 out=pathlib.Path(out);out.mkdir(parents=True,exist_ok=True);rows=[];preds=[];targets=[];masks=[]
 for i,e,step,r in samples(split,limit):
  t=time.monotonic();pred=np.asarray(predict(r,i),np.float32)[:15,:10];elapsed=time.monotonic()-t
  assert pred.shape==(15,10) and np.isfinite(pred).all(),('invalid_prediction',pred.shape)
  target=np.zeros((15,10),np.float32);mask=np.zeros((15,10),bool);n=len(r['actions']);target[:n]=r['actions'];mask[:n]=True;err=(pred-target)
  row=dict(index=i,episode_id=e['episode_id'],question=e['question'],step=step,phase=r['phase'],valid_horizon=n,seconds=elapsed,first_abs_error=np.abs(err[0]).tolist(),horizon_abs_error=np.abs(err[:n]).mean(0).tolist())
  rows.append(row);preds.append(pred);targets.append(target);masks.append(mask)
  with (out/'predictions.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
  if i%20==0:print('EVAL',split,i,flush=True)
 np.savez_compressed(out/'predictions.npz',prediction=np.stack(preds),target=np.stack(targets),mask=np.stack(masks))
 summary={'status':'OFFLINE_ACTION_EVALUATION_COMPLETE','split':split,'samples':len(rows),'episodes':len(set(x['episode_id'] for x in rows)),'first_action_mae_by_channel':np.mean([x['first_abs_error'] for x in rows],axis=0).tolist(),'horizon_mae_by_channel':np.mean([x['horizon_abs_error'] for x in rows],axis=0).tolist(),'latency_p50_s':float(np.median([x['seconds'] for x in rows[1:]])) if len(rows)>1 else None,'latency_p95_s':float(np.quantile([x['seconds'] for x in rows[1:]],.95)) if len(rows)>1 else None,'first_request_s':rows[0]['seconds'],'closed_loop_success_rate':None,'channel_units':['m/s','m/s','rad/s','m','m','m','rad','rad','rad','closure_fraction']}
 (out/'SUMMARY.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
