import pathlib,json,csv,numpy as np
R=pathlib.Path(__file__).resolve().parents[1];C=R/'new_experiment/campaign_v1';O=R/'analysis/new_offline';O.mkdir(parents=True,exist_ok=True)
seeds=json.loads((C/'PROTOCOL.json').read_text())['seeds'];loaded={};refs=None
for model in ['pi05','gr00t']:
 for seed in seeds:
  p=C/f'{model}_seed{seed}_test/evaluation';s=json.loads((p/'SUMMARY.json').read_text());assert s['samples']==320 and s['episodes']==80
  rows=[json.loads(l) for l in (p/'predictions.jsonl').read_text().splitlines()];key=[(x['episode_id'],x['step']) for x in rows]
  with np.load(p/'predictions.npz',allow_pickle=False) as z:pred,target,mask=[z[k].copy() for k in ['prediction','target','mask']]
  if refs is None:refs=(key,target,mask,rows)
  else:assert key==refs[0] and np.array_equal(target,refs[1]) and np.array_equal(mask,refs[2]),'evaluation sample mismatch'
  assert len(rows)==320 and all(len(set(x['episode_id'] for x in rows[i:i+4]))==1 for i in range(0,320,4))
  loaded[model,seed]=np.abs(pred[:,0,:]-target[:,0,:]).reshape(80,4,10).mean(1)
groups={'base_vx_m_s':[0],'base_vy_m_s':[1],'base_yaw_rad_s':[2],'tcp_translation_m':[3,4,5],'tcp_rotation_rad':[6,7,8],'gripper_closure':[9]};records=[];diffs=[]
questions=np.array([refs[3][i*4]['question'] for i in range(80)]);rng=np.random.default_rng(20261111);boot_seeds=rng.integers(0,3,(10000,3));boot_eps=np.concatenate([rng.choice(np.flatnonzero(questions==q),(10000,10),replace=True) for q in sorted(set(questions))],axis=1)
for g,dims in groups.items():
 vals={m:np.stack([loaded[m,s][:,dims].mean(1) for s in seeds]) for m in ['pi05','gr00t']}
 delta=vals['gr00t']-vals['pi05'];boot=delta[boot_seeds[:,:,None],boot_eps[:,None,:]].mean(axis=(1,2));lo,hi=np.quantile(boot,[.025,.975]);diffs.append(dict(metric=g,pi05_mean=float(vals['pi05'].mean()),gr00t_mean=float(vals['gr00t'].mean()),delta_gr00t_minus_pi05=float(delta.mean()),ci95_low=float(lo),ci95_high=float(hi),lower_is_better=True))
 for m in vals:
  for j,seed in enumerate(seeds):records.append(dict(model=m,seed=seed,metric=g,mae=float(vals[m][j].mean())))
byq=[]
for q in sorted(set(questions)):
 for g,dims in groups.items():
  for m in ['pi05','gr00t']:byq.append(dict(question=q,metric=g,model=m,mae=float(np.mean([loaded[m,s][questions==q][:,dims].mean() for s in seeds]))))
for name,rows in [('matched_test_summary.csv',diffs),('per_seed.csv',records),('per_question.csv',byq)]:
 with (O/name).open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
summary={'status':'OFFLINE_COMPARISON_COMPLETE','test_episodes':80,'test_observations':320,'training_seeds':3,'comparison_unit':'mean first-action physical-unit MAE per episode','ci':'paired crossed bootstrap of seed pairs and episodes stratified by question, 10000 resamples','limitations':['three training seeds only','four deterministic observations per held-out episode','LoRA placement and parameter counts differ','not closed-loop or real-robot success','inference diffusion steps differ; shared GPU latency is diagnostic only','multiple metrics; confidence intervals are descriptive, not a familywise significance claim'],'metrics':diffs,'closed_loop_success_rate':None}
(O/'SUMMARY.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
