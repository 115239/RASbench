import pathlib,json,csv,numpy as np
R=pathlib.Path(__file__).resolve().parents[1];C=R/'new_experiment/phase_coverage';O=R/'analysis/phase_coverage';O.mkdir(exist_ok=True,parents=True)
seeds=[20261111,20261112,20261113];loaded={};reference=None
for model in ['pi05','gr00t']:
 for seed in seeds:
  p=C/f'{model}_seed{seed}/evaluation';rows=[json.loads(x) for x in (p/'predictions.jsonl').read_text().splitlines()]
  key=[(x['episode_id'],x['step']) for x in rows]
  with np.load(p/'predictions.npz',allow_pickle=False) as z:pred=z['prediction'];target=z['target'];mask=z['mask']
  if reference is None:reference=(key,target,mask,rows)
  else:assert key==reference[0] and np.array_equal(target,reference[1]) and np.array_equal(mask,reference[2])
  frozen=[json.loads(x) for x in (C/'SAMPLES.jsonl').read_text().splitlines()]
  assert key==[(x['episode_id'],x['step']) for x in frozen]
  assert len(key)==802 and len(set(x[0] for x in key))==80
  loaded[model,seed]=np.abs(pred[:,0,:]-target[:,0,:])
phases=np.array([x['phase'] for x in rows]);episodes=np.array([x['episode_id'] for x in rows]);questions=np.array([x['question'] for x in rows]);records=[];groups={'base_vx_m_s':[0],'base_yaw_rad_s':[2],'tcp_translation_m':[3,4,5],'tcp_rotation_rad':[6,7,8],'gripper_closure':[9]}
rng=np.random.default_rng(20261111)
for phase in sorted(set(phases)):
 indices=np.flatnonzero(phases==phase);assert len(set(episodes[indices]))==len(indices)
 for metric,dims in groups.items():
  a={m:np.stack([loaded[m,s][indices][:,dims].mean(1) for s in seeds]) for m in ['pi05','gr00t']}
  lo=hi=None
  if len(indices)>=10:
   bs=rng.integers(0,3,(10000,3));be=np.concatenate([rng.choice(np.flatnonzero(questions[indices]==q),(10000,np.sum(questions[indices]==q)),replace=True) for q in sorted(set(questions[indices]))],axis=1)
   delta=a['gr00t']-a['pi05'];boot=delta[bs[:,:,None],be[:,None,:]].mean((1,2));lo,hi=map(float,np.quantile(boot,[.025,.975]))
  records.append(dict(phase=phase,metric=metric,episodes=len(indices),pi05_mae=float(a['pi05'].mean()),gr00t_mae=float(a['gr00t'].mean()),delta_gr00t_minus_pi05=float((a['gr00t']-a['pi05']).mean()),ci95_low=lo,ci95_high=hi))
with (O/'per_phase.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
(O/'SUMMARY.json').write_text(json.dumps({'status':'SUPPLEMENTARY_OFFLINE_COMPLETE','samples':802,'episodes':80,'training_seeds':3,'metrics':records,'ci':'paired seed and within-question episode bootstrap; descriptive only','rare_phases':'AISLE_CENTER and LOADED_TURN each one episode; no confidence intervals','closed_loop_success_rate':None},indent=2))
print('phase coverage complete')
