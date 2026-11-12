import pathlib,json,numpy as np,csv,collections
R=pathlib.Path(__file__).resolve().parents[1];C=R/'new_experiment/campaign_v1';O=R/'analysis/new_offline';O.mkdir(parents=True,exist_ok=True);out=[]
threshold=np.array([.005,.005,.01,.0005,.0005,.0005,.002,.002,.002,.05])
for model in ['pi05','gr00t']:
 for seed in [20261111,20261112,20261113]:
  p=C/f'{model}_seed{seed}_test/evaluation';rows=[json.loads(l) for l in (p/'predictions.jsonl').read_text().splitlines()]
  with np.load(p/'predictions.npz',allow_pickle=False) as z:pred=z['prediction'][:,0,:];target=z['target'][:,0,:]
  phases=np.array([r['phase'] for r in rows]);err=np.abs(pred-target)
  for phase in ['ALL']+sorted(set(phases)):
   valid=np.ones(len(rows),bool) if phase=='ALL' else phases==phase
   for j in range(10):
    active=valid&(np.abs(target[:,j])>threshold[j]);out.append({'model':model,'seed':seed,'phase':phase,'channel':j,'samples':int(valid.sum()),'mae':float(err[valid,j].mean()),'active_samples':int(active.sum()),'active_mae':float(err[active,j].mean()) if active.any() else None,'active_sign_agreement':float((np.sign(pred[active,j])==np.sign(target[active,j])).mean()) if active.any() and j<9 else None})
with (O/'phase_and_active_action_metrics.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
(O/'ACTIVE_METRIC_DEFINITION.json').write_text(json.dumps({'activity_thresholds_by_channel':threshold.tolist(),'units':['m/s','m/s','rad/s','m','m','m','rad','rad','rad','closure_fraction'],'sign_agreement_excludes_absolute_gripper':True,'note':'diagnostic metrics; not success rates'},indent=2))
