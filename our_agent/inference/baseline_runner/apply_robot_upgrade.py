"""Build-time organizer robot revision; preserve formal topology and all scene files."""
from pathlib import Path
import json,hashlib,shutil
B=Path('/opt/safety_embodiment/baseline');root=Path('/opt/safety_embodiment');runner=root/'competition_runner/runner'
manifest=json.loads((B/'ROBOT_MANIFEST.json').read_text());newsha=next(r['sha256'] for r in manifest['files'] if r['path']==manifest['entrypoint']);oldsha='7ca91879191f18f71184f7511c39250d4b52dbec503335eaf33e412b35f42d18'
output=B/'provenance/robot_upgrade';output.mkdir(parents=True,exist_ok=True);changes=[]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def update(p,data):
 prior=sha(p);target=output/p.relative_to(root);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
 p.write_text(data);changes.append({'path':str(p),'before_sha256':prior,'sha256':sha(p),'original':str(target)})
support=root/'question/unified_q01_q24_20260813/assets/warehouse/runtime_support'
for row in manifest['files']:
 source=B/row['path'];assert sha(source)==row['sha256'];dest=support/'M20_piper'/source.relative_to(B/'assets/M20_piper');dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
 changes.append({'path':str(dest),'sha256':sha(dest),'copied_from':str(source)})
p=support/'asset_manifest.json';v=json.loads(p.read_text());assert v['robot_usd_sha256']==oldsha
v.update(robot_usd='M20_piper/piper_piper_d435i_sensors.usda',robot_usd_sha256=newsha)
update(p,json.dumps(v,ensure_ascii=False,indent=2)+'\n')
profiles=list(runner.glob('_vendor_*/robolab_action_schema/robot_profiles.py'))
assert profiles
for p in profiles:
 text=p.read_text();assert text.count(oldsha)==1;update(p,text.replace(oldsha,newsha))
p=runner/'m20_runtime/runners/m20_fourview_runner.py';text=p.read_text();assert text.count(oldsha)==1
anchor='        source_layer = Sdf.Layer.FindOrOpen(str(EXPECTED_USD))'
assert text.count(anchor)==1
text=text.replace(anchor,'        source_stage = Usd.Stage.Open(str(EXPECTED_USD))\n        source_layer = source_stage.Flatten() if source_stage is not None else None')
update(p,text.replace(oldsha,newsha))
# Runtime PYTHONPATH also contains these compatible support libraries.
for p in (B/'code/runner/vendor').glob('_vendor_*/robolab_action_schema/robot_profiles.py'):
 text=p.read_text()
 if oldsha in text:update(p,text.replace(oldsha,newsha))
(B/'IMAGE_ROBOT_UPGRADE.json').write_text(json.dumps({'entrypoint_sha256':newsha,'base_robot_sha256':oldsha,'changes':changes,'scene_files_changed':False},ensure_ascii=False,indent=2)+'\n')
code_manifest=B/'CODE_MANIFEST.json'
if code_manifest.exists():
 v=json.loads(code_manifest.read_text())
 for row in changes:
  p=Path(row['path'])
  if p.is_relative_to(B):v['files'][str(p.relative_to(B))]=sha(p)
 for p in output.rglob('*'):
  if p.is_file():v['files'][str(p.relative_to(B))]=sha(p)
 v['files']['IMAGE_ROBOT_UPGRADE.json']=sha(B/'IMAGE_ROBOT_UPGRADE.json')
 code_manifest.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
print('ROBOT_UPGRADE',newsha,len(changes),flush=True)
