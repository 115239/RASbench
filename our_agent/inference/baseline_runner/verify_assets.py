"""Verify all packaged bytes without network or host resources."""
import json,hashlib,time,argparse
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
b=Path('/opt/safety_embodiment/baseline');checks={}
for row in json.loads((b/'PAYLOAD_MANIFEST.json').read_text())['files']:checks[row['path']]=row['sha256']
checks.update(json.loads((b/'CODE_MANIFEST.json').read_text())['files'])
for row in json.loads((b/'IMAGE_ROBOT_UPGRADE.json').read_text())['changes']:
 checks[row['path']]=row['sha256']
fail=[];size=0;start=time.time()
for i,(name,expected) in enumerate(checks.items(),1):
 path=b/name
 try:
  with path.open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
  size+=path.stat().st_size
  if actual!=expected:fail.append({'path':name,'error':'hash mismatch'})
 except OSError as e:fail.append({'path':name,'error':str(e)})
 if i%1000==0:print(i,len(checks),flush=True)
a.output.parent.mkdir(parents=True,exist_ok=True)
a.output.write_text(json.dumps({'files':len(checks),'bytes':size,'failures':fail,'elapsed_s':time.time()-start,'offline':True},indent=2))
print('VERIFIED',len(checks),'FAILURES',len(fail),flush=True)
raise SystemExit(bool(fail))
