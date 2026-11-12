"""Read-only recording checks; reports are separate from original attempt status."""
import argparse,json
from pathlib import Path
from contestant_run import validate_hdf5
p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args()
rows=[]
for f in a.root.glob('attempts/*/*/episode/submission.hdf5'):
 try: rows.append({'path':str(f),'valid':True,**validate_hdf5(f)})
 except Exception as e:rows.append({'path':str(f),'valid':False,'error':str(e)})
(a.root/'POSTFIX_RECORDING_VALIDATION.json').write_text(json.dumps({'recordings':rows,'all_valid':bool(rows) and all(r['valid'] for r in rows)},indent=2))
print(rows)
