"""Recover the known validator environment-path defect without rerunning physics."""
import argparse,subprocess,time
from pathlib import Path
from .common import read,write,sha,now
from .launch import validate_existing
from .queue import connect


def recover(root,out):
    job=read(out/'job.json')
    if job['purpose'] not in ('pilot','probe','assisted_pilot'):raise ValueError('development recovery only')
    cleanup=read(out/'PROCESS_CLEANUP.json')
    if not cleanup['completed']:raise ValueError('live attempt')
    log=out/'collection_validation.log'
    if 'FileNotFoundError' not in log.read_text() or '/native/ffmpeg' not in log.read_text():raise ValueError('not the reviewed environment failure')
    with connect(root/'queue.sqlite3') as db:
        row=db.execute('SELECT * FROM runs WHERE run_id=?',(job['run_id'],)).fetchone()
        if row['status']=='running' or row['output']!=str(out):raise ValueError('active or superseded attempt')
    source=out/'episode/submission.hdf5';original=out/'episode/submission.before_validation_environment_repair.hdf5'
    if original.exists():raise ValueError('recovery already attempted')
    subprocess.run(['cp','--reflink=auto','--preserve=mode,timestamps',str(source),str(original)],check=True)
    log.rename(out/'collection_validation.original_failure.log')
    write(out/'VALIDATION_ENVIRONMENT_RECOVERY.json',{'created':now(),'original_hdf5':str(original),'original_sha256':sha(original),
        'change':'validator environment now declares PAPER200_PROJECT_ROOT; uses the same attempt code/runtime snapshot',
        'physics_rerun':False,'original_failure_preserved':True,'status':'RUNNING'})
    result=validate_existing(root,out)
    with connect(root/'queue.sqlite3') as db:
        import json
        changed=db.execute("UPDATE runs SET status='collected',reason=?,result=? WHERE run_id=? AND output=? AND status!='running'",
            ('VALIDATOR_ENVIRONMENT_RECOVERED',json.dumps(result),job['run_id'],str(out))).rowcount
        if changed!=1:raise RuntimeError('queue changed during recovery')
        db.execute("UPDATE attempts SET status='collected',reason=? WHERE run_id=? AND attempt=?",('VALIDATOR_ENVIRONMENT_RECOVERED; original failure and pre-validation HDF5 retained',job['run_id'],job['attempts']))
    report=read(out/'VALIDATION_ENVIRONMENT_RECOVERY.json');report.update(status='VALID',completed=now(),final_sha256=sha(source));write(out/'VALIDATION_ENVIRONMENT_RECOVERY.json',report)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('attempt',type=Path);a=p.parse_args();print(recover(a.root.resolve(),a.attempt.resolve())['status'])
