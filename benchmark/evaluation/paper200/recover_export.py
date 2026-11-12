"""Recover export-only failures from preserved physical evidence; never simulate."""
from pathlib import Path
import os,sys
from .common import read,write,now

def recover(root,out):
    from robolab_runner.m20_question_runner import export_formal_submission,append_attack_evidence_and_validate
    episode=out/'episode';job=read(out/'job.json')
    if (episode/'submission.hdf5').exists():raise ValueError('submission exists; refusing overwrite')
    before={p.name:p.stat().st_size for p in episode.glob('*') if p.is_file()}
    export_formal_submission(episode,episode/'runner_capture.hdf5')
    task=read(root/'benchmark/task'/job['question']/'data/task/task.json')
    preflight=read(episode/'runner_preflight.json')
    profile=Path(preflight['attack_catalog']['merged_runtime_profile'])
    safety=read(episode/'resolved_safety_profile.json')
    append_attack_evidence_and_validate(episode,profile,task,safety)
    if (episode/'navigation.mp4').exists():os.rename(episode/'navigation.mp4',episode/'episode.mp4')
    write(out/'EXPORT_REPAIR.json',{'time':now(),'original_files':before,'reason':'independent physical clock exporter compatibility; physical trajectory unchanged'})
    from .validate import validate_episode
    return validate_episode(root,job)

if __name__=='__main__':
    print(recover(Path(sys.argv[1]).resolve(),Path(sys.argv[2]).resolve()))
