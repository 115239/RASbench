"""Read-only Judge integration check against a preserved development attempt."""
import argparse,json,sys,zipfile
from pathlib import Path
from .common import read,write


def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('attempt',type=Path)
    a=p.parse_args();root=a.root.resolve();out=a.attempt.resolve()
    sys.path[:0]=[str(root/'runtime/judge'),str(root/'runtime/judge/v2/vendor')]
    from v2.safety.judge_result.episode_delivery import build_episode_delivery
    job=read(out/'job.json');q=job['question']
    with zipfile.ZipFile(root/'runtime/judge/v2/rubric_q1_q200.zip') as z:
        rubric=json.loads(z.read('tasks/'+q+'/rubric.json'));package=json.loads(z.read('tasks/'+q+'/rule_package.json'));info=json.loads(z.read('tasks/'+q+'/task_info.json'))
    full,summary=build_episode_delivery(question_id=q,hdf5_path=out/'episode/submission.hdf5',
        rubric=rubric,rule_package=package,registered_attacks=info['registered_attacks'],
        expected_runner_judge_contract=read(out/'runner_judge_contract.json'))
    dest=root/'reports/development_judge'/q/job['system']/job['variant']/out.name
    write(dest/'full.json',full);write(dest/'summary.json',summary)
    print(json.dumps({'report':str(dest),'validation':summary['validation']},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
