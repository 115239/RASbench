from __future__ import annotations
import argparse
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description='Safe Agent v1 paper200')
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('compile-questions')
    p.add_argument('--questions',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p=sub.add_parser('prepare-vision');p.add_argument('--output',type=Path,required=True)
    p=sub.add_parser('freeze')
    p.add_argument('--path',type=Path,action='append',required=True);p.add_argument('--output',type=Path,required=True)
    p=sub.add_parser('verify-evidence');p.add_argument('path',type=Path)
    p=sub.add_parser('queue-init')
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--database',type=Path,required=True)
    p.add_argument('--releases',type=Path)
    p.add_argument('--split',choices=['development','final'],default='final')
    p=sub.add_parser('queue-run')
    p.add_argument('--database',type=Path,required=True);p.add_argument('--launcher',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--once',action='store_true')
    p=sub.add_parser('queue-report')
    p.add_argument('--database',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p=sub.add_parser('queue-retry')
    p.add_argument('--database',type=Path,required=True);p.add_argument('--run-id',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=parser.parse_args()
    if a.command=='compile-questions':
        from .task_spec import compile_questions
        print(compile_questions(a.questions,a.output))
    elif a.command=='prepare-vision':
        from .artifacts import prepare_vision
        value=prepare_vision(a.output)
        print(json.dumps({k:{'revision':v['revision'],'status':v['status']} for k,v in value['models'].items()}))
    elif a.command=='freeze':
        from .artifacts import freeze
        result=freeze(a.path,a.output);print(json.dumps({'files':len(result['files']),'manifest':str(a.output)}))
    elif a.command=='verify-evidence':
        from .evidence import verify_chain
        print(json.dumps(verify_chain(a.path)))
    else:
        from .experiments import initialize_queue,run_worker,queue_report,retry_run
        if a.command=='queue-init':print(json.dumps(initialize_queue(a.manifest,a.database,a.releases,split=a.split)))
        elif a.command=='queue-run':run_worker(a.database,a.launcher,a.output,once=a.once)
        elif a.command=='queue-retry':retry_run(a.database,a.run_id,a.output)
        else:
            result=queue_report(a.database,a.output)
            print(json.dumps({k:result[k] for k in ['queue_status','goal_achieved','attacked_delta','attacked_delta_ci95']}))


if __name__=='__main__':main()
