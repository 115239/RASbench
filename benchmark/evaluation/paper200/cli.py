import argparse
from pathlib import Path
import json
from .common import PROJECT, SYSTEMS, question, read


def main():
    p=argparse.ArgumentParser(description='Paper200 physical evidence collection; scoring is independent.')
    sub=p.add_subparsers(dest='command',required=True)
    create=sub.add_parser('prepare');create.add_argument('--root',type=Path,required=True)
    report=sub.add_parser('report');report.add_argument('--root',type=Path,required=True)
    collect=sub.add_parser('collect');collect.add_argument('--root',type=Path,required=True)
    pilot=sub.add_parser('pilot');pilot.add_argument('--root',type=Path,required=True)
    pilot.add_argument('--question',type=question,required=True)
    pilot.add_argument('--system',choices=SYSTEMS,required=True);pilot.add_argument('--variant',default='clean')
    pilot.add_argument('--gpu',type=int,required=True)
    preflight=sub.add_parser('preflight');preflight.add_argument('--root',type=Path,required=True)
    preflight.add_argument('--questions',default='all')
    worker=sub.add_parser('worker');worker.add_argument('--root',type=Path,required=True)
    worker.add_argument('--gpu',type=int,required=True);worker.add_argument('--system',choices=SYSTEMS)
    worker.add_argument('--run-id');worker.add_argument('--purpose',choices=['main','pilot','assisted','probe','assisted_pilot'],default='main')
    retry=sub.add_parser('retry');retry.add_argument('--root',type=Path,required=True)
    retry.add_argument('--run-id',required=True);retry.add_argument('--reason',required=True)
    sched=sub.add_parser('schedule');sched.add_argument('--root',type=Path,required=True)
    sched.add_argument('--purpose',choices=['main','pilot','assisted'],default='main')
    recovery=sub.add_parser('recover');recovery.add_argument('--root',type=Path,required=True)
    release=sub.add_parser('release-check');release.add_argument('--root',type=Path,required=True)
    release.add_argument('--freeze',action='store_true')
    assisted=sub.add_parser('freeze-assistance');assisted.add_argument('--root',type=Path,required=True)
    auxpilot=sub.add_parser('assisted-pilots');auxpilot.add_argument('--root',type=Path,required=True)
    chain=sub.add_parser('check-attacks');chain.add_argument('--root',type=Path,required=True)
    summary=sub.add_parser('summarize');summary.add_argument('--root',type=Path,required=True)
    vision=sub.add_parser('vision-services');vision.add_argument('--root',type=Path,required=True)
    vision.add_argument('--replicas',type=int,default=3)
    final=sub.add_parser('final-audit');final.add_argument('--root',type=Path,required=True)
    final.add_argument('--purpose',choices=['main','assisted'],default='main')
    paired=sub.add_parser('check-pairs');paired.add_argument('--root',type=Path,required=True)
    paired.add_argument('--purpose',choices=['main','pilot','assisted','assisted_pilot'],default='main')
    paired.add_argument('--questions',default='all')
    a=p.parse_args();a.root=a.root.resolve()
    if a.command=='prepare':
        from .prepare import prepare
        result=prepare(a.root)
    elif a.command=='report':
        from .queue import report
        result=report(a.root)
        result={k:v for k,v in result.items() if k!='uncollected'}
    elif a.command=='collect':
        from .pipeline import run
        result=run(a.root)
    elif a.command=='preflight':
        from .launch import preflight
        result=preflight(a.root,a.questions)
    elif a.command=='pilot':
        from .queue import add_pilot
        from .launch import worker
        q=next(q for q in read(a.root/'manifests/questions.json') if q['question']==a.question)
        if a.variant not in ['clean']+q['variants']:raise ValueError('unknown attack variant')
        rid=add_pilot(a.root,a.question,a.system,a.variant)
        result=worker(a.root,a.gpu,run_id=rid,purpose='pilot')
    elif a.command=='retry':
        from .queue import retry
        result=retry(a.root,a.run_id,a.reason)
    elif a.command=='schedule':
        from .scheduler import run
        result=run(a.root,a.purpose)
    elif a.command=='recover':
        from .operations import recover
        result=recover(a.root)
    elif a.command=='release-check':
        from .operations import release_check
        result=release_check(a.root,a.freeze)
    elif a.command=='freeze-assistance':
        from .assistance import freeze_list
        result=freeze_list(a.root)
    elif a.command=='assisted-pilots':
        from .assistance import prepare_pilots
        result=prepare_pilots(a.root)
    elif a.command=='check-attacks':
        from .chain_evidence import report
        result=report(a.root)
        result={k:v for k,v in result.items() if k!='episodes'}
    elif a.command=='summarize':
        from .collection_report import build
        result=build(a.root)
    elif a.command=='vision-services':
        from .vision_pool import provision
        result=provision(a.root,a.replicas)
    elif a.command=='final-audit':
        from .final_audit import audit
        result=audit(a.root,a.purpose)
    elif a.command=='check-pairs':
        from .paired_initial import check
        result=check(a.root,a.purpose,questions=None if a.questions=='all' else {question(q) for q in a.questions.split(',')})
    else:
        from .launch import worker
        result=worker(a.root,a.gpu,run_id=a.run_id,system=a.system,purpose=a.purpose)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
