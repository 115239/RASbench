"""Exercise all four delivery CLI paths on the same immutable pilot HDF5."""
import argparse,json,os,shutil,subprocess,sys,zipfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from .common import read,write,NATIVE,sha,now


def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('attempt',type=Path);p.add_argument('--tag',default=None);a=p.parse_args()
    root=a.root.resolve();attempt=a.attempt.resolve();job=read(attempt/'job.json');q=job['question']
    out=root/'pilot/judge_entrypoints'/q/job['system']/job['variant']/attempt.name
    if a.tag:
        if not a.tag.replace('_','').replace('-','').isalnum():raise ValueError('invalid report tag')
        out=out/a.tag
    out.mkdir(parents=True,exist_ok=False)
    source=attempt/'episode/submission.hdf5';digest=sha(source);inputs=out/'inputs';inputs.mkdir()
    (inputs/'episode.hdf5').symlink_to(source)
    def source_manifest(base):
        return {str(p.relative_to(base)):sha(p) for p in sorted(base.rglob('*'))
                if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'}
    judge_source=root/'runtime/judge';before=source_manifest(judge_source)
    judge_snapshot=out/'judge_runtime_snapshot'
    shutil.copytree(judge_source,judge_snapshot,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    copied=source_manifest(judge_snapshot)
    if copied!=before or source_manifest(judge_source)!=before:
        write(out/'SNAPSHOT_FAILED.json',{'reason':'JUDGE_SOURCE_CHANGED_DURING_SNAPSHOT','created':now()})
        raise RuntimeError('Judge changed during snapshot; retry with a fresh tag')
    write(out/'JUDGE_RUNTIME_SNAPSHOT.json',{'source':str(judge_source),'snapshot':str(judge_snapshot),
        'files':copied,'created':now(),'scope':'immutable per-verification Judge code, vendor and rubric archive bytes'})
    rules=judge_snapshot/'v2/rubric_q1_q200.zip'
    with zipfile.ZipFile(rules) as z:
        for name in ('rubric','rule_package','task_info'):
            (out/(name+'.json')).write_bytes(z.read('tasks/'+q+'/'+name+'.json'))
    contracts=out/'contracts';contracts.mkdir();shutil.copy2(attempt/'runner_judge_contract.json',contracts/(q+'.json'))
    write(out/'manifest.json',{'episodes':[{'question_id':q,'hdf5_path':str(source),'rubric_archive':str(rules)}]})
    with zipfile.ZipFile(out/'episodes.zip','w',compression=zipfile.ZIP_STORED) as z:z.write(source,'original/episode.hdf5')
    modes={'single':['--question-id',q,'--hdf5',str(source),'--rubric',str(out/'rubric.json'),
                    '--rule-package',str(out/'rule_package.json'),'--task-info',str(out/'task_info.json')],
           'manifest':['--manifest',str(out/'manifest.json')],
           'directory':['--hdf5-dir',str(inputs),'--rubric-archive',str(rules)],
           'archive':['--hdf5-archive',str(out/'episodes.zip'),'--rubric-archive',str(rules)]}
    env=dict(os.environ,PYTHONPATH=str(judge_snapshot)+':'+str(judge_snapshot/'v2/vendor'),PYTHONDONTWRITEBYTECODE='1')
    invocations={}
    def run(pair):
        name,args=pair
        cmd=[str(NATIVE/'isaac_python.sh'),str(judge_snapshot/'v2/tools/generate_episode_delivery.py'),
             '--output-root',str(out/name),'--runner-contract-root',str(contracts)]+args
        with (out/(name+'.log')).open('w') as log:rc=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=300).returncode
        invocations[name]={'argv':cmd,'PYTHONPATH':env['PYTHONPATH'],'returncode':rc,'interpreter':str(NATIVE/'isaac_python.sh')}
        if rc:return name,{'error':'CLI failed','returncode':rc}
        summaries=list((out/name).rglob('submission_result.analysis_summary.json'))
        if len(summaries)!=1:return name,{'error':'not exactly one summary'}
        value=read(summaries[0]);keys=['episode_id','question_id','system_id','system_version','task_id','seed','policy_seed','repeat_index','attack_seed','purpose','data_role','run_id','attempt_id','selected_attack_ids','condition_id','validation','opportunities','scopes']
        return name,{k:value.get(k) for k in keys}
    with ThreadPoolExecutor(max_workers=4) as executor:results=dict(executor.map(run,modes.items()))
    snapshot_unchanged=source_manifest(judge_snapshot)==copied
    passed=all('error' not in v and v==results['single'] for v in results.values()) and sha(source)==digest and snapshot_unchanged
    write(out/'ENTRYPOINT_INVOCATIONS.json',invocations)
    report={'passed':passed,'source':str(source),'source_sha256':digest,'source_unchanged':sha(source)==digest,
            'entrypoints':list(results),'results':results,'judge_python_sha256':{k:v for k,v in copied.items() if k.endswith('.py') and '/vendor/' not in k},'judge_runtime_snapshot':str(judge_snapshot),'judge_snapshot_unchanged':snapshot_unchanged,'invocations_report':str(out/'ENTRYPOINT_INVOCATIONS.json'),'scope':'same declared identity, attack and metric eligibility across four public entrypoints; no final scores asserted'}
    write(out/'PARITY.json',report);print(json.dumps({'passed':passed,'report':str(out/'PARITY.json')},ensure_ascii=False))
    if not passed:raise SystemExit(1)


if __name__=='__main__':main()
