"""Indexes and capacity estimates; never infer scores from collection status."""
import csv,json,statistics,shutil
from collections import defaultdict,Counter
from pathlib import Path
from .common import read,write,now
from .queue import connect,report


def build(root):
    state=report(root);questions={q['question']:q for q in read(root/'manifests/questions.json')}
    with connect(root/'queue.sqlite3') as db:
        jobs=[dict(r) for r in db.execute('SELECT * FROM runs ORDER BY purpose,system,CAST(substr(question,2) AS INTEGER),variant')]
        attempts=[dict(r) for r in db.execute('SELECT * FROM attempts ORDER BY run_id,attempt')]
    lineage_path=root/'manifests/resume_repaired.json'
    lineage={r['run_id']:r for r in read(lineage_path)['runs']} if lineage_path.exists() else {}
    measurements=defaultdict(list);aux_measurements=defaultdict(list);index=[];failures=[]
    for job in jobs:
        entry={k:job[k] for k in ['run_id','purpose','system','question','variant','seed','status','attempts','output','reason']}
        entry['score_status']='NOT_SCORED';entry['task_score']=None
        if job['run_id'] in lineage:
            entry['source_run_id']=lineage[job['run_id']]['source_run_id']
            entry['source_output']=lineage[job['run_id']]['source_output']
        if job['status']=='collected':
            out=Path(job['output']);v=read(out/'COLLECTION_VALIDATION.json')
            entry.update({k:v.get(k) for k in ['hdf5','mp4','hdf5_sha256','mp4_sha256','simulation_duration_s','inference_count','termination']})
            entry['attack_evidence']=v.get('attack_evidence')
            stamp=out/'SIZE_MEASUREMENT.json'
            if stamp.exists():size=read(stamp)['bytes']
            else:
                size=sum(p.stat().st_size for p in out.rglob('*') if p.is_file());write(stamp,{'bytes':size,'time':now()})
            a=next(a for a in attempts if a['run_id']==job['run_id'] and a['attempt']==job['attempts'])
            elapsed=(a['finished']-a['started']) if a['finished'] else None
            entry.update(bytes=size,elapsed_wall_s=elapsed)
            if elapsed:measurements[(job['system'],questions[job['question']]['task_type'])].append((elapsed,size))
            if elapsed and job['purpose'] in ('assisted','assisted_pilot'):
                aux_measurements[(job['system'],questions[job['question']]['task_type'])].append((elapsed,size))
            if entry['termination']:failures.append({'category':'physical_or_policy_termination',**entry})
        elif job['status']=='infrastructure_failure':failures.append({'category':'infrastructure_failure',**entry})
        index.append(entry)
    estimates={':'.join(k):{'samples':len(v),'mean_wall_s':statistics.mean(x[0] for x in v),
        'mean_bytes':statistics.mean(x[1] for x in v),'max_bytes':max(x[1] for x in v)} for k,v in measurements.items()}
    remaining=[j for j in jobs if j['purpose']=='main' and j['status']!='collected']
    fallback=list(x for values in measurements.values() for x in values)
    gpu_seconds=0.;estimated_bytes=0.;unmeasured=[]
    if fallback:
        for j in remaining:
            key=(j['system'],questions[j['question']]['task_type']);values=measurements.get(key)
            if not values:values=fallback;unmeasured.append(':'.join(key))
            gpu_seconds+=statistics.mean(x[0] for x in values);estimated_bytes+=statistics.mean(x[1] for x in values)
    aux_bytes=0.;aux_seconds=0.;aux_unmeasured=[]
    if fallback:
        for j in jobs:
            if j['purpose']!='assisted' or j['status']=='collected':continue
            key=(j['system'],questions[j['question']]['task_type']);values=aux_measurements.get(key)
            if not values:values=measurements.get(key,fallback);aux_unmeasured.append(':'.join(key))
            aux_seconds+=statistics.mean(x[0] for x in values);aux_bytes+=statistics.mean(x[1] for x in values)
    disk=shutil.disk_usage(root)
    audit=root/'reports/FINAL_AUDIT_main.json'
    remaining_main_complete=state['remaining_collection_complete'] and audit.exists() and read(audit).get('passed',False)
    main_complete=remaining_main_complete and not state['legacy_audit_required']
    from .resume_repaired import expected_matrix,key
    expected_auxiliary=expected_matrix(root,'assisted')
    auxiliary_total=len(expected_auxiliary)
    auxiliary_rows=[j for j in jobs if j['purpose']=='assisted']
    auxiliary_matrix_matches=len(auxiliary_rows)==auxiliary_total and {key(j) for j in auxiliary_rows}==expected_auxiliary
    auxiliary_collected=sum(j['purpose']=='assisted' and j['status']=='collected' for j in jobs)
    auxiliary_audit=root/'reports/FINAL_AUDIT_assisted.json'
    auxiliary_complete=auxiliary_matrix_matches and auxiliary_total==auxiliary_collected and auxiliary_audit.exists() and read(auxiliary_audit).get('passed',False)
    summary={'time':now(),'main_expected':state['main_expected'],'main_collected':state['main_collected'],
        'main_planned':state['main_planned'],'main_parked':state['main_parked'],
        'main_enqueued':state['main_enqueued'],'collection_scope':state['collection_scope'],
        'remaining_main_collection_complete':remaining_main_complete,
        'remaining_all_collection_complete':remaining_main_complete and auxiliary_complete,
        'legacy_audit_required':state['legacy_audit_required'],
        'legacy_audit_status':('REQUIRED' if remaining_main_complete and auxiliary_complete else 'DEFERRED_UNTIL_REMAINING_COMPLETE') if state['legacy_audit_required'] else 'NOT_APPLICABLE',
        'whole_paper_complete':main_complete and auxiliary_complete,
        'collection_complete':main_complete,'all_deliverables_complete':main_complete and auxiliary_complete,'score_status':'NOT_SCORED',
        'pilot_collected':sum(j['purpose']=='pilot' and j['status']=='collected' for j in jobs),
        'handler_probe_collected':sum(j['purpose']=='probe' and j['status']=='collected' for j in jobs),
        'handler_probe_scope':'controlled-trigger physical interface diagnostics; excluded from main and assisted performance',
        'assisted_expected':auxiliary_total,'assisted_enqueued':len(auxiliary_rows),'assisted_collected':auxiliary_collected,'assisted_complete':auxiliary_complete,
        'groups':state['groups'],'resource_measurements':estimates,'free_disk_bytes':disk.free,
        'estimated_remaining_main_bytes':estimated_bytes,'estimated_remaining_main_worker_seconds':gpu_seconds,
        'estimated_remaining_assisted_bytes':aux_bytes,'estimated_remaining_assisted_worker_seconds':aux_seconds,
        'estimated_remaining_total_bytes':estimated_bytes+aux_bytes,
        'assisted_estimate_fallback_groups':sorted(set(aux_unmeasured)),
        'estimate_limitations':'pilot sample means; attack durations, concurrency contention and auxiliary runs may increase cost',
        'unmeasured_system_task_groups':sorted(set(unmeasured))}
    write(root/'reports/data_index.json',index);write(root/'reports/failures.json',failures)
    write(root/'reports/attempt_index.json',attempts);write(root/'reports/COLLECTION_REPORT.json',summary)
    with (root/'reports/data_index.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        fields=['run_id','purpose','system','question','variant','seed','status','attempts','hdf5','mp4','termination','score_status']
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(index)
    (root/'reports/采集报告.md').write_text(
        '# 200 题实验采集进度\n\n更新时间：'+summary['time']+'\n\n'
        '主实验总计划：%d；旧版搁置待审：%d；本版本剩余矩阵有效采集：**%d / %d**。预检有效采集：%d。辅助实验有效采集：%d。\n\n'%(
            summary['main_planned'],summary['main_parked'],summary['main_collected'],summary['main_expected'],summary['pilot_collected'],summary['assisted_collected'])+
        ('剩余主实验与辅助实验已完成，须逐项审查搁置旧版数据并重跑有问题的回合；整篇实验尚未完成。' if summary['remaining_all_collection_complete'] and summary['legacy_audit_required'] else
         '主实验与辅助实验采集完成、待评分。' if summary['all_deliverables_complete'] else
         '主实验采集完成、待评分；辅助实验尚未完成。' if summary['collection_complete'] else
         '采集尚未完成；未执行最终评分。')+
        '\n\n失败、超时与未触发攻击如实保留。`data_index.csv` 和 `data_index.json` 提供数据索引；'
        '`attempt_index.json` 保留所有尝试；`failures.json` 区分任务终止与基础设施失败。\n\n'
        '剩余主实验粗略存储估算：%.1f GiB；辅助实验：%.1f GiB；当前剩余空间：%.1f GiB。'
        '缺少辅助实测分组时暂用其他预检估算，完整分组样本和限制见 `COLLECTION_REPORT.json`。\n'%(
            estimated_bytes/1024**3,aux_bytes/1024**3,disk.free/1024**3),encoding='utf-8')
    return summary
