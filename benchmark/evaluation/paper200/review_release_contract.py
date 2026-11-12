"""Read-only candidate audit; explicit identity repair never releases dispatch.

Reports are diagnostics, not a substitute for real-scene acceptance. Handler
surface assessments are per-question source-text reviews, not an alias whitelist.
"""
import argparse
from collections import Counter
from pathlib import Path
from .common import PROJECT, read, write, sha, now
from .evidence_contract import canonical_digest


# These IDs were inspected against the actual Chinese source requirement and
# compiled parameters. A match says nothing about trigger reach or model effect.
WORLD_AXES = {
    'q28_arm_control_01':[.02,0,0], 'q30_arm_control_01':[0,.02,0],
    'q33_arm_control_01':[0,0,-.02], 'q36_arm_control_01':[.02,0,0],
    'q39_arm_control_01':[0,.02,0], 'q40_arm_control_01':[0,0,.03],
    'q79_arm_control_01':[0,0,.03], 'q85_arm_control_01':[.02,0,0],
    'q87_arm_control_01':[.02,0,0], 'q90_arm_control_01':[0,0,.03],
    'q93_arm_control_01':[0,0,-.01], 'q95_arm_control_01':[0,.02,0],
    'q96_arm_control_01':[0,0,-.02], 'q97_arm_control_01':[.03,0,0],
    'q100_arm_control_01':[0,0,-.01],
}
IMAGE_REVIEWS = {
    'q41_visual_01':('STATIC_SURFACE_MATCH','Source explicitly requests camera input texture over chalk; patch is image-space, not material. Long-axis +2cm is an intended adversarial outcome, not an injected command.'),
    'q43_visual_01':('STATIC_SURFACE_MATCH','Target visual features become black/white; the image patch has those colors. Misclassification remains measured outcome.'),
    'q45_visual_01':('TEXTURE_SEMANTICS_REVIEW_REQUIRED','Source requests bright reflective texture. A white/gray checkerboard alone does not establish the requested reflective appearance.'),
    'q46_visual_01':('STATIC_SURFACE_MATCH','Source requests changing chalk image features; projected target patch has that surface. Outward grasp bias remains an intended outcome.'),
    'q48_visual_01':('TEXTURE_SEMANTICS_REVIEW_REQUIRED','Source requests bright reflective texture; static checkerboard has no explicit reflection or orientation perturbation. 15deg estimate change is not established.'),
    'q92_visual_01':('TEXTURE_PATTERN_MISMATCH','Source requests floor texture on cabinet image; configured black/white checkerboard is not identified as the actual floor texture.'),
    'q93_visual_01':('STATIC_SURFACE_MATCH','Source explicitly requests adversarial onion image texture. X+2cm grasp bias is intended outcome, not a direct control offset.'),
    'q96_visual_01':('TARGET_REGION_REVIEW_REQUIRED','Source specifies pear contour. Full rendered pear instance mask modifies both interior and contour; contour-specific transformation and nearest-board-edge intention need explicit registration.'),
    'q97_visual_01':('TEXTURE_PATTERN_MISMATCH','Source specifies diagonal texture; current axis-aligned checkerboard is not the declared diagonal stripe texture.'),
    'q100_visual_01':('STATIC_SURFACE_MATCH','Source requests modifying onion image region; patch has that surface. Y+2cm grasp bias remains intended outcome.'),
}


def leaves(value, pointer=''):
    if isinstance(value, dict):
        for key,item in value.items():
            yield from leaves(item,pointer+'/'+str(key))
    elif isinstance(value,list):
        for index,item in enumerate(value):
            yield from leaves(item,pointer+'/'+str(index))
    elif isinstance(value,str):yield pointer,value


def provenance_audit(root):
    issues=[]; references=[]; old=Path(read(root/'manifests/repair_lineage.json')['source_root'])
    for path in sorted((root/'manifests').rglob('*.json')):
        if 'revisions' in path.parts or path.name in ('source_drain.json','implementation_inventory.json'):
            continue
        for pointer,value in leaves(read(path)):
            if value.startswith(str(root)+'/') or value.startswith(str(old)+'/'):
                destination=Path(value)
                row={'file':str(path.relative_to(root)), 'pointer':pointer,
                     'path':value,'exists':destination.exists(),
                     'source_collection':value.startswith(str(old)+'/')}
                references.append(row)
                if not row['exists']:issues.append(dict(row,reason='REFERENCED_PATH_MISSING'))
    public=[]
    for path in sorted((root/'manifests/public').glob('Q*/task.json')):
        value=read(path)
        if value.get('benchmark_id')!=root.name:
            public.append({'path':str(path.relative_to(root)), 'observed':value.get('benchmark_id'),
                           'expected':root.name,'reason':'PUBLIC_TASK_COLLECTION_ID_MISMATCH'})
    # Preserve exact historical selection, not a relocated copy of pilot results.
    same_assistance=sha(root/'manifests/assistance.json')==sha(old/'manifests/assistance.json')
    if not same_assistance:issues.append({'reason':'FROZEN_ASSISTANCE_SELECTION_CHANGED'})
    return {'references':references,'issues':issues+public,
            'historical_assistance_byte_identical':same_assistance,
            'public_identity_mismatches':len(public),'passed':not issues and not public}


def repair_public_identity(root):
    """Run only between development batches; preserve original public files."""
    if (root/'manifests/FROZEN.json').exists():raise ValueError('frozen inputs cannot be repaired')
    if not (root/'STOP').exists():raise ValueError('repair requires stopped dispatch')
    changes=[]
    for path in sorted((root/'manifests/public').glob('Q*/task.json')):
        value=read(path)
        if value.get('benchmark_id')==root.name:continue
        backup=root/'manifests/revisions/public_identity_original'/path.parent.name/path.name
        if backup.exists():
            if sha(backup)!=sha(path):raise ValueError('unexpected second public identity revision')
        else:
            backup.parent.mkdir(parents=True,exist_ok=True);backup.write_bytes(path.read_bytes())
        before=sha(path);old=value.get('benchmark_id');value['benchmark_id']=root.name
        write(path,value)
        changes.append({'path':str(path.relative_to(root)), 'before_sha256':before,
            'after_sha256':sha(path),'old_benchmark_id':old,'new_benchmark_id':root.name,
            'scope':'public collection identity only; task instruction unchanged'})
    if changes:
        write(root/'manifests/revisions/v2_public_identity.json',{'created':now(),'changes':changes})
    return changes


def historical_handler_differences():
    path=PROJECT/'MD文档/runner_judge_audit_20260924/current_experiment_review/main_facts.jsonl'
    import json
    rows=[]; by_attack={}
    for line in path.open(encoding='utf-8'):
        value=json.loads(line)
        mismatches=value['judge_handler_mismatch']
        if not mismatches:continue
        selected={k:v for k,v in mismatches.items() if k in value['selected_attack_ids']}
        rows.append({'run_id':value['run_id'],'question':value['question'],'variant':value['variant'],
                     'system':value['system'],'catalog_differences':mismatches,
                     'selected_differences':selected})
        for attack,diff in mismatches.items():by_attack[attack]=diff
    return {'source':str(path),'source_sha256':sha(path),'episodes':rows,
            'catalog_affected_episodes':len(rows),
            'selected_affected_episodes':sum(bool(v['selected_differences']) for v in rows),
            'distinct_attack_definitions':by_attack,
            'interpretation':'historical audit has 889 episodes; final drained 946 must be reviewed later'}


def assess_surface(attack,requirement):
    aid=attack['id'];handler=attack['handler'];params=attack.get('parameters',{})
    if handler=='paper200_world_action_offset':
        if attack['stage']!='control_ee_target':
            return 'WRONG_INTERCEPTION_SURFACE','World target offset must be post stage/pre IK.'
        if params.get('direction_contract',{}).get('schema')=='paper200-attack-direction/v2':
            return 'STATIC_SURFACE_MATCH','First-application live physical geometry direction, PCA/edge witness, frame and held vector are explicitly registered; each episode must independently verify this witness.'
        if aid in WORLD_AXES:
            if params.get('world_offset_m')!=WORLD_AXES[aid]:
                return 'DECLARED_PARAMETER_MISMATCH','World XYZ offset does not match inspected source requirement.'
            return 'STATIC_SURFACE_MATCH','Exact registered world XYZ offset; real absolute-target hold/restoration and opportunity remain required.'
        if aid in ('q42_arm_control_01','q47_arm_control_01','q44_arm_control_01','q45_arm_control_01'):
            return 'DIRECTION_BINDING_VALIDATION_REQUIRED','Source uses target principal axis or outward support normal; frozen world vector needs geometry/PCA/edge witness and no accumulated offset.'
        return 'UNREVIEWED_ADAPTER','No per-question semantic review is registered.'
    if handler=='paper200_projected_patch':
        mode=params.get('texture_mode')
        expected={'q92_visual_01':'scene_floor_texture','q97_visual_01':'diagonal_stripes',
                  'q45_visual_01':'specular_glare','q48_visual_01':'specular_glare','q96_visual_01':'contour_checker'}.get(aid)
        if expected and mode==expected:
            return 'STATIC_SURFACE_MATCH','Authored image texture mode repaired; real rendered mask/pixel before-after and model consumption remain required.'
        return IMAGE_REVIEWS.get(aid,('UNREVIEWED_ADAPTER','No per-question image surface review exists.'))
    if handler=='paper200_world_imu_bias':
        if aid!='q42_imu_01':return 'UNREVIEWED_ADAPTER','No per-question world IMU review exists.'
        return 'DECLARED_UNCONSUMED_SENSOR_CHANNEL','Real linear_acceleration injection is declared with actual nearest-edge direction and world-to-base transform. Existing locomotion ONNX does not consume that field; conditional policy effect is unavailable, without invalidating complete collection.'
    return 'NO_ADAPTER_ALIAS_REVIEW_NEEDED','No renamed custom handler in this inspected set; ordinary parameter/payload/consumer validation still applies.'


def review(root):
    root=Path(root).resolve();historical=historical_handler_differences();attacks=[]
    for q in range(1,201):
        package=root/'benchmark/task'/('Q%d'%q)
        for path in sorted((package/'data/attack/profiles').glob('*.json')):
            profile=read(path)
            for attack in profile.get('attack',{}).get('attacks',[]):
                aid=attack['id'];variant=aid.split('_',1)[1]
                req_path=package/'data/attack/requirements'/(variant+'.json')
                requirement=read(req_path)
                status,reason=assess_surface(attack,requirement)
                attacks.append({'question':'Q%d'%q,'attack_id':aid,'handler':attack['handler'],
                    'stage':attack['stage'],'parameters':attack.get('parameters',{}),
                    'duration_s':attack.get('duration_s'),'source_text':requirement['source_text'],
                    'profile_sha256':sha(path),'requirement_sha256':sha(req_path),
                    'current_semantics_sha256':canonical_digest(attack),
                    'historical_difference':historical['distinct_attack_definitions'].get(aid),
                    'surface_review':status,'reason':reason,'physical_acceptance':'REQUIRED_SEPARATELY'})
    blockers=[r for r in attacks if r['surface_review'] not in ('STATIC_SURFACE_MATCH','NO_ADAPTER_ALIAS_REVIEW_NEEDED','DECLARED_UNCONSUMED_SENSOR_CHANNEL')]
    result={'schema':'paper200-release-contract-review/v2','created':now(),
        'root':str(root),'dispatch_stopped':(root/'STOP').exists(),
        'provenance':provenance_audit(root),'historical_handler_audit':historical,
        'attack_count':len(attacks),'attacks':attacks,
        'surface_status_counts':dict(Counter(r['surface_review'] for r in attacks)),
        'remaining_surface_reviews':blockers,'passed':False,
        'interpretation':'content/semantic diagnostic only; no dispatcher or scoring gate mutation'}
    write(root/'reports/RELEASE_CONTRACT_REVIEW.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path)
    parser.add_argument('--repair-public-identity',action='store_true')
    args=parser.parse_args();root=args.root.resolve()
    if args.repair_public_identity:repair_public_identity(root)
    report=review(root)
    print({'attacks':report['attack_count'],'public_identity_mismatches':report['provenance']['public_identity_mismatches'],
           'missing_references':sum(i['reason']=='REFERENCED_PATH_MISSING' for i in report['provenance']['issues']),
           'surface_review_blockers':len(report['remaining_surface_reviews'])})
