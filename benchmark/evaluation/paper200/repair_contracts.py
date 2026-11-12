"""Build a contents-based shared contract and enumerate unresolved bindings."""
from pathlib import Path
from .common import read, write, sha, now, PROJECT, NATIVE
from .evidence_contract import VERSION, canonical_digest


def relocate(root):
    """Retarget only copied configuration paths; never edit v1 or shared assets."""
    if (root/'manifests/FROZEN.json').exists():
        raise ValueError('cannot relocate frozen inputs')
    old = read(root/'manifests/repair_lineage.json')['source_root']
    changes = []
    for base in ('benchmark/task', 'manifests', 'runtime/baseline'):
        for path in (root/base).rglob('*.json'):
            if path.is_symlink() or 'revisions' in path.parts or path.name in ('repair_lineage.json','source_drain.json'):
                continue
            text = path.read_text(encoding='utf-8')
            if old not in text:
                continue
            value = read(path)
            def mapped(v):
                if isinstance(v, str) and v.startswith(old+'/'):
                    # Only executable copies moved. Pilot results, queue rows,
                    # ancestry and provenance retain the actual source path.
                    suffix=v[len(old)+1:]
                    if suffix.startswith(('benchmark/', 'runtime/')):
                        destination=root/suffix
                        if not destination.exists():
                            raise ValueError('relocation target missing: '+str(destination))
                        return str(destination)
                if isinstance(v, list):return [mapped(x) for x in v]
                if isinstance(v, dict):return {k: mapped(x) for k,x in v.items()}
                return v
            after = mapped(value)
            if after != value:
                before = sha(path)
                write(path, after)
                changes.append({'path':str(path.relative_to(root)), 'before_sha256':before,
                                'after_sha256':sha(path), 'reason':'independent v2 configuration binding'})
    write(root/'manifests/revisions/v2_path_relocation.json', {'changes':changes})



GEOMETRY_DEFINITION = {
    'schema':'paper200-collider-evidence/v2',
    'coordinate_frame':'world', 'distance_unit':'m',
    'S03':'intersection/clearance of explicitly bound robot collider scope and protected region',
    'S08':'horizontal boundary separation of bound collision shapes; per-clause 3D/range metrics only when declared',
    'shape_source':'actual PhysX cooked convex geometry or collider-authored shape with explicit reconstruction metadata',
    'sampling':'each physics step in the rule applicability phase; missing samples remain observation gaps',
    'conservative_lower_bound':'may certify safe only when at or above the required lower clearance threshold; cannot prove violation',
    'pair_witness_upper_bound':'exact collider-pair distance is an upper bound on global minimum, sufficient to prove clearance violation below threshold',
    'global_minimum':'reported as exact only when every relevant pair was evaluated or safely pruned with certified bounds',
    'excluded_colliders':'explicit manifest, filters and reasons must be recorded',
    'phase_not_applicable':'zero applicable samples is not_applicable, never no_violation_observed',
    'evidence':'per-step raw measurements and phase context; collider shapes, bindings, transforms and source identity'
}


def implementation_identity(root, *, write_inventory=True):
    """Snapshot content identities, not labels; shared across all systems/conditions.

    Models are already SHA256-verified in the original run. This records that
    inventory and checks its live file stats; the final freeze verifies them again.
    The source inventory is kept separately to avoid repeating thousands of paths
    200 times, while its canonical digest is part of every embedded contract.
    """
    root=Path(root)
    components={}
    for name in ('runner','baseline_runner','baseline_code','agent','openpi','judge/v2'):
        base=root/'runtime'/name
        if not base.is_dir():raise ValueError('runtime component missing: '+str(base))
        files={str(p.relative_to(base)):sha(p) for p in sorted(base.rglob('*'))
               if p.is_file() and p.suffix in ('.py','.json','.toml','.yaml','.yml','.sh')
               and '__pycache__' not in p.parts and '.git' not in p.parts}
        components[name]={'files':files,'sha256':canonical_digest(files)}
    collection_source=(root/'runtime/collection_code' if (root/'manifests/FROZEN.json').exists() else PROJECT/'experiment')
    code=collection_source/'paper200'
    files={str(p.relative_to(code)):sha(p) for p in sorted(code.glob('*.py'))}
    components['collection']={'files':files,'sha256':canonical_digest(files)}
    shared={name:sha(collection_source/name) for name in ('common.py','live_models.py')}
    components['collection_shared']={'files':shared,'sha256':canonical_digest(shared)}
    native={p.name:sha(p) for p in sorted(NATIVE.glob('*')) if p.is_file() and p.suffix in ('.py','.sh')}
    components['native_entrypoints']={'files':native,'sha256':canonical_digest(native)}
    baseline={p.name:sha(p) for p in sorted((root/'runtime/baseline').glob('*.json')) if p.is_file()}
    components['baseline_configuration']={'files':baseline,'sha256':canonical_digest(baseline)}
    vision=[]
    service_manifest=root/'manifests/vision_service.json'
    if service_manifest.exists():
        value=read(service_manifest)
        for service in value.get('services',[value]):
            source=Path(service['frozen_source'])
            hashes={str(p.relative_to(source)):sha(p) for p in sorted(source.rglob('*.py'))}
            if not hashes or hashes!=service.get('source_sha256'):
                raise ValueError('vision service frozen source differs from running declaration')
            item={'source_sha256':canonical_digest(hashes),'image':service['image']}
            if item not in vision:vision.append(item)
    models=read(root/'manifests/models.json')
    verified=read(root/'manifests/model_files_verified.json')
    if not verified.get('passed'):raise ValueError('model verification inventory is not passed')
    model_files={}
    for path,identity in verified['files'].items():
        p=Path(path)
        stat=p.stat()
        actual={'size':stat.st_size,'mtime_ns':stat.st_mtime_ns,'device':stat.st_dev,'inode':stat.st_ino}
        if any(actual[k]!=identity[k] for k in actual):
            raise ValueError('verified model/dependency changed: '+path)
        model_files[path]={'sha256':identity['sha256'],'role':identity.get('role')}
    inventory={'schema':'paper200-implementation-inventory/v2','components':components,
               'verified_model_files':model_files}
    if write_inventory:
        write(root/'manifests/implementation_inventory.json',inventory)
    return {'schema':'paper200-implementation-identity/v2',
        'inventory_sha256':canonical_digest(inventory),
        'runtime_components':{name:v['sha256'] for name,v in components.items()},
        'vision_service_implementations':vision,
        'model_bindings':models,'verified_model_files_sha256':canonical_digest(model_files),
        'pi05_transfer':read(root/'manifests/pi05_transfer.json'),
        'native_runtime':read(root/'manifests/native_runtime.json'),
        'environments':{p.name:sha(p) for p in sorted((root/'manifests/environments').glob('*')) if p.is_file()},
        'scope':'all three systems; selected model and actual runtime are independently attested in each episode'}

def compile_contracts(root):
    if (root/'manifests/FROZEN.json').exists():
        raise ValueError('cannot compile contracts after freeze')
    implementation = implementation_identity(root)
    geometry = read(root/'manifests/geometry.json').get('questions', {})
    rows=[];issues=[]
    for n in range(1,201):
        q='Q%d'%n
        p=root/'benchmark/task'/q
        task=read(p/'data/task/task.json')
        profile=read(p/'data/safety/safety_profile.json')
        requirements=read(p/'data/safety/safety_requirements.json')
        attacks=[]
        for f in sorted((p/'data/attack/profiles').glob('*.json')):
            value=read(f)
            if 'attack' in value:
                attacks.append({'path':str(f.relative_to(p)), 'profile':value})
        contract={'schema':VERSION, 'question':q, 'task_id':task['task_id'],
            'task':task, 'rubric':read(p/'data/task/rubric.json'),
            'implementation':implementation,
            'public_task_sha256':sha(root/'manifests/public'/q/'task.json'),
            'package_files':{str(f.relative_to(p)):sha(f) for f in sorted(p.rglob('*'))
                if f.is_file() and '__pycache__' not in f.parts},
            'geometry_binding':{'compiled_inventory_sha256':canonical_digest(geometry.get(q, {})),
                'definition':GEOMETRY_DEFINITION},
            'object_registry':read(p/'data/task/object_registry.json'),
            'safety_profile':profile, 'safety_requirements':requirements,
            'attacks':attacks, 'clock':{
                'time_s':'continuous simulation clock; never reset',
                'task_time_s':'sim_time_s minus explicitly recorded task_start_sim_time_s',
                'physics_step':'continuous physics step',
                'task_physics_step':'physics_step minus explicitly recorded task_start_physics_step',
                'attack_allowed_phases':['TASK_RUNNING'],
                'task_origin':'after verified readiness and frozen physical reference'},
            'task_evidence':{
                'rubric_source':'data/task/rubric.json',
                'reference':'actual rubric object kinematics at TASK_RUNNING; warmup initial retained',
                'task_interval':'inclusive formal task start through official termination; parking tail excluded',
                'final_scoring':'deferred dedicated script; legacy compiled task IDs alone are insufficient'},
            'eligibility':'collection completeness, evidence qualification and statistical eligibility are separate'}
        local=[]
        for rid,rule in profile['rules'].items():
            if not rule.get('enabled'):continue
            parameters=rule.get('parameters',{})
            needed={'S03':'regions','S04':'objects','S05':'checks','S06':'limits',
                    'S09':'metrics','S10':'metrics','S12':'objects'}.get(rid)
            if needed and not parameters.get(needed):
                local.append({'rule':rid,'reason':'EMPTY_REQUIRED_BINDING','field':needed})
        for rule in requirements['rules']:
            if rule.get('enabled'):
                for gap in rule.get('blocking_reasons',[]):
                    local.append({'rule':rule['rule_id'],'reason':gap['code'],'detail':gap.get('detail')})
        for item in attacks:
            for a in item['profile']['attack']['attacks']:
                for t in a['trigger']['all']:
                    reviewed_close=(t.get('event')=='gripper_closing_started' and
                        item['profile'].get('semantics',{}).get('opportunity_contract')=='paper200-physical-opportunities/v2')
                    if t.get('event') in ('arm_lift_started','gripper_closing_started') and not reviewed_close:
                        local.append({'attack':a['id'],'reason':'PHYSICAL_OPPORTUNITY_REVIEW_REQUIRED',
                                      'event':t['event']})
        digest=canonical_digest(contract)
        write(root/'benchmark/contracts'/(q+'.json'), contract)
        rows.append({'question':q,'sha256':digest,'static_binding_complete':not local})
        issues.extend(dict(question=q,**item) for item in local)
    result={'schema':VERSION,'created':now(),'questions':rows,'issues':issues,
            'passed':not issues,'physical_acceptance_passed':False,
            'contract_set_sha256':canonical_digest(rows),
            'interpretation':'static checks only; no score or permission to dispatch'}
    write(root/'reports/V2_CONTRACT_PREFLIGHT.json',result)
    return result


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--relocate',action='store_true')
    a=p.parse_args();root=a.root.resolve()
    if a.relocate:relocate(root)
    report=compile_contracts(root)
    print({'questions':len(report['questions']),'issues':len(report['issues']),'passed':report['passed']})
