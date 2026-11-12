"""Bind all persistent material profiles to their existing texture file bytes."""
import argparse,copy,shutil
from pathlib import Path
from .common import read,write,sha,now


def repair(root):
    root=Path(root).resolve()
    if not (root/'STOP').exists() or (root/'manifests/FROZEN.json').exists():
        raise ValueError('material binding repair requires stopped unfrozen candidate')
    plans=[]
    for path in sorted((root/'benchmark/task').glob('*/data/attack/profiles/*.json')):
        original=read(path);value=copy.deepcopy(original);bindings=[]
        package=path.parents[3]
        for attack in value.get('attack',{}).get('attacks',[]):
            if attack.get('stage')!='immediate_visual_material':continue
            if attack['handler']!='install_visual_material_override':raise ValueError('unexpected material handler')
            parameters=attack['parameters'];relative=Path(parameters['texture_asset'])
            if relative.is_absolute() or '..' in relative.parts:raise ValueError('texture outside package')
            asset=(package/relative).resolve(strict=True);asset.relative_to(package.resolve())
            digest=sha(asset)
            if parameters.get('texture_sha256') not in (None,digest):raise ValueError('existing texture hash disagrees')
            parameters['texture_sha256']=digest
            bindings.append({'attack_id':attack['id'],'texture_asset':str(asset.relative_to(root)),
                'texture_sha256':digest,'texture_bytes':asset.stat().st_size})
        if bindings:plans.append((path,original,value,bindings))
    if len(plans)!=60 or sum(len(p[3]) for p in plans)!=60:raise ValueError('expected exactly 60 material profiles')
    records=[]
    for path,original,value,bindings in plans:
        before=sha(path);changed=value!=original
        backup=root/'manifests/revisions/material_hash_original'/path.relative_to(root/'benchmark/task')
        if changed:
            if backup.exists():raise ValueError('original backup already exists for changed profile')
            backup.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,backup);write(path,value)
        verify=read(path)
        for attack in verify['attack']['attacks']:
            if attack.get('stage')=='immediate_visual_material':
                attack['parameters'].pop('texture_sha256',None)
        prior=copy.deepcopy(original)
        for attack in prior['attack']['attacks']:
            if attack.get('stage')=='immediate_visual_material':attack['parameters'].pop('texture_sha256',None)
        if prior!=verify:raise AssertionError('repair changed material semantics')
        records.append({'profile':str(path.relative_to(root)),'before_sha256':before,'after_sha256':sha(path),
            'original_profile':str(backup.relative_to(root)) if backup.exists() else None,
            'changed_this_run':changed,'bindings':bindings,'only_added_parameter':'texture_sha256',
            'all_other_fields_equal':True})
    report=root/'manifests/revisions/material_texture_hash_binding.json'
    result={'schema':'paper200-material-texture-content-binding/v2','created':now(),'profiles':60,
        'changed':sum(r['changed_this_run'] for r in records),'source':'existing packaged texture bytes; no texture changes',
        'scope':'same expected texture content for Runner and Judge across all three systems',
        'trigger_duration_handler_channels_unchanged':True,'records':records}
    if result['changed'] or not report.exists():write(report,result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();r=repair(a.root)
    print('MATERIAL_TEXTURE_BINDING',r['profiles'],r['changed'])
