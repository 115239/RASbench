"""Read-only verification of actual rendered image attacks and neural inputs."""
import argparse,json
from pathlib import Path
import h5py,numpy as np
from .common import read,write,sha,now
from .attack_proof import verify_effect,text


def check(out):
    out=Path(out).resolve();ep=out/'episode';scope=read(ep/'VISUAL_CONTROL.json');issues=[]
    original=scope['original_attack'];diagnostic=scope['diagnostic_attack']
    for field in ('id','handler','stage','parameters','duration_s'):
        if original.get(field)!=diagnostic.get(field):issues.append('ORIGINAL_PAYLOAD_OR_DURATION_CHANGED:'+field)
    if scope['original_condition_opportunity_eligible'] is not False or scope['formal_or_assisted_matrix_member'] is not False:
        issues.append('CONTROL_SCOPE_NOT_EXPLICIT')
    source=ep/'submission.hdf5';before=sha(source)
    with h5py.File(source,'r') as f:
        found=[]
        def visit(name,obj):
            if isinstance(obj,h5py.Group) and name.endswith('/'+original['id']) and 'applications' in obj:found.append(obj)
        f.visititems(visit)
        if len(found)!=1:raise ValueError('selected application group is not unique')
        proof=verify_effect(f,found[0],original['stage'],diagnostic)
        if not proof['passed']:issues.extend(proof['issues'])
        collection=f['extensions/paper200_collection'];samples=[]
        for record in collection['visual_masks'].values():
            meta=json.loads(text(record['metadata_json'][()]))
            if meta.get('attack_id')!=original['id']:continue
            a=record['image_before'][...];b=record['image_after'][...];mask=record['mask'][...]
            changed=np.any(a!=b,axis=-1);outside=int(np.count_nonzero(changed & ~mask))
            if outside:issues.append('PIXELS_CHANGED_OUTSIDE_DECLARED_MASK')
            samples.append({'physics_step':meta['physics_step'],'visible_mask_pixels':int(mask.sum()),
                'changed_pixels':int(changed.sum()),'outside_mask_changed_pixels':outside,
                'original_pixel_sha256':meta['image_before_sha256'],'attacked_pixel_sha256':meta['image_after_sha256']})
        if not samples or not any(s['changed_pixels']>0 for s in samples):issues.append('NO_REAL_VISIBLE_TARGET_MUTATION')
        forwards=[]
        for key,record in collection['model_boundaries'].items():
            if 'neural_receipt_json' not in record:continue
            receipt=json.loads(text(record['neural_receipt_json'][()]))
            forwards.append({'request':key,'verified':bool(record.attrs.get('neural_evidence_verified',False)),
                'forward_executed':receipt['forward_executed'],'consumed_front_request_ids':receipt['consumed_request_ids']['front'],
                'raw_output_sha256':receipt['raw_output_sha256']})
        if not any(r['verified'] and r['forward_executed'] for r in forwards):issues.append('NO_ACTUAL_NEURAL_FORWARD')
    if sha(source)!=before:raise RuntimeError('source changed during read-only verification')
    result={'schema':'paper200-real-visual-control-acceptance/v2','checked':now(),'passed':not issues,
        'issues':sorted(set(issues)),'scope':'actual USD renderer, declared image transformation, actual neural input and raw output',
        'original_task_conditions':False,'original_trigger_acceptance':False,'formal_task_score':None,
        'attack':original['id'],'parameters':original['parameters'],'camera':scope['camera'],
        'samples':samples,'neural_forwards':forwards,'strict_effect_proof':proof,
        'source_hdf5_sha256':before,'source_unchanged':True,'control_scope_sha256':sha(ep/'VISUAL_CONTROL.json')}
    write(out/'VISUAL_CONTROL_ACCEPTANCE.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('attempt',type=Path);a=p.parse_args()
    r=check(a.attempt);print('VISUAL_CONTROL_ACCEPTANCE',r['passed'],r['issues'])
