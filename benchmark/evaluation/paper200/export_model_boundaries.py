import json
import hashlib
"""Embed actual per-request before/after/consumed inputs and returned outputs."""
import h5py
import numpy as np


def embed(collection,episode):
    root=collection.create_group('model_boundaries');root.attrs['complete']=False
    directory=episode/'attack_evidence/raw_payloads/vla_requests'
    ids=[]
    for path in sorted(directory.glob('request_*.npz')):
        with np.load(path,allow_pickle=False) as record:
            rid=int(record['request_id']);ids.append(rid)
            group=root.create_group(str(rid));group.attrs.update(request_id=rid,
                physics_step=int(record['physics_step']),sim_time_s=float(record['time_s']),
                receipt_status='inference_returned')
            fields={'prompt':('original_prompt','attacked_prompt','model_consumed_prompt'),
                'state':('original_state','attacked_state','model_consumed_state'),
                'front':('original_m20_camera_0_rgb','forwarded_m20_camera_0_rgb','model_consumed_image'),
                'rear':('original_m20_camera_1_rgb','forwarded_m20_camera_1_rgb','model_consumed_rear_image'),
                'wrist':('original_piper_wrist_rgb','forwarded_piper_wrist_rgb','model_consumed_wrist_image')}
            for name,keys in fields.items():
                target=group.create_group(name)
                for label,key in zip(('original','attacked','consumed'),keys):
                    if label=='consumed' and key not in record:
                        target[label]=target['attacked']
                        target.attrs['consumed_source']='direct forwarded request; Agent prepare_request not active'
                        continue
                    value=record[key]
                    if value.dtype.kind=='U':
                        target.create_dataset(label,data=np.asarray(value,dtype=h5py.string_dtype('utf-8')))
                    else:target.create_dataset(label,data=value,compression='gzip' if value.ndim else None)
            response=json.loads(str(record['raw_response_json'])) if 'raw_response_json' in record else {}
            receipt=response.get('_paper_neural_receipt')
            group.attrs['neural_evidence_verified']=False
            if receipt is not None:
                group.create_dataset('neural_receipt_json',data=np.frombuffer(json.dumps(receipt,ensure_ascii=False).encode(),np.uint8))
                call=int(receipt['call_id']);model_root=(episode.parent/'model').resolve()
                wire=model_root/('input_%06d.msgpack'%call)
                wire_ok=wire.is_file() and hashlib.sha256(wire.read_bytes()).hexdigest()==receipt.get('input_wire_sha256')
                output_ok=not receipt.get('forward_executed')
                raw=receipt.get('raw_output_path')
                if raw:
                    path=__import__('pathlib').Path(raw)
                    path=(path if path.is_absolute() else model_root/path).resolve()
                    if model_root in path.parents and path.is_file():
                        data=path.read_bytes();digest=hashlib.sha256(data).hexdigest()
                        output_ok=digest==receipt.get('raw_output_sha256')
                        group.create_dataset('neural_raw_output_npz',data=np.frombuffer(data,np.uint8),compression='gzip')
                        group.attrs['neural_raw_output_sha256']=digest
                group.attrs['neural_evidence_verified']=bool(wire_ok and output_ok and call==rid)
                group.attrs['neural_forward_executed']=bool(receipt.get('forward_executed'))
            group.create_dataset('raw_actions',data=record['vla_raw_actions'],compression='gzip')
            group.create_dataset('returned_actions',data=record['vla_attacked_actions'],compression='gzip')
    if len(ids)!=len(set(ids)):raise ValueError('duplicated model request IDs')
    root.attrs.update(complete=True,request_count=len(ids))
    payloads=collection.create_group('interception_payloads')
    for path in sorted((episode/'attack_evidence/raw_payloads').glob('*.jsonl')):
        if path.name=='consumer_receipts.jsonl':continue
        dataset=payloads.create_dataset(path.stem,shape=(0,),maxshape=(None,),dtype=np.uint8,
            chunks=(1048576,),compression='gzip')
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(1048576),b''):
                start=len(dataset);dataset.resize(start+len(block),axis=0);dataset[start:]=np.frombuffer(block,np.uint8)
        dataset.attrs['source_file']=str(path.relative_to(episode))

    visual=collection.create_group('visual_masks')
    for path in sorted((episode/'attack_evidence/visual_masks').glob('*.npz')):
        group=visual.create_group(path.stem)
        with np.load(path,allow_pickle=False) as record:
            for key in record.files:
                value=record[key]
                if value.dtype.kind=='U':value=np.asarray(value,dtype=h5py.string_dtype('utf-8'))
                group.create_dataset(key,data=value,compression='gzip' if value.ndim else None)
        metadata=path.with_suffix('.json')
        if metadata.exists() and 'metadata_json' not in group:group.create_dataset('metadata_json',data=np.frombuffer(metadata.read_bytes(),np.uint8))
    for source,name in (('sensor_lidar','lidar_payloads'),('lidar_consumers','lidar_consumers')):
        target=collection.create_group(name)
        for path in sorted((episode/'attack_evidence/raw_payloads'/source).glob('*.npz')):
            group=target.create_group(path.stem)
            with np.load(path,allow_pickle=False) as record:
                for key in record.files:
                    value=record[key]
                    if value.dtype.kind=='U':value=np.asarray(value,dtype=h5py.string_dtype('utf-8'))
                    group.create_dataset(key,data=value,compression='gzip' if value.ndim else None)
