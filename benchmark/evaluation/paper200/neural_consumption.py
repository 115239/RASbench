"""Join actual neural forwards to their source request, including NAV buffers."""
import hashlib,json
import numpy as np


def receipt(request,text):
    if 'neural_receipt_json' not in request:return None
    value=json.loads(text(request['neural_receipt_json'][()]))
    return value if request.attrs.get('neural_evidence_verified',False) else None


def model_use(boundaries,request,modality,text,changed_indices=None):
    """A true service response is distinct from a true neural forward."""
    rid=int(request.attrs['request_id']);service=receipt(request,text)
    if service is None:return {'service_verified':False,'neural_verified':False,'reason':'NEURAL_EXECUTION_RECEIPT_MISSING'}
    for consumer in boundaries.values():
        forward=receipt(consumer,text)
        if not forward or not forward.get('forward_executed'):continue
        for source in forward.get('consumed_sources',{}).get(modality,[]):
            if source.get('request_id')!=rid:continue
            try:
                actual=request[modality+'/consumed'][()]
                if modality=='prompt':data=text(actual).encode('utf-8')
                else:data=np.ascontiguousarray(actual).tobytes()
                if hashlib.sha256(data).hexdigest()!=source['value_sha256']:continue
                fields={'front':'observation/m20_camera_0_rgb','rear':'observation/m20_camera_1_rgb',
                    'wrist':'observation/piper_wrist_rgb','state':'observation/state','prompt':'prompt'}
                if source.get('source_field')!=fields[modality]:continue
                shape=[] if modality=='prompt' else list(np.asarray(actual).shape)
                dtype='utf8' if modality=='prompt' else str(np.asarray(actual).dtype)
                if source.get('shape')!=shape or source.get('dtype')!=dtype:continue
                if modality=='state':
                    indices=np.asarray(forward.get('state_indices',[]),int)
                    if len(indices)==0:continue
                    if changed_indices is None:
                        before=request['state/original'][...];after=request['state/attacked'][...]
                        changed_indices=np.flatnonzero(before!=after)
                    if not set(indices.tolist()).intersection(np.asarray(changed_indices,int).tolist()):continue
                return {'service_verified':True,'neural_verified':True,'neural_call_id':forward['call_id'],
                    'source_request_id':rid,'modality':modality,'state_indices':forward.get('state_indices',[]) if modality=='state' else None,
                    'consumer_path':consumer.name}
            except (KeyError,ValueError,TypeError,IndexError):continue
    return {'service_verified':True,'neural_verified':False,'source_request_id':rid,
        'reason':'REQUEST_MODALITY_NOT_USED_BY_RECORDED_NEURAL_FORWARD','service_decision':service.get('service_decision')}
