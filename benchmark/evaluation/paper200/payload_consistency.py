"""Check application summaries against actual saved values, including digests."""
import hashlib
import numpy as np


def matches(record,value):
    if isinstance(record,dict) and record.get('type')=='ndarray':
        try:
            array=np.asarray(value,dtype=record['dtype'])
            if list(array.shape)!=list(record['shape']):return False
            if 'sha256' in record and hashlib.sha256(array.tobytes()).hexdigest()!=record['sha256']:return False
            if 'values' in record and not np.array_equal(array,np.asarray(record['values'],dtype=array.dtype)):return False
            return True
        except (KeyError,ValueError,TypeError):return False
    if isinstance(record,dict):
        if not isinstance(value,dict):return False
        keys=set(record)-{'attack_metadata'}
        return keys==set(value)-{'attack_metadata'} and all(matches(record[k],value[k]) for k in keys)
    if isinstance(record,str) or isinstance(value,str):return record==value
    try:
        a=np.asarray(record);b=np.asarray(value)
        if a.dtype.kind in 'biuf' and b.dtype.kind in 'biuf':return a.shape==b.shape and np.allclose(a,b,rtol=0,atol=1e-7,equal_nan=False)
        return np.array_equal(a,b)
    except (TypeError,ValueError):return record==value
