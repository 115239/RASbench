"""Lossless, compressed per-step rule measurements for independent Judge audit."""
import json
import math
from pathlib import Path
import h5py
import numpy as np

def safe(value):
    if isinstance(value,dict): return {str(k):safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [safe(v) for v in value]
    if isinstance(value,np.ndarray): return safe(value.tolist())
    if isinstance(value,np.generic): return safe(value.item())
    if isinstance(value,float) and not math.isfinite(value):return {'nonfinite_numeric':repr(value)}
    return value

class ObservationWriter:
    def __init__(self,path,profile_sha256):
        self.file=h5py.File(Path(path),'x');self.pending=[];self.last_step=None
        self.file.attrs.update(schema='runner-v0925-safety-observations/1',complete=False,
                               profile_sha256=profile_sha256)
        self.file.create_dataset('json_bytes',shape=(0,),maxshape=(None,),dtype='u1',chunks=(262144,),compression='gzip')
        self.file.create_dataset('offsets',data=np.array([0],np.int64),maxshape=(None,),chunks=(256,),compression='gzip')
    def context(self,name,value):
        raw=json.dumps(safe(value),ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()
        self.file.create_dataset(name,data=np.frombuffer(raw,np.uint8),compression='gzip')
    def append(self,point,rules,sample=None):
        step=point['physics_step']
        if self.last_step is not None and step!=self.last_step+1:raise ValueError('safety observation step gap')
        row={'point':point,'rules':rules,'sample':sample}
        self.pending.append(json.dumps(safe(row),ensure_ascii=False,separators=(',',':'),allow_nan=False).encode())
        self.last_step=step
        if len(self.pending)>=256:self.flush()
    def flush(self):
        if not self.pending:return
        f=self.file;n=len(f['json_bytes']);payload=b''.join(self.pending)
        f['json_bytes'].resize(n+len(payload),axis=0);f['json_bytes'][n:]=np.frombuffer(payload,np.uint8)
        offsets=n+np.cumsum([len(x) for x in self.pending]);count=len(f['offsets'])
        f['offsets'].resize(count+len(offsets),axis=0);f['offsets'][count:]=offsets
        self.pending.clear()
    def close(self):
        self.flush();self.file.attrs['complete']=True;self.file.close()
