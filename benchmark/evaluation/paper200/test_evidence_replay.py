"""Replay retained real physical samples through both lossless writers."""
import argparse,json,tempfile
from pathlib import Path
import h5py,numpy as np
from rubric_evidence_io import RubricEvidenceWriter as Original,ROOT,object_path,contact_path
from .physics_evidence import RubricEvidenceWriter as Buffered
from .common import write


def replay(source,output):
    output.mkdir(parents=True,exist_ok=True)
    with h5py.File(source,'r') as src,tempfile.TemporaryDirectory(dir=output) as work:
        manifest=json.loads(bytes(src[ROOT+'/manifest_json'][...]).decode())
        count=min(300,len(src[ROOT+'/sim_time_s']))
        if count<257:raise ValueError('test must cross a 256-step flush boundary')
        paths=[]
        for name,cls in [('reference',Original),('buffered',Buffered)]:
            path=Path(work)/(name+'.hdf5');paths.append(path);w=cls(path,manifest)
            w.initial([{k:v[...] for k,v in src[object_path(o['prim'])+'/initial'].items()} for o in manifest['objects']])
            for i in range(count):
                rows=[{k:v[i] for k,v in src[object_path(o['prim'])].items() if isinstance(v,h5py.Dataset)} for o in manifest['objects']]
                contacts={'physics_steps':int(src[ROOT+'/physics_steps'][i])}
                for field in ['net_forces_w','impulse_w','active_fraction']:
                    contacts[field]=np.asarray([src[contact_path(p['source'],p['target'])+'/'+field][i].reshape(-1) if field!='active_fraction'
                        else src[contact_path(p['source'],p['target'])+'/'+field][i] for p in manifest['contacts']])
                w.step(i,float(src[ROOT+'/sim_time_s'][i]),rows,contacts)
            w.complete(count);w.close()
        checked=[]
        with h5py.File(paths[0],'r') as a,h5py.File(paths[1],'r') as b:
            def compare(name,obj):
                if isinstance(obj,h5py.Dataset):
                    if obj.dtype!=b[name].dtype or not np.array_equal(obj[...],b[name][...]):raise AssertionError(name)
                    checked.append(name)
            a.visititems(compare)
        result={'passed':True,'source':str(source),'physical_samples':count,'datasets_bitwise_equal':len(checked),
                'scope':'lossless recording optimization; no task success or scoring claim'}
        write(output/'BUFFERED_WRITER_REPLAY.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(json.dumps(replay(a.source,a.output)))
