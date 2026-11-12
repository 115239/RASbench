"""Bound the exported OpenEXR worker pools; image values are unchanged."""
import ctypes,os
from pathlib import Path
from .common import write,now


def configure(output,threads=4):
    paths={line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
           if line.endswith(('/libcarb.imaging.plugin.so','/_omni_replicator_exrwriter.so'))}
    records=[]
    for path in sorted(paths):
        library=ctypes.CDLL(path,mode=os.RTLD_NOLOAD|os.RTLD_LOCAL)
        get=library._ZN7Imf_3_217globalThreadCountEv;get.argtypes=[];get.restype=ctypes.c_int
        set_count=library._ZN7Imf_3_220setGlobalThreadCountEi;set_count.argtypes=[ctypes.c_int];set_count.restype=None
        before=get();set_count(threads);after=get()
        if after!=threads:raise RuntimeError('OpenEXR thread-pool setting did not take effect')
        records.append({'library':path,'before':before,'after':after})
    write(output/'CPU_THREAD_POOLS.json',{'time':now(),'pools':records,
          'scope':'OpenEXR worker count only; same decoding/encoding algorithms and values',
          'process_threads':len(list(Path('/proc/self/task').iterdir()))})
