"""Select an isolated client-library extension, preserving the shared Isaac environment."""
from pathlib import Path
import os
from .common import read,write,now

def configure(args):
    cfg=read(os.environ['THREE_SCENES_RUN_CONFIG']);binding=cfg.get('native_runtime')
    if not binding or not binding.get('enabled'):return
    folder=Path(cfg['collection_root'])/'runtime/native_extensions'
    args.kit_args+=' --ext-folder '+str(folder)+' --enable '+binding['extension']

def verify(output):
    cfg=read(os.environ['THREE_SCENES_RUN_CONFIG']);binding=cfg.get('native_runtime')
    if not binding or not binding.get('enabled'):return
    import omni.client
    libraries=sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                      if 'libomniclient.so' in line or 'libomniverse_connection.so' in line})
    version=omni.client.get_version()
    if not version.startswith(binding['client_version']+'-') or len(libraries)!=2 or not all('/runtime/native_extensions/' in p for p in libraries):
        raise RuntimeError('isolated native client identity mismatch')
    write(output/'NATIVE_RUNTIME.json',{'time':now(),'version':version,'libraries':libraries,
                                      'shared_environment_modified':False,'verified':True})
