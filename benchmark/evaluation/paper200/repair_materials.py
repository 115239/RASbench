"""Restore missing dependencies from byte-identical copies of the same asset family."""
from pathlib import Path
import re,shutil
from .common import read,write,sha,now

def restore(root):
    if (root/'manifests/FROZEN.json').exists():raise ValueError('frozen inputs cannot be repaired')
    family=root/'benchmark/assets/floor8_external/MarketGen/Goods/Props/Snacks'
    names=['OmniUe4Base.mdl','OmniUe4Function.mdl','TX_Bag_Display_NN_01a_RMA.png']
    canonical={};records=[]
    for name in names:
        sources=sorted(family.rglob(name));hashes={sha(p) for p in sources}
        if len(hashes)!=1:raise ValueError('ambiguous dependency source '+name)
        canonical[name]=(sources[0],next(iter(hashes)),len(sources))
    for mdl in sorted(family.glob('SM_Bag_Display_NN_*/MI_Bag_Display_NN_01a.mdl')):
        content=mdl.read_text(encoding='utf-8-sig')
        for name in names:
            relative=name if name.endswith('.mdl') else 'MI_Bag_Display_NN_01a/'+name
            used=('::'+name[:-4]) in content if name.endswith('.mdl') else relative in content
            destination=mdl.parent/relative
            if not used or destination.exists():continue
            source,digest,count=canonical[name];destination.parent.mkdir(parents=True,exist_ok=True)
            temporary=destination.with_suffix(destination.suffix+'.restore');shutil.copy2(source,temporary);temporary.replace(destination)
            records.append({'restored':str(destination.relative_to(root)),'source':str(source.relative_to(root)),
                            'sha256':digest,'identical_existing_copies':count,'previously_missing':True})
    report=root/'manifests/revisions/material_dependencies.json'
    previous=read(report).get('restored',[]) if report.exists() else []
    value={'time':now(),'restored':previous+records,'method':'copy matching named dependency after all existing asset-family copies have equal SHA256',
           'synthetic_materials':False,'applies_to':'all systems and all variants sharing floor8 assets'}
    write(report,value);return value

if __name__=='__main__':
    import sys,json
    print(json.dumps(restore(Path(sys.argv[1]).resolve()),indent=2))
