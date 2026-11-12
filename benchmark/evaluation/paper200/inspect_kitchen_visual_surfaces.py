"""Static USD material/geometry survey for Q92 image attack bindings."""
from pathlib import Path
from .common import read,write
ROOT=Path(__file__).resolve().parents[2]/'datasets/paper200/collection_20260924_v2'
from isaacsim import SimulationApp
app=SimulationApp({'headless':True,'active_gpu':0,'physics_gpu':0,'multi_gpu':False,'extra_args':['--ext-folder',str(ROOT/'runtime/native_extensions'),'--enable','omni.client.lib-1.1.1','--/plugins/carb.tasking.plugin/threadCount=8']})
try:
 from pxr import Usd,UsdGeom,UsdShade
 import numpy as np
 stage=Usd.Stage.Open(read(ROOT/'manifests/geometry.json')['questions']['Q92']['scene'])
 cache=UsdGeom.BBoxCache(Usd.TimeCode.Default(),[UsdGeom.Tokens.default_,UsdGeom.Tokens.render,UsdGeom.Tokens.proxy]);rows=[]
 for p in stage.Traverse():
  path=str(p.GetPath())
  if not p.IsA(UsdGeom.Mesh) or not any(k in path for k in ('architecture_kitchen_living_room','kitchen_cabinet_and_island')):continue
  box=cache.ComputeWorldBound(p).ComputeAlignedRange();mat=UsdShade.MaterialBindingAPI(p).ComputeBoundMaterial()[0]
  attrs=[]
  if mat:
   for child in Usd.PrimRange(mat.GetPrim()):
    for attr in child.GetAttributes():
     val=attr.Get()
     if val is not None and any(k in attr.GetName().lower() for k in ('file','texture','diffuse','color')):
      attrs.append({'prim':str(child.GetPath()),'attribute':attr.GetName(),'value':str(val),'resolved':getattr(val,'resolvedPath',None)})
  rows.append({'prim':path,'world_aabb':[list(box.GetMin()),list(box.GetMax())], 'material':str(mat.GetPath()) if mat else None,'material_attributes':attrs})
 write(ROOT/'reports/KITCHEN_VISUAL_SURFACES.json',{'rows':rows});print('meshes',len(rows),flush=True)
finally:app.close()
