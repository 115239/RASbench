from pathlib import Path
import sys,numpy as np
from .common import read,write
root=Path(sys.argv[1]).resolve()
from isaacsim import SimulationApp
app=SimulationApp({'headless':True,'active_gpu':5,'physics_gpu':5,'multi_gpu':False,'extra_args':['--/app/extensions/fsWatcherEnabled=false']})
from pxr import Usd,UsdGeom,UsdShade
geo=read(root/'manifests/geometry.json')['questions'];result={}
for q in ['Q8','Q108','Q114','Q191','Q192']:
 stage=Usd.Stage.Open(geo[q]['scene']);bbox=UsdGeom.BBoxCache(Usd.TimeCode.Default(),['default','render','proxy']);out=[]
 if q=='Q8':
  for p in stage.Traverse():
   name=str(p.GetPath())
   if p.IsA(UsdGeom.Gprim) and '/Cars/' not in name:
    bound=bbox.ComputeWorldBound(p).ComputeAlignedRange();a=np.array([bound.GetMin(),bound.GetMax()])
    if np.isfinite(a).all() and np.linalg.norm(np.maximum(0,np.maximum(a[0,:2]-[-23.36,-79.52],[-23.36,-79.52]-a[1,:2])))<12 and max(a[1,:2]-a[0,:2])<25:out.append({'prim':name,'aabb':a.tolist()})
 else:
  target=read(root/'benchmark/task'/q/'data/task/task.json')['tracked_objects']['target_0']
  for p in Usd.PrimRange(stage.GetPrimAtPath(target),Usd.TraverseInstanceProxies()):
   if not (p.IsA(UsdGeom.Mesh) or p.IsA(UsdGeom.Subset)):continue
   mat,_=UsdShade.MaterialBindingAPI(p).ComputeBoundMaterial();props={}
   if mat:
    for shader in Usd.PrimRange(mat.GetPrim()):
     for a in shader.GetAttributes():
      if any(s in a.GetName().lower() for s in ['diffuse','albedo','texture','opacity','transmission','color','file']):
       props[str(shader.GetPath())+'.'+a.GetName()]=str(a.Get())
   out.append({'prim':str(p.GetPath()),'type':str(p.GetTypeName()),'material':str(mat.GetPath()) if mat else None,'properties':props})
 result[q]=out
write(root/'reports/scene_details.json',result)
app.close()
