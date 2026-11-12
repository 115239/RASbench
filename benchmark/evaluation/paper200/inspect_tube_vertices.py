from pathlib import Path
from .common import read,write
ROOT=Path(__file__).resolve().parents[2]/'datasets/paper200/collection_20260924_v2'
from isaacsim import SimulationApp
app=SimulationApp({'headless':True,'active_gpu':0,'physics_gpu':0,'multi_gpu':False,'extra_args':['--ext-folder',str(ROOT/'runtime/native_extensions'),'--enable','omni.client.lib-1.1.1','--/plugins/carb.tasking.plugin/threadCount=8']})
try:
 from pxr import Usd,UsdGeom
 import numpy as np
 stage=Usd.Stage.Open(read(ROOT/'manifests/geometry.json')['questions']['Q130']['scene'])
 p=stage.GetPrimAtPath('/World/Starting_point_01/task/SM_ShampooTube_33/SM_ShampooTube_02');mesh=UsdGeom.Mesh(p)
 pts=np.array(mesh.GetPointsAttr().Get());matrix=np.array(UsdGeom.XformCache().GetLocalToWorldTransform(p));world=np.c_[pts,np.ones(len(pts))]@matrix
 write(ROOT/'reports/TUBE_VERTICES.json',{'points_world':world[:,:3].tolist(),'indices':list(mesh.GetFaceVertexIndicesAttr().Get()),'counts':list(mesh.GetFaceVertexCountsAttr().Get()),'local_points':pts.tolist(),'matrix':matrix.tolist()})
 print('done',flush=True)
finally:app.close()
