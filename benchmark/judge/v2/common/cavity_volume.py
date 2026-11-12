"""Closed physical cavity surface tests; supports concave vessel interiors."""
import numpy as np
from scipy.spatial import ConvexHull

TOL=1e-8 # arithmetic tolerance on local mesh coordinates, never an S08 margin

def point_inside(point,triangles):
    p=np.asarray(point,float);t=np.asarray(triangles,float);a=t[:,0]-p;b=t[:,1]-p;c=t[:,2]-p
    # A point on an actual triangle is on the permitted cavity boundary.
    normal=np.cross(b-a,c-a);norm=np.linalg.norm(normal,axis=1)
    for tri,n,length in zip(t,normal,norm):
        if length<=1e-14 or abs(np.dot(p-tri[0],n))/length>TOL:continue
        v0=tri[1]-tri[0];v1=tri[2]-tri[0];v2=p-tri[0];mat=np.array([[v0@v0,v0@v1],[v0@v1,v1@v1]])
        if abs(np.linalg.det(mat))<1e-20:continue
        uv=np.linalg.solve(mat,[v2@v0,v2@v1])
        if np.all(uv>=-TOL) and uv.sum()<=1+TOL:return True
    la=np.linalg.norm(a,axis=1);lb=np.linalg.norm(b,axis=1);lc=np.linalg.norm(c,axis=1)
    numerator=np.einsum('ij,ij->i',a,np.cross(b,c))
    denominator=la*lb*lc+np.einsum('ij,ij->i',a,b)*lc+np.einsum('ij,ij->i',b,c)*la+np.einsum('ij,ij->i',c,a)*lb
    winding=float((2*np.arctan2(numerator,denominator)).sum())/(4*np.pi)
    return abs(winding)>.5

def triangle_cuts_convex(triangle,planes):
    """Clip the boundary triangle to the strict interior of a convex part."""
    polygon=list(np.asarray(triangle,float))
    for plane in planes:
        if not polygon:return False
        out=[];previous=polygon[-1];dp=float(previous@plane[:3]+plane[3]+TOL)
        for current in polygon:
            dc=float(current@plane[:3]+plane[3]+TOL)
            if (dc<0)!=(dp<0):out.append(previous+(current-previous)*(dp/(dp-dc)))
            if dc<0:out.append(current)
            previous=current;dp=dc
        polygon=out
    if len(polygon)<3:return False
    area=sum(np.linalg.norm(np.cross(polygon[j]-polygon[0],polygon[j+1]-polygon[0])) for j in range(1,len(polygon)-1))/2
    return bool(area>TOL*TOL)

def convex_inside(vertices,triangles):
    points=np.unique(np.asarray(vertices,float),axis=0)
    if not all(point_inside(p,triangles) for p in points):return False
    hull=ConvexHull(points)
    # Vertex-only containment is insufficient at concave corners: a part can
    # bridge over a cavity wall even when all of its vertices are inside.
    return not any(triangle_cuts_convex(t,hull.equations) for t in triangles)

def component_point_inside(point,components,axis,offset):
    height=float(np.dot(point,axis)+offset)
    active=[c for c in components if c['minimum_height']-TOL<=height<=c['maximum_height']+TOL]
    return bool(active) and all(point_inside(point,c['triangles_local']) for c in active)

def convex_inside_components(vertices,components,axis,offset):
    points=np.unique(np.asarray(vertices,float),axis=0)
    if not all(component_point_inside(p,components,axis,offset) for p in points):return False
    hull=ConvexHull(points)
    # Component closure caps are membership bookkeeping only. An object may
    # cross an overlapping body/rim component's cap while remaining in the bin.
    return not any(triangle_cuts_convex(t,hull.equations) for c in components for t in c['actual_boundary_local'])
