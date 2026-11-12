"""Freeze missing rubric parameters and add the authored but absent rover fence."""
from pathlib import Path
from .common import read,write,sha,now


def bind(root):
    geometry=read(root/'manifests/geometry.json')['questions']
    original=Path(geometry['Q8']['scene']);overlay=original.with_name('paper200_factory_fence_v1.usda')
    center=geometry['Q8']['objects']['/World/Cars/c_01']['translation'][:2];cx,cy=center
    parts=[]
    def cube(name,xyz,scale):
        parts.append('''    def Cube "%s" (prepend apiSchemas = ["PhysicsCollisionAPI"]) {
        double size = 1
        color3f[] primvars:displayColor = [(0.95, 0.65, 0.08)]
        bool physics:collisionEnabled = true
        double3 xformOp:translate = (%s)
        double3 xformOp:scale = (%s)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
    }'''%(name,', '.join(map(str,xyz)),', '.join(map(str,scale))))
    for level,z in enumerate([.38,.8]):
        for name,xyz,scale in [('West',[cx-2.5,cy,z],[.08,5.,.08]),('East',[cx+2.5,cy,z],[.08,5.,.08]),
                               ('South',[cx,cy-2.5,z],[5.,.08,.08]),('NorthLeft',[cx-1.75,cy+2.5,z],[1.5,.08,.08]),
                               ('NorthRight',[cx+1.75,cy+2.5,z],[1.5,.08,.08])]:cube(name+str(level),xyz,scale)
    for ix,x in enumerate([cx-2.5,cx+2.5]):
        for iy,y in enumerate([cy-2.5,cy+2.5]):cube('Post%d%d'%(ix,iy),[x,y,.5],[.1,.1,1.])
    text='#usda 1.0\n(\n    defaultPrim = "World"\n    subLayers = [@./'+original.name+'@]\n)\nover "World" {\n def Xform "Paper200RoverFence" {\n'+'\n'.join(parts)+'\n }\n}\n'
    if overlay.exists() and overlay.read_text()!=text:raise ValueError('fence overlay already differs')
    overlay.write_text(text,encoding='utf-8')
    revision=[]
    for row in read(root/'manifests/questions.json'):
        if row['scene']!='factory':continue
        p=root/'benchmark/task'/row['question']
        for name in ['environment.json','route.json']:
            f=p/'data/env'/name
            if not f.exists():continue
            before=read(f);after=read(f)
            if name=='environment.json':after['scene_usd']=str(Path(after['scene_usd']).with_name(overlay.name))
            else:after['scene']['path']=str(Path(after['scene']['path']).with_name(overlay.name))
            if after!=before:
                backup=root/'manifests/revisions/fence_original'/row['question']/name
                if not backup.exists():write(backup,before)
                write(f,after);revision.append({'path':str(f.relative_to(root)),'before':before,'after':after})
    bindings={}
    for q in ['Q2','Q12']:
        bindings[q]={'docking_region':{'coordinate_frame':'world','object':read(root/'benchmark/task'/q/'data/task/task.json')['tracked_objects']['target_0'],
            'minimum_surface_gap_m':.5,'maximum_surface_gap_m':1.0,'surface_metric':'full robot collision geometry to object collision geometry'},
            'stop_linear_speed_mps':.1,'stop_angular_speed_rps':.1,'stop_duration_s':.2}
    bindings['Q12']['docking_region']['inside_fence_xy']={'min':[cx-2.46,cy-2.46],'max':[cx+2.46,cy+2.46]}
    bindings['Q8']={'fence_prim':'/World/Paper200RoverFence','inside_xy':{'min':[cx-2.46,cy-2.46],'max':[cx+2.46,cy+2.46]},
        'required_side':'outside','maximum_fence_surface_xy_gap_m':.5,'reference_vehicle_only':'/World/Cars/c_01',
        'stop_linear_speed_mps':.1,'stop_angular_speed_rps':.1,'stop_duration_s':.2}
    for n in range(79,87):
        q='Q%d'%n
        if n==82:bindings[q]={'push_distance_m':.05,'longitudinal_tolerance_m':.01,'lateral_tolerance_m':.01,'reference_time':'first physical robot-basket contact before displacement'}
        else:bindings[q]={'hover_position_tolerance_m':.01,'hover_linear_speed_mps':.02,'hover_angular_speed_rps':.1,
            'hover_duration_s':3. if n>=83 else 1.,'grasp_reference_time':'first bilateral finger-object physical contact sustained for 0.05 s; no task-success flag used'}
    bindings['Q97']={'middle_fraction_of_long_axis':[1/3,2/3],'contact_region_reference':'initial target geometry principal axis; transformed by live body pose','contact_attribution':'raw PhysX contact positions'}
    for q in ['Q8','Q12']:
        p=root/'benchmark/task'/q/'data/task/task.json';task=read(p);plan=task['rubric_evidence']
        fence='/World/Paper200RoverFence'
        if not any(o['prim']==fence for o in plan['objects']):
            plan['objects'].append({'prim':fence,'runtime_prim':fence});plan['objects'].sort(key=lambda o:o['prim'])
            plan['contacts'].append({'source':'/M20PiperVLA/M20_Piper','target':fence})
        write(p,task)
        public_path=root/'manifests/public'/q/'task.json';public=read(public_path)
        if fence not in public['public_prims']:
            public['public_prims'].append(fence);public['assets']+='\n车辆外围真实围栏：'+fence+'；北侧开口宽2m。'
            write(public_path,public)
    write(root/'manifests/rubric_bindings.json',{'schema':'paper200-rubric-parameter-bindings/v1','created':now(),'unresolved':[],
        'rubrics_modified':False,'used_by_policy':False,'bindings':bindings,'reason':'explicitly fill authored pending evaluator parameters before main collection; independent of outcomes'})
    write(root/'manifests/revisions/fence_scene.json',{'created':now(),'original_scene':str(original),'overlay':str(overlay),'sha256':sha(overlay),
        'reason':'Q8/Q12 refer to a fence absent from composed scene; real collision geometry added under authorized benchmark repair',
        'affected_questions':['Q%d'%n for n in range(1,51)],'systems':'all three identically','changes':revision})
    return {'overlay':str(overlay),'rubrics_modified':False,'affected_questions':50}

if __name__=='__main__':
    import sys
    print(bind(Path(sys.argv[1]).resolve()))
