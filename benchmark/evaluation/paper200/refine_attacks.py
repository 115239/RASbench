"""Freeze explicit attack bindings; retain original profiles and exact diffs."""
from pathlib import Path
import copy,re
from .common import read,write,now,sha


def refine(root):
    geometry=read(root/'manifests/geometry.json')['questions'];changes=[];issues=[]
    def event(name):return {'type':'interaction_event','event':name,'op':'truthy'}
    def field(name,value,op='eq'):return {'type':'field','field':name,'op':op,'value':value}
    for row in read(root/'manifests/questions.json'):
        q=row['question'];n=int(q[1:]);p=root/'benchmark/task'/q;task=read(p/'data/task/task.json');reg=task['tracked_objects'];geo=geometry[q]['objects']
        for f in sorted((p/'data/attack/profiles').glob('*.json')):
            profile=read(f)
            if 'attack' not in profile:continue
            saved=root/'manifests/revisions/original_attack_profiles'/q/f.name
            original=read(saved) if saved.exists() else copy.deepcopy(profile)
            profile=copy.deepcopy(original);req=read(p/profile['source_requirement']);source=req['source_text'];codes={x['code'] for x in req['unresolved']}
            for attack in profile['attack']['attacks']:
                params=attack['parameters'];trigger=attack['trigger']['all'];first=re.split('发动|触发[。；]|。',source)[0]
                # Compile actual control/phase events instead of the old t=0 placeholders.
                if '返程' in first:
                    trigger=[t for t in trigger if t.get('field')!='simulation.task_time_s' and t.get('event')!='return_started']+[event('return_started')]
                    if '转向' in first:trigger+=[field('command.yaw_abs',.1 if '0.1' in first else 1e-6,'gte')]
                elif '开始抬升' in first:trigger=[event('arm_lift_started')]
                elif '开始闭合' in first:trigger=[event('gripper_closing_started')]
                elif '开始推动' in first:trigger=[event('push_started')]
                elif '开始绕行' in first:trigger=[event('navigation_turn_started')]
                elif '开始导航' in first:trigger=[field('mission.phase','NAV_OUTBOUND')]
                if '去程' in first:trigger.append(field('mission.phase','NAV_OUTBOUND'))
                if n==111 and req['family']=='base_control':trigger=[field('command.yaw_abs',1e-6,'gt')]
                if '首次小于' in first and not any(x in first for x in ['小于等于','小于或等于','小于或等于']):
                    for t in trigger:
                        if t.get('type')=='distance_to_object':t['op']='lt'
                if n==21 and req['family']=='base_control':
                    trigger.append({'type':'field','field':'mission.spawn_distance_m','op':'lte','value':1.0})
                attack['trigger']['all']=trigger
                if attack['handler']=='lidar_phantom_add_frame' and '宽0.6m、高0.8m、厚0.1m' in source:
                    params.update(forward_distance_m=1.,width_m=.6,height_m=.8,thickness_m=.1,
                        x_range_m=[.95,1.05],y_range_m=[-.3,.3],z_range_m=[-.4,.4])
                    if '返程方向' in source:
                        params.update(anchor_frame='world_at_trigger',bottom_at_trigger_ground=True,direction_binding='return_destination')
                # Complete null point-cloud selectors using actual registered geometry.
                if attack['handler']=='lidar_object_dropout_frame' and not params.get('target_prim_path') and not params.get('world_aabb'):
                    key={62:'reference_0',95:'support_0',100:'destination_0'}.get(n,'target_0')
                    if n==98 and req['variant_id']=='lidar_01':key='reference_1'
                    path=reg[key];obj=geo[path]
                    if n==46:path=reg['support_0'].split('/Geometry/')[0];obj=geometry['Q31']['objects'][path]
                    if n==42:path=reg['support_0'];obj=geo[path]
                    if n in [5,11,13,18,22]:params={'world_aabb':{'min':obj['aabb'][0],'max':obj['aabb'][1]}}
                    elif n in [62,95,100,111] or (n==98 and req['variant_id']=='lidar_02'):
                        params={'target_prim_path':path,'include_descendants':True}
                    else:
                        box=copy.deepcopy(obj['aabb']);center=[(a+b)/2 for a,b in zip(*box)]
                        if n in [42,44]:
                            target=geo[reg['target_0']]['aabb'];mid=[(a+b)/2 for a,b in zip(*target)];radius=.05 if n==42 else .02
                            box=[[x-radius for x in mid],[x+radius for x in mid]]
                            if n==44:box[0][2]=target[1][2]-.02;box[1][2]=target[1][2]+.02
                        elif n==94:
                            # Near half-space is frozen relative to the published spawn.
                            spawn=(read(p/'data/env/environment.json').get('spawn') or read(p/'data/env/route.json')['spawn'])['xyz'];axis=max(range(2),key=lambda i:abs(spawn[i]-center[i]))
                            box[0 if spawn[axis]>center[axis] else 1][axis]=center[axis]
                        else:
                            spawn=(read(p/'data/env/environment.json').get('spawn') or read(p/'data/env/route.json')['spawn'])['xyz'];axis=max(range(2),key=lambda i:abs(spawn[i]-center[i]))
                            box[0 if spawn[axis]>center[axis] else 1][axis]=center[axis]
                            if n in [57,69,60]:box[1][2]=box[0][2]+.85*(box[1][2]-box[0][2])
                        params={'world_aabb':{'min':box[0],'max':box[1]}}
                    attack['parameters']=params
                # Explicit image-space patches are projected from the target geometry.
                if 'CAMERA_2D_MASK_REQUIRED' in codes:
                    attack.update(handler='paper200_projected_patch',mode='intercept',stage='request_image',release_delay_s=0.0)
                    params={'target_prim':reg['target_0'],'color_a_rgb':[255,255,255] if '反光' in source else [0,0,0],
                            'color_b_rgb':[40,40,40] if '反光' in source else [255,255,255],'tile_px':4}
                    attack['parameters']=params;attack['marker_policy']['confirmation']['strategy']='normalized_payload_sha256_change'
                # The original compiler incorrectly recolored the task target for distractor attacks.
                if n in [2,6,31,34,36] and req['family']=='visual':
                    other={2:6,6:2,31:32,34:35,36:39}[n]
                    path=read(root/'benchmark/task'/('Q%d'%other)/'data/task/task.json')['tracked_objects']['target_0']
                    params['target_prims']=[path]
                    params['expected_affected_mesh_prims']=geometry['Q%d'%other]['objects'][path]['meshes']
                elif attack['handler']=='install_visual_material_override':
                    meshes=[]
                    for target in params['target_prims']:
                        matches=[v for record in geometry.values() for k,v in record['objects'].items() if k==target and record['scene']==geometry[q]['scene']]
                        if matches:meshes+=matches[0]['meshes']
                        else:issues.append({'question':q,'variant':req['variant_id'],'reason':'visual mesh inventory missing','prim':target})
                    if meshes:params['expected_affected_mesh_prims']=sorted(set(meshes))
                    if n==102:
                        params['target_prims']=[m for m in meshes if 'BackPlate' in m.rsplit('/',1)[-1]]
                        params['expected_affected_mesh_prims']=params['target_prims'][:]
                    if n==114:
                        params['target_prims']=[m for m in meshes if m.endswith('/Section0')]
                        params['expected_affected_mesh_prims']=params['target_prims'][:]
                    if n==85:
                        params['target_prims']=[reg['target_0'],reg['reference_0']]
                        params['expected_affected_mesh_prims']=sorted({m for x in params['target_prims'] for m in geo[x]['meshes']})
                # World-frame translations must be transformed into the common arm-frame ABI.
                match=re.search(r'世界\s*([XYZ])\s*轴[^，。；]*?([+＋−-]?\s*\d+(?:\.\d+)?)\s*(cm|mm|m)',source)
                if req['family']=='arm_control' and match:
                    axis='XYZ'.index(match[1]);amount=float(match[2].replace(' ','').replace('＋','+').replace('−','-'))*{'cm':.01,'mm':.001,'m':1}[match[3]]
                    offset=[0.,0.,0.];offset[axis]=amount
                    attack.update(handler='paper200_world_action_offset',parameters={'world_offset_m':offset})
                # Keep source gaps explicit until geometry-specific bindings are audited.
                if 'WORLD_DIRECTION_VECTOR_REQUIRED' in codes:
                    import numpy as np
                    if '桌外' in source or '桌边外' in source:
                        target=np.asarray(geo[reg['target_0']]['aabb']).mean(axis=0)
                        table=np.asarray(geo[reg['support_0']]['aabb']);options=[]
                        for axis in range(2):
                            for edge,sign in [(0,-1),(1,1)]:options.append((abs(target[axis]-table[edge,axis]),axis,sign))
                        _,axis,sign=min(options);direction=np.zeros(3);direction[axis]=sign
                    else:
                        direction=np.asarray(geo[reg['target_0']]['principal_axes_w'][0])
                        if '短轴' in source:direction=np.cross([0,0,1],direction)
                        direction/=np.linalg.norm(direction)
                        if direction[np.argmax(np.abs(direction))]<0:direction=-direction
                    if req['family']=='arm_control':
                        length=float(re.search(r'偏移\s*[+＋]?\s*(\d+(?:\.\d+)?)\s*cm',source)[1])*.01
                        attack.update(handler='paper200_world_action_offset',parameters={'world_offset_m':(direction*length).tolist(),'direction_binding':'composed target PCA or nearest support edge, frozen before collection'})
                    elif req['family']=='imu':
                        attack.update(handler='paper200_world_imu_bias',parameters={'world_bias_mps2':(direction*.5).tolist()})
                    else:
                        attack.update(handler='paper200_projected_patch',mode='intercept',stage='request_image',release_delay_s=0.0,
                            parameters={'target_prim':reg['target_0'],'color_a_rgb':[255,255,255],'color_b_rgb':[40,40,40],
                                        'tile_px':4,'intended_bias_world_m':(direction*.03).tolist(),'intended_bias_is_not_guaranteed_effect':True})
                        attack['marker_policy']['confirmation']['strategy']='normalized_payload_sha256_change'
                unresolved=codes-{'YAW_COMPONENT_TRIGGER_REQUIRED','RETURN_PHASE_EVENT_REQUIRED','LIDAR_LOCAL_ROI_REQUIRED','CAMERA_2D_MASK_REQUIRED','VISUAL_TARGET_MESH_REQUIRED','WORLD_DIRECTION_VECTOR_REQUIRED'}
                for code in unresolved:issues.append({'question':q,'variant':req['variant_id'],'reason':code})
            if profile!=original:
                dest=root/'manifests/revisions/original_attack_profiles'/q/f.name
                if not dest.exists():write(dest,original)
                write(f,profile);changes.append({'path':str(f.relative_to(root)),'original':str(dest.relative_to(root)),'after_sha256':sha(f),'source_text':source,'reason':'bind authored trigger/affected geometry/coordinate frame; no changes based on outcomes'})
    write(root/'manifests/revisions/attack_bindings.json',{'created':now(),'changes':changes,'remaining_issues':issues})
    print('revised',len(changes),'remaining',len(issues))
if __name__=='__main__':
    import sys
    refine(Path(sys.argv[1]).resolve())
