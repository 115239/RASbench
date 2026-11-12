"""Image-space floor texture, diagonal stripes and specular glare attacks.

All modifications use the rendered instance mask, preserving occlusion and all
pixels outside the declared visible target. Desired grasp/perception errors are
outcomes, never forcibly applied to controller outputs.
"""
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from PIL import Image


def mask_from_frame(frame,targets,shape):
    segmentation=frame.get('instance_id_segmentation')
    if not isinstance(segmentation,dict) or 'data' not in segmentation:
        raise ValueError('visual attack instance mask is unavailable')
    labels=segmentation.get('info',{}).get('idToLabels',{});ids=[]
    for key,label in labels.items():
        if isinstance(label,dict):label=label.get('primPath','')
        if any(str(label)==p or str(label).startswith(p+'/') for p in targets):ids.append(int(key))
    mask=np.isin(np.asarray(segmentation['data']).squeeze(),ids)
    if mask.shape!=tuple(shape):
        mask=np.asarray(Image.fromarray(mask).resize((shape[1],shape[0]),Image.Resampling.NEAREST))
    return mask,ids,labels,np.asarray(segmentation['data'])


def transform_image(image,mask,parameters,texture=None):
    """Pure image transformation; dtype/range remain the actual camera ABI."""
    original=np.asarray(image)
    if original.dtype!=np.uint8 or original.ndim!=3 or original.shape[2]<3:
        raise ValueError('registered image attack requires uint8 HWC RGB input')
    result=original.copy();ys,xs=np.where(mask)
    if not len(xs):return result
    x0,x1=int(xs.min()),int(xs.max())+1;y0,y1=int(ys.min()),int(ys.max())+1
    width,height=x1-x0,y1-y0;mode=parameters['texture_mode']
    yy,xx=np.mgrid[0:height,0:width];rgb=original[y0:y1,x0:x1,:3].astype(float)
    if mode=='scene_floor_texture':
        if texture is None:raise ValueError('floor texture pixels missing')
        patch=np.asarray(Image.fromarray(np.asarray(texture,dtype=np.uint8)).convert('RGB').resize((width,height),Image.Resampling.BILINEAR))
    elif mode=='contour_checker':
        tile=int(parameters['tile_px'])
        if tile<1:raise ValueError('contour tile width must be positive')
        alternate=(xx//tile+yy//tile)%2==0
        patch=np.where(alternate[...,None],np.asarray(parameters['color_a_rgb']),np.asarray(parameters['color_b_rgb']))
    elif mode=='diagonal_stripes':
        angle=np.deg2rad(float(parameters['stripe_angle_deg']))
        period=float(parameters['stripe_period_px'])
        if period<=0:raise ValueError('stripe period must be positive')
        # Stripe direction is angle, so the phase runs along its normal.
        phase=(-xx*np.sin(angle)+yy*np.cos(angle))/period
        stripe=np.mod(phase,1.)<float(parameters['bright_fraction'])
        a=np.asarray(parameters['color_a_rgb']);b=np.asarray(parameters['color_b_rgb'])
        patch=np.where(stripe[...,None],a,b)
    elif mode=='specular_glare':
        # Additive clipped highlight lobes create actual saturation and a bright
        # reflective streak, preserving the underlying observed tube elsewhere.
        angle=np.deg2rad(float(parameters['streak_angle_deg']))
        u=(xx+.5)/max(width,1)-.5;v=(yy+.5)/max(height,1)-.5
        across=-u*np.sin(angle)+v*np.cos(angle)
        along=u*np.cos(angle)+v*np.sin(angle)
        sigma=float(parameters['streak_width_fraction'])
        if sigma<=0:raise ValueError('glare width must be positive')
        glare=np.exp(-.5*(across/sigma)**2)*np.exp(-.5*(along/.6)**2)
        glare+=float(parameters['secondary_strength'])*np.exp(-.5*((across-.22)/(sigma*.6))**2)
        patch=np.clip(rgb+255.*float(parameters['highlight_gain'])*glare[...,None],0,255)
    else:raise ValueError('unsupported frozen texture mode '+mode)
    cropped=result[y0:y1,x0:x1,:3]
    local=mask[y0:y1,x0:x1];cropped[local]=np.rint(patch).astype(np.uint8)[local]
    return result


def restrict_mask(mask,parameters):
    if parameters.get('mask_region')!='inner_contour':return mask
    from scipy.ndimage import binary_erosion
    width=int(parameters['contour_width_px'])
    if width<1:raise ValueError('contour width must be positive')
    return mask & ~binary_erosion(mask,iterations=width,border_value=0)


def texture_patch(image,parameters,context):
    from . import attack_runtime
    from .common import write,sha
    collector=attack_runtime.CURRENT_COLLECTOR
    if collector is None:raise ValueError('no camera attack adapter')
    targets=parameters.get('target_prims') or [parameters['target_prim']]
    frame=collector.cameras[0].get_current_frame()
    mask,ids,labels,instance_ids=mask_from_frame(frame,targets,np.asarray(image).shape[:2])
    mask=restrict_mask(mask,parameters)
    texture=None;texture_identity=None
    if parameters['texture_mode']=='scene_floor_texture':
        package=Path(os.environ['M20_QUESTION_PACKAGE_ROOT']).resolve(strict=True)
        path=(package/parameters['texture_asset']).resolve(strict=True);path.relative_to(package)
        identity=sha(path)
        if identity!=parameters['texture_sha256']:raise ValueError('frozen floor texture changed')
        texture=np.asarray(Image.open(path).convert('RGB'));texture_identity={'path':str(path),'sha256':identity}
    result=transform_image(image,mask,parameters,texture)
    changed=np.any(np.asarray(image)[...,:3]!=result[...,:3],axis=-1)
    if np.any(changed & ~mask):raise AssertionError('texture modification escaped target instance mask')
    record={'schema':'paper200-image-texture/v2','event':'visual_attack_instance_mask',
        'attack_id':context['attack_id'],'texture_mode':parameters['texture_mode'],
        'target_prims':targets,'matched_ids':ids,'idToLabels':labels,'physics_step':int(collector.physics_step),
        'visible_target_pixels':int(mask.sum()),'changed_pixels':int(changed.sum()),
        'mask_source':'rendered instance IDs; occlusion respected',
        'texture_identity':texture_identity,'image_surface':'front camera policy input',
        'parameters':parameters,'intended_perception_error_is_measured_outcome':True,
        'mask_sha256':hashlib.sha256(mask.tobytes()).hexdigest(),
        'image_before_sha256':hashlib.sha256(np.asarray(image).tobytes()).hexdigest(),
        'image_after_sha256':hashlib.sha256(result.tobytes()).hexdigest()}
    out=getattr(collector,'paper_visual_evidence_dir',None)
    if out is not None:
        out=Path(out);out.mkdir(exist_ok=True)
        stem='%s_%09d'%(context['attack_id'],collector.physics_step)
        # Requests can share a physics step; keep all actual applications.
        count=0
        while (out/(stem+'.json')).exists():count+=1;stem='%s_%09d_%03d'%(context['attack_id'],collector.physics_step,count)
        write(out/(stem+'.json'),record)
        values={'image_before':np.asarray(image),'image_after':result,'mask':mask,
                'instance_ids':instance_ids,'metadata_json':np.frombuffer(json.dumps(record,sort_keys=True).encode(),dtype=np.uint8)}
        if texture is not None:
            values['texture_rgb']=texture
            values['texture_asset_bytes']=np.frombuffer(path.read_bytes(),dtype=np.uint8)
        np.savez_compressed(out/(stem+'.npz'),**values)
        record['evidence_stem']=stem
    collector.attack_boundary.recorder.append(record)
    return result


def install(handlers):
    previous=handlers['paper200_projected_patch']
    def dispatch(image,parameters,context):
        if parameters.get('texture_mode'):return texture_patch(image,parameters,context)
        return previous(image,parameters,context)
    handlers['paper200_projected_patch']=dispatch
