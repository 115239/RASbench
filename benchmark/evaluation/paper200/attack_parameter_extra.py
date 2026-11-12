"""Independent geometry/image parameter audit from persisted actual payloads.

This module deliberately does not import an attack implementation. A verified
sample is not proof that every frame in a duration was sampled; the independent
application/consumer and lifecycle audits must also pass.
"""
import hashlib
import io
import json
import numpy as np


def same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all() and np.allclose(a, b, atol=2e-6, rtol=0)


def select(sample, p, points, text):
    target = p.get('target_prim_path')
    if target:
        # Maps can evolve as RTX sees new objects. A final sidecar map is not
        # accepted as the identity of an earlier sample.
        mapping = json.loads(text(sample['stable_id_map_json'][()]))
        ids = sample['original_object_ids'][...]
        if ids.shape != (len(points),):
            raise ValueError('LIDAR_OBJECT_ID_ALIGNMENT_MISMATCH')
        target = target.rstrip('/')
        matching = [int(key) for key, value in mapping.items() if value == target or
                    (p.get('include_descendants', True) and value.startswith(target + '/'))]
        return np.isin(ids, matching)
    for key in ('world_aabb', 'sensor_aabb'):
        if key not in p:
            continue
        box = p[key]
        bounds = np.asarray([box['min'], box['max']] if isinstance(box, dict) else box, float)
        if bounds.shape != (2, 3) or not np.isfinite(bounds).all() or np.any(bounds[0] > bounds[1]):
            raise ValueError('LIDAR_INVALID_EXPECTED_BOUNDS')
        coordinates = points
        if key == 'world_aabb':
            matrix = sample['original_sensor_to_world'][...]
            coordinates = points @ matrix[:3, :3].T + matrix[:3, 3]
        return np.all((coordinates >= bounds[0]) & (coordinates <= bounds[1]), axis=1)
    return np.ones(len(points), bool)


def phantom(p, step):
    if p.get('anchor_frame') == 'world_at_trigger':
        raise ValueError('WORLD_PHANTOM_TRIGGER_AND_GROUND_RECEIPT_REQUIRED')
    count = int(p.get('point_count', p.get('count', p.get('phantom_count', 180))))
    if count < 1 or count > 100000:
        raise ValueError('LIDAR_PHANTOM_COUNT_OUT_OF_BOUNDS')
    ranges = [p.get(k, d) for k, d in [('x_range_m', [.35, .85]),
              ('y_range_m', [-.2, .2]), ('z_range_m', [-.30, -.26])]]
    mode = 'dense_wall' if p.get('geometry') == 'wall' else p.get('sampling', 'seeded_uniform')
    if mode == 'wall_grid':
        sizes = [int(p.get('layers', 1)), int(p.get('columns', 128)), int(p.get('rows', 64))]
        if min(sizes) < 1 or np.prod(sizes) > 100000:
            raise ValueError('LIDAR_PHANTOM_GRID_OUT_OF_BOUNDS')
        return np.column_stack([v.ravel() for v in np.meshgrid(
            *(np.linspace(*span, n) for span, n in zip(ranges, sizes)), indexing='ij')]).astype(np.float32)
    if mode == 'linspace':
        return np.column_stack([np.linspace(*span, count) for span in ranges]).astype(np.float32)
    divisor = int(p.get('seed_step_divisor', 1))
    if divisor <= 0:
        raise ValueError('LIDAR_INVALID_SEED_DIVISOR')
    rng = np.random.default_rng(np.random.SeedSequence([int(p.get('seed', 0)), step // divisor]))
    if mode == 'dense_wall':
        center, thickness = float(p.get('x_m', 1.2)), float(p.get('thickness_m', .08))
        ranges[0] = [center - thickness / 2, center + thickness / 2]
    elif mode != 'seeded_uniform':
        raise ValueError('LIDAR_PHANTOM_SAMPLING_UNSUPPORTED')
    result = np.column_stack([rng.uniform(*span, count) for span in ranges]).astype(np.float32)
    if mode == 'dense_wall' and float(p.get('noise_std_m', .01)):
        result += rng.normal(0, float(p.get('noise_std_m', .01)), result.shape).astype(np.float32)
    return result


def anchored_points(p, receipt):
    """Reconstruct the world wall from the actual trigger and ground hit."""
    position = np.asarray(receipt['trigger_position'], float)
    yaw = float(receipt['trigger_yaw'])
    if position.shape != (3,) or not np.isfinite(position).all() or not np.isfinite(yaw):
        raise ValueError('WORLD_PHANTOM_TRIGGER_POSE_INVALID')
    direction = np.array([np.cos(yaw), np.sin(yaw), 0.])
    if p.get('direction_binding') == 'return_destination':
        direction[:2] = np.asarray(receipt['return_destination_xy']) - position[:2]
        length = np.linalg.norm(direction)
        if length < 1e-8:
            raise ValueError('WORLD_PHANTOM_RETURN_DIRECTION_UNDEFINED')
        direction /= length
    ground_hit = np.asarray(receipt['ground_hit_position'], float)
    if (ground_hit.shape != (3,) or not np.isfinite(ground_hit).all() or
            not same(ground_hit[:2], position[:2]) or
            not position[2]-5.25-2e-6 <= ground_hit[2] <= position[2]-.25+2e-6):
        raise ValueError('WORLD_PHANTOM_GROUND_RAYCAST_MISMATCH')
    side = np.array([-direction[1], direction[0], 0.])
    center = position + direction*float(p['forward_distance_m'])
    sizes = [int(p['layers']), int(p['columns']), int(p['rows'])]
    if min(sizes) < 1 or np.prod(sizes) > 100000:
        raise ValueError('WORLD_PHANTOM_GRID_OUT_OF_BOUNDS')
    xx, yy, zz = np.meshgrid(np.linspace(-p['thickness_m']/2, p['thickness_m']/2, sizes[0]),
        np.linspace(-p['width_m']/2, p['width_m']/2, sizes[1]),
        np.linspace(ground_hit[2], ground_hit[2]+p['height_m'], sizes[2]), indexing='ij')
    wanted = center + xx.reshape(-1, 1)*direction + yy.reshape(-1, 1)*side
    wanted[:, 2] = zz.ravel()
    if not same(wanted, receipt['world_points']):
        raise ValueError('WORLD_PHANTOM_ANCHOR_GEOMETRY_MISMATCH')
    return wanted


def lidar(sample, p, handler, text, anchor=None):
    before, after = sample['original_points'][...], sample['attacked_points'][...]
    if before.ndim != 2 or before.shape[1] != 3 or after.ndim != 2 or after.shape[1] != 3:
        raise ValueError('LIDAR_POINT_SHAPE_INVALID')
    transform = sample['original_sensor_to_world'][...]
    if not same(transform, sample['attacked_sensor_to_world'][...]) or transform.shape != (4, 4):
        raise ValueError('LIDAR_FRAME_TRANSFORM_CHANGED')
    if not same(transform[3], [0, 0, 0, 1]) or not same(transform[:3, :3].T @ transform[:3, :3], np.eye(3)):
        raise ValueError('LIDAR_TRANSFORM_NOT_RIGID')
    if (text(sample['original_frame_id'][()]) != 'm20_rtx_lidar' or
            text(sample['attacked_frame_id'][()]) != 'm20_rtx_lidar' or
            not same(sample['original_timestamp_s'][()], sample['attacked_timestamp_s'][()])):
        raise ValueError('LIDAR_FRAME_ID_OR_TIMESTAMP_CHANGED')
    mask = None
    if handler in ('lidar_object_dropout_frame', 'paper200_roi_dropout'):
        if not any(k in p for k in ('target_prim_path', 'world_aabb', 'sensor_aabb')):
            raise ValueError('LIDAR_DROPOUT_SELECTOR_MISSING')
        mask = select(sample, p, before, text)
        wanted = before[~mask]
    elif handler == 'lidar_point_drift_frame':
        mask = select(sample, p, before, text)
        angle = float(p.get('yaw_bias_rad', 0))
        c, s = np.cos(angle), np.sin(angle)
        rotation = np.asarray([[c, -s, 0], [s, c, 0], [0, 0, 1]], np.float32)
        wanted = before.copy()
        wanted[mask] = (before[mask] * float(p.get('range_scale', 1))) @ rotation.T + np.asarray(p.get('translation_xyz_m', [0, 0, 0]), np.float32)
    elif handler == 'lidar_phantom_add_frame':
        if p.get('anchor_frame') == 'world_at_trigger':
            if anchor is None:
                raise ValueError('WORLD_PHANTOM_TRIGGER_AND_GROUND_RECEIPT_REQUIRED')
            world = anchored_points(p, anchor)
            added = ((world-transform[:3, 3]) @ transform[:3, :3]).astype(np.float32)
        else:
            added = phantom(p, int(sample['physics_step'][()]))
        wanted = np.concatenate([before, added])
    else:
        raise ValueError('LIDAR_PARAMETER_HANDLER_UNSUPPORTED')
    if not same(wanted, after):
        raise ValueError('LIDAR_PARAMETER_POINTS_MISMATCH')
    if 'original_object_ids' in sample:
        original_ids, attacked_ids = sample['original_object_ids'][...], sample['attacked_object_ids'][...]
        if handler in ('lidar_object_dropout_frame', 'paper200_roi_dropout'):
            wanted_ids = original_ids[~mask]
        elif handler == 'lidar_phantom_add_frame':
            wanted_ids = np.concatenate([original_ids, np.full(len(wanted) - len(before), p.get('phantom_object_id', -1), dtype=original_ids.dtype)])
        else:
            wanted_ids = original_ids
        if not np.array_equal(wanted_ids, attacked_ids):
            raise ValueError('LIDAR_PARAMETER_IDS_MISMATCH')


def image_expected(before, mask, p, record):
    from PIL import Image
    result = before.copy()
    if before.dtype != np.uint8 or before.ndim != 3 or before.shape[2] != 3:
        raise ValueError('IMAGE_RGB_ABI_UNSUPPORTED')
    ys, xs = np.where(mask)
    if not len(xs):
        return result
    mode = p.get('texture_mode')
    if mode is None:
        yy, xx = np.indices(before.shape[:2]);tile = max(1, int(p.get('tile_px', 8)))
        colors = np.where(((xx // tile + yy // tile) % 2)[..., None],
                          p.get('color_a_rgb', [0, 255, 0]), p.get('color_b_rgb', [15, 90, 15]))
        result[mask] = colors[mask]
        return result
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
    width, height = x1-x0, y1-y0;yy, xx = np.mgrid[:height, :width]
    if mode == 'scene_floor_texture':
        texture_key = 'texture_encoded_bytes' if 'texture_encoded_bytes' in record else 'texture_asset_bytes'
        raw = record[texture_key][...].tobytes()
        if hashlib.sha256(raw).hexdigest() != p['texture_sha256']:
            raise ValueError('IMAGE_TEXTURE_CONTENT_HASH_MISMATCH')
        patch = np.asarray(Image.open(io.BytesIO(raw)).convert('RGB').resize((width, height), Image.Resampling.BILINEAR))
    elif mode == 'contour_checker':
        tile = int(p['tile_px'])
        if tile < 1:
            raise ValueError('IMAGE_CONTOUR_TILE_INVALID')
        even = (xx//tile + yy//tile) % 2 == 0
        patch = np.where(even[..., None], p['color_a_rgb'], p['color_b_rgb'])
    elif mode == 'diagonal_stripes':
        angle = np.deg2rad(float(p['stripe_angle_deg']));period = float(p['stripe_period_px'])
        if period <= 0:
            raise ValueError('IMAGE_STRIPE_PERIOD_INVALID')
        bright = np.mod((-xx*np.sin(angle)+yy*np.cos(angle))/period, 1) < float(p['bright_fraction'])
        patch = np.where(bright[..., None], p['color_a_rgb'], p['color_b_rgb'])
    elif mode == 'specular_glare':
        angle = np.deg2rad(float(p['streak_angle_deg']));sigma = float(p['streak_width_fraction'])
        if sigma <= 0:
            raise ValueError('IMAGE_GLARE_WIDTH_INVALID')
        u, v = (xx+.5)/width-.5, (yy+.5)/height-.5
        across, along = -u*np.sin(angle)+v*np.cos(angle), u*np.cos(angle)+v*np.sin(angle)
        glare = np.exp(-.5*(across/sigma)**2)*np.exp(-.5*(along/.6)**2)
        glare += float(p['secondary_strength'])*np.exp(-.5*((across-.22)/(sigma*.6))**2)
        patch = np.clip(before[y0:y1, x0:x1].astype(float)+255*float(p['highlight_gain'])*glare[..., None], 0, 255)
    else:
        raise ValueError('IMAGE_PARAMETER_MODE_UNSUPPORTED')
    local = mask[y0:y1, x0:x1]
    result[y0:y1, x0:x1][local] = np.rint(patch).astype(np.uint8)[local]
    return result


def verify(collection, attack, spec, stage, exact, rows, text):
    handler, p = spec.get('handler'), spec.get('parameters', {})
    issues, checked = [], 0
    try:
        if spec.get('stage') != stage:
            raise ValueError('EXPECTED_ATTACK_STAGE_MISMATCH')
        if stage == 'sensor_lidar':
            if handler not in ('lidar_object_dropout_frame', 'paper200_roi_dropout',
                               'lidar_point_drift_frame', 'lidar_phantom_add_frame'):
                raise ValueError('PARAMETER_SEMANTICS_NOT_VERIFIED:'+str(handler))
            anchors = [row for row in rows(collection, 'consumer_receipts_jsonl') if
                       row.get('consumer') == 'phantom_anchor' and row.get('attack_id') == spec.get('id')]
            if p.get('anchor_frame') == 'world_at_trigger' and len(anchors) != 1:
                raise ValueError('WORLD_PHANTOM_UNIQUE_ANCHOR_RECEIPT_REQUIRED')
            for sample in collection['lidar_payloads'].values():
                lidar(sample, p, handler, text, anchors[0] if anchors else None);checked += 1
        elif stage == 'request_image' and handler == 'paper200_projected_patch':
            from PIL import Image
            requests = collection['model_boundaries'];records = collection['visual_masks']
            covered = set()
            for record in records.values():
                meta = json.loads(text(record['metadata_json'][()]))
                if meta.get('attack_id') != spec['id']:
                    continue
                targets = p.get('target_prims') or [p['target_prim']]
                labels = meta['idToLabels'];ids = []
                for key, label in labels.items():
                    path = label.get('primPath', '') if isinstance(label, dict) else label
                    if any(path == target or path.startswith(target.rstrip('/')+'/') for target in targets):
                        ids.append(int(key))
                before_key = 'image_before' if 'image_before' in record else 'before'
                after_key = 'image_after' if 'image_after' in record else 'after'
                before, after = record[before_key][...], record[after_key][...]
                mask = np.isin(record['instance_ids'][...].squeeze(), ids)
                if mask.shape != before.shape[:2]:
                    mask = np.asarray(Image.fromarray(mask).resize((before.shape[1], before.shape[0]), Image.Resampling.NEAREST))
                if p.get('mask_region') == 'inner_contour':
                    from scipy.ndimage import binary_erosion
                    width = int(p['contour_width_px'])
                    if width < 1:
                        raise ValueError('IMAGE_CONTOUR_WIDTH_INVALID')
                    mask = mask & ~binary_erosion(mask, iterations=width, border_value=0)
                elif p.get('mask_region') not in (None, 'full'):
                    raise ValueError('IMAGE_MASK_REGION_UNSUPPORTED')
                if not np.array_equal(mask, record['mask'][...]):
                    raise ValueError('IMAGE_INSTANCE_TARGET_MASK_MISMATCH')
                wanted = image_expected(before, mask, p, record)
                if not np.array_equal(wanted, after):
                    raise ValueError('IMAGE_NUMERIC_PARAMETER_MISMATCH')
                matches = [key for key, request in requests.items() if
                    int(request.attrs['physics_step']) == int(meta['physics_step']) and
                    np.array_equal(before, request['front/original'][...]) and
                    np.array_equal(after, request['front/attacked'][...])]
                if not matches:
                    raise ValueError('IMAGE_MODEL_BOUNDARY_JOIN_MISSING')
                covered.update(matches);checked += 1
            for key, request in requests.items():
                for camera in ('front', 'rear', 'wrist'):
                    if camera in request and not np.array_equal(request[camera+'/original'][...], request[camera+'/attacked'][...]):
                        if camera != 'front' or key not in covered:
                            issues.append('IMAGE_CHANGED_REQUEST_WITHOUT_TARGET_PARAMETER_PROOF')
        elif stage == 'immediate_visual_material':
            expected = set(p['expected_affected_mesh_prims'])
            if not expected or not p.get('texture_sha256'):
                raise ValueError('MATERIAL_FROZEN_MESH_OR_TEXTURE_HASH_MISSING')
            for row in rows(collection, 'consumer_receipts_jsonl'):
                if row.get('consumer') != 'usd_renderer_binding' or row.get('attack_id') != spec.get('id'):
                    continue
                if set(row['actual_materials']) != expected or set(row['original_materials']) != expected:
                    raise ValueError('MATERIAL_TARGET_MESH_PARAMETER_MISMATCH')
                if any(value != row['override_material'] for value in row['actual_materials'].values()):
                    raise ValueError('MATERIAL_ACTUAL_BINDING_MISMATCH')
                if row['texture_sha256'] != p['texture_sha256']:
                    raise ValueError('MATERIAL_ACTUAL_TEXTURE_CONTENT_MISMATCH')
                checked += 1
            # Continuous binding hold and restoration belong to the consumer
            # proof; this result only certifies the actual mesh/texture values.
        else:
            raise ValueError('PARAMETER_SEMANTICS_NOT_VERIFIED:'+str(handler))
    except (KeyError, TypeError, IndexError) as exc:
        issues.append('PARAMETER_SOURCE_EVIDENCE_MISSING:'+str(exc))
    except ValueError as exc:
        issues.append(str(exc))
    if checked == 0:
        issues.append('NO_NUMERIC_PARAMETER_EVIDENCE_VERIFIED')
    return {'passed':checked > 0 and not issues, 'checked_rows':checked, 'handler':handler,
            'issues':list(dict.fromkeys(issues)),
            'scope':'persisted geometric/image parameters; consumer and complete temporal coverage audited separately'}
