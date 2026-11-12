"""Versioned projections of organizer-supplied, content-verified contracts."""
from copy import deepcopy
import hashlib
import json
import numpy as np
import h5py
from .evidence_io import object_dataset, decode
from .paper200_contract import canonical_digest

KINDS = dict(zip(('S%02d' % i for i in range(1, 15)), (
    'forbidden_contact', 'self_collision', 'region_exclusion', 'unexpected_drop',
    'discrete_state_hold', 'contact_force_upper_bound', 'body_posture_limit',
    'min_separation_distance', 'linear_speed_limit', 'angular_speed_limit',
    'joint_limit', 'displacement_rotation_limit', 'transport_tilt_limit', 'placement_exclusion')))

def safety_config(profile):
    return {'safety_version': profile.get('rule_set_version'),
            'constraint_set_id': profile['profile_id'], 'activation': deepcopy(profile['activation']),
            'safety_types': {rid: {'enabled': row.get('enabled', False), 'description': KINDS.get(rid, rid),
                'rule': {**deepcopy(row), 'kind': row.get('kind') or KINDS.get(rid)}}
                for rid, row in profile['rules'].items()}}

def frozen_registry(contract):
    family = {'request_image': 'image', 'request_state': 'state_feedback', 'state_feedback': 'state_feedback', 'request_prompt': 'prompt', 'response_action': 'vla_output_control', 'control_base': 'vla_output_control', 'control_joint': 'vla_output_control', 'control_gripper': 'vla_output_control', 'control_ee_target': 'vla_output_control', 'immediate_robot_control': 'vla_output_control', 'immediate_gripper': 'vla_output_control', 'immediate_visual_material': 'image', 'sensor_lidar': 'lidar', 'immediate_lidar': 'lidar', 'sensor_imu': 'imu', 'immediate_imu': 'imu', 'sensor_force_torque': 'state_feedback', 'sensor_wheel_encoder': 'state_feedback', 'sensor_joint_encoder': 'state_feedback'}
    result = []
    for entry in contract.get('attacks', []):
        for attack in entry['profile'].get('attack', {}).get('attacks', []):
            result.append({'attack_id': attack['id'], 'handler': attack['handler'], 'stage': attack['stage'],
                'family': family.get(attack['stage'], attack['stage']), 'contract': deepcopy(attack),
                '_verified_frozen_contract': True,
                'source': 'hdf5:/extensions/paper200_collection/runner_judge_contract_json'})
    return result

def augment_identity(identity, hdf5):
    group = hdf5.get('/extensions/paper200_collection')
    if group is None:
        return
    contract = object_dataset(group, 'runner_judge_contract_json')
    inputs = object_dataset(group, 'input_identity_json')
    declared = object_dataset(group, 'identity_json')
    model = object_dataset(group, 'model_identity_json')
    task = contract.get('task', {})
    values = {'scene_id': task.get('policy_scene'), 'attack_seed': declared.get('attack_seed'),
              'runner_contract_digest': canonical_digest(contract) if contract else None}
    state = group.get('formal_initial_state')
    if isinstance(state, h5py.Group):
        h = hashlib.sha256()
        for name in sorted(state):
            ds = state[name]
            if not isinstance(ds, h5py.Dataset):
                continue
            raw = ds[()]
            h.update(name.encode())
            if getattr(raw, 'dtype', None) is not None and raw.dtype.kind not in 'OSU':
                h.update(str(raw.shape).encode()); h.update(str(raw.dtype).encode()); h.update(raw.tobytes())
            else:
                h.update(json.dumps(np.asarray(raw).astype(str).tolist(), sort_keys=True).encode())
        values['initial_state_digest'] = h.hexdigest()
    for key, value in values.items():
        if identity.get(key) is None and value is not None:
            identity[key] = value
            identity['identity_sources'][key] = 'derived:verified_paper200_collection/' + key
    meta = identity['source_metadata']
    for key, value in {'safety_config_digest': contract.get('safety_profile'),
        'attack_config_digest': contract.get('attacks'), 'object_binding_digest': contract.get('object_registry'),
        'scene_digest': inputs.get('scene_revisions'), 'model_digest': model}.items():
        if value:
            meta.setdefault(key, canonical_digest(value))
    # Bytes and canonical content are deliberately separate namespaces.
    meta['recorded_contract_canonical_sha256'] = canonical_digest(contract) if contract else None

def load_trusted_contract(path, expected_sha256):
    from pathlib import Path
    raw = Path(path).read_bytes()
    if not expected_sha256 or hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError('TRUSTED_CONTRACT_SOURCE_SHA256_MISMATCH')
    return decode(raw, str(path))
