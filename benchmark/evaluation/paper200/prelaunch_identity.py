"""Reject stale Runner/Judge inputs before allocating models or starting physics.

A visual diagnostic starts from the same verified base contract. Its explicit
private trigger override is added later by control_visual; it is never accepted
as a replacement for the root benchmark contract here.
"""
from pathlib import Path
from .common import read, sha, SYSTEMS
from .evidence_contract import VERSION, canonical_digest
from .repair_contracts import implementation_identity, GEOMETRY_DEFINITION


def verify(root, job, expected_digest=None):
    root = Path(root)
    question = job['question']
    contract_path = root/'benchmark/contracts'/(question + '.json')
    contract = read(contract_path)
    problems = []
    if contract.get('schema') != VERSION or contract.get('question') != question:
        problems.append('QUESTION_CONTRACT_IDENTITY_MISMATCH')
    if 'diagnostic_control' in contract:
        problems.append('PRIVATE_DIAGNOSTIC_CONTRACT_CANNOT_REPLACE_ROOT_CONTRACT')
    if job['system'] not in SYSTEMS:
        problems.append('UNKNOWN_SYSTEM')
    package = root/'benchmark/task'/question
    actual_files = {str(path.relative_to(package)): sha(path) for path in sorted(package.rglob('*'))
                    if path.is_file() and '__pycache__' not in path.parts}
    if actual_files != contract.get('package_files'):
        old_files = contract.get('package_files', {})
        changed = sorted(key for key in set(actual_files) | set(old_files) if actual_files.get(key) != old_files.get(key))
        problems.append('PACKAGE_FILES_CHANGED:' + ','.join(changed))
    public_sha = sha(root/'manifests/public'/question/'task.json')
    if public_sha != contract.get('public_task_sha256'):
        problems.append('PUBLIC_TASK_SPEC_CHANGED_OR_UNBOUND')
    embedded = {'task': 'data/task/task.json', 'rubric': 'data/task/rubric.json',
                'object_registry': 'data/task/object_registry.json',
                'safety_profile': 'data/safety/safety_profile.json',
                'safety_requirements': 'data/safety/safety_requirements.json'}
    for name, relative in embedded.items():
        if contract.get(name) != read(package/relative):
            problems.append('EMBEDDED_CONTENT_CHANGED:' + name)
    if contract.get('task_id') != contract.get('task', {}).get('task_id'):
        problems.append('TASK_ID_MISMATCH')
    attacks = []
    for path in sorted((package/'data/attack/profiles').glob('*.json')):
        value = read(path)
        if 'attack' in value:
            attacks.append({'path': str(path.relative_to(package)), 'profile': value})
    if contract.get('attacks') != attacks:
        problems.append('ATTACK_DEFINITIONS_CHANGED')
    selected = 'data/attack/profiles/q%d_%s.json' % (int(question[1:]), job['variant'])
    if job['variant'] != 'clean' and selected not in {item['path'] for item in attacks}:
        problems.append('SELECTED_ATTACK_NOT_IN_CONTRACT')
    geometry = read(root/'manifests/geometry.json').get('questions', {}).get(question, {})
    if contract.get('geometry_binding') != {'compiled_inventory_sha256': canonical_digest(geometry),
                                          'definition': GEOMETRY_DEFINITION}:
        problems.append('GEOMETRY_BINDING_CHANGED')
    contract_digest = canonical_digest(contract)
    if expected_digest is not None and contract_digest != expected_digest:
        problems.append('CONTRACT_CHANGED_DURING_LAUNCH_PREPARATION')
    # Fail cheaply on a changed scene/package before hashing executable components.
    if problems:
        raise ValueError('PRELAUNCH_CONTRACT_STALE: ' + '; '.join(problems))
    actual_implementation = implementation_identity(root, write_inventory=False)
    if contract.get('implementation') != actual_implementation:
        old = contract.get('implementation', {})
        changed = sorted(key for key in set(old) | set(actual_implementation)
                         if old.get(key) != actual_implementation.get(key))
        raise ValueError('PRELAUNCH_CONTRACT_STALE: IMPLEMENTATION_CHANGED:' + ','.join(changed))
    return {'schema': 'paper200-prelaunch-identity/v2', 'status': 'CURRENT',
            'question': question, 'system': job['system'], 'purpose': job['purpose'],
            'contract_sha256': contract_digest, 'contract_file_sha256': sha(contract_path),
            'package_files_sha256': canonical_digest(actual_files), 'public_task_sha256': public_sha,
            'implementation_sha256': canonical_digest(actual_implementation),
            'private_diagnostic_override_permitted': job['purpose'] == 'control_visual',
            'diagnostic_scope': 'Base contract verified; later private trigger override must retain original contract and explicit control scope.' if job['purpose'] == 'control_visual' else None}
