"""OpenPI NumPy-msgpack compatible codec, with no pickle/object deserialization."""
import math
import numpy as np
from .contracts import ALIASES


def encode(value):
    if isinstance(value, (np.ndarray, np.generic)) and value.dtype.kind not in 'biuf':
        raise ValueError("unsupported array dtype")
    if isinstance(value, np.ndarray):
        return {b'__ndarray__': True, b'data': value.tobytes(), b'dtype': value.dtype.str, b'shape': value.shape}
    if isinstance(value, np.generic):
        return {b'__npgeneric__': True, b'data': value.item(), b'dtype': value.dtype.str}
    raise TypeError("unsupported msgpack value")


def decode(value):
    if b'__ndarray__' in value:
        dtype, shape, data = np.dtype(value[b'dtype']), value[b'shape'], value[b'data']
        if dtype.kind not in 'biuf' or len(shape) > 4 or any(type(n) is not int or n < 0 for n in shape):
            raise ValueError("invalid ndarray descriptor")
        if math.prod(shape) * dtype.itemsize != len(data) or len(data) > 32 * 1024**2:
            raise ValueError("invalid ndarray byte size")
        return np.frombuffer(data, dtype=dtype).reshape(shape).copy()
    if b'__npgeneric__' in value:
        dtype = np.dtype(value[b'dtype'])
        if dtype.kind not in 'biuf':
            raise ValueError("invalid scalar dtype")
        return dtype.type(value[b'data'])
    return value


def pack(value):
    import msgpack
    return msgpack.packb(value, default=encode)


def unpack(value):
    import msgpack
    if not isinstance(value, bytes):
        raise ValueError("expected binary msgpack")
    return msgpack.unpackb(value, object_hook=decode)


def policy_observation(clean):
    result = {"prompt": clean["prompt"]}
    for name, aliases in ALIASES.items():
        if name in clean:
            for alias in aliases:
                result[alias] = clean[name]
    return result


METADATA = {
    "robot_id": "deeprobotics-m20-pro+agilex-piper",
    "action_schema": "m20-piper-unified-action/v1",
    "observation_schema": "m20-piper-front-rear-wrist-observation/v2",
    "action_dim": 10, "state_dim": 25, "control_hz": 10,
    "image_keys": ["image", "rear_image", "wrist_image"],
    "request_key_aliases": ALIASES,
    "policy_family": "m20-runtime-assurance-proxy",
}
