"""Strict, encoding-aware readers for recorded JSON; never modify evidence."""
import json
import numpy as np
import h5py

def _pairs(rows):
    out = {}
    for key, value in rows:
        if key in out:
            raise ValueError('duplicate JSON key: ' + key)
        out[key] = value
    return out

def decode(value, path='<json>'):
    try:
        if isinstance(value, np.ndarray):
            if value.dtype == np.uint8 and value.ndim == 1:
                value = value.tobytes()
            elif value.ndim == 0:
                value = value.item()
            else:
                raise ValueError('expected scalar text or one-dimensional uint8 JSON')
        if isinstance(value, bytes):
            value = value.decode('utf-8')
        if not isinstance(value, str):
            raise ValueError('expected UTF-8 JSON text')
        return json.loads(value, object_pairs_hook=_pairs,
                          parse_constant=lambda value: (_ for _ in ()).throw(ValueError('nonfinite JSON: ' + value)))
    except (ValueError, TypeError, UnicodeError) as exc:
        raise ValueError('INVALID_JSON_EVIDENCE:' + path + ':' + str(exc)) from exc

def records(group, name):
    item = group.get(name)
    if not isinstance(item, h5py.Dataset):
        return []
    raw = item[()]
    if isinstance(raw, np.ndarray) and raw.dtype.kind in 'OSU' and raw.ndim == 1:
        values = list(raw)
    else:
        if isinstance(raw, np.ndarray) and raw.dtype == np.uint8 and raw.ndim == 1:
            raw = raw.tobytes()
        if name.endswith('jsonl'):
            if isinstance(raw, bytes):
                raw = raw.decode('utf-8')
            values = [line for line in raw.splitlines() if line.strip()]
        else:
            values = [raw]
    result = []
    for i, raw in enumerate(values):
        value = decode(raw, item.name + ':' + str(i))
        rows = value if isinstance(value, list) else [value]
        if any(not isinstance(row, dict) for row in rows):
            raise ValueError('INVALID_JSON_EVIDENCE:' + item.name + ':expected object records')
        result.extend(rows)
    return result

def object_dataset(group, name):
    rows = records(group, name)
    if len(rows) > 1:
        raise ValueError('INVALID_JSON_EVIDENCE:' + group.name + '/' + name + ':multiple objects')
    return rows[0] if rows else {}
