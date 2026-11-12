# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded MessagePack encoding for numeric NumPy arrays."""

from __future__ import annotations

from typing import Any, Mapping

import msgpack
import numpy as np

from robolab_runner.errors import ProtocolError

_ARRAY_MARKER = "__robolab_ndarray_v1__"
_MAX_DEPTH = 32
_MAX_CONTAINER_ITEMS = 65_536


def _encode(value: Any, depth: int = 0) -> Any:
    if depth > _MAX_DEPTH:
        raise ProtocolError("message nesting exceeds the protocol limit")
    if isinstance(value, np.ndarray):
        dtype = value.dtype
        if dtype.hasobject or dtype.fields or dtype.kind not in {"b", "i", "u", "f"}:
            raise ProtocolError(f"unsafe ndarray dtype: {dtype}")
        if value.ndim > 8 or any(dimension < 0 for dimension in value.shape):
            raise ProtocolError(f"unsafe ndarray shape: {value.shape}")
        canonical_dtype = np.dtype(dtype.name).newbyteorder("<")
        contiguous = np.ascontiguousarray(value, dtype=canonical_dtype)
        return {
            _ARRAY_MARKER: True,
            "dtype": dtype.name,
            "shape": list(value.shape),
            "data": contiguous.tobytes(order="C"),
        }
    if isinstance(value, np.generic):
        return _encode(value.item(), depth + 1)
    if isinstance(value, Mapping):
        if len(value) > _MAX_CONTAINER_ITEMS:
            raise ProtocolError("MessagePack map has too many entries")
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ProtocolError("MessagePack map keys must be strings")
            result[key] = _encode(item, depth + 1)
        return result
    if isinstance(value, (list, tuple)):
        if len(value) > _MAX_CONTAINER_ITEMS:
            raise ProtocolError("MessagePack array has too many entries")
        return [_encode(item, depth + 1) for item in value]
    if value is None or isinstance(value, (str, bytes, bool, int, float)):
        if isinstance(value, float) and not np.isfinite(value):
            raise ProtocolError("message scalar contains NaN or infinity")
        return value
    raise ProtocolError(f"unsupported MessagePack value: {type(value).__name__}")


def _decode(value: Any, depth: int = 0) -> Any:
    if depth > _MAX_DEPTH:
        raise ProtocolError("message nesting exceeds the protocol limit")
    if isinstance(value, dict):
        if value.get(_ARRAY_MARKER) is True:
            if set(value) != {_ARRAY_MARKER, "dtype", "shape", "data"}:
                raise ProtocolError("malformed ndarray envelope")
            dtype_text = value.get("dtype")
            shape_value = value.get("shape")
            data = value.get("data")
            if not isinstance(dtype_text, str) or not isinstance(shape_value, list) or not isinstance(data, bytes):
                raise ProtocolError("malformed ndarray fields")
            try:
                dtype = np.dtype(dtype_text)
            except TypeError as error:
                raise ProtocolError(f"invalid ndarray dtype {dtype_text!r}") from error
            if dtype.name != dtype_text or dtype.hasobject or dtype.fields or dtype.kind not in {"b", "i", "u", "f"}:
                raise ProtocolError(f"unsafe ndarray dtype: {dtype_text!r}")
            if len(shape_value) > 8:
                raise ProtocolError("ndarray rank exceeds 8")
            shape: list[int] = []
            items = 1
            for dimension in shape_value:
                if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 0:
                    raise ProtocolError("ndarray dimensions must be non-negative integers")
                shape.append(dimension)
                items *= dimension
            expected_bytes = items * dtype.itemsize
            if expected_bytes != len(data):
                raise ProtocolError(
                    f"ndarray payload has {len(data)} bytes, expected {expected_bytes} for {dtype_text}{tuple(shape)}"
                )
            wire_dtype = dtype.newbyteorder("<")
            array = np.frombuffer(data, dtype=wire_dtype).reshape(tuple(shape))
            return np.array(array, dtype=dtype, order="C", copy=True)
        return {str(key): _decode(item, depth + 1) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode(item, depth + 1) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        raise ProtocolError("message scalar contains NaN or infinity")
    return value


def pack_message(value: Mapping[str, Any], *, max_bytes: int) -> bytes:
    try:
        packed = msgpack.packb(_encode(value), use_bin_type=True, strict_types=True)
    except (OverflowError, ValueError, TypeError, msgpack.PackException) as error:
        raise ProtocolError(f"failed to encode MessagePack: {error}") from error
    if len(packed) > max_bytes:
        raise ProtocolError(f"message uses {len(packed)} bytes, exceeding max_message_bytes={max_bytes}")
    return packed


def unpack_message(data: bytes, *, max_bytes: int) -> dict[str, Any]:
    if len(data) > max_bytes:
        raise ProtocolError(f"message uses {len(data)} bytes, exceeding max_message_bytes={max_bytes}")
    try:
        value = msgpack.unpackb(
            data,
            raw=False,
            strict_map_key=True,
            max_bin_len=max_bytes,
            max_str_len=max_bytes,
            max_array_len=_MAX_CONTAINER_ITEMS,
            max_map_len=_MAX_CONTAINER_ITEMS,
            ext_hook=lambda code, payload: (_ for _ in ()).throw(
                ProtocolError(f"MessagePack extension type {code} is forbidden")
            ),
        )
    except ProtocolError:
        raise
    except (ValueError, TypeError, msgpack.UnpackException) as error:
        raise ProtocolError(f"failed to decode MessagePack: {error}") from error
    decoded = _decode(value)
    if not isinstance(decoded, dict):
        raise ProtocolError("protocol message root must be a map")
    return decoded


__all__ = ["pack_message", "unpack_message"]
