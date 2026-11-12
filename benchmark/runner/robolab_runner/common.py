# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical JSON, hashes, and no-overwrite atomic publication."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from robolab_runner.errors import RunnerError


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def canonical_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise RunnerError(f"value is not canonical JSON: {error}") from error


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"required file does not exist: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RunnerError(f"failed to read JSON {source}: {error}") from error
    if not isinstance(value, dict):
        raise RunnerError(f"JSON root must be an object: {source}")
    return value


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError:
        pass


def publish_no_replace(source: str | Path, target: str | Path) -> None:
    source_path = Path(source).expanduser().resolve()
    target_path = Path(target).expanduser().resolve()
    if source_path.parent != target_path.parent:
        raise RunnerError("atomic publication requires source and target in the same directory")
    try:
        os.link(source_path, target_path)
    except FileExistsError:
        raise FileExistsError(f"refusing to overwrite existing artifact: {target_path}") from None
    except OSError as error:
        raise RunnerError(f"failed to atomically publish {target_path}: {error}") from error
    try:
        source_path.unlink()
    except Exception:
        try:
            target_path.unlink()
        except OSError:
            pass
        raise
    _fsync_directory(target_path.parent)


def write_json_atomic(path: str | Path, value: Mapping[str, Any]) -> Path:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite existing report: {target}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".partial", dir=target.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(dict(value), stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        publish_no_replace(temporary, target)
        return target
    finally:
        if temporary.exists():
            temporary.unlink()


__all__ = [
    "canonical_json",
    "load_json",
    "publish_no_replace",
    "sha256_file",
    "sha256_json",
    "utc_now",
    "write_json_atomic",
]
