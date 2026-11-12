"""Build and validate release-time HDF5 requirements from trusted Rule Packages."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import PurePosixPath
from typing import Any

import h5py
import numpy as np

from .safety_task_state import parse_usd_attribute_value_path
from .v4_hdf5_schema import (
    DATA_ROOT,
    SCHEMA_NAME,
    SCHEMA_VERSION,
    SchemaValidationError,
    entity_id,
)

REQUIREMENTS_SCHEMA = "robolab-action-hdf5-requirements/v1"
_REQUIREMENTS_KEYS = {
    "requirements_schema",
    "hdf5_schema",
    "task_id",
    "identity_mode",
    "datasets",
    "articulations",
}
_ARTICULATION_KEYS = {"entity_id", "prim_path", "joint_paths", "body_paths"}


def _selector_usages(package: Mapping[str, Any]) -> Iterable[dict[str, Any]]:
    fields = ("items", "hard_gates", "ai_items", "fatal_checks", "outcome_checks")

    def walk(value: object, *, owner_id: str, location: str) -> Iterable[dict[str, Any]]:
        if isinstance(value, Mapping):
            inputs = value.get("inputs")
            if isinstance(inputs, Mapping):
                for input_name, selector in inputs.items():
                    if isinstance(selector, Mapping) and "path" in selector:
                        yield {
                            "owner_id": owner_id,
                            "input_name": str(input_name),
                            "location": f"{location}.inputs.{input_name}",
                            "selector": dict(selector),
                        }
            for key, child in value.items():
                if key != "inputs":
                    yield from walk(
                        child,
                        owner_id=owner_id,
                        location=f"{location}.{key}",
                    )
        elif isinstance(value, list):
            for index, child in enumerate(value):
                yield from walk(
                    child,
                    owner_id=owner_id,
                    location=f"{location}[{index}]",
                )

    for field in fields:
        for index, item in enumerate(package.get(field, [])):
            yield from walk(
                item,
                owner_id=str(item.get("id", "")),
                location=f"{field}[{index}]",
            )


def collect_hdf5_selectors(package: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return a stable, JSON-serializable projection of all package selectors."""

    return sorted(
        _selector_usages(package),
        key=lambda item: (
            str(item["selector"].get("path", "")),
            item["owner_id"],
            item["input_name"],
            item["location"],
        ),
    )


def _stream_companion_paths(path: str) -> tuple[str, ...]:
    if (
        "/feedback/articulation/" not in path
        and "/sensors/contact/" not in path
    ):
        return ()
    group, field = path.rsplit("/", 1)
    if field in {"step_index", "sim_time_s"}:
        return ()
    return (f"{group}/step_index", f"{group}/sim_time_s")


def _articulation_axes(context: Mapping[str, Any]) -> dict[str, tuple[int, int]]:
    result: dict[str, tuple[int, int]] = {}
    for articulation in context.get("articulations", []):
        if isinstance(articulation, Mapping):
            result[str(articulation.get("entity_id", ""))] = (
                len(articulation.get("joint_paths", [])),
                len(articulation.get("body_paths", [])),
            )
    return result


def infer_dataset_contract(
    path: str,
    trajectory_context: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Infer the formal V4 source-dataset contract for one selector path."""

    parts = PurePosixPath(path).parts
    if len(parts) < 4 or parts[1:3] != ("data", "demo_0"):
        return None
    tail = parts[3:]
    axes = _articulation_axes(trajectory_context or {})

    if len(tail) == 2 and tail[0] == "tracked_pose":
        widths = {
            "position": 3,
            "orientation": 4,
            "linear_velocity": 3,
            "angular_velocity": 3,
        }
        if tail[1] in widths:
            return {"dtype": "float32", "shape": ["T", widths[tail[1]]]}

    if len(tail) == 4 and tail[0] in {"initial_state", "states"}:
        state, kind, entity, field = tail
        if kind == "rigid_object" and field in {"root_pose", "root_velocity"}:
            width = 7 if field == "root_pose" else 6
            return {
                "dtype": "float32",
                "shape": [width] if state == "initial_state" else ["T", width],
            }
        if kind == "articulation":
            joint_count, _ = axes.get(entity, (None, None))
            widths: dict[str, int | None] = {
                "root_pose": 7,
                "root_velocity": 6,
                "joint_position": joint_count,
                "joint_velocity": joint_count,
            }
            if field in widths and widths[field] is not None:
                width = int(widths[field])
                return {
                    "dtype": "float32",
                    "shape": [width] if state == "initial_state" else ["T", width],
                }

    if len(tail) == 4 and tail[:2] == ("feedback", "articulation"):
        _, _, entity, field = tail
        joint_count, body_count = axes.get(entity, (None, None))
        if field in {"step_index", "sim_time_s"}:
            return {
                "dtype": "uint64" if field == "step_index" else "float64",
                "shape": ["S"],
            }
        if field in {
            "joint_position",
            "joint_velocity",
            "joint_position_target",
            "joint_velocity_target",
            "joint_effort_target",
            "applied_torque",
            "computed_torque",
        } and joint_count is not None:
            return {"dtype": "float32", "shape": ["S", joint_count]}
        if field in {"body_pose", "body_velocity"} and body_count is not None:
            return {
                "dtype": "float32",
                "shape": ["S", body_count, 7 if field == "body_pose" else 6],
            }

    if len(tail) == 3 and tail[0] == "task_reference" and tail[2] == "pose":
        return {"dtype": "float32", "shape": [7]}

    if len(tail) == 4 and tail[:2] == ("states", "gripper") and tail[3] == "closure":
        return {"dtype": "float32", "shape": ["T"], "range": [0.0, 1.0]}

    if (
        len(tail) == 5
        and tail[0] in {"initial_state", "states"}
        and tail[1] == "usd_attribute"
        and tail[4] == "value"
    ):
        try:
            parse_usd_attribute_value_path(path)
        except SchemaValidationError:
            return None
        return {
            "dtype": "uint8",
            "shape": [] if tail[0] == "initial_state" else ["T"],
            "range": [0, 1],
        }

    if len(tail) == 4 and tail[:2] == ("sensors", "contact"):
        pair, field = tail[2], tail[3]
        if field in {"step_index", "sim_time_s"}:
            return {
                "dtype": "uint64" if field == "step_index" else "float64",
                "shape": ["S"],
            }
        if field == "net_forces_w":
            contact_count = 1
            matching = [
                (entity_id, body_count)
                for entity_id, (_joint_count, body_count) in axes.items()
                if pair.startswith(f"{entity_id}__with__")
            ]
            if matching:
                _, contact_count = max(matching, key=lambda item: len(item[0]))
            return {"dtype": "float32", "shape": ["S", contact_count, 3]}
    return None


def build_hdf5_requirements(
    package: Mapping[str, Any],
    *,
    task_id: str,
    identity_mode: str,
    runtime_stage_manifest: str | None = None,
) -> dict[str, Any]:
    """Project one trusted Rule Package into a release-time evidence manifest."""

    context = package.get("trajectory_context", {})
    if not isinstance(context, Mapping):
        context = {}
    if not isinstance(identity_mode, str) or not identity_mode.strip():
        raise SchemaValidationError("identity_mode must be non-empty")
    if not isinstance(task_id, str) or not task_id or task_id != task_id.strip():
        raise SchemaValidationError("task_id must be non-empty without surrounding whitespace")

    by_path: dict[str, dict[str, Any]] = {}
    for usage in collect_hdf5_selectors(package):
        selector = usage["selector"]
        path = str(selector["path"])
        contract = infer_dataset_contract(path, context)
        if contract is None:
            raise SchemaValidationError(
                f"Rule Package selector is outside the formal V4 evidence surface: {path}"
            )
        compact_usage = {
            "owner_id": usage["owner_id"],
            "input_name": usage["input_name"],
            "location": usage["location"],
            "selector": selector,
        }
        existing = by_path.get(path)
        if existing is None:
            by_path[path] = {
                "path": path,
                **contract,
                "rank": len(contract["shape"]),
                "required": True,
                "usages": [compact_usage],
            }
        else:
            if any(existing.get(key) != value for key, value in contract.items()):
                raise SchemaValidationError(
                    f"one HDF5 path has inconsistent inferred contracts: {path}"
                )
            existing.pop("reason", None)
            existing["usages"].append(compact_usage)

        for companion_path in _stream_companion_paths(path):
            companion_contract = infer_dataset_contract(companion_path, context)
            if companion_contract is None:  # pragma: no cover - paired with a known stream
                raise SchemaValidationError(
                    f"cannot infer required stream companion: {companion_path}"
                )
            companion = by_path.get(companion_path)
            if companion is None:
                by_path[companion_path] = {
                    "path": companion_path,
                    **companion_contract,
                    "rank": len(companion_contract["shape"]),
                    "required": True,
                    "reason": "stream_alignment",
                    "usages": [],
                }
            elif any(
                companion.get(key) != value
                for key, value in companion_contract.items()
            ):
                raise SchemaValidationError(
                    f"one HDF5 path has inconsistent inferred contracts: {companion_path}"
                )

    result: dict[str, Any] = {
        "requirements_schema": REQUIREMENTS_SCHEMA,
        "hdf5_schema": {"name": SCHEMA_NAME, "version": SCHEMA_VERSION},
        "task_id": task_id,
        "identity_mode": identity_mode,
        "datasets": [by_path[path] for path in sorted(by_path)],
        "articulations": list(context.get("articulations", [])),
    }
    if runtime_stage_manifest is not None:
        result["runtime_stage_manifest"] = runtime_stage_manifest
    validate_hdf5_requirements(result, package=package)
    return result


def _canonical_selector_occurrences(
    value: Iterable[Mapping[str, Any]],
) -> Counter[str]:
    return Counter(
        json.dumps(
            {
                "owner_id": item.get("owner_id"),
                "input_name": item.get("input_name"),
                "location": item.get("location"),
                "selector": item.get("selector"),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        for item in value
    )


def _validate_runtime_stage_manifest_reference(value: object) -> None:
    if not isinstance(value, str) or not value:
        raise SchemaValidationError(
            "HDF5 requirements runtime_stage_manifest must be a non-empty path"
        )
    if "\\" in value:
        raise SchemaValidationError(
            "HDF5 requirements runtime_stage_manifest must use POSIX separators"
        )
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise SchemaValidationError(
            "HDF5 requirements runtime_stage_manifest must be a safe relative POSIX path"
        )


def _validate_axis_paths(
    value: object,
    name: str,
    root_path: str,
    *,
    allow_physical_siblings: bool,
) -> None:
    if not isinstance(value, list) or any(not isinstance(path, str) for path in value):
        raise SchemaValidationError(f"{name} must be a list of absolute Prim paths")
    if value != sorted(set(value)):
        raise SchemaValidationError(f"{name} must be strictly sorted and unique")
    prefix = root_path.rstrip("/") + "/"
    for path in value:
        try:
            entity_id(path)
        except SchemaValidationError as error:
            raise SchemaValidationError(f"{name} contains an invalid Prim path: {path!r}") from error
        if (
            not allow_physical_siblings
            and path != root_path
            and not path.startswith(prefix)
        ):
            raise SchemaValidationError(
                f"{name} Prim path must be at or below articulation root {root_path!r}: {path!r}"
            )


def _validate_requirement_articulations(
    value: object,
    *,
    allow_physical_siblings: bool,
) -> None:
    if not isinstance(value, list):
        raise SchemaValidationError("HDF5 requirements articulations must be a list")
    seen: set[str] = set()
    for index, articulation in enumerate(value):
        name = f"articulations[{index}]"
        if not isinstance(articulation, Mapping) or set(articulation) != _ARTICULATION_KEYS:
            raise SchemaValidationError(f"{name} has an invalid key set")
        prim_path = articulation.get("prim_path")
        if not isinstance(prim_path, str):
            raise SchemaValidationError(f"{name}.prim_path must be an absolute Prim path")
        try:
            expected_entity_id = entity_id(prim_path)
        except SchemaValidationError as error:
            raise SchemaValidationError(f"{name}.prim_path is invalid: {error}") from error
        if articulation.get("entity_id") != expected_entity_id:
            raise SchemaValidationError(f"{name}.entity_id does not match prim_path")
        if expected_entity_id in seen:
            raise SchemaValidationError(
                f"HDF5 requirements contain duplicate articulation: {expected_entity_id}"
            )
        seen.add(expected_entity_id)
        _validate_axis_paths(
            articulation.get("joint_paths"),
            f"{name}.joint_paths",
            prim_path,
            allow_physical_siblings=allow_physical_siblings,
        )
        _validate_axis_paths(
            articulation.get("body_paths"),
            f"{name}.body_paths",
            prim_path,
            allow_physical_siblings=allow_physical_siblings,
        )


def validate_hdf5_requirements(
    requirements: Mapping[str, Any],
    *,
    package: Mapping[str, Any] | None = None,
) -> None:
    allowed_root_keys = set(_REQUIREMENTS_KEYS)
    if "runtime_stage_manifest" in requirements:
        allowed_root_keys.add("runtime_stage_manifest")
    if set(requirements) != allowed_root_keys:
        raise SchemaValidationError("HDF5 requirements has an invalid top-level key set")
    if requirements.get("requirements_schema") != REQUIREMENTS_SCHEMA:
        raise SchemaValidationError(
            f"requirements_schema must be {REQUIREMENTS_SCHEMA!r}"
        )
    schema = requirements.get("hdf5_schema")
    if schema != {"name": SCHEMA_NAME, "version": SCHEMA_VERSION}:
        raise SchemaValidationError("HDF5 requirements reference an unsupported schema")
    task_id = requirements.get("task_id")
    if not isinstance(task_id, str) or not task_id or task_id != task_id.strip():
        raise SchemaValidationError(
            "HDF5 requirements task_id must be non-empty without surrounding whitespace"
        )
    identity_mode = requirements.get("identity_mode")
    if not isinstance(identity_mode, str) or not identity_mode.strip():
        raise SchemaValidationError("HDF5 requirements identity_mode must be non-empty")
    if "runtime_stage_manifest" in requirements:
        _validate_runtime_stage_manifest_reference(
            requirements.get("runtime_stage_manifest")
        )
    package_schema = package.get("package_schema") if package is not None else None
    _validate_requirement_articulations(
        requirements.get("articulations"),
        # The official R1 Pro authors ArticulationRootAPI on base_link while
        # physically owned arm/wheel bodies and joints are siblings.  Only the
        # embodied package line may use that USD joint-graph ownership model;
        # safety/generic packages keep the stricter ancestry rule.
        allow_physical_siblings=(
            package_schema
            in {
                "robolab-action-rule-package/v0.5.2",
                "robolab-action-rule-package/v0.5.3",
            }
            or identity_mode == "embodied_judge_preflight"
        ),
    )
    datasets = requirements.get("datasets")
    if not isinstance(datasets, list):
        raise SchemaValidationError("HDF5 requirements datasets must be a list")
    paths: list[str] = []
    usages: list[Mapping[str, Any]] = []
    allowed_dataset_keys = {
        "path",
        "dtype",
        "shape",
        "rank",
        "required",
        "range",
        "reason",
        "usages",
    }
    for index, dataset in enumerate(datasets):
        if not isinstance(dataset, Mapping):
            raise SchemaValidationError(f"datasets[{index}] must be an object")
        unknown_keys = set(dataset) - allowed_dataset_keys
        if unknown_keys:
            raise SchemaValidationError(
                f"datasets[{index}] has unsupported keys: {sorted(unknown_keys)}"
            )
        path = dataset.get("path")
        if not isinstance(path, str) or not path.startswith(f"{DATA_ROOT}/"):
            raise SchemaValidationError(f"datasets[{index}].path is invalid")
        paths.append(path)
        shape = dataset.get("shape")
        if not isinstance(shape, list) or dataset.get("rank") != len(shape):
            raise SchemaValidationError(f"datasets[{index}] has an invalid shape/rank")
        if any(
            requirement not in {"T", "S"}
            and (
                not isinstance(requirement, int)
                or isinstance(requirement, bool)
                or requirement < 1
            )
            for requirement in shape
        ):
            raise SchemaValidationError(
                f"datasets[{index}].shape contains an invalid dimension"
            )
        dtype = dataset.get("dtype")
        if dtype not in {"float32", "float64", "uint8", "uint64"}:
            raise SchemaValidationError(f"datasets[{index}].dtype is invalid")
        if dataset.get("required") is not True:
            raise SchemaValidationError(f"datasets[{index}].required must be true")
        dataset_usages = dataset.get("usages")
        is_stream_companion = dataset.get("reason") == "stream_alignment"
        if not isinstance(dataset_usages, list) or (
            not dataset_usages and not is_stream_companion
        ):
            raise SchemaValidationError(
                f"datasets[{index}].usages must be non-empty unless it is a stream companion"
            )
        if is_stream_companion and dataset_usages:
            raise SchemaValidationError(
                f"datasets[{index}] stream companion must not contain selector usages"
            )
        if "reason" in dataset and not is_stream_companion:
            raise SchemaValidationError(f"datasets[{index}].reason is invalid")
        if "range" in dataset:
            value_range = dataset["range"]
            if (
                not isinstance(value_range, list)
                or len(value_range) != 2
                or any(
                    isinstance(value, bool) or not isinstance(value, (int, float))
                    for value in value_range
                )
                or not np.isfinite(np.asarray(value_range, dtype=np.float64)).all()
                or value_range[0] > value_range[1]
            ):
                raise SchemaValidationError(f"datasets[{index}].range is invalid")
        for usage in dataset_usages:
            if (
                not isinstance(usage, Mapping)
                or set(usage) != {"owner_id", "input_name", "location", "selector"}
                or any(
                    not isinstance(usage.get(key), str) or not usage.get(key)
                    for key in ("owner_id", "input_name", "location")
                )
                or not isinstance(usage.get("selector"), Mapping)
                or usage["selector"].get("path") != path
            ):
                raise SchemaValidationError(f"datasets[{index}] has an invalid selector usage")
            usages.append(usage)
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise SchemaValidationError("HDF5 requirement dataset paths must be unique and sorted")
    if package is not None:
        expected = _canonical_selector_occurrences(collect_hdf5_selectors(package))
        actual = _canonical_selector_occurrences(usages)
        if actual != expected:
            raise SchemaValidationError(
                "HDF5 requirements and Rule Package selectors are not bidirectionally aligned"
            )

        direct_paths = {
            str(usage["selector"]["path"])
            for usage in collect_hdf5_selectors(package)
        }
        expected_companions = {
            companion
            for usage in collect_hdf5_selectors(package)
            for companion in _stream_companion_paths(str(usage["selector"]["path"]))
        } - direct_paths
        actual_companions = {
            str(dataset["path"])
            for dataset in datasets
            if dataset.get("reason") == "stream_alignment"
        }
        if actual_companions != expected_companions:
            raise SchemaValidationError(
                "HDF5 requirements stream companions are not bidirectionally aligned"
            )


def _shape_matches(actual: tuple[int, ...], expected: list[object], frame_count: int) -> bool:
    if len(actual) != len(expected):
        return False
    for size, requirement in zip(actual, expected):
        if requirement == "T" and size != frame_count:
            return False
        if requirement == "S" and size < 1:
            return False
        if isinstance(requirement, int) and size != requirement:
            return False
    return True


def validate_requirements_against_submission(
    hdf5_file: h5py.File,
    requirements: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate every generated evidence requirement against one real HDF5."""

    validate_hdf5_requirements(requirements)
    expected_task_id = str(requirements["task_id"])
    metadata = hdf5_file.get("/metadata")
    if not isinstance(metadata, h5py.Group) or "task_id" not in metadata.attrs:
        raise SchemaValidationError("HDF5 metadata task_id is missing")
    raw_task_id = metadata.attrs["task_id"]
    if isinstance(raw_task_id, bytes):
        try:
            raw_task_id = raw_task_id.decode("utf-8")
        except UnicodeDecodeError as error:
            raise SchemaValidationError("HDF5 metadata task_id is not UTF-8") from error
    if not isinstance(raw_task_id, str) or raw_task_id != expected_task_id:
        raise SchemaValidationError(
            f"HDF5 metadata task_id does not match requirements: "
            f"{raw_task_id!r} != {expected_task_id!r}"
        )

    tracked = hdf5_file.get(f"{DATA_ROOT}/tracked_pose/position")
    if not isinstance(tracked, h5py.Dataset) or tracked.ndim < 1:
        raise SchemaValidationError("cannot determine HDF5 control-step count")
    frame_count = int(tracked.shape[0])
    errors: list[str] = []
    checked = 0
    stream_lengths: dict[str, list[tuple[str, int]]] = {}
    for requirement in requirements["datasets"]:
        path = str(requirement["path"])
        node = hdf5_file.get(path)
        if not isinstance(node, h5py.Dataset):
            errors.append(f"missing dataset: {path}")
            continue
        checked += 1
        expected_dtype = str(requirement["dtype"])
        if node.dtype != np.dtype(expected_dtype):
            errors.append(
                f"dtype mismatch: {path} {node.dtype} != {expected_dtype}"
            )
        if not _shape_matches(tuple(node.shape), list(requirement["shape"]), frame_count):
            errors.append(
                f"shape mismatch: {path} {tuple(node.shape)} != {requirement['shape']}"
            )
        if requirement["shape"] and requirement["shape"][0] == "S" and node.ndim:
            stream_lengths.setdefault(path.rsplit("/", 1)[0], []).append(
                (path, int(node.shape[0]))
            )
        if node.dtype.kind == "f" and not np.isfinite(np.asarray(node[...])).all():
            errors.append(f"non-finite values: {path}")
        value_range = requirement.get("range")
        if value_range is not None:
            values = np.asarray(node[...])
            if np.any(values < value_range[0]) or np.any(values > value_range[1]):
                errors.append(f"value range mismatch: {path}")
        for usage in requirement["usages"]:
            selector = usage["selector"]
            if "index" in selector:
                index = selector["index"]
                if (
                    not isinstance(index, int)
                    or isinstance(index, bool)
                    or index < 0
                    or node.ndim < 2
                    or index >= node.shape[1]
                ):
                    errors.append(f"index out of bounds: {path}[{index}]")
            if "slice" in selector:
                bounds = selector["slice"]
                valid_bounds = (
                    isinstance(bounds, list)
                    and len(bounds) == 2
                    and all(
                        isinstance(value, int)
                        and not isinstance(value, bool)
                        and value >= 0
                        for value in bounds
                    )
                    and bounds[0] < bounds[1]
                )
                start, stop = bounds if valid_bounds else ("?", "?")
                if (
                    not valid_bounds
                    or node.ndim == 0
                    or int(stop) > node.shape[-1]
                ):
                    errors.append(f"slice out of bounds: {path}[{start}:{stop}]")
    for group in sorted(stream_lengths):
        entries = stream_lengths[group]
        if len({length for _path, length in entries}) != 1:
            rendered = ", ".join(f"{path}={length}" for path, length in entries)
            errors.append(f"stream datasets are not aligned: {group} ({rendered})")

        step_path = f"{group}/step_index"
        time_path = f"{group}/sim_time_s"
        step_node = hdf5_file.get(step_path)
        time_node = hdf5_file.get(time_path)
        if not isinstance(step_node, h5py.Dataset) or not isinstance(time_node, h5py.Dataset):
            continue
        steps = np.asarray(step_node[...])
        times = np.asarray(time_node[...])
        if steps.size > 1 and np.any(np.diff(steps.astype(np.int64)) < 0):
            errors.append(f"stream step_index decreases: {step_path}")
        if np.any(steps >= frame_count):
            errors.append(f"stream step_index is out of range: {step_path}")
        if times.size > 1 and np.any(np.diff(times) < 0.0):
            errors.append(f"stream sim_time_s decreases: {time_path}")
    if errors:
        raise SchemaValidationError(
            "HDF5 requirement coverage failed: " + "; ".join(errors)
        )
    return {
        "status": "VALID",
        "task_id": requirements["task_id"],
        "required_dataset_count": len(requirements["datasets"]),
        "checked_dataset_count": checked,
    }
