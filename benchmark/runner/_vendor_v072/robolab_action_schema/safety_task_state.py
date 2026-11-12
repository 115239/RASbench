"""Safety-only HDF5 evidence for control-aligned USD boolean attributes."""

from __future__ import annotations

import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Iterable, Optional
from urllib.parse import quote, unquote

import h5py
import numpy as np

from .v4_hdf5_schema import DATA_ROOT, SchemaValidationError, entity_id

SAFETY_USD_ATTRIBUTE_EVIDENCE_SCHEMA = (
    "robolab-safety-usd-attribute-evidence/v0.1"
)
SAFETY_USD_ATTRIBUTE_EVIDENCE_METADATA_KEY = "safety_usd_attribute_evidence_schema"
USD_ATTRIBUTE_INITIAL_ROOT = f"{DATA_ROOT}/initial_state/usd_attribute"
USD_ATTRIBUTE_STATE_ROOT = f"{DATA_ROOT}/states/usd_attribute"
USD_BOOLEAN_ATTRIBUTE_NAMESPACE = "state:"


@dataclass(frozen=True, order=True)
class UsdBooleanAttributeSpec:
    """One bool state attribute discovered from the final composed Stage."""

    prim_path: str
    attribute_name: str


@dataclass(frozen=True)
class UsdBooleanAttributeEvidenceValidation:
    attribute_count: int
    frame_count: int


@dataclass(frozen=True)
class UsdBooleanAttributeRecord:
    initial_value: bool
    values: object


def discover_usd_boolean_attributes(stage: Any) -> tuple[UsdBooleanAttributeSpec, ...]:
    """Discover the complete ``state:*`` bool surface on a composed USD Stage.

    Discovery deliberately depends only on the final Stage, never on task_id,
    a Rule Package, or a per-task manifest.  An empty result is valid.
    """

    traverse = getattr(stage, "Traverse", None)
    if not callable(traverse):
        raise SchemaValidationError("USD Stage must provide Traverse()")
    discovered: set[UsdBooleanAttributeSpec] = set()
    try:
        prims = traverse()
        for prim in prims:
            prim_path = str(prim.GetPath())
            entity_id(prim_path)
            for attribute in prim.GetAttributes():
                attribute_name = str(attribute.GetName())
                if not attribute_name.startswith(USD_BOOLEAN_ATTRIBUTE_NAMESPACE):
                    continue
                suffix = attribute_name[len(USD_BOOLEAN_ATTRIBUTE_NAMESPACE) :]
                if not suffix:
                    raise SchemaValidationError(
                        "USD state attribute name must have a non-empty suffix"
                    )
                attribute_id(attribute_name)
                if str(attribute.GetTypeName()) != "bool":
                    continue
                spec = UsdBooleanAttributeSpec(prim_path, attribute_name)
                if spec in discovered:
                    raise SchemaValidationError(
                        "final USD Stage contains a duplicate boolean state attribute: "
                        f"{prim_path}.{attribute_name}"
                    )
                discovered.add(spec)
    except SchemaValidationError:
        raise
    except Exception as error:
        raise SchemaValidationError(
            f"cannot discover USD boolean state attributes: {error}"
        ) from error
    return tuple(sorted(discovered))


def attribute_id(attribute_name: str) -> str:
    """Encode one NFC USD attribute name as a safe HDF5 path component."""

    if (
        not isinstance(attribute_name, str)
        or not attribute_name
        or attribute_name != attribute_name.strip()
        or unicodedata.normalize("NFC", attribute_name) != attribute_name
        or "/" in attribute_name
    ):
        raise SchemaValidationError(
            "USD attribute name must be non-empty NFC text without '/' or whitespace padding"
        )
    return quote(attribute_name, safe="-._~", encoding="utf-8", errors="strict")


def usd_attribute_value_path(
    prim_path: str,
    attribute_name: str,
    *,
    initial: bool = False,
) -> str:
    root = USD_ATTRIBUTE_INITIAL_ROOT if initial else USD_ATTRIBUTE_STATE_ROOT
    return f"{root}/{entity_id(prim_path)}/{attribute_id(attribute_name)}/value"


def parse_usd_attribute_value_path(path: str) -> tuple[str, str, bool]:
    """Parse and validate one canonical USD attribute evidence dataset path."""

    if not isinstance(path, str):
        raise SchemaValidationError("USD attribute evidence path must be text")
    roots = (
        (USD_ATTRIBUTE_INITIAL_ROOT, True),
        (USD_ATTRIBUTE_STATE_ROOT, False),
    )
    for root, initial in roots:
        prefix = root + "/"
        if not path.startswith(prefix):
            continue
        suffix = path[len(prefix) :]
        parts = suffix.split("/")
        if len(parts) != 3 or parts[2] != "value" or not all(parts[:2]):
            break
        encoded_entity, encoded_attribute = parts[:2]
        try:
            prim_path = unquote(encoded_entity, encoding="utf-8", errors="strict")
            attribute_name = unquote(
                encoded_attribute, encoding="utf-8", errors="strict"
            )
            expected = usd_attribute_value_path(
                prim_path,
                attribute_name,
                initial=initial,
            )
        except (UnicodeDecodeError, SchemaValidationError) as error:
            raise SchemaValidationError(
                f"invalid USD attribute evidence path: {path}"
            ) from error
        if path != expected:
            raise SchemaValidationError(
                f"non-canonical USD attribute evidence path: {path}"
            )
        return prim_path, attribute_name, initial
    raise SchemaValidationError(f"invalid USD attribute evidence path: {path}")


def _metadata_text(attributes: h5py.AttributeManager, name: str) -> str:
    value = attributes.get(name)
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except UnicodeDecodeError as error:
            raise SchemaValidationError(f"metadata.{name} must be UTF-8") from error
    if not isinstance(value, str) or value != value.strip() or not value:
        raise SchemaValidationError(f"metadata.{name} must be normalized non-empty text")
    return value


def _group_or_none(hdf5_file: h5py.File, path: str) -> Optional[h5py.Group]:
    node = hdf5_file.get(path)
    if node is None:
        return None
    if not isinstance(node, h5py.Group):
        raise SchemaValidationError(f"USD attribute evidence root must be a group: {path}")
    return node


def _canonical_attribute_name(encoded: str, path: str) -> str:
    try:
        decoded = unquote(encoded, encoding="utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise SchemaValidationError(
            f"USD attribute evidence has invalid UTF-8 identity: {path}"
        ) from error
    if attribute_id(decoded) != encoded:
        raise SchemaValidationError(
            f"USD attribute evidence has non-canonical attribute identity: {path}"
        )
    if (
        not decoded.startswith(USD_BOOLEAN_ATTRIBUTE_NAMESPACE)
        or not decoded[len(USD_BOOLEAN_ATTRIBUTE_NAMESPACE) :]
    ):
        raise SchemaValidationError(
            f"USD boolean evidence attribute must match state:*: {path}"
        )
    return decoded


def _collect_leaf_paths(root: h5py.Group) -> dict[tuple[str, str], h5py.Dataset]:
    result: dict[tuple[str, str], h5py.Dataset] = {}
    for encoded_entity in sorted(root.keys()):
        entity_group = root[encoded_entity]
        if not isinstance(entity_group, h5py.Group):
            raise SchemaValidationError(
                f"USD attribute entity must be a group: {entity_group.name}"
            )
        try:
            prim_path = unquote(encoded_entity, encoding="utf-8", errors="strict")
            canonical_entity = entity_id(prim_path)
        except (UnicodeDecodeError, SchemaValidationError) as error:
            raise SchemaValidationError(
                f"USD attribute evidence has invalid entity identity: {entity_group.name}"
            ) from error
        if canonical_entity != encoded_entity:
            raise SchemaValidationError(
                f"USD attribute evidence has non-canonical entity identity: {entity_group.name}"
            )
        for encoded_attribute in sorted(entity_group.keys()):
            attribute_group = entity_group[encoded_attribute]
            if not isinstance(attribute_group, h5py.Group):
                raise SchemaValidationError(
                    f"USD attribute identity must be a group: {attribute_group.name}"
                )
            _canonical_attribute_name(encoded_attribute, attribute_group.name)
            if set(attribute_group.keys()) != {"value"}:
                raise SchemaValidationError(
                    "USD boolean attribute group must contain exactly the value dataset: "
                    f"{attribute_group.name}"
                )
            value = attribute_group["value"]
            if not isinstance(value, h5py.Dataset):
                raise SchemaValidationError(
                    f"USD boolean attribute value must be a dataset: {value.name}"
                )
            result[(encoded_entity, encoded_attribute)] = value
    return result


def _validate_uint8_boolean(dataset: h5py.Dataset, shape: tuple[int, ...]) -> None:
    if dataset.dtype != np.dtype(np.uint8) or tuple(dataset.shape) != shape:
        raise SchemaValidationError(
            "USD boolean attribute evidence has invalid dtype/shape: "
            f"{dataset.name} {dataset.dtype} {tuple(dataset.shape)}, "
            f"expected uint8 {shape}"
        )
    values = np.asarray(dataset[...], dtype=np.uint8)
    if np.any((values != 0) & (values != 1)):
        raise SchemaValidationError(
            f"USD boolean attribute evidence must contain only 0 or 1: {dataset.name}"
        )


def validate_usd_boolean_attribute_evidence(
    hdf5_file: h5py.File,
) -> UsdBooleanAttributeEvidenceValidation:
    """Validate the optional extension and its exact initial/current closure."""

    initial_root = _group_or_none(hdf5_file, USD_ATTRIBUTE_INITIAL_ROOT)
    state_root = _group_or_none(hdf5_file, USD_ATTRIBUTE_STATE_ROOT)
    metadata = hdf5_file.get("/metadata")
    if not isinstance(metadata, h5py.Group):
        raise SchemaValidationError("HDF5 is missing /metadata")
    declared = SAFETY_USD_ATTRIBUTE_EVIDENCE_METADATA_KEY in metadata.attrs
    present = initial_root is not None or state_root is not None
    if not present:
        if declared:
            raise SchemaValidationError(
                "USD attribute evidence metadata is present without evidence datasets"
            )
        return UsdBooleanAttributeEvidenceValidation(0, 0)
    if initial_root is None or state_root is None:
        raise SchemaValidationError(
            "USD attribute evidence requires both initial_state and states roots"
        )
    if not declared or _metadata_text(
        metadata.attrs, SAFETY_USD_ATTRIBUTE_EVIDENCE_METADATA_KEY
    ) != SAFETY_USD_ATTRIBUTE_EVIDENCE_SCHEMA:
        raise SchemaValidationError(
            "metadata.safety_usd_attribute_evidence_schema must identify v0.1"
        )

    actions = hdf5_file.get(f"{DATA_ROOT}/actions")
    if not isinstance(actions, h5py.Dataset) or actions.ndim < 1:
        raise SchemaValidationError("cannot determine control-step count for USD attributes")
    frame_count = int(actions.shape[0])
    initial = _collect_leaf_paths(initial_root)
    states = _collect_leaf_paths(state_root)
    if not initial or set(initial) != set(states):
        raise SchemaValidationError(
            "USD attribute initial/current identities must be non-empty and exactly equal"
        )
    for identity in sorted(initial):
        _validate_uint8_boolean(initial[identity], ())
        _validate_uint8_boolean(states[identity], (frame_count,))
    return UsdBooleanAttributeEvidenceValidation(len(initial), frame_count)


def write_usd_boolean_attribute_evidence(
    hdf5_file: h5py.File,
    records: Mapping[tuple[str, str], UsdBooleanAttributeRecord],
) -> UsdBooleanAttributeEvidenceValidation:
    """Write one exact extension closure into an open unpublished HDF5 file."""

    if not isinstance(records, Mapping) or not records:
        raise SchemaValidationError("USD boolean attribute records must be non-empty")
    metadata = hdf5_file.get("/metadata")
    actions = hdf5_file.get(f"{DATA_ROOT}/actions")
    if not isinstance(metadata, h5py.Group) or not isinstance(actions, h5py.Dataset):
        raise SchemaValidationError(
            "USD attribute evidence requires existing metadata and actions"
        )
    if USD_ATTRIBUTE_INITIAL_ROOT in hdf5_file or USD_ATTRIBUTE_STATE_ROOT in hdf5_file:
        raise SchemaValidationError("USD attribute evidence roots already exist")
    frame_count = int(actions.shape[0]) if actions.ndim >= 1 else 0
    if frame_count < 1:
        raise SchemaValidationError("USD attribute evidence requires at least one action")
    metadata.attrs[
        SAFETY_USD_ATTRIBUTE_EVIDENCE_METADATA_KEY
    ] = SAFETY_USD_ATTRIBUTE_EVIDENCE_SCHEMA
    for identity in sorted(records):
        if (
            not isinstance(identity, tuple)
            or len(identity) != 2
            or not all(isinstance(item, str) for item in identity)
        ):
            raise SchemaValidationError(
                "USD boolean attribute record keys must be (prim_path, attribute_name)"
            )
        record = records[identity]
        if not isinstance(record, UsdBooleanAttributeRecord) or not isinstance(
            record.initial_value, (bool, np.bool_)
        ):
            raise SchemaValidationError(
                "USD boolean attribute record requires a boolean initial_value"
            )
        values = np.asarray(record.values)
        if values.dtype not in {np.dtype(np.bool_), np.dtype(np.uint8)}:
            raise SchemaValidationError(
                "USD boolean attribute record values must use bool or uint8 dtype"
            )
        if values.shape != (frame_count,) or np.any((values != 0) & (values != 1)):
            raise SchemaValidationError(
                f"USD boolean attribute record values must contain [T] boolean 0/1: {identity}"
            )
        prim_path, attribute_name = identity
        hdf5_file.create_dataset(
            usd_attribute_value_path(prim_path, attribute_name, initial=True),
            data=np.uint8(record.initial_value),
        )
        hdf5_file.create_dataset(
            usd_attribute_value_path(prim_path, attribute_name),
            data=values.astype(np.uint8, copy=False),
        )
    return validate_usd_boolean_attribute_evidence(hdf5_file)


def write_discovered_usd_boolean_attribute_evidence(
    hdf5_file: h5py.File,
    discovered_attributes: Iterable[UsdBooleanAttributeSpec],
    records: Mapping[tuple[str, str], UsdBooleanAttributeRecord],
) -> UsdBooleanAttributeEvidenceValidation:
    """Write records only when they exactly close the discovered Stage surface."""

    discovered = tuple(discovered_attributes)
    if any(not isinstance(item, UsdBooleanAttributeSpec) for item in discovered):
        raise SchemaValidationError(
            "discovered USD boolean attributes must use UsdBooleanAttributeSpec"
        )
    if len(set(discovered)) != len(discovered):
        raise SchemaValidationError("discovered USD boolean attributes must be unique")
    expected = {(item.prim_path, item.attribute_name) for item in discovered}
    if not isinstance(records, Mapping) or set(records) != expected:
        raise SchemaValidationError(
            "USD boolean attribute records must exactly match final Stage discovery"
        )
    if not discovered:
        return validate_usd_boolean_attribute_evidence(hdf5_file)
    return write_usd_boolean_attribute_evidence(hdf5_file, records)
