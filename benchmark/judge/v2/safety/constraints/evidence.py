"""Strict reader for the safety-private ``/research`` HDF5 extension."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from types import MappingProxyType
from typing import Any, Union

import h5py
import numpy as np

from .schema import AttackMarkerPoint, EvidencePoint, ViolationWindow

MISSING_SAFETY_EVIDENCE = "MISSING_SAFETY_EVIDENCE"
INVALID_OR_MISSING_TIMEBASE = "INVALID_OR_MISSING_TIMEBASE"
INVALID_SAFETY_EVIDENCE = "INVALID_SAFETY_EVIDENCE"
_END_REASONS = frozenset(
    {"cleared", "observation_gap", "episode_end_still_active"}
)


class _InvalidSafetyEvidence(ValueError):
    pass


@dataclass(frozen=True)
class RecordingSegment:
    stream_id: str
    fully_observed: bool
    error_reason: str


@dataclass(frozen=True)
class ConstraintEvidence:
    configuration: Mapping[str, Any]
    events: tuple[ViolationWindow, ...]
    recordings: tuple[RecordingSegment, ...]
    rule_status: Mapping[str, str]
    unknown_rules: tuple[str, ...]
    reason_codes_by_type: Mapping[str, tuple[str, ...]]
    global_reason_codes: tuple[str, ...]
    attack_markers: tuple[AttackMarkerPoint, ...]
    attack_reason_codes: tuple[str, ...]

    @property
    def applied_attacks(self) -> tuple[AttackMarkerPoint, ...]:
        """Compatibility view containing only markers with an applied effect."""

        return tuple(marker for marker in self.attack_markers if marker.effect_applied)


def load_constraint_evidence(
    hdf5_path: Union[str, PathLike],
) -> ConstraintEvidence:
    """Read the fixed safety evidence surface without reconstructing time."""

    path = Path(hdf5_path).expanduser().resolve()
    try:
        with h5py.File(path, "r") as hdf5_file:
            if "/research" not in hdf5_file:
                return _missing_evidence()
            research = hdf5_file["/research"]
            if (
                research.attrs.get("schema_name") != "compact-task-safety"
                or int(research.attrs.get("schema_version", -1)) != 1
                or not bool(research.attrs.get("complete", False))
            ):
                return _missing_evidence()
            try:
                configuration = _read_blob(research, "configuration")
                raw_events = _read_rows(research, "safety_events")
                raw_recordings = _read_rows(research, "recording")
                summary = _read_blob(research, "summary")
            except (KeyError, TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError):
                return _missing_evidence()
            try:
                raw_attacks = _read_rows(research, "attack_markers")
            except (
                KeyError,
                TypeError,
                ValueError,
                UnicodeDecodeError,
                json.JSONDecodeError,
            ):
                raw_attacks = None
    except OSError:
        return _missing_evidence()

    issues = defaultdict(set)
    attack_issues = set()
    if raw_attacks is None:
        raw_attacks = ()
        attack_issues.add("MISSING_ATTACK_EVIDENCE")

    attack_markers = []
    for row_index, row in enumerate(raw_attacks):
        if (
            not isinstance(row, Mapping)
            or not isinstance(row.get("effect_applied"), bool)
            or not isinstance(row.get("marker_only"), bool)
        ):
            attack_issues.add("INVALID_ATTACK_EVIDENCE")
            continue
        try:
            attack_markers.append(_parse_attack_marker(row, row_index))
        except (KeyError, TypeError, ValueError):
            attack_issues.add("INVALID_ATTACK_TIMEBASE")

    events = []
    for row_index, row in enumerate(raw_events):
        type_id = row.get("type_id") if isinstance(row, Mapping) else None
        if not isinstance(type_id, str) or not type_id:
            return _missing_evidence()
        try:
            events.append(_parse_event(row, row_index))
        except _InvalidSafetyEvidence:
            issues[type_id].add(INVALID_SAFETY_EVIDENCE)
        except (KeyError, TypeError, ValueError):
            issues[type_id].add(INVALID_OR_MISSING_TIMEBASE)

    recordings = []
    for row in raw_recordings:
        stream_id = row.get("stream_id") if isinstance(row, Mapping) else None
        if not isinstance(stream_id, str) or not stream_id:
            return _missing_evidence()
        try:
            recordings.append(_parse_recording(row))
        except (KeyError, TypeError, ValueError):
            issues[stream_id].add(INVALID_OR_MISSING_TIMEBASE)

    if not isinstance(configuration, Mapping) or not isinstance(summary, Mapping):
        return _missing_evidence()
    raw_status = summary.get("rule_status")
    raw_unknown = summary.get("unknown_rules", ())
    if not isinstance(raw_status, Mapping) or not isinstance(raw_unknown, list):
        return _missing_evidence()
    rule_status = {}
    for type_id, status in raw_status.items():
        if not isinstance(type_id, str) or not isinstance(status, str):
            return _missing_evidence()
        rule_status[type_id] = status
    if any(not isinstance(type_id, str) for type_id in raw_unknown):
        return _missing_evidence()

    normalized_issues = {
        type_id: tuple(sorted(reason_codes))
        for type_id, reason_codes in sorted(issues.items())
    }
    return ConstraintEvidence(
        configuration=MappingProxyType(dict(configuration)),
        events=tuple(events),
        recordings=tuple(recordings),
        rule_status=MappingProxyType(dict(sorted(rule_status.items()))),
        unknown_rules=tuple(sorted(set(raw_unknown))),
        reason_codes_by_type=MappingProxyType(normalized_issues),
        global_reason_codes=(),
        attack_markers=tuple(
            sorted(
                attack_markers,
                key=lambda item: (item.physics_step, item.time_s, item.row_index),
            )
        ),
        attack_reason_codes=tuple(sorted(attack_issues)),
    )


def _missing_evidence() -> ConstraintEvidence:
    return ConstraintEvidence(
        configuration=MappingProxyType({}),
        events=(),
        recordings=(),
        rule_status=MappingProxyType({}),
        unknown_rules=(),
        reason_codes_by_type=MappingProxyType({}),
        global_reason_codes=(MISSING_SAFETY_EVIDENCE,),
        attack_markers=(),
        attack_reason_codes=("MISSING_ATTACK_EVIDENCE",),
    )


def _parse_attack_marker(
    row: Mapping[str, Any], row_index: int
) -> AttackMarkerPoint:
    observation_gap = row.get("first_confirmed_after_observation_gap", False)
    if not isinstance(observation_gap, bool):
        raise TypeError("attack observation-gap flag must be boolean")
    return AttackMarkerPoint(
        attack_id=_text(row["attack_id"]),
        physics_step=_physics_step(row["physics_step"]),
        time_s=_finite_number(row["time_s"]),
        row_index=row_index,
        first_confirmed_after_observation_gap=observation_gap,
        effect_applied=row["effect_applied"],
        marker_only=row["marker_only"],
    )


def _read_rows(research: h5py.Group, name: str):
    group = research[name]
    offsets = group["offsets"][...]
    payload = group["utf8"][...].tobytes()
    values = [int(value) for value in offsets]
    if not values or values[0] != 0 or values[-1] != len(payload):
        raise ValueError("invalid row offsets")
    if any(left > right for left, right in zip(values[:-1], values[1:])):
        raise ValueError("row offsets are not monotonic")
    return [
        json.loads(payload[left:right].decode("utf-8"))
        for left, right in zip(values[:-1], values[1:])
    ]


def _read_blob(research: h5py.Group, name: str):
    raw = research[name][...]
    if isinstance(raw, bytes):
        payload = raw
    elif isinstance(raw, np.ndarray):
        payload = raw.tobytes()
    else:
        payload = bytes(raw)
    return json.loads(payload.decode("utf-8"))


def _parse_event(row: Mapping[str, Any], row_index: int) -> ViolationWindow:
    type_id = _text(row["type_id"])
    criterion = _text(row["criterion"])
    raw_objects = row["objects"]
    if not isinstance(raw_objects, list) or any(
        not isinstance(value, str) or not value for value in raw_objects
    ):
        raise ValueError("invalid event objects")
    start = _point(row["start_physics_step"], row["start_time_s"])
    end = _point(row["end_physics_step"], row["end_time_s"])
    first = _marker(row["first"])
    worst = _marker(row["worst"])
    last_observed = _point(
        row["last_observed_physics_step"], row["last_observed_time_s"]
    )
    end_reason = _text(row["end_reason"])
    if end_reason not in _END_REASONS:
        raise ValueError("invalid end reason")
    marker_pairs = (
        (start, first),
        (first, worst),
        (worst, last_observed),
        (last_observed, end),
    )
    if any(left.physics_step > right.physics_step for left, right in marker_pairs):
        raise ValueError("event physics steps are not ordered")
    if any(left.time_s > right.time_s for left, right in marker_pairs):
        raise ValueError("event timestamps are not ordered")
    threshold = row.get("threshold")
    _ensure_json_safe(threshold)
    return ViolationWindow(
        type_id=type_id,
        criterion=criterion,
        objects=tuple(raw_objects),
        start=start,
        end=end,
        end_reason=end_reason,
        first=first,
        worst=worst,
        last_observed=last_observed,
        threshold=threshold,
        unit=_text(row.get("unit", ""), allow_empty=True),
        comparison=_text(row.get("comparison", ""), allow_empty=True),
        row_index=row_index,
    )


def _parse_recording(row: Mapping[str, Any]) -> RecordingSegment:
    stream_id = _text(row["stream_id"])
    fully_observed = row["fully_observed"]
    if not isinstance(fully_observed, bool):
        raise TypeError("fully_observed must be boolean")
    start = _point(row["start_physics_step"], row["start_time_s"])
    last = _point(row["last_physics_step"], row["last_observed_time_s"])
    end_time = _finite_number(row["end_time_s"])
    if start.physics_step > last.physics_step or start.time_s > last.time_s:
        raise ValueError("recording interval is not ordered")
    if last.time_s > end_time:
        raise ValueError("recording end time precedes last observation")
    error_reason = row.get("error_reason", "")
    if not isinstance(error_reason, str):
        raise TypeError("recording error_reason must be text")
    return RecordingSegment(stream_id, fully_observed, error_reason)


def _marker(value: object) -> EvidencePoint:
    if not isinstance(value, Mapping):
        raise TypeError("event marker must be an object")
    marker_value = value.get("value")
    _ensure_json_safe(marker_value)
    return EvidencePoint(
        physics_step=_physics_step(value["physics_step"]),
        time_s=_finite_number(value["time_s"]),
        value=marker_value,
    )


def _point(step: object, time_s: object) -> EvidencePoint:
    return EvidencePoint(
        physics_step=_physics_step(step),
        time_s=_finite_number(time_s),
    )


def _physics_step(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("physics step must be a non-negative integer")
    return value


def _finite_number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("timestamp must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0:
        raise ValueError("timestamp must be finite and non-negative")
    return normalized


def _text(value: object, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        raise ValueError("value must be text")
    return value


def _ensure_json_safe(value: object) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise _InvalidSafetyEvidence("non-finite JSON number")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise _InvalidSafetyEvidence("JSON object key must be text")
            _ensure_json_safe(item)
        return
    if isinstance(value, list):
        for item in value:
            _ensure_json_safe(item)
        return
    raise _InvalidSafetyEvidence("unsupported JSON evidence value")


__all__ = [
    "ConstraintEvidence",
    "RecordingSegment",
    "load_constraint_evidence",
]
