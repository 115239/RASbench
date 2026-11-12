"""Config-driven termination registry for task-independent episode control."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence


TerminationHandler = Callable[[Mapping[str, Any], Mapping[str, Any]], bool]


def _lookup(data: Mapping[str, Any], dotted: str) -> Any:
    value: Any = data
    for part in dotted.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return value


def _distance(a: Sequence[float], b: Sequence[float], axes: str = "xyz") -> float:
    indices = (0, 1) if axes == "xy" else (0, 1, 2)
    return math.sqrt(sum((float(a[i]) - float(b[i])) ** 2 for i in indices))


class TerminationRegistry:
    """Maps stable JSON evaluator names to audited Python implementations."""

    def __init__(self) -> None:
        self._handlers: dict[str, TerminationHandler] = {}

    def register(self, name: str, handler: TerminationHandler) -> None:
        normalized = str(name).strip()
        if not normalized or "." in normalized or "/" in normalized or normalized in self._handlers:
            raise ValueError(f"invalid or duplicate termination evaluator: {name!r}")
        self._handlers[normalized] = handler

    def evaluate(self, spec: Mapping[str, Any], snapshot: Mapping[str, Any]) -> bool:
        name = str(spec.get("evaluator", "")).strip()
        if name not in self._handlers:
            raise KeyError(f"unregistered termination evaluator: {name!r}")
        return bool(self._handlers[name](snapshot, dict(spec.get("params", {}))))

    def describe(self) -> list[str]:
        return sorted(self._handlers)


@dataclass(frozen=True)
class TerminationDecision:
    status: str
    reason: str | None
    checks: dict[str, bool]


class TerminationEvaluator:
    """Evaluates success/failure groups without task-specific runner branches."""

    def __init__(self, config: Mapping[str, Any], registry: TerminationRegistry) -> None:
        self.config = dict(config)
        self.registry = registry
        self._stable_counts: dict[str, int] = {}
        self._stable_since_s: dict[str, float] = {}

    def evaluate(self, snapshot: Mapping[str, Any]) -> TerminationDecision:
        checks: dict[str, bool] = {}
        for outcome in ("failure", "success"):
            group = dict(self.config.get(outcome, {}))
            conditions = list(group.get("conditions", []))
            if not conditions:
                continue
            values = []
            for index, condition in enumerate(conditions):
                key = str(condition.get("id", f"{outcome}_{index}"))
                raw_value = self.registry.evaluate(condition, snapshot)
                if "stable_duration_s" in condition:
                    duration_s = max(0.0, float(condition["stable_duration_s"]))
                    elapsed_s = _lookup(snapshot, "episode.elapsed_s")
                    if elapsed_s is None or not math.isfinite(float(elapsed_s)):
                        raise ValueError(
                            "stable_duration_s termination requires finite "
                            "episode.elapsed_s"
                        )
                    now_s = float(elapsed_s)
                    if raw_value:
                        since_s = self._stable_since_s.setdefault(key, now_s)
                        value = now_s - since_s >= duration_s
                    else:
                        self._stable_since_s.pop(key, None)
                        value = False
                else:
                    required_steps = max(1, int(condition.get("stable_steps", 1)))
                    self._stable_counts[key] = (
                        self._stable_counts.get(key, 0) + 1 if raw_value else 0
                    )
                    value = self._stable_counts[key] >= required_steps
                checks[key] = value
                values.append(value)
            mode = str(group.get("mode", "any" if outcome == "failure" else "all"))
            matched = all(values) if mode == "all" else any(values)
            if matched:
                return TerminationDecision(outcome, str(group.get("reason", outcome)), checks)
        return TerminationDecision("running", None, checks)


def build_standard_termination_registry() -> TerminationRegistry:
    registry = TerminationRegistry()
    registry.register(
        "route_completed",
        lambda s, p: int(_lookup(s, "route.reached_count") or 0)
        >= int(_lookup(s, "route.total_count") or 0) > 0,
    )
    registry.register(
        "robot_near_target",
        lambda s, p: _distance(
            _lookup(s, "robot.position_xyz"), p["target_xyz"], str(p.get("axes", "xy"))
        ) <= float(p["distance_m"]),
    )
    registry.register(
        "object_near_target",
        lambda s, p: _distance(
            _lookup(s, f"objects.{p['object_id']}.position_xyz"),
            _lookup(s, f"objects.{p['target_id']}.position_xyz")
            if p.get("target_id") else p["target_xyz"],
            str(p.get("axes", "xy")),
        ) <= float(p["distance_m"]),
    )
    registry.register(
        "object_lifted",
        lambda s, p: float(_lookup(s, f"metrics.object_lift_m.{p['object_id']}") or 0.0)
        >= float(p["height_m"]),
    )
    registry.register(
        "object_displaced",
        lambda s, p: _distance(
            _lookup(s, f"objects.{p['object_id']}.position_xyz"),
            _lookup(s, f"objects.{p['object_id']}.initial_position_xyz"),
            str(p.get("axes", "xy")),
        ) >= float(p["distance_m"]),
    )
    registry.register(
        "object_axis_displacement",
        lambda s, p: (
            float(_lookup(s, f"objects.{p['object_id']}.position_xyz")[int(p["axis"])])
            - float(_lookup(s, f"objects.{p['object_id']}.initial_position_xyz")[int(p["axis"])])
        ) * float(p.get("direction", 1.0)) >= float(p["distance_m"]),
    )
    registry.register(
        "grasp_confirmed",
        lambda s, p: bool(_lookup(s, "interaction.grasp_confirmed"))
        and (not p.get("object_id") or _lookup(s, "interaction.grasped_object_id") == p["object_id"]),
    )
    registry.register(
        "interaction_event",
        lambda s, p: any(
            str(event.get("type", event.get("event", ""))) == str(p["event"])
            and (not p.get("object_id") or str(event.get("object_id")) == str(p["object_id"]))
            and (not p.get("target_id") or str(event.get("target_id")) == str(p["target_id"]))
            for event in (_lookup(s, "interaction.events") or [])
            if isinstance(event, Mapping)
        ),
    )
    registry.register(
        "robot_tilt_exceeded",
        lambda s, p: max(abs(float(x)) for x in (_lookup(s, "robot.rpy") or [0.0, 0.0])[:2])
        > float(p["radians"]),
    )
    registry.register("robot_fallen", _robot_fallen)
    registry.register(
        "robot_height_below",
        lambda s, p: float((_lookup(s, "robot.position_xyz") or [0.0, 0.0, math.inf])[2])
        < float(p["height_m"]),
    )
    registry.register(
        "maximum_actions",
        lambda s, p: int(_lookup(s, "episode.executed_actions") or 0) >= int(p["count"]),
    )
    registry.register(
        "maximum_duration",
        lambda s, p: float(_lookup(s, "episode.elapsed_s") or 0.0) >= float(p["seconds"]),
    )
    registry.register(
        "field_compare",
        lambda s, p: _field_compare(_lookup(s, str(p["field"])), str(p.get("op", "eq")), p.get("value")),
    )
    return registry


def _robot_fallen(snapshot: Mapping[str, Any], params: Mapping[str, Any]) -> bool:
    """Confirm a physical fall instead of treating a transient tilt as failure."""

    rpy = _lookup(snapshot, "robot.rpy")
    position = _lookup(snapshot, "robot.position_xyz")
    initial_position = _lookup(snapshot, "robot.initial_position_xyz")
    if rpy is None or position is None or initial_position is None:
        return False
    try:
        roll = float(rpy[0])
        pitch = float(rpy[1])
        current_height = float(position[2])
        initial_height = float(initial_position[2])
    except (IndexError, TypeError, ValueError):
        return False
    values = (roll, pitch, current_height, initial_height)
    if not all(math.isfinite(value) for value in values):
        return False

    # Dot(body-up, world-up) = cos(roll) * cos(pitch).  This avoids Euler-axis
    # bias while remaining compatible with the physical RPY snapshot.
    up_dot = max(-1.0, min(1.0, math.cos(roll) * math.cos(pitch)))
    tilt_radians = math.acos(up_dot)
    height_drop_m = initial_height - current_height
    fallen_tilt = float(params.get("tilt_radians", math.radians(70.0)))
    hard_tilt = float(params.get("hard_tilt_radians", math.radians(85.0)))
    minimum_drop = float(params.get("minimum_height_drop_m", 0.12))
    return tilt_radians >= hard_tilt or (
        tilt_radians >= fallen_tilt and height_drop_m >= minimum_drop
    )


def _field_compare(value: Any, op: str, target: Any) -> bool:
    operations = {
        "eq": lambda: value == target,
        "ne": lambda: value != target,
        "gt": lambda: value is not None and value > target,
        "gte": lambda: value is not None and value >= target,
        "lt": lambda: value is not None and value < target,
        "lte": lambda: value is not None and value <= target,
        "truthy": lambda: bool(value),
        "falsy": lambda: not bool(value),
    }
    if op not in operations:
        raise ValueError(f"unsupported termination operator: {op!r}")
    return bool(operations[op]())
