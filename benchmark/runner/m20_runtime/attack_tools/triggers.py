"""Domain triggers for M20 distance, actions, interactions, and legacy fields."""
from __future__ import annotations
import math
from typing import Any, Mapping, Sequence

def lookup(data: Mapping[str, Any], dotted: str) -> Any:
    value: Any = data
    for part in dotted.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return value

def compare(value: Any, op: str, target: Any = None) -> bool:
    operations = {
        "eq": lambda: value == target, "ne": lambda: value != target,
        "gt": lambda: value is not None and value > target,
        "gte": lambda: value is not None and value >= target,
        "lt": lambda: value is not None and value < target,
        "lte": lambda: value is not None and value <= target,
        "truthy": lambda: bool(value), "falsy": lambda: not bool(value),
        "in": lambda: value in target if target is not None else False,
        "not_in": lambda: value not in target if target is not None else True,
    }
    if op not in operations:
        raise ValueError(f"unsupported trigger operator: {op!r}")
    return bool(operations[op]())

class TriggerEvaluator:
    kinds = frozenset({
        "field",
        "distance_to_object",
        "distance_to_any_object",
        "action_performed",
        "interaction_event",
    })

    def matches(self, condition: Mapping[str, Any], snapshot: Mapping[str, Any]) -> bool:
        kind = str(condition.get("type", "field"))
        if kind not in self.kinds:
            raise ValueError(f"unsupported trigger type: {kind!r}")
        if kind == "field":
            return compare(lookup(snapshot, str(condition["field"])),
                           str(condition.get("op", "eq")), condition.get("value"))
        if kind in {"distance_to_object", "distance_to_any_object"}:
            value = (
                self.distance(condition, snapshot)
                if kind == "distance_to_object"
                else self.distance_to_any(condition, snapshot)
            )
            target = condition.get("distance_m", condition.get("value"))
            if target is None:
                raise ValueError(f"{kind} requires distance_m")
            return compare(value, str(condition.get("op", "lte")), float(target))
        if kind == "action_performed":
            matched = self.action(condition, snapshot)
        else:
            matched = self.interaction(condition, snapshot)
        return compare(matched, str(condition.get("op", "truthy")), condition.get("value"))

    @staticmethod
    def _has_path(data: Mapping[str, Any], dotted: str) -> bool:
        value: Any = data
        for part in dotted.split("."):
            if not isinstance(value, Mapping) or part not in value:
                return False
            value = value[part]
        return True

    def observable(self, condition: Mapping[str, Any], snapshot: Mapping[str, Any]) -> bool:
        """Return whether this snapshot contains the facts needed by a trigger.

        This is deliberately separate from ``matches``.  The handoff field
        contract distinguishes a known-false trigger from an observation gap;
        treating both as ``False`` would make
        ``first_confirmed_after_observation_gap`` impossible to establish.
        """
        kind = str(condition.get("type", "field"))
        if kind not in self.kinds:
            raise ValueError(f"unsupported trigger type: {kind!r}")
        if kind == "field":
            return self._has_path(snapshot, str(condition["field"]))
        if kind == "distance_to_object":
            object_id = str(condition["object_id"])
            metric = str(condition.get("metric", "prim_origin"))
            if metric == "world_aabb_xy":
                return self._has_path(
                    snapshot, f"boundary_distances.{object_id}"
                )
            return (
                self._has_path(snapshot, f"distances.{object_id}")
                or (
                    self._has_path(snapshot, "robot.position_xyz")
                    and self._has_path(snapshot, f"objects.{object_id}.position_xyz")
                )
            )
        if kind == "distance_to_any_object":
            object_ids = condition.get("object_ids")
            if (
                not isinstance(object_ids, Sequence)
                or isinstance(object_ids, (str, bytes))
                or not object_ids
            ):
                raise ValueError("distance_to_any_object requires non-empty object_ids")
            return any(
                self.observable(
                    {**dict(condition), "type": "distance_to_object", "object_id": str(object_id)},
                    snapshot,
                )
                for object_id in object_ids
            )
        if kind == "action_performed":
            return "action" in snapshot or "actions" in snapshot
        return self._has_path(snapshot, "interaction.events") or "events" in snapshot

    def all_match_observed(
        self,
        conditions: Sequence[Mapping[str, Any]],
        snapshot: Mapping[str, Any],
    ) -> bool | None:
        """Three-valued AND: true, false, or unknown due to missing evidence."""
        if not conditions:
            return False
        unknown = False
        for condition in conditions:
            if not self.observable(condition, snapshot):
                unknown = True
                continue
            if not self.matches(condition, snapshot):
                return False
        return None if unknown else True

    def distance(self, condition: Mapping[str, Any], snapshot: Mapping[str, Any]) -> float | None:
        object_id = str(condition["object_id"])
        metric = str(condition.get("metric", "prim_origin"))
        if metric == "world_aabb_xy":
            explicit = lookup(snapshot, f"boundary_distances.{object_id}")
            return None if explicit is None else float(explicit)
        if metric not in {"prim_origin", "prim_origin_xy"}:
            raise ValueError(f"unsupported distance metric: {metric!r}")
        explicit = lookup(snapshot, f"distances.{object_id}")
        if explicit is not None:
            return float(explicit)
        robot = lookup(snapshot, "robot.position_xyz")
        target = lookup(snapshot, f"objects.{object_id}.position_xyz")
        if robot is None or target is None:
            return None
        robot_xyz, target_xyz = [float(x) for x in robot[:3]], [float(x) for x in target[:3]]
        if not all(math.isfinite(x) for x in robot_xyz + target_xyz):
            raise ValueError("distance trigger coordinates must be finite")
        axes = "xy" if metric == "prim_origin_xy" else condition.get("axes", "xyz")
        indices = (0, 1) if axes == "xy" else (0, 1, 2)
        return math.sqrt(sum((robot_xyz[i] - target_xyz[i]) ** 2 for i in indices))

    def distance_to_any(
        self,
        condition: Mapping[str, Any],
        snapshot: Mapping[str, Any],
    ) -> float | None:
        object_ids = condition.get("object_ids")
        if (
            not isinstance(object_ids, Sequence)
            or isinstance(object_ids, (str, bytes))
            or not object_ids
        ):
            raise ValueError("distance_to_any_object requires non-empty object_ids")
        distances = []
        for object_id in object_ids:
            item = dict(condition)
            item["object_id"] = str(object_id)
            distance = self.distance(item, snapshot)
            if distance is not None:
                distances.append(distance)
        return min(distances) if distances else None

    @staticmethod
    def action(condition: Mapping[str, Any], snapshot: Mapping[str, Any]) -> bool:
        expected, names = str(condition["action"]), []
        candidates = [snapshot.get("action", {}), *snapshot.get("actions", [])]
        for item in candidates:
            if isinstance(item, Mapping):
                names.extend(str(item[k]) for k in ("name", "type", "id") if item.get(k) is not None)
            elif item:
                names.append(str(item))
        return expected in names

    @staticmethod
    def interaction(condition: Mapping[str, Any], snapshot: Mapping[str, Any]) -> bool:
        expected, object_id = str(condition["event"]), condition.get("object_id")
        events = lookup(snapshot, "interaction.events") or snapshot.get("events", [])
        if not isinstance(events, Sequence) or isinstance(events, (str, bytes)):
            return False
        for item in events:
            if isinstance(item, Mapping):
                name = item.get("type", item.get("event", item.get("name")))
                if str(name) == expected and (object_id is None or str(item.get("object_id")) == str(object_id)):
                    return True
            elif str(item) == expected and object_id is None:
                return True
        return False

    def all_match(self, conditions: Sequence[Mapping[str, Any]], snapshot: Mapping[str, Any]) -> bool:
        return bool(conditions) and all(self.matches(item, snapshot) for item in conditions)
