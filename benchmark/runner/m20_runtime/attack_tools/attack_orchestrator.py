"""Config-driven attack scheduling for M20 simulation experiments."""
from __future__ import annotations
import copy
import hashlib
import json
import math
from dataclasses import dataclass, field
from numbers import Integral
from typing import Any, Callable, Mapping

from .attack_registry import AttackRegistry
from .audit import summarize
from .triggers import TriggerEvaluator

AttackHandler = Callable[[Any, Mapping[str, Any], Mapping[str, Any]], Any]

@dataclass
class AttackSpec:
    attack_id: str
    mode: str
    stage: str
    handler: str
    trigger: list[dict[str, Any]]
    active_while: list[dict[str, Any]] | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    effect_enabled: bool = True
    once: bool = True
    priority: int = 100
    max_applications: int | None = None
    duration_s: float | None = None
    release_delay_s: float | None = None

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AttackSpec":
        trigger = value.get("trigger", {})
        spec = cls(
            attack_id=str(value["id"]), mode=str(value["mode"]), stage=str(value["stage"]),
            handler=str(value["handler"]), trigger=list(trigger.get("all", [])),
            active_while=(
                None
                if value.get("active_while") is None
                else list(value["active_while"].get("all", []))
            ),
            parameters=dict(value.get("parameters", {})), enabled=bool(value.get("enabled", True)),
            effect_enabled=bool(value.get("effect_enabled", True)),
            once=bool(value.get("once", True)), priority=int(value.get("priority", 100)),
            max_applications=None if value.get("max_applications") is None else int(value["max_applications"]),
            duration_s=None if value.get("duration_s") is None else float(value["duration_s"]),
            release_delay_s=None if value.get("release_delay_s") is None else float(value["release_delay_s"]),
        )
        if spec.mode not in {"next_vla", "immediate", "intercept"}:
            raise ValueError(f"invalid mode for {spec.attack_id}: {spec.mode!r}")
        valid_stages = {
            "request_state", "request_image", "request_prompt", "response_action",
            "state_feedback", "sensor_lidar", "sensor_imu", "sensor_force_torque",
            "sensor_wheel_encoder", "sensor_joint_encoder",
            "control_base", "control_joint", "control_gripper", "control_ee_target",
            "immediate_robot_control", "immediate_gripper", "immediate_lidar", "immediate_imu",
            "immediate_visual_material",
        }
        if spec.stage not in valid_stages:
            raise ValueError(f"invalid stage for {spec.attack_id}: {spec.stage!r}")
        if spec.mode == "next_vla" and spec.stage not in {
            "request_state", "request_image", "request_prompt", "response_action"
        }:
            raise ValueError(f"next_vla stage mismatch for {spec.attack_id}")
        if spec.mode == "immediate" and not spec.stage.startswith("immediate_"):
            raise ValueError(f"immediate stage mismatch for {spec.attack_id}")
        if spec.mode == "intercept" and spec.stage.startswith("immediate_"):
            raise ValueError(f"intercept stage mismatch for {spec.attack_id}")
        if spec.max_applications is not None and spec.max_applications <= 0:
            raise ValueError("max_applications must be positive")
        if spec.duration_s is not None and spec.duration_s <= 0:
            raise ValueError("duration_s must be positive")
        if spec.release_delay_s is not None and spec.release_delay_s < 0:
            raise ValueError("release_delay_s must be non-negative")
        return spec

class AttackOrchestrator:
    """Evaluate domain triggers and dispatch registered attacks at typed stages."""

    def __init__(self, config: Mapping[str, Any],
                 handlers: Mapping[str, AttackHandler] | AttackRegistry, *,
                 audit: Callable[[dict[str, Any]], None] | None = None) -> None:
        self.registry = handlers if isinstance(handlers, AttackRegistry) else None
        self.handlers = handlers.handlers() if self.registry else dict(handlers)
        self.audit = audit or (lambda record: None)
        self.evaluator = TriggerEvaluator()
        self.specs = sorted((AttackSpec.from_dict(x) for x in config.get("attacks", [])),
                            key=lambda x: x.priority)
        missing = sorted({x.handler for x in self.specs} - self.handlers.keys())
        if missing:
            raise ValueError(f"unregistered attack handlers: {missing}")
        if self.registry:
            for spec in self.specs:
                self.registry.resolve(spec.handler, spec.stage)
        self.triggered: set[str] = set()
        self.armed: list[tuple[AttackSpec, dict[str, Any]]] = []
        self.response_pipeline: list[tuple[AttackSpec, dict[str, Any]]] = []
        self.active_intercepts: list[tuple[AttackSpec, dict[str, Any]]] = []
        self.application_counts: dict[str, int] = {}
        self.completed: set[str] = set()
        self.latest_time_s = 0.0
        self.stop_reasons: dict[str, str] = {}
        self.condition_counts: dict[str, int] = {}
        self.condition_instances: dict[str, str] = {}
        self.condition_clocks: dict[str, tuple[float, int]] = {}
        self.condition_observation_gaps: set[str] = set()
        self.effect_instances: set[str] = set()
        self.last_condition_markers: list[str] = []

    @staticmethod
    def _snapshot_time(snapshot: Mapping[str, Any]) -> float:
        simulation = snapshot.get("simulation", {})
        return float(simulation.get("time_s", 0.0)) if isinstance(simulation, Mapping) else 0.0

    @staticmethod
    def _snapshot_clock(snapshot: Mapping[str, Any]) -> tuple[float, int] | None:
        simulation = snapshot.get("simulation", {})
        if not isinstance(simulation, Mapping):
            return None
        # Marker clocks share the continuous physical evidence axis. The formal
        # task-relative clock and its explicit origin remain in the snapshot.
        time_value = simulation.get("time_s")
        step_value = simulation.get("physics_step")
        if (
            isinstance(time_value, bool)
            or not isinstance(time_value, (int, float))
            or not math.isfinite(float(time_value))
            or float(time_value) < 0.0
            or isinstance(step_value, bool)
            or not isinstance(step_value, Integral)
            or step_value < 0
        ):
            return None
        return float(time_value), int(step_value)

    @staticmethod
    def _payload_digest(value: Any) -> str:
        payload = json.dumps(
            summarize(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _condition_marker(
        self,
        spec: AttackSpec,
        snapshot: Mapping[str, Any],
        *,
        observation_gap: bool,
    ) -> str:
        clock = self._snapshot_clock(snapshot)
        if clock is None:
            raise ValueError(
                f"{spec.attack_id}: trigger confirmed without task time/physics step"
            )
        count = self.condition_counts.get(spec.attack_id, 0) + 1
        self.condition_counts[spec.attack_id] = count
        instance = f"{spec.attack_id}:{count}"
        self.condition_instances[spec.attack_id] = instance
        self.condition_clocks[instance] = clock
        evidence = {
            "event": "condition_triggered",
            "mode": spec.mode,
            "stage": spec.stage,
            "configured_trigger": {"all": copy.deepcopy(spec.trigger)},
            "observed_snapshot": copy.deepcopy(dict(snapshot)),
        }
        self.audit({
            "event": "condition_triggered",
            "attack_id": spec.attack_id,
            "attack_instance_id": instance,
            "time_s": clock[0],
            "physics_step": clock[1],
            "marker_only": True,
            "effect_applied": False,
            "first_confirmed_after_observation_gap": observation_gap,
            "evidence": evidence,
        })
        return instance

    def _effect_marker(
        self,
        spec: AttackSpec,
        before: Any,
        after: Any,
        context: Mapping[str, Any] | None,
    ) -> None:
        instance = self.condition_instances.get(spec.attack_id)
        if instance is None or instance in self.effect_instances:
            return
        before_digest = self._payload_digest(before)
        after_digest = self._payload_digest(after)
        if before_digest == after_digest:
            return
        runtime = context or {}
        time_value = runtime.get("time_s")
        step_value = runtime.get("physics_step")
        if (
            isinstance(time_value, bool)
            or not isinstance(time_value, (int, float))
            or not math.isfinite(float(time_value))
            or isinstance(step_value, bool)
            or not isinstance(step_value, Integral)
        ):
            time_value, step_value = self.condition_clocks[instance]
        self.audit({
            "event": "effect_first_applied",
            "attack_id": spec.attack_id,
            "attack_instance_id": instance,
            "time_s": float(time_value),
            "physics_step": int(step_value),
            "marker_only": False,
            "effect_applied": True,
            "first_confirmed_after_observation_gap": False,
            "evidence": {
                "event": "effect_first_applied",
                "stage": spec.stage,
                "executor": spec.handler,
                "target_before_sha256": before_digest,
                "target_after_sha256": after_digest,
            },
        })
        self.effect_instances.add(instance)

    def _boundary_time(self, context: Mapping[str, Any] | None) -> float:
        boundary_context = context or {}
        value = boundary_context.get("time_s")
        if not isinstance(value, (int, float)):
            runtime = boundary_context.get("runtime", {})
            value = runtime.get("time_s") if isinstance(runtime, Mapping) else None
        return float(value) if isinstance(value, (int, float)) else self.latest_time_s

    def observe(self, snapshot: Mapping[str, Any], immediate_target: Any = None) -> list[str]:
        activated: list[str] = []
        self.last_condition_markers = []
        if snapshot.get('simulation', {}).get('lifecycle_phase') not in (None, 'TASK_RUNNING'):
            return activated
        frozen = copy.deepcopy(dict(snapshot))
        now = self._snapshot_time(snapshot)
        self.latest_time_s = now
        retained_active = []
        for active_spec, active_context in self.active_intercepts:
            retention_conditions = (
                active_spec.trigger
                if active_spec.active_while is None
                else active_spec.active_while
            )
            inside = self.evaluator.all_match(retention_conditions, snapshot)
            runtime = active_context.setdefault("_attack_runtime", {})
            if inside:
                runtime["last_inside_s"] = now
            elapsed = now - float(runtime.get("activated_s", now))
            outside_elapsed = now - float(runtime.get("last_inside_s", now))
            reason = None
            if active_spec.duration_s is not None and elapsed >= active_spec.duration_s:
                reason = "duration_reached"
            elif not inside and active_spec.release_delay_s is not None and outside_elapsed >= active_spec.release_delay_s:
                reason = "left_trigger_region"
            if reason:
                self.completed.add(active_spec.attack_id); self.stop_reasons[active_spec.attack_id] = reason
                self.audit({"event":"intercept_deactivated","attack_id":active_spec.attack_id,
                            "stage":active_spec.stage,"active":False,"activated_at":runtime.get("activated_s"),
                            "last_trigger_true_at":runtime.get("last_inside_s"),
                            "application_count":self.application_counts.get(active_spec.attack_id,0),
                            "max_applications":active_spec.max_applications,"duration_s":active_spec.duration_s,
                            "release_delay_s":active_spec.release_delay_s,"stop_reason":reason,
                            "elapsed_attack_s":elapsed})
            else:
                retained_active.append((active_spec, active_context))
        self.active_intercepts = retained_active
        for spec in self.specs:
            if (
                not spec.enabled
                or spec.attack_id in self.completed
                or (spec.once and spec.attack_id in self.triggered)
                or (
                    not spec.effect_enabled
                    and spec.attack_id in self.condition_instances
                )
            ):
                continue
            if spec.mode == "intercept" and any(x.attack_id == spec.attack_id for x, _ in self.active_intercepts):
                continue
            condition = self.evaluator.all_match_observed(spec.trigger, snapshot)
            if condition is None:
                self.condition_observation_gaps.add(spec.attack_id)
                continue
            if not condition:
                continue
            instance = self._condition_marker(
                spec,
                frozen,
                observation_gap=spec.attack_id in self.condition_observation_gaps,
            )
            self.triggered.add(spec.attack_id)
            self.last_condition_markers.append(spec.attack_id)
            if not spec.effect_enabled:
                self.audit({
                    "event": "condition_recorded_without_effect",
                    "attack_id": spec.attack_id,
                    "attack_instance_id": instance,
                    "stage": spec.stage,
                    "effect_enabled": False,
                })
                continue
            activated.append(spec.attack_id)
            if spec.mode == "immediate":
                target = immediate_target.get(spec.stage) if isinstance(immediate_target, Mapping) else immediate_target
                if target is None:
                    raise RuntimeError(f"{spec.attack_id}: no target supplied for {spec.stage}")
                self._call(spec, target, frozen)
                event = "immediate_executed"
            elif spec.mode == "intercept":
                frozen["_attack_runtime"] = {"active":True,"activated_s":now,"last_inside_s":now,
                                               "application_count":0,"stop_reason":None,
                                               "attack_instance_id":instance}
                self.active_intercepts.append((spec, frozen))
                event = "intercept_activated"
            else:
                self.armed.append((spec, frozen))
                event = "next_vla_armed"
            self.audit({"event": event, "attack_id": spec.attack_id, "stage": spec.stage,
                        "attack_instance_id": instance,
                        "active":spec.mode=="intercept","activated_at":now,"last_trigger_true_at":now,
                        "application_count":0,"max_applications":spec.max_applications,
                        "duration_s":spec.duration_s,"release_delay_s":spec.release_delay_s,"stop_reason":None,
                        "trigger_snapshot": frozen})
        return activated

    def observe_markers_only(self, snapshot: Mapping[str, Any]) -> list[str]:
        """Evaluate configured triggers without arming or applying any attack."""
        marked: list[str] = []
        self.last_condition_markers = []
        if snapshot.get('simulation', {}).get('lifecycle_phase') not in (None, 'TASK_RUNNING'):
            return marked
        frozen = copy.deepcopy(dict(snapshot))
        self.latest_time_s = self._snapshot_time(snapshot)
        for spec in self.specs:
            # Marker-only mode records the first confirmed condition for each
            # configured attack class in the episode, matching the handoff
            # contract even when an executable profile uses ``once=false``.
            if not spec.enabled or spec.attack_id in self.condition_instances:
                continue
            condition = self.evaluator.all_match_observed(spec.trigger, snapshot)
            if condition is None or self._snapshot_clock(snapshot) is None:
                self.condition_observation_gaps.add(spec.attack_id)
                continue
            if not condition:
                continue
            self._condition_marker(
                spec,
                frozen,
                observation_gap=spec.attack_id in self.condition_observation_gaps,
            )
            self.triggered.add(spec.attack_id)
            marked.append(spec.attack_id)
            self.last_condition_markers.append(spec.attack_id)
        return marked

    def process_intercept(self, stage: str, payload: Any,
                          context: Mapping[str, Any] | None = None) -> Any:
        result = copy.deepcopy(payload)
        retained: list[tuple[AttackSpec, dict[str, Any]]] = []
        for spec, trigger_snapshot in self.active_intercepts:
            if spec.stage != stage:
                retained.append((spec, trigger_snapshot))
                continue
            sequence_round = None
            if spec.handler == 'replace_prompt_sequence' and spec.parameters.get('delivery') == 'once_per_scheduled_request':
                count = self.application_counts.get(spec.attack_id, 0)
                steps = spec.parameters['steps']
                if count >= len(steps):
                    continue
                origin = trigger_snapshot['simulation']['task_start_sim_time_s']
                task_time = self._boundary_time(context) - origin
                due = float(steps[count]['at_s'])
                if task_time < due - 1e-7:
                    retained.append((spec, trigger_snapshot))
                    continue
                if task_time > due + 1e-7:
                    self.audit({'event': 'prompt_round_not_delivered', 'attack_id': spec.attack_id,
                                'round_index': count, 'scheduled_task_time_s': due,
                                'actual_task_time_s': task_time, 'reason': 'NO_REQUEST_AT_SCHEDULED_TIME'})
                    self.completed.add(spec.attack_id)
                    self.stop_reasons[spec.attack_id] = 'prompt_schedule_incomplete'
                    continue
                sequence_round = count
                context = dict(context or {}, prompt_round_index=count, task_time_s=task_time)
            before = copy.deepcopy(result)
            result = self._call(spec, result, trigger_snapshot, context)
            self._effect_marker(spec, before, result, context)
            count = self.application_counts.get(spec.attack_id, 0) + 1
            self.application_counts[spec.attack_id] = count
            trigger_snapshot.setdefault("_attack_runtime", {})["application_count"] = count
            self.audit({"event": "payload_intercepted", "attack_id": spec.attack_id,
                        "attack_instance_id": self.condition_instances.get(spec.attack_id),
                        "stage": stage, "application": count, "before": before, "after": result,
                        "prompt_round_index": sequence_round,
                        "runtime_context": dict(context or {}),
                        "elapsed_attack_s":max(0.0, self._boundary_time(context)-float(trigger_snapshot.get("_attack_runtime",{}).get("activated_s",self.latest_time_s)))})
            if spec.max_applications is None or count < spec.max_applications:
                retained.append((spec, trigger_snapshot))
            else:
                reason="application_limit_reached"; self.completed.add(spec.attack_id); self.stop_reasons[spec.attack_id]=reason
                self.audit({"event":"intercept_deactivated","attack_id":spec.attack_id,"stage":stage,
                            "active":False,"application_count":count,"max_applications":spec.max_applications,
                            "duration_s":spec.duration_s,"release_delay_s":spec.release_delay_s,
                            "stop_reason":reason,"elapsed_attack_s":max(0.0, self._boundary_time(context)-float(trigger_snapshot.get("_attack_runtime",{}).get("activated_s",self.latest_time_s)))})
        self.active_intercepts = retained
        return result

    def process_vla_request(
        self,
        observation: Mapping[str, Any],
        context: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        request = copy.deepcopy(dict(observation))
        runtime_context = dict(context or {})
        armed, self.armed, self.response_pipeline = self.armed, [], []
        for spec, snapshot in armed:
            if spec.stage == "response_action":
                self.response_pipeline.append((spec, snapshot))
                continue
            key = str(spec.parameters.get("key", self._default_key(spec.stage)))
            if key not in request:
                raise KeyError(f"{spec.attack_id}: request key {key!r} not found")
            before = copy.deepcopy(request[key])
            request[key] = self._call(spec, request[key], snapshot)
            self._effect_marker(spec, before, request[key], runtime_context)
            self.audit({"event": "request_modified", "attack_id": spec.attack_id,
                        "attack_instance_id": self.condition_instances.get(spec.attack_id),
                        "stage": spec.stage, "key": key, "before": before, "after": request[key],
                        "runtime_context": runtime_context})
        for stage, key in (("request_image", "observation/image"),
                           ("request_state", "observation/state"), ("request_prompt", "prompt")):
            if key in request:
                stage_context = dict(runtime_context)
                stage_context["key"] = key
                request[key] = self.process_intercept(stage, request[key], stage_context)
        return request

    def process_vla_response(
        self,
        response: Mapping[str, Any],
        context: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        result = copy.deepcopy(dict(response))
        runtime_context = dict(context or {})
        pipeline, self.response_pipeline = self.response_pipeline, []
        for spec, snapshot in pipeline:
            key = str(spec.parameters.get("key", self._default_key(spec.stage)))
            if key not in result:
                raise KeyError(f"{spec.attack_id}: response key {key!r} not found")
            before = copy.deepcopy(result[key])
            result[key] = self._call(spec, result[key], snapshot)
            self._effect_marker(spec, before, result[key], runtime_context)
            self.audit({"event": "response_modified", "attack_id": spec.attack_id,
                        "attack_instance_id": self.condition_instances.get(spec.attack_id),
                        "stage": spec.stage, "key": key, "before": before, "after": result[key],
                        "runtime_context": runtime_context})
        if "actions" in result:
            result["actions"] = self.process_intercept(
                "response_action", result["actions"], runtime_context
            )
        return result

    @staticmethod
    def _default_key(stage: str) -> str:
        return {"request_image": "observation/image", "request_state": "observation/state",
                "request_prompt": "prompt", "response_action": "actions"}[stage]

    def _call(self, spec: AttackSpec, payload: Any, snapshot: Mapping[str, Any],
              runtime_context: Mapping[str, Any] | None = None) -> Any:
        context = {"attack_id": spec.attack_id, "mode": spec.mode, "stage": spec.stage,
                   "snapshot": snapshot, "runtime": dict(runtime_context or {})}
        result=self.handlers[spec.handler](payload, spec.parameters, context)
        recorder=getattr(self,'_paper_consumer_recorder',None)
        if recorder is not None:
            recorder.capture_consumption({'consumer':'attack_handler_boundary','receipt_status':'handler_returned',
                'attack_id':spec.attack_id,'stage':spec.stage,'handler':spec.handler,'parameters':spec.parameters,
                'physics_step':context['runtime'].get('physics_step'),
                'runtime':context['runtime'],
                'snapshot':{key:snapshot[key] for key in ('robot','arm','locomotion','objects','_attack_runtime') if key in snapshot}})
        return result

    def reset(self) -> None:
        self.triggered.clear()
        self.armed.clear()
        self.response_pipeline.clear()
        self.active_intercepts.clear()
        self.application_counts.clear()
        self.completed.clear()
        self.stop_reasons.clear()
        self.condition_counts.clear()
        self.condition_instances.clear()
        self.condition_clocks.clear()
        self.condition_observation_gaps.clear()
        self.effect_instances.clear()
        self.last_condition_markers = []
        self.latest_time_s = 0.0
