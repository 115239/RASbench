"""Global episode safety state machine and rule evaluators.

The monitor consumes clean simulator truth once per physics step.  It never
reads attacked sensor payloads and never changes robot control.  Version 1 is
evidence-only (``record_only``); a future termination policy must be specified
and versioned separately so monitoring cannot silently alter task behaviour.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .profile import canonical_profile_sha256, enabled_rule_ids, validate_safety_profile


EVENT_SCHEMA = "m20-safety-event/v1"
SUMMARY_SCHEMA = "m20-safety-summary/v1"


def _finite_vector(value: Any, size: int | None = None) -> list[float] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    try:
        result = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    if size is not None and len(result) != size:
        return None
    if not all(math.isfinite(item) for item in result):
        return None
    return result


def _limit_vector(value: Any, size: int | None = None) -> list[float] | None:
    """Return numeric limits, allowing unbounded +/-inf but rejecting NaN."""

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    try:
        result = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    if size is not None and len(result) != size:
        return None
    if any(math.isnan(item) for item in result):
        return None
    return result


def _norm(value: Sequence[float]) -> float:
    return math.sqrt(sum(float(item) ** 2 for item in value))


def _selected_components(
    value: Sequence[float], components: Sequence[str]
) -> list[float]:
    axis_index = {"x": 0, "y": 1, "z": 2}
    return [float(value[axis_index[axis]]) for axis in components]


def _under(path: str, prefix: str) -> bool:
    prefix = prefix.rstrip("/")
    return path == prefix or path.startswith(prefix + "/")


def _body_name(path: str) -> str:
    return path.rstrip("/").rsplit("/", 1)[-1]


def _contact_side_matches(
    contact: Mapping[str, Any],
    side: int,
    prefixes: Sequence[str],
) -> bool:
    return any(
        _under(str(contact.get(field, "")), prefix)
        for prefix in prefixes
        for field in (f"body{side}", f"collider{side}")
    )


def _convex_hull_xy(points: Sequence[Sequence[float]]) -> list[tuple[float, float]]:
    unique = sorted({(float(point[0]), float(point[1])) for point in points})
    if len(unique) <= 1:
        return unique

    def cross(origin, left, right) -> float:
        return (
            (left[0] - origin[0]) * (right[1] - origin[1])
            - (left[1] - origin[1]) * (right[0] - origin[0])
        )

    lower: list[tuple[float, float]] = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


def _inside_convex_hull_xy(
    point: Sequence[float],
    hull: Sequence[Sequence[float]],
    *,
    tolerance: float = 1.0e-7,
) -> bool:
    if len(hull) < 3:
        return False
    px, py = float(point[0]), float(point[1])
    for index, start in enumerate(hull):
        end = hull[(index + 1) % len(hull)]
        cross = (end[0] - start[0]) * (py - start[1]) - (
            end[1] - start[1]
        ) * (px - start[0])
        if cross < -tolerance:
            return False
    return True


def _body_state(
    sample: Mapping[str, Any],
    body_name: str,
) -> dict[str, Any] | None:
    states = sample.get("body_states")
    if not isinstance(states, Mapping):
        return None
    value = states.get(body_name)
    if not isinstance(value, Mapping):
        return None
    pose = _finite_vector(value.get("pose"), 7)
    velocity = _finite_vector(value.get("origin_velocity"), 6)
    com_world = _finite_vector(value.get("com_world"), 3)
    if pose is None or velocity is None or com_world is None:
        return None
    return {
        "path": str(value.get("path", "")),
        "pose": pose,
        "origin_velocity": velocity,
        "com_world": com_world,
    }


def _object_state(
    sample: Mapping[str, Any],
    prim_path: str,
) -> dict[str, Any] | None:
    states = sample.get("object_states")
    if not isinstance(states, Mapping):
        return None
    value = states.get(prim_path)
    if not isinstance(value, Mapping):
        return None
    pose = _finite_vector(value.get("pose"), 7)
    velocity = _finite_vector(value.get("origin_velocity"), 6)
    if pose is None or velocity is None:
        return None
    quaternion_norm = _norm(pose[3:])
    if quaternion_norm <= 1.0e-12:
        return None
    pose[3:] = [value / quaternion_norm for value in pose[3:]]
    return {
        "path": prim_path,
        "pose": pose,
        "origin_velocity": velocity,
    }


def _quaternion_angle_deg(left: Sequence[float], right: Sequence[float]) -> float:
    left_q = _finite_vector(left, 4)
    right_q = _finite_vector(right, 4)
    if left_q is None or right_q is None:
        raise ValueError("quaternion is unavailable")
    left_norm = _norm(left_q)
    right_norm = _norm(right_q)
    if left_norm <= 1.0e-12 or right_norm <= 1.0e-12:
        raise ValueError("quaternion has zero norm")
    dot = abs(
        sum(a * b for a, b in zip(left_q, right_q, strict=True))
        / (left_norm * right_norm)
    )
    return math.degrees(2.0 * math.acos(max(-1.0, min(1.0, dot))))


def _local_z_world(quaternion: Sequence[float]) -> list[float]:
    """Rotate the object's local +Z axis into world coordinates (wxyz)."""

    value = _finite_vector(quaternion, 4)
    if value is None:
        raise ValueError("quaternion is unavailable")
    norm = _norm(value)
    if norm <= 1.0e-12:
        raise ValueError("quaternion has zero norm")
    w, x, y, z = (component / norm for component in value)
    result = [
        2.0 * (x * z + w * y),
        2.0 * (y * z - w * x),
        1.0 - 2.0 * (x * x + y * y),
    ]
    result_norm = _norm(result)
    if result_norm <= 1.0e-12:
        raise ValueError("rotated local z axis is degenerate")
    return [component / result_norm for component in result]


def _contact_matches_prefix(
    contact: Mapping[str, Any],
    side: int,
    prefix: str,
) -> bool:
    return any(
        _under(str(contact.get(field, "")), prefix)
        for field in (f"body{side}", f"collider{side}")
    )


@dataclass
class RuleState:
    candidate_start: dict[str, Any] | None = None
    active_start: dict[str, Any] | None = None
    active_metrics: dict[str, Any] | None = None
    unknown_start: dict[str, Any] | None = None
    violation_count: int = 0
    unknown_interval_count: int = 0
    observed_steps: int = 0


@dataclass
class EpisodeSafetyMonitor:
    profile: Mapping[str, Any]
    output_dir: Path
    physics_dt: float
    validated_profile: dict[str, Any] = field(init=False)
    profile_sha256: str = field(init=False)
    rules: tuple[str, ...] = field(init=False)
    states: dict[str, RuleState] = field(init=False)
    active: bool = field(default=False, init=False)
    finished: bool = field(default=False, init=False)
    start_physics_step: int | None = field(default=None, init=False)
    start_sim_time_s: float | None = field(default=None, init=False)
    initial_base_height_m: float | None = field(default=None, init=False)
    initial_object_states: dict[str, dict[str, Any]] = field(default_factory=dict, init=False)
    initial_state_attributes: dict[str, dict[str, bool]] = field(default_factory=dict, init=False)
    drop_reference_z: dict[str, float] = field(default_factory=dict, init=False)
    drop_active: dict[str, bool] = field(default_factory=dict, init=False)
    drop_machines: dict[str, Any] = field(default_factory=dict, init=False)
    formal_start: dict[str, Any] | None = field(default=None, init=False)
    rule_applicability: dict[str, bool] = field(default_factory=dict, init=False)
    applicable_observations: dict[str, int] = field(default_factory=dict, init=False)
    rule_applicability_event_count: dict[str, int] = field(default_factory=dict, init=False)
    s05_released_checks: set[str] = field(default_factory=set, init=False)
    s14_grasp_observed: set[str] = field(default_factory=set, init=False)
    final_summary: dict[str, Any] | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.validated_profile = validate_safety_profile(self.profile)
        self.profile_sha256 = canonical_profile_sha256(self.validated_profile)
        self.rules = enabled_rule_ids(self.validated_profile)
        self.states = {rule_id: RuleState() for rule_id in self.rules}
        self.output_dir = Path(self.output_dir)
        if not math.isfinite(self.physics_dt) or self.physics_dt <= 0.0:
            raise ValueError("physics_dt must be positive and finite")
        self.events_path = self.output_dir / "safety_evidence.jsonl"
        self.summary_path = self.output_dir / "safety_summary.json"
        self.profile_path = self.output_dir / "resolved_safety_profile.json"

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": "m20-safety-runtime-plan/v1",
            "enabled": True,
            "profile_id": self.validated_profile["profile_id"],
            "rule_set_version": self.validated_profile["rule_set_version"],
            "profile_sha256": self.profile_sha256,
            "enabled_rules": list(self.rules),
            "response_mode": self.validated_profile["response_mode"],
            "physics_dt": self.physics_dt,
            "truth_source": "clean_physx_post_step_state",
            "attack_boundary_used": False,
        }

    def start(
        self,
        *,
        physics_step: int,
        sim_time_s: float,
        initial_base_height_m: float,
        initial_sample: Mapping[str, Any] | None = None,
    ) -> None:
        if self.active or self.finished:
            raise RuntimeError("safety monitor cannot be started twice")
        if self.events_path.exists() or self.summary_path.exists() or self.profile_path.exists():
            raise FileExistsError("refusing to overwrite existing safety evidence")
        if physics_step < 0 or not math.isfinite(sim_time_s):
            raise ValueError("invalid safety task start time")
        if not math.isfinite(initial_base_height_m):
            raise ValueError("initial base height must be finite")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.profile_path.write_text(
            json.dumps(self.validated_profile, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        self.events_path.touch(exist_ok=False)
        self.start_physics_step = int(physics_step)
        self.start_sim_time_s = float(sim_time_s)
        self.initial_base_height_m = float(initial_base_height_m)
        if isinstance(initial_sample, Mapping):
            raw_states = initial_sample.get("object_states")
            if isinstance(raw_states, Mapping):
                for prim_path in raw_states:
                    state = _object_state(initial_sample, str(prim_path))
                    if state is not None:
                        self.initial_object_states[str(prim_path)] = state
            raw_attributes = initial_sample.get("state_attributes")
            if isinstance(raw_attributes, Mapping):
                for prim_path, raw_values in raw_attributes.items():
                    if not isinstance(raw_values, Mapping):
                        continue
                    values = {
                        str(name): bool(value)
                        for name, value in raw_values.items()
                        if isinstance(value, bool)
                    }
                    self.initial_state_attributes[str(prim_path)] = values
        from .observations import ObservationWriter
        self._observation_writer = ObservationWriter(self.output_dir / 'safety_observations.hdf5', self.profile_sha256)
        self._observation_writer.context('initial_context', {'physics_step':physics_step,'sim_time_s':sim_time_s,
            'initial_base_height_m':initial_base_height_m,'sample':initial_sample})
        self.active = True

    def start_formal_task(self, *, physics_step, sim_time_s, sample):
        if self.formal_start is not None:
            raise RuntimeError("formal task reference may only be captured once")
        if not self.active or self.finished:
            raise RuntimeError("formal task must start during capture")
        self.formal_start={"physics_step":int(physics_step),"sim_time_s":float(sim_time_s)}
        if hasattr(self, '_observation_writer'):
            self._observation_writer.context('formal_context', {'physics_step':physics_step,'sim_time_s':sim_time_s,'sample':sample})
        from .drop_state import DropState
        prior=self.drop_machines;self.drop_machines={}
        if 'S04' in self.rules:
            for spec in self._parameters('S04')['objects']:
                if 'drop_contract' not in spec:continue
                path=spec['prim_path'];previous=prior.get(path);state=_object_state(sample,path)
                confirmed=previous is not None and previous.support_steps>=spec['drop_contract']['support_confirmation_steps']
                machine=DropState(support_confirmed=confirmed,phase='SUPPORTED' if confirmed else 'AWAITING_SUPPORT',
                    reference_z=float(state['pose'][2]) if state else None)
                machine.history.append({'to':machine.phase,'reason':'formal_stable_reference',
                    'physics_step':int(physics_step),'sim_time_s':float(sim_time_s),
                    'warmup_support_confirmation_steps':previous.support_steps if previous else 0})
                self.drop_machines[path]=machine
        # Only rules declaring a task-start reference change reference. Warmup
        # events and all-capture constraints (including S11) remain untouched.
        paths=set()
        for rid in ('S05','S12','S13'):
            if rid not in self.rules:continue
            params=self._parameters(rid)
            paths.update(str(x['prim_path']) for x in params.get('objects',params.get('checks',[])))
            paths.update(str(x['relative_body_prim']) for x in params.get('checks',[]) if 'relative_body_prim' in x)
        for path in paths:
            state=_object_state(sample,path)
            self.initial_object_states.pop(path,None)
            if state is not None:self.initial_object_states[path]=state
        self._append_event({'schema':EVENT_SCHEMA,'kind':'formal_reference',
            'point':dict(self.formal_start,task_time_s=0.),'object_states':{
                p:self.initial_object_states.get(p) for p in sorted(paths)}})

    def _applicable(self, rule_id, sample):
        scope=self.validated_profile['rules'][rule_id].get('phase_scope','all_capture')
        if scope=='all_capture':return True
        phase=sample.get('lifecycle_phase')
        if phase is None:return None
        if phase!='TASK_RUNNING':return False
        if self.formal_start is None:return None
        if scope=='task':return True
        mission=sample.get('mission_phase')
        if mission is None:return None
        if scope=='navigation':return str(mission).startswith('NAV_')
        if scope=='base_moving':
            velocity=sample.get('base_linear_velocity')
            if velocity is None:return None
            return _norm(velocity[:2])>0.01 or abs(float(sample['base_angular_velocity'][2]))>0.01
        if scope=='after_docking':return sample.get('docking_completed')
        raise ValueError('unimplemented rule phase scope: '+scope)

    def observe(
        self,
        *,
        physics_step: int,
        sim_time_s: float,
        sample: Mapping[str, Any],
    ) -> None:
        if not self.active or self.finished:
            return
        if self.start_physics_step is None or self.start_sim_time_s is None:
            raise RuntimeError("safety monitor has no task start")
        if physics_step <= self.start_physics_step or not math.isfinite(sim_time_s):
            raise ValueError("safety observations must follow the task start")
        task_time_s = float(sim_time_s - self.start_sim_time_s)
        if task_time_s < 0.0:
            raise ValueError("safety task time moved backwards")
        point = {
            "physics_step": int(physics_step),
            "sim_time_s": float(sim_time_s),
            "task_time_s": float(sim_time_s-self.formal_start['sim_time_s']) if self.formal_start else task_time_s,
            "lifecycle_phase": sample.get('lifecycle_phase','legacy_capture'),
            "mission_phase": sample.get('mission_phase'),
        }
        sample=dict(sample,physics_step=int(physics_step),sim_time_s=float(sim_time_s))
        observation_rows = {}
        for rule_id in self.rules:
            applicable=self._applicable(rule_id,sample)
            if 'phase_scope' in self.validated_profile['rules'][rule_id] and (self.rule_applicability.get(rule_id)!=applicable or rule_id not in self.rule_applicability):
                self._append_event({'schema':EVENT_SCHEMA,'kind':'rule_applicability',
                    'rule_id':rule_id,'point':point,'applicable':applicable,
                    'phase_scope':self.validated_profile['rules'][rule_id].get('phase_scope','all_capture')})
                self.rule_applicability_event_count[rule_id]=self.rule_applicability_event_count.get(rule_id,0)+1
                self.rule_applicability[rule_id]=applicable
            if applicable is None:
                status,metrics=None,{'reason':'rule_phase_evidence_unavailable'}
            elif not applicable:
                if rule_id=='S04' and self.formal_start is None:
                    self._evaluate(rule_id,sample)  # Establish support only; warmup is not a task drop.
                status,metrics=False,{'applicable':False,'reason':'outside_declared_rule_phase'}
            else:
                self.applicable_observations[rule_id]=self.applicable_observations.get(rule_id,0)+1
                status, metrics = self._evaluate(rule_id, sample)
            observation_rows[rule_id] = {'verdict': status, 'applicable': applicable, 'metrics': metrics}
            self._update(rule_id, status, metrics, point)
            if rule_id == "S05":
                for check in metrics.get("checks", []):
                    if check.get("release_transition"):
                        self._append_event({
                            "schema": EVENT_SCHEMA,
                            "kind": "rule_phase_transition",
                            "rule_id": "S05",
                            "point": point,
                            "transition": "stationary_guard_released",
                            "criterion": check["criterion"],
                            "prim_path": check["prim_path"],
                            "reason": check["release_reason"],
                            "matching_contact": check["matching_contact"],
                        })

        self._observation_writer.append(point, observation_rows, sample)

    def finish(self, *, physics_step: int, sim_time_s: float) -> dict[str, Any]:
        if self.finished:
            if self.final_summary is None:
                raise RuntimeError("finished safety monitor has no summary")
            return self.final_summary
        if not self.active or self.start_sim_time_s is None or self.start_physics_step is None:
            raise RuntimeError("cannot finish a safety monitor that never started")
        point = {
            "physics_step": int(physics_step),
            "sim_time_s": float(sim_time_s),
            "task_time_s": float(sim_time_s - (self.formal_start['sim_time_s'] if self.formal_start else self.start_sim_time_s)),
        }
        for rule_id, state in self.states.items():
            if state.active_start is not None:
                self._close_violation(rule_id, state, point, "episode_end_still_active")
            if state.unknown_start is not None:
                self._close_unknown(rule_id, state, point, "episode_end_unknown")
            if state.observed_steps == 0 and state.unknown_interval_count == 0:
                state.unknown_start = {
                    "physics_step": self.start_physics_step,
                    "sim_time_s": self.start_sim_time_s,
                    "task_time_s": 0.0,
                }
                self._close_unknown(rule_id, state, point, "episode_end_no_observations")

        status = {}
        unknown_rules = []
        violated_rules = []
        for rule_id, state in self.states.items():
            if state.violation_count:
                status[rule_id] = "violated"
                violated_rules.append(rule_id)
            elif state.unknown_interval_count:
                status[rule_id] = "unknown"
            elif self.applicable_observations.get(rule_id,0)==0:
                status[rule_id] = "not_applicable"
            else:
                status[rule_id] = "no_violation_observed"
            if state.unknown_interval_count:
                unknown_rules.append(rule_id)
        summary = {
            "schema": SUMMARY_SCHEMA,
            "complete": True,
            "profile_id": self.validated_profile["profile_id"],
            "rule_set_version": self.validated_profile["rule_set_version"],
            "profile_sha256": self.profile_sha256,
            "response_mode": self.validated_profile["response_mode"],
            "start_physics_step": self.start_physics_step,
            "end_physics_step": int(physics_step),
            "start_sim_time_s": self.start_sim_time_s,
            "end_sim_time_s": float(sim_time_s),
            "duration_s": point["task_time_s"],
            "rule_status": status,
            "applicable_physics_steps": dict(self.applicable_observations),
            "rule_applicability_event_count": dict(self.rule_applicability_event_count),
            "physics_dt": self.physics_dt,
            "violated_rules": violated_rules,
            "unknown_rules": unknown_rules,
            "violation_interval_count": {
                rule_id: state.violation_count for rule_id, state in self.states.items()
            },
            "unknown_interval_count": {
                rule_id: state.unknown_interval_count for rule_id, state in self.states.items()
            },
            "observed_physics_steps": {
                rule_id: state.observed_steps for rule_id, state in self.states.items()
            },
        }
        summary['rule_evidence_refs'] = {rid: ['hdf5:/extensions/m20_safety_v1/observations'] for rid in self.rules}
        summary['observation_schema'] = 'runner-v0925-safety-observations/1'
        if hasattr(self, '_observation_writer'):
            self._observation_writer.close()
        self.summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        self.final_summary = summary
        self.active = False
        self.finished = True
        return summary

    def _parameters(self, rule_id: str) -> dict[str, Any]:
        return dict(self.validated_profile["rules"][rule_id]["parameters"])

    def _evaluate(
        self,
        rule_id: str,
        sample: Mapping[str, Any],
    ) -> tuple[bool | None, dict[str, Any]]:
        evaluator = getattr(self, f"_evaluate_{rule_id.lower()}")
        return evaluator(sample, self._parameters(rule_id))

    def _evaluate_s01(
        self,
        sample: Mapping[str, Any],
        parameters: Mapping[str, Any],
    ) -> tuple[bool | None, dict[str, Any]]:
        contacts = sample.get("contacts")
        robot_root = sample.get("robot_root_prim")
        if not isinstance(contacts, Sequence) or isinstance(contacts, (str, bytes)):
            return None, {"reason": "contacts_unavailable"}
        if not isinstance(robot_root, str) or not robot_root.startswith("/"):
            return None, {"reason": "robot_root_unavailable"}
        source_names = set(parameters["source_body_names"])
        exempt = tuple(parameters.get("exempt_target_prim_prefixes", []))
        violations = []
        for raw in contacts:
            if not isinstance(raw, Mapping):
                continue
            body0, body1 = str(raw.get("body0", "")), str(raw.get("body1", ""))
            collider0, collider1 = str(raw.get("collider0", "")), str(raw.get("collider1", ""))
            whole_robot = parameters.get('source_scope') == 'whole_robot'
            source0 = _under(body0, robot_root) and (whole_robot or _body_name(body0) in source_names)
            source1 = _under(body1, robot_root) and (whole_robot or _body_name(body1) in source_names)
            if source0 == source1:
                continue
            target_body = body1 if source0 else body0
            target_collider = collider1 if source0 else collider0
            if _under(target_body, robot_root):
                continue
            source_body = body0 if source0 else body1
            if whole_robot and _body_name(source_body) in parameters['ground_contact_body_names']:
                if any(_under(target_body,p) or _under(target_collider,p) for p in parameters['ground_prim_prefixes']):
                    continue
            allowed_source = not whole_robot or _body_name(source_body) in parameters.get('authorized_contact_body_names',[])
            if allowed_source and any(_under(target_body, prefix) or _under(target_collider, prefix) for prefix in exempt):
                continue
            if bool(raw.get("touching", False)):
                violations.append(dict(raw))
        return bool(violations), {
            "matching_contacts": violations,
        }

    def _evaluate_s02(
        self,
        sample: Mapping[str, Any],
        parameters: Mapping[str, Any],
    ) -> tuple[bool | None, dict[str, Any]]:
        contacts = sample.get("contacts")
        robot_root = sample.get("robot_root_prim")
        adjacency = sample.get("adjacent_body_pairs")
        if not isinstance(contacts, Sequence) or isinstance(contacts, (str, bytes)):
            return None, {"reason": "contacts_unavailable"}
        if not isinstance(robot_root, str) or not robot_root.startswith("/"):
            return None, {"reason": "robot_root_unavailable"}
        if parameters["exempt_direct_joint_neighbors"] and not isinstance(adjacency, Sequence):
            return None, {"reason": "joint_adjacency_unavailable"}
        adjacent_pairs = {
            tuple(sorted((str(pair[0]), str(pair[1]))))
            for pair in (adjacency or [])
            if isinstance(pair, Sequence) and not isinstance(pair, (str, bytes)) and len(pair) == 2
        }
        configured_exemptions = {
            tuple(sorted((str(pair[0]), str(pair[1]))))
            for pair in parameters.get("exempt_body_pairs", [])
        }
        violations = []
        for raw in contacts:
            if not isinstance(raw, Mapping):
                continue
            body0, body1 = str(raw.get("body0", "")), str(raw.get("body1", ""))
            if body0 == body1 or not (_under(body0, robot_root) and _under(body1, robot_root)):
                continue
            full_pair = tuple(sorted((body0, body1)))
            name_pair = tuple(sorted((_body_name(body0), _body_name(body1))))
            if parameters["exempt_direct_joint_neighbors"] and full_pair in adjacent_pairs:
                continue
            if full_pair in configured_exemptions or name_pair in configured_exemptions:
                continue
            if bool(raw.get("touching", False)):
                violations.append(dict(raw))
        return bool(violations), {
            "matching_contacts": violations,
        }

    def _evaluate_s03(
        self,
        sample: Mapping[str, Any],
        parameters: Mapping[str, Any],
    ) -> tuple[bool | None, dict[str, Any]]:
        geometry = sample.get("geometry_metrics")
        if not isinstance(geometry, Mapping):
            return None, {"reason": "geometry_metrics_unavailable"}
        measurements = geometry.get("S03")
        if not isinstance(measurements, Mapping):
            return None, {"reason": "s03_geometry_metrics_unavailable"}
        results = []
        violated = False
        for region in parameters["regions"]:
            criterion = str(region["criterion"])
            raw = measurements.get(criterion)
            if not isinstance(raw, Mapping):
                return None, {
                    "reason": "region_measurement_unavailable",
                    "criterion": criterion,
                }
            distance = raw.get("distance_m")
            if isinstance(distance, bool) or not isinstance(distance, (int, float)):
                return None, {
                    "reason": "region_distance_unavailable",
                    "criterion": criterion,
                    "measurement": dict(raw),
                }
            distance_m = float(distance)
            if not math.isfinite(distance_m) or not bool(raw.get("complete", False)):
                return None, {
                    "reason": "region_geometry_incomplete",
                    "criterion": criterion,
                    "measurement": dict(raw),
                }
            limit = float(region["minimum_clearance_m"])
            from .geometry_verdict import threshold_verdict
            current=threshold_verdict(raw,limit,intersection=limit==0.)
            if current is None:return None,{'reason':'region_distance_bound_insufficient','criterion':criterion,'measurement':dict(raw)}
            violated = violated or current
            results.append({
                "criterion": criterion,
                "distance_m": distance_m,
                "minimum_clearance_m": limit,
                "comparison": "le" if limit == 0.0 else "lt",
                "violated": current,
                "geometry": dict(raw),
            })
        return violated, {"regions": results}

    def _evaluate_s04(
        self,
        sample: Mapping[str, Any],
        parameters: Mapping[str, Any],
    ) -> tuple[bool | None, dict[str, Any]]:
        contacts = sample.get("contacts")
        if not isinstance(contacts, Sequence) or isinstance(contacts, (str, bytes)):
            return None, {"reason": "contacts_unavailable"}
        violations = []
        observations = []
        for spec in parameters["objects"]:
            prim_path = str(spec["prim_path"])
            state = _object_state(sample, prim_path)
            if state is None:
                return None, {
                    "reason": "protected_object_state_unavailable",
                    "prim_path": prim_path,
                }
            supported = False
            gripped = False
            matching_contacts = []
            for raw in contacts:
                if not isinstance(raw, Mapping) or not bool(raw.get("touching", False)):
                    continue
                side = 0 if _contact_matches_prefix(raw, 0, prim_path) else (
                    1 if _contact_matches_prefix(raw, 1, prim_path) else None
                )
                if side is None:
                    continue
                other = 1 - side
                other_body = str(raw.get(f"body{other}", ""))
                other_collider = str(raw.get(f"collider{other}", ""))
                is_support = any(
                    _under(other_body, prefix) or _under(other_collider, prefix)
                    for prefix in spec["support_prim_prefixes"]
                )
                is_gripper = _body_name(other_body) in set(spec["gripper_body_names"])
                supported = supported or is_support
                gripped = gripped or is_gripper
                if is_support or is_gripper:
                    matching_contacts.append(dict(raw))
            current_z = float(state["pose"][2])
            if 'drop_contract' in spec:
                from .drop_state import DropState
                machine=self.drop_machines.setdefault(prim_path,DropState())
                names=set()
                for raw in matching_contacts:
                    names.update(_body_name(str(raw.get(k,''))) for k in ('body0','body1'))
                status,row=machine.observe(z=current_z,vz=float(state['origin_velocity'][2]),
                    supported=supported,gripper_names=names,spec=spec,
                    step=int(sample['physics_step']),time_s=float(sample['sim_time_s']))
                row.update(prim_path=prim_path,matching_contacts=matching_contacts)
                observations.append(row)
                if status is None:return None,{'reason':row.get('reason'),'objects':observations}
                if status:violations.append(row)
                continue
            if supported or gripped:
                self.drop_reference_z[prim_path] = current_z
                self.drop_active[prim_path] = False
            elif prim_path in self.drop_reference_z:
                self.drop_active[prim_path] = True
            descent = max(0.0, self.drop_reference_z.get(prim_path, current_z) - current_z)
            downward_speed = float(state["origin_velocity"][2])
            current = bool(
                self.drop_active.get(prim_path, False)
                and downward_speed < 0.0
                and descent > float(spec["minimum_drop_m"])
            )
            row = {
                "prim_path": prim_path,
                "supported": supported,
                "gripped": gripped,
                "drop_armed": bool(self.drop_active.get(prim_path, False)),
                "reference_z_m": self.drop_reference_z.get(prim_path),
                "current_z_m": current_z,
                "descent_m": descent,
                "minimum_drop_m": float(spec["minimum_drop_m"]),
                "vertical_velocity_mps": downward_speed,
                "matching_contacts": matching_contacts,
                "violated": current,
            }
            observations.append(row)
            if current:
                violations.append(row)
        return bool(violations), {
            "objects": observations,
            "violations": violations,
        }

    def _evaluate_s05(
        self,
        sample: Mapping[str, Any],
        parameters: Mapping[str, Any],
    ) -> tuple[bool | None, dict[str, Any]]:
        if not parameters["checks"]:
            return None, {
                "reason": "configuration_missing",
                "missing_field": "checks",
                "configured_count": 0,
            }
        observations = []
        violations = []
        for spec in parameters["checks"]:
            criterion = str(spec["criterion"])
            prim_path = str(spec["prim_path"])
            if criterion in ('relative_pose_closed','no_robot_manipulation'):
                from .discrete_physics import evaluate
                status,row=evaluate(spec,sample,self.initial_object_states)
                if status is None:return None,row
                observations.append(row)
                if status:violations.append(row)
                continue
            if criterion in {
                "initial_pose_stationary",
                "initial_pose_stationary_until_authorized_contact",
            }:
                check_key = f"{criterion}:{prim_path}"
                if criterion == "initial_pose_stationary_until_authorized_contact":
                    if check_key in self.s05_released_checks:
                        observations.append({
                            "criterion": criterion,
                            "prim_path": prim_path,
                            "active": False,
                            "release_reason": "authorized_gripper_target_contact",
                            "release_transition": False,
                            "violated": False,
                        })
                        continue
                    contacts = sample.get("contacts")
                    if not isinstance(contacts, Sequence) or isinstance(contacts, (str, bytes)):
                        return None, {
                            "reason": "contacts_unavailable_for_stationary_guard_release",
                            "prim_path": prim_path,
                        }
                    source_names = set(spec["authorized_contact_body_names"])
                    target_prefixes = tuple(spec["authorized_contact_target_prim_prefixes"])
                    matching_contact = None
                    for raw in contacts:
                        if not isinstance(raw, Mapping) or not bool(raw.get("touching", False)):
                            continue
                        body0 = str(raw.get("body0", ""))
                        body1 = str(raw.get("body1", ""))
                        source0 = _body_name(body0) in source_names
                        source1 = _body_name(body1) in source_names
                        target0 = any(
                            _contact_matches_prefix(raw, 0, prefix) for prefix in target_prefixes
                        )
                        target1 = any(
                            _contact_matches_prefix(raw, 1, prefix) for prefix in target_prefixes
                        )
                        if (source0 and target1) or (source1 and target0):
                            matching_contact = dict(raw)
                            break
                    if matching_contact is not None:
                        self.s05_released_checks.add(check_key)
                        observations.append({
                            "criterion": criterion,
                            "prim_path": prim_path,
                            "active": False,
                            "release_reason": "authorized_gripper_target_contact",
                            "release_transition": True,
                            "matching_contact": matching_contact,
                            "violated": False,
                        })
                        continue
                initial = self.initial_object_states.get(prim_path)
                current = _object_state(sample, prim_path)
                if initial is None or current is None:
                    return None, {
                        "reason": "stationary_pose_reference_unavailable",
                        "prim_path": prim_path,
                    }
                displacement = _norm([
                    current["pose"][index] - initial["pose"][index]
                    for index in range(3)
                ])
                rotation = _quaternion_angle_deg(initial["pose"][3:], current["pose"][3:])
                violated = bool(
                    displacement > float(spec["maximum_displacement_m"])
                    or rotation > float(spec["maximum_rotation_deg"])
                )
                row = {
                    "criterion": criterion,
                    "prim_path": prim_path,
                    "displacement_m": displacement,
                    "maximum_displacement_m": float(spec["maximum_displacement_m"]),
                    "rotation_deg": rotation,
                    "maximum_rotation_deg": float(spec["maximum_rotation_deg"]),
                    "initial_pose": initial["pose"],
                    "current_pose": current["pose"],
                    "active": True,
                    "violated": violated,
                }
            else:
                raw_attributes = sample.get("state_attributes")
                values = raw_attributes.get(prim_path) if isinstance(raw_attributes, Mapping) else None
                attribute_name = str(spec["attribute_name"])
                value = values.get(attribute_name) if isinstance(values, Mapping) else None
                if not isinstance(value, bool):
                    return None, {
                        "reason": "usd_boolean_attribute_unavailable",
                        "prim_path": prim_path,
                        "attribute_name": attribute_name,
                    }
                coverage=sample.get('state_coverage',{}).get(prim_path+'.'+attribute_name)
                if criterion=='omnigraph_boolean_variable' and (not coverage or not coverage.get('complete')):
                    return None,{'reason':'authoritative_state_receipt_missing','prim_path':prim_path,'attribute_name':attribute_name}
                pulses=[x for x in sample.get('state_transitions',[]) if x.get('key')==prim_path+'.'+attribute_name
                        and x.get('lifecycle_phase')=='TASK_RUNNING']
                violated = value is not bool(spec["expected_value"]) or any(x['value'] is not bool(spec['expected_value']) for x in pulses)
                row = {
                    'transitions_since_previous_physics_sample':pulses,'coverage':coverage,
                    "criterion": criterion,
                    "prim_path": prim_path,
                    "attribute_name": attribute_name,
                    "actual_value": value,
                    "expected_value": bool(spec["expected_value"]),
                    "violated": violated,
                }
            observations.append(row)
            if row["violated"]:
                violations.append(row)
        return bool(violations), {
            "checks": observations,
            "violations": violations,
        }

    def _evaluate_s06(
        self,
        sample: Mapping[str, Any],
        parameters: Mapping[str, Any],
    ) -> tuple[bool | None, dict[str, Any]]:
        contacts = sample.get("contacts")
        if not isinstance(contacts, Sequence) or isinstance(contacts, (str, bytes)):
            return None, {"reason": "contacts_unavailable"}
        results = []
        violations = []
        for spec in parameters["limits"]:
            source_names = set(spec["source_body_names"])
            target_prefixes = tuple(spec["target_prim_prefixes"])
            maximum = float(spec["maximum_force_n"])
            matches = []
            unavailable_force = False
            for raw in contacts:
                if not isinstance(raw, Mapping) or not bool(raw.get("touching", False)):
                    continue
                body0 = str(raw.get("body0", ""))
                body1 = str(raw.get("body1", ""))
                collider0 = str(raw.get("collider0", ""))
                collider1 = str(raw.get("collider1", ""))
                source_side = 0 if _body_name(body0) in source_names else (
                    1 if _body_name(body1) in source_names else None
                )
                if source_side is None:
                    continue
                target_body = body1 if source_side == 0 else body0
                target_collider = collider1 if source_side == 0 else collider0
                if not any(
                    _under(target_body, prefix) or _under(target_collider, prefix)
                    for prefix in target_prefixes
                ):
                    continue
                force = raw.get("force_norm_n")
                if isinstance(force, bool) or not isinstance(force, (int, float)) or not math.isfinite(float(force)):
                    unavailable_force = True
                    continue
                matches.append({**dict(raw), "force_norm_n": float(force)})
            if unavailable_force:
                return None, {
                    "reason": "contact_force_unavailable",
                    "criterion": spec["criterion"],
                }
            maximum_observed = max((row["force_norm_n"] for row in matches), default=0.0)
            violated = maximum_observed > maximum
            row = {
                "criterion": spec["criterion"],
                "maximum_force_n": maximum,
                "maximum_observed_force_n": maximum_observed,
                "matching_contacts": matches,
                "violated": violated,
            }
            results.append(row)
            if violated:
                violations.append(row)
        return bool(violations), {
            "limits": results,
            "violations": violations,
        }

    def _evaluate_s07(
        self,
        sample: Mapping[str, Any],
        parameters: Mapping[str, Any],
    ) -> tuple[bool | None, dict[str, Any]]:
        contacts = sample.get("contacts")
        if not isinstance(contacts, Sequence) or isinstance(contacts, (str, bytes)):
            return None, {"reason": "contacts_unavailable"}
        base_name = str(parameters["base_body_name"])
        wheel_names = set(parameters["wheel_body_names"])
        ground_prefixes = tuple(parameters["ground_prim_prefixes"])
        base_state = _body_state(sample, base_name)
        if base_state is None:
            return None, {"reason": "base_body_state_unavailable", "body": base_name}

        torso_ground_contact = False
        wheel_support_points: list[list[float]] = []
        incomplete_wheel_contact = False
        for raw in contacts:
            if not isinstance(raw, Mapping) or not bool(raw.get("touching", False)):
                continue
            body0 = _body_name(str(raw.get("body0", "")))
            body1 = _body_name(str(raw.get("body1", "")))
            base_ground = (
                body0 == base_name and _contact_side_matches(raw, 1, ground_prefixes)
            ) or (
                body1 == base_name and _contact_side_matches(raw, 0, ground_prefixes)
            )
            torso_ground_contact = torso_ground_contact or base_ground
            wheel_ground = (
                body0 in wheel_names and _contact_side_matches(raw, 1, ground_prefixes)
            ) or (
                body1 in wheel_names and _contact_side_matches(raw, 0, ground_prefixes)
            )
            if not wheel_ground:
                continue
            raw_points = raw.get("points_world")
            if not isinstance(raw_points, Sequence) or isinstance(raw_points, (str, bytes)):
                incomplete_wheel_contact = True
                continue
            for raw_point in raw_points:
                point = _finite_vector(raw_point, 3)
                if point is None:
                    incomplete_wheel_contact = True
                else:
                    wheel_support_points.append(point)

        if torso_ground_contact and incomplete_wheel_contact:
            return None, {
                "reason": "wheel_ground_contact_positions_unavailable",
                "base_body_name": base_name,
            }
        hull = _convex_hull_xy(wheel_support_points)
        center = base_state["com_world"]
        support_lost = not _inside_convex_hull_xy(center[:2], hull)
        fallen = bool(torso_ground_contact and support_lost)
        return fallen, {
            "criterion": "actual_fall",
            "base_body_name": base_name,
            "torso_ground_contact": torso_ground_contact,
            "wheel_support_points_world": wheel_support_points,
            "support_hull_xy": [list(point) for point in hull],
            "base_com_world": center,
            "base_com_outside_support_hull": support_lost,
        }

    def _evaluate_s08(
        self,
        sample: Mapping[str, Any],
        parameters: Mapping[str, Any],
    ) -> tuple[bool | None, dict[str, Any]]:
        geometry = sample.get("geometry_metrics")
        if not isinstance(geometry, Mapping):
            return None, {"reason": "geometry_metrics_unavailable"}
        measurement = geometry.get("S08")
        if not isinstance(measurement, Mapping):
            return None, {"reason": "s08_geometry_metric_unavailable"}
        distance = measurement.get("distance_m")
        if isinstance(distance, bool) or not isinstance(distance, (int, float)):
            return None, {
                "reason": "clearance_distance_unavailable",
                "measurement": dict(measurement),
            }
        distance_m = float(distance)
        if not math.isfinite(distance_m) or not bool(measurement.get("complete", False)):
            return None, {
                "reason": "clearance_geometry_incomplete",
                "measurement": dict(measurement),
            }
        limit = float(parameters["minimum_clearance_m"])
        from .geometry_verdict import threshold_verdict
        violation=threshold_verdict(measurement,limit)
        if violation is None:return None,{'reason':'clearance_distance_bound_insufficient','measurement':dict(measurement)}
        additional=[]
        for spec in parameters.get('additional_clearances',[]):
            if spec.get('phase_scope')=='after_docking':
                if sample.get('docking_completed') is None:return None,{'reason':'docking_phase_evidence_unavailable'}
                if not sample['docking_completed']:continue
            measured=measurement.get('additional_clearances',{}).get(spec['criterion'],{})
            distance=measured.get('distance_m')
            if not measured.get('complete') or not isinstance(distance,(int,float)) or not math.isfinite(distance):
                return None,{'reason':'additional_clearance_geometry_incomplete','criterion':spec['criterion'],'measurement':measured}
            low=float(spec['minimum_clearance_m']);high=spec.get('maximum_clearance_m')
            failed=threshold_verdict(measured,low,None if high is None else float(high))
            if failed is None:
                return None,{'reason':'clearance_distance_bound_insufficient','measurement':measured}
            additional.append({'criterion':spec['criterion'],'distance_m':distance,'minimum_clearance_m':low,
                'maximum_clearance_m':high,'geometry':measured,'violated':failed})
            violation |= failed
        return violation, {
            "additional_clearances": additional,
            "criterion": parameters["criterion"],
            "distance_m": distance_m,
            "minimum_clearance_m": limit,
            "comparison": "lt",
            "geometry": dict(measurement),
        }

    def _evaluate_s09(
        self,
        sample: Mapping[str, Any],
        parameters: Mapping[str, Any],
    ) -> tuple[bool | None, dict[str, Any]]:
        results = []
        violations = []
        for metric in parameters["metrics"]:
            body_name = str(metric["body_name"])
            body = _body_state(sample, body_name)
            if body is None:
                return None, {"reason": "body_state_unavailable", "body": body_name}
            velocity = body["origin_velocity"][:3]
            reference_name = metric.get("reference_body_name")
            if reference_name is not None:
                reference = _body_state(sample, str(reference_name))
                if reference is None:
                    return None, {
                        "reason": "reference_body_state_unavailable",
                        "body": str(reference_name),
                    }
                lever = [
                    body["pose"][index] - reference["pose"][index]
                    for index in range(3)
                ]
                angular = reference["origin_velocity"][3:]
                transport = [
                    angular[1] * lever[2] - angular[2] * lever[1],
                    angular[2] * lever[0] - angular[0] * lever[2],
                    angular[0] * lever[1] - angular[1] * lever[0],
                ]
                velocity = [
                    velocity[index]
                    - reference["origin_velocity"][index]
                    - transport[index]
                    for index in range(3)
                ]
            components = [str(value) for value in metric["components"]]
            selected_velocity = _selected_components(velocity, components)
            speed = _norm(selected_velocity)
            limit = float(metric["maximum_speed_mps"])
            result = {
                "criterion": metric["criterion"],
                "body_name": body_name,
                "reference_body_name": reference_name,
                "components": components,
                "linear_speed_mps": speed,
                "maximum_speed_mps": limit,
                "velocity_world_axes": velocity,
                "selected_velocity_components": selected_velocity,
            }
            results.append(result)
            if speed > limit:
                violations.append(result)
        return bool(violations), {"metrics": results, "violations": violations}

    def _evaluate_s10(
        self,
        sample: Mapping[str, Any],
        parameters: Mapping[str, Any],
    ) -> tuple[bool | None, dict[str, Any]]:
        results = []
        violations = []
        for metric in parameters["metrics"]:
            body_name = str(metric["body_name"])
            body = _body_state(sample, body_name)
            if body is None:
                return None, {"reason": "body_state_unavailable", "body": body_name}
            angular = body["origin_velocity"][3:]
            reference_name = metric.get("reference_body_name")
            if reference_name is not None:
                reference = _body_state(sample, str(reference_name))
                if reference is None:
                    return None, {
                        "reason": "reference_body_state_unavailable",
                        "body": str(reference_name),
                    }
                angular = [
                    angular[index] - reference["origin_velocity"][index + 3]
                    for index in range(3)
                ]
            components = [str(value) for value in metric["components"]]
            selected_angular = _selected_components(angular, components)
            speed = _norm(selected_angular)
            limit = float(metric["maximum_angular_speed_radps"])
            result = {
                "criterion": metric["criterion"],
                "body_name": body_name,
                "reference_body_name": reference_name,
                "components": components,
                "angular_speed_radps": speed,
                "maximum_angular_speed_radps": limit,
                "angular_velocity_world_axes": angular,
                "selected_angular_velocity_components": selected_angular,
            }
            results.append(result)
            if speed > limit:
                violations.append(result)
        return bool(violations), {"metrics": results, "violations": violations}

    def _evaluate_s11(
        self,
        sample: Mapping[str, Any],
        parameters: Mapping[str, Any],
    ) -> tuple[bool | None, dict[str, Any]]:
        names = sample.get("joint_names")
        if not isinstance(names, Sequence) or isinstance(names, (str, bytes)):
            return None, {"reason": "joint_names_unavailable"}
        names = [str(name) for name in names]
        count = len(names)
        position = _finite_vector(sample.get("joint_positions"), count)
        velocity = _finite_vector(sample.get("joint_velocities"), count)
        effort = _finite_vector(sample.get("joint_efforts"), count)
        # Revolute wheel joints can be formally unbounded.  Infinite position
        # bounds are valid evidence and simply cannot be exceeded; NaN remains
        # unavailable and therefore fails closed to an unknown observation.
        lower = _limit_vector(sample.get("joint_lower_limits"), count)
        upper = _limit_vector(sample.get("joint_upper_limits"), count)
        prismatic = sample.get("prismatic_joint_names")
        if any(value is None for value in (position, velocity, effort, lower, upper)):
            return None, {"reason": "joint_state_or_limits_unavailable"}
        if not isinstance(prismatic, Sequence) or isinstance(prismatic, (str, bytes)):
            return None, {"reason": "joint_type_inventory_unavailable"}
        prismatic_names = {str(name) for name in prismatic}
        if parameters["velocity_limit_source"] == "robot_profile":
            limits = _finite_vector(sample.get("joint_velocity_limits"), count)
            if limits is None or any(limit <= 0.0 for limit in limits):
                return None, {"reason": "robot_profile_velocity_limits_unavailable"}
            velocity_limits = dict(zip(names, limits, strict=True))
        else:
            explicit = dict(parameters["maximum_velocity_by_joint"])
            missing = sorted(set(names) - set(explicit))
            if missing:
                return None, {"reason": "explicit_velocity_limits_missing", "joints": missing}
            velocity_limits = {name: float(explicit[name]) for name in names}

        if parameters["effort_limit_source"] == "robot_profile":
            limits = _finite_vector(sample.get("joint_effort_limits"), count)
            if limits is None or any(limit <= 0.0 for limit in limits):
                return None, {"reason": "robot_profile_effort_limits_unavailable"}
            effort_limits = dict(zip(names, limits, strict=True))
        else:
            explicit = dict(parameters["maximum_effort_by_joint"])
            missing = sorted(set(names) - set(explicit))
            if missing:
                return None, {
                    "reason": "explicit_effort_limits_missing",
                    "joints": missing,
                }
            effort_limits = {name: float(explicit[name]) for name in names}

        position_violations = []
        velocity_violations = []
        effort_violations = []
        for index, name in enumerate(names):
            tolerance = float(
                parameters[
                    "prismatic_position_tolerance_m"
                    if name in prismatic_names
                    else "revolute_position_tolerance_rad"
                ]
            )
            if position[index] < lower[index] - tolerance or position[index] > upper[index] + tolerance:
                position_violations.append({
                    "joint": name,
                    "position": position[index],
                    "lower": lower[index],
                    "upper": upper[index],
                    "tolerance": tolerance,
                    "unit": "m" if name in prismatic_names else "rad",
                })
            limit = velocity_limits[name]
            if abs(velocity[index]) > limit:
                velocity_violations.append({
                    "joint": name,
                    "velocity": velocity[index],
                    "limit": limit,
                    "unit": "m/s" if name in prismatic_names else "rad/s",
                })
            effort_limit = effort_limits[name]
            if abs(effort[index]) > effort_limit:
                effort_violations.append({
                    "joint": name,
                    "effort": effort[index],
                    "limit": effort_limit,
                    "unit": "N" if name in prismatic_names else "N*m",
                })
        return bool(position_violations or velocity_violations or effort_violations), {
            "position_violations": position_violations,
            "velocity_violations": velocity_violations,
            "effort_violations": effort_violations,
            "velocity_limit_source": parameters["velocity_limit_source"],
            "effort_limit_source": parameters["effort_limit_source"],
        }

    def _evaluate_s12(
        self,
        sample: Mapping[str, Any],
        parameters: Mapping[str, Any],
    ) -> tuple[bool | None, dict[str, Any]]:
        if not parameters["objects"]:
            return None, {
                "reason": "configuration_missing",
                "missing_field": "objects",
                "configured_count": 0,
            }
        observations = []
        violations = []
        for spec in parameters["objects"]:
            prim_path = str(spec["prim_path"])
            initial = self.initial_object_states.get(prim_path)
            current = _object_state(sample, prim_path)
            if initial is None or current is None:
                return None, {
                    "reason": "protected_pose_reference_unavailable",
                    "prim_path": prim_path,
                }
            displacement = _norm([
                current["pose"][index] - initial["pose"][index]
                for index in range(3)
            ])
            rotation = _quaternion_angle_deg(initial["pose"][3:], current["pose"][3:])
            violated = bool(
                displacement > float(spec["maximum_displacement_m"])
                or rotation > float(spec["maximum_rotation_deg"])
            )
            row = {
                "prim_path": prim_path,
                "reference": spec["reference"],
                "displacement_m": displacement,
                "maximum_displacement_m": float(spec["maximum_displacement_m"]),
                "rotation_deg": rotation,
                "maximum_rotation_deg": float(spec["maximum_rotation_deg"]),
                "initial_pose": initial["pose"],
                "current_pose": current["pose"],
                "violated": violated,
            }
            observations.append(row)
            if violated:
                violations.append(row)
        return bool(violations), {
            "objects": observations,
            "violations": violations,
        }

    def _evaluate_s13(
        self,
        sample: Mapping[str, Any],
        parameters: Mapping[str, Any],
    ) -> tuple[bool | None, dict[str, Any]]:
        observations = []
        violations = []
        for spec in parameters["objects"]:
            prim_path = str(spec["prim_path"])
            initial = self.initial_object_states.get(prim_path)
            current = _object_state(sample, prim_path)
            if initial is None or current is None:
                return None, {
                    "reason": "tilt_reference_unavailable",
                    "prim_path": prim_path,
                }
            initial_z = _local_z_world(initial["pose"][3:])
            current_z = _local_z_world(current["pose"][3:])
            dot = max(
                -1.0,
                min(
                    1.0,
                    sum(
                        left * right
                        for left, right in zip(initial_z, current_z, strict=True)
                    ),
                ),
            )
            tilt_deg = math.degrees(math.acos(dot))
            maximum = float(spec["maximum_tilt_deg"])
            violated = tilt_deg > maximum
            row = {
                "prim_path": prim_path,
                "reference": spec["reference"],
                "initial_local_z_world": initial_z,
                "current_local_z_world": current_z,
                "tilt_deg": tilt_deg,
                "maximum_tilt_deg": maximum,
                "comparison": "gt",
                "initial_pose": initial["pose"],
                "current_pose": current["pose"],
                "violated": violated,
            }
            observations.append(row)
            if violated:
                violations.append(row)
        return bool(violations), {
            "objects": observations,
            "violations": violations,
        }

    def _evaluate_s14(
        self,
        sample: Mapping[str, Any],
        parameters: Mapping[str, Any],
    ) -> tuple[bool | None, dict[str, Any]]:
        contacts = sample.get("contacts")
        robot_root = sample.get("robot_root_prim")
        if not isinstance(contacts, Sequence) or isinstance(contacts, (str, bytes)):
            return None, {"reason": "contacts_unavailable"}
        if not isinstance(robot_root, str) or not robot_root.startswith("/"):
            return None, {"reason": "robot_root_unavailable"}
        observations = []
        violations = []
        for spec in parameters["placements"]:
            criterion = str(spec["criterion"])
            object_path = str(spec["object_prim_path"])
            forbidden_prefixes = tuple(spec["forbidden_prim_prefixes"])
            configured_grippers = set(spec["gripper_body_names"])
            gripper_contacts: set[str] = set()
            forbidden_contacts = []
            object_contacts = []
            for raw in contacts:
                if not isinstance(raw, Mapping) or not bool(raw.get("touching", False)):
                    continue
                matches0 = _contact_matches_prefix(raw, 0, object_path)
                matches1 = _contact_matches_prefix(raw, 1, object_path)
                if matches0 == matches1:
                    continue
                object_side = 0 if matches0 else 1
                other = 1 - object_side
                other_body = str(raw.get(f"body{other}", ""))
                other_collider = str(raw.get(f"collider{other}", ""))
                object_contacts.append(dict(raw))
                if (
                    _under(other_body, robot_root)
                    and _body_name(other_body) in configured_grippers
                ):
                    gripper_contacts.add(_body_name(other_body))
                if any(
                    _under(other_body, prefix) or _under(other_collider, prefix)
                    for prefix in forbidden_prefixes
                ):
                    forbidden_contacts.append(dict(raw))
            fully_gripped = configured_grippers.issubset(gripper_contacts)
            if fully_gripped:
                self.s14_grasp_observed.add(criterion)
            grasp_observed = criterion in self.s14_grasp_observed
            eligible = bool(spec["include_initial_placement"] or grasp_observed)
            violated = bool(forbidden_contacts and not fully_gripped and eligible)
            row = {
                "criterion": criterion,
                "object_prim_path": object_path,
                "forbidden_prim_prefixes": list(forbidden_prefixes),
                "configured_gripper_body_names": sorted(configured_grippers),
                "contacting_gripper_body_names": sorted(gripper_contacts),
                "fully_gripped": fully_gripped,
                "grasp_observed_since_task_start": grasp_observed,
                "include_initial_placement": bool(spec["include_initial_placement"]),
                "placement_eligible": eligible,
                "forbidden_contacts": forbidden_contacts,
                "object_contacts": object_contacts,
                "violated": violated,
            }
            observations.append(row)
            if violated:
                violations.append(row)
        return bool(violations), {
            "placements": observations,
            "violations": violations,
            "placement_semantics": (
                "actual_forbidden_contact_without_all_configured_gripper_contacts"
            ),
        }

    def _update(
        self,
        rule_id: str,
        status: bool | None,
        metrics: dict[str, Any],
        point: dict[str, Any],
    ) -> None:
        state = self.states[rule_id]
        if status is None:
            if state.active_start is not None:
                self._close_violation(rule_id, state, point, "observation_gap")
            state.candidate_start = None
            if state.unknown_start is None:
                state.unknown_start = {**point, "metrics": metrics}
            return

        state.observed_steps += 1
        if state.unknown_start is not None:
            self._close_unknown(rule_id, state, point, "observation_resumed")
        if status:
            state.active_metrics = metrics
            if state.active_start is None:
                state.active_start = dict(point)
        else:
            state.candidate_start = None
            if state.active_start is not None:
                self._close_violation(rule_id, state, point, "cleared")

    def _close_violation(
        self,
        rule_id: str,
        state: RuleState,
        point: dict[str, Any],
        reason: str,
    ) -> None:
        if state.active_start is None:
            return
        event = {
            "schema": EVENT_SCHEMA,
            "kind": "violation_interval",
            "rule_id": rule_id,
            "start": state.active_start,
            "end": point,
            "end_reason": reason,
            "last_metrics": state.active_metrics,
        }
        self._append_event(event)
        state.violation_count += 1
        state.active_start = None
        state.active_metrics = None
        state.candidate_start = None

    def _close_unknown(
        self,
        rule_id: str,
        state: RuleState,
        point: dict[str, Any],
        reason: str,
    ) -> None:
        if state.unknown_start is None:
            return
        event = {
            "schema": EVENT_SCHEMA,
            "kind": "observation_gap",
            "rule_id": rule_id,
            "start": state.unknown_start,
            "end": point,
            "end_reason": reason,
        }
        self._append_event(event)
        state.unknown_interval_count += 1
        state.unknown_start = None

    def _append_event(self, event: Mapping[str, Any]) -> None:
        with self.events_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")
