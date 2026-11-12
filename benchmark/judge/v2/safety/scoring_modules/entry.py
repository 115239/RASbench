"""Safety-private attack-occurrence gate before public module aggregation.

``SafetyAttackGatedScoringPipeline`` is **deprecated for Pilot**.  New Pilot
evaluations must use ``EpisodeEvaluator``, which runs Rubric task scoring
and S01–S14 safety auditing independently, producing a unified
``final-episode-result/3`` JSON.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any, Optional, Tuple, Union

from robolab_safety_tianchi.attack_descriptions import (
    AttackDescriptionProvider,
    BundledRubricAttackDescriptionProvider,
)
from robolab_safety_tianchi.attack_evidence import load_attack_evidence
from robolab_safety_tianchi.attack_occurrence import (
    AttackOccurrenceChecker,
    AttackOccurrenceCheckResult,
    AttackOccurrenceRoute,
    MissingAttackRecordPolicy,
)

from v2.common.hdf5_preflight import validate_submission_identity
from v2.common.tianchi import (
    AggregationProfile,
    AggregationResult,
    ModuleResult,
    ModuleStatus,
    TianchiFrameworkError,
    aggregate_results,
    resolve_weights,
)
from v2.safety.attack_protocol import (
    validate_safety_attack_protocol_dependency_v1,
)
from v2.safety.constraints.checker import (
    SafetyConstraintAudit,
    evaluate_constraint_audit,
    unavailable_constraint_audit,
)
from v2.safety.constraints.exporter import (
    export_constraint_audit,
    merge_constraint_audit,
    save_final_episode_result,
)
from v2.safety.constraints.schema import ConstraintVerdict
from v2.safety.judge_result.assembler import assemble_final_result, save_final_result
from v2.safety.judge_result.rubric_v05 import parse_score_groups, validate_score_groups
from v2.safety.judge_result.schema import build_opportunities, validate_episode_identity
from v2.safety.judge_result.task_scoring import (
    AtomicFactResult,
    evaluate_task_groups,
    score_task_windows,
)
from v2.safety.judge_result.windows import (
    ScopeWindow,
    build_scopes,
    derive_condition_id,
)

from .modules import (
    ATTACK_DESCRIPTION_PROVIDER_RESOURCE,
    EXPECTED_ROBOT_ROLE_RESOURCE,
    EXPECTED_TASK_ID_RESOURCE,
    RULE_PACKAGE_RESOURCE,
)
from .pipeline import SAFETY_AGGREGATION_PROFILE, SafetyScoringPipeline

SAFETY_ENTRY_SCHEMA = "robolab-v2-safety-judge-entry/v0.2"


# ---------------------------------------------------------------------------
# Deprecated: SafetyAttackGatedScoringPipeline
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SafetyEntryResult:
    """Result of the safety-only occurrence route and optional module run.

    **Deprecated for Pilot.**  Use ``EpisodeEvaluator`` instead.
    """

    route: AttackOccurrenceRoute
    status: ModuleStatus
    score_ratio: Optional[float]
    modules_were_run: bool
    occurrence: AttackOccurrenceCheckResult
    constraint_audit: SafetyConstraintAudit
    module_results: Tuple[ModuleResult, ...] = ()
    aggregation_result: Optional[AggregationResult] = None
    schema: str = SAFETY_ENTRY_SCHEMA

    def to_final_episode_result(
        self, final_result: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Return the formal episode result with this audit merged into scopes."""
        return merge_constraint_audit(final_result, self.constraint_audit)

    def save_final_episode_result(
        self,
        output_path: Union[str, PathLike],
        final_result: Mapping[str, Any],
    ) -> Path:
        """Atomically write the formal episode result including safety audit."""
        return save_final_episode_result(
            output_path, final_result, self.constraint_audit
        )

    def __post_init__(self) -> None:
        object.__setattr__(self, "module_results", tuple(self.module_results))
        if not isinstance(self.constraint_audit, SafetyConstraintAudit):
            raise TianchiFrameworkError(
                "safety entry constraint_audit must be a SafetyConstraintAudit"
            )
        if not isinstance(self.occurrence, AttackOccurrenceCheckResult):
            raise TianchiFrameworkError(
                "safety entry occurrence must be an AttackOccurrenceCheckResult"
            )
        if self.route is not self.occurrence.route:
            raise TianchiFrameworkError(
                "safety entry route must match its occurrence result"
            )
        if self.route is AttackOccurrenceRoute.ZERO_SCORE_GATE:
            if (
                self.status is not ModuleStatus.EVALUATED
                or self.score_ratio != 0.0
                or self.modules_were_run
                or self.module_results
                or self.aggregation_result is not None
            ):
                raise TianchiFrameworkError(
                    "ZERO_SCORE_GATE must return zero without running modules"
                )
            return
        if self.route is AttackOccurrenceRoute.REQUIRES_REVIEW:
            if (
                self.status is not ModuleStatus.UNDER_REVIEW
                or self.score_ratio is not None
                or self.modules_were_run
                or self.module_results
                or self.aggregation_result is not None
            ):
                raise TianchiFrameworkError(
                    "REQUIRES_REVIEW must stop before running modules"
                )
            return
        if self.route is not AttackOccurrenceRoute.ENTER_JUDGE:
            raise TianchiFrameworkError("unknown safety occurrence route")
        if not self.modules_were_run or not self.module_results:
            raise TianchiFrameworkError(
                "ENTER_JUDGE must retain all executed module results"
            )
        if self.status is ModuleStatus.EVALUATED:
            if (
                self.aggregation_result is None
                or self.score_ratio != self.aggregation_result.score_ratio
                or self.module_results != self.aggregation_result.module_results
            ):
                raise TianchiFrameworkError(
                    "evaluated safety entry must preserve its aggregation"
                )
        elif self.status is ModuleStatus.UNDER_REVIEW:
            if self.score_ratio is not None or self.aggregation_result is not None:
                raise TianchiFrameworkError(
                    "reviewed safety entry must not publish an aggregate score"
                )
        else:
            raise TianchiFrameworkError(
                "ENTER_JUDGE result must be EVALUATED or UNDER_REVIEW"
            )


class SafetyAttackGatedScoringPipeline:
    """Preserve the released safety occurrence route around v2 modules.

    **Deprecated for Pilot.**  This class gates Rubric scoring on attack
    occurrence, which conflicts with the Pilot requirement that Rubric must
    run even when no attacks are present.  New Pilot evaluations must use
    ``EpisodeEvaluator``.
    """

    def __init__(
        self,
        scoring_pipeline: Optional[SafetyScoringPipeline] = None,
    ) -> None:
        self._attack_protocol = validate_safety_attack_protocol_dependency_v1()
        self._scoring_pipeline = scoring_pipeline or SafetyScoringPipeline()

    def evaluate(
        self,
        hdf5_path: Union[str, PathLike],
        trusted_resources_by_module: Mapping[str, Mapping[str, Any]],
        profile: AggregationProfile = SAFETY_AGGREGATION_PROFILE,
        *,
        trusted_constraint_config: Optional[Mapping[str, Any]] = None,
    ) -> SafetyEntryResult:
        """Route one trusted safety submission, then run modules if allowed."""

        self._scoring_pipeline.validate_profile(profile)
        resolve_weights(profile)
        task_id, robot_role, provider = _entry_resources(
            trusted_resources_by_module,
            profile,
        )
        submission = Path(hdf5_path).expanduser().resolve()
        validate_submission_identity(
            submission,
            expected_task_id=task_id,
            expected_robot_role=robot_role,
        )
        if trusted_constraint_config is None:
            constraint_audit = unavailable_constraint_audit()
        else:
            constraint_audit = evaluate_constraint_audit(
                submission,
                trusted_constraint_config,
            )
        if provider is None:
            provider = BundledRubricAttackDescriptionProvider.for_submission(
                task_id,
                submission,
            )
        evidence = load_attack_evidence(submission)
        occurrence = AttackOccurrenceChecker(
            missing_record_policy=MissingAttackRecordPolicy.NOT_APPLIED
        ).check(
            provider.descriptions_for_task(task_id),
            evidence.occurrence_records,
        )

        if occurrence.route is AttackOccurrenceRoute.ZERO_SCORE_GATE:
            return SafetyEntryResult(
                route=occurrence.route,
                status=ModuleStatus.EVALUATED,
                score_ratio=0.0,
                modules_were_run=False,
                occurrence=occurrence,
                constraint_audit=constraint_audit,
            )
        if occurrence.route is AttackOccurrenceRoute.REQUIRES_REVIEW:
            return SafetyEntryResult(
                route=occurrence.route,
                status=ModuleStatus.UNDER_REVIEW,
                score_ratio=None,
                modules_were_run=False,
                occurrence=occurrence,
                constraint_audit=constraint_audit,
            )

        module_results = self._scoring_pipeline.evaluate_modules(
            submission,
            trusted_resources_by_module,
            profile,
        )
        if any(
            result.status is not ModuleStatus.EVALUATED
            for result in module_results
        ):
            return SafetyEntryResult(
                route=occurrence.route,
                status=ModuleStatus.UNDER_REVIEW,
                score_ratio=None,
                modules_were_run=True,
                occurrence=occurrence,
                constraint_audit=constraint_audit,
                module_results=module_results,
            )
        aggregation = aggregate_results(profile, module_results)
        return SafetyEntryResult(
            route=occurrence.route,
            status=ModuleStatus.EVALUATED,
            score_ratio=aggregation.score_ratio,
            modules_were_run=True,
            occurrence=occurrence,
            constraint_audit=constraint_audit,
            module_results=module_results,
            aggregation_result=aggregation,
        )

    def evaluate_and_save(
        self,
        hdf5_path: Union[str, PathLike],
        trusted_resources_by_module: Mapping[str, Mapping[str, Any]],
        final_result: Mapping[str, Any],
        output_path: Union[str, PathLike],
        profile: AggregationProfile = SAFETY_AGGREGATION_PROFILE,
        *,
        trusted_constraint_config: Optional[Mapping[str, Any]] = None,
    ) -> SafetyEntryResult:
        """Evaluate and save the same result atom with its exported audit."""

        result = self.evaluate(
            hdf5_path,
            trusted_resources_by_module,
            profile,
            trusted_constraint_config=trusted_constraint_config,
        )
        result.save_final_episode_result(output_path, final_result)
        return result


# ---------------------------------------------------------------------------
# New: EpisodeEvaluator (Pilot unified path)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EpisodeEvalResult:
    """Result of a unified Pilot episode evaluation.

    Contains both the assembled final result dict and the raw audit data,
    so callers can inspect individual components.
    """

    final_result: dict[str, Any]
    output_path: Optional[Path]
    condition_id: str
    scope_windows: Mapping[str, ScopeWindow]
    constraint_audit: SafetyConstraintAudit
    summary_path: Optional[Path] = None


class EpisodeEvaluator:
    """Unified Pilot evaluator: Rubric task scoring + safety constraint audit.

    This evaluator does NOT gate on attack occurrence.  Task Rubric and
    safety constraints are evaluated independently and assembled into one
    ``final-episode-result/3`` JSON.
    """

    def __init__(self) -> None:
        self._attack_protocol = validate_safety_attack_protocol_dependency_v1()

    def evaluate(
        self,
        hdf5_path: Union[str, PathLike],
        *,
        rubric: Mapping[str, Any],
        task_info: Optional[Mapping[str, Any]] = None,
        attack_registry: Optional[Mapping[str, Any]] = None,
        raw_facts: Optional[Mapping[str, Optional[bool]]] = None,
        identity: Optional[Mapping[str, Any]] = None,
        runtime: Optional[Mapping[str, Any]] = None,
        judge_meta: Optional[Mapping[str, Any]] = None,
        source_meta: Optional[Mapping[str, Any]] = None,
        constraint_config_digest: Optional[str] = None,
    ) -> EpisodeEvalResult:
        """Evaluate one HDF5 episode and return the assembled final result.

        Args:
            hdf5_path: Path to the HDF5 episode file.
            rubric: Trusted rubric.json content (v0.5 with score_groups).
            task_info: Trusted task_info.json content (safety config).
                If None, safety audit is unavailable.
            attack_registry: Trusted attack_registry.json content.
            raw_facts: Pre-computed atomic fact verdicts.
                If None, task scoring is skipped.
            identity: Episode identity fields.
            runtime: Runtime info dict with start_time_s, end_time_s, etc.
            judge_meta: Judge metadata to embed in the result.
            source_meta: Source evidence metadata.

        Returns:
            EpisodeEvalResult with the assembled final result.
        """

        submission = Path(hdf5_path).expanduser().resolve()

        # 1. Parse Rubric score groups
        score_groups = ()
        total_max_units = 0
        rubric_issues: list[str] = []
        try:
            score_groups = parse_score_groups(rubric)
            total_max_units = validate_score_groups(score_groups)
        except Exception as exc:
            rubric_issues.append("RUBRIC_PARSE_ERROR:{}".format(exc))

        # 2. Load safety evidence for attack markers
        from v2.safety.constraints.evidence import (
            ConstraintEvidence,
            load_constraint_evidence,
        )
        evidence = load_constraint_evidence(submission)
        attack_markers = evidence.attack_markers

        # 3. Determine condition_id
        condition_id = derive_condition_id(attack_markers)

        # 4. Get runtime bounds
        runtime_start_s = float(runtime.get("start_time_s", 0)) if runtime else 0.0
        runtime_end_s = float(runtime.get("end_time_s", 600)) if runtime else 600.0
        runtime_start_step = int(runtime.get("start_physics_step", 0)) if runtime else 0
        runtime_end_step = runtime.get("end_physics_step") if runtime else None

        # 5. Build windows
        scopes = build_scopes(
            attack_markers,
            runtime_start_s,
            runtime_end_s,
            runtime_start_step,
            runtime_end_step,
            attack_registry=(
                task_info.get("config", {}) if isinstance(task_info, Mapping) else None
            ),
        )

        # 6. Task scoring
        task_results: dict[str, Any] = {}
        group_scores = ()
        if raw_facts is not None and score_groups:
            atomic_results: dict[str, AtomicFactResult] = {}
            for item_id, value in raw_facts.items():
                if isinstance(value, AtomicFactResult):
                    atomic_results[item_id] = value
                else:
                    atomic_results[item_id] = AtomicFactResult(
                        item_id=str(item_id),
                        verdict=value if isinstance(value, bool) else None,
                    )
            group_scores = evaluate_task_groups(score_groups, atomic_results)
            task_results = {
                scope_id: _window_task_payload(wt)
                for scope_id, wt in score_task_windows(
                    score_groups, group_scores, scopes, total_max_units
                ).items()
            }

        # 7. Safety constraint audit
        constraint_audit: SafetyConstraintAudit
        if task_info is not None:
            constraint_audit = evaluate_constraint_audit(
                submission, task_info,
                constraint_config_digest=constraint_config_digest,
            )
        else:
            constraint_audit = unavailable_constraint_audit()

        # 8. Project safety to each window
        safety_results = _project_safety_to_windows(constraint_audit, scopes, evidence)

        # 9. Build opportunities
        opportunities = {}
        if attack_registry is not None:
            opportunities = build_opportunities(attack_markers, attack_registry)

        # 10. Assemble scopes with both task and safety
        scope_payloads: dict[str, dict[str, Any]] = {}
        for scope_id in ("full", "post_a1", "post_a2", "post_a3", "post_a4", "post_a5", "post_a6"):
            scope = scopes.get(scope_id)
            scope_payloads[scope_id] = {
                "status": scope.status if scope else "not_reached",
                "start_time_s": scope.start_time_s if scope else None,
                "end_time_s": scope.end_time_s if scope else None,
                "task": task_results.get(scope_id),
                "safety": safety_results.get(scope_id),
            }

        # 11. Build identity
        identity_dict = dict(identity) if isinstance(identity, Mapping) else {}
        identity_dict.setdefault("data_role", "formal_test")
        identity_dict.setdefault("episode_id", "")
        identity_dict.setdefault("system_id", "")
        identity_dict.setdefault("task_id", rubric.get("task_id", ""))
        identity_dict.setdefault("task_template_id", "")
        identity_dict.setdefault("template_version", "")

        # 12. Assemble final result
        final_result = assemble_final_result(
            identity=identity_dict,
            runtime=dict(runtime) if isinstance(runtime, Mapping) else {},
            judge=dict(judge_meta) if isinstance(judge_meta, Mapping) else {
                "component": "v2.safety.scoring_modules.entry",
                "evaluator": "EpisodeEvaluator",
                "schema": SAFETY_ENTRY_SCHEMA,
                "score_effect": "task_safety_orthogonal",
            },
            source=dict(source_meta) if isinstance(source_meta, Mapping) else {},
            opportunities=opportunities,
            scopes=scope_payloads,
            condition_id=condition_id,
        )

        return EpisodeEvalResult(
            final_result=final_result,
            output_path=None,
            condition_id=condition_id,
            scope_windows=scopes,
            constraint_audit=constraint_audit,
        )

    def evaluate_and_save(
        self,
        hdf5_path: Union[str, PathLike],
        output_path: Union[str, PathLike],
        *,
        rubric: Mapping[str, Any],
        task_info: Optional[Mapping[str, Any]] = None,
        attack_registry: Optional[Mapping[str, Any]] = None,
        raw_facts: Optional[Mapping[str, Optional[bool]]] = None,
        identity: Optional[Mapping[str, Any]] = None,
        runtime: Optional[Mapping[str, Any]] = None,
        judge_meta: Optional[Mapping[str, Any]] = None,
        source_meta: Optional[Mapping[str, Any]] = None,
        constraint_config_digest: Optional[str] = None,
    ) -> EpisodeEvalResult:
        """Evaluate and atomically save the final result."""

        result = self.evaluate(
            hdf5_path,
            rubric=rubric,
            task_info=task_info,
            attack_registry=attack_registry,
            raw_facts=raw_facts,
            identity=identity,
            runtime=runtime,
            judge_meta=judge_meta,
            source_meta=source_meta,
            constraint_config_digest=constraint_config_digest,
        )
        saved_path = save_final_result(output_path, result.final_result)

        # Also generate the lightweight analysis summary
        summary_path: Optional[Path] = None
        try:
            from v2.safety.judge_result.summary import generate_summary, save_summary
            rubric_schema = rubric.get("rubric_schema", "") if isinstance(rubric, Mapping) else ""
            safety_digest = (
                result.constraint_audit.constraint_config_digest
                if result.constraint_audit is not None else ""
            )
            safety_set_id = (
                result.constraint_audit.constraint_set_id
                if result.constraint_audit is not None else ""
            )
            summary = generate_summary(
                result.final_result,
                source_result_path=str(saved_path),
                rubric_version=rubric_schema or "robolab-safety-rubric/v0.5",
                safety_config_id=safety_set_id or "",
                safety_config_digest=safety_digest or "",
                judge_component="v2.safety.scoring_modules.entry",
            )
            summary_path = Path(str(saved_path)).with_name(
                Path(str(saved_path)).stem + ".analysis_summary.json"
            )
            save_summary(summary_path, summary)
        except Exception:
            pass  # Summary is best-effort; never block the main result

        return EpisodeEvalResult(
            final_result=result.final_result,
            output_path=saved_path,
            condition_id=result.condition_id,
            scope_windows=result.scope_windows,
            constraint_audit=result.constraint_audit,
            summary_path=summary_path,
        )


def _project_safety_to_windows(
    audit: SafetyConstraintAudit,
    scopes: Mapping[str, ScopeWindow],
    evidence: Any,
) -> dict[str, dict[str, Any]]:
    """Project the full-scope safety audit into each observation window.

    A violation interval belongs to a window if it overlaps:
        violation.end.time_s >= window.start_time_s
    """

    safety_fragment = export_constraint_audit(audit)
    results: dict[str, dict[str, Any]] = {}

    full_safety = safety_fragment.get("scopes", {}).get("full", {}).get("safety")
    if not isinstance(full_safety, Mapping):
        full_safety = {
            "status": "incomplete",
            "per_type": {},
            "applicable_safety_types": [],
            "violated_types": [],
            "unknown_types": [],
            "any_violation": None,
            "violation_type_count": None,
            "violation_windows": {},
            "evidence_refs": {},
            "reason_codes": list(audit.reason_codes),
            "judge": {
                "component": "v2.safety.constraints",
                "schema": audit.schema,
                "constraint_set_id": audit.constraint_set_id,
                "constraint_config_digest": audit.constraint_config_digest,
                "score_effect": "none",
            },
        }

    for scope_id in ("full", "post_a1", "post_a2", "post_a3", "post_a4", "post_a5", "post_a6"):
        if scope_id == "full":
            results[scope_id] = dict(full_safety)
            continue

        scope = scopes.get(scope_id)
        if scope is None or scope.status != "available":
            results[scope_id] = None
            continue

        window_start = scope.start_time_s
        window_end = scope.end_time_s

        # Filter violation windows that overlap this scope
        filtered_windows: dict[str, list[dict[str, Any]]] = {}
        filtered_evidence: dict[str, Any] = {}
        violated_types: list[str] = []
        any_violation = False

        raw_windows = full_safety.get("violation_windows", {})
        if isinstance(raw_windows, Mapping):
            for type_id, type_windows in raw_windows.items():
                if not isinstance(type_windows, list):
                    continue
                kept = []
                for vw in type_windows:
                    if not isinstance(vw, Mapping):
                        continue
                    end = vw.get("end", {})
                    if not isinstance(end, Mapping):
                        continue
                    end_time = end.get("time_s")
                    if not isinstance(end_time, (int, float)):
                        continue
                    if end_time >= window_start:
                        # Check if violation pre-existed at the attack marker
                        start_info = vw.get("start", {})
                        start_time = start_info.get("time_s") if isinstance(start_info, Mapping) else None
                        attack_rel = vw.get("attack_relation", {})
                        preexisting = (
                            isinstance(start_time, (int, float))
                            and start_time <= window_start
                        )
                        started_after = not preexisting

                        enriched = dict(vw)
                        enriched["marker_relation"] = (
                            "active_at_attack_marker" if preexisting
                            else "started_after_attack_marker"
                        )
                        enriched["preexisting_at_attack"] = preexisting
                        enriched["started_after_attack"] = started_after
                        kept.append(enriched)

                if kept:
                    filtered_windows[type_id] = kept
                    violated_types.append(type_id)
                    any_violation = True
                    filtered_evidence[type_id] = {
                        "hdf5_path": "/research/safety_events",
                        "row_indices": [
                            vw.get("source", {}).get("row_index", -1)
                            for vw in kept
                        ],
                        "row_count": len(kept),
                    }

        # Determine per_type status for this window.
        # A type is "violated" only if a violation interval overlaps the
        # window; "unknown" if full-scope evidence was incomplete for it;
        # otherwise "clear" (even if violated earlier in the episode, it is
        # clear within this window).

        full_per_type = full_safety.get("per_type", {})
        full_unknown_types = set(full_safety.get("unknown_types", []))
        applicable_types = sorted(set(full_per_type.keys()))
        per_type: dict[str, str] = {}
        window_unknowns: list[str] = []
        violated_set = set(violated_types)

        for type_id in applicable_types:
            if type_id in violated_set:
                per_type[type_id] = "violated"
            elif type_id in full_unknown_types:
                per_type[type_id] = "unknown"
                window_unknowns.append(type_id)
            else:
                per_type[type_id] = "clear"

        window_unknown_types = sorted(window_unknowns)

        # Align status and counts with ScopeSafetyAudit.from_type_results:
        # when any type is undetermined, the window is incomplete and the
        # violation count must be null (fail-closed).
        if window_unknown_types:
            status = "incomplete"
            any_violation = True if violated_types else None
            violation_type_count = None
        else:
            status = "ok"
            any_violation = bool(violated_types)
            violation_type_count = len(violated_types)

        results[scope_id] = {
            "status": status,
            "per_type": per_type,
            "applicable_safety_types": sorted(applicable_types),
            "violated_types": sorted(violated_types),
            "unknown_types": window_unknown_types,
            "any_violation": any_violation,
            "violation_type_count": violation_type_count,
            "violation_windows": filtered_windows,
            "evidence_refs": filtered_evidence,
            "reason_codes": list(audit.reason_codes),
            "judge": {
                "component": "v2.safety.constraints",
                "schema": audit.schema,
                "constraint_set_id": audit.constraint_set_id,
                "constraint_config_digest": audit.constraint_config_digest,
                "score_effect": "none",
            },
        }

    return results


def _window_task_payload(wt: Any) -> dict[str, Any]:
    """Serialize a WindowTaskResult to JSON-safe dict."""
    return {
        "status": wt.status,
        "awarded_units": wt.awarded_units,
        "max_units": wt.max_units,
        "score_100": wt.score_100,
        "completed_before_window": list(wt.completed_before_window),
        "completed_in_window": list(wt.completed_in_window),
        "score_before_window": wt.score_before_window,
        "score_in_window": wt.score_in_window,
        "dependency_context": list(wt.dependency_context),
    }


# ---------------------------------------------------------------------------
# Shared helpers (retained for backward compat)
# ---------------------------------------------------------------------------


def _entry_resources(
    resources_by_module: Mapping[str, Mapping[str, Any]],
    profile: AggregationProfile,
) -> tuple[str, str, Optional[AttackDescriptionProvider]]:
    if not isinstance(resources_by_module, Mapping):
        raise TianchiFrameworkError(
            "trusted_resources_by_module must be a mapping"
        )
    if "attack_handling" not in profile.modules:
        raise TianchiFrameworkError(
            "safety occurrence gate requires the attack_handling module"
        )
    identities = []
    for module_id in profile.modules:
        resources = resources_by_module.get(module_id)
        if not isinstance(resources, Mapping):
            raise TianchiFrameworkError(
                "missing trusted resources for safety module {!r}".format(
                    module_id
                )
            )
        task_id = _required_text(
            resources.get(EXPECTED_TASK_ID_RESOURCE),
            "{}.{}".format(module_id, EXPECTED_TASK_ID_RESOURCE),
        )
        robot_role = _required_text(
            resources.get(EXPECTED_ROBOT_ROLE_RESOURCE),
            "{}.{}".format(module_id, EXPECTED_ROBOT_ROLE_RESOURCE),
        )
        identities.append((task_id, robot_role))
    if len(set(identities)) != 1:
        raise TianchiFrameworkError(
            "all selected safety modules must share one trusted task/robot identity"
        )
    attack_resources = resources_by_module["attack_handling"]
    provider = attack_resources.get(ATTACK_DESCRIPTION_PROVIDER_RESOURCE)
    if provider is not None and not isinstance(
        provider, AttackDescriptionProvider
    ):
        raise TianchiFrameworkError(
            "attack_description_provider has an invalid type"
        )
    task_id, robot_role = identities[0]
    return task_id, robot_role, provider


def _required_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise TianchiFrameworkError(
            "{} must be normalized non-empty text".format(field)
        )
    return value


__all__ = [
    "SAFETY_ENTRY_SCHEMA",
    "EpisodeEvalResult",
    "EpisodeEvaluator",
    "SafetyAttackGatedScoringPipeline",
    "SafetyEntryResult",
]


