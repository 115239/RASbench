"""Safety competition ScoringModules over the public v2 interfaces."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from robolab_action_judge.package import (
    V06_PACKAGE_SCHEMA,
    V07_PACKAGE_SCHEMA,
    V08_PACKAGE_SCHEMA,
)
from robolab_safety_tianchi.attack_ai_config import (
    create_deployment_attack_semantic_judge,
)
from robolab_safety_tianchi.attack_descriptions import (
    AttackDescriptionProvider,
    BundledRubricAttackDescriptionProvider,
)
from robolab_safety_tianchi.attack_handling import (
    ATTACK_HANDLING_SCORING_SCHEMA,
    AttackHandlingScoreStatus,
)
from robolab_safety_tianchi.attack_judging import (
    AttackSemanticJudge,
    evaluate_attack_handling_submission,
)
from robolab_safety_tianchi.attack_semantic import AttackSemanticJudgeError
from robolab_safety_tianchi.physical_safety import (
    PHYSICAL_SAFETY_SCORING_SCHEMA,
    PhysicalSafetyScoreStatus,
    evaluate_physical_safety_submission,
)
from robolab_safety_tianchi.task_execution import (
    TASK_EXECUTION_SCORING_SCHEMA,
    TASK_EXECUTION_V08_SCORING_SCHEMA,
)

from v2.common.judge_core import evaluate_raw_facts
from v2.common.hdf5_preflight import validate_submission_identity
from v2.common.rule_packages import materialize_rule_package
from v2.common.tianchi import ModuleResult, ModuleStatus, TianchiFrameworkError
from v2.safety.attack_protocol import (
    validate_safety_attack_protocol_dependency_v1,
)
from v2.safety.judge_scorer import score_safety_judge


RULE_PACKAGE_RESOURCE = "rule_package"
EXPECTED_TASK_ID_RESOURCE = "expected_task_id"
EXPECTED_ROBOT_ROLE_RESOURCE = "expected_robot_role"
WEIGHT_POLICY_RESOURCE = "weight_policy"
ATTACK_DESCRIPTION_PROVIDER_RESOURCE = "attack_description_provider"
ATTACK_SEMANTIC_JUDGE_RESOURCE = "attack_semantic_judge"
SAFETY_ROBOT_ROLE = "m20_piper"


class SafetyJudgeModule:
    """Public Judge Core followed by the safety task-execution scorer."""

    module_id = "task_execution"

    def __init__(self, trusted_resources: Mapping[str, Any]) -> None:
        self._package, self._task_id, self._robot_role = _common_resources(
            trusted_resources
        )
        weight_policy = trusted_resources.get(WEIGHT_POLICY_RESOURCE)
        if weight_policy is not None and not isinstance(weight_policy, Mapping):
            raise TianchiFrameworkError("weight_policy must be a mapping")
        self._weight_policy = (
            dict(weight_policy) if isinstance(weight_policy, Mapping) else None
        )
        self.module_version = (
            TASK_EXECUTION_V08_SCORING_SCHEMA
            if self._package.get("package_schema")
            in {V06_PACKAGE_SCHEMA, V07_PACKAGE_SCHEMA, V08_PACKAGE_SCHEMA}
            else TASK_EXECUTION_SCORING_SCHEMA
        )

    def evaluate(self, hdf5_path: Path) -> ModuleResult:
        raw_facts = evaluate_raw_facts(
            self._package,
            hdf5_path,
            expected_task_id=self._task_id,
            expected_robot_role=self._robot_role,
        )
        return score_safety_judge(
            self._package,
            raw_facts,
            self._weight_policy,
        )


class SafetyPhysicalSafetyModule:
    """Adapter around the released deterministic physical-safety scorer."""

    module_id = "physical_safety"
    module_version = PHYSICAL_SAFETY_SCORING_SCHEMA

    def __init__(self, trusted_resources: Mapping[str, Any]) -> None:
        self._package, self._task_id, self._robot_role = _common_resources(
            trusted_resources
        )

    def evaluate(self, hdf5_path: Path) -> ModuleResult:
        validate_submission_identity(
            hdf5_path,
            expected_task_id=self._task_id,
            expected_robot_role=self._robot_role,
        )
        score = evaluate_physical_safety_submission(
            self._package,
            hdf5_path,
            expected_task_id=self._task_id,
        )
        status = (
            ModuleStatus.EVALUATED
            if score.status is PhysicalSafetyScoreStatus.EVALUATED
            else ModuleStatus.UNDER_REVIEW
        )
        return ModuleResult(
            module_id=self.module_id,
            module_version=score.scoring_schema,
            status=status,
            score_ratio=score.score_ratio,
            report={
                "policy_id": score.policy_id,
                "policy_digest": score.policy_digest,
                "task_entity_ids": score.task_entity_ids,
                "candidate_evidence": tuple(score.candidate_evidence),
                "assessments": tuple(
                    _physical_assessment_report(item)
                    for item in score.assessments
                ),
                "reason_code": score.reason_code,
                "issues": tuple(score.issues),
            },
        )


class SafetyAttackHandlingModule:
    """Adapter around the released attack-occurrence and response scorer."""

    module_id = "attack_handling"
    module_version = ATTACK_HANDLING_SCORING_SCHEMA

    def __init__(self, trusted_resources: Mapping[str, Any]) -> None:
        self._attack_protocol = validate_safety_attack_protocol_dependency_v1()
        if not isinstance(trusted_resources, Mapping):
            raise TianchiFrameworkError("trusted_resources must be a mapping")
        self._task_id = _required_text(
            trusted_resources.get(EXPECTED_TASK_ID_RESOURCE),
            EXPECTED_TASK_ID_RESOURCE,
        )
        self._robot_role = _required_text(
            trusted_resources.get(EXPECTED_ROBOT_ROLE_RESOURCE),
            EXPECTED_ROBOT_ROLE_RESOURCE,
        )
        if self._robot_role != SAFETY_ROBOT_ROLE:
            raise TianchiFrameworkError(
                "safety scoring requires robot_role {!r}".format(
                    SAFETY_ROBOT_ROLE
                )
            )
        provider = trusted_resources.get(ATTACK_DESCRIPTION_PROVIDER_RESOURCE)
        if provider is not None and not isinstance(
            provider, AttackDescriptionProvider
        ):
            raise TianchiFrameworkError(
                "attack_description_provider has an invalid type"
            )
        semantic_judge = trusted_resources.get(ATTACK_SEMANTIC_JUDGE_RESOURCE)
        if semantic_judge is not None and not isinstance(
            semantic_judge, AttackSemanticJudge
        ):
            raise TianchiFrameworkError(
                "attack_semantic_judge has an invalid type"
            )
        self._provider = provider
        self._semantic_judge = semantic_judge

    def evaluate(self, hdf5_path: Path) -> ModuleResult:
        validate_submission_identity(
            hdf5_path,
            expected_task_id=self._task_id,
            expected_robot_role=self._robot_role,
        )
        provider = self._provider
        if provider is None:
            provider = BundledRubricAttackDescriptionProvider.for_submission(
                self._task_id,
                hdf5_path,
            )
        semantic_judge = self._semantic_judge
        if semantic_judge is None:
            semantic_judge = create_deployment_attack_semantic_judge()
        try:
            evaluation = evaluate_attack_handling_submission(
                self._task_id,
                hdf5_path,
                provider,
                semantic_judge,
            )
        except AttackSemanticJudgeError as error:
            return ModuleResult(
                module_id=self.module_id,
                module_version=self.module_version,
                status=ModuleStatus.UNDER_REVIEW,
                score_ratio=None,
                report={
                    "reason_code": "ATTACK_SEMANTIC_AI_CONTRACT_FAILED",
                    "message": str(error),
                },
            )
        score = evaluation.score
        status = (
            ModuleStatus.EVALUATED
            if score.status is AttackHandlingScoreStatus.EVALUATED
            else ModuleStatus.UNDER_REVIEW
        )
        return ModuleResult(
            module_id=self.module_id,
            module_version=score.scoring_schema,
            status=status,
            score_ratio=score.score_ratio,
            report={
                "scoring_schema": score.scoring_schema,
                "attack_protocol": {
                    "name": self._attack_protocol.protocol_name,
                    "primary_version": self._attack_protocol.primary_version,
                    "compatible_versions": (
                        self._attack_protocol.compatible_versions
                    ),
                    "reader_view_schema": (
                        self._attack_protocol.reader_view_schema
                    ),
                },
                "occurrence_schema": evaluation.occurrence.schema,
                "evidence_schema": evaluation.evidence.schema,
                "planned_attack_count": score.planned_attack_count,
                "applied_attack_count": score.applied_attack_count,
                "anomaly_ratio": score.anomaly_ratio,
                "response_ratio": score.response_ratio,
                "contributions": tuple(
                    _attack_contribution_report(item)
                    for item in score.contributions
                ),
            },
        )


def _common_resources(
    trusted_resources: Mapping[str, Any],
) -> tuple[Mapping[str, Any], str, str]:
    if not isinstance(trusted_resources, Mapping):
        raise TianchiFrameworkError("trusted_resources must be a mapping")
    if RULE_PACKAGE_RESOURCE not in trusted_resources:
        raise TianchiFrameworkError("module requires trusted rule_package")
    package = materialize_rule_package(trusted_resources[RULE_PACKAGE_RESOURCE])
    if package.get("package_schema") not in {
        V06_PACKAGE_SCHEMA,
        V07_PACKAGE_SCHEMA,
        V08_PACKAGE_SCHEMA,
    }:
        raise TianchiFrameworkError(
            "safety module requires Rule Package v0.6, v0.7, or v0.8"
        )
    task_id = _required_text(
        trusted_resources.get(EXPECTED_TASK_ID_RESOURCE),
        EXPECTED_TASK_ID_RESOURCE,
    )
    robot_role = _required_text(
        trusted_resources.get(EXPECTED_ROBOT_ROLE_RESOURCE),
        EXPECTED_ROBOT_ROLE_RESOURCE,
    )
    if robot_role != SAFETY_ROBOT_ROLE:
        raise TianchiFrameworkError(
            "safety scoring requires robot_role {!r}".format(
                SAFETY_ROBOT_ROLE
            )
        )
    if package.get("task_id") != task_id:
        raise TianchiFrameworkError(
            "expected_task_id does not match safety Rule Package"
        )
    return package, task_id, robot_role


def _required_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise TianchiFrameworkError(
            "{} must be normalized non-empty text".format(name)
        )
    return value


def _physical_assessment_report(value: object) -> Mapping[str, object]:
    violation = value.violation
    return {
        "task_entity_id": value.task_entity_id,
        "contact_path": value.contact_path,
        "sample_count": value.sample_count,
        "peak_force_n": value.peak_force_n,
        "violation": None
        if violation is None
        else {
            "body_index": violation.body_index,
            "body_path": violation.body_path,
            "sample_index": violation.sample_index,
            "step_index": violation.step_index,
            "sim_time_s": violation.sim_time_s,
            "force_xyz_n": violation.force_xyz_n,
            "force_norm_n": violation.force_norm_n,
        },
    }


def _attack_contribution_report(value: object) -> Mapping[str, object]:
    return {
        "dimension_code": value.dimension_code,
        "attack_id": value.attack_id,
        "status": value.status.value,
        "anomaly_value": value.anomaly_value,
        "response_value": value.response_value,
        "score_ratio": value.score_ratio,
        "report": dict(value.report),
    }


__all__ = [
    "ATTACK_DESCRIPTION_PROVIDER_RESOURCE",
    "ATTACK_SEMANTIC_JUDGE_RESOURCE",
    "EXPECTED_ROBOT_ROLE_RESOURCE",
    "EXPECTED_TASK_ID_RESOURCE",
    "RULE_PACKAGE_RESOURCE",
    "SAFETY_ROBOT_ROLE",
    "WEIGHT_POLICY_RESOURCE",
    "SafetyAttackHandlingModule",
    "SafetyJudgeModule",
    "SafetyPhysicalSafetyModule",
]
