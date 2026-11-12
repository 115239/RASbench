"""Unified episode-result assembly for Pilot Rubric + Safety Constraints."""

from __future__ import annotations

from v2.safety.judge_result.schema import (
    FINAL_EPISODE_RESULT_VERSION,
    ConditionId,
    FinalEpisodeResult,
    OpportunityInfo,
    ScopeResult,
    ScopeStatus,
    ValidationStatus,
    build_opportunities,
    validate_episode_identity,
)
from v2.safety.judge_result.rubric_v05 import (
    SAFETY_RUBRIC_V05_SCHEMA,
    ScoreGroup,
    parse_score_groups,
    validate_score_groups,
)
from v2.safety.judge_result.windows import (
    ScopeWindow,
    build_scopes,
)
from v2.safety.judge_result.task_scoring import (
    AtomicFactResult,
    GroupScore,
    WindowTaskResult,
    evaluate_task_groups,
    score_task_windows,
)
from v2.safety.judge_result.assembler import (
    assemble_final_result,
    save_final_result,
)

__all__ = [
    # schema
    "ConditionId",
    "FINAL_EPISODE_RESULT_VERSION",
    "FinalEpisodeResult",
    "OpportunityInfo",
    "ScopeResult",
    "ScopeStatus",
    "ValidationStatus",
    "build_opportunities",
    "validate_episode_identity",
    # rubric_v05
    "SAFETY_RUBRIC_V05_SCHEMA",
    "ScoreGroup",
    "parse_score_groups",
    "validate_score_groups",
    # windows
    "ScopeWindow",
    "build_scopes",
    # task_scoring
    "AtomicFactResult",
    "GroupScore",
    "WindowTaskResult",
    "evaluate_task_groups",
    "score_task_windows",
    # assembler
    "assemble_final_result",
    "save_final_result",
]