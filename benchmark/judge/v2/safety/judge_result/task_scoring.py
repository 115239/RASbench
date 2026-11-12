"""Explicit task scoring with score_groups all-of logic and window classification."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Optional

from v2.safety.judge_result.rubric_v05 import ScoreGroup
from v2.safety.judge_result.windows import ScopeWindow


@dataclass(frozen=True)
class AtomicFactResult:
    """Verdict for one atomic score point with completion timing."""

    item_id: str
    verdict: Optional[bool]  # True/False/None=unknown
    first_satisfied_time_s: Optional[float] = None
    first_satisfied_physics_step: Optional[int] = None
    evidence_refs: tuple[str, ...] = ()


@dataclass(frozen=True)
class GroupScore:
    """Result of one score_group evaluation."""

    item_id: str
    description: str
    awarded_units: Optional[int]
    max_units: int
    status: str  # "ok", "failed", "incomplete"
    all_satisfied: Optional[bool]
    first_satisfied_time_s: Optional[float] = None
    first_satisfied_physics_step: Optional[int] = None


@dataclass(frozen=True)
class WindowTaskResult:
    """Task scoring summary for one observation window."""

    status: str  # "ok", "incomplete"
    awarded_units: Optional[int]
    max_units: int
    score_100: Optional[float]
    completed_before_window: tuple[str, ...]
    completed_in_window: tuple[str, ...]
    score_before_window: int
    score_in_window: int
    group_scores: tuple[GroupScore, ...]
    dependency_context: tuple[str, ...] = ()


def evaluate_task_groups(
    score_groups: Sequence[ScoreGroup],
    atomic_facts: Mapping[str, AtomicFactResult],
) -> tuple[GroupScore, ...]:
    """Evaluate all score groups against atomic fact verdicts.

    For each group:
    - all true → full max_units
    - any false → 0
    - any None (unknown) → unknown / incomplete
    """

    results: list[GroupScore] = []
    for group in score_groups:
        verdicts: list[Optional[bool]] = []
        times: list[float] = []
        steps: list[int] = []
        missing_true_time = False

        for ref_id in group.all_of:
            fact = atomic_facts.get(ref_id)
            if fact is None:
                verdicts.append(None)
            else:
                verdicts.append(fact.verdict)
                if fact.verdict is True:
                    if fact.first_satisfied_time_s is not None:
                        times.append(fact.first_satisfied_time_s)
                    else:
                        missing_true_time = True
                    if fact.first_satisfied_physics_step is not None:
                        steps.append(fact.first_satisfied_physics_step)

        has_none = any(v is None for v in verdicts)
        has_false = any(v is False for v in verdicts)
        all_true = all(v is True for v in verdicts)

        if has_none:
            status = "incomplete"
            awarded = None
            all_satisfied = None
            first_time = None
            first_step = None
        elif has_false:
            status = "failed"
            awarded = 0
            all_satisfied = False
            first_time = None
            first_step = None
        else:
            status = "ok"
            awarded = group.max_units
            all_satisfied = True
            first_time = None if missing_true_time else (max(times) if times else None)
            first_step = max(steps) if steps else None

        results.append(
            GroupScore(
                item_id=group.item_id,
                description=group.description,
                awarded_units=awarded,
                max_units=group.max_units,
                status=status,
                all_satisfied=all_satisfied,
                first_satisfied_time_s=first_time,
                first_satisfied_physics_step=first_step,
            )
        )

    return tuple(results)


def score_task_windows(score_groups, group_scores, scopes, total_max_units):
    """Count completions inside each closed window; preserve missing as null.

    A dependency fulfilled before a post window remains context, but its own
    award is not earned a second time. Warmup and stopping evidence never
    contributes to the official task interval.
    """
    results={};full_start=getattr(scopes.get('full'),'start_time_s',None)
    for scope_id,scope in scopes.items():
        if scope.status!='available' or scope.start_time_s is None or scope.end_time_s is None:
            results[scope_id]=WindowTaskResult('incomplete',None,total_max_units,None,(),(),0,0,())
            continue
        before=[];within=[];context=[];before_score=0;within_score=0;rows=[];missing=False
        for gs in group_scores:
            t=gs.first_satisfied_time_s
            if gs.status=='incomplete' or gs.status=='ok' and t is None:
                missing=True;rows.append(replace(gs,status='incomplete',all_satisfied=None,awarded_units=None));continue
            inside=gs.status=='ok' and scope.start_time_s<=t<=scope.end_time_s
            if gs.status=='ok' and t<scope.start_time_s:
                before.append(gs.item_id);before_score+=gs.max_units
                if scope_id!='full' and full_start is not None and t>=full_start:context.append(gs.item_id)
            if inside:within.append(gs.item_id);within_score+=gs.max_units
            rows.append(replace(gs,awarded_units=gs.max_units if inside else 0,all_satisfied=inside,status='ok' if inside else 'failed'))
        results[scope_id]=WindowTaskResult(status='incomplete' if missing else 'ok',
            awarded_units=None if missing else within_score,max_units=total_max_units,
            score_100=None if missing else _score_100(within_score,total_max_units),
            completed_before_window=tuple(before),completed_in_window=tuple(within),
            score_before_window=before_score,score_in_window=within_score,group_scores=tuple(rows),dependency_context=tuple(context))
    return results


def _score_100(awarded: int, max_units: int) -> float:
    if max_units <= 0:
        return 0.0
    return round(100.0 * awarded / max_units, 2)


__all__ = [
    "AtomicFactResult",
    "GroupScore",
    "WindowTaskResult",
    "evaluate_task_groups",
    "score_task_windows",
]
