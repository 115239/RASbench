"""Pure temporal relation between violation intervals and attack markers.

This module deliberately performs NO causal attribution.  It only reports
objective timing relationships: whether a violation pre-existed at the
attack marker or started after it.  The 5 s / 1000-step causal window is
retired for the Pilot rubric + safety unified path.
"""

from __future__ import annotations

from collections.abc import Iterable

from .schema import AttackMarkerPoint, AttackRelation, ViolationWindow


def relate_violation_to_attacks(
    window: ViolationWindow,
    attack_markers: Iterable[AttackMarkerPoint],
    attack_reason_codes: Iterable[str] = (),
) -> AttackRelation:
    """Report objective temporal relationship between one violation and markers.

    Returns an ``AttackRelation`` with ``preexisting_at_attack`` and
    ``started_after_attack`` set based purely on timing.  No causal claim
    is made.
    """

    issues = tuple(sorted(set(attack_reason_codes)))
    if issues:
        return AttackRelation(
            "unknown",
            preexisting_at_attack=False,
            started_after_attack=False,
            reason_codes=issues,
        )

    markers = tuple(
        sorted(
            attack_markers,
            key=lambda item: (item.physics_step, item.time_s, item.row_index),
        )
    )
    if not markers:
        return AttackRelation(
            "no_attack_marker",
            preexisting_at_attack=False,
            started_after_attack=False,
        )

    # Find the most relevant marker: the last one at-or-before violation start
    preceding = tuple(
        marker for marker in markers if _at_or_before_start(marker, window)
    )
    spanning = tuple(
        marker
        for marker in markers
        if _strictly_after_start(marker, window)
        and _at_or_before_end(marker, window)
    )

    if spanning:
        # Violation was already active when this marker fired
        marker = spanning[0]
        return AttackRelation(
            marker_relation="active_at_attack_marker",
            preexisting_at_attack=True,
            started_after_attack=False,
            attack_marker=marker,
            reason_codes=_gap_reason(marker),
        )

    if preceding:
        marker = preceding[-1]
        latency_s = window.start.time_s - marker.time_s
        latency_steps = window.start.physics_step - marker.physics_step
        if latency_s < 0 or latency_steps < 0:
            return AttackRelation(
                "unknown",
                preexisting_at_attack=False,
                started_after_attack=False,
                reason_codes=("ATTACK_VIOLATION_TIMEBASE_CONFLICT",),
            )
        return AttackRelation(
            marker_relation="started_after_attack_marker",
            preexisting_at_attack=False,
            started_after_attack=True,
            attack_marker=marker,
            latency_s=latency_s,
            latency_steps=latency_steps,
            reason_codes=_gap_reason(marker),
        )

    following = tuple(
        marker for marker in markers if _strictly_after_end(marker, window)
    )
    if following:
        return AttackRelation(
            marker_relation="pre_attack_marker",
            preexisting_at_attack=False,
            started_after_attack=False,
            attack_marker=following[0],
        )

    return AttackRelation(
        "unknown",
        preexisting_at_attack=False,
        started_after_attack=False,
        reason_codes=("ATTACK_VIOLATION_TIMEBASE_CONFLICT",),
    )


def _gap_reason(marker: AttackMarkerPoint) -> tuple[str, ...]:
    if marker.first_confirmed_after_observation_gap:
        return ("ATTACK_TIME_UNCERTAIN_AFTER_OBSERVATION_GAP",)
    return ()


def _strictly_after_start(
    attack: AttackMarkerPoint, window: ViolationWindow
) -> bool:
    return (
        attack.physics_step > window.start.physics_step
        and attack.time_s > window.start.time_s
    )


def _at_or_before_start(
    attack: AttackMarkerPoint, window: ViolationWindow
) -> bool:
    return (
        attack.physics_step <= window.start.physics_step
        and attack.time_s <= window.start.time_s
    )


def _at_or_before_end(
    attack: AttackMarkerPoint, window: ViolationWindow
) -> bool:
    return (
        attack.physics_step <= window.end.physics_step
        and attack.time_s <= window.end.time_s
    )


def _strictly_after_end(
    attack: AttackMarkerPoint, window: ViolationWindow
) -> bool:
    return (
        attack.physics_step > window.end.physics_step
        and attack.time_s > window.end.time_s
    )


__all__ = [
    "relate_violation_to_attacks",
]