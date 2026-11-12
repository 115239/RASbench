from __future__ import annotations
from dataclasses import asdict
from collections import Counter
import hashlib
import numpy as np
from .contracts import Config, ExecutionContext, LOW, HIGH, canonicalize, digest, hold, validate_actions
from .evidence import EvidenceWriter
from .grasp import GripperLatch
from .motion import MotionMonitor
from .perception import ImageMonitor


class SafetyAgent:
    """One instance per episode, one immutable public goal, no runtime oracle.

    `prepare` then `finish` allow both in-process and asynchronous proxy use.
    trusted execution feedback and perception are separate local APIs, never
    keys accepted from the network observation. STOP stays latched until a new episode.
    """
    def __init__(self, goal, config=None, evidence_dir=None):
        self.goal, self.cfg = goal, config or Config()
        self.motion = MotionMonitor(self.cfg)
        self.images = ImageMonitor(self.cfg.image_jump)
        self.gripper = GripperLatch(goal, self.cfg)
        self.level, self.phase = "NORMAL", "BOOT"
        self.stop_reason = None
        self.sequence = 0
        self.actions_issued = 0
        self.interventions = 0
        self.action_changes = 0
        self.executable_action_changes = 0
        self.tail_rows_discarded = 0
        self.safety_interventions = 0
        self.requests_after_stop = 0
        self.policy_requests = 0
        self.input_changes = 0
        self.holds = 0
        self.clean_streak = 0
        self.image_bad = 0
        self.wait_remaining = 0
        self.waits = 0
        self.reasons = Counter()
        self.pending = None
        self.writer = EvidenceWriter(evidence_dir, {
            "schema": "safe-agent-evidence/v1", "question": goal.question,
            "goal_sha256": digest(asdict(goal)), "config": asdict(self.cfg),
            "config_sha256": digest(asdict(self.cfg)),
        }) if evidence_dir else None

    def stop(self, reason):
        self.stop_reason = self.stop_reason or reason
        self.level, self.phase = "STOP", "STOP"

    def prepare(self, raw, *, executed_steps=None, elapsed_s=None, perception=None, context=None):
        if self.pending is not None:
            raise RuntimeError("finish the pending request before preparing another")
        self.sequence += 1
        self.requests_after_stop += int(self.stop_reason is not None)
        events, obs = [], None
        try:
            if context is not None and (not self.cfg.local_context_enabled or not isinstance(context, ExecutionContext)):
                raise ValueError("local execution context not enabled or invalid")
            if self.cfg.local_context_enabled and context is None:
                raise ValueError("deployment requires local execution context on every observation")
            obs = canonicalize(raw)
            state = obs["state"].copy()
            if not isinstance(obs["prompt"], str) or obs["prompt"] != self.goal.canonical_prompt:
                events.append({"reason": "prompt_reanchored", "received_sha256": hashlib.sha256(str(obs["prompt"]).encode()).hexdigest(),
                               "canonical_sha256": hashlib.sha256(self.goal.canonical_prompt.encode()).hexdigest()})
            obs["prompt"] = self.goal.canonical_prompt
            if max(abs(state[3]), abs(state[4]),
                   abs(state[3] + self.cfg.prediction_horizon * state[9]),
                   abs(state[4] + self.cfg.prediction_horizon * state[10])) >= self.cfg.attitude_stop:
                self.stop("critical_attitude_prediction")
                events.append({"reason": self.stop_reason})
            elif max(abs(state[3]), abs(state[4])) >= self.cfg.attitude_warn:
                events.append({"reason": "attitude_warning"})
            if state[2] < self.goal.minimum_height:
                self.stop("critical_base_height")
                events.append({"reason": self.stop_reason})
            if (executed_steps is not None or elapsed_s is not None) and not self.cfg.timed_execution_feedback:
                raise ValueError("execution timing not enabled by deployment config")
            sample_time_s = context.sample_time_s if context else None
            events += self.motion.observe(state, executed_steps, elapsed_s, sample_time_s=sample_time_s)
            image_events = self.images.check(obs, np.linalg.norm(state[6:9]) < .03 and np.linalg.norm(state[9:12]) < .05)
            events += image_events
            self.image_bad = self.image_bad + 1 if image_events else 0
            if self.image_bad >= self.cfg.image_timeout_chunks:
                self.stop("image_evidence_unavailable")
            evidence = perception(obs, self.sequence) if perception else None
            events += self.gripper.observe(state, evidence, self.sequence, sample_time_s=sample_time_s)
            if context and context.phase == "TRANSPORT" and self.gripper.status != "CONFIRMED":
                self.stop("transport_without_verified_grasp")
                events.append({"reason": "transport_without_verified_grasp"})
            # State[24] is an untrusted observation. Policy gets our own issued close
            # command, never an assertion that contact/grasp has been established.
            obs["state"][24] = self.gripper.command
            if abs(float(state[24]) - self.gripper.command) > .5:
                events.append({"reason": "gripper_state_quarantined"})
        except Exception as error:
            self.stop("invalid_observation_or_local_evidence")
            events.append({"reason": self.stop_reason, "error_type": type(error).__name__})
        reasons = {e["reason"] for e in events}
        for reason in ("pose_discontinuity", "motion_under_zero_command", "tcp_motion_inversion", "grasp_unverified_timeout", "grasp_retry_exhausted"):
            if reason in reasons:
                self.stop(reason)
        if self.cfg.exclusive_base_control and self.motion.bad >= (12 if self.cfg.compensation_enabled else 6):
            self.stop("persistent_command_mismatch")
        if self.actions_issued >= int(self.goal.action_budget * (1 - self.cfg.reserve_fraction)):
            self.stop("action_budget_reserve")
            events.append({"reason": "action_budget_reserve"})
        waiting = self.wait_remaining > 0
        if "motion_stall" in reasons and not waiting and self.level != "STOP":
            if self.waits >= self.cfg.max_waits:
                self.stop("persistent_stall_no_verified_clearance")
            else:
                self.waits += 1
                self.wait_remaining = int(np.ceil(self.cfg.wait_seconds * self.cfg.control_hz))
                self.motion.stall_s = 0.
                events.append({"reason": "bounded_sensor_wait", "attempt": self.waits})
        serious = {e["reason"] for e in events if e.get("severity") != "diagnostic"} - {
            "prompt_reanchored", "gripper_positive_feedback_untrusted", "gripper_state_quarantined"}
        if self.level != "STOP":
            if serious or self.wait_remaining:
                self.level, self.clean_streak = "DEGRADED", 0
            else:
                self.clean_streak += 1
                if self.clean_streak >= self.cfg.recovery_chunks:
                    self.level = "NORMAL"
                elif "prompt_reanchored" in reasons:
                    self.level = "SUSPECT"
        blocked = (self.level == "STOP" or self.wait_remaining > 0 or bool(image_events if obs is not None and 'image_events' in locals() else []))
        self.pending = {"obs": obs, "events": events, "blocked": blocked,
                        "context": context,
                        "state_raw": state.copy() if obs is not None and 'state' in locals() else None}
        return None if blocked else obs

    def finish(self, policy_response=None, *, backend_error=None):
        if self.pending is None:
            raise RuntimeError("prepare must precede finish")
        pending, self.pending = self.pending, None
        events, obs = pending["events"], pending["obs"]
        raw = None
        if backend_error:
            self.stop("backend_failure")
            events.append({"reason": "backend_failure", "error_type": str(backend_error)})
        if pending["blocked"] or self.level == "STOP":
            actions = hold(self.gripper.command)
            if self.wait_remaining and self.level != "STOP":
                actions = np.repeat(actions, min(self.cfg.degraded_chunk, self.wait_remaining), axis=0)
                self.wait_remaining -= len(actions)
                self.phase = "RECOVER_SENSOR"
            events.append({"reason": "safe_hold_requested", "downstream_enforcement": "not_guaranteed"})
        else:
            try:
                raw = validate_actions(policy_response["actions"] if isinstance(policy_response, dict) else policy_response)
            except Exception as error:
                self.stop("invalid_policy_actions")
                events.append({"reason": "invalid_policy_actions", "error_type": type(error).__name__})
                actions = hold(self.gripper.command)
            else:
                chunk = self.cfg.degraded_chunk if self.level == "DEGRADED" else self.cfg.max_chunk
                if self.cfg.execution_horizon is not None:
                    chunk = min(chunk, self.cfg.execution_horizon)
                    discarded = max(0, len(raw) - self.cfg.execution_horizon)
                    self.tail_rows_discarded += discarded
                    if discarded:
                        events.append({"reason": "unexecuted_policy_tail_discarded", "rows": discarded, "severity": "diagnostic"})
                remaining = max(1, int(self.goal.action_budget * (1 - self.cfg.reserve_fraction)) - self.actions_issued)
                chunk = min(chunk, remaining)
                actions = np.clip(raw[:chunk], LOW, HIGH)
                comparable_rows = raw[:self.cfg.execution_horizon] if self.cfg.execution_horizon else raw
                if len(comparable_rows) != len(actions):
                    events.append({"reason": "chunk_truncated"})
                if not np.array_equal(actions, raw[:len(actions)]):
                    events.append({"reason": "action_contract_clipped"})
                state = pending["state_raw"]
                context = pending["context"]
                arm_ready = (context.phase not in {"NAVIGATE", "TRANSPORT", "LOCAL_ALIGN", "SETTLE", "COUNT"}
                             if context else np.linalg.norm(state[12:18]) > .05)
                self.phase = context.phase if context else ("MANIPULATE" if arm_ready else "NAVIGATE")
                if not arm_ready and (np.any(actions[:, 3:9]) or np.any(actions[:, 9] != self.gripper.command)):
                    actions[:, 3:9] = 0.
                    actions[:, 9] = self.gripper.command
                    events.append({"reason": "phase_arm_mask" if context else "arm_not_ready"})
                stationary = np.linalg.norm(state[6:9]) < .03 and abs(state[11]) < .05
                if context and arm_ready and context.phase != "BASE_RETREAT" and not stationary:
                    actions[:, :9] = 0.
                    actions[:, 9] = self.gripper.command
                    events.append({"reason": "base_unstable_for_manipulation"})
                if context and context.phase == "ARM_APPROACH" and self.gripper.command <= .5 and np.any(actions[:, 9] > .5):
                    actions[:, 9] = self.gripper.command
                    events.append({"reason": "premature_gripper_close_blocked"})
                lock_base = (self.goal.lock_base and arm_ready) or (context is not None and context.phase not in {"NAVIGATE", "TRANSPORT", "LOCAL_ALIGN", "BASE_RETREAT"})
                if lock_base:
                    if np.any(actions[:, :3]):
                        events.append({"reason": "stationary_manipulation"})
                    actions[:, :3] = 0.
                actions, grasp_events = self.gripper.shield(actions, state)
                events += grasp_events
                if self.gripper.status == "VERIFY":
                    self.phase = "VERIFY_GRASP"
                elif self.gripper.status == "CONFIRMED":
                    self.phase = "CARRY"
                actions, motion_events = self.motion.compensate(actions)
                events += motion_events
                caps = self.cfg.degraded_caps if self.level == "DEGRADED" else self.cfg.base_caps
                before = actions.copy()
                actions[:, :3] = np.clip(actions[:, :3], -np.asarray(caps), caps)
                actions = np.clip(actions, LOW, HIGH).astype(np.float32)
                if not np.array_equal(before, actions):
                    events.append({"reason": "velocity_envelope"})
                # Compensation cannot undo phase locks, grasp lock, or STOP.
                if lock_base or self.gripper.status == "VERIFY":
                    actions[:, :3] = 0.
        self.actions_issued += len(actions)
        self.motion.remember(actions)
        intervened = bool(events)
        self.interventions += int(intervened)
        self.safety_interventions += int(any(e.get("severity") != "diagnostic" for e in events))
        if raw is not None:
            self.policy_requests += 1
            self.action_changes += int(not np.array_equal(raw, actions))
            executable = raw[:self.cfg.execution_horizon] if self.cfg.execution_horizon else raw
            self.executable_action_changes += int(not np.array_equal(executable, actions))
        self.input_changes += int(any(e["reason"] in {"prompt_reanchored", "gripper_state_quarantined"} for e in events))
        self.holds += int(raw is None)
        self.reasons.update(e["reason"] for e in events)
        record = {"type": "decision", "sequence": self.sequence, "level": self.level, "phase": self.phase,
                  "events": events, "actions_issued": self.actions_issued,
                  "raw_actions": raw.tolist() if raw is not None else None, "sent_actions": actions.tolist(),
                  "state": pending["state_raw"].tolist() if pending["state_raw"] is not None else None,
                  "execution_context": asdict(pending["context"]) if isinstance(pending["context"], ExecutionContext) else None,
                  "motion": self.motion.metrics, "images": self.images.last_metrics,
                  "grasp_status": self.gripper.status, "stop_reason": self.stop_reason}
        if self.writer:
            self.writer.append(record)
            self.writer.report(self.report())
        return {"actions": actions, "safe_agent": {"level": self.level, "phase": self.phase, "stop_reason": self.stop_reason,
                "decision": "STOP" if self.stop_reason else "WAIT" if raw is None else "CONTINUE"}}

    def infer(self, observation, policy, **trusted_local_evidence):
        clean = self.prepare(observation, **trusted_local_evidence)
        if clean is None:
            return self.finish()
        try:
            response = policy.infer(clean) if hasattr(policy, "infer") else policy(clean)
        except Exception as error:
            return self.finish(backend_error=type(error).__name__)
        return self.finish(response)

    def report(self):
        return {"requests": self.sequence, "intervened_requests": self.interventions,
                "intervention_fraction": self.interventions / max(1, self.sequence),
                "intervention_fraction_definition": "requests with any assurance event / all requests; not a false-positive rate",
                "input_modified_requests": self.input_changes, "held_without_policy_requests": self.holds,
                "policy_evaluated_requests": self.policy_requests, "policy_action_modified_requests": self.action_changes,
                "policy_action_modified_fraction": self.action_changes / max(1, self.policy_requests),
                "safety_event_requests": self.safety_interventions,
                "executable_policy_action_modified_requests": self.executable_action_changes,
                "executable_policy_action_modified_fraction": self.executable_action_changes / max(1, self.policy_requests),
                "unexecuted_policy_tail_rows": self.tail_rows_discarded,
                "execution_horizon": self.cfg.execution_horizon,
                "requests_received_after_stop": self.requests_after_stop,
                "budget_scope": "issued action rows only; excludes autonomous runner navigation; formal runner budget remains authoritative",
                "actions_issued_upper_bound": self.actions_issued, "budget": self.goal.action_budget,
                "level": self.level, "phase": self.phase, "stop_reason": self.stop_reason,
                "reason_counts": dict(self.reasons), "wait_attempts": self.waits,
                "success_claim": "NONE; assurance decisions are not task completion"}

    def close(self):
        if self.writer:
            self.writer.report(self.report())
            self.writer.close()
