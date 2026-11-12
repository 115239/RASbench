from __future__ import annotations
from collections import deque
import numpy as np
from .contracts import wrap


class MotionMonitor:
    """Detect consequences using root pose; never infer a hidden attack identity.

    Fit only with explicit executed-step feedback and exclusive VLA control.
    Wall-clock request intervals are not simulation time. Saturated rows,
    changing commands and non-planar motion are excluded from identification.
    """
    def __init__(self, config):
        self.cfg = config
        self.previous = None
        self.previous_time = None
        self.sent = None
        self.stall_s = 0.
        self.bad = 0
        self.tcp_bad = 0
        self.zero_motion_count = 0
        self.samples = [deque(maxlen=24) for _ in range(3)]
        self.models = {}
        self.model_age = {}
        self.metrics = {}

    def observe(self, state, executed_steps=None, elapsed_s=None, *, sample_time_s=None):
        events = []
        if (executed_steps is None) != (elapsed_s is None):
            raise ValueError("execution feedback must include both count and elapsed simulation time")
        if sample_time_s is not None:
            if not np.isfinite(sample_time_s) or sample_time_s < 0 or (self.previous_time is not None and sample_time_s <= self.previous_time):
                raise ValueError("observation clock must be finite and strictly increasing")
        sample_dt = sample_time_s - self.previous_time if sample_time_s is not None and self.previous_time is not None else None
        self.previous_time = sample_time_s
        if elapsed_s is None and self.models:
            self.models.clear()
            for samples in self.samples:
                samples.clear()
            events.append({"reason": "affine_models_revoked_missing_timing"})
        if self.previous is None or self.sent is None:
            self.previous = state.copy()
            return events
        explicit = executed_steps is not None and elapsed_s is not None
        if explicit:
            if type(executed_steps) is not int or not 1 <= executed_steps <= len(self.sent) or not np.isfinite(elapsed_s) or elapsed_s <= 0:
                raise ValueError("invalid trusted execution feedback")
            n, dt = executed_steps, elapsed_s
        else:
            n, dt = len(self.sent), len(self.sent) / self.cfg.control_hz
        timing = "trusted_execution" if explicit else "nominal_unverified"
        aligned = sample_dt is None or not explicit or abs(sample_dt - elapsed_s) <= max(.01, .05 * sample_dt)
        if sample_dt is not None:
            dt = sample_dt
            timing = "local_observation_clock"
        if not aligned:
            self.models.clear()
            for samples in self.samples:
                samples.clear()
            events.append({"reason": "execution_observation_interval_mismatch", "severity": "diagnostic"})
        # Observation spacing does not certify the command that ran between
        # samples. Hybrid navigation is driven by a separate controller.
        comparable = self.cfg.exclusive_base_control and aligned
        prior = self.previous
        displacement = state[:2] - prior[:2]
        mid_yaw = prior[5] + .5 * wrap(float(state[5] - prior[5]))
        c, s = np.cos(mid_yaw), np.sin(mid_yaw)
        measured = np.array([(c*displacement[0] + s*displacement[1]) / dt,
                             (-s*displacement[0] + c*displacement[1]) / dt,
                             wrap(float(state[5] - prior[5])) / dt])
        expected = self.sent[:n, :3].mean(axis=0)
        predicted = expected.copy()
        if self.cfg.compensation_enabled:
            for d, (a, b, _) in self.models.items():
                predicted[d] = a * expected[d] + b
        residual = measured - predicted
        # TCP action is a per-action displacement, not velocity. Compare sums
        # only while the base frame stays still; never mix metres with m/s.
        tcp_expected = self.sent[:n, 3:6].sum(axis=0)
        tcp_measured = state[18:21] - prior[18:21]
        stable_frame = np.linalg.norm(displacement) < .01 and abs(wrap(float(state[5] - prior[5]))) < .03
        tcp_opposite = stable_frame and bool(np.any((np.abs(tcp_expected) > .003) & (np.abs(tcp_measured) > .002) & (tcp_expected * tcp_measured < 0)))
        self.tcp_bad = self.tcp_bad + 1 if tcp_opposite else 0
        if self.tcp_bad >= 2:
            events.append({"reason": "tcp_motion_inversion", "expected_displacement": tcp_expected.tolist(),
                           "observed_displacement": tcp_measured.tolist()})
        self.metrics = {"measured": measured.tolist(), "expected": predicted.tolist(), "sent": expected.tolist(),
                        "residual": residual.tolist(), "timing": timing,
                        "dt": float(dt), "exclusive_base_control": self.cfg.exclusive_base_control,
                        "command_comparison_eligible": comparable,
                        "inverse_identification_eligible": bool(explicit and comparable)}
        if np.linalg.norm(displacement) > max(1., dt * 2.):
            if explicit or sample_dt is not None:
                events.append({"reason": "pose_discontinuity"})
            else:
                events.append({"reason": "pose_displacement_without_timing", "severity": "diagnostic"})
        moving = comparable and np.linalg.norm(predicted[:2]) > .035
        if moving and np.linalg.norm(displacement) / dt < .008:
            self.stall_s += dt
        else:
            self.stall_s = 0.
        if self.stall_s >= self.cfg.stall_seconds:
            events.append({"reason": "motion_stall", "seconds": self.stall_s})
        mismatch = bool(np.any(np.abs(residual) > self.cfg.residual_threshold))
        # Sign disagreement is identifiable even at low speeds (timing-free).
        opposite = bool(np.any((np.abs(predicted) > .035) & (np.abs(measured) > .02) & (predicted * measured < 0)))
        self.bad = self.bad + 1 if comparable and (mismatch or opposite) else 0
        if self.bad >= 2:
            events.append({"reason": "command_motion_mismatch", **self.metrics})
        # A stop command can itself be corrupted downstream. Report explicitly.
        self.zero_motion_count = self.zero_motion_count + 1 if np.linalg.norm(predicted) < .005 and np.linalg.norm(measured) > .08 else 0
        if self.zero_motion_count >= 3:
            reason = "motion_under_zero_command" if comparable else "unattributed_motion_under_zero_command"
            events.append({"reason": reason, "severity": "safety" if comparable else "diagnostic", **self.metrics})
        if explicit and comparable and max(abs(state[3]), abs(state[4])) < .15:
            for d in range(3):
                if d in self.models:
                    a, b, _ = self.models[d]
                    self.model_age[d] = self.model_age.get(d, 0) + 1
                    if abs(measured[d] - (a * expected[d] + b)) > self.cfg.residual_threshold[d] or self.model_age[d] > 12:
                        del self.models[d]
                        self.samples[d].clear()
                        events.append({"reason": "affine_model_revoked", "dimension": d})
                cap = self.cfg.base_caps[d]
                if self.sent[:n, d].std() > .005 or abs(measured[d]) > .90 * cap:
                    continue
                self.samples[d].append((float(expected[d]), float(measured[d])))
                fit = self._fit(d)
                if fit is not None and d not in self.models:
                    self.models[d] = fit
                    self.model_age[d] = 0
                    events.append({"reason": "affine_model_identified", "dimension": d, "model": list(fit)})
        self.previous = state.copy()
        return events

    def _fit(self, d):
        xy = np.array(self.samples[d])
        if len(xy) < self.cfg.affine_min_samples or xy[:, 0].std() < self.cfg.affine_std:
            return None
        x, y = xy[:, 0], xy[:, 1]
        matrix = np.column_stack((x, np.ones(len(x))))
        beta = np.linalg.lstsq(matrix, y, rcond=None)[0]
        for _ in range(4):
            residual = y - matrix @ beta
            scale = max(.003, float(np.median(np.abs(residual))) * 1.4826)
            weights = np.minimum(1., 1.5 * scale / np.maximum(np.abs(residual), 1e-9))
            beta = np.linalg.lstsq(matrix * weights[:, None]**.5, y * weights**.5, rcond=None)[0]
        a, b = map(float, beta)
        variance = float(np.sum((y - y.mean())**2))
        r2 = 1. - float(np.sum((y - matrix @ beta)**2)) / max(variance, 1e-10)
        if r2 < self.cfg.affine_r2 or not .25 <= abs(a) <= 6 or abs(b) > .3 * self.cfg.base_caps[d]:
            return None
        if abs(a - 1) < .25 and abs(b) < .02:
            return None
        return a, b, r2

    def compensate(self, actions):
        values = actions.copy()
        events = []
        if self.cfg.compensation_enabled:
            for d, (a, b, r2) in self.models.items():
                values[:, d] = (values[:, d] - b) / a
                events.append({"reason": "affine_compensation", "dimension": d, "a": a, "b": b, "r2": r2})
        return values, events

    def remember(self, actions):
        self.sent = actions.copy()
