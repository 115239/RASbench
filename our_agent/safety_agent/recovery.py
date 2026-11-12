"""Budgeted, phase-local recovery; no attack IDs, timers or simulator facts."""
from collections import Counter


class RecoveryManager:
    def __init__(self, spec, config):
        if spec.action_budget is None or spec.duration_s is None:
            raise ValueError('bind official runtime budgets before starting an episode')
        self.spec, self.cfg = spec, config
        self.state = 'NORMAL'
        self.attempts = Counter()
        self.bad = self.clean = 0
        self.recovery_seconds = 0.
        self.recovery_actions = 0
        self.issued_actions = 0
        self.started = self.last_time = self.attempt_start = None
        self.phase = None
        self.reason = None
        self.stop_reason = None
        self.operation = None
        self.completed_recoveries = 0

    def stop(self, reason):
        self.state, self.stop_reason = 'STOP', self.stop_reason or reason
        return self.state

    def account_action(self):
        self.issued_actions += 1
        if self.state == 'RECOVERING':
            self.recovery_actions += 1

    def tick(self, time_s, phase, reasons=(), severe=None):
        phase = str(phase).upper()
        if self.started is None:
            self.started = time_s
        if self.last_time is not None and time_s < self.last_time:
            raise ValueError('recovery clock moved backwards')
        dt = 0. if self.last_time is None else time_s - self.last_time
        self.last_time = time_s
        if self.state == 'RECOVERING':
            self.recovery_seconds += dt
        if self.state == 'STOP':
            return self.state
        if severe:
            return self.stop(severe)
        if time_s - self.started >= self.spec.duration_s or self.issued_actions >= self.spec.action_budget:
            return self.stop('formal_budget_exhausted')
        if self.phase != phase and self.state != 'RECOVERING':
            self.phase = phase
            self.bad = self.clean = 0
            self.state = 'NORMAL'
        if (self.recovery_seconds >= self.spec.duration_s * self.cfg.recovery_fraction
                or self.recovery_actions >= max(1, int(self.spec.action_budget * self.cfg.recovery_fraction))):
            return self.stop('recovery_budget_exhausted')
        self.bad = self.bad + 1 if reasons else 0
        self.clean = self.clean + 1 if not reasons else 0
        if self.state == 'RECOVERING':
            if self.clean >= self.cfg.resume_samples:
                self.state = 'RESUME'
                self.completed_recoveries += 1
            elif time_s - self.attempt_start >= self.cfg.recovery_attempt_s:
                self.begin(time_s, self.phase, reasons or (self.reason,))
        elif self.bad >= self.cfg.confirm_samples:
            if self.cfg.recovery:
                self.begin(time_s, phase, reasons)
            else:
                self.stop('persistent_anomaly_recovery_disabled')
        elif reasons:
            self.state = 'SUSPECT'
        elif self.state in {'RESUME', 'SUSPECT'}:
            self.state = 'NORMAL'
        return self.state

    def begin(self, time_s, phase, reasons):
        if self.attempts[phase] >= self.cfg.max_recoveries_per_phase:
            self.stop('phase_recovery_exhausted')
            return
        self.attempts[phase] += 1
        self.attempt_start = time_s
        self.state, self.clean = 'RECOVERING', 0
        self.reason = next(iter(reasons), 'uncertain_observation')
        joined = ' '.join(str(r) for r in reasons)
        if 'prompt' in joined:
            self.operation = 'RESTORE_TASK'
        elif any(x in joined for x in ('image', 'target', 'vision')):
            self.operation = 'REOBSERVE'
        elif any(x in joined for x in ('grasp', 'gripper')):
            self.operation = 'REGRASP'
        elif 'PLACE' in phase:
            self.operation = 'VERIFY_PLACEMENT'
        elif any(x in phase for x in ('NAV', 'TRANSPORT', 'RETURN')):
            self.operation = 'REPLAN'
        else:
            self.operation = 'REALIGN'

    def report(self):
        return {'state': self.state, 'stop_reason': self.stop_reason,
                'phase_attempts': dict(self.attempts), 'operation': self.operation,
                'recovery_seconds': self.recovery_seconds, 'recovery_actions': self.recovery_actions,
                'issued_actions': self.issued_actions, 'completed_recoveries': self.completed_recoveries}
