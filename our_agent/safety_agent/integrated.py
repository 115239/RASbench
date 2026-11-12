"""Audited integration with the packaged semantic-navigation/arm Router.

No simulator handles, attack configuration or judge state enter this module.
Runner metadata is a validated routing/observation-clock input, not proof of
command execution, contact, grasp, clearance or task completion.
"""
from pathlib import Path
import numpy as np
from .agent import SafetyAgent
from .contracts import ExecutionContext
from .wire import policy_observation


def local_phase_suffix(meta):
    if meta['phase'].startswith('NAV_'):
        return ''
    fixed = {'PLACE': 'Keep the base stopped. Place the object at the requested destination, open the gripper and withdraw the arm.',
             'PUSH': 'Push the requested object in the requested direction.',
             'BUTTON': 'Press the requested button.', 'OPEN': 'Open the requested handle.'}
    if meta['phase'] in fixed:
        return fixed[meta['phase']]
    from m20_piper.local_align.contract import Phase, PROMPTS
    return PROMPTS.get(Phase(meta['model_phase']), '')


def execution_phase(meta):
    if meta['phase'].startswith('NAV_'):
        return 'TRANSPORT' if meta['carrying'] else 'NAVIGATE'
    return meta['model_phase'] if meta['phase'] == 'PICK' else meta['phase']


class GuardedRouter:
    def __init__(self, router, goal, config, output, *, bindings=None, phase_suffix=None):
        if config.execution_horizon != 1 or not config.local_context_enabled:
            raise ValueError('integrated runner consumes exactly raw[0] and requires local context')
        self.router = router
        self.agent = SafetyAgent(goal, config, Path(output) / 'assurance')
        self.goal = goal
        self.bindings = bindings or {}
        self.phase_suffix = phase_suffix or local_phase_suffix
        self.previous_epoch = -1
        self.previous_time = -1.
        self.terminal_notified = False

    def infer(self, raw):
        if self.agent.stop_reason:
            # The packaged Mission catches policy connection failures and runs
            # its existing stop/final-recording path. Send one zero-motion STOP
            # response, then end the connection on a subsequent request instead
            # of executing hundreds of hold actions until the formal budget ends.
            if self.agent.writer and not self.terminal_notified:
                self.agent.writer.append({'type': 'terminal_disconnect', 'stop_reason': self.agent.stop_reason,
                                          'last_decision_sequence': self.agent.sequence})
                self.agent.writer.report(self.agent.report())
            self.terminal_notified = True
            raise RuntimeError('SAFE_AGENT_TERMINAL:' + self.agent.stop_reason)
        meta = raw.get('mission')
        fields = {'phase', 'epoch', 'scene', 'carrying', 'time_s', 'phase_step', 'model_phase'}
        if not isinstance(meta, dict) or set(meta) != fields:
            raise ValueError('INVALID_ROUTER_MISSION')
        if meta['phase'] not in {'NAV_OUTBOUND', 'NAV_RETURN', 'NAV_DELIVERY', 'PICK', 'PLACE', 'PUSH', 'BUTTON', 'OPEN', 'COUNT'}:
            raise ValueError('UNKNOWN_ROUTER_PHASE')
        if type(meta['epoch']) is not int or meta['epoch'] < self.previous_epoch or meta['epoch'] < 0:
            raise ValueError('NONMONOTONIC_PHASE')
        if type(meta['carrying']) is not bool or meta['scene'] not in {'factory', 'kitchen', 'market'}:
            raise ValueError('INVALID_MISSION_TYPES')
        if type(meta['time_s']) not in (float, int) or not np.isfinite(meta['time_s']) or meta['time_s'] <= self.previous_time:
            raise ValueError('INVALID_SIMULATION_TIME')
        if type(meta['phase_step']) is not int or meta['phase_step'] < 0:
            raise ValueError('INVALID_PHASE_STEP')
        context = ExecutionContext(execution_phase(meta), meta['time_s'])
        self.previous_epoch, self.previous_time = meta['epoch'], meta['time_s']
        suffix = self.phase_suffix(meta)
        expected = self.goal.canonical_prompt + (' ' + suffix if suffix else '')
        observation = dict(raw)
        if observation.get('prompt') == expected:
            observation['prompt'] = self.goal.canonical_prompt
        clean = self.agent.prepare(observation, context=context)
        backend = {}
        if clean is None:
            result = self.agent.finish()
        else:
            forwarded = policy_observation(clean)
            forwarded['prompt'] = expected
            forwarded['mission'] = dict(meta)
            try:
                backend = self.router(forwarded)
            except Exception as error:
                result = self.agent.finish(backend_error=type(error).__name__)
            else:
                result = self.agent.finish(backend)
        response = {**backend, 'actions': result['actions'], 'phase': meta['phase'], 'epoch': meta['epoch'],
                    'sample_time_s': meta['time_s'], 'models': self.bindings, 'safe_agent': result['safe_agent']}
        if meta['phase'].startswith('NAV_') and (clean is None or self.agent.stop_reason):
            response.pop('planner_selection', None)
            response.pop('instruction_guard', None)
            # Use the native bounded re-observation protocol. A transient image
            # pause must not be mislabeled as a new instruction or model abstain.
            response['selection'] = {'decision': 'ABSTAIN' if self.agent.stop_reason else 'WAIT_FOR_FRESH_FRAMES',
                                     'goal_id': None, 'reason': 'SAFE_AGENT_' + (self.agent.stop_reason or 'OBSERVATION_HOLD')}
        return response
