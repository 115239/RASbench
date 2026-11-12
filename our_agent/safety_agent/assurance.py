"""Episode runtime for v1: perception, recovery, policy and actuator interventions."""
from __future__ import annotations
from dataclasses import asdict, replace
import copy
import numpy as np
from .contracts import canonicalize, validate_actions, LOW, HIGH, hold
from .evidence import EvidenceWriter
from .final_guard import FinalGuard
from .fusion import ObservationFusion
from .recovery import RecoveryManager
from .runtime_types import RuntimeConfig, TemporalGate, VisualEvidence
from .wire import policy_observation


class AssuranceAgent:
    def __init__(self, spec, episode_id, config=None, output=None):
        self.spec, self.episode_id = spec, episode_id
        config = config or RuntimeConfig()
        limits = spec.requirements.get('numeric_limits', {})
        self.cfg = replace(config, **{k: (max if k == 'clearance_m' else min)(getattr(config, k), v)
                                    for k, v in limits.items() if k in {'clearance_m','base_speed_mps','yaw_rate_rps'}})
        self.clock = TemporalGate(episode_id)
        self.fusion = ObservationFusion(self.cfg)
        self.recovery = RecoveryManager(spec, self.cfg)
        self.guard = FinalGuard(self.cfg)
        self.frame = self.fused = self.visual = None
        self.grasped = False
        self.grasp_streak = 0
        self.gripper_command = 0.
        self.close_time = None
        self.pending_commands = {}
        self.next_command = 0
        self.model_phase = ''
        self.policy_request_id = None
        self.actuator_reasons = []
        self.previous_receipt = None
        self.counts = {'policy_requests': 0, 'policy_interventions': 0,
                       'actuator_commands': 0, 'actuator_interventions': 0, 'receipts': 0,
                       'joint_commands': 0, 'joint_interventions': 0}
        self.writer = EvidenceWriter(output, {'schema': 'safe-agent-evidence/v2',
                  'task_identity': spec.identity, 'episode_id': episode_id,
                  'task': asdict(spec), 'config': asdict(self.cfg)}) if output else None

    def log(self, event):
        if self.writer:
            self.writer.append(event)

    def observe(self, frame, visual=None):
        self.clock.observe(frame)
        self.frame, self.fused = frame, self.fusion.update(frame)
        reasons = list(self.fused['reasons']) + self.actuator_reasons
        self.actuator_reasons = []
        self.accept_visual(visual, frame)
        if self.close_time is not None and frame.time_s-self.close_time > 1. and not self.grasped:
            if any(x in (frame.phase + self.model_phase).upper() for x in ('TRANSPORT', 'CARRY', 'LIFT')):
                reasons.append('grasp_unverified')
        if not self.cfg.visual_only:
            self.recovery.tick(frame.time_s, frame.phase, reasons, self.fused['severe'])
        else:
            self.recovery.tick(frame.time_s, frame.phase)
        self.log({'type': 'observation', 'sequence': frame.sequence, 'time_s': frame.time_s,
                  'phase': frame.phase, 'reliability': self.fused['reliability'], 'reasons': reasons,
                  'visual': asdict(visual) if visual else None, 'recovery': self.recovery.report()})

    def accept_visual(self, visual, frame):
        if visual is not None:
            if (not isinstance(visual, VisualEvidence) or visual.episode_id != self.episode_id
                    or visual.sequence > frame.sequence or visual.time_s > frame.time_s
                    or frame.time_s-visual.time_s > self.cfg.visual_max_age_s):
                raise ValueError('stale or foreign visual evidence')
            if self.visual is not None:
                if visual.sequence < self.visual.sequence or visual.time_s < self.visual.time_s:
                    raise ValueError('replayed visual evidence')
                if visual.sequence == self.visual.sequence:
                    if asdict(visual) != asdict(self.visual):
                        raise ValueError('changed visual evidence reused a sequence')
                    return
                if visual.time_s <= self.visual.time_s:
                    raise ValueError('visual timestamp was reused')
            # Count fresh camera evidence, never repeated controller ticks.
            if self.visual is None or visual.sequence > self.visual.sequence:
                self.grasp_streak = self.grasp_streak + 1 if visual.grasp_confirmed() else 0
                # Carrying is latched. Missing/stale views must never authorize opening.
                self.grasped = self.grasped or self.grasp_streak >= 2
            self.visual = visual

    @property
    def release_allowed(self):
        return bool(self.frame and self.visual and 'PLACE' in self.frame.phase
                    and self.frame.time_s-self.visual.time_s <= self.cfg.visual_max_age_s
                    and self.visual.supported and self.visual.placement_authorized
                    and np.linalg.norm(self.fused['state'][6:8]) <= .05)

    def prepare_policy(self, raw):
        self.counts['policy_requests'] += 1
        if self.cfg.mode == 'off':
            return raw
        if self.frame is None:
            raise ValueError('observation required before policy')
        clean = canonicalize(raw)
        changed = False
        if not self.cfg.visual_only:
            suffix = raw.get('safe_phase_instruction', '')
            # Suffix is set by the local runner bridge, never accepted from a remote policy.
            clean['prompt'] = self.spec.instruction + (' ' + suffix if suffix else '')
            clean['state'] = self.fused['state'].copy()
            changed = raw.get('prompt') != clean['prompt']
        if (self.visual and self.visual.target_visible and self.visual.target_consistent
                and not self.frame.phase.upper().startswith('NAV')):
            # Add grounded context, without introducing goal IDs or simulator coordinates.
            if self.frame.time_s-self.visual.time_s <= self.cfg.visual_max_age_s:
                import json
                clean['prompt'] += (' Visual localization (boxes x1,y1,x2,y2 normalized 0..1000): '
                                    + json.dumps(self.visual.target_boxes, sort_keys=True)
                                    + '. Track this object by shape and spatial relations; reobserve if uncertain.')
            changed = True
        if not self.cfg.visual_only and self.fused['images']:
            clean.update(self.fused['images'])
            for name in self.frame.images:
                if name not in self.fused['images']:
                    clean[name] = np.zeros_like(self.frame.images[name])
        forwarded = policy_observation(clean)
        if 'mission' in raw:
            forwarded['mission'] = copy.deepcopy(raw['mission'])
        self.counts['policy_interventions'] += int(changed)
        self.log({'type': 'policy_input', 'sequence': self.frame.sequence, 'changed': changed,
                  'request_id':self.policy_request_id,
                  'mode': self.cfg.mode, 'recovery': self.recovery.report()})
        return forwarded if self.cfg.mode == 'on' else raw

    def finish_policy(self, response):
        # Both modes preserve the full native response, including routing metadata.
        if self.cfg.mode == 'off':
            return response
        raw = validate_actions(response['actions'])
        out = copy.deepcopy(response)
        actions = raw[:1].copy()
        if not self.cfg.visual_only:
            actions = np.clip(actions, LOW, HIGH)
            state = self.recovery.state
            if state == 'STOP':
                actions = hold(self.gripper_command)
            elif state == 'RECOVERING':
                actions[:, :3] = 0.
                op = self.recovery.operation
                if op in {'REOBSERVE', 'REPLAN', 'VERIFY_PLACEMENT'}:
                    actions = hold(self.gripper_command)
                elif op == 'REGRASP':
                    # Open only when evidence says the object is still supported.
                    actions[:, 3:9] = 0.
                    if self.visual and self.visual.supported and not self.grasped:
                        elapsed = self.frame.time_s - self.recovery.attempt_start
                        actions[:, 9] = 0. if elapsed < .5 else 1.
                else:
                    actions[:, 3:9] *= .3
            if self.grasped and actions[0, 9] <= .5 and not self.release_allowed:
                actions[:, 9] = 1.
            if actions[0, 9] > .5 and self.gripper_command <= .5:
                self.close_time = self.frame.time_s
            if actions[0, 9] <= .5:
                self.close_time = None
                if self.release_allowed:
                    self.grasped = False
                    self.grasp_streak = 0
            self.gripper_command = float(actions[0, 9])
        out['actions'] = actions.astype(np.float32)
        if self.cfg.mode == 'on':
            out['safe_agent'] = {'schema': 'safe-agent-decision/v2', **self.recovery.report(),
                                 'grasp_verified': self.grasped}
            if self.recovery.state in {'RECOVERING', 'STOP'} and 'selection' in out:
                out['selection'] = {'decision': 'ABSTAIN' if self.recovery.state == 'STOP' else 'WAIT_FOR_FRESH_FRAMES',
                                    'goal_id': None, 'reason': 'SAFE_AGENT_' + self.recovery.state}
        self.log({'type': 'policy_output', 'sequence': self.frame.sequence,
                  'request_id':self.policy_request_id,
                  'post_attack_actions': raw.tolist(), 'proposed_guard_actions': actions.tolist(),
                  'applied': self.cfg.mode == 'on', 'recovery': self.recovery.report()})
        return out if self.cfg.mode == 'on' else response

    def base_command(self, candidate):
        if self.frame is None:
            raise ValueError('no sensor sample for actuator guard')
        out, reasons = self.guard.base(candidate, frame=self.frame, status=self.recovery.state,
                                       grasped=self.grasped, lidar=self.fused['lidar'])
        self.actuator_reasons = [r for r in reasons if r == 'clearance_unresolved']
        self.next_command += 1
        command_id = self.next_command
        self.pending_commands[command_id] = (self.frame.sequence, self.frame.time_s, out.copy())
        self.counts['actuator_commands'] += 1
        self.counts['actuator_interventions'] += int(not np.array_equal(np.asarray(candidate), out))
        self.log({'type': 'base_command', 'command_id': command_id, 'sequence': self.frame.sequence,
                  'policy_request_id':self.policy_request_id, 'control_source':self.frame.phase,
                  'time_s': self.frame.time_s, 'post_attack_candidate': np.asarray(candidate).tolist(),
                  'applied_command': out.tolist(), 'reasons': reasons})
        return out, command_id

    def receipt(self, receipt):
        issued = self.pending_commands.get(receipt.command_id)
        if issued is None or issued[0] != receipt.observation_sequence or abs(issued[1]-receipt.start_s) > 1e-6:
            raise ValueError('receipt has no matching issued command')
        if not np.allclose(issued[2], receipt.applied_base, atol=1e-7, rtol=0):
            raise ValueError('applied command differs from guarded command')
        self.clock.receipt(receipt)
        self.previous_receipt = receipt
        del self.pending_commands[receipt.command_id]
        self.counts['receipts'] += 1
        self.log({'type': 'execution_receipt', **asdict(receipt), 'task_success_claim': None})

    def report(self):
        count = self.counts['actuator_commands'] + self.counts['joint_commands']
        interventions = self.counts['actuator_interventions'] + self.counts['joint_interventions']
        return {**self.counts, 'command_count':count,'interventions':interventions,
                'intervention_rate':interventions/count if count else None,
                'mode': self.cfg.mode, 'recovery': self.recovery.report(),
                'pending_receipts': len(self.pending_commands), 'task_score': None,
                'grasp_verified_by_agent': self.grasped, 'success_claim': 'EXTERNAL_SCORER_ONLY'}

    def close(self):
        if self.writer and not self.writer.stream.closed:
            self.writer.report(self.report())
            self.writer.close()
