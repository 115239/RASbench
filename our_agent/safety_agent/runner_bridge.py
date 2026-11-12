"""Runner adapter. Importing this module never imports Isaac, attacks, or a judge."""
from __future__ import annotations
from dataclasses import asdict, replace
from pathlib import Path
import json
import numpy as np
from .assurance import AssuranceAgent
from .contracts import canonicalize
from .runtime_types import RuntimeConfig, ObservationFrame, ExecutionReceipt
from .task_spec import TaskSpec


class RunnerBridge:
    def __init__(self, spec, episode_id, config, output, *, action_budget, duration_s, vision=None):
        self.agent = AssuranceAgent(spec.bind(action_budget=action_budget, duration_s=duration_s),
                                    episode_id, config, output)
        self.sequence = 0
        self.vision = vision
        self.latest_images = {}
        self.last_time = -1.
        self.pending = None
        self.latest_receipt = None
        self.phase_suffix = ''
        self.camera_mounts = {}

    def observe(self, state, time_s, phase, *, images=None, lidar=None, lidar_time_s=None, imu=None, encoder=None,
                wheel_velocity=None, wheel_radius_m=None, proprioception_time_s=None):
        # A policy request can share the preceding control sample's timestamp.
        if time_s < self.last_time:
            raise ValueError('runner observation time moved backwards')
        if images:
            self.latest_images = images
        if time_s == self.last_time:
            if images:
                from .calibration import camera_projections
                frame = replace(self.agent.frame, state=np.asarray(state, dtype=np.float32), images=images,
                                phase=phase, calibration=camera_projections(state, self.camera_mounts) if self.camera_mounts else {})
                self.agent.frame = frame
                self.agent.fused = self.agent.fusion.update(frame)
            if images and self.vision and (self.agent.visual is None or self.agent.visual.sequence < self.sequence):
                from .calibration import camera_projections
                frame = replace(self.agent.frame, images=images,
                    calibration=camera_projections(state, self.camera_mounts) if self.camera_mounts else {})
                self.agent.accept_visual(self.vision(frame, self.agent.spec), frame)
            return
        self.sequence += 1
        from .calibration import camera_projections
        calibration = camera_projections(state, self.camera_mounts) if images and self.camera_mounts else {}
        frame = ObservationFrame(self.agent.episode_id, self.sequence, float(time_s), phase,
                                 np.asarray(state, dtype=np.float32), images or {}, lidar, lidar_time_s, imu, encoder, calibration,
                                 wheel_velocity, wheel_radius_m, proprioception_time_s)
        visual = self.vision(frame, self.agent.spec) if self.vision and images else None
        self.agent.observe(frame, visual)
        self.last_time = time_s

    def start_formal_task(self, time_s, physics_step):
        if getattr(self,'formal_task_origin',None) is not None:
            raise ValueError('formal task origin already registered')
        recovery=self.agent.recovery
        if recovery.state not in ('NORMAL','RESUME') or recovery.issued_actions:
            raise ValueError('Agent not ready for formal task start')
        self.formal_task_origin={'sim_time_s':float(time_s),'physics_step':int(physics_step)}
        self.agent.log({'type':'formal_task_started','origin':self.formal_task_origin,
                        'warmup_recovery_seconds':recovery.recovery_seconds})
        recovery.started=float(time_s)
        recovery.recovery_seconds=0.
        recovery.recovery_actions=0

    def prepare_request(self, raw, *, time_s, phase, phase_instruction='', request_id=None):
        clean = canonicalize(raw)
        self.agent.policy_request_id = request_id
        self.agent.model_phase = str(raw.get('mission', {}).get('model_phase', '')).upper()
        images = {k: clean[k] for k in ('image', 'rear_image', 'wrist_image') if k in clean}
        self.observe(clean['state'], time_s, phase, images=images)
        enriched = dict(raw)
        enriched['safe_phase_instruction'] = phase_instruction
        # off/shadow must not leak the local-only extra field to the model.
        if self.agent.cfg.mode in {'off', 'shadow'}:
            self.agent.prepare_policy(enriched)
            return raw
        return self.agent.prepare_policy(enriched)

    def finish_response(self, response):
        return self.agent.finish_policy(response)

    def proprioception(self, angular, gravity, wheel, radius):
        from .proprioception import fuse_proprioception
        a,g,w,reasons=fuse_proprioception(angular,gravity,wheel,fused=self.agent.fused,
                                         radius=radius,config=self.agent.cfg)
        if reasons:
            self.agent.log({'type':'proprioception_repair','sequence':self.sequence,'time_s':self.last_time,
                'post_attack_angular':np.asarray(angular).tolist(),'post_attack_gravity':np.asarray(gravity).tolist(),
                'post_attack_wheel':np.asarray(wheel).tolist(),'consumed_angular':a.tolist(),
                'consumed_gravity':g.tolist(),'consumed_wheel':w.tolist(),'reasons':reasons,
                'applied':self.agent.cfg.mode=='on' and not self.agent.cfg.visual_only})
        return a,g,w

    def base(self, command):
        if self.pending is not None:
            raise ValueError('previous actuator command has no execution receipt')
        value, cid = self.agent.base_command(command)
        self.pending = {'id': cid, 'start': self.last_time, 'sequence': self.sequence,
                        'base': value.copy(), 'positions': (), 'velocities': ()}
        return value

    def joints(self, positions, velocities, **robot_limits):
        result = self.agent.guard.joints(positions, velocities, state=self.agent.frame.state,
                    status=self.agent.recovery.state, grasped=self.agent.grasped,
                    release_allowed=self.agent.release_allowed,
                    require_closed=('PICK' in self.agent.spec.stages and self.agent.model_phase in
                                    {'CONTACT_CLOSE', 'LIFT_VERIFY', 'RETRACT_LOADED', 'CARRY_VERIFY', 'BASE_RETREAT'}),
                    **robot_limits)
        p, v, reasons = result
        self.agent.counts['joint_commands'] += 1
        self.agent.counts['joint_interventions'] += int(not (np.array_equal(p,positions,equal_nan=True)
                                                       and np.array_equal(v,velocities,equal_nan=True)))
        if self.pending is None:
            raise ValueError('joint command requires current base transaction')
        def values(a):
            return tuple(float(x) if np.isfinite(x) else None for x in a)
        self.pending.update(positions=values(p), velocities=values(v))
        self.agent.log({'type': 'joint_command', 'command_id': self.pending['id'],
                        'post_attack_positions': values(positions), 'post_attack_velocities': values(velocities),
                        'applied_positions': values(p), 'applied_velocities': values(v), 'reasons': reasons})
        return p, v

    def executed(self, end_s, source):
        if self.pending is None:
            raise ValueError('execution has no issued command')
        p = self.pending
        receipt = ExecutionReceipt(self.agent.episode_id, p['id'], p['sequence'], source,
                    p['start'], end_s, tuple(float(x) for x in p['base']), p['positions'], p['velocities'])
        self.agent.receipt(receipt)
        self.latest_receipt = asdict(receipt)
        self.pending = None
        return self.latest_receipt

    def close(self):
        if (self.agent.writer and not self.agent.writer.stream.closed and self.vision
                and hasattr(self.vision,'metrics')):
            self.agent.log({'type':'vision_latency_summary',**self.vision.metrics()})
        self.agent.close()

    def account_task_action(self):
        self.agent.recovery.account_action()

    def not_advanced(self):
        if self.pending is not None:
            cid = self.pending['id']
            self.agent.log({'type': 'command_without_physics_advance', 'command_id': cid})
            self.agent.pending_commands.pop(cid)
            self.pending = None


def from_environment(output, *, action_budget, duration_s):
    import os
    mode = os.environ.get('SAFE_AGENT_MODE', 'off')
    task_path = os.environ.get('SAFE_AGENT_TASK_SPEC')
    # Legacy runs remain byte-compatible unless explicitly enrolled in a paired experiment.
    if not task_path and mode == 'off':
        return None
    if not task_path:
        raise ValueError('SAFE_AGENT_TASK_SPEC is required for on/shadow')
    options = json.loads(Path(os.environ['SAFE_AGENT_CONFIG']).read_text()) if os.environ.get('SAFE_AGENT_CONFIG') else {}
    options['mode'] = mode
    vision = None
    if os.environ.get('SAFE_AGENT_VISION_ENDPOINT'):
        from .vision_client import VisionClient
        vision = VisionClient(os.environ['SAFE_AGENT_VISION_ENDPOINT'])
    for name, actual, cast in [('SAFE_AGENT_ACTION_BUDGET', action_budget, int), ('SAFE_AGENT_DURATION_S', duration_s, float)]:
        if name in os.environ and cast(os.environ[name]) != actual:
            raise ValueError('frozen runner budget mismatch: ' + name)
    return RunnerBridge(TaskSpec.load(task_path), os.environ.get('SAFE_AGENT_EPISODE_ID', str(Path(output).resolve())), RuntimeConfig(**options),
                        Path(output) / 'assurance_v1', action_budget=action_budget,
                        duration_s=duration_s, vision=vision)
