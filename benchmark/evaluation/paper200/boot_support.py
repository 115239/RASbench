"""Measured motor-only support during pre-task warmup, never during policy work.

The nominal STOP stance can keep redistributing contact load on uneven collision
meshes. After the initial 0.3 seconds of ordinary settling, latch the measured legal leg
configuration and blend the motor targets to it; wheel drives brake physically.
All joint commands still pass through the normal attack and assurance boundary.
No root/joint state, effort limit, contact filter, or readiness threshold is changed.
"""
import json
import numpy as np

SCHEMA = 'paper200-measured-warmup-support/v2'
WARMUP_WHEEL_DAMPING = 60.0

def _audit(c, row):
    row.update(schema=SCHEMA, physics_step=int(c.physics_step),
               phase=c.paper_lifecycle.phase)
    c.vla_trace.append(row)


def _native_values(value):
    if value is None:
        return None
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=float).reshape(-1).tolist()


def _drive_readback(c):
    kp, kd = c.controller.get_gains()
    return {'joint_order':list(c.names), 'kp':_native_values(kp),
            'kd':_native_values(kd),
            'max_efforts':_native_values(c.controller.get_max_efforts()),
            'source':'live_ArticulationController_PhysX_get_gains_and_get_max_efforts'}


def apply(c, command, measured, positions, wheel_velocities, namespace):
    clock = getattr(c, 'paper_lifecycle', None)
    if clock is None:
        return positions, wheel_velocities
    state = getattr(c, '_paper_boot_support', None)
    active = clock.phase == 'WARMUP'
    if not active:
        if state is not None and not state['released']:
            # set_stationary_drive_mode has already handled this tick's command.
            # Restore its exact configured gains, including MOVE and Piper gains.
            stationary = bool(c.stationary_drive_mode)
            c.controller.set_gains(
                kps=(c.stationary_kps if stationary else c.locomotion_kps).copy(),
                kds=(c.stationary_kds if stationary else c.locomotion_kds).copy(),
                save_to_usd=False)
            c.stationary_leg_entry_targets = None
            c.stationary_control_steps = 0
            state['released'] = True
            _audit(c, {'event':'warmup_support_released',
                       'normal_targets_unchanged':True,
                       'normal_stationary_mode':stationary,
                       'native_drive_readback':_drive_readback(c)})
        return positions, wheel_velocities
    command = np.asarray(command, dtype=float)
    if command.shape != (3,) or not np.isfinite(command).all() or np.max(abs(command)) > 1e-6:
        raise RuntimeError('WARMUP_SUPPORT_REQUIRES_ZERO_BASE_COMMAND')
    leg_names = namespace['LEG_JOINTS']; wheel_names = namespace['WHEEL_JOINTS']
    # Brake from the first WARMUP tick. Dining/Market previously retained the
    # weaker locomotion wheel damping even while commanding a stationary stop.
    kp = c.stationary_kps.copy(); kd = c.stationary_kds.copy()
    for name in leg_names: kp[c.index[name]]=200.; kd[c.index[name]]=20.
    for name in wheel_names: kp[c.index[name]]=0.; kd[c.index[name]]=WARMUP_WHEEL_DAMPING
    c.controller.set_gains(kps=kp, kds=kd, save_to_usd=False)
    if state is None:
        state = {'released':False, 'goal':None}
        c._paper_boot_support = state
        _audit(c, {'event':'warmup_support_brake_started', 'wheel_kp_kd':[0.,WARMUP_WHEEL_DAMPING],
                   'native_drive_readback':_drive_readback(c)})
    # Preserve the motor-controlled initial drop, then latch an actually
    # supported posture instead of repeatedly pulling toward nominal stance.
    if int(c.physics_step) * float(namespace['PHYSICS_DT']) < .3:
        return positions, {name:0. for name in wheel_names}
    indices = [c.index[name] for name in leg_names]
    if state['goal'] is None:
        goal = np.asarray(measured, dtype=float)[indices].copy()
        lo = np.asarray(c.articulation['lower'])[indices] + .003
        hi = np.asarray(c.articulation['upper'])[indices] - .003
        if goal.shape != (12,) or not np.isfinite(goal).all() or np.any(goal < lo) or np.any(goal > hi):
            raise RuntimeError('WARMUP_SUPPORT_MEASURED_STANCE_OUT_OF_LIMITS')
        state = {'goal':goal, 'start':np.asarray([positions[n] for n in leg_names]),
                 'start_step':int(c.physics_step), 'released':False}
        c._paper_boot_support = state
        _audit(c, {'event':'warmup_support_latched', 'leg_order':list(leg_names),
                   'measured_leg_rad':goal.tolist(), 'start_leg_target_rad':state['start'].tolist(),
                   'state25':np.asarray(c.state()).tolist(), 'wheel_target_rad_s':[0.]*4,
                   'leg_kp_kd':[200.,20.], 'wheel_kp_kd':[0.,WARMUP_WHEEL_DAMPING],
                   'source':'measured_joint_encoder_after_0.3_second_settle',
                   'root_or_joint_state_writes':False, 'effort_limits_changed':False})
    if state['released']:
        raise RuntimeError('WARMUP_SUPPORT_REENTERED_WITHOUT_EPISODE_RESET')
    progress = min(1., max(0., (int(c.physics_step)-state['start_step'])*float(namespace['PHYSICS_DT'])/.3))
    blend = 10*progress**3-15*progress**4+6*progress**5
    target = state['start'] + blend*(state['goal']-state['start'])
    positions = dict(positions); wheel_velocities = dict(wheel_velocities)
    for name, value in zip(leg_names, target): positions[name] = float(value)
    for name in wheel_names: wheel_velocities[name] = 0.
    return positions, wheel_velocities
