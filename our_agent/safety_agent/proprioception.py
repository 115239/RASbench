"""Repair low-level sensor inputs using independent public observed state.

The method never accepts a clean sensor copy, attack metadata or a nominal motor
command. An unreliable reference causes abstention from numerical repair.
"""
import numpy as np
from .calibration import rotation_rpy


def fuse_proprioception(angular, gravity, wheel, *, fused, radius, config):
    raw=[np.asarray(v) for v in (angular,gravity,wheel)]
    if any(not np.isfinite(v).all() for v in raw) or raw[0].shape!=(3,) or raw[1].shape!=(3,) or raw[2].shape!=(4,):
        raise ValueError('invalid low-level public sample')
    out=[v.copy() for v in raw];reasons=[]
    if fused['reliability']['state']>=.4:
        state=fused['state']
        if np.linalg.norm(out[0]-state[9:12])>.3:
            out[0]=np.asarray(state[9:12],dtype=raw[0].dtype).copy()
            reasons.append('imu_angular_reconstructed_from_observed_state')
        reference=rotation_rpy(state[3:6]).T@np.array([0.,0.,-1.])
        if np.linalg.norm(out[1]-reference)>.25:
            out[1]=reference.astype(raw[1].dtype)
            reasons.append('gravity_reconstructed_from_observed_attitude')
        # Conservative wheel-speed envelope accommodates turning/slip without
        # claiming exact wheel odometry or fabricating contact information.
        cap=max(.8,(np.linalg.norm(state[6:8])+abs(state[11])*.5+.15)/radius*1.5)
        if np.max(np.abs(out[2]))>cap:
            out[2]=np.clip(out[2],-cap,cap)
            reasons.append('wheel_feedback_outlier_projection')
    if config.mode!='on' or config.visual_only:
        return (*raw,reasons)
    return (*out,reasons)
