"""Final command projection. The input is the post-attack candidate, never a clean copy."""
from __future__ import annotations
import numpy as np


class FinalGuard:
    def __init__(self, config):
        self.cfg = config
        self.last_base = np.zeros(3)
        self.last_positions = None
        self.last_velocities = None
        self.last_time = None

    def base(self, candidate, *, frame, status, grasped=False, lidar=None):
        raw = np.asarray(candidate, dtype=float)
        if raw.shape != (3,) or not np.isfinite(raw).all():
            raise ValueError('invalid base candidate')
        out, reasons = raw.copy(), []
        dt = .02 if self.last_time is None else max(1e-4, min(.1, frame.time_s-self.last_time))
        if status == 'STOP':
            out[:] = 0.
            reasons.append('terminal_brake')
        else:
            speed = self.cfg.base_speed_mps * (.6 if grasped else 1.)
            norm = np.linalg.norm(out[:2])
            if norm > speed:
                out[:2] *= speed / norm
                reasons.append('base_speed_limit')
            out[2] = np.clip(out[2], -self.cfg.yaw_rate_rps, self.cfg.yaw_rate_rps)
            delta = out[:2] - self.last_base[:2]
            n = np.linalg.norm(delta)
            if n > self.cfg.base_acceleration * dt:
                out[:2] = self.last_base[:2] + delta * (self.cfg.base_acceleration * dt / n)
                reasons.append('base_acceleration_limit')
            out[2] = np.clip(out[2], self.last_base[2]-self.cfg.yaw_acceleration*dt,
                            self.last_base[2]+self.cfg.yaw_acceleration*dt)
            if lidar is not None and len(lidar):
                # Sensor extrinsic must have been applied by the public adapter.
                points = np.asarray(lidar)
                points = points[(points[:, 2] > -.15) & (points[:, 2] < 1.5), :2]
                if len(points) and np.linalg.norm(out[:2]) > 0:
                    horizon = self.cfg.prediction_s
                    permitted = False
                    for scale in (1., .5, .25, 0.):
                        sample = np.linspace(0., horizon, 6)[:, None] * out[None, :2] * scale
                        distances = np.linalg.norm(points[None, :, :] - sample[:, None, :], axis=2)
                        if distances.min() >= self.cfg.footprint_radius_m + self.cfg.clearance_m:
                            out[:2] *= scale
                            permitted = True
                            if scale != 1.:
                                reasons.append('predicted_clearance_projection')
                            break
                    if not permitted:
                        out[:] = 0.
                        reasons.append('clearance_unresolved')
            if status == 'RECOVERING':
                out[:] = 0.
                reasons.append('recovery_base_hold')
        self.last_base, self.last_time = out.copy(), frame.time_s
        applied = out if self.cfg.mode == 'on' and self.cfg.final_guard and not self.cfg.visual_only else raw
        return (applied.astype(np.float32) if self.cfg.mode == 'on' and self.cfg.final_guard
                and not self.cfg.visual_only else np.asarray(candidate).copy()), reasons

    def joints(self, positions, velocities, *, lower, upper, velocity_limits, names,
               state, dt, status, grasped=False, release_allowed=False, require_closed=False, wheel_radius_m=None):
        p, v = np.asarray(positions, dtype=float), np.asarray(velocities, dtype=float)
        lo, hi, cap = map(lambda x: np.asarray(x, dtype=float), (lower, upper, velocity_limits))
        n = len(names)
        if any(a.shape != (n,) for a in (p, v, lo, hi, cap)) or np.isinf(p).any() or np.isinf(v).any():
            raise ValueError('invalid articulation command layout')
        if np.any(lo > hi) or not np.isfinite(cap).all() or np.any(cap <= 0):
            raise ValueError('invalid public robot limits')
        outp, outv, reasons = np.clip(p, lo, hi), np.clip(v, -cap, cap), []
        dt = max(1e-4, min(.1, dt))
        for i, name in enumerate(names):
            if 'wheel' in name and wheel_radius_m is not None and np.isfinite(outv[i]):
                if wheel_radius_m<=0:raise ValueError('invalid wheel radius')
                wheel_cap=self.cfg.base_speed_mps*(.6 if grasped else 1.)/wheel_radius_m
                outv[i]=np.clip(outv[i],-wheel_cap,wheel_cap)
                if self.last_velocities is not None and np.isfinite(self.last_velocities[i]):
                    step=self.cfg.base_acceleration*dt/wheel_radius_m
                    outv[i]=np.clip(outv[i],self.last_velocities[i]-step,self.last_velocities[i]+step)
            if name in [f'joint{j}' for j in range(1, 7)] and np.isfinite(outp[i]):
                j = int(name[5:])-1
                anchor = float(state[12+j])
                bound = min(cap[i], self.cfg.arm_speed_rps) * dt
                outp[i] = np.clip(outp[i], anchor-bound, anchor+bound)
            elif self.last_positions is not None and np.isfinite(outp[i]) and np.isfinite(self.last_positions[i]):
                outp[i] = np.clip(outp[i], self.last_positions[i]-cap[i]*dt, self.last_positions[i]+cap[i]*dt)
            if grasped and not release_allowed and name in {'joint7', 'joint8'}:
                if np.isfinite(outv[i]):outv[i]=0.
                if self.last_positions is not None and np.isfinite(self.last_positions[i]):
                    if abs(outp[i]) > abs(self.last_positions[i]):
                        outp[i] = self.last_positions[i]
                        reasons.append('unsupported_release_blocked')
            if require_closed and not release_allowed and name in {'joint7', 'joint8'}:
                if np.isfinite(outv[i]):outv[i]=0.
                anchor = self.last_positions[i] if self.last_positions is not None else p[i]
                target = np.clip(0., lo[i], hi[i])
                outp[i] = np.clip(target, anchor-cap[i]*dt, anchor+cap[i]*dt) if np.isfinite(anchor) else target
                reasons.append('task_closure_projection')
        if status == 'STOP':
            outv[np.isfinite(outv)] = 0.
            if self.last_positions is not None:
                mask = np.isfinite(self.last_positions)
                outp[mask] = self.last_positions[mask]
            reasons.append('terminal_joint_hold')
        if not np.array_equal(p, outp, equal_nan=True) or not np.array_equal(v, outv, equal_nan=True):
            reasons.append('joint_envelope_projection')
        self.last_positions, self.last_velocities = outp.copy(), outv.copy()
        if self.cfg.mode != 'on' or not self.cfg.final_guard or self.cfg.visual_only:
            return np.asarray(positions).copy(), np.asarray(velocities).copy(), reasons
        return outp.astype(np.float32), outv.astype(np.float32), reasons
