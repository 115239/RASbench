"""Causal consistency checks using attacked/public sensor samples only."""
from __future__ import annotations
import numpy as np
from .contracts import wrap


class ObservationFusion:
    def __init__(self, config):
        self.cfg = config
        self.previous = None
        self.previous_images = {}
        self.last = None
        self.image_reasons = []

    def update(self, frame):
        s = np.asarray(frame.state, dtype=float)
        repaired = s.copy()
        reasons, reliability = [], {'state': 1., 'lidar': 1., 'imu': 1., 'encoder': 1.}
        severe = None
        valid_images = {}
        for name, value in frame.images.items():
            a = np.asarray(value)
            if a.dtype != np.uint8 or a.ndim != 3 or a.shape[2] != 3 or min(a.shape[:2]) < 16:
                reliability[name] = 0.
                continue
            small = a[::max(1, a.shape[0] // 32), ::max(1, a.shape[1] // 32)].astype(float) / 255.
            old = self.previous_images.get(name)
            jump = old is not None and old.shape == small.shape and np.abs(old-small).mean() > .30
            # A single anomalous view does not stop the whole policy.
            reliability[name] = .25 if jump else 1.
            if not jump:
                valid_images[name] = a
            self.previous_images[name] = small
        if frame.images:
            self.image_reasons = ['all_images_unreliable'] if not valid_images else []
        reasons.extend(self.image_reasons)
        dt = None
        if self.previous is not None:
            dt = frame.time_s - self.previous.time_s
            if dt > 0:
                prior = self.previous.state
                yaw = float(prior[5]) + .5 * wrap(float(s[5] - prior[5]))
                c, sn = np.cos(yaw), np.sin(yaw)
                velocity = np.array([[c, sn], [-sn, c]]) @ (s[:2] - prior[:2]) / dt
                wz = wrap(float(s[5] - prior[5])) / dt
                if np.linalg.norm(velocity) <= 2 and abs(wz) <= 3:
                    if np.linalg.norm(velocity - s[6:8]) > .15 or abs(wz-s[11]) > .25:
                        reasons.append('velocity_feedback_inconsistent')
                        reliability['state'] = .4
                        repaired[6:8], repaired[11] = velocity, wz
                else:
                    reasons.append('pose_feedback_discontinuous')
                    reliability['state'] = .1
                # Gravity/attitude has to agree over time before catastrophic termination.
                if max(abs(s[3]), abs(s[4])) > self.cfg.attitude_stop_rad:
                    if max(abs(prior[3]), abs(prior[4])) > self.cfg.attitude_stop_rad:
                        severe = 'persistent_critical_attitude'
                    else:
                        reasons.append('attitude_suspect')
        if frame.imu is not None:
            imu = np.asarray(frame.imu, dtype=float)
            if imu.shape != (3,) or not np.isfinite(imu).all():
                reliability['imu'] = 0.
                reasons.append('imu_invalid')
            elif np.linalg.norm(imu - repaired[9:12]) > .3:
                reliability['imu'] = .2
                reasons.append('imu_inconsistent')
        if frame.encoder is not None:
            enc = np.asarray(frame.encoder, dtype=float)
            if enc.shape != (6,) or not np.isfinite(enc).all() or np.max(np.abs(enc-s[12:18])) > .1:
                reliability['encoder'] = .2
                reasons.append('encoder_inconsistent')
        if frame.wheel_velocity is not None:
            wheel = np.asarray(frame.wheel_velocity, dtype=float)
            if frame.proprioception_time_s is None or frame.time_s-frame.proprioception_time_s > self.cfg.sensor_max_age_s:
                reliability['encoder'] = 0.
                reasons.append('encoder_stale')
            elif wheel.shape != (4,) or not np.isfinite(wheel).all() or not frame.wheel_radius_m:
                reliability['encoder'] = 0.
                reasons.append('encoder_invalid')
            elif (np.max(np.abs(wheel))*frame.wheel_radius_m > max(.30, np.linalg.norm(repaired[6:8])*3+.1)
                  and abs(repaired[11]) < .15):
                reliability['encoder'] = .2
                reasons.append('wheel_motion_inconsistent')
        lidar = frame.lidar
        if lidar is not None:
            if frame.time_s - frame.lidar_time_s > self.cfg.sensor_max_age_s:
                reliability['lidar'], lidar = 0., None
                reasons.append('lidar_stale')
            elif len(lidar) == 0:
                reliability['lidar'] = .2
                reasons.append('lidar_empty')
        self.previous = frame
        self.last = {'state': repaired.astype(np.float32), 'images': valid_images,
                     'lidar': lidar, 'reliability': reliability, 'reasons': reasons,
                     'severe': severe, 'dt': dt}
        return self.last
