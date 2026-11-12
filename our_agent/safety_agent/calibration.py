"""Camera projections from fixed public mount calibration and observed base pose."""
import numpy as np


def rotation_rpy(angles):
    r, p, y = angles
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                     [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr], [-sp, cp*sr, cp*cr]])


def camera_projections(state, mounts):
    rot, origin = rotation_rpy(state[3:6]), np.asarray(state[:3])
    result = {}
    for name, mount in mounts.items():
        eye = origin + rot @ np.asarray(mount['eye_base'])
        target = origin + rot @ np.asarray(mount['target_base'])
        forward = target-eye
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, [0., 0., 1.])
        if np.linalg.norm(right) < 1e-8:
            continue
        right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        r = np.stack([right, down, forward])
        k = np.asarray(mount['intrinsics'])
        result[name] = {'projection_world_to_pixel': (k @ np.c_[r, -r @ eye]).tolist(),
                        'resolution': mount['resolution']}
    return result
