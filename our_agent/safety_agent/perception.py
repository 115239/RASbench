"""Observable image anomalies, not a semantic object detector or depth estimator."""
from __future__ import annotations
import hashlib
import numpy as np


class ImageMonitor:
    def __init__(self, threshold=.25):
        self.previous = {}
        self.threshold = threshold
        self.last_metrics = {}
        self.quarantined_color = {}

    def check(self, obs, stationary):
        events = []
        for key in ("image", "rear_image", "wrist_image"):
            if key not in obs:
                events.append({"reason": "image_missing", "view": key})
                continue
            image = np.asarray(obs[key])
            if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 16 or image.dtype != np.uint8:
                events.append({"reason": "image_invalid", "view": key})
                continue
            # Deterministic bounded thumbnail, retaining original pixels for policy.
            yy = np.linspace(0, image.shape[0] - 1, 64).astype(int)
            xx = np.linspace(0, image.shape[1] - 1, 64).astype(int)
            thumb = image[yy[:, None], xx].astype(np.float32) / 255.
            prior = self.previous.get(key)
            delta = float(np.abs(thumb - prior).mean()) if prior is not None else 0.
            red = (thumb[..., 0] > thumb[..., 1] + .15) & (thumb[..., 0] > thumb[..., 2] + .15)
            green = (thumb[..., 1] > thumb[..., 0] + .15) & (thumb[..., 1] > thumb[..., 2] + .15)
            color = (float(red.mean()), float(green.mean()))
            old = self.last_metrics.get(key, {}).get("color", color)
            remap = old[0] - color[0] > .08 and color[1] - old[1] > .08
            if remap and key not in self.quarantined_color:
                self.quarantined_color[key] = old
            baseline = self.quarantined_color.get(key)
            if baseline is not None:
                if baseline[0] - color[0] > .08 and color[1] - baseline[1] > .08:
                    remap = True
                else:
                    del self.quarantined_color[key]
            # Periodicity on original sampling, accepting arbitrary patch period 2..24.
            small = image[::max(1, image.shape[0] // 256), ::max(1, image.shape[1] // 256)].astype(np.float32) / 255.
            periodic = 0.
            height, width = small.shape[:2]
            rh, rw = height * 2 // 3, width * 2 // 3
            regions = [small] + [small[y:y+rh,x:x+rw] for y in (0,height-rh) for x in (0,width-rw)]
            for region in regions:
                for shift in range(2, min(25, min(region.shape[:2]) // 3)):
                    d1 = float(np.abs(region[:, shift:] - region[:, :-shift]).mean())
                    d2 = float(np.abs(region[:, 2*shift:] - region[:, :-2*shift]).mean())
                    e1 = float(np.abs(region[shift:] - region[:-shift]).mean())
                    e2 = float(np.abs(region[2*shift:] - region[:-2*shift]).mean())
                    periodic = max(periodic, min(d1 - d2, e1 - e2))
            self.last_metrics[key] = {"delta": delta, "periodicity": periodic, "color": color,
                                      "sha256": hashlib.sha256(image.tobytes()).hexdigest()}
            if (stationary and delta > self.threshold) or remap:
                events.append({"reason": "image_temporal_anomaly", "view": key, "delta": delta, "color_remap": bool(remap)})
            if periodic > .10:
                events.append({"reason": "image_periodic_texture", "view": key, "score": periodic})
            self.previous[key] = thumb
        return events
