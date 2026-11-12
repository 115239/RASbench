"""Local vision-service client: allowlisted public pixels, state and text only."""
from __future__ import annotations
from dataclasses import fields
import urllib.request
import time
import numpy as np
from .runtime_types import VisualEvidence
from .wire import pack, unpack


class VisionClient:
    def __init__(self, endpoint, timeout=45.):
        if not endpoint.startswith(('http://127.0.0.1:', 'http://localhost:')):
            raise ValueError('vision service must be local')
        self.endpoint, self.timeout = endpoint, timeout
        self.last_error = None
        self.requests = self.failures = 0
        self.total_latency_s = 0.

    def __call__(self, frame, spec):
        started = time.perf_counter()
        self.requests += 1
        payload = {'episode_id': frame.episode_id, 'sequence': frame.sequence, 'time_s': frame.time_s,
                   'phase': frame.phase, 'instruction': spec.instruction,
                   'constraints': spec.constraints, 'state': frame.state,
                   'images': frame.images, 'calibration': frame.calibration}
        try:
            request = urllib.request.Request(self.endpoint + '/observe', data=pack(payload),
                                             headers={'Content-Type': 'application/msgpack'})
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                result = unpack(response.read(32 * 1024**2))
            allowed = {f.name for f in fields(VisualEvidence)}
            if set(result) != allowed:
                raise ValueError('invalid vision evidence fields')
            result['views'], result['uncertainty'] = tuple(result['views']), tuple(result['uncertainty'])
            self.last_error = None
            return VisualEvidence(**result)
        except Exception as error:
            self.failures += 1
            self.last_error = type(error).__name__
            return VisualEvidence(frame.episode_id, frame.sequence, frame.time_s,
                                  uncertainty=('vision_service_unavailable:' + self.last_error,))
        finally:
            self.total_latency_s += time.perf_counter()-started

    def metrics(self):
        return {'requests':self.requests,'failed_requests':self.failures,'extra_wall_latency_s':self.total_latency_s}
