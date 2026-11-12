from __future__ import annotations
import numpy as np
from .contracts import PerceptionEvidence


class GripperLatch:
    def __init__(self, goal, config):
        self.goal, self.cfg = goal, config
        self.command = 0.
        self.status = "EMPTY"
        self.verify_age = 0
        self.confirmations = 0
        self.regrasp = 0
        self.missing = 0
        self.close_tcp_z = None
        self.now = None
        self.verify_started = None

    def observe(self, state, evidence, sequence, *, sample_time_s=None):
        events = []
        self.now = sample_time_s
        if self.command > .5 and state[24] < .2:
            events.append({"reason": "gripper_feedback_disagreement"})
        if self.command < .2 and state[24] > .8:
            events.append({"reason": "gripper_positive_feedback_untrusted"})
        valid = (isinstance(evidence, PerceptionEvidence)
                 and evidence.sequence == sequence
                 and evidence.object_id == self.goal.object_id
                 and len(set(evidence.views) & {"image", "rear_image", "wrist_image"}) >= 2
                 and evidence.object_tracks_tcp and np.isfinite(evidence.lift_m)
                 and evidence.lift_m >= .03 and .9 <= evidence.confidence <= 1.)
        if self.command > .5:
            self.verify_age += 1
            self.confirmations = self.confirmations + 1 if valid else 0
            if self.confirmations >= 2:
                self.status, self.missing = "CONFIRMED", 0
            elif self.status == "CONFIRMED":
                self.missing += 1
                # Every missing evidence frame blocks transport immediately.
                self.status = "VERIFY"
                self.verify_age = 0
                self.verify_started = sample_time_s
                self.regrasp += 1
                events.append({"reason": "grasp_evidence_lost", "retry": self.regrasp})
            expired = (sample_time_s - self.verify_started >= self.cfg.grasp_verify_seconds
                       if sample_time_s is not None and self.verify_started is not None
                       else self.verify_age >= self.cfg.grasp_verify_chunks)
            if self.status == "VERIFY" and expired:
                events.append({"reason": "grasp_unverified_timeout"})
            if self.regrasp > self.cfg.max_regrasp:
                events.append({"reason": "grasp_retry_exhausted"})
        return events

    def shield(self, actions, state):
        events = []
        out = actions.copy()
        # Split at each gripper intent change: a chunk cannot close then move/release unseen.
        mask = out[:, 9] > .5
        changes = np.flatnonzero(mask != mask[0])
        if len(changes):
            out = out[:int(changes[0])]
            events.append({"reason": "gripper_chunk_boundary"})
        if out[0, 9] > .5 and self.command <= .5:
            self.status, self.verify_age, self.confirmations = "VERIFY", 0, 0
            self.verify_started = self.now
            self.close_tcp_z = float(state[20])
            # Obtain feedback after closure before any further manipulation.
            out = out[:1]
            out[:, :9] = 0.
            events.append({"reason": "grasp_verification_started"})
        if self.status == "VERIFY":
            out[:, :3] = 0.
            out[:, 3:9] = np.clip(out[:, 3:9], -.005, .005)
            out[:, 9] = 1.
            events.append({"reason": "unverified_grasp_transport_blocked"})
        if self.status == "CONFIRMED" and out[0, 9] <= .5:
            # Destination proximity is necessary, not sufficient for successful placement.
            dest = self.goal.destination
            near = dest is not None and np.linalg.norm(state[:2] - np.asarray(dest[:2])) < .6
            stationary = np.linalg.norm(state[6:8]) < .03
            if not (near and stationary):
                out[:, 9] = 1.
                events.append({"reason": "release_outside_destination_blocked"})
            else:
                out = out[:1]
                out[:, :9] = 0.
                self.status = "EMPTY"
                events.append({"reason": "release_requested_unverified_placement"})
        self.command = float(out[-1, 9])
        return out, events
