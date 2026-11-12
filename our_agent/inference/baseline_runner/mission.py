"""Simulation-independent phase ownership and budget rules."""
from dataclasses import dataclass
import math

class Stop(RuntimeError):
    pass

def validate_response(response, phase, epoch, sample_time, models):
    if response.get("phase") != phase or response.get("epoch") != epoch or response.get("sample_time_s") != sample_time:
        raise Stop("STALE_MODEL_RESPONSE")
    if response.get("models") != models:
        raise Stop("MODEL_BINDING_MISMATCH")

@dataclass
class Budget:
    start: float
    duration: float
    maximum_actions: int
    actions: int = 0

    def remaining(self, now):
        return max(0., self.start + self.duration - now)

    def check(self, now, seconds=.1):
        if not math.isfinite(now):
            raise Stop("NONFINITE_CLOCK")
        if self.remaining(now) + 1e-8 < seconds:
            raise Stop("FORMAL_DURATION_EXHAUSTED")
        if self.actions >= self.maximum_actions:
            raise Stop("FORMAL_ACTION_BUDGET_EXHAUSTED")

class Mission:
    def __init__(self, backend, task, pick_seconds=30., place_seconds=30.):
        self.b = backend
        self.task = task
        self.windows = {"PICK": pick_seconds, "PUSH": pick_seconds, "BUTTON": pick_seconds,
                        "OPEN": pick_seconds, "PLACE": place_seconds}
        self.epoch = 0
        self.phase = "BOOT"
        self.events = []
        self.carrying = False
        self.grasp_success = False
        self.place_success = False
        self.last_operation = None

    def switch(self, phase):
        self.epoch += 1
        self.phase = phase
        event = {"phase": phase, "epoch": self.epoch, "time_s": self.b.now(), "status": "STARTED"}
        self.events.append(event)
        self.b.event(event)
        self.b.barrier(self.epoch)
        return event

    def run(self):
        failure = None
        incomplete = []
        try:
            if hasattr(self.b, "initialize"):
                self.b.initialize()
            for phase in self.task["stages"]:
                event = self.switch(phase)
                if phase.startswith("UNSUPPORTED_"):
                    incomplete.append(phase)
                    event.update(status="UNIMPLEMENTED")
                elif phase.startswith("NAV_"):
                    event.update(self.b.navigate(phase, self.epoch, self.carrying))
                    self.carrying = bool(event.get("carrying", self.carrying))
                elif phase == "HANDOFF":
                    self.carrying = self.b.handoff(self.epoch, self.carrying, 10.)
                    event.update(status="COMPLETE", carrying=self.carrying)
                else:
                    outcome = self.b.manipulate(phase, self.epoch, self.windows[phase])
                    self.last_operation = phase
                    event.update(outcome)
                    if phase == "PLACE":
                        self.place_success |= bool(outcome.get("physical_success"))
                    else:
                        self.grasp_success |= bool(outcome.get("grasp_success"))
                    self.carrying = bool(outcome.get("carrying", False))
                event["end_time_s"] = self.b.now()
                self.b.event(dict(event))
        except Exception as error:
            failure = f"{type(error).__name__}: {error}"
            if self.events:
                self.events[-1].update(status="STOPPED", failure=failure, end_time_s=self.b.now())
                self.b.event(dict(self.events[-1]))
        finally:
            try:
                self.b.stop(self.epoch + 1)
            except Exception as error:
                failure = (failure + "; " if failure else "") + "STOP_FAILED:" + str(error)
        return {"schema": "continuous-paper200-result/v1" if self.task.get("benchmark_id") else "continuous-q01-q24-result/v1",
                "question": self.task["question"], "benchmark_id": self.task.get("benchmark_id"),
                "task_identity": self.task.get("task_identity"),
                "flow_completed": failure is None and not incomplete,
                "physical_grasp_success": self.grasp_success, "physical_place_success": self.place_success,
                "return_arrived": any(e["phase"] in ("NAV_RETURN", "NAV_DELIVERY") and e.get("status") == "COMPLETE" for e in self.events),
                "failure": failure, "stopped_stage": self.phase, "unimplemented_stages": incomplete,
                "events": self.events, "continuous_world": True, "task_accepted": False,
                "formal_task_score": None, "hardware_authorized": False}
