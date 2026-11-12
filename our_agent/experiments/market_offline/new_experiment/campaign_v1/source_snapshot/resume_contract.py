"""Validate data/recipe identity and explicit continuation schedule changes."""


def validate_resume(old, requested, step, restart_schedule=False):
    if requested["schedule_steps"] <= step:
        raise ValueError("target steps must exceed checkpoint step")
    result = dict(requested)
    if restart_schedule:
        allowed = {"lr", "schedule_steps", "scheduler_origin"}
        if {k: v for k, v in old.items() if k not in allowed} != {
            k: v for k, v in result.items() if k not in allowed
        }:
            raise ValueError("restart schedule cannot change data, seed, batch or phase mix")
        result["scheduler_origin"] = step
    else:
        if "scheduler_origin" in old:
            result["scheduler_origin"] = old["scheduler_origin"]
        if old != result:
            raise ValueError("resume data or recipe changed; schedule changes require --restart-schedule")
    return result
