import json

import h5py

from m20_runtime.attack_tools.attack_orchestrator import AttackOrchestrator
from robolab_runner.attack_hdf5_enricher import (
    _attack_marker_rows,
    _trigger_physics_step,
    _write_json_rows,
)


def _config():
    return {
        "attacks": [
            {
                "id": "q102_control",
                "mode": "intercept",
                "stage": "control_base",
                "handler": "identity",
                "trigger": {
                    "all": [
                        {
                            "type": "field",
                            "field": "simulation.task_time_s",
                            "op": "gte",
                            "value": 5.0,
                        }
                    ]
                },
                "once": False,
            }
        ]
    }


def _snapshot(time_s, physics_step):
    return {
        "simulation": {
            "time_s": 100.0 + time_s,
            "physics_step": 20000 + physics_step,
            "task_time_s": time_s,
            "task_physics_step": physics_step,
        }
    }


def test_marker_only_evaluates_trigger_without_arming_effect():
    events = []
    orchestrator = AttackOrchestrator(
        _config(), {"identity": lambda payload, parameters, context: payload}, audit=events.append
    )

    assert orchestrator.observe_markers_only(_snapshot(4.9, 980)) == []
    assert orchestrator.observe_markers_only(_snapshot(5.0, 1000)) == ["q102_control"]
    assert orchestrator.observe_markers_only(_snapshot(6.0, 1200)) == []
    assert orchestrator.active_intercepts == []
    assert orchestrator.armed == []

    markers = [event for event in events if event["event"] == "condition_triggered"]
    assert markers == [
        {
            "event": "condition_triggered",
            "attack_id": "q102_control",
            "attack_instance_id": "q102_control:1",
            "time_s": 5.0,
            "physics_step": 1000,
            "marker_only": True,
            "effect_applied": False,
            "first_confirmed_after_observation_gap": False,
            "evidence": markers[0]["evidence"],
        }
    ]
    assert markers[0]["evidence"]["event"] == "condition_triggered"


def test_observation_gap_is_preserved_on_first_confirmed_marker():
    events = []
    orchestrator = AttackOrchestrator(
        _config(), {"identity": lambda payload, parameters, context: payload}, audit=events.append
    )
    assert orchestrator.observe_markers_only({"simulation": {"task_time_s": 5.0}}) == []
    assert orchestrator.observe_markers_only(_snapshot(5.1, 1020)) == ["q102_control"]
    marker = next(event for event in events if event["event"] == "condition_triggered")
    assert marker["first_confirmed_after_observation_gap"] is True


def test_trigger_physics_step_uses_exact_condition_marker_step():
    marker = {
        "event": "condition_triggered",
        "time_s": 5.0,
        "physics_step": 1000,
        # A later runtime boundary must never replace the trigger step.
        "runtime_context": {"time_s": 5.01, "physics_step": 1002},
    }
    assert _trigger_physics_step(marker) == 1000
    assert _trigger_physics_step(None) is None
    assert _trigger_physics_step({"physics_step": True}) is None


def test_effect_marker_reuses_condition_instance_and_requires_payload_change():
    events = []
    orchestrator = AttackOrchestrator(
        _config(),
        {"identity": lambda payload, parameters, context: {**payload, "yaw": -payload["yaw"]}},
        audit=events.append,
    )
    assert orchestrator.observe(_snapshot(5.0, 1000)) == ["q102_control"]
    result = orchestrator.process_intercept(
        "control_base", {"yaw": 0.5}, {"time_s": 5.01, "physics_step": 1002}
    )
    assert result == {"yaw": -0.5}
    markers = [event for event in events if event["event"] in {"condition_triggered", "effect_first_applied"}]
    assert [event["event"] for event in markers] == ["condition_triggered", "effect_first_applied"]
    assert {event["attack_instance_id"] for event in markers} == {"q102_control:1"}
    assert markers[1]["marker_only"] is False
    assert markers[1]["effect_applied"] is True


def test_all_conditions_are_marked_but_only_selected_effect_is_armed():
    events = []
    config = {
        "attacks": [
            {
                "id": "selected_control",
                "mode": "intercept",
                "stage": "control_base",
                "handler": "identity",
                "trigger": {
                    "all": [{
                        "type": "field",
                        "field": "simulation.task_time_s",
                        "op": "gte",
                        "value": 5.0,
                    }]
                },
                "effect_enabled": True,
            },
            {
                "id": "unselected_visual",
                "mode": "immediate",
                "stage": "immediate_visual_material",
                "handler": "identity",
                "trigger": {
                    "all": [{
                        "type": "field",
                        "field": "simulation.task_time_s",
                        "op": "gte",
                        "value": 5.0,
                    }]
                },
                "effect_enabled": False,
            },
        ]
    }
    orchestrator = AttackOrchestrator(
        config,
        {"identity": lambda payload, parameters, context: payload},
        audit=events.append,
    )

    assert orchestrator.observe(_snapshot(5.0, 1000)) == ["selected_control"]
    assert orchestrator.last_condition_markers == [
        "selected_control",
        "unselected_visual",
    ]
    assert [spec.attack_id for spec, _ in orchestrator.active_intercepts] == [
        "selected_control"
    ]
    assert orchestrator.armed == []
    assert orchestrator.observe(_snapshot(5.1, 1020)) == []
    condition_ids = [
        event["attack_id"]
        for event in events
        if event["event"] == "condition_triggered"
    ]
    assert condition_ids == ["selected_control", "unselected_visual"]


def test_hdf5_attack_markers_use_handoff_offset_layout(tmp_path):
    audit = [
        {
            "event": "condition_triggered",
            "attack_id": "q102_control",
            "attack_instance_id": "q102_control:1",
            "time_s": 5.0,
            "physics_step": 1000,
            "marker_only": True,
            "effect_applied": False,
            "first_confirmed_after_observation_gap": False,
            "evidence": {"event": "condition_triggered"},
        }
    ]
    rows = _attack_marker_rows(audit, _config()["attacks"])
    path = tmp_path / "markers.hdf5"
    with h5py.File(path, "w") as output:
        group = output.create_group("attack_markers")
        _write_json_rows(group, rows)
    with h5py.File(path, "r") as stored:
        group = stored["attack_markers"]
        offsets = group["offsets"][...]
        raw = group["utf8"][...]
        decoded = [
            json.loads(raw[int(start):int(end)].tobytes().decode("utf-8"))
            for start, end in zip(offsets[:-1], offsets[1:])
        ]
        assert group.attrs["layout"] == "offset-indexed-json/v1"
        assert decoded == rows
