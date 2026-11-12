import json
from pathlib import Path

from robolab_runner.m20_question_runner import (
    build_attack_marker_catalog,
    sha256,
)


SCENARIO_ID = "q102-scenario"


def _profile(attack_id: str, stage: str, mode: str, handler: str) -> dict:
    return {
        "schema": "m20-attack-experiment/v3",
        "question_id": SCENARIO_ID,
        "scenario_id": SCENARIO_ID,
        "experiment": {"name": attack_id, "output_dir": "runner_output"},
        "attack": {
            "enabled": True,
            "attacks": [
                {
                    "id": attack_id,
                    "enabled": True,
                    "mode": mode,
                    "stage": stage,
                    "handler": handler,
                    "trigger": {
                        "all": [
                            {
                                "type": "field",
                                "field": "simulation.task_time_s",
                                "op": "gte",
                                "value": 0.0,
                            }
                        ]
                    },
                }
            ],
        },
    }


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_catalog_keeps_all_conditions_and_selects_one_effect(tmp_path):
    package = tmp_path / "Q102"
    profile_root = package / "data/attack/profiles"
    control_path = profile_root / "control.json"
    visual_path = profile_root / "visual.json"
    control = _profile(
        "q102_control",
        "control_base",
        "intercept",
        "affine_actions",
    )
    visual = _profile(
        "q102_visual",
        "immediate_visual_material",
        "immediate",
        "install_visual_material_override",
    )
    _write(control_path, control)
    _write(visual_path, visual)
    _write(
        profile_root / "index.json",
        {
            "schema": "m20-question-attack-profile-index/v1",
            "questions": {
                SCENARIO_ID: {
                    "profile": control_path.name,
                    "sha256": sha256(control_path),
                }
            },
            "variants": {
                "control": {
                    "profile": control_path.name,
                    "sha256": sha256(control_path),
                },
                "visual": {
                    "profile": visual_path.name,
                    "sha256": sha256(visual_path),
                },
            },
        },
    )
    contract = {
        "scenario_id": SCENARIO_ID,
        "attack_config": "data/attack/profiles/control.json",
        "attack_variants": {
            "control": "data/attack/profiles/control.json",
            "visual": "data/attack/profiles/visual.json",
        },
    }

    merged, audit = build_attack_marker_catalog(
        package=package,
        contract=contract,
        scenario_id=SCENARIO_ID,
        selected_attack=visual,
        selected_source_path=visual_path,
    )

    specs = merged["attack"]["attacks"]
    assert [spec["id"] for spec in specs] == ["q102_control", "q102_visual"]
    assert {spec["id"]: spec["effect_enabled"] for spec in specs} == {
        "q102_control": False,
        "q102_visual": True,
    }
    assert audit["configured_attack_ids"] == ["q102_control", "q102_visual"]
    assert audit["selected_effect_attack_ids"] == ["q102_visual"]
