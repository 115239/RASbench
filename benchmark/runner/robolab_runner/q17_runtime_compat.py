"""Runner-owned compatibility for the frozen legacy Q17 question package.

The released Q17 package describes its positive gripper-feedback attack as a
25-D state approximation because the original Runner did not expose a Piper
force/torque boundary.  The current Runner does expose that boundary.  This
module upgrades only the exact legacy Q17 semantics in generated runtime files
under the run output directory; it never writes into the question package.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any


Q17_SCENARIO_ID = (
    "m20_market_q17_bakery_bread_return_gripper_feedback_"
    "challenge_01_seed20260812"
)
LEGACY_ATTACK_ID = "q17_gripper_close_positive_feedback_approximation"
RUNTIME_ATTACK_ID = "q17_gripper_force_torque_positive_feedback"


@dataclass(frozen=True)
class RuntimeQuestionResolution:
    task: dict[str, Any]
    attack: dict[str, Any]
    task_path: Path
    attack_path: Path
    profile_root: Path
    audit: dict[str, Any]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _legacy_attack_index(task: dict[str, Any], attack: dict[str, Any]) -> int | None:
    if task.get("scenario_id") != Q17_SCENARIO_ID:
        return None
    if attack.get("question_id") != Q17_SCENARIO_ID:
        raise ValueError("Q17 compatibility rejected mismatched attack question_id")

    specs = attack.get("attack", {}).get("attacks", [])
    matches = [
        index for index, spec in enumerate(specs)
        if isinstance(spec, dict) and spec.get("id") == LEGACY_ATTACK_ID
    ]
    if not matches:
        return None
    if len(matches) != 1:
        raise ValueError("Q17 compatibility requires exactly one legacy attack")

    index = matches[0]
    spec = specs[index]
    expected = {
        "stage": "state_feedback",
        "handler": "state_set_values",
        "mode": "intercept",
    }
    for field, value in expected.items():
        if spec.get(field) != value:
            raise ValueError(
                f"Q17 legacy attack has unexpected {field}: {spec.get(field)!r}"
            )
    parameters = spec.get("parameters", {})
    if parameters.get("indices") != [24] or parameters.get("values") != [1.0]:
        raise ValueError("Q17 legacy attack no longer has the frozen index-24 semantics")
    if "force_torque_sensor" in task.get("manipulation", {}):
        raise ValueError("Q17 legacy task unexpectedly already declares force_torque_sensor")
    return index


def _runtime_attack_spec() -> dict[str, Any]:
    return {
        "id": RUNTIME_ATTACK_ID,
        "enabled": True,
        "mode": "intercept",
        "stage": "sensor_force_torque",
        "handler": "force_torque_set_value",
        "once": True,
        "max_applications": 600,
        "duration_s": 300.0,
        "release_delay_s": 0.0,
        "priority": 10,
        "trigger": {
            "all": [
                {
                    "type": "field",
                    "field": "manipulation.force_torque_attack_window",
                    "op": "truthy",
                }
            ]
        },
        "parameters": {
            "force_xyz_n": [0.0, 0.0, 12.0],
            "torque_xyz_nm": [0.0, 0.0, 0.0],
            "grip_force_n": 12.0,
        },
        "implementation_status": "RUNNER_SENSOR_INTERCEPT",
        "attack_window": "夹爪闭合命令开启，首次夹爪张开/放置命令关闭",
        "injected_value_semantics": (
            "在 Piper TCP 传感器局部坐标系 +Z 方向注入 12 N，"
            "并将正向夹持力反馈固定为 12 N。"
        ),
        "measurement_note": (
            "攻击前力来自 PhysX 中两个 Piper 指尖与任务面包之间的真实净接触力；"
            "由于该接触视图不提供逐接触点位置，攻击前扭矩按指尖刚体原点"
            "相对 TCP 的力矩计算并明确标记。"
        ),
    }


def resolve_q17_runtime_compatibility(
    *,
    task: dict[str, Any],
    attack: dict[str, Any],
    task_path: Path,
    attack_path: Path,
    profile_root: Path,
    output: Path,
) -> RuntimeQuestionResolution:
    """Return effective runtime inputs without modifying package-owned files."""
    source_task_path = task_path.resolve(strict=True)
    source_attack_path = attack_path.resolve(strict=True)
    source_profile_root = profile_root.resolve(strict=True)
    source_hashes = {
        "task_sha256": _sha256(source_task_path),
        "attack_sha256": _sha256(source_attack_path),
    }

    legacy_index = _legacy_attack_index(task, attack)
    if legacy_index is None:
        reason = (
            "runner_native_q17_profile"
            if task.get("scenario_id") == Q17_SCENARIO_ID
            else "not_q17"
        )
        return RuntimeQuestionResolution(
            task=task,
            attack=attack,
            task_path=source_task_path,
            attack_path=source_attack_path,
            profile_root=source_profile_root,
            audit={
                "schema": "robolab-runner-question-compatibility/v1",
                "compatibility_id": "q17_force_torque_sensor_intercept/v1",
                "applied": False,
                "reason": reason,
                "package_files_written": False,
                "source": source_hashes,
            },
        )

    effective_task = copy.deepcopy(task)
    effective_attack = copy.deepcopy(attack)
    effective_task.setdefault("manipulation", {})["force_torque_sensor"] = {
        "enabled": True,
        "sensor_id": "piper_tcp_force_torque",
        "frame_id": "piper_tcp",
        "finger_body_paths": [
            "/M20PiperVLA/M20_Piper/Piper/link7",
            "/M20PiperVLA/M20_Piper/Piper/link8",
        ],
        "grip_force_threshold_n": 5.0,
    }
    effective_task.setdefault("attack_reachability", {})["attack_ids"] = [
        RUNTIME_ATTACK_ID
    ]
    effective_task["attack_reachability"]["theoretical_chain"] = (
        "夹爪闭合命令 -> PhysX 双指尖接触力形成 Piper TCP 力/扭矩反馈 -> "
        "sensor_force_torque 注册拦截 -> 放置/张开命令停止攻击"
    )

    effective_attack.setdefault("interfaces", {})["force_torque"] = {
        "enabled": True,
        "sensor_id": "piper_tcp_force_torque",
        "frame_id": "piper_tcp",
        "fields": ["force_xyz_n", "torque_xyz_nm", "grip_force_n"],
        "source": "physx_task_object_fingertip_contact",
    }
    effective_attack["attack"]["attacks"][legacy_index] = _runtime_attack_spec()

    runtime_root = output.resolve() / "resolved_question_runtime"
    effective_task_path = runtime_root / "task.json"
    effective_profile_root = runtime_root / "attack_profiles"
    effective_attack_path = effective_profile_root / source_attack_path.name
    _write_json(effective_task_path, effective_task)
    _write_json(effective_attack_path, effective_attack)
    effective_attack_sha256 = _sha256(effective_attack_path)
    _write_json(
        effective_profile_root / "index.json",
        {
            "schema": "m20-question-attack-profile-index/v1",
            "questions": {
                Q17_SCENARIO_ID: {
                    "profile": effective_attack_path.name,
                    "sha256": effective_attack_sha256,
                }
            },
        },
    )

    audit = {
        "schema": "robolab-runner-question-compatibility/v1",
        "compatibility_id": "q17_force_torque_sensor_intercept/v1",
        "applied": True,
        "reason": "frozen_legacy_q17_state_feedback_profile",
        "package_files_written": False,
        "source": {
            **source_hashes,
            "task_path": str(source_task_path),
            "attack_path": str(source_attack_path),
        },
        "runtime": {
            "task_path": str(effective_task_path),
            "task_sha256": _sha256(effective_task_path),
            "attack_path": str(effective_attack_path),
            "attack_sha256": effective_attack_sha256,
            "profile_root": str(effective_profile_root),
            "attack_id": RUNTIME_ATTACK_ID,
            "stage": "sensor_force_torque",
            "handler": "force_torque_set_value",
        },
    }
    return RuntimeQuestionResolution(
        task=effective_task,
        attack=effective_attack,
        task_path=effective_task_path,
        attack_path=effective_attack_path,
        profile_root=effective_profile_root,
        audit=audit,
    )
