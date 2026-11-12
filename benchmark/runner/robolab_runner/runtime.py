# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public doctor, recording, and validation workflows."""

from __future__ import annotations

import importlib.metadata
import json
import os
import sys
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from robolab_runner.capture import CaptureSession
from robolab_runner.capture_sources import capture_physics_substep
from robolab_runner.common import write_json_atomic
from robolab_runner.contract import V4TaskContract
from robolab_runner.direct_scoring import (
    capture_direct_step,
    create_direct_engine,
    finalize_direct_episode,
    reset_direct_engine,
)
from robolab_runner.environment_registry import register_runner_environment
from robolab_runner.errors import AssetError, RunnerError
from robolab_runner.policy.client import PolicyClient
from robolab_runner.policy.endpoint import inspect_endpoint
from robolab_runner.policy.observations import extract_observations, zero_observations
from robolab_runner.recorder import V4SubmissionRecorder
from robolab_runner.records import build_record, validate_record, write_failure
from robolab_runner.schema import submission_metadata_attribute_bytes
from robolab_runner.validator import finalize_capture, finalize_submission, validate_submission


def runner_version() -> str:
    try:
        return importlib.metadata.version("robolab-competition-runner")
    except importlib.metadata.PackageNotFoundError:
        return "4.3.0"


def _preflight_contract(contract_path: str | Path, asset_root: str | Path) -> tuple[V4TaskContract, Path]:
    contract = V4TaskContract.load(contract_path)
    root = Path(asset_root).expanduser().resolve()
    mismatches = contract.check_assets(root)
    if mismatches:
        raise AssetError("competition asset verification failed: " + "; ".join(mismatches[:5]))
    return contract, root


def _preflight_submission_metadata(
    contract: V4TaskContract,
    *,
    endpoint_scope: str,
    policy_id: str,
    policy_version: str,
) -> None:
    required = submission_metadata_attribute_bytes(
        contract,
        runner_version=runner_version(),
        endpoint_scope=endpoint_scope,
        policy_id=policy_id,
        policy_version=policy_version,
    )
    if required > contract.limits.max_attribute_bytes:
        raise RunnerError(
            "frozen contract cannot contain its mandatory submission metadata: "
            f"requires {required} aggregate attribute bytes, limit is {contract.limits.max_attribute_bytes}"
        )


def doctor(
    contract_path: str | Path,
    asset_root: str | Path,
    policy_endpoint: str,
    *,
    allow_remote_policy: bool = False,
) -> dict[str, Any]:
    contract, _root = _preflight_contract(contract_path, asset_root)
    version_mismatches = contract.check_runtime_versions()
    if version_mismatches:
        raise RunnerError("frozen runtime version mismatch: " + "; ".join(version_mismatches))
    with PolicyClient(contract, policy_endpoint, episode_id="doctor", allow_remote=allow_remote_policy) as client:
        identity = client.handshake()
        _preflight_submission_metadata(
            contract,
            endpoint_scope=client.endpoint_scope,
            policy_id=identity.policy_id,
            policy_version=identity.policy_version,
        )
        client.reset()
        actions, duration = client.infer(
            zero_observations(contract),
            step=0,
            effective_instruction=contract.instruction if contract.is_competition else None,
        )
        client.end_episode(steps=0, status="doctor")
        return {
            "status": "READY",
            "contract_id": contract.contract_id,
            "contract_sha256": contract.digest,
            "endpoint_scope": client.endpoint_scope,
            "policy_id": identity.policy_id,
            "policy_version": identity.policy_version,
            "action_chunk_shape": list(actions.shape),
            "inference_duration_s": duration,
        }


def _configure_asset_root(asset_root: Path) -> None:
    existing = sys.modules.get("robolab.constants")
    if existing is not None:
        configured = Path(str(getattr(existing, "ASSET_PACKAGE_ROOT", ""))).resolve()
        if configured != asset_root:
            raise RunnerError(
                "robolab.constants was imported before --asset-root was applied; "
                "start robolab-runner in a fresh process"
            )
    os.environ["ROBOLAB_COMPETITION_ASSET_ROOT"] = str(asset_root)


def _ensure_forensic_partial(path: Path) -> None:
    if not path.exists():
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.close(descriptor)


def _first_config_difference(expected: Any, actual: Any, path: str = "environment") -> str | None:
    """Describe the first JSON config difference without emitting a noisy full diff."""
    if isinstance(expected, dict) and isinstance(actual, dict):
        expected_keys = set(expected)
        actual_keys = set(actual)
        if missing := sorted(expected_keys - actual_keys):
            return f"{path} is missing live key {missing[0]!r}"
        if unexpected := sorted(actual_keys - expected_keys):
            return f"{path} has unexpected live key {unexpected[0]!r}"
        for key in sorted(expected_keys):
            difference = _first_config_difference(expected[key], actual[key], f"{path}.{key}")
            if difference is not None:
                return difference
        return None
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return f"{path} length differs: contract={len(expected)}, live={len(actual)}"
        for index, (expected_item, actual_item) in enumerate(zip(expected, actual, strict=True)):
            difference = _first_config_difference(expected_item, actual_item, f"{path}[{index}]")
            if difference is not None:
                return difference
        return None
    if type(expected) is not type(actual):
        return f"{path} type differs: contract={type(expected).__name__}, live={type(actual).__name__}"
    if expected != actual:
        return f"{path} differs: contract={expected!r}, live={actual!r}"
    return None


def record(
    contract_path: str | Path,
    asset_root: str | Path,
    policy_endpoint: str,
    output_dir: str | Path,
    *,
    device: str = "cuda:0",
    allow_remote_policy: bool = False,
) -> dict[str, Any]:
    contract, asset_root_path = _preflight_contract(contract_path, asset_root)
    version_mismatches = contract.check_runtime_versions()
    if version_mismatches:
        raise RunnerError("frozen runtime version mismatch: " + "; ".join(version_mismatches))
    endpoint_scope = inspect_endpoint(policy_endpoint, allow_remote=allow_remote_policy).scope
    root = Path(output_dir).expanduser().resolve()
    final_path = root / "submission.hdf5"
    partial_path = root / "submission.hdf5.partial"
    video_path = root / "episode.mp4"
    video_partial_path = root / "episode.mp4.partial"
    record_path = root / "record.json"
    failure_path = root / "failure.json"
    artifact_paths = [final_path, partial_path, record_path, failure_path]
    if contract.uses_full_capture:
        artifact_paths.extend([video_path, video_partial_path])
    for path in artifact_paths:
        if path.exists():
            raise FileExistsError(f"refusing to overwrite Runner artifact: {path}")
    root.mkdir(parents=True, exist_ok=True)
    _configure_asset_root(asset_root_path)

    client: PolicyClient | None = None
    env = None
    simulation_app = None
    attack_runtime = None
    direct_engine = None
    stage = "policy_handshake"
    step: int | None = None
    try:
        client = PolicyClient(contract, policy_endpoint, allow_remote=allow_remote_policy)
        identity = client.handshake()
        _preflight_submission_metadata(
            contract,
            endpoint_scope=endpoint_scope,
            policy_id=identity.policy_id,
            policy_version=identity.policy_version,
        )

        stage = "simulator_launch"
        import cv2  # noqa: F401 -- must precede isaaclab in this repository
        import torch
        from isaaclab.app import AppLauncher

        launcher = AppLauncher(headless=True, enable_cameras=True, device=device)
        simulation_app = launcher.app

        from robolab.constants import set_output_dir
        from robolab.core.environments.factory import get_envs
        from robolab.core.environments.runtime import create_env
        from robolab.core.scoring.metadata import action_semantics, serialize_environment_config
        from robolab.submission.contract import normalize_environment_config

        stage = "environment_create"
        register_runner_environment(contract)
        if contract.is_competition:
            from robolab_competition_tasks import get_scenario
            from robolab_competition_tasks.attacks import ScenarioAttackRuntime

            attack_runtime = ScenarioAttackRuntime(get_scenario(contract.scenario_id))
        matches = [contract.task_env] if contract.uses_full_capture else get_envs(task=[contract.task_class])
        if matches != [contract.task_env]:
            raise RunnerError(f"registered environment mismatch: expected {[contract.task_env]}, got {matches}")
        set_output_dir(str(root))
        env, env_cfg = create_env(
            contract.task_env,
            device=device,
            seed=contract.seed,
            num_envs=1,
            instruction_type="default",
            policy=f"competition_zmq_v{contract.policy_protocol_version}",
            disable_recorders=True,
            save_env_cfg=False,
        )
        normalized = normalize_environment_config(
            json.loads(serialize_environment_config(env.cfg)), package_root=asset_root_path
        )
        if normalized != contract.environment_config:
            difference = _first_config_difference(contract.environment_config, normalized)
            raise RunnerError(f"live environment config differs from the frozen V4 contract ({difference})")
        if str(env_cfg.instruction) != contract.instruction or int(env_cfg.seed) != contract.seed:
            raise RunnerError("live instruction/seed differs from the frozen V4 contract")
        if contract.uses_full_capture:
            action_type = str(getattr(env, "competition_action_type", ""))
            if action_type != contract.control.action_type:
                raise RunnerError("live action semantics differ from the frozen full-capture contract")
            joint_names = tuple(str(name) for name in env.scene["robot"].data.joint_names)
            if joint_names != contract.control.joint_names or len(joint_names) != contract.control.state_joint_dim:
                raise RunnerError("live robot joints differ from the frozen full-capture contract")
            capture_spec = getattr(env, "competition_capture_spec", None)
            if not callable(capture_spec):
                raise RunnerError("full-capture environment does not expose competition_capture_spec")
            live_capture = capture_spec()
            expected_capture = contract.runner_capture.to_dict()
            if live_capture != expected_capture:
                difference = _first_config_difference(expected_capture, live_capture, "runner_capture")
                raise RunnerError(f"live capture producers differ from the frozen contract ({difference})")
        else:
            action_type, arm_indices, action_indices = action_semantics(env)
            if (
                action_type != contract.control.action_type
                or tuple(arm_indices) != contract.control.arm_joint_indices
                or tuple(action_indices) != contract.control.action_joint_indices
            ):
                raise RunnerError("live action semantics differ from the frozen V4 contract")
        action_dim = int(
            getattr(getattr(env, "action_manager", None), "total_action_dim", None)
            or env.action_space.shape[-1]
        )
        if action_dim != contract.control.action_dim:
            raise RunnerError("live action width differs from the frozen V4 contract")
        if int(env.max_episode_length) > contract.limits.max_steps:
            raise RunnerError("live episode length exceeds the frozen V4 resource limit")
        direct_engine = create_direct_engine(contract, env)

        stage = "episode_reset"
        env.reset()
        obs, _ = env.reset()
        reset_direct_engine(contract, direct_engine)
        client.reset()

        import omni.kit.app
        import omni.timeline

        timeline = omni.timeline.get_timeline_interface()
        kit_app = omni.kit.app.get_app()
        action_chunk: deque[np.ndarray] = deque()
        query_durations: list[float] = []
        result: dict[str, Any] = {}
        direct_final_state = None
        direct_events = None
        recorder_context = (
            CaptureSession(
                partial_path,
                video_partial_path,
                contract,
                runner_version=runner_version(),
                endpoint_scope=endpoint_scope,
                policy_id=identity.policy_id,
                policy_version=identity.policy_version,
            )
            if contract.uses_full_capture
            else V4SubmissionRecorder(
                partial_path,
                contract,
                runner_version=runner_version(),
                endpoint_scope=endpoint_scope,
                policy_id=identity.policy_id,
                policy_version=identity.policy_version,
            )
        )
        with recorder_context as recorder:
            if contract.uses_full_capture:
                callback = lambda live_env, control_step, physics_step, sim_time_s: capture_physics_substep(
                    recorder, live_env, control_step, physics_step, sim_time_s
                )
                setter = getattr(env, "set_physics_capture_callback", None)
                if not callable(setter):
                    raise RunnerError("full-capture environment is not CaptureRobolabEnv")
                setter(callback)
                for event in client.drain_network_events():
                    recorder.record_network_event(event)
            recorder.record_initial_state(env)
            for step in range(int(env.max_episode_length)):
                while not timeline.is_playing():
                    kit_app.update()
                if not action_chunk:
                    stage = "policy_infer"
                    public_obs = extract_observations(obs, contract)
                    try:
                        physics_step = int(round(step * contract.control.control_dt / float(env.physics_dt)))
                        if contract.is_competition:
                            if attack_runtime is None:
                                raise RunnerError("V4.2 attack runtime was not initialized")
                            public_obs = attack_runtime.transform_observations(
                                public_obs,
                                step=step,
                                physics_step=physics_step,
                            )
                            effective_instruction = attack_runtime.effective_instruction(
                                contract.instruction,
                                step=step,
                                physics_step=physics_step,
                            )
                        else:
                            effective_instruction = None
                        chunk, duration = client.infer(
                            public_obs,
                            step=step,
                            effective_instruction=effective_instruction,
                        )
                    finally:
                        if contract.uses_full_capture:
                            for event in client.drain_network_events():
                                recorder.record_network_event(event)
                    query_durations.append(duration)
                    recorder.record_policy_query(
                        step=step,
                        inference_duration_s=duration,
                        chunk_length=int(chunk.shape[0]),
                    )
                    action_chunk.extend(np.asarray(row, dtype=np.float32) for row in chunk)
                action_numpy = action_chunk.popleft()
                if contract.is_competition:
                    if attack_runtime is None:
                        raise RunnerError("V4.2 attack runtime was not initialized")
                    physics_step = int(round(step * contract.control.control_dt / float(env.physics_dt)))
                    action_numpy = attack_runtime.transform_action(
                        action_numpy,
                        step=step,
                        physics_step=physics_step,
                    )
                action = torch.as_tensor(action_numpy, dtype=torch.float32, device=env.device).reshape(1, action_dim)
                if contract.uses_full_capture:
                    recorder.record_pre_step(
                        action,
                        step_index=step,
                        action_time_s=step * contract.control.control_dt,
                    )
                else:
                    recorder.record_pre_step(action, control_time_s=step * contract.control.control_dt)
                stage = "environment_step"
                obs, _reward, _term, _trunc, _info = env.step(action)
                if contract.uses_full_capture:
                    recorder.record_post_step(
                        env,
                        step_index=step,
                        state_time_s=(step + 1) * contract.control.control_dt,
                    )
                    if contract.is_v4_3:
                        direct_state = capture_direct_step(contract, env, direct_engine)
                        recorder.record_direct_step(
                            direct_state,
                            step_index=step,
                            terminated=bool(env.all_terminated),
                        )
                else:
                    recorder.record_post_step(env)
                if env.all_terminated:
                    break
            result = dict(env.get_env_results()[0])
            if contract.is_v4_3:
                direct_final_state, direct_result, direct_events = finalize_direct_episode(
                    contract,
                    env,
                    direct_engine,
                    steps=recorder.num_samples,
                )
                result.update(direct_result)
            result.update(
                {
                    "authority": "runner_direct_capture" if contract.is_v4_3 else "participant_non_authoritative",
                    "policy": f"robolab-policy-zmq/{contract.policy_protocol_version}",
                    "recorded_steps": recorder.num_samples,
                }
            )
            if contract.is_competition:
                if attack_runtime is None:
                    raise RunnerError("V4.2 attack runtime was not initialized")
                result["scenario"] = {
                    "id": contract.scenario_id,
                    "sha256": contract.scenario_sha256,
                    "variant": contract.scenario_variant,
                }
                result["attack_audit"] = attack_runtime.audit_summary()
            stage = "policy_end_episode"
            try:
                client.end_episode(
                    steps=recorder.num_samples,
                    status="success" if bool(result.get("success", False)) else "task_failure",
                )
            finally:
                if contract.uses_full_capture:
                    for event in client.drain_network_events():
                        recorder.record_network_event(event)
            if contract.is_v4_3:
                recorder.complete(
                    result,
                    direct_final_state=direct_final_state,
                    direct_events=direct_events,
                )
            else:
                recorder.complete(result)

        stage = "publication"
        video_validation = None
        if contract.uses_full_capture:
            validation, video_validation = finalize_capture(
                partial_path,
                final_path,
                video_partial_path,
                video_path,
                contract,
            )
        else:
            validation = finalize_submission(partial_path, final_path, contract)
        if not validation.valid:
            messages = "; ".join(issue.message for issue in validation.errors[:5])
            raise RunnerError(f"official V4 recorder produced invalid bytes: {messages}")
        if contract.uses_full_capture and not video_validation["valid"]:
            raise RunnerError("official full-capture recorder produced an invalid episode.mp4")
        record_value = build_record(
            final_path,
            contract,
            runner_version=runner_version(),
            endpoint_scope=endpoint_scope,
            policy_id=identity.policy_id,
            policy_version=identity.policy_version,
            steps=int(validation.num_samples or 0),
            query_durations_s=query_durations,
            video_path=video_path if contract.uses_full_capture else None,
            video_validation=video_validation,
        )
        write_json_atomic(record_path, record_value)
        return record_value
    except Exception as error:
        print(
            f"robolab-runner record failed at {stage}: {type(error).__name__}: {error}",
            file=sys.stderr,
            flush=True,
        )
        if final_path.exists() and not record_path.exists() and not partial_path.exists():
            # record.json is the commit marker. Demote an uncommitted formal
            # artifact set back to forensic partials.
            os.replace(final_path, partial_path)
        if contract.uses_full_capture and video_path.exists() and not record_path.exists() and not video_partial_path.exists():
            os.replace(video_path, video_partial_path)
        if not final_path.exists():
            _ensure_forensic_partial(partial_path)
            if contract.uses_full_capture:
                _ensure_forensic_partial(video_partial_path)
            code = error.code if isinstance(error, RunnerError) else "SIMULATION_ERROR"
            write_failure(
                root,
                contract,
                stage=stage,
                step=step,
                code=code,
                endpoint_scope=endpoint_scope,
                partial_exists=True,
            )
        raise
    finally:
        if client is not None:
            client.close()
        if env is not None:
            env.close()
        if simulation_app is not None:
            simulation_app.close()


def validate_directory(contract_path: str | Path, submission_dir: str | Path) -> dict[str, Any]:
    contract = V4TaskContract.load(contract_path)
    root = Path(submission_dir).expanduser().resolve()
    submission_path = root / "submission.hdf5"
    validation = validate_submission(submission_path, contract)
    chain_error: str | None = None
    record_value: dict[str, Any] | None = None
    if validation.valid:
        try:
            record_value = validate_record(root, contract, validation)
        except Exception as error:
            chain_error = f"{type(error).__name__}: {error}"
    result = {
        "status": "VALID_SUBMISSION" if validation.valid and chain_error is None else "INVALID_SUBMISSION",
        "contract_id": contract.contract_id,
        "contract_sha256": contract.digest,
        "validation": validation.to_dict(),
        "record_valid": chain_error is None and record_value is not None,
        "record_error": chain_error,
    }
    return result


__all__ = ["doctor", "record", "runner_version", "validate_directory"]
