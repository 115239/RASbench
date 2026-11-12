# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hard-coded task-family registration boundary for the public Runner."""

from __future__ import annotations

from robolab_runner.contract import SPOT_TASK_FAMILY, V4TaskContract
from robolab_runner.errors import RunnerError

SPOT_TASK_ENV = "SpotWarehouseNavigation-v0"
SPOT_TASK_CLASS = "SpotWarehouseNavigationTask"


def register_runner_environment(contract: V4TaskContract) -> None:
    """Register only known code; a contract can never provide an import path."""
    if contract.is_competition:
        from robolab_competition_tasks import get_scenario, register_scenario

        if contract.scenario_id is None:
            raise RunnerError("competition contract is missing its official scenario id")
        try:
            scenario = get_scenario(contract.scenario_id)
        except ValueError as error:
            raise RunnerError(str(error)) from error
        expected = {
            "scenario_sha256": scenario.digest,
            "scenario_variant": scenario.variant,
            "scenario_schedule_seed": scenario.schedule_seed,
            "task_env": scenario.gym_id,
            "task_class": scenario.task_class,
            "task_family": scenario.task_family,
            "instruction": scenario.instruction,
            "seed": scenario.scene_seed,
        }
        actual = {name: getattr(contract, name) for name in expected}
        if actual != expected:
            differences = [
                f"{name}: contract={actual[name]!r}, registry={value!r}"
                for name, value in expected.items()
                if actual[name] != value
            ]
            raise RunnerError(
                "competition contract differs from the official scenario registry (" + "; ".join(differences) + ")"
            )
        if (
            contract.control.action_type != scenario.action_type
            or contract.control.action_dim != scenario.action_dim
            or contract.control.control_dt != scenario.control_dt
            or contract.limits.max_steps != scenario.max_steps
        ):
            raise RunnerError("competition control or episode limits differ from the official scenario registry")
        register_scenario(scenario.scenario_id)
        return
    if contract.is_v4_1:
        if (
            contract.task_family != SPOT_TASK_FAMILY
            or contract.task_env != SPOT_TASK_ENV
            or contract.task_class != SPOT_TASK_CLASS
        ):
            raise RunnerError("V4.1 contract does not match the Runner Spot whitelist")
        from robolab.registrations.spot_warehouse import register_spot_warehouse_env

        register_spot_warehouse_env()
        return
    from robolab.registrations.droid.auto_env_registrations_jointpos import auto_register_droid_envs

    auto_register_droid_envs(task_dirs=["benchmark"], task=[contract.task_class])


__all__ = ["SPOT_TASK_CLASS", "SPOT_TASK_ENV", "register_runner_environment"]
