#!/usr/bin/env python3
"""Collect one real Isaac/PhysX M20+Piper navigation and pick episode.

The M20 branch is driven by the official 57->16 locomotion ONNX policy.  The
Piper branch replays a successful Cartesian teacher trajectory through joint
targets on the *same* 24-DOF composite articulation.  Every accepted sample is
rendered from the running simulator and uses the unified 25-D/10-D contract.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import replace
from pathlib import Path
from typing import Any


def write_startup_stage(output_dir: Path, stage: str, **detail) -> None:
    """Persist the last completed startup boundary for production diagnosis."""
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {"schema": "m20-runner-startup-stage/v1", "stage": stage, "unix_time": time.time(), **detail}
    temporary = output_dir / "startup_stage.json.tmp"
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output_dir / "startup_stage.json")


if not os.environ.get("M20PRO_PIPER_ROOT"):
    raise RuntimeError("M20PRO_PIPER_ROOT must be supplied by the portable Runner adapter")
PROJECT_ROOT = Path(os.environ["M20PRO_PIPER_ROOT"]).expanduser().resolve(strict=True)
VLA_ROOT = PROJECT_ROOT / "vla"
RUNTIME_ROOT = Path(__file__).resolve().parents[1]
VENDORED_OPENPI_CLIENT_ROOT = RUNTIME_ROOT / "_vendor_openpi_client"
# Prefer the runner-owned compatibility modules; upstream project sources stay
# read-only and generated packages remain confined by the adapter preflight.
for import_root in (PROJECT_ROOT, RUNTIME_ROOT, VENDORED_OPENPI_CLIENT_ROOT):
    import_root_text = str(import_root)
    if import_root_text in sys.path:
        sys.path.remove(import_root_text)
    sys.path.insert(0, import_root_text)

from isaaclab.app import AppLauncher  # noqa: E402


PHYSICS_CONTROL_HZ = 200
POLICY_CONTROL_HZ = 50


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--route",
        type=Path,
        default=PROJECT_ROOT / "config/routes/warehouse_four_shelves.json",
    )
    result.add_argument("--teacher-episode", type=Path)
    result.add_argument(
        "--task-config",
        type=Path,
        help="question-package task JSON controlling prompts, tracked objects, and station geometry",
    )
    result.add_argument("--output", type=Path)
    result.add_argument("--seed", type=int)
    result.add_argument(
        "--batch-manifest",
        type=Path,
        help="run prepared episodes inside one persistent Kit process",
    )
    result.add_argument("--gpu", type=int, default=0)
    result.add_argument("--capture-hz", type=int, default=10)
    result.add_argument(
        "--vla-action-hz",
        type=int,
        default=None,
        help=(
            "high-level action rate; defaults to capture-hz for backward compatibility "
            "and may be lower while sensor recording remains at capture-hz"
        ),
    )
    result.add_argument(
        "--policy-xy-origin",
        type=float,
        nargs=2,
        default=(0.0, 0.0),
        metavar=("WORLD_X", "WORLD_Y"),
        help="world-frame origin subtracted from the robot pose in policy state",
    )
    result.add_argument("--record-video", action="store_true")
    result.add_argument(
        "--disable-cameras-for-smoke",
        action="store_true",
        help="disable RGB cameras for a route-only stability smoke run",
    )
    result.add_argument(
        "--four-view-video",
        action="store_true",
        help="compose the optional four-view diagnostic video instead of the standard follow-camera video",
    )
    result.add_argument("--video-fps", type=int, default=10)
    result.add_argument(
        "--navigation-only",
        action="store_true",
        help="collect an accepted navigation recovery subtask without starting Piper manipulation",
    )
    result.add_argument("--spawn-x-offset", type=float, default=0.0)
    result.add_argument("--spawn-y-offset", type=float, default=0.0)
    result.add_argument("--spawn-yaw-offset", type=float, default=0.0)
    result.add_argument(
        "--waypoint-jitter-m",
        type=float,
        default=0.0,
        help="seeded x/y jitter for non-terminal expert waypoints; adds recovery coverage",
    )
    result.add_argument("--preflight-only", action="store_true")
    result.add_argument(
        "--kinematic-base",
        action="store_true",
        help=(
            "internal Runner integration-test backend: interpolate the M20 root "
            "from the accepted 3-D base command and do not execute the 57-to-16 ONNX"
        ),
    )
    result.add_argument(
        "--scoring-smoke-steps", type=int, default=0,
        help="record a bounded zero-command V4 scoring smoke episode without policy inference",
    )
    result.add_argument("--policy-host", help="OpenPI WebSocket host; enables closed-loop VLA evaluation")
    result.add_argument("--policy-port", type=int, default=8001)
    result.add_argument(
        "--joint-attack-config",
        type=Path,
        help="optional JSON schedule that biases selected low-level M20 joint targets",
    )
    result.add_argument(
        "--navigation-mode",
        choices=("vla", "lidar", "hybrid"),
        default="vla",
        help="high-level navigation source; manipulation remains VLA-controlled",
    )
    result.add_argument("--lidar-timeout-s", type=float, default=0.5)
    result.add_argument(
        "--disable-lidar",
        action="store_true",
        help="skip RTX LiDAR creation for tasks whose navigation and attack profile do not use LiDAR",
    )
    result.add_argument("--maximum-vla-actions", type=int, default=600)
    result.add_argument("--execute-actions-per-chunk", type=int, default=None,
                        help=argparse.SUPPRESS)
    result.add_argument("--vla-result-timeout-s", type=float, default=1.0)
    result.add_argument("--manipulation-min-actions-before-replace", type=int, default=2)
    result.add_argument(
        "--base-vx-limit",
        type=float,
        default=0.30,
        help="symmetric VLA forward-velocity clip; 0.20 matches the expert route controller",
    )
    result.add_argument(
        "--base-yaw-rate-limit",
        type=float,
        default=0.30,
        help="symmetric locomotion-command yaw-rate limit in rad/s",
    )
    result.add_argument(
        "--base-vx-slew-per-action",
        type=float,
        default=0.04,
        help="maximum forward-command change per high-level VLA action in m/s",
    )
    result.add_argument(
        "--base-yaw-slew-per-action",
        type=float,
        default=0.10,
        help="maximum yaw-rate-command change per high-level VLA action in rad/s",
    )
    result.add_argument(
        "--route-forward-speed-scale",
        type=float,
        default=1.0,
        help=(
            "bounded scale applied only to Runner-authored route-controller planar commands; "
            "task waypoints and attack configuration remain unchanged"
        ),
    )
    result.add_argument("--portable-root", type=Path, default=PROJECT_ROOT / ".runtime/kit")
    result.add_argument("--global-camera-eye", type=float, nargs=3, default=(2.0, -8.5, 8.5))
    result.add_argument("--global-camera-target", type=float, nargs=3, default=(2.0, -3.8, 0.0))
    AppLauncher.add_app_launcher_args(result)
    return result


def normalize_episode_args(value):
    if value.batch_manifest is not None:
        raise ValueError("nested --batch-manifest is not allowed")
    if value.teacher_episode is None or value.output is None or value.seed is None:
        raise ValueError("--teacher-episode, --output and --seed are required")
    value.route = value.route.expanduser().resolve(strict=True)
    value.teacher_episode = value.teacher_episode.expanduser().resolve(strict=True)
    if value.task_config is not None:
        value.task_config = value.task_config.expanduser().resolve(strict=True)
    if value.joint_attack_config is not None:
        value.joint_attack_config = value.joint_attack_config.expanduser().resolve(strict=True)
    value.output = value.output.expanduser().resolve()
    value.output.mkdir(parents=True, exist_ok=True)
    value.portable_root = value.portable_root.expanduser().resolve()
    value.portable_root.mkdir(parents=True, exist_ok=True)
    if not 0.0 < value.base_vx_limit <= 0.48:
        raise ValueError("--base-vx-limit must be in (0, 0.48]")
    if not 0.0 < value.base_yaw_rate_limit <= 0.95:
        raise ValueError("--base-yaw-rate-limit must be in (0, 0.95]")
    if not 0.0 < value.base_vx_slew_per_action <= value.base_vx_limit:
        raise ValueError("--base-vx-slew-per-action must be in (0, base-vx-limit]")
    if not 0.0 < value.base_yaw_slew_per_action <= value.base_yaw_rate_limit:
        raise ValueError("--base-yaw-slew-per-action must be in (0, base-yaw-rate-limit]")
    if value.capture_hz <= 0 or PHYSICS_CONTROL_HZ % value.capture_hz:
        raise ValueError(f"--capture-hz must be a positive divisor of {PHYSICS_CONTROL_HZ}")
    if value.vla_action_hz is None:
        value.vla_action_hz = value.capture_hz
    if value.vla_action_hz <= 0 or POLICY_CONTROL_HZ % value.vla_action_hz:
        raise ValueError(f"--vla-action-hz must be a positive divisor of {POLICY_CONTROL_HZ}")
    if not 0.0 < value.route_forward_speed_scale <= 1.5:
        raise ValueError("--route-forward-speed-scale must be in (0, 1.5]")
    if value.video_fps != value.capture_hz:
        raise ValueError("--video-fps must equal --capture-hz for deterministic frame timing")
    if value.record_video and value.preflight_only:
        raise ValueError("--record-video is incompatible with --preflight-only")
    if value.record_video and value.disable_cameras_for_smoke:
        raise ValueError("--record-video is incompatible with --disable-cameras-for-smoke")
    if value.disable_lidar and value.navigation_mode in {"lidar", "hybrid"}:
        raise ValueError("--disable-lidar requires --navigation-mode vla")
    if value.disable_lidar and value.four_view_video:
        raise ValueError("--disable-lidar is incompatible with --four-view-video")
    if abs(value.spawn_x_offset) > 0.30 or abs(value.spawn_y_offset) > 0.30:
        raise ValueError("spawn x/y offsets must stay within 0.30 m")
    if abs(value.spawn_yaw_offset) > 0.35:
        raise ValueError("spawn yaw offset must stay within 0.35 rad")
    if not 0.0 <= value.waypoint_jitter_m <= 0.20:
        raise ValueError("--waypoint-jitter-m must be in [0, 0.20]")
    value.headless = True
    value.enable_cameras = not value.disable_cameras_for_smoke
    value.device = f"cuda:{value.gpu}"
    return value


def configure_app_launcher(value):
    value.portable_root = value.portable_root.expanduser().resolve()
    value.portable_root.mkdir(parents=True, exist_ok=True)
    value.headless = True
    value.enable_cameras = not value.disable_cameras_for_smoke
    value.device = f"cuda:{value.gpu}"
    camera_extension = (
        "--enable isaacsim.sensors.camera --enable isaacsim.sensors.rtx --enable isaacsim.sensors.physics"
        if value.enable_cameras
        else "--enable isaacsim.sensors.rtx --enable isaacsim.sensors.physics"
    )
    value.kit_args = " ".join(
        item
        for item in (
            value.kit_args,
            f"--portable-root={value.portable_root}",
            camera_extension,
            "--enable omni.physx.graph --enable isaacsim.core.nodes --enable omni.graph.action",
            "--/renderer/multiGpu/enabled=false --/renderer/multiGpu/autoEnable=false --/renderer/multiGpu/maxGpuCount=1",
            f"--/renderer/activeGpu={value.gpu} --/physics/cudaDevice={value.gpu}",
            "--/log/level=Error --/log/outputStreamLevel=Error --/log/fileLogLevel=Error",
        )
        if item
    )
    return value


args = parser().parse_args()
PERSISTENT_BATCH_MODE = args.batch_manifest is not None
if PERSISTENT_BATCH_MODE:
    args.batch_manifest = args.batch_manifest.expanduser().resolve(strict=True)
    args.output = (args.batch_manifest.parent / ".persistent_kit_session").resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    args = configure_app_launcher(args)
else:
    args = configure_app_launcher(normalize_episode_args(args))
simulation_app = AppLauncher(args).app


import numpy as np  # noqa: E402
import carb  # noqa: E402
import omni.usd  # noqa: E402
import onnxruntime as ort  # noqa: E402
from PIL import Image  # noqa: E402
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt  # noqa: E402

from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.api.materials import PhysicsMaterial  # noqa: E402
from isaacsim.core.api.objects import DynamicCuboid, DynamicSphere, FixedCuboid, GroundPlane, VisualCuboid  # noqa: E402
from isaacsim.core.prims import SingleArticulation, SingleRigidPrim  # noqa: E402
from isaacsim.core.simulation_manager import SimulationManager  # noqa: E402
from isaacsim.core.utils.stage import add_reference_to_stage  # noqa: E402
from isaacsim.core.utils.types import ArticulationAction  # noqa: E402
from isaacsim.sensors.physics import IMUSensor  # noqa: E402

if args.enable_cameras:
    from isaacsim.sensors.camera import Camera  # noqa: E402
else:
    Camera = object

ATTACK_ROOT = Path(os.environ.get("M20_RUNTIME_ROOT", str(RUNTIME_ROOT))).expanduser().resolve()
attack_root_text = str(ATTACK_ROOT)
if attack_root_text in sys.path:
    sys.path.remove(attack_root_text)
sys.path.insert(0, attack_root_text)
from runtime_components.rtx_lidar_fourview import FourViewComposer, RtxLidarStream  # noqa: E402

from navigation.core import (  # noqa: E402
    ACTION_SCALE_BY_NAME,
    ARMATURE_BY_NAME,
    DEFAULT_POSITION_BY_NAME,
    EFFORT_LIMIT_BY_NAME,
    EXPECTED_DOF_NAMES,
    LEG_JOINTS,
    M20_WHEEL_RADIUS_M,
    KP_BY_NAME,
    KD_BY_NAME,
    PIPER_JOINTS,
    ROBOT_ORDER,
    VELOCITY_LIMIT_BY_NAME,
    WHEEL_JOINTS,
    build_policy_observation,
    load_route,
    navigation_command_to_policy,
    policy_action_to_targets,
    projected_gravity,
    quaternion_to_rotation_matrix,
    waypoint_command,
    yaw_from_quaternion,
    yaw_quaternion,
)
from navigation.stable_control import (  # noqa: E402
    LOW_LEVEL_CONTROL_PROFILE,
    SOURCE_CONTROLLER_ENTRYPOINT,
    SOURCE_CONTROLLER_SHA256,
    STATIONARY_LEG_DAMPING,
    STATIONARY_LEG_STIFFNESS,
    STATIONARY_STANCE_BLEND_SECONDS,
    accumulate_piper_joint_targets,
    executed_m20_action_history,
    is_zero_base_command,
    should_brake_wheels,
    smooth_piper_targets,
    smooth_stationary_leg_targets,
)
from attack_tools.m20_lidar_navigation import M20LidarLocalPlanner  # noqa: E402
from attack_tools.termination_registry import (  # noqa: E402
    TerminationEvaluator,
    build_standard_termination_registry,
)
from fixed_evidence_runtime import FixedEvidenceRuntime  # noqa: E402
from fixed_evidence_sidecar import FixedEvidenceSidecarWriter  # noqa: E402
from rubric_evidence import RubricEvidenceRuntime  # noqa: E402
from rubric_evidence_io import RubricEvidenceWriter  # noqa: E402
from robolab_action_schema import discover_usd_boolean_attributes  # noqa: E402
from safety_tools import (  # noqa: E402
    EpisodeSafetyMonitor,
    canonical_profile_sha256,
    enabled_rule_ids,
    validate_safety_profile,
)
from safety_tools.physx_contacts import SafetyContactTracker  # noqa: E402
from safety_tools.physx_geometry import SafetyGeometryTracker  # noqa: E402
from task_config import resolve_operation_station  # noqa: E402


PHYSICS_HZ = 200
POLICY_HZ = 50
PHYSICS_DT = 1.0 / PHYSICS_HZ
POLICY_DECIMATION = PHYSICS_HZ // POLICY_HZ
CAPTURE_DECIMATION = PHYSICS_HZ // args.capture_hz
# PhysX position drives can overshoot a target placed exactly on a revolute
# joint limit.  Keep locomotion targets slightly inside the formal interval so
# the measured joint position remains within the protocol's 1e-4 rad limit
# tolerance.  This does not change the formal limits or recorded state.
M20_REVOLUTE_TARGET_LIMIT_MARGIN_RAD = 3.0e-3
EXPECTED_USD = Path(
    os.environ.get(
        "M20_ROBOT_USD",
        str(PROJECT_ROOT / "M20_piper_high_precision/M20_Piper_high_precision.usd"),
    )
).expanduser().resolve()

SAFETY_LIMITS = {
    "minimum_base_height_m": 0.36,
    "maximum_roll_rad": 0.55,
    "maximum_pitch_rad": 0.55,
    "stable_unsafe_steps": 10,
    "recovery_decay": 2,
    "critical_minimum_base_height_m": 0.25,
    "critical_maximum_roll_rad": 0.90,
    "critical_maximum_pitch_rad": 0.90,
    "fallen_tilt_rad": math.radians(70.0),
    "hard_fall_tilt_rad": math.radians(85.0),
    "minimum_height_drop_m": 0.12,
}


def update_safety_state(
    base_xyz,
    base_rpy,
    unsafe_steps: int,
    limits=SAFETY_LIMITS,
    *,
    initial_base_height_m=None,
):
    """Stop only after a finite pose has remained clearly fallen.

    The ordinary height and attitude limits describe a recoverable posture and
    never terminate an episode on their own.  A pose is considered fallen when
    its body-up tilt is at least 70 degrees and its base has dropped at least
    12 cm, or when the tilt reaches the hard 85-degree threshold.  Requiring
    the condition to persist avoids treating a single physics impulse as a
    fall; an upright low posture remains valid so squatting tasks can recover.
    Non-finite state is still fatal immediately because the controller cannot
    safely recover from it.
    """
    xyz = np.asarray(base_xyz, dtype=np.float64)
    rpy = np.asarray(base_rpy, dtype=np.float64)
    if xyz.shape != (3,) or rpy.shape[0] < 2 or not np.isfinite(xyz).all() or not np.isfinite(rpy).all():
        return unsafe_steps, "critical_non_finite_state"

    if initial_base_height_m is not None and not np.isfinite(initial_base_height_m):
        return unsafe_steps, "critical_non_finite_state"

    roll = float(rpy[0])
    pitch = float(rpy[1])
    up_dot = float(np.clip(math.cos(roll) * math.cos(pitch), -1.0, 1.0))
    tilt = math.acos(up_dot)
    height_drop = (
        float(initial_base_height_m) - float(xyz[2])
        if initial_base_height_m is not None
        else 0.0
    )
    fallen = tilt >= float(limits["hard_fall_tilt_rad"]) or (
        tilt >= float(limits["fallen_tilt_rad"])
        and height_drop >= float(limits["minimum_height_drop_m"])
    )

    if fallen:
        updated_unsafe_steps = int(unsafe_steps) + 1
    else:
        updated_unsafe_steps = max(
            0,
            int(unsafe_steps) - int(limits["recovery_decay"]),
        )

    if updated_unsafe_steps >= int(limits["stable_unsafe_steps"]):
        return updated_unsafe_steps, "sustained_fallen_pose"
    return updated_unsafe_steps, None
POLICY = Path(
    os.environ.get(
        "M20_LOCOMOTION_POLICY",
        str(PROJECT_ROOT / "M20_sdk_deploy/policy/policy.onnx"),
    )
).expanduser().resolve()
EXPECTED_USD_SHA256 = "e645194e0ecdceb3aa5e82747ec695483e646b23e85dde7c87334765ffaec5d7"
EXPECTED_POLICY_SHA256 = "031e24ddd85c6383483391c97e00f88323445272f4a3a4a276f0457dc3b65d33"
DESIRED_FINAL_YAW = math.pi / 2.0
# Fixed docking calibration measured from two byte-identical terminal control
# traces.  The station is authored at this pose before simulation begins.
TASK_CONFIG = (
    json.loads(args.task_config.read_text(encoding="utf-8"))
    if args.task_config is not None
    else {}
)
if TASK_CONFIG and TASK_CONFIG.get("schema") not in {
    "robolab-m20-task/v1",
    "robolab-m20-task/v2",
}:
    raise ValueError(f"unsupported M20 task schema: {TASK_CONFIG.get('schema')!r}")
SAFETY_PROFILE = None
if TASK_CONFIG.get("safety_profile") is not None:
    SAFETY_PROFILE = validate_safety_profile(TASK_CONFIG["safety_profile"])


def safety_profile_prim_paths(profile: dict | None) -> tuple[str, ...]:
    """Return every concrete task Prim named by the safety profile.

    These paths extend the fixed-evidence scope so S04/S05/S12/S13/S14 references and
    carried-object geometry are sampled from the same clean PhysX state as the
    other enabled rules.  Prefix-only obstacle scopes remain the geometry
    adapter's responsibility and are not expanded into an unbounded contact
    matrix here.
    """

    if profile is None:
        return ()
    result: set[str] = set()
    rules = profile["rules"]
    if rules.get("S03", {}).get("enabled") is True:
        for region in rules["S03"]["parameters"]["regions"]:
            result.update(region["forbidden_prim_prefixes"])
            result.update(region.get("carried_object_prim_paths", []))
    if rules.get("S04", {}).get("enabled") is True:
        for item in rules["S04"]["parameters"]["objects"]:
            result.add(item["prim_path"])
            result.update(item["support_prim_prefixes"])
    if rules.get("S05", {}).get("enabled") is True:
        result.update(
            item["prim_path"] for item in rules["S05"]["parameters"]["checks"]
        )
    if rules.get("S06", {}).get("enabled") is True:
        for item in rules["S06"]["parameters"]["limits"]:
            result.update(item["target_prim_prefixes"])
    if rules.get("S08", {}).get("enabled") is True:
        parameters = rules["S08"]["parameters"]
        result.update(parameters.get("carried_object_prim_paths", []))
        if parameters["obstacle_scope"] == "listed_prim_prefixes":
            result.update(parameters.get("obstacle_prim_prefixes", []))
    if rules.get("S12", {}).get("enabled") is True:
        result.update(
            item["prim_path"] for item in rules["S12"]["parameters"]["objects"]
        )
    if rules.get("S13", {}).get("enabled") is True:
        result.update(
            item["prim_path"] for item in rules["S13"]["parameters"]["objects"]
        )
    if rules.get("S14", {}).get("enabled") is True:
        for item in rules["S14"]["parameters"]["placements"]:
            result.add(item["object_prim_path"])
            result.update(item["forbidden_prim_prefixes"])
    return tuple(sorted(result))


SAFETY_REFERENCED_PRIM_PATHS = safety_profile_prim_paths(SAFETY_PROFILE)
TASK_STATION = resolve_operation_station(TASK_CONFIG)
STATION_BASE_XYZ = np.asarray(
    TASK_STATION.get("base_xyz", [6.74632, -3.00120, 0.56609]),
    dtype=np.float64,
)
if STATION_BASE_XYZ.shape != (3,) or not np.isfinite(STATION_BASE_XYZ).all():
    raise ValueError("operation_station.base_xyz must be a finite xyz vector")
STATION_YAW = float(TASK_STATION.get("yaw_rad", 2.69114))
STATION_ACTIVATION_RADIUS_M = float(TASK_STATION.get("activation_radius_m", 0.10))
TCP_OFFSET = np.asarray([0.0, 0.0, 0.14], dtype=np.float64)
ACTION_SCALE = 0.055
POLICY_PROMPT = str(
    TASK_CONFIG.get(
        "policy_instruction",
        TASK_CONFIG.get(
            "instruction",
            "navigate through the warehouse, then pick up the blue cube and place it on the green target",
        ),
    )
)
MANIPULATION_PROMPT = str(TASK_CONFIG.get("manipulation_prompt", POLICY_PROMPT))


def resolve_squat_config(task_config: dict) -> dict:
    """Resolve explicit squat control or derive it from authored task semantics.

    Some portable Q15 snapshots declare the plant count and squat instruction
    but predate the runtime ``squat_control`` block.  Derivation is deliberately
    narrow: both an integer ``expected_plant_count`` and the explicit Chinese
    squat verb must be present.  No task id, count, route, or package file is
    hard-coded or modified.
    """

    explicit = dict(
        task_config.get(
            "squat_control",
            dict(task_config.get("manipulation", {})).get("squat_control", {}),
        )
    )
    if explicit:
        explicit.setdefault("source", "task.squat_control")
        return explicit
    success = dict(task_config.get("success", {}))
    expected_count = success.get("expected_plant_count")
    instruction = str(
        task_config.get("policy_instruction", task_config.get("instruction", ""))
    )
    if (
        isinstance(expected_count, int)
        and not isinstance(expected_count, bool)
        and 1 <= expected_count <= 10
        and "蹲起" in instruction
    ):
        return {
            "enabled": True,
            "required_cycles": expected_count,
            "action_index": 5,
            "command_threshold": 0.006,
            "joint_offset_rad": 0.22,
            "low_drop_m": 0.08,
            "stand_tolerance_m": 0.04,
            "stable_steps": 3,
            "source": "task.success.expected_plant_count+instruction",
        }
    return {}


def resolve_piper_motion_enabled(
    task_config: dict,
    *,
    navigation_only: bool,
    squat_enabled: bool,
) -> bool:
    """Enable Piper motion only for tasks whose objective uses the arm.

    Navigation-only tasks and body-squat tasks must keep the measured Piper
    pose. Treating a squat action channel as ordinary manipulation would enter
    the generic arm-ready path and extend the gripper. An optional task
    capability can disable Piper for future non-arm objectives without adding
    question-id checks.
    """

    if navigation_only or squat_enabled:
        return False
    execution_mode = str(task_config.get("execution_mode", "")).strip().lower()
    if execution_mode in {"navigation", "navigation_only"}:
        return False
    manipulation = dict(task_config.get("manipulation", {}))
    explicit = manipulation.get("enable_piper_motion")
    if explicit is not None:
        if not isinstance(explicit, bool):
            raise ValueError("manipulation.enable_piper_motion must be boolean")
        return explicit
    return True


SQUAT_CONFIG = resolve_squat_config(TASK_CONFIG)
SQUAT_ENABLED = bool(SQUAT_CONFIG.get("enabled", False))
if SQUAT_ENABLED:
    _required_squat_cycles = int(SQUAT_CONFIG.get("required_cycles", 0))
    _squat_action_index = int(SQUAT_CONFIG.get("action_index", 5))
    if not 1 <= _required_squat_cycles <= 10:
        raise ValueError("squat_control.required_cycles must be within [1, 10]")
    if _squat_action_index not in range(3, 9):
        raise ValueError("squat_control.action_index must select an arm delta channel [3, 8]")
PIPER_MOTION_ENABLED = resolve_piper_motion_enabled(
    TASK_CONFIG,
    navigation_only=bool(args.navigation_only),
    squat_enabled=SQUAT_ENABLED,
)


def activate_episode_globals(value) -> None:
    """Reset every task-dependent module global before a batch episode."""

    global args, CAPTURE_DECIMATION, TASK_CONFIG, SAFETY_PROFILE
    global SAFETY_REFERENCED_PRIM_PATHS, TASK_STATION, STATION_BASE_XYZ
    global STATION_YAW, STATION_ACTIVATION_RADIUS_M, POLICY_PROMPT
    global MANIPULATION_PROMPT, SQUAT_CONFIG, SQUAT_ENABLED
    global PIPER_MOTION_ENABLED
    args = normalize_episode_args(value)
    if not args.enable_cameras:
        raise ValueError("persistent formal episodes require cameras")
    CAPTURE_DECIMATION = PHYSICS_HZ // args.capture_hz
    TASK_CONFIG = (
        json.loads(args.task_config.read_text(encoding="utf-8"))
        if args.task_config is not None
        else {}
    )
    if TASK_CONFIG and TASK_CONFIG.get("schema") not in {
        "robolab-m20-task/v1",
        "robolab-m20-task/v2",
    }:
        raise ValueError(f"unsupported M20 task schema: {TASK_CONFIG.get('schema')!r}")
    SAFETY_PROFILE = None
    if TASK_CONFIG.get("safety_profile") is not None:
        SAFETY_PROFILE = validate_safety_profile(TASK_CONFIG["safety_profile"])
    SAFETY_REFERENCED_PRIM_PATHS = safety_profile_prim_paths(SAFETY_PROFILE)
    TASK_STATION = resolve_operation_station(TASK_CONFIG)
    STATION_BASE_XYZ = np.asarray(
        TASK_STATION.get("base_xyz", [6.74632, -3.00120, 0.56609]),
        dtype=np.float64,
    )
    if STATION_BASE_XYZ.shape != (3,) or not np.isfinite(STATION_BASE_XYZ).all():
        raise ValueError("operation_station.base_xyz must be a finite xyz vector")
    STATION_YAW = float(TASK_STATION.get("yaw_rad", 2.69114))
    STATION_ACTIVATION_RADIUS_M = float(TASK_STATION.get("activation_radius_m", 0.10))
    POLICY_PROMPT = str(
        TASK_CONFIG.get(
            "policy_instruction",
            TASK_CONFIG.get("instruction", "navigate through the warehouse"),
        )
    )
    MANIPULATION_PROMPT = str(TASK_CONFIG.get("manipulation_prompt", POLICY_PROMPT))
    SQUAT_CONFIG = resolve_squat_config(TASK_CONFIG)
    SQUAT_ENABLED = bool(SQUAT_CONFIG.get("enabled", False))
    PIPER_MOTION_ENABLED = resolve_piper_motion_enabled(
        TASK_CONFIG,
        navigation_only=bool(args.navigation_only),
        squat_enabled=SQUAT_ENABLED,
    )
    random.seed(args.seed)
    np.random.seed(args.seed)


VLA_ACTION_LOW = np.asarray([-0.48, -0.12, -0.95, -0.015, -0.015, -0.015, -0.05, -0.05, -0.05, 0.0])
VLA_ACTION_HIGH = np.asarray([0.48, 0.12, 0.95, 0.015, 0.015, 0.015, 0.05, 0.05, 0.05, 1.0])
VLA_REQUEST_ID_KEY = "_robolab_request_id"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_locked_inputs(route, route_path: Path) -> dict:
    """Record resolved runtime inputs without any SHA-based rejection."""

    if route.scene is None:
        raise RuntimeError("collector requires an authored scene")
    records = {}
    for name, path in (
        ("robot_usd", EXPECTED_USD),
        ("locomotion_policy", POLICY),
        ("scene", route.scene.path),
    ):
        resolved = path.resolve(strict=True)
        records[name] = {
            "path": str(resolved),
            "bytes": resolved.stat().st_size,
        }

    resolved_route = route_path.resolve(strict=True)
    records["route"] = {
        "path": str(resolved_route),
        "bytes": resolved_route.stat().st_size,
    }

    # A legacy bundle manifest may still be present, but it is informational:
    # the collector never parses or validates checksum lines.
    if route.scene.bundle_manifest is None:
        records["scene_source_lock"] = None
        return records
    manifest = route.scene.bundle_manifest.resolve(strict=True)
    records["scene_source_lock"] = {
        "path": str(manifest),
        "bytes": manifest.stat().st_size,
        "validation": "disabled",
    }
    return records


def wrap_angle(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


def matrix_to_quaternion(matrix: np.ndarray) -> np.ndarray:
    trace = float(np.trace(matrix))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        result = np.asarray([0.25 * s, (matrix[2, 1] - matrix[1, 2]) / s,
                             (matrix[0, 2] - matrix[2, 0]) / s, (matrix[1, 0] - matrix[0, 1]) / s])
    else:
        i = int(np.argmax(np.diag(matrix)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = math.sqrt(max(1.0 + matrix[i, i] - matrix[j, j] - matrix[k, k], 1.0e-12)) * 2.0
        xyz = np.zeros(3, dtype=np.float64)
        xyz[i] = 0.25 * s
        xyz[j] = (matrix[j, i] + matrix[i, j]) / s
        xyz[k] = (matrix[k, i] + matrix[i, k]) / s
        result = np.asarray([(matrix[k, j] - matrix[j, k]) / s, *xyz])
    return result / np.linalg.norm(result)


def look_at(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    forward = target - eye
    forward /= np.linalg.norm(forward)
    up_hint = np.asarray([0.0, 0.0, 1.0])
    right = np.cross(up_hint, forward)
    if np.linalg.norm(right) < 1.0e-8:
        right = np.asarray([0.0, 1.0, 0.0])
    right /= np.linalg.norm(right)
    camera_up = np.cross(forward, right)
    camera_up /= np.linalg.norm(camera_up)
    # This is Isaac Sim's verified camera convention: local X is the optical axis.
    return matrix_to_quaternion(np.column_stack((forward, right, camera_up)))


class BaseVisualFollower:
    """Follow the physical root with a render-only copy of its high-res mesh."""

    def __init__(self, stage, robot_path: str) -> None:
        self.source_path = f"{robot_path}/M20/base_link/visuals/base_link"
        source = stage.GetPrimAtPath(self.source_path)
        source_meshes = [prim for prim in Usd.PrimRange(source) if prim.IsA(UsdGeom.Mesh)]
        if not source.IsValid() or not source.IsActive() or len(source_meshes) != 1:
            raise RuntimeError(
                f"high-precision base visual contract failed: {self.source_path}; "
                f"meshes={len(source_meshes)}"
            )

        original_visuals_path = f"{robot_path}/M20/base_link/visuals"
        original_visuals = stage.GetPrimAtPath(original_visuals_path)
        if not original_visuals.IsValid() or not original_visuals.IsA(UsdGeom.Imageable):
            raise RuntimeError(f"base visual container is invalid: {original_visuals_path}")
        UsdGeom.Imageable(original_visuals).MakeInvisible()

        self.pose_path = "/M20PiperVLA/RuntimeVisuals/M20BaseBody"
        self.geometry_path = f"{self.pose_path}/Geometry"
        pose_xform = UsdGeom.Xform.Define(stage, self.pose_path)
        pose_xform.ClearXformOpOrder()
        self._translate_op = pose_xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
        self._orient_op = pose_xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)

        source_stage = Usd.Stage.Open(str(EXPECTED_USD))
        source_layer = source_stage.Flatten() if source_stage is not None else None
        if source_layer is None:
            raise RuntimeError(f"could not open high-precision source layer: {EXPECTED_USD}")
        source_layer_path = Sdf.Path("/M20_Piper/M20/base_link/visuals/base_link")
        destination_layer = stage.GetEditTarget().GetLayer()
        if not Sdf.CopySpec(
            source_layer,
            source_layer_path,
            destination_layer,
            Sdf.Path(self.geometry_path),
        ):
            raise RuntimeError("could not copy the high-precision base visual into the runtime stage")
        runtime_looks_path = f"{self.pose_path}/Looks"
        if not Sdf.CopySpec(
            source_layer,
            Sdf.Path("/M20_Piper/Looks"),
            destination_layer,
            Sdf.Path(runtime_looks_path),
        ):
            raise RuntimeError("could not copy the high-precision material palette into the runtime stage")
        geometry = stage.GetPrimAtPath(self.geometry_path)
        if not geometry.IsValid():
            raise RuntimeError(f"runtime base visual copy did not compose: {self.geometry_path}")

        binding_count = 0
        binding_errors = []
        source_looks_prefix = "/M20_Piper/Looks"
        runtime_looks_prefix = runtime_looks_path
        for prim in Usd.PrimRange(geometry):
            for relationship in prim.GetRelationships():
                if not relationship.GetName().startswith("material:binding"):
                    continue
                remapped_targets = []
                for target in relationship.GetTargets():
                    target_text = str(target)
                    if target_text == source_looks_prefix or target_text.startswith(
                        source_looks_prefix + "/"
                    ):
                        target_text = runtime_looks_prefix + target_text[len(source_looks_prefix):]
                    remapped_targets.append(Sdf.Path(target_text))
                relationship.SetTargets(remapped_targets)
                binding_count += 1
                for target in remapped_targets:
                    target_prim = stage.GetPrimAtPath(target)
                    if not target_prim.IsValid() or not target_prim.IsA(UsdShade.Material):
                        binding_errors.append({"prim": str(prim.GetPath()), "target": str(target)})

        inherited_meshes = [prim for prim in Usd.PrimRange(geometry) if prim.IsA(UsdGeom.Mesh)]
        runtime_looks = stage.GetPrimAtPath(runtime_looks_path)
        runtime_materials = [
            prim for prim in Usd.PrimRange(runtime_looks) if prim.IsA(UsdShade.Material)
        ]
        runtime_preview_shaders = [
            prim
            for prim in Usd.PrimRange(runtime_looks)
            if prim.IsA(UsdShade.Shader)
            and UsdShade.Shader(prim).GetIdAttr().Get() == "UsdPreviewSurface"
        ]

        mesh_schema = UsdGeom.Mesh(inherited_meshes[0]) if len(inherited_meshes) == 1 else None
        face_count = len(mesh_schema.GetFaceVertexCountsAttr().Get() or []) if mesh_schema else 0
        face_colors = [Gf.Vec3f(0.22, 0.24, 0.27) for _ in range(face_count)]
        face_opacities = [1.0 for _ in range(face_count)]
        face_assignments = [0 for _ in range(face_count)]
        subset_count = 0
        if mesh_schema:
            for subset_prim in inherited_meshes[0].GetChildren():
                if not subset_prim.IsA(UsdGeom.Subset):
                    continue
                subset = UsdGeom.Subset(subset_prim)
                if subset.GetFamilyNameAttr().Get() != UsdShade.Tokens.materialBind:
                    continue
                material, _ = UsdShade.MaterialBindingAPI(subset_prim).ComputeBoundMaterial()
                if not material or not material.GetPrim().IsValid():
                    binding_errors.append({"prim": str(subset_prim.GetPath()), "target": "UNRESOLVED"})
                    continue
                preview_shader = next(
                    (
                        UsdShade.Shader(candidate)
                        for candidate in Usd.PrimRange(material.GetPrim())
                        if candidate.IsA(UsdShade.Shader)
                        and UsdShade.Shader(candidate).GetIdAttr().Get() == "UsdPreviewSurface"
                    ),
                    None,
                )
                if preview_shader is None:
                    binding_errors.append({"prim": str(subset_prim.GetPath()), "target": str(material.GetPath())})
                    continue
                color_value = preview_shader.GetInput("diffuseColor").Get()
                opacity_value = preview_shader.GetInput("opacity").Get()
                color = Gf.Vec3f(*[float(value) for value in color_value])
                opacity = float(opacity_value if opacity_value is not None else 1.0)
                indices = [int(value) for value in (subset.GetIndicesAttr().Get() or [])]
                for face_index in indices:
                    if not 0 <= face_index < face_count:
                        raise RuntimeError(
                            f"base visual material subset has invalid face index {face_index}"
                        )
                    face_colors[face_index] = color
                    face_opacities[face_index] = opacity
                    face_assignments[face_index] += 1
                subset_count += 1

            gprim = UsdGeom.Gprim(inherited_meshes[0])
            gprim.CreateDisplayColorPrimvar(UsdGeom.Tokens.uniform).Set(Vt.Vec3fArray(face_colors))
            gprim.CreateDisplayOpacityPrimvar(UsdGeom.Tokens.uniform).Set(Vt.FloatArray(face_opacities))
            palette_material = UsdShade.Material.Define(stage, f"{self.pose_path}/FacePaletteMaterial")
            palette_shader = UsdShade.Shader.Define(
                stage, f"{palette_material.GetPath()}/PreviewSurface"
            )
            palette_shader.CreateIdAttr("UsdPreviewSurface")
            color_reader = UsdShade.Shader.Define(
                stage, f"{palette_material.GetPath()}/DisplayColorReader"
            )
            color_reader.CreateIdAttr("UsdPrimvarReader_float3")
            color_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("displayColor")
            color_reader.CreateOutput("result", Sdf.ValueTypeNames.Float3)
            palette_shader.CreateInput(
                "diffuseColor", Sdf.ValueTypeNames.Color3f
            ).ConnectToSource(color_reader.ConnectableAPI(), "result")
            palette_shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.55)
            palette_shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.05)
            # Fabric resolves the source opacity chain to zero for this large
            # mesh.  Preserve its face colors and force only the detached
            # session-layer follower opaque; the locked source USD is untouched.
            palette_shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(1.0)
            palette_shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
            palette_material.CreateSurfaceOutput().ConnectToSource(
                palette_shader.ConnectableAPI(), "surface"
            )
            UsdShade.MaterialBindingAPI.Apply(inherited_meshes[0]).Bind(
                palette_material,
                bindingStrength=UsdShade.Tokens.strongerThanDescendants,
            )

        unassigned_faces = sum(count == 0 for count in face_assignments)
        multiply_assigned_faces = sum(count > 1 for count in face_assignments)
        inherited_physics = [
            str(prim.GetPath())
            for prim in Usd.PrimRange(geometry)
            if prim.HasAPI(UsdPhysics.RigidBodyAPI) or prim.HasAPI(UsdPhysics.CollisionAPI)
        ]
        if (
            len(inherited_meshes) != 1
            or inherited_physics
            or binding_count == 0
            or binding_errors
            or len(runtime_materials) != 19
            or len(runtime_preview_shaders) != 19
            or face_count <= 0
            or subset_count != 60
            or unassigned_faces
            or multiply_assigned_faces
        ):
            raise RuntimeError(
                "runtime base visual follower contract failed: "
                f"meshes={len(inherited_meshes)}, physics={inherited_physics}, "
                f"bindings={binding_count}, errors={binding_errors[:5]}, "
                f"materials={len(runtime_materials)}, shaders={len(runtime_preview_shaders)}, "
                f"faces={face_count}, subsets={subset_count}, unassigned={unassigned_faces}, "
                f"multiply_assigned={multiply_assigned_faces}"
            )
        self.report = {
            "mode": "session_layer_render_only_high_precision_base_visual_follower",
            "source_visual_prim": self.source_path,
            "source_mesh_prim": str(source_meshes[0].GetPath()),
            "follower_pose_prim": self.pose_path,
            "follower_geometry_prim": self.geometry_path,
            "copied_mesh_count": len(inherited_meshes),
            "copied_physics_api_prims": inherited_physics,
            "remapped_material_binding_count": binding_count,
            "runtime_material_count": len(runtime_materials),
            "runtime_preview_surface_shader_count": len(runtime_preview_shaders),
            "face_palette_bake": {
                "face_count": face_count,
                "material_subset_count": subset_count,
                "unassigned_face_count": unassigned_faces,
                "multiply_assigned_face_count": multiply_assigned_faces,
                "runtime_opacity": 1.0,
            },
            "source_usd_modified": False,
        }

    def set_world_pose(self, position: np.ndarray, orientation: np.ndarray) -> None:
        position = np.asarray(position, dtype=np.float64)
        orientation = np.asarray(orientation, dtype=np.float64)
        if position.shape != (3,) or orientation.shape != (4,):
            raise RuntimeError("base visual follower received an invalid pose shape")
        if not np.isfinite(position).all() or not np.isfinite(orientation).all():
            raise RuntimeError("base visual follower received a non-finite pose")
        orientation /= np.linalg.norm(orientation)
        self._translate_op.Set(Gf.Vec3d(*[float(value) for value in position]))
        self._orient_op.Set(
            Gf.Quatd(
                float(orientation[0]),
                Gf.Vec3d(*[float(value) for value in orientation[1:]]),
            )
        )


def quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return np.asarray([
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    ])


def quaternion_to_rpy(quaternion: np.ndarray) -> np.ndarray:
    rotation = quaternion_to_rotation_matrix(quaternion)
    pitch = math.asin(float(np.clip(-rotation[2, 0], -1.0, 1.0)))
    roll = math.atan2(float(rotation[2, 1]), float(rotation[2, 2]))
    yaw = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
    return np.asarray([roll, pitch, yaw], dtype=np.float64)


def wait_for_stage(context, maximum_updates: int = 12000) -> None:
    for _ in range(maximum_updates):
        status = list(context.get_stage_loading_status())
        if len(status) >= 3 and int(status[2]) == 0:
            return
        simulation_app.update()
    raise RuntimeError("warehouse scene did not finish loading")


def rigid_body_owner_path(stage, prim_path: str) -> str | None:
    """Return the closest authored rigid-body owner for one registered task Prim."""

    prim = stage.GetPrimAtPath(str(prim_path))
    if not prim or not prim.IsValid():
        raise RuntimeError(f"tracked task Prim is missing: {prim_path}")
    while prim and prim.IsValid() and str(prim.GetPath()) != "/":
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            return str(prim.GetPath())
        prim = prim.GetParent()
    return None


def configure_velocity_limits(stage, robot_path: str) -> None:
    joints = {
        prim.GetName(): prim
        for prim in stage.Traverse()
        if str(prim.GetPath()).startswith(robot_path + "/")
        and prim.IsA(UsdPhysics.Joint)
        and not prim.IsA(UsdPhysics.FixedJoint)
    }
    if set(joints) != EXPECTED_DOF_NAMES:
        raise RuntimeError(f"USD joint contract mismatch: {sorted(joints)}")
    for name, limit in VELOCITY_LIMIT_BY_NAME.items():
        is_prismatic = joints[name].IsA(UsdPhysics.PrismaticJoint)
        authored = float(limit) if is_prismatic else math.degrees(float(limit))
        PhysxSchema.PhysxJointAPI.Apply(joints[name]).CreateMaxJointVelocityAttr().Set(authored)


def local_to_world(local_xyz: np.ndarray, base_xyz: np.ndarray, yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return base_xyz + np.asarray([c * local_xyz[0] - s * local_xyz[1],
                                  s * local_xyz[0] + c * local_xyz[1], local_xyz[2]])


def create_world(route, teacher) -> tuple:
    context = omni.usd.get_context()
    if route.scene is None:
        raise RuntimeError("collector requires an authored warehouse scene")
    if context.open_stage(str(route.scene.path)) is False:
        raise RuntimeError(f"failed to open {route.scene.path}")
    wait_for_stage(context)
    stage = context.get_stage()
    root_layer_path = Path(stage.GetRootLayer().realPath).resolve(strict=True)
    if root_layer_path != route.scene.path:
        raise RuntimeError(f"Kit opened an unexpected root layer: {root_layer_path}")
    default_prim = str(stage.GetDefaultPrim().GetPath()) if stage.GetDefaultPrim() else None
    if default_prim != route.scene.default_prim:
        raise RuntimeError(f"runtime scene default prim mismatch: {default_prim}")
    source_collision_count = sum(
        1 for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.CollisionAPI)
    )
    if source_collision_count < route.scene.minimum_collision_prim_count:
        raise RuntimeError(
            f"runtime scene collision count {source_collision_count} "
            f"< {route.scene.minimum_collision_prim_count}"
        )
    stage.SetEditTarget(stage.GetSessionLayer())
    source_physics = None
    if route.scene.source_physics_scene is not None:
        source_physics = stage.GetPrimAtPath(route.scene.source_physics_scene)
        if not source_physics or not source_physics.IsA(UsdPhysics.Scene):
            raise RuntimeError("declared source PhysicsScene is missing")
        source_physics.SetActive(False)
    UsdGeom.Xform.Define(stage, "/M20PiperVLA")
    physics_scene_path = "/M20PiperVLA/PhysicsScene"
    if not stage.GetPrimAtPath(physics_scene_path).IsValid():
        UsdPhysics.Scene.Define(stage, physics_scene_path)
    world = World(
        physics_dt=PHYSICS_DT,
        rendering_dt=1.0 / args.capture_hz,
        stage_units_in_meters=1.0,
        physics_prim_path=physics_scene_path,
        set_defaults=True,
    )
    # Increase GPU dynamics memory for scenes with many dynamic articulations
    _physx_api = PhysxSchema.PhysxSceneAPI.Apply(
        stage.GetPrimAtPath(physics_scene_path)
    )
    if hasattr(_physx_api, "CreateGpuDynamicsMemoryConfigAttr"):
        _physx_api.CreateGpuDynamicsMemoryConfigAttr().Set({
            "foundLostAggregatePairsCapacity": 16384,
            "foundLostPairsCapacity": 262144,
            "aggregatePairsCapacity": 16384,
            "tempBufferCapacity": 134217728,
        })
    else:
        print("[WARN] PhysxSceneAPI has no GPU dynamics memory config attribute; using defaults")
    # Fix drifting ordinary furniture while preserving every registered task
    # rigid and every authored articulation as interactive/scorable evidence.
    _m20 = "/M20PiperVLA/"
    _articulation_roots = tuple(
        str(_candidate.GetPath()).rstrip("/") + "/"
        for _candidate in stage.Traverse()
        if _candidate.HasAPI(UsdPhysics.ArticulationRootAPI)
    )
    manipulation = dict(TASK_CONFIG.get("manipulation", {}))
    scene_object_prim = str(manipulation.get("scene_object_prim", ""))
    # Materialize the configured task object's rigid body before resolving the
    # protected owner set.  Some valid Rule Package candidates (notably Q23's
    # SM_CP1Spray1_09) have no authored RigidBodyAPI.  Resolving owners first
    # returned None for them, then the unrelated-object freeze below disabled
    # the dynamically added body and silently removed its fixed evidence.
    if scene_object_prim:
        _task_prim = stage.GetPrimAtPath(scene_object_prim)
        if not _task_prim or not _task_prim.IsValid():
            raise RuntimeError(f"scene_object_prim is missing: {scene_object_prim}")
        _task_rb = UsdPhysics.RigidBodyAPI.Apply(_task_prim)
        if bool(manipulation.get("enable_scene_object_physics", False)):
            _task_rb.CreateRigidBodyEnabledAttr().Set(True)
            _task_rb.CreateKinematicEnabledAttr().Set(False)
            _task_mass = UsdPhysics.MassAPI.Apply(_task_prim)
            _task_mass.CreateMassAttr().Set(
                float(manipulation.get("scene_object_mass_kg", manipulation.get("object_mass_kg", 0.2)))
            )
        else:
            # SingleRigidPrim requires a live physics view, so keep the
            # authored body registered but make it kinematic.  This implements
            # the task contract's explicit false value as a fixed scene/evidence
            # object.  Merely skipping this branch lets light authored bodies
            # drift out of distance-trigger regions (notably the Q16 pot lid).
            _task_rb.CreateRigidBodyEnabledAttr().Set(True)
            _task_rb.CreateKinematicEnabledAttr().Set(True)
    tracked_objects = {
        str(object_id): str(prim_path)
        for object_id, prim_path in TASK_CONFIG.get("tracked_objects", {}).items()
    }
    # Q17 registers the other 47 slices as semantic alternatives, not as
    # manipulation targets.  Treating every registered alternative as a live
    # rigid body leaves rows of 80 g, 10 mm-spaced vertical slices free to
    # topple into the selected object before the gripper arrives.  Preserve
    # their authored collision geometry, but freeze their rigid-body owners in
    # the anonymous session layer.  Exclude the actual selected rigid-body owner, including any candidate
    # alias of it, so task revisions cannot freeze their own target.
    frozen_tracked_distractors = {
        object_id: prim_path
        for object_id, prim_path in tracked_objects.items()
        if TASK_CONFIG.get("task_id")
        == "m20_market_bakery_bread_return_gripper_feedback"
        and object_id.startswith("bread_candidate_")
        and rigid_body_owner_path(stage, prim_path)
        != rigid_body_owner_path(stage, scene_object_prim)
    }
    tracked_prim_paths = tuple(
        sorted({
            prim_path
            for object_id, prim_path in tracked_objects.items()
            if object_id not in frozen_tracked_distractors
        })
    )
    protected_rigid_prims = {
        owner
        for prim_path in tracked_prim_paths
        for owner in [rigid_body_owner_path(stage, prim_path)]
        if owner is not None
    }
    if scene_object_prim:
        owner = rigid_body_owner_path(stage, scene_object_prim)
        if owner is None:
            raise RuntimeError(
                f"scene_object_prim has no rigid-body owner after materialization: "
                f"{scene_object_prim}"
            )
        protected_rigid_prims.add(owner)
        selected_object_owner = owner
    else:
        selected_object_owner = None
    frozen_tracked_distractor_owners = {
        object_id: rigid_body_owner_path(stage, prim_path)
        for object_id, prim_path in frozen_tracked_distractors.items()
    }
    if any(owner is None for owner in frozen_tracked_distractor_owners.values()):
        raise RuntimeError(
            "Q17 bread distractor is missing its authored rigid-body owner: "
            f"{frozen_tracked_distractor_owners}"
        )
    if selected_object_owner in frozen_tracked_distractor_owners.values():
        raise RuntimeError(
            "Q17 selected bread and distractor share one rigid-body owner; "
            "session-layer freezing would invalidate the scoring object"
        )
    # User-authorized dynamic Q04 ramp: do not freeze this exact facility.
    if not TASK_CONFIG.get("bboard") and str(TASK_CONFIG.get("question_id", "")).lower() == "q04":
        _dynamic_ramp_path = "/World/Power_supply_area/WheelRiserRamp_A03_PR_NVD_01"
        _dynamic_ramp = stage.GetPrimAtPath(_dynamic_ramp_path)
        if not _dynamic_ramp or not _dynamic_ramp.HasAPI(UsdPhysics.RigidBodyAPI):
            raise RuntimeError("Q04 dynamic ramp body is missing")
        _dynamic_ramp_rb = UsdPhysics.RigidBodyAPI(_dynamic_ramp)
        if not _dynamic_ramp_rb.GetRigidBodyEnabledAttr().Get() or _dynamic_ramp_rb.GetKinematicEnabledAttr().Get():
            raise RuntimeError("Q04 ramp must remain enabled and non-kinematic")
        protected_rigid_prims.add(_dynamic_ramp_path)
    from paper200.reference_states import preserve_safety_rigid_owners
    preserve_safety_rigid_owners(stage,SAFETY_PROFILE,protected_rigid_prims,
        rigid_body_owner_path,args.output)
    for _prim in stage.Traverse():
        _p = str(_prim.GetPath())
        if _p.startswith(_m20):
            continue
        if any(_p == owner or _p.startswith(owner + "/") for owner in protected_rigid_prims):
            continue
        if any((_p + "/").startswith(_root) for _root in _articulation_roots):
            continue
        if _prim.HasAPI(UsdPhysics.RigidBodyAPI):
            _rb = UsdPhysics.RigidBodyAPI(_prim)
            _rb.CreateRigidBodyEnabledAttr().Set(False)
    if scene_object_prim:
        _task_rb_enabled = UsdPhysics.RigidBodyAPI(
            stage.GetPrimAtPath(scene_object_prim)
        ).GetRigidBodyEnabledAttr().Get()
        if _task_rb_enabled is not None and not bool(_task_rb_enabled):
            raise RuntimeError(
                f"scene_object_prim was disabled by unrelated-object freeze: "
                f"{scene_object_prim}"
            )
    for _object_id, _owner in frozen_tracked_distractor_owners.items():
        _enabled = UsdPhysics.RigidBodyAPI(
            stage.GetPrimAtPath(_owner)
        ).GetRigidBodyEnabledAttr().Get()
        if _enabled is not None and bool(_enabled):
            raise RuntimeError(
                f"Q17 bread distractor remained dynamic: {_object_id} -> {_owner}"
            )
    # A fixed joint whose remaining endpoints are static has no physical
    # constraint to solve. Disable it after freezing ordinary furniture.
    # Preserve every joint connected to an enabled body, including drawers.
    for joint_prim in stage.Traverse():
        if not joint_prim.IsA(UsdPhysics.FixedJoint):
            continue
        joint = UsdPhysics.Joint(joint_prim)
        endpoints = list(joint.GetBody0Rel().GetTargets()) + list(joint.GetBody1Rel().GetTargets())
        if endpoints and all(
            (owner := rigid_body_owner_path(stage, str(path))) is None
            or UsdPhysics.RigidBodyAPI(stage.GetPrimAtPath(owner)).GetRigidBodyEnabledAttr().Get() is False
            for path in endpoints
        ):
            joint.CreateJointEnabledAttr().Set(False)
    robot_path = "/M20PiperVLA/M20_Piper"
    articulation_path = f"{robot_path}/M20/base_link"
    add_reference_to_stage(str(EXPECTED_USD), robot_path)
    configure_velocity_limits(stage, robot_path)

    arm_origin = STATION_BASE_XYZ + np.asarray([0.0, 0.0, 0.0888])
    object_local = np.asarray(manipulation.get("object_local_xyz", teacher["object_position"][0]), dtype=np.float64)
    goal_local = np.asarray(manipulation.get("goal_local_xyz", teacher["goal_position"][0]), dtype=np.float64)
    if object_local.shape != (3,) or goal_local.shape != (3,):
        raise ValueError("manipulation object_local_xyz and goal_local_xyz must be xyz vectors")
    # Compact pad: supports both poses without extending into the M20 footprint.
    table_center_local = 0.5 * (object_local + goal_local)
    table_center_local[2] = -0.025
    table_scale = np.asarray([
        abs(object_local[0] - goal_local[0]) + 0.09,
        abs(object_local[1] - goal_local[1]) + 0.09,
        0.05,
    ])
    table_position = local_to_world(table_center_local, arm_origin, STATION_YAW)
    object_position = np.asarray(
        manipulation.get("object_world_xyz", local_to_world(object_local, arm_origin, STATION_YAW)),
        dtype=np.float64,
    )
    goal_position = np.asarray(
        manipulation.get("goal_world_xyz", local_to_world(goal_local, arm_origin, STATION_YAW)),
        dtype=np.float64,
    )
    if object_position.shape != (3,) or goal_position.shape != (3,):
        raise ValueError("manipulation object_world_xyz and goal_world_xyz must be xyz vectors")
    material = PhysicsMaterial(
        prim_path="/M20PiperVLA/Materials/HighFriction",
        static_friction=1.5,
        dynamic_friction=1.2,
        restitution=0.0,
    )
    for surface_name, surface_config in (
        ("SourceSurfaceAdapter", manipulation.get("source_surface_adapter")),
        ("TargetSurfaceAdapter", manipulation.get("target_surface_adapter")),
    ):
        if surface_config:
            world.scene.add(FixedCuboid(
                prim_path=f"/M20PiperVLA/Station/{surface_name}",
                name=surface_name.lower(),
                position=np.asarray(surface_config["world_xyz"], dtype=np.float64),
                scale=np.asarray(surface_config["scale_xyz"], dtype=np.float64),
                visible=False,
                physics_material=material,
            ))
    floor_contract = {
        "mode": "authored_scene_collisions_unchanged",
        "source_scene_modified": False,
        "session_layer_floor_edits": False,
    }
    if route.name.startswith("factory_"):
        # The copied factory scene has one broad authored FloorCollider whose
        # contact response is unstable for the M20 wheel/leg policy.  Replace
        # only that collider in the anonymous session layer.  Other scenes,
        # especially the training-distribution warehouse, keep all authored
        # collision geometry exactly as composed.
        floor_candidates = [
            prim
            for prim in stage.Traverse()
            if prim.GetName() == "FloorCollider" and prim.HasAPI(UsdPhysics.CollisionAPI)
        ]
        if len(floor_candidates) != 1:
            raise RuntimeError(
                f"factory scene must expose exactly one FloorCollider: {len(floor_candidates)}"
            )
        floor_prim = floor_candidates[0]
        floor_cube = UsdGeom.Cube(floor_prim)
        floor_size = float(floor_cube.GetSizeAttr().Get())
        floor_matrix = UsdGeom.XformCache().GetLocalToWorldTransform(floor_prim)
        half = 0.5 * floor_size
        floor_top_z = max(
            float(floor_matrix.Transform(Gf.Vec3d(x, y, z))[2])
            for x in (-half, half)
            for y in (-half, half)
            for z in (-half, half)
        )
        floor_prim.SetActive(False)
        floor_adapter_material = PhysicsMaterial(
            prim_path="/M20PiperVLA/Materials/FactoryFloorAdapter",
            static_friction=0.5,
            dynamic_friction=0.5,
            restitution=0.8,
        )
        world.scene.add(GroundPlane(
            prim_path="/M20PiperVLA/FactoryFloorAdapter",
            name="factory_floor_adapter",
            size=200.0,
            z_position=floor_top_z,
            visible=False,
            physics_material=floor_adapter_material,
        ))
        floor_contract = {
            "mode": "factory_session_layer_ground_plane_adapter",
            "source_floor_prim": str(floor_prim.GetPath()),
            "source_floor_deactivated_in_session_layer": not floor_prim.IsActive(),
            "adapter_prim": "/M20PiperVLA/FactoryFloorAdapter",
            "top_z_m": floor_top_z,
            "material_prim": floor_adapter_material.prim_path,
            "static_friction": floor_adapter_material.get_static_friction(),
            "dynamic_friction": floor_adapter_material.get_dynamic_friction(),
            "restitution": floor_adapter_material.get_restitution(),
            "visual_scene_unchanged": True,
            "source_scene_modified": False,
            "session_layer_floor_edits": True,
        }
    table = None
    if bool(manipulation.get("create_support_table", True)):
        table = world.scene.add(FixedCuboid(
            prim_path="/M20PiperVLA/Station/Table",
            name="pick_table",
            position=table_position,
            scale=table_scale,
            color=np.asarray([0.34, 0.25, 0.15]),
            physics_material=material,
        ))
    if scene_object_prim:
        source_object = stage.GetPrimAtPath(scene_object_prim)
        if not source_object or not source_object.HasAPI(UsdPhysics.RigidBodyAPI):
            raise RuntimeError(f"scene_object_prim must be an existing rigid body: {scene_object_prim}")
        cube = world.scene.add(SingleRigidPrim(
            scene_object_prim,
            name=str(manipulation.get("object_name", "task_object")),
            reset_xform_properties=False,
        ))
        object_position = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(source_object).ExtractTranslation())
        scoring_object_prim = scene_object_prim
    else:
        object_scale = np.asarray(manipulation.get("object_scale_xyz", [0.04, 0.04, 0.04]))
        object_color = np.asarray(manipulation.get("object_color_rgb", [0.08, 0.28, 0.95]))
        object_common = dict(
            prim_path="/M20PiperVLA/Station/Object",
            name=str(manipulation.get("object_name", "task_object")),
            position=object_position,
            color=object_color,
            visible=PIPER_MOTION_ENABLED,
            mass=float(manipulation.get("object_mass_kg", 0.02)),
            physics_material=material,
        )
        if manipulation.get("object_shape", "cuboid") == "sphere":
            cube = world.scene.add(DynamicSphere(
                **object_common,
                radius=float(manipulation.get("object_radius_m", 0.04)),
            ))
        else:
            cube = world.scene.add(DynamicCuboid(**object_common, scale=object_scale))
        scoring_object_prim = "/M20PiperVLA/Station/Object"
    object_contact_api = PhysxSchema.PhysxContactReportAPI.Apply(stage.GetPrimAtPath(scoring_object_prim))
    object_contact_api.CreateThresholdAttr().Set(0.0)
    world.scene.add(VisualCuboid(
        prim_path="/M20PiperVLA/Station/Goal",
        name="green_goal",
        position=np.asarray([goal_position[0], goal_position[1], goal_position[2]]),
        scale=np.asarray([0.09, 0.09, 0.006]),
        color=np.asarray([0.05, 0.85, 0.18]),
        visible=False,
    ))
    spawn_xyz = np.asarray(route.spawn_xyz, dtype=np.float64) + np.asarray([
        args.spawn_x_offset,
        args.spawn_y_offset,
        0.0,
    ])
    robot = world.scene.add(SingleArticulation(
        # Bind the prim that owns ArticulationRootAPI so PhysX/Fabric updates
        # the complete high-precision robot in the live recording.
        prim_path=articulation_path,
        name="m20_piper_vla",
        position=spawn_xyz,
        orientation=yaw_quaternion(route.spawn_yaw_rad + args.spawn_yaw_offset),
    ))
    # The mobile chassis must be able to dock underneath the arm-height pad.
    # Filter only concrete M20 collision shapes.  Filtering against the
    # articulation root also suppresses object contacts with Piper fingers,
    # because M20 and Piper belong to the same 24-DOF articulation.
    m20_collision_paths = [
        prim.GetPath()
        for prim in stage.Traverse()
        if str(prim.GetPath()).startswith(f"{robot_path}/M20/")
        and prim.HasAPI(UsdPhysics.CollisionAPI)
    ]
    if not m20_collision_paths:
        raise RuntimeError("no concrete M20 collision prims found for station filtering")
    if table is not None:
        table_filter = UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath("/M20PiperVLA/Station/Table"))
        table_filter.CreateFilteredPairsRel().SetTargets(m20_collision_paths)
    cube_filter = UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath(scoring_object_prim))
    cube_filter.CreateFilteredPairsRel().SetTargets(m20_collision_paths)

    # Formal V4 evidence uses the PhysX contact reporter.  Apply it to every
    # enabled rigid body before reset.  Applying the API after PhysX has built
    # an SDF-backed actor invalidates existing tensor views; this occurs for
    # Q16's zero-DOF pot articulation and makes its contact sensor disappear.
    # Ordinary furniture was disabled above, so this set matches the enabled
    # A/D source surface without enabling unrelated static collision owners.
    contact_body_paths = []
    for prim in stage.Traverse():
        prim_path = str(prim.GetPath())
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        rigid_body_api = UsdPhysics.RigidBodyAPI(prim)
        rigid_body_enabled = rigid_body_api.GetRigidBodyEnabledAttr().Get()
        if rigid_body_enabled is not None and not bool(rigid_body_enabled):
            continue
        report_api = PhysxSchema.PhysxContactReportAPI.Apply(prim)
        report_api.CreateThresholdAttr().Set(0.0)
        contact_body_paths.append(prim_path)
    if not contact_body_paths:
        raise RuntimeError("no rigid bodies found for V4 contact evidence")
    carb.settings.get_settings().set_bool("/physics/disableContactProcessing", False)

    cameras = []
    if args.enable_cameras:
        # Keep the three robot-mounted VLA cameras explicit and stable in both
        # runtime requests and recorded artifacts.
        camera_specs = [
            ("m20_camera_0", (960, 540)),
            ("m20_camera_1", (960, 540)),
            ("piper_wrist", (960, 540)),
        ]
        if args.record_video:
            follow_resolution = (960, 540) if args.four_view_video else (1280, 720)
            camera_specs.append(("follow", follow_resolution))
        for name, resolution in camera_specs:
            cameras.append(world.scene.add(Camera(
                prim_path=f"/M20PiperVLA/Sensors/{name}_camera",
                name=f"{name}_camera",
                frequency=args.capture_hz,
                resolution=resolution,
                position=np.asarray([0.0, 0.0, 2.0]),
                orientation=np.asarray([1.0, 0.0, 0.0, 0.0]),
            )))
    lidar_stream = None
    if args.enable_cameras and not args.disable_lidar:
        write_startup_stage(args.output, "before_rtx_lidar_create")
        lidar_stream = RtxLidarStream.create(
            prim_path="/M20PiperVLA/Sensors/rtx_lidar",
            output_dir=args.output / "lidar",
            config={
                "model": "Example_Rotary",
                "translation_xyz": [0.28, 0.0, 0.42],
                "aux_output_level": "FULL",
                "max_range_m": 12.0,
            },
        )
    imu_sensor = world.scene.add(IMUSensor(
        prim_path=f"{articulation_path}/imu_sensor",
        name="m20_base_imu",
        frequency=POLICY_HZ,
        translation=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        orientation=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        linear_acceleration_filter_size=1,
        angular_velocity_filter_size=1,
        orientation_filter_size=1,
    ))
    write_startup_stage(
        args.output,
        "after_rtx_lidar_create_before_world_reset"
        if lidar_stream is not None
        else "before_world_reset_without_rtx_lidar",
    )
    world.reset()
    if lidar_stream is not None:
        write_startup_stage(args.output, "after_world_reset_before_rtx_lidar_initialize")
        # Flush the reset-triggered Hydra/Replicator work for the three camera
        # render products before LidarRtx synchronously creates its own product.
        for _ in range(8):
            world.render()
        write_startup_stage(args.output, "after_world_reset_render_flush_before_rtx_lidar_initialize")
        lidar_stream.initialize_after_world_reset()
        for _ in range(4):
            world.render()
        write_startup_stage(args.output, "after_rtx_lidar_initialize")
    else:
        write_startup_stage(args.output, "after_world_reset_without_rtx_lidar")
    base_visual_follower = BaseVisualFollower(stage, robot_path)
    base_position, base_orientation = robot.get_world_pose()
    base_visual_follower.set_world_pose(base_position, base_orientation)
    active_physics_scenes = sorted(
        str(prim.GetPath())
        for prim in stage.Traverse()
        if prim.IsActive() and prim.IsA(UsdPhysics.Scene)
    )
    if active_physics_scenes != ["/M20PiperVLA/PhysicsScene"]:
        raise RuntimeError(f"runtime must have exactly one active PhysicsScene: {active_physics_scenes}")
    active_collision_count = sum(
        1
        for prim in stage.Traverse()
        if prim.IsActive() and prim.HasAPI(UsdPhysics.CollisionAPI)
    )
    if active_collision_count < route.scene.minimum_collision_prim_count:
        raise RuntimeError(
            f"runtime active collision count {active_collision_count} "
            f"< {route.scene.minimum_collision_prim_count}"
        )
    goal_marker_visibility = str(
        UsdGeom.Imageable(
            stage.GetPrimAtPath("/M20PiperVLA/Station/Goal")
        ).ComputeVisibility()
    )
    if goal_marker_visibility != str(UsdGeom.Tokens.invisible):
        raise RuntimeError(
            f"goal marker must be hidden, got {goal_marker_visibility}"
        )
    synthetic_object_visibility = None
    if not scene_object_prim:
        synthetic_object_visibility = str(
            UsdGeom.Imageable(
                stage.GetPrimAtPath("/M20PiperVLA/Station/Object")
            ).ComputeVisibility()
        )
        if (
            not PIPER_MOTION_ENABLED
            and synthetic_object_visibility != str(UsdGeom.Tokens.invisible)
        ):
            raise RuntimeError(
                "non-manipulation synthetic object marker must be hidden, got "
                f"{synthetic_object_visibility}"
            )
    runtime_scene_contract = {
        "root_layer": str(root_layer_path),
        "root_layer_sha256": sha256(root_layer_path),
        "default_prim": default_prim,
        "source_collision_prim_count": source_collision_count,
        "active_collision_prim_count": active_collision_count,
        "minimum_collision_prim_count": route.scene.minimum_collision_prim_count,
        "source_physics_scene": route.scene.source_physics_scene,
        "source_physics_scene_deactivated_in_session_layer": (
            not source_physics.IsActive() if source_physics is not None else None
        ),
        "active_physics_scenes": active_physics_scenes,
        "robot_asset_prim_path": robot_path,
        "robot_articulation_prim_path": articulation_path,
        "imu_sensor": {
            "prim_path": str(imu_sensor.prim_path),
            "frame": "M20 base_link local frame",
            "frequency_hz": POLICY_HZ,
            "read_gravity": True,
            "linear_acceleration_source": "isaacsim.sensors.physics.IMUSensor",
        },
        "floor_collision_contract": floor_contract,
        "tracked_distractor_freeze": {
            "mode": "q17_semantic_alternatives_static_collision",
            "source_scene_modified": False,
            "session_layer_only": True,
            "selected_dynamic_object": scene_object_prim or None,
            "frozen_object_count": len(frozen_tracked_distractor_owners),
            "frozen_rigid_body_owners": dict(
                sorted(frozen_tracked_distractor_owners.items())
            ),
        },
        "runtime_visual_contract": base_visual_follower.report,
        "task_visual_markers": {
            "goal_visibility": goal_marker_visibility,
            "synthetic_object_visibility": synthetic_object_visibility,
            "piper_motion_enabled": PIPER_MOTION_ENABLED,
        },
    }
    return (
        world,
        robot,
        cube,
        table,
        cameras,
        object_position,
        goal_position,
        base_visual_follower,
        lidar_stream,
        imu_sensor,
        runtime_scene_contract,
    )


def configure_articulation(robot):
    names = list(robot.dof_names)
    if len(names) != 24 or set(names) != EXPECTED_DOF_NAMES:
        raise RuntimeError(f"PhysX 24-DOF contract mismatch: {names}")
    controller = robot.get_articulation_controller()
    kps = np.asarray([KP_BY_NAME[name] for name in names], dtype=np.float32)
    kds = np.asarray([KD_BY_NAME[name] for name in names], dtype=np.float32)
    # Match the high-PD Piper configuration that produced the source teacher
    # demonstrations.  M20 leg/wheel gains remain on the deployment contract.
    for joint_number in range(1, 4):
        kps[names.index(f"joint{joint_number}")] = 400.0
        kds[names.index(f"joint{joint_number}")] = 80.0
    for joint_number in range(4, 7):
        kps[names.index(f"joint{joint_number}")] = 200.0
        kds[names.index(f"joint{joint_number}")] = 40.0
    for joint_number in (7, 8):
        kps[names.index(f"joint{joint_number}")] = 800.0
        kds[names.index(f"joint{joint_number}")] = 20.0
    efforts = np.asarray([EFFORT_LIMIT_BY_NAME[name] for name in names], dtype=np.float32)
    controller.set_gains(kps=kps, kds=kds, save_to_usd=False)
    controller.set_max_efforts(efforts)
    wheel_indices = np.asarray([names.index(name) for name in WHEEL_JOINTS], dtype=np.int32)
    wheel_armatures = np.asarray([[ARMATURE_BY_NAME[name] for name in WHEEL_JOINTS]], dtype=np.float32)
    robot._articulation_view.set_armatures(wheel_armatures, joint_indices=wheel_indices)
    initial = np.asarray([DEFAULT_POSITION_BY_NAME.get(name, 0.0) for name in names], dtype=np.float32)
    initial[names.index("joint7")] = 0.025
    initial[names.index("joint8")] = -0.025
    robot.set_joint_positions(initial)
    robot.set_joint_velocities(np.zeros(24, dtype=np.float32))
    controller.apply_action(ArticulationAction(joint_positions=initial.copy()))
    limits = np.asarray(robot._articulation_view.get_dof_limits(), dtype=np.float64)[0]
    return {
        "names": names,
        "index": {name: names.index(name) for name in names},
        "controller": controller,
        "locomotion_kps": kps.copy(),
        "locomotion_kds": kds.copy(),
        "lower": limits[:, 0],
        "upper": limits[:, 1],
    }


class Collector:
    def __init__(
        self,
        route,
        teacher,
        world,
        robot,
        cube,
        goal_position,
        cameras,
        articulation,
        base_visual_follower,
        lidar_stream,
        imu_sensor,
    ):
        self.route = route
        self.teacher = teacher
        self.world = world
        self.robot = robot
        self.cube = cube
        self.goal_position = np.asarray(goal_position, dtype=np.float64)
        self.cameras = cameras
        self.articulation = articulation
        self.base_visual_follower = base_visual_follower
        self.lidar_stream = lidar_stream
        self.imu_sensor = imu_sensor
        self.four_view = FourViewComposer()
        rng = np.random.default_rng(args.seed)
        self.navigation_waypoints_xy = np.asarray(route.waypoints_xy, dtype=np.float64)
        if args.waypoint_jitter_m > 0.0:
            # Keep the last three points fixed: their relative geometry is the
            # calibrated docking approach that sets the Piper/table heading.
            jitter = rng.uniform(
                -args.waypoint_jitter_m,
                args.waypoint_jitter_m,
                size=(len(self.navigation_waypoints_xy) - 3, 2),
            )
            self.navigation_waypoints_xy[:-3] += jitter
        self.names = articulation["names"]
        self.index = articulation["index"]
        self.controller = articulation["controller"]
        self.gripper_max_opening = float(
            min(
                articulation["upper"][self.index["joint7"]],
                -articulation["lower"][self.index["joint8"]],
            )
        )
        if not math.isfinite(self.gripper_max_opening) or self.gripper_max_opening <= 0.0:
            raise RuntimeError(
                f"invalid Piper finger opening limit: {self.gripper_max_opening}"
            )
        self.robot_order = np.asarray([self.index[name] for name in ROBOT_ORDER], dtype=np.int32)
        self.body_names = list(robot._articulation_view.body_names)
        for required in ("arm_base_link", "gripper_base"):
            if required not in self.body_names:
                raise RuntimeError(f"missing body {required}: {self.body_names}")
        self.arm_body = self.body_names.index("arm_base_link")
        self.gripper_body = self.body_names.index("gripper_base")
        self.session = None if args.kinematic_base else ort.InferenceSession(
            str(POLICY), providers=["CPUExecutionProvider"]
        )
        self.previous_m20_action = np.zeros(16, dtype=np.float32)
        self.stationary_leg_entry_targets: np.ndarray | None = None
        self.stationary_control_steps = 0
        self.stationary_activation_count = 0
        self.locomotion_kps = np.asarray(
            articulation["locomotion_kps"], dtype=np.float32
        ).copy()
        self.locomotion_kds = np.asarray(
            articulation["locomotion_kds"], dtype=np.float32
        ).copy()
        self.stationary_kps = self.locomotion_kps.copy()
        self.stationary_kds = self.locomotion_kds.copy()
        for name in LEG_JOINTS:
            self.stationary_kps[self.index[name]] = STATIONARY_LEG_STIFFNESS
            self.stationary_kds[self.index[name]] = STATIONARY_LEG_DAMPING
        # The factory and kitchen floor profiles need stronger zero-speed braking.
        # Keep the Market profile and MOVE gains at their calibrated values.
        if str(TASK_CONFIG.get("task_id", "")).startswith(("m20_factory_", "m20_kitchen_", "m20_parlor_")):
            for name in WHEEL_JOINTS:
                self.stationary_kds[self.index[name]] = 6.0
        self.stationary_drive_mode = False
        # The expert controller treats Piper targets as commanded state.  Keep
        # that state independent of measured sag so zero VLA deltas really hold
        # the last pose instead of ratcheting the arm downward.
        self.piper_command_targets: np.ndarray | None = None
        self.current_control_phase = "initialize"
        self.physics_step = 0
        self.task_start_physics_step: int | None = None
        self._clock_audit = {"callbacks": 0, "simulated_seconds": 0.0}
        def _record_physics_tick(dt):
            self._clock_audit["callbacks"] += 1
            self._clock_audit["simulated_seconds"] += float(dt)
        world.add_physics_callback("release_clock_audit", _record_physics_tick)
        initial_base_position, initial_base_orientation = self.robot.get_world_pose()
        self.kinematic_base_position = np.asarray(
            initial_base_position, dtype=np.float64
        ).copy()
        self.kinematic_base_standing_z = float(self.kinematic_base_position[2])
        self.kinematic_base_yaw = float(
            yaw_from_quaternion(initial_base_orientation)
        )
        self.kinematic_command = np.zeros(3, dtype=np.float32)
        self.kinematic_position_indices = np.asarray(
            [self.index[name] for name in (*LEG_JOINTS, *PIPER_JOINTS)],
            dtype=np.int32,
        )
        self.kinematic_position_targets = np.asarray(
            self.robot.get_joint_positions(), dtype=np.float32
        )[self.kinematic_position_indices].copy()
        self.m20_camera_0_images = []
        self.m20_camera_1_images = []
        self.piper_wrist_images = []
        # Compatibility aliases used by the original submission writer.
        self.images = self.m20_camera_0_images
        self.wrist_images = self.piper_wrist_images
        self.states = []
        self.imu_angular_velocities = []
        self.imu_linear_accelerations = []
        self.imu_projected_gravities = []
        self.actions = []
        self.phases = []
        self.times = []
        self.object_heights = []
        self.scoring_active = False
        self.scoring_initial: dict[str, np.ndarray] = {}
        self.scoring_samples: list[dict[str, np.ndarray | float]] = []
        runtime_articulation_prim = Path(str(robot.prim_path))
        if len(runtime_articulation_prim.parents) < 2:
            raise RuntimeError(f"unexpected M20 articulation Prim: {runtime_articulation_prim}")
        # ArticulationRootAPI lives on M20/base_link, while the coupled Piper
        # joints are siblings under the shared M20_Piper evidence entity.
        self.scoring_articulation_prim = str(runtime_articulation_prim.parents[1])
        self.scoring_object_prim = str(
            TASK_CONFIG.get("manipulation", {}).get(
                "scene_object_prim", "/M20PiperVLA/Station/Object"
            )
        )
        self.scoring_target_prim = "/M20PiperVLA/Station/Goal"
        self.scoring_joint_paths, self.scoring_joint_indices = self._standard_joint_order()
        self.scoring_body_paths, self.scoring_body_indices = self._standard_body_order()
        self.scoring_contact_body_paths = list(self.scoring_body_paths)
        physics_view = SimulationManager.get_physics_sim_view()
        self.scoring_contact_views = [
            physics_view.create_rigid_contact_view(
                body_path,
                filter_patterns=[self.scoring_object_prim],
                max_contact_data_count=64,
            )
            for body_path in self.scoring_contact_body_paths
        ]
        # Attack interaction events are sampled at policy frequency, while a
        # fingertip contact can begin and end between two policy snapshots.
        # Keep a clean PhysX-step latch for the three terminal gripper bodies;
        # this never mutates the object and is not used for Judge truth.
        attack_grasp_body_names = {"gripper_base", "link7", "link8"}
        self.attack_grasp_contact_indices = [
            index
            for index, path in enumerate(self.scoring_contact_body_paths)
            if Path(path).name in attack_grasp_body_names
        ]
        if not self.attack_grasp_contact_indices:
            raise RuntimeError("no Piper gripper contact bodies are available")
        self._attack_grasp_initial_height: float | None = None
        self._attack_grasp_contact_latch: dict[str, Any] | None = None
        self._attack_grasp_confirmed_evidence: dict[str, Any] | None = None
        force_torque_config = dict(
            TASK_CONFIG.get("manipulation", {}).get("force_torque_sensor", {})
        )
        self.force_torque_sensor_enabled = bool(
            force_torque_config.get("enabled", False)
        )
        self.force_torque_sensor_id = str(
            force_torque_config.get("sensor_id", "piper_tcp_force_torque")
        )
        self.force_torque_frame_id = str(
            force_torque_config.get("frame_id", "piper_tcp")
        )
        self.force_torque_grip_threshold_n = float(
            force_torque_config.get("grip_force_threshold_n", 5.0)
        )
        self.force_torque_contact_indices: list[int] = []
        if self.force_torque_sensor_enabled:
            configured_finger_paths = [
                str(value) for value in force_torque_config.get("finger_body_paths", [])
            ]
            if len(configured_finger_paths) != 2:
                raise ValueError(
                    "force_torque_sensor.finger_body_paths must name exactly two Piper fingers"
                )
            missing_finger_paths = sorted(
                set(configured_finger_paths) - set(self.scoring_contact_body_paths)
            )
            if missing_finger_paths:
                raise RuntimeError(
                    "force/torque finger bodies are absent from the final articulation: "
                    f"{missing_finger_paths}"
                )
            self.force_torque_contact_indices = [
                self.scoring_contact_body_paths.index(path)
                for path in configured_finger_paths
            ]
            if not 0.0 < self.force_torque_grip_threshold_n <= 1000.0:
                raise ValueError(
                    "force_torque_sensor.grip_force_threshold_n must be within (0, 1000]"
                )
        task_id = str(TASK_CONFIG.get("task_id", "")).strip()
        instruction = str(TASK_CONFIG.get("instruction", "")).strip()
        if not task_id or not instruction:
            raise ValueError("v0.6.0 fixed evidence requires task_id and instruction")
        # The bridge below is deliberately fail-closed to the one frozen Q17
        # task upgraded by q17_runtime_compat.  Other tasks, including future
        # force/torque tasks, retain the pre-existing state semantics.
        self.q17_force_feedback_bridge_enabled = bool(
            self.force_torque_sensor_enabled
            and str(TASK_CONFIG.get("question_id", "")).strip().lower() == "q17"
            and task_id == "m20_market_bakery_bread_return_gripper_feedback"
        )
        stage = omni.usd.get_context().get_stage()
        self.final_stage = stage
        # v0.7.2: freeze the complete state:* bool surface from the final
        # composed Stage.  task_id, attack profile and Rule Package never
        # participate in this discovery.
        discovered_usd_boolean_attributes = discover_usd_boolean_attributes(stage)
        concrete_usd_boolean_attributes = []
        unavailable_usd_boolean_attributes = []
        for spec in discovered_usd_boolean_attributes:
            prim = stage.GetPrimAtPath(spec.prim_path)
            attribute = prim.GetAttribute(spec.attribute_name) if prim and prim.IsValid() else None
            value = attribute.Get() if attribute and attribute.IsValid() else None
            if isinstance(value, (bool, np.bool_)):
                concrete_usd_boolean_attributes.append(spec)
            else:
                unavailable_usd_boolean_attributes.append({
                    "prim_path": spec.prim_path,
                    "attribute_name": spec.attribute_name,
                    "reason": "no_concrete_bool_value_on_final_composed_stage",
                })
        self.usd_boolean_attributes = tuple(concrete_usd_boolean_attributes)
        (args.output / "usd_boolean_attribute_discovery.json").write_text(
            json.dumps({
                "schema": "m20-usd-boolean-attribute-discovery/v1",
                "policy": "record_concrete_bool_values_only_without_default_invention",
                "discovered_count": len(discovered_usd_boolean_attributes),
                "recorded_count": len(self.usd_boolean_attributes),
                "unavailable_count": len(unavailable_usd_boolean_attributes),
                "unavailable": unavailable_usd_boolean_attributes,
            }, indent=2) + "\n",
            encoding="utf-8",
        )
        self.fixed_evidence_runtime = FixedEvidenceRuntime(
            None,
            stage,
            task_id=task_id,
            instruction=instruction,
            robot_role="m20_piper",
            formal_stage_digest=str(route.scene.sha256),
            physics_dt=PHYSICS_DT,
            origin_xyz=np.zeros(3, dtype=np.float32),
            evidence_prim_paths=tuple(sorted({
                *(str(value) for value in TASK_CONFIG.get("tracked_objects", {}).values()),
                self.scoring_object_prim,
                *SAFETY_REFERENCED_PRIM_PATHS,
            })),
        )
        fixed_manifest = self.fixed_evidence_runtime.manifest()
        self.fixed_evidence_manifest = fixed_manifest
        (args.output / "fixed_evidence_runtime_plan_v3.json").write_text(
            json.dumps(fixed_manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        self.fixed_evidence_sidecar = FixedEvidenceSidecarWriter(
            args.output / "fixed_evidence_sidecar.hdf5",
            fixed_manifest,
            usd_boolean_attributes=self.usd_boolean_attributes,
        )
        self.fixed_evidence_completed = False
        self.rubric_evidence_runtime = None
        self.rubric_evidence_writer = None
        if TASK_CONFIG.get("rubric_evidence"):
            self.rubric_evidence_runtime = RubricEvidenceRuntime(
                stage, TASK_CONFIG["rubric_evidence"], PHYSICS_DT,
                sim_view=self.fixed_evidence_runtime.sim_view,
            )
            rubric_manifest = self.rubric_evidence_runtime.manifest()
            (args.output / "rubric_evidence_runtime_plan.json").write_text(
                json.dumps(rubric_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            self.rubric_evidence_writer = RubricEvidenceWriter(
                args.output / "rubric_evidence_sidecar.hdf5", rubric_manifest)
        success_contact = dict(TASK_CONFIG.get("success", {}).get("contact", {}))
        self.success_contact_target_id = success_contact.get("target_id")
        self.success_contact_target_prim = success_contact.get("target_prim")
        self.success_contact_threshold_n = float(success_contact.get("minimum_force_n", 0.01))
        self.success_contact_views = []
        if self.success_contact_target_prim:
            stage = omni.usd.get_context().get_stage()
            target_prefix = str(self.success_contact_target_prim).rstrip("/") + "/"
            target_root = stage.GetPrimAtPath(str(self.success_contact_target_prim))
            target_colliders = sorted({
                *(
                    [str(target_root.GetPath())]
                    if target_root and (
                        target_root.HasAPI(UsdPhysics.CollisionAPI)
                        or target_root.HasAPI(UsdPhysics.RigidBodyAPI)
                    ) else []
                ),
                *(
                    str(prim.GetPath()) for prim in stage.Traverse()
                    if str(prim.GetPath()).startswith(target_prefix)
                    and (
                        prim.HasAPI(UsdPhysics.CollisionAPI)
                        or prim.HasAPI(UsdPhysics.RigidBodyAPI)
                    )
                ),
            })
            if not target_colliders:
                raise RuntimeError(
                    "success contact target has no collision shapes or rigid bodies: "
                    f"{self.success_contact_target_prim}"
                )
            self.success_contact_views = [
                physics_view.create_rigid_contact_view(
                    self.scoring_object_prim,
                    filter_patterns=[target_collider],
                    max_contact_data_count=64,
                )
                for target_collider in target_colliders
            ]
        self.latest_unified_action = np.zeros(10, dtype=np.float32)
        self.vla_trace: list[dict] = []
        self.policy_execution_events: list[dict[str, Any]] = []
        self.policy_xy_origin = np.asarray(args.policy_xy_origin, dtype=np.float32)
        self.video_frame_count = 0
        self.video_frames = args.output / ".video_frames"
        self._direct_command_override: np.ndarray | None = None
        self._direct_command_until_step = -1
        self._gripper_release_opening = 0.05
        self._gripper_release_until_step = -1
        self._lidar_filter = None
        self._latest_lidar_frame: dict | None = None
        self._latest_lidar_step = -1
        self._lidar_planner = M20LidarLocalPlanner()
        self._imu_transform = None
        self._visual_material_override: dict[str, Any] | None = None
        self.attack_boundary = None
        self.assurance_bridge = None
        self._assurance_initialized = False
        self._force_torque_attack_window_open = False
        self._force_torque_placement_completed = False
        self._q17_force_feedback_positive = False
        self._q17_force_feedback_attack_ids: tuple[str, ...] = ()
        self._q17_force_feedback_payload_changed = False
        self._q17_clean_grip_force_n = math.nan
        self._q17_attacked_grip_force_n = math.nan
        self._pending_q17_policy_bridge: dict[str, Any] | None = None
        self.task_config = TASK_CONFIG
        self.squat_config = resolve_squat_config(self.task_config)
        self.squat_enabled = bool(self.squat_config.get("enabled", False))
        self.piper_motion_enabled = resolve_piper_motion_enabled(
            self.task_config,
            navigation_only=bool(args.navigation_only),
            squat_enabled=self.squat_enabled,
        )
        if self.piper_motion_enabled != PIPER_MOTION_ENABLED:
            raise RuntimeError("Piper task capability changed after scene creation")
        self.piper_hold_targets = self.measured_piper_targets()
        if not self.piper_motion_enabled:
            self.piper_command_targets = self.piper_hold_targets.copy()
        self.squat_required_cycles = int(self.squat_config.get("required_cycles", 0))
        self.squat_action_index = int(self.squat_config.get("action_index", 5))
        self.squat_command_threshold = float(self.squat_config.get("command_threshold", 0.006))
        self.squat_joint_offset_rad = float(self.squat_config.get("joint_offset_rad", 0.22))
        self.squat_low_drop_m = float(self.squat_config.get("low_drop_m", 0.08))
        self.squat_stand_tolerance_m = float(self.squat_config.get("stand_tolerance_m", 0.04))
        self.squat_stable_steps = int(self.squat_config.get("stable_steps", 3))
        self.squat_cycles = 0
        self.squat_phase = "standing"
        self.squat_low_steps = 0
        self.squat_stand_steps = 0
        self.squat_command = 0
        self.squat_minimum_height_m = math.inf
        self.squat_standing_reference_m: float | None = None
        self.squat_trace: list[dict] = []
        if self.squat_enabled:
            if not 0.0 < self.squat_command_threshold <= 0.015:
                raise ValueError("squat_control.command_threshold must be within (0, 0.015]")
            if not 0.05 <= self.squat_joint_offset_rad <= 0.35:
                raise ValueError("squat_control.joint_offset_rad must be within [0.05, 0.35]")
            if not 0.05 <= self.squat_low_drop_m <= 0.18:
                raise ValueError("squat_control.low_drop_m must be within [0.05, 0.18]")
            if not 0.01 <= self.squat_stand_tolerance_m <= 0.08:
                raise ValueError("squat_control.stand_tolerance_m must be within [0.01, 0.08]")
            if not 1 <= self.squat_stable_steps <= 20:
                raise ValueError("squat_control.stable_steps must be within [1, 20]")
        manipulation = dict(self.task_config.get("manipulation", {}))
        self.task_object_id = str(manipulation.get("object_id", "task_object"))
        self.contact_distance_m = float(manipulation.get("contact_distance_m", 0.12))
        self.tracked_object_prims = {
            str(key): str(value)
            for key, value in self.task_config.get("tracked_objects", {}).items()
        }
        self.tracked_object_prims.setdefault(
            self.task_object_id, "/M20PiperVLA/Station/Object"
        )
        self.safety_profile = SAFETY_PROFILE
        self.safety_monitor: EpisodeSafetyMonitor | None = None
        self.safety_contact_tracker: SafetyContactTracker | None = None
        self.safety_geometry_tracker: SafetyGeometryTracker | None = None
        self.safety_enabled_rules: tuple[str, ...] = ()
        self.safety_prismatic_joint_names: tuple[str, ...] = ()
        self.safety_state_prim_paths: tuple[str, ...] = ()
        self.safety_joint_velocity_limits = np.asarray(
            [VELOCITY_LIMIT_BY_NAME[name] for name in self.names],
            dtype=np.float64,
        )
        self.safety_joint_effort_limits = np.asarray(
            [EFFORT_LIMIT_BY_NAME[name] for name in self.names],
            dtype=np.float64,
        )
        if self.safety_profile is not None:
            self.safety_enabled_rules = enabled_rule_ids(self.safety_profile)
            joint_prims = {
                prim.GetName(): prim
                for prim in self.final_stage.Traverse()
                if str(prim.GetPath()).startswith(self.scoring_articulation_prim + "/")
                and prim.IsA(UsdPhysics.Joint)
                and not prim.IsA(UsdPhysics.FixedJoint)
                and prim.GetName() in self.names
            }
            if "S11" in self.safety_enabled_rules and set(joint_prims) != set(self.names):
                raise RuntimeError(
                    "S11 cannot inventory every articulation joint on the final Stage: "
                    f"missing={sorted(set(self.names) - set(joint_prims))}, "
                    f"unexpected={sorted(set(joint_prims) - set(self.names))}"
                )
            self.safety_prismatic_joint_names = tuple(sorted(
                name
                for name, prim in joint_prims.items()
                if prim.IsA(UsdPhysics.PrismaticJoint)
            ))
            required_safety_body_names: set[str] = set()
            if "S01" in self.safety_enabled_rules:
                source_names = set(
                    self.safety_profile["rules"]["S01"]["parameters"]["source_body_names"]
                )
                required_safety_body_names.update(source_names)
            if "S04" in self.safety_enabled_rules:
                for item in self.safety_profile["rules"]["S04"]["parameters"]["objects"]:
                    required_safety_body_names.update(item["gripper_body_names"])
            if "S06" in self.safety_enabled_rules:
                for item in self.safety_profile["rules"]["S06"]["parameters"]["limits"]:
                    required_safety_body_names.update(item["source_body_names"])
            if "S14" in self.safety_enabled_rules:
                for item in self.safety_profile["rules"]["S14"]["parameters"]["placements"]:
                    required_safety_body_names.update(item["gripper_body_names"])
            if "S07" in self.safety_enabled_rules:
                parameters = self.safety_profile["rules"]["S07"]["parameters"]
                required_safety_body_names.add(parameters["base_body_name"])
                required_safety_body_names.update(parameters["wheel_body_names"])
            for rule_id in ("S09", "S10"):
                if rule_id not in self.safety_enabled_rules:
                    continue
                for metric in self.safety_profile["rules"][rule_id]["parameters"]["metrics"]:
                    required_safety_body_names.add(metric["body_name"])
                    if metric.get("reference_body_name") is not None:
                        required_safety_body_names.add(metric["reference_body_name"])
            missing_safety_bodies = sorted(
                required_safety_body_names - set(self.body_names)
            )
            if missing_safety_bodies:
                raise RuntimeError(
                    "safety profile bodies are absent from the final articulation: "
                    f"{missing_safety_bodies}"
                )
            self.safety_monitor = EpisodeSafetyMonitor(
                self.safety_profile,
                args.output,
                PHYSICS_DT,
            )
            state_paths: set[str] = set()
            if "S03" in self.safety_enabled_rules:
                for item in self.safety_profile["rules"]["S03"]["parameters"]["regions"]:
                    state_paths.update(item.get("carried_object_prim_paths", []))
            if "S04" in self.safety_enabled_rules:
                state_paths.update(
                    item["prim_path"]
                    for item in self.safety_profile["rules"]["S04"]["parameters"]["objects"]
                )
            if "S05" in self.safety_enabled_rules:
                state_paths.update(
                    item["prim_path"]
                    for item in self.safety_profile["rules"]["S05"]["parameters"]["checks"]
                )
            if "S05" in self.safety_enabled_rules:
                state_paths.update(item['relative_body_prim'] for item in self.safety_profile['rules']['S05']['parameters']['checks'] if 'relative_body_prim' in item)
            if "S08" in self.safety_enabled_rules:
                state_paths.update(
                    self.safety_profile["rules"]["S08"]["parameters"].get(
                        "carried_object_prim_paths", []
                    )
                )
            if "S12" in self.safety_enabled_rules:
                state_paths.update(
                    item["prim_path"]
                    for item in self.safety_profile["rules"]["S12"]["parameters"]["objects"]
                )
            if "S13" in self.safety_enabled_rules:
                state_paths.update(
                    item["prim_path"]
                    for item in self.safety_profile["rules"]["S13"]["parameters"]["objects"]
                )
            if "S14" in self.safety_enabled_rules:
                state_paths.update(
                    item["object_prim_path"]
                    for item in self.safety_profile["rules"]["S14"]["parameters"]["placements"]
                )
            self.safety_state_prim_paths = tuple(sorted(state_paths))
            missing_state_prims = [
                path
                for path in self.safety_state_prim_paths
                if not self.final_stage.GetPrimAtPath(path).IsValid()
            ]
            if missing_state_prims:
                raise RuntimeError(
                    "safety profile state Prims are absent from the final Stage: "
                    f"{missing_state_prims}"
                )
            if {"S01", "S02", "S04", "S06", "S07", "S14"} & set(self.safety_enabled_rules):
                self.safety_contact_tracker = SafetyContactTracker(
                    self.final_stage,
                    robot_root_prim=self.scoring_articulation_prim,
                    physics_dt=PHYSICS_DT,
                )
            if {"S03", "S08"} & set(self.safety_enabled_rules):
                self.safety_geometry_tracker = SafetyGeometryTracker(
                    self.final_stage,
                    self.safety_profile,
                    robot_root_prim=self.scoring_articulation_prim,
                )
            (args.output / "safety_runtime_plan.json").write_text(
                json.dumps(self.safety_runtime_manifest(), ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        self.joint_attack_config = None
        self.joint_attack_samples = 0
        self.joint_attack_active_samples = 0
        self.joint_attack_trace_path = args.output / "joint_attack_trace.jsonl"
        if args.joint_attack_config is not None:
            self.joint_attack_config = json.loads(
                args.joint_attack_config.read_text(encoding="utf-8")
            )
            if self.joint_attack_config.get("schema") != "m20-joint-command-attack/v1":
                raise ValueError("unsupported M20 joint attack schema")
            offsets = self.joint_attack_config.get("joint_offsets_rad")
            if not isinstance(offsets, dict) or not offsets:
                raise ValueError("joint attack requires non-empty joint_offsets_rad")
            unknown = sorted(set(offsets) - set(LEG_JOINTS))
            if unknown:
                raise ValueError(f"joint attack names unknown M20 leg joints: {unknown}")
            start = int(self.joint_attack_config["start_physics_step"])
            end = int(self.joint_attack_config["end_physics_step"])
            if start < 0 or end < start:
                raise ValueError("invalid joint attack physics-step interval")
            if self.joint_attack_trace_path.exists():
                raise FileExistsError(
                    f"refusing to overwrite joint attack trace {self.joint_attack_trace_path}"
                )
        if args.record_video:
            if self.video_frames.exists():
                raise FileExistsError(f"refusing to reuse video frame directory {self.video_frames}")
            self.video_frames.mkdir(parents=True)

    def link_transform(self, body_index: int) -> tuple[np.ndarray, np.ndarray]:
        value = np.asarray(self.robot._articulation_view._physics_view.get_link_transforms(), dtype=np.float64)[0, body_index]
        quaternion = np.asarray([value[6], value[3], value[4], value[5]], dtype=np.float64)
        return value[:3], quaternion / np.linalg.norm(quaternion)

    def _standard_joint_order(self) -> tuple[list[str], np.ndarray]:
        stage = omni.usd.get_context().get_stage()
        prefix = self.scoring_articulation_prim.rstrip("/") + "/"
        paths = sorted(
            str(prim.GetPath()) for prim in stage.Traverse()
            if str(prim.GetPath()).startswith(prefix) and prim.IsA(UsdPhysics.Joint)
            and prim.GetName() in self.names
        )
        if len(paths) != len(self.names):
            raise RuntimeError(f"V4 scoring joint ordering mismatch: {len(paths)} != {len(self.names)}")
        indices = np.asarray([self.index[Path(path).name] for path in paths], dtype=np.int32)
        return paths, indices

    def _standard_body_order(self) -> tuple[list[str], np.ndarray]:
        stage = omni.usd.get_context().get_stage()
        prefix = self.scoring_articulation_prim.rstrip("/") + "/"
        candidates = sorted(
            str(prim.GetPath()) for prim in stage.Traverse()
            if str(prim.GetPath()).startswith(prefix) and prim.HasAPI(UsdPhysics.RigidBodyAPI)
        )
        runtime_by_name: dict[str, list[int]] = {}
        for index, name in enumerate(self.body_names):
            runtime_by_name.setdefault(str(name), []).append(index)
        paths, indices = [], []
        for path in candidates:
            matches = runtime_by_name.get(Path(path).name, [])
            if len(matches) == 1:
                paths.append(path)
                indices.append(matches[0])
        if len(paths) != len(self.body_names):
            raise RuntimeError(f"V4 scoring body ordering mismatch: {len(paths)} != {len(self.body_names)}")
        return paths, np.asarray(indices, dtype=np.int32)

    @staticmethod
    def _pose(position, orientation) -> np.ndarray:
        position = np.asarray(position, dtype=np.float32).reshape(3)
        orientation = np.asarray(orientation, dtype=np.float32).reshape(4)
        norm = float(np.linalg.norm(orientation))
        if not np.isfinite(position).all() or not np.isfinite(orientation).all() or norm <= 1e-12:
            raise RuntimeError("V4 scoring pose is non-finite or has a zero quaternion")
        return np.concatenate((position, orientation / norm)).astype(np.float32)

    @staticmethod
    def _velocity(linear, angular) -> np.ndarray:
        value = np.concatenate((np.asarray(linear).reshape(3), np.asarray(angular).reshape(3))).astype(np.float32)
        if not np.isfinite(value).all():
            raise RuntimeError("V4 scoring velocity is non-finite")
        return value

    def _scoring_snapshot(self) -> dict[str, np.ndarray]:
        root_position, root_orientation = self.robot.get_world_pose()
        object_position, object_orientation = self.cube.get_world_pose()
        transforms = np.asarray(
            self.robot._articulation_view._physics_view.get_link_transforms(), dtype=np.float32
        )[0, self.scoring_body_indices]
        body_pose = np.concatenate((transforms[:, :3], transforms[:, [6, 3, 4, 5]]), axis=1)
        body_pose[:, 3:] /= np.linalg.norm(body_pose[:, 3:], axis=1, keepdims=True)
        body_velocity = np.asarray(
            self.robot._articulation_view._physics_view.get_link_velocities(), dtype=np.float32
        )[0, self.scoring_body_indices]
        return {
            "robot_root_pose": self._pose(root_position, root_orientation),
            "robot_root_velocity": self._velocity(
                self.robot.get_linear_velocity(), self.robot.get_angular_velocity()
            ),
            "robot_joint_position": np.asarray(self.robot.get_joint_positions(), dtype=np.float32)[self.scoring_joint_indices],
            "robot_joint_velocity": np.asarray(self.robot.get_joint_velocities(), dtype=np.float32)[self.scoring_joint_indices],
            "robot_body_pose": body_pose.astype(np.float32),
            "robot_body_velocity": body_velocity.astype(np.float32),
            "object_root_pose": self._pose(object_position, object_orientation),
            "object_root_velocity": self._velocity(
                self.cube.get_linear_velocity(), self.cube.get_angular_velocity()
            ),
            "contact_net_forces_w": self._contact_net_forces_w(),
        }

    def _contact_net_forces_w(self) -> np.ndarray:
        forces = []
        for view in self.scoring_contact_views:
            matrix = view.get_contact_force_matrix(dt=PHYSICS_DT)
            value = matrix.detach().cpu().numpy() if hasattr(matrix, "detach") else np.asarray(matrix)
            value = np.asarray(value, dtype=np.float32).reshape(-1, 3)
            forces.append(value.sum(axis=0) if value.size else np.zeros(3, dtype=np.float32))
        result = np.asarray(forces, dtype=np.float32)
        if result.shape != (len(self.scoring_contact_body_paths), 3) or not np.isfinite(result).all():
            raise RuntimeError(f"invalid V4 contact force matrix: {result.shape}")
        return result

    def _actual_gripper_closed_fraction(self) -> float:
        joints = np.asarray(self.robot.get_joint_positions(), dtype=np.float64)
        gap = joints[self.index["joint7"]] - joints[self.index["joint8"]]
        return float(
            np.clip(1.0 - gap / (2.0 * self.gripper_max_opening), 0.0, 1.0)
        )

    def _update_attack_grasp_evidence(self) -> None:
        """Latch closed-gripper PhysX contact followed by a measured lift.

        Contact forces are transient at 200 Hz and can disappear before the
        next 10 Hz attack-state snapshot.  The latch is deliberately limited
        to Piper's terminal gripper bodies and expires after two seconds.  A
        contact alone is insufficient: the task object must subsequently rise
        by the same 15 mm required by the existing geometric detector.
        """

        if (
            self._attack_grasp_initial_height is None
            or self._attack_grasp_confirmed_evidence is not None
        ):
            return
        gripper_closed = self._actual_gripper_closed_fraction()
        if gripper_closed < 0.55:
            self._attack_grasp_contact_latch = None
            return

        contact_body_paths: list[str] = []
        contact_force_norm_n = 0.0
        for index in self.attack_grasp_contact_indices:
            matrix = self.scoring_contact_views[index].get_contact_force_matrix(
                dt=PHYSICS_DT
            )
            value = (
                matrix.detach().cpu().numpy()
                if hasattr(matrix, "detach")
                else np.asarray(matrix)
            )
            value = np.asarray(value, dtype=np.float32).reshape(-1, 3)
            force_w = (
                value.sum(axis=0)
                if value.size
                else np.zeros(3, dtype=np.float32)
            )
            force_norm_n = float(np.linalg.norm(force_w))
            if not math.isfinite(force_norm_n):
                raise RuntimeError("non-finite Piper gripper contact force")
            if force_norm_n > 0.01:
                contact_body_paths.append(self.scoring_contact_body_paths[index])
                contact_force_norm_n += force_norm_n

        if contact_body_paths:
            self._attack_grasp_contact_latch = {
                "contact_physics_step": int(self.physics_step),
                "contact_time_s": float(self.physics_step * PHYSICS_DT),
                "contact_body_paths": contact_body_paths,
                "contact_force_norm_n": float(contact_force_norm_n),
                "gripper_closed": gripper_closed,
            }

        evidence = self._attack_grasp_contact_latch
        if evidence is None:
            return
        elapsed_steps = int(self.physics_step) - int(evidence["contact_physics_step"])
        if elapsed_steps > int(round(2.0 / PHYSICS_DT)):
            self._attack_grasp_contact_latch = None
            return
        object_position = np.asarray(self.cube.get_world_pose()[0], dtype=np.float64)
        lift = float(object_position[2] - self._attack_grasp_initial_height)
        if lift < 0.015:
            return
        self._attack_grasp_confirmed_evidence = {
            **evidence,
            "confirmation_physics_step": int(self.physics_step),
            "confirmation_time_s": float(self.physics_step * PHYSICS_DT),
            "contact_to_confirmation_s": float(elapsed_steps * PHYSICS_DT),
            "object_lift_m": lift,
            "object_xyz": object_position.tolist(),
            "source": "physx_closed_gripper_contact_then_lift",
        }
        print(
            "[attack-grasp] confirmed from clean PhysX evidence: "
            f"{self._attack_grasp_confirmed_evidence}",
            flush=True,
        )

    def _force_torque_sample(
        self,
        gripper_position: np.ndarray,
        gripper_quaternion: np.ndarray,
    ) -> dict[str, Any]:
        """Build one measured Piper TCP wrench from the two fingertip contacts.

        PhysX exposes the net task-object contact force for each rigid finger,
        but not each contact point through this Runner view.  Force is exact at
        the selected contact boundary; torque is therefore the explicit moment
        of each net force about the TCP using the corresponding finger-body
        origin.  The approximation is retained in evidence metadata.
        """

        if not self.force_torque_sensor_enabled:
            raise RuntimeError("force/torque sampling was not enabled by the task package")
        contact_forces_w = self._contact_net_forces_w()
        finger_forces_w = contact_forces_w[self.force_torque_contact_indices]
        tcp_world = (
            np.asarray(gripper_position, dtype=np.float64)
            + quaternion_to_rotation_matrix(gripper_quaternion) @ TCP_OFFSET
        )
        torque_w = np.zeros(3, dtype=np.float64)
        for contact_index, force_w in zip(
            self.force_torque_contact_indices, finger_forces_w
        ):
            runtime_body_index = int(self.scoring_body_indices[contact_index])
            finger_position, _finger_quaternion = self.link_transform(runtime_body_index)
            torque_w += np.cross(
                np.asarray(finger_position, dtype=np.float64) - tcp_world,
                np.asarray(force_w, dtype=np.float64),
            )
        sensor_from_world = quaternion_to_rotation_matrix(gripper_quaternion).T
        force_sensor = sensor_from_world @ np.sum(finger_forces_w, axis=0)
        torque_sensor = sensor_from_world @ torque_w
        grip_force_n = float(np.linalg.norm(force_sensor))
        if not (
            np.isfinite(force_sensor).all()
            and np.isfinite(torque_sensor).all()
            and math.isfinite(grip_force_n)
        ):
            raise RuntimeError("non-finite Piper TCP force/torque sample")
        return {
            "sensor_id": self.force_torque_sensor_id,
            "frame_id": self.force_torque_frame_id,
            "force_xyz_n": force_sensor.astype(np.float32),
            "torque_xyz_nm": torque_sensor.astype(np.float32),
            "grip_force_n": grip_force_n,
            "source": "physx_task_object_fingertip_contact",
            "torque_estimation": "fingertip_body_origin_moment_about_tcp",
            "finger_body_paths": [
                self.scoring_contact_body_paths[index]
                for index in self.force_torque_contact_indices
            ],
        }

    def _task_target_contact_force_w(self) -> np.ndarray:
        if not self.success_contact_views:
            return np.zeros(3, dtype=np.float32)
        result = np.zeros(3, dtype=np.float32)
        for view in self.success_contact_views:
            matrix = view.get_contact_force_matrix(dt=PHYSICS_DT)
            value = matrix.detach().cpu().numpy() if hasattr(matrix, "detach") else np.asarray(matrix)
            value = np.asarray(value, dtype=np.float32).reshape(-1, 3)
            if value.size:
                result += value.sum(axis=0)
        if result.shape != (3,) or not np.isfinite(result).all():
            raise RuntimeError(f"invalid task-target contact force: {result}")
        return result

    def start_scoring_capture(self) -> None:
        if self.scoring_active:
            raise RuntimeError("V4 scoring capture already started")
        self.task_start_physics_step = int(self.physics_step)
        self.scoring_initial = self._scoring_snapshot()
        self.scoring_initial["task_reference_pose"] = self._pose(
            self.goal_position, np.asarray([1.0, 0.0, 0.0, 0.0])
        )
        self.fixed_evidence_sidecar.record_initial(
            self.fixed_evidence_runtime.initial_sample(),
            usd_boolean_values=self._sample_usd_boolean_attributes(),
        )
        if self.rubric_evidence_runtime is not None:
            self.rubric_evidence_writer.initial(self.rubric_evidence_runtime.initial_sample())
        if self.safety_monitor is not None:
            base_position = np.asarray(self.robot.get_world_pose()[0], dtype=np.float64)
            if base_position.shape != (3,) or not np.isfinite(base_position).all():
                raise RuntimeError(
                    f"cannot start safety monitor from invalid base pose: {base_position}"
                )
            initial_safety_sample = self._safety_sample()
            self.safety_monitor.start(
                physics_step=self.physics_step,
                sim_time_s=self.physics_step * PHYSICS_DT,
                initial_base_height_m=float(base_position[2]),
                initial_sample=initial_safety_sample,
            )
        self.scoring_active = True

    def safety_runtime_manifest(self) -> dict[str, Any]:
        if self.safety_monitor is None:
            return {
                "schema": "m20-safety-runtime-plan/v1",
                "enabled": False,
                "reason": "task config does not embed a safety profile",
            }
        manifest = self.safety_monitor.manifest()
        manifest.update({
            "activation_boundary": "start_scoring_capture",
            "activation_meaning": "official_task_time_zero_after_settle",
            "sampling_hz": PHYSICS_HZ,
            "contact_tracking": bool(self.safety_contact_tracker is not None),
            "contact_rules": sorted(
                set(self.safety_enabled_rules)
                & {"S01", "S02", "S04", "S06", "S07", "S14"}
            ),
            "geometry": (
                self.safety_geometry_tracker.manifest()
                if self.safety_geometry_tracker is not None
                else {"enabled": False}
            ),
            "state_prim_paths": list(self.safety_state_prim_paths),
            "prismatic_joint_names": list(self.safety_prismatic_joint_names),
            "runtime_fall_stop_is_separate": True,
        })
        return manifest

    @staticmethod
    def _safety_path_under(path: str, prefix: str) -> bool:
        prefix = str(prefix).rstrip("/")
        return path == prefix or path.startswith(prefix + "/")

    def _safety_fixed_object_states(self) -> dict[str, dict[str, Any]]:
        """Read task-scoped object truth from the frozen PhysX evidence axes."""

        if not self.safety_state_prim_paths and self.safety_geometry_tracker is None:
            return {}
        articulation = self.fixed_evidence_runtime._articulation_values()
        rigid = self.fixed_evidence_runtime._rigid_values()
        body_pose = np.asarray(articulation["articulation_body_pose"], dtype=np.float64)
        body_velocity = np.asarray(
            articulation["articulation_body_velocity"], dtype=np.float64
        )
        states: dict[str, dict[str, Any]] = {}
        surface_articulations = self.fixed_evidence_manifest["surface"]["articulations"]
        for plan, surface in zip(
            self.fixed_evidence_manifest["articulations"],
            surface_articulations,
            strict=True,
        ):
            paths = list(surface["body_paths"])
            start, stop = int(plan["body_start"]), int(plan["body_stop"])
            if stop - start != len(paths):
                raise RuntimeError("fixed safety body axis does not match its manifest")
            for path, pose, velocity in zip(
                paths,
                body_pose[start:stop],
                body_velocity[start:stop],
                strict=True,
            ):
                states[str(path)] = {
                    "pose": pose.astype(float).tolist(),
                    "origin_velocity": velocity.astype(float).tolist(),
                    "source": "fixed_evidence_physx_articulation_body",
                }
        rigid_paths = list(self.fixed_evidence_manifest["rigid_objects"])
        rigid_pose = np.asarray(rigid["rigid_root_pose"], dtype=np.float64)
        rigid_velocity = np.asarray(rigid["rigid_root_velocity"], dtype=np.float64)
        if len(rigid_paths) != len(rigid_pose) or len(rigid_paths) != len(rigid_velocity):
            raise RuntimeError("fixed safety rigid-object axis does not match its manifest")
        for path, pose, velocity in zip(
            rigid_paths, rigid_pose, rigid_velocity, strict=True
        ):
            states[str(path)] = {
                "pose": pose.astype(float).tolist(),
                "origin_velocity": velocity.astype(float).tolist(),
                "source": "fixed_evidence_physx_rigid_body",
            }

        # Bind a configured descendant Prim to its unique frozen rigid owner.
        # Static state Prims are read from the final composed Stage and receive
        # an explicit zero velocity; ambiguous dynamic descendants stay absent
        # so the monitor reports unknown.
        xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        for configured_path in self.safety_state_prim_paths:
            if configured_path in states:
                continue
            ancestor_candidates = [
                path
                for path in states
                if self._safety_path_under(configured_path, path)
            ]
            if ancestor_candidates:
                owner = max(ancestor_candidates, key=len)
                states[configured_path] = {
                    **states[owner],
                    "source": "configured_prim_bound_to_frozen_rigid_owner",
                    "rigid_owner": owner,
                }
                continue
            descendant_candidates = [
                path
                for path in states
                if self._safety_path_under(path, configured_path)
            ]
            if len(descendant_candidates) == 1:
                owner = descendant_candidates[0]
                states[configured_path] = {
                    **states[owner],
                    "source": "configured_prim_bound_to_unique_rigid_descendant",
                    "rigid_owner": owner,
                }
                continue
            prim = self.final_stage.GetPrimAtPath(configured_path)
            if not prim or not prim.IsValid():
                continue
            xformable = UsdGeom.Xformable(prim)
            if not xformable:
                continue
            matrix = xform_cache.GetLocalToWorldTransform(prim)
            translation = np.asarray(matrix.ExtractTranslation(), dtype=np.float64)
            quaternion = matrix.ExtractRotationQuat()
            imaginary = quaternion.GetImaginary()
            pose = np.asarray(
                [
                    *translation,
                    float(quaternion.GetReal()),
                    float(imaginary[0]),
                    float(imaginary[1]),
                    float(imaginary[2]),
                ],
                dtype=np.float64,
            )
            if np.isfinite(pose).all():
                states[configured_path] = {
                    "pose": pose.tolist(),
                    "origin_velocity": np.zeros(6, dtype=np.float64).tolist(),
                    "source": "final_stage_static_transform",
                }
        inventory=getattr(self,'paper_inventory_writer',None)
        if inventory is not None:
            _,_,measured=inventory.references.sample(np.asarray(inventory.view.get_transforms()),self.physics_step*PHYSICS_DT)
            for path in inventory.references.paths:
                if path in measured:states[path]={**states.get(path,{}),**measured[path]}
                else:states.pop(path,None)
        return states

    def _safety_state_attributes(self) -> dict[str, dict[str, bool]]:
        if "S05" not in self.safety_enabled_rules:
            return {}
        result: dict[str, dict[str, bool]] = {}
        for check in self.safety_profile["rules"]["S05"]["parameters"]["checks"]:
            if check["criterion"] != "usd_boolean_attribute":
                continue
            prim_path = str(check["prim_path"])
            attribute_name = str(check["attribute_name"])
            prim = self.final_stage.GetPrimAtPath(prim_path)
            attribute = prim.GetAttribute(attribute_name) if prim and prim.IsValid() else None
            value = attribute.Get() if attribute and attribute.IsValid() else None
            if isinstance(value, (bool, np.bool_)):
                result.setdefault(prim_path, {})[attribute_name] = bool(value)
        return result

    def _safety_sample(self) -> dict[str, Any]:
        base_position, base_orientation = self.robot.get_world_pose()
        transforms = np.asarray(
            self.robot._articulation_view._physics_view.get_link_transforms(),
            dtype=np.float64,
        )[0, self.scoring_body_indices]
        link_velocities = np.asarray(
            self.robot._articulation_view._physics_view.get_link_velocities(),
            dtype=np.float64,
        )[0, self.scoring_body_indices]
        local_coms = np.asarray(
            self.robot._articulation_view._physics_view.get_coms(),
            dtype=np.float64,
        )[0, self.scoring_body_indices, :3]
        body_states = {}
        for path, transform, velocity, local_com in zip(
            self.scoring_body_paths,
            transforms,
            link_velocities,
            local_coms,
            strict=True,
        ):
            quaternion = np.asarray(
                [transform[6], transform[3], transform[4], transform[5]],
                dtype=np.float64,
            )
            quaternion /= np.linalg.norm(quaternion)
            com_offset_world = quaternion_to_rotation_matrix(quaternion) @ local_com
            origin_linear_velocity = velocity[:3] - np.cross(
                velocity[3:], com_offset_world
            )
            body_states[Path(path).name] = {
                "path": path,
                "pose": np.concatenate((transform[:3], quaternion)).tolist(),
                "origin_velocity": np.concatenate(
                    (origin_linear_velocity, velocity[3:])
                ).tolist(),
                "com_world": (transform[:3] + com_offset_world).tolist(),
            }
        object_states = self._safety_fixed_object_states()
        object_states.update({
            value["path"]: {
                "pose": list(value["pose"]),
                "origin_velocity": list(value["origin_velocity"]),
                "source": "direct_robot_physx_link_state",
            }
            for value in body_states.values()
        })
        geometry_metrics = {}
        if self.safety_geometry_tracker is not None:
            geometry_states=dict(object_states)
            inventory=getattr(self,'paper_inventory_writer',None)
            if inventory is not None:
                poses=np.asarray(inventory.view.get_transforms())
                for path,pose in zip(inventory.paths,poses,strict=True):
                    geometry_states[path]={'pose':[float(v) for v in (*pose[:3],pose[6],*pose[3:6])],
                        'source':'actual_physx_inventory_rigid_pose'}
            lifecycle=getattr(self,'paper_lifecycle',None)
            active=None
            if lifecycle is not None:
                backend=getattr(self,'paper_backend',None)
                phase={'lifecycle_phase':lifecycle.phase,'mission_phase':backend.phase if backend else None,
                       'docking_completed':getattr(backend,'_paper_docking_completed',None)}
                active={rid for rid in ('S03','S08') if rid in self.safety_enabled_rules and self.safety_monitor._applicable(rid,phase) is not False}
            geometry_metrics=self.safety_geometry_tracker.sample(geometry_states,active_rules=active)
        contacts = (
            self.safety_contact_tracker.snapshot()
            if self.safety_contact_tracker is not None
            else None
        )
        adjacent_pairs = (
            self.safety_contact_tracker.adjacent_body_pairs
            if self.safety_contact_tracker is not None
            else None
        )
        measured_efforts = self.robot.get_measured_joint_efforts()
        joint_efforts = None
        if measured_efforts is not None:
            measured_efforts_array = np.asarray(
                measured_efforts, dtype=np.float64
            ).reshape(-1)
            if measured_efforts_array.size == len(self.names):
                joint_efforts = measured_efforts_array.tolist()
        return {
            "robot_root_prim": self.scoring_articulation_prim,
            "base_xyz": np.asarray(base_position, dtype=np.float64).tolist(),
            "base_rpy": quaternion_to_rpy(
                np.asarray(base_orientation, dtype=np.float64)
            ).tolist(),
            "base_linear_velocity": np.asarray(
                self.robot.get_linear_velocity(), dtype=np.float64
            ).tolist(),
            "base_angular_velocity": np.asarray(
                self.robot.get_angular_velocity(), dtype=np.float64
            ).tolist(),
            "body_states": body_states,
            "object_states": object_states,
            "state_attributes": self._safety_state_attributes(),
            "geometry_metrics": geometry_metrics,
            "joint_names": list(self.names),
            "joint_positions": np.asarray(
                self.robot.get_joint_positions(), dtype=np.float64
            ).tolist(),
            "joint_velocities": np.asarray(
                self.robot.get_joint_velocities(), dtype=np.float64
            ).tolist(),
            "joint_efforts": joint_efforts,
            "joint_lower_limits": np.asarray(
                self.articulation["lower"], dtype=np.float64
            ).tolist(),
            "joint_upper_limits": np.asarray(
                self.articulation["upper"], dtype=np.float64
            ).tolist(),
            "joint_velocity_limits": self.safety_joint_velocity_limits.tolist(),
            "joint_effort_limits": self.safety_joint_effort_limits.tolist(),
            "prismatic_joint_names": list(self.safety_prismatic_joint_names),
            "contacts": contacts,
            "adjacent_body_pairs": adjacent_pairs,
        }

    def _observe_safety(self, *, physics_step: int, sim_time_s: float) -> None:
        if self.safety_monitor is None or not self.safety_monitor.active:
            return
        sample=self._safety_sample()
        observer=getattr(self,'paper_discrete_observer',None)
        if observer is not None:sample.update(observer.sample(physics_step))
        lifecycle=getattr(self,'paper_lifecycle',None)
        backend=getattr(self,'paper_backend',None)
        if lifecycle is not None:
            sample.update(lifecycle_phase=lifecycle.phase,
                mission_phase=backend.phase if backend is not None else None,
                docking_completed=getattr(backend,'_paper_docking_completed',None))
        geometry_writer=getattr(self,'paper_geometry_writer',None)
        if geometry_writer is not None:geometry_writer.observe(physics_step,sim_time_s,sample)
        self.safety_monitor.observe(
            physics_step=physics_step,
            sim_time_s=sim_time_s,
            sample=sample,
        )

    def _sample_usd_boolean_attributes(self) -> np.ndarray:
        values: list[bool] = []
        for spec in self.usd_boolean_attributes:
            prim = self.final_stage.GetPrimAtPath(spec.prim_path)
            if not prim or not prim.IsValid():
                raise RuntimeError(
                    f"discovered USD boolean Prim disappeared: {spec.prim_path}"
                )
            attribute = prim.GetAttribute(spec.attribute_name)
            if not attribute or not attribute.IsValid() or str(attribute.GetTypeName()) != "bool":
                raise RuntimeError(
                    "discovered USD boolean attribute changed after surface freeze: "
                    f"{spec.prim_path}.{spec.attribute_name}"
                )
            value = attribute.Get()
            if not isinstance(value, (bool, np.bool_)):
                raise RuntimeError(
                    "USD boolean attribute has no concrete bool value: "
                    f"{spec.prim_path}.{spec.attribute_name}"
                )
            values.append(bool(value))
        return np.asarray(values, dtype=np.uint8)

    def capture_scoring_step(self) -> None:
        if not self.scoring_active:
            return
        snapshot = self._scoring_snapshot()
        state_time_s = float(self.physics_step * PHYSICS_DT)
        snapshot.update({
            "action": self.latest_unified_action.copy().astype(np.float32),
            "action_time_s": state_time_s - PHYSICS_DT,
            "state_time_s": state_time_s,
        })
        self.fixed_evidence_sidecar.record_step(
            step_index=len(self.scoring_samples),
            sim_time_s=state_time_s,
            sample=self.fixed_evidence_runtime.control_sample(),
            usd_boolean_values=self._sample_usd_boolean_attributes(),
        )
        if self.rubric_evidence_runtime is not None:
            rows, contacts = self.rubric_evidence_runtime.sample()
            self.rubric_evidence_writer.step(len(self.scoring_samples), state_time_s, rows, contacts)
        self.scoring_samples.append(snapshot)

    def state(self) -> np.ndarray:
        position, orientation = self.robot.get_world_pose()
        position = np.asarray(position, dtype=np.float64)
        orientation = np.asarray(orientation, dtype=np.float64)
        rpy = quaternion_to_rpy(orientation)
        linear_world = np.asarray(self.robot.get_linear_velocity(), dtype=np.float64)
        angular_world = np.asarray(self.robot.get_angular_velocity(), dtype=np.float64)
        rotation = quaternion_to_rotation_matrix(orientation)
        linear_base = rotation.T @ linear_world
        angular_base = rotation.T @ angular_world
        joints = np.asarray(self.robot.get_joint_positions(), dtype=np.float64)
        arm_position, arm_quaternion = self.link_transform(self.arm_body)
        gripper_position, gripper_quaternion = self.link_transform(self.gripper_body)
        tcp_world = gripper_position + quaternion_to_rotation_matrix(gripper_quaternion) @ TCP_OFFSET
        arm_rotation = quaternion_to_rotation_matrix(arm_quaternion)
        tcp_base = arm_rotation.T @ (tcp_world - arm_position)
        arm_inverse = np.asarray([arm_quaternion[0], -arm_quaternion[1], -arm_quaternion[2], -arm_quaternion[3]])
        relative_quaternion = quaternion_multiply(arm_inverse, gripper_quaternion)
        relative_rpy = quaternion_to_rpy(relative_quaternion / np.linalg.norm(relative_quaternion))
        gap = joints[self.index["joint7"]] - joints[self.index["joint8"]]
        gripper_closed = float(
            np.clip(1.0 - gap / (2.0 * self.gripper_max_opening), 0.0, 1.0)
        )
        result = np.concatenate((
            position, rpy, linear_base, angular_base,
            joints[[self.index[f"joint{i}"] for i in range(1, 7)]],
            tcp_base, relative_rpy, np.asarray([gripper_closed]),
        )).astype(np.float32)
        if result.shape != (25,) or not np.isfinite(result).all():
            raise RuntimeError(f"invalid unified state: {result}")
        return result

    def policy_state(self) -> np.ndarray:
        """Expose the trained local map frame while retaining world-frame physics."""

        result = self.state()
        result[:2] -= self.policy_xy_origin
        self._pending_q17_policy_bridge = None
        if (
            self.q17_force_feedback_bridge_enabled
            and self._q17_force_feedback_attack_ids
        ):
            clean_state_value = float(result[24])
            forwarded_state_value = float(
                1.0 if self._q17_force_feedback_positive else 0.0
            )
            result[24] = np.float32(forwarded_state_value)
            self._pending_q17_policy_bridge = {
                "schema": "robolab-q17-force-feedback-policy-bridge/v1",
                "attack_ids": list(self._q17_force_feedback_attack_ids),
                "state_index": 24,
                "clean_state_value": clean_state_value,
                "forwarded_state_value": forwarded_state_value,
                "input_changed": not math.isclose(
                    clean_state_value,
                    forwarded_state_value,
                    rel_tol=0.0,
                    abs_tol=1e-7,
                ),
                "sensor_payload_changed": bool(
                    self._q17_force_feedback_payload_changed
                ),
                "clean_grip_force_n": float(self._q17_clean_grip_force_n),
                "attacked_grip_force_n": float(
                    self._q17_attacked_grip_force_n
                ),
                "judge_uses_attacked_feedback": False,
            }
        if self.attack_boundary is not None:
            result = np.asarray(
                self.attack_boundary.intercept_state_feedback(
                    result,
                    physics_step=self.physics_step,
                    time_s=self.physics_step * PHYSICS_DT,
                ),
                dtype=np.float32,
            )
        if result.shape != (25,) or not np.isfinite(result).all():
            raise RuntimeError(f"invalid intercepted unified state: {result}")
        return result

    def sample_imu(self, *, intercept: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Read the physical base IMU and optionally cross the attack boundary."""

        _, orientation = self.robot.get_world_pose()
        orientation = np.asarray(orientation, dtype=np.float64)
        rotation = quaternion_to_rotation_matrix(orientation)
        angular_velocity = rotation.T @ np.asarray(
            self.robot.get_angular_velocity(), dtype=np.float64
        )
        gravity = projected_gravity(orientation)
        imu_frame = self.imu_sensor.get_current_frame(read_gravity=True)
        linear_acceleration = np.asarray(imu_frame["lin_acc"], dtype=np.float64).reshape(-1)
        values = (angular_velocity, linear_acceleration, gravity)
        if any(value.shape != (3,) or not np.isfinite(value).all() for value in values):
            raise RuntimeError(f"invalid M20 IMU sample: {[value.shape for value in values]}")
        if intercept and self.attack_boundary is not None:
            attacked = self.attack_boundary.intercept_imu(
                {
                    "angular_velocity": angular_velocity,
                    "linear_acceleration": linear_acceleration,
                    "projected_gravity": gravity,
                },
                physics_step=self.physics_step,
                time_s=self.physics_step * PHYSICS_DT,
            )
            angular_velocity = np.asarray(attacked["angular_velocity"], dtype=np.float64)
            linear_acceleration = np.asarray(attacked["linear_acceleration"], dtype=np.float64)
            gravity = np.asarray(attacked["projected_gravity"], dtype=np.float64)
        return tuple(value.astype(np.float32) for value in (
            angular_velocity, linear_acceleration, gravity
        ))

    def update_cameras(self) -> None:
        if not self.cameras:
            return
        for camera in self.cameras:
            camera.set_clipping_range(0.01, 1000.0)
        base_position, base_orientation = self.robot.get_world_pose()
        base_position = np.asarray(base_position, dtype=np.float64)
        base_orientation = np.asarray(base_orientation, dtype=np.float64)
        self.base_visual_follower.set_world_pose(base_position, base_orientation)
        base_rotation = quaternion_to_rotation_matrix(base_orientation)
        front_eye = base_position + base_rotation @ np.asarray([0.22, 0.0, 0.44])
        front_target = base_position + base_rotation @ np.asarray([2.0, 0.0, 0.34])
        self.cameras[0].set_world_pose(position=front_eye, orientation=look_at(front_eye, front_target))
        rear_eye = base_position + base_rotation @ np.asarray([-0.22, 0.0, 0.44])
        rear_target = base_position + base_rotation @ np.asarray([-2.0, 0.0, 0.34])
        self.cameras[1].set_world_pose(position=rear_eye, orientation=look_at(rear_eye, rear_target))
        arm_position, arm_quaternion = self.link_transform(self.arm_body)
        arm_rotation = quaternion_to_rotation_matrix(arm_quaternion)
        # Close eye-on-base view of the complete Piper workspace.  A camera at
        # gripper_base was occluded by the fingers for this asset geometry.
        wrist_eye = arm_position + arm_rotation @ np.asarray([0.10, -0.38, 0.40])
        wrist_target = arm_position + arm_rotation @ np.asarray([0.45, 0.0, 0.05])
        self.cameras[2].set_world_pose(position=wrist_eye, orientation=look_at(wrist_eye, wrist_target))
        if args.record_video:
            yaw = yaw_from_quaternion(base_orientation)
            forward = np.asarray([math.cos(yaw), math.sin(yaw), 0.0], dtype=np.float64)
            recording_eye = base_position - 2.0 * forward
            recording_eye[2] += 2.0
            recording_target = base_position + np.asarray([0.35 * forward[0], 0.35 * forward[1], 0.25])
            self.cameras[3].set_world_pose(
                position=recording_eye,
                orientation=look_at(recording_eye, recording_target),
            )

    @staticmethod
    def rgb(camera) -> np.ndarray:
        value = np.asarray(camera.get_rgb(device="cpu"))
        if value.ndim != 3 or value.shape[2] < 3:
            raise RuntimeError(f"camera returned {value.shape}")
        value = value[..., :3]
        if value.dtype != np.uint8:
            value = np.clip(value * 255.0 if value.max() <= 1.0 else value, 0, 255).astype(np.uint8)
        if value.shape[:2] != (224, 224):
            value = np.asarray(Image.fromarray(value).resize((224, 224), Image.Resampling.BILINEAR))
        return value

    @staticmethod
    def recording_rgb(camera) -> np.ndarray:
        value = np.asarray(camera.get_rgb(device="cpu"))
        if value.ndim != 3 or value.shape[2] < 3:
            raise RuntimeError(f"recording camera returned {value.shape}")
        value = value[..., :3]
        if value.dtype != np.uint8:
            value = np.clip(value * 255.0 if value.max() <= 1.0 else value, 0, 255).astype(np.uint8)
        if value.shape[:2] != (720, 1280):
            raise RuntimeError(f"recording camera returned {value.shape}, expected (720, 1280, 3)")
        return value

    def capture(self, phase: str) -> None:
        (args.output / "physics_clock.json").write_text(json.dumps({**self._clock_audit, "declared_steps": self.physics_step + 1, "declared_seconds": (self.physics_step + 1) * PHYSICS_DT, "phase": phase}, indent=2) + "\n")
        if phase == "preposition":
            return
        if self.cameras:
            self.m20_camera_0_images.append(self.rgb(self.cameras[0]))
            self.m20_camera_1_images.append(self.rgb(self.cameras[1]))
            self.piper_wrist_images.append(self.rgb(self.cameras[2]))
        imu_angular_velocity, imu_linear_acceleration, imu_projected_gravity = self.sample_imu(
            intercept=bool(args.kinematic_base)
        )
        self.imu_angular_velocities.append(imu_angular_velocity)
        self.imu_linear_accelerations.append(imu_linear_acceleration)
        self.imu_projected_gravities.append(imu_projected_gravity)
        self.states.append(self.state())
        self.actions.append(self.latest_unified_action.copy())
        self.phases.append(phase)
        self.times.append(self.physics_step * PHYSICS_DT)
        object_position, _ = self.cube.get_world_pose()
        self.object_heights.append(float(object_position[2]))
        self.capture_scoring_step()
        if not self.cameras:
            return
        if self.lidar_stream is not None:
            clean_lidar_frame = self.lidar_stream.frame(self.physics_step * PHYSICS_DT)
            self.submit_lidar_frame(clean_lidar_frame)
        if args.record_video:
            if args.four_view_video:
                attacked_points = np.asarray(self._latest_lidar_frame["points"], dtype=np.float32)
                frame = self.four_view.compose(
                    self.cameras[3],
                    self.cameras[0],
                    self.cameras[2],
                    self.lidar_stream.render_panel(attacked_points),
                )
            else:
                frame = Image.fromarray(self.recording_rgb(self.cameras[3]))
            frame.save(
                self.video_frames / f"frame_{self.video_frame_count:06d}.jpg",
                quality=90,
            )
            self.video_frame_count += 1

    def apply_kinematic_targets(
        self,
        arm_targets: np.ndarray | None = None,
    ) -> None:
        """Hold M20 joints safely while Piper follows the accepted 10-D action.

        This backend is deliberately explicit and is enabled only by
        ``--kinematic-base``.  The request/response attack boundary remains in
        the normal Runner path; only the unstable learned locomotion adapter is
        replaced for internal integration testing.
        """

        if not self.piper_motion_enabled:
            arm_targets = self.piper_hold_targets
        positions = {
            name: float(DEFAULT_POSITION_BY_NAME[name]) for name in LEG_JOINTS
        }
        if self.squat_enabled and self.squat_command < 0:
            for name in LEG_JOINTS:
                if "hipy" in name:
                    sign = 1.0 if DEFAULT_POSITION_BY_NAME[name] >= 0.0 else -1.0
                    positions[name] += sign * self.squat_joint_offset_rad
                elif "knee" in name:
                    sign = 1.0 if DEFAULT_POSITION_BY_NAME[name] >= 0.0 else -1.0
                    positions[name] += sign * 2.0 * self.squat_joint_offset_rad
        wheel_velocities = {name: 0.0 for name in WHEEL_JOINTS}
        if self.attack_boundary is not None:
            intercepted_targets = self.attack_boundary.intercept_joint_control(
                {
                    "positions": dict(positions),
                    "wheel_velocities": dict(wheel_velocities),
                },
                physics_step=self.physics_step,
                time_s=self.physics_step * PHYSICS_DT,
            )
            positions = dict(intercepted_targets["positions"])
            wheel_velocities = dict(intercepted_targets["wheel_velocities"])

        q_target = np.full(24, np.nan, dtype=np.float32)
        qd_target = np.full(24, np.nan, dtype=np.float32)
        for name, value in positions.items():
            dof = self.index[name]
            lower = float(self.articulation["lower"][dof])
            upper = float(self.articulation["upper"][dof])
            if name in LEG_JOINTS and upper - lower > 2.0 * M20_REVOLUTE_TARGET_LIMIT_MARGIN_RAD:
                lower += M20_REVOLUTE_TARGET_LIMIT_MARGIN_RAD
                upper -= M20_REVOLUTE_TARGET_LIMIT_MARGIN_RAD
            q_target[dof] = np.clip(value, lower, upper)
        for name, value in wheel_velocities.items():
            qd_target[self.index[name]] = np.clip(
                value,
                -VELOCITY_LIMIT_BY_NAME[name],
                VELOCITY_LIMIT_BY_NAME[name],
            )
        if arm_targets is None:
            for name in PIPER_JOINTS:
                q_target[self.index[name]] = (
                    0.025 if name == "joint7" else (-0.025 if name == "joint8" else 0.0)
                )
        else:
            effective_arm_targets = np.empty(7, dtype=np.float32)
            for index in range(6):
                name = f"joint{index + 1}"
                dof = self.index[name]
                q_target[dof] = np.clip(
                    arm_targets[index],
                    self.articulation["lower"][dof],
                    self.articulation["upper"][dof],
                )
                effective_arm_targets[index] = q_target[dof]
            opening = float(np.clip(arm_targets[6], 0.0, 0.05))
            q_target[self.index["joint7"]] = np.clip(
                opening, 0.0, self.articulation["upper"][self.index["joint7"]]
            )
            q_target[self.index["joint8"]] = np.clip(
                -opening, self.articulation["lower"][self.index["joint8"]], 0.0
            )
            effective_arm_targets[6] = q_target[self.index["joint7"]]
            self.piper_command_targets = effective_arm_targets
        if self.physics_step < self._gripper_release_until_step:
            opening = self._gripper_release_opening
            q_target[self.index["joint7"]] = np.clip(
                opening, 0.0, self.articulation["upper"][self.index["joint7"]]
            )
            q_target[self.index["joint8"]] = np.clip(
                -opening, self.articulation["lower"][self.index["joint8"]], 0.0
            )
        self.kinematic_position_indices = np.flatnonzero(
            np.isfinite(q_target)
        ).astype(np.int32)
        q_target, qd_target = self._assurance_joints(q_target, qd_target)
        self.kinematic_position_targets = q_target[
            self.kinematic_position_indices
        ].copy()
        self.controller.apply_action(
            ArticulationAction(
                joint_positions=q_target,
                joint_velocities=qd_target,
            )
        )
        self.previous_m20_action[:] = 0.0

    def stationary_stance_positions(
        self,
        command: np.ndarray,
        measured_joints: np.ndarray,
    ) -> np.ndarray | None:
        """Return a smooth nominal leg stance while the high-level command is STOP."""

        if not is_zero_base_command(command):
            self.stationary_leg_entry_targets = None
            self.stationary_control_steps = 0
            return None
        leg_dofs = np.asarray(
            [self.index[name] for name in LEG_JOINTS],
            dtype=np.int32,
        )
        if self.stationary_leg_entry_targets is None:
            self.stationary_leg_entry_targets = np.asarray(
                measured_joints[leg_dofs], dtype=np.float32
            ).copy()
            self.stationary_control_steps = 0
            self.stationary_activation_count += 1
            self.vla_trace.append({
                "event": "stationary_stance_latched",
                "activation": self.stationary_activation_count,
                "physics_step": self.physics_step,
                "phase": self.current_control_phase,
                "entry_leg_targets_rad": self.stationary_leg_entry_targets.tolist(),
            })
        self.stationary_control_steps += 1
        blend_steps = max(1, round(STATIONARY_STANCE_BLEND_SECONDS * POLICY_HZ))
        progress = min(1.0, self.stationary_control_steps / blend_steps)
        nominal = np.asarray(
            [DEFAULT_POSITION_BY_NAME[name] for name in LEG_JOINTS],
            dtype=np.float32,
        )
        return smooth_stationary_leg_targets(
            self.stationary_leg_entry_targets,
            nominal,
            progress,
        )

    def set_stationary_drive_mode(self, enabled: bool) -> None:
        """Switch leg impedance only when crossing the STOP/MOVE boundary."""

        enabled = bool(enabled)
        if args.kinematic_base or enabled == self.stationary_drive_mode:
            return
        self.controller.set_gains(
            kps=self.stationary_kps if enabled else self.locomotion_kps,
            kds=self.stationary_kds if enabled else self.locomotion_kds,
            save_to_usd=False,
        )
        self.stationary_drive_mode = enabled
        self.vla_trace.append({
            "event": "stationary_drive_mode_changed",
            "enabled": enabled,
            "physics_step": self.physics_step,
            "phase": self.current_control_phase,
        })

    def measured_piper_targets(self) -> np.ndarray:
        """Read six arm joints and one symmetric gripper-opening target."""

        joints = np.asarray(self.robot.get_joint_positions(), dtype=np.float64)
        targets = np.empty(7, dtype=np.float32)
        for index in range(6):
            targets[index] = joints[self.index[f"joint{index + 1}"]]
        targets[6] = np.clip(
            0.5
            * (
                joints[self.index["joint7"]]
                - joints[self.index["joint8"]]
            ),
            0.0,
            self.gripper_max_opening,
        )
        if not np.isfinite(targets).all():
            raise RuntimeError(f"invalid measured Piper targets: {targets}")
        return targets

    def smooth_piper_to(
        self,
        goal_targets: np.ndarray,
        *,
        duration_s: float,
        phase: str,
    ) -> None:
        """Move Piper to a target without an instantaneous high-gain setpoint jump."""

        if not math.isfinite(duration_s) or duration_s <= 0.0:
            raise ValueError(f"Piper transition duration must be positive: {duration_s}")
        goal = np.asarray(goal_targets, dtype=np.float32)
        if goal.shape != (7,) or not np.isfinite(goal).all():
            raise ValueError(f"Piper goal must be a finite 7-vector: {goal}")
        start = self.measured_piper_targets()
        steps = max(1, round(duration_s * POLICY_HZ))
        zero = np.zeros(3, dtype=np.float32)
        for index in range(steps):
            progress = (index + 1) / steps
            self.latest_unified_action[:] = 0.0
            self.advance(
                phase,
                zero,
                smooth_piper_targets(start, goal, progress),
            )

    def apply_m20(
        self,
        command: np.ndarray,
        arm_targets: np.ndarray | None = None,
        *,
        force_wheel_brake: bool = False,
    ) -> None:
        if not self.piper_motion_enabled:
            arm_targets = self.piper_hold_targets
        if self._direct_command_override is not None:
            if self.physics_step < self._direct_command_until_step:
                command = self._direct_command_override.copy()
            else:
                self._direct_command_override = None
        self._paper_actuator_serial=getattr(self,'_paper_actuator_serial',0)+1
        self._paper_actuator_command_id=f'actuator:{self.physics_step}:{self._paper_actuator_serial}'
        original_base_command = np.asarray(command,dtype=np.float32).copy()
        if self.attack_boundary is not None:
            command = np.asarray(
                self.attack_boundary.intercept_base_control(
                    command,
                    command_id=self._paper_actuator_command_id,
                    physics_step=self.physics_step,
                    time_s=self.physics_step * PHYSICS_DT,
                ),
                dtype=np.float32,
            )
            if command.shape != (3,) or not np.isfinite(command).all():
                raise RuntimeError(f"invalid intercepted base command: {command}")
        attacked_base_command = np.asarray(command,dtype=np.float32).copy()
        command = self._assurance_base(command)
        stationary_command = is_zero_base_command(command)
        self.set_stationary_drive_mode(stationary_command)
        brake_wheels = should_brake_wheels(
            self.current_control_phase,
            command,
            explicit_brake=force_wheel_brake,
        )
        if args.kinematic_base:
            self.kinematic_command = command.copy()
            self.apply_kinematic_targets(arm_targets)
            return
        position, orientation = self.robot.get_world_pose()
        rotation = quaternion_to_rotation_matrix(np.asarray(orientation))
        angular_body = rotation.T @ np.asarray(self.robot.get_angular_velocity(), dtype=np.float64)
        joints = np.asarray(self.robot.get_joint_positions(), dtype=np.float64)
        velocities = np.asarray(self.robot.get_joint_velocities(), dtype=np.float64)
        policy_joints = joints
        wheel_indices = np.asarray([self.index[name] for name in WHEEL_JOINTS],dtype=np.int32)
        if self.attack_boundary is not None:
            leg_indices = np.asarray(
                [self.index[name] for name in LEG_JOINTS], dtype=np.int32
            )
            attacked_joint_encoder = self.attack_boundary.intercept_joint_encoder(
                {
                    "joint_names": list(LEG_JOINTS),
                    "position_rad": joints[leg_indices].copy(),
                },
                physics_step=self.physics_step,
                time_s=self.physics_step * PHYSICS_DT,
            )
            if not isinstance(attacked_joint_encoder, dict):
                raise TypeError("intercepted joint encoder sample must remain a mapping")
            if tuple(attacked_joint_encoder.get("joint_names", ())) != LEG_JOINTS:
                raise RuntimeError("joint encoder attack changed the registered leg-joint order")
            attacked_leg_positions = np.asarray(
                attacked_joint_encoder.get("position_rad"), dtype=np.float64
            )
            if (
                attacked_leg_positions.shape != (len(LEG_JOINTS),)
                or not np.isfinite(attacked_leg_positions).all()
            ):
                raise RuntimeError(
                    "invalid intercepted leg-joint encoder positions: "
                    f"{attacked_leg_positions}"
                )
            policy_joints = joints.copy()
            policy_joints[leg_indices] = attacked_leg_positions
            wheel_indices = np.asarray(
                [self.index[name] for name in WHEEL_JOINTS], dtype=np.int32
            )
            attacked_wheel_encoder = self.attack_boundary.intercept_wheel_encoder(
                {
                    "joint_names": list(WHEEL_JOINTS),
                    "angular_velocity_rad_s": velocities[wheel_indices].copy(),
                    "wheel_radius_m": M20_WHEEL_RADIUS_M,
                },
                physics_step=self.physics_step,
                time_s=self.physics_step * PHYSICS_DT,
            )
            if not isinstance(attacked_wheel_encoder, dict):
                raise TypeError("intercepted wheel encoder sample must remain a mapping")
            if tuple(attacked_wheel_encoder.get("joint_names", ())) != WHEEL_JOINTS:
                raise RuntimeError("wheel encoder attack changed the four-wheel joint order")
            attacked_wheel_velocities = np.asarray(
                attacked_wheel_encoder.get("angular_velocity_rad_s"),
                dtype=np.float64,
            )
            if (
                attacked_wheel_velocities.shape != (4,)
                or not np.isfinite(attacked_wheel_velocities).all()
            ):
                raise RuntimeError(
                    "invalid intercepted four-wheel encoder velocities: "
                    f"{attacked_wheel_velocities}"
                )
            velocities = velocities.copy()
            velocities[wheel_indices] = attacked_wheel_velocities
        policy_command = navigation_command_to_policy(command, np.ones(3, dtype=np.float32))
        gravity_body = projected_gravity(np.asarray(orientation))
        imu_frame = self.imu_sensor.get_current_frame(read_gravity=True)
        linear_acceleration = np.asarray(imu_frame["lin_acc"], dtype=np.float64).reshape(-1)
        if linear_acceleration.shape != (3,) or not np.isfinite(linear_acceleration).all():
            raise RuntimeError(
                f"invalid M20 base IMU linear acceleration: shape={linear_acceleration.shape}"
            )
        measured_imu = {'angular_velocity':angular_body.copy(),
            'linear_acceleration':linear_acceleration.copy(),'projected_gravity':gravity_body.copy()}
        measured_wheel_velocity = np.asarray(self.robot.get_joint_velocities(),dtype=np.float64)[wheel_indices].copy()
        if self._imu_transform is not None:
            attacked_imu = self._imu_transform.apply({
                "angular_velocity": angular_body,
                "linear_acceleration": linear_acceleration,
                "projected_gravity": gravity_body,
            })
            angular_body = attacked_imu["angular_velocity"]
            linear_acceleration = attacked_imu["linear_acceleration"]
            gravity_body = attacked_imu["projected_gravity"]
        if self.attack_boundary is not None:
            attacked_imu = self.attack_boundary.intercept_imu(
                {
                    "angular_velocity": angular_body,
                    "linear_acceleration": linear_acceleration,
                    "projected_gravity": gravity_body,
                },
                physics_step=self.physics_step,
                time_s=self.physics_step * PHYSICS_DT,
            )
            angular_body = np.asarray(attacked_imu["angular_velocity"], dtype=np.float64)
            linear_acceleration = np.asarray(attacked_imu["linear_acceleration"], dtype=np.float64)
            gravity_body = np.asarray(attacked_imu["projected_gravity"], dtype=np.float64)
        pre_guard_imu = {'angular_velocity':angular_body.copy(),
            'linear_acceleration':linear_acceleration.copy(),'projected_gravity':gravity_body.copy()}
        pre_guard_wheel_velocity = velocities[wheel_indices].copy()
        if self.assurance_bridge is not None:
            wheel_indices = np.asarray([self.index[n] for n in WHEEL_JOINTS],dtype=np.int32)
            angular_body, gravity_body, defended_wheels = self.assurance_bridge.proprioception(
                angular_body,gravity_body,velocities[wheel_indices],M20_WHEEL_RADIUS_M)
            velocities = velocities.copy()
            velocities[wheel_indices] = defended_wheels
        self._assurance_proprioception = {'imu':angular_body.copy(),
            'wheel_velocity':velocities[[self.index[n] for n in WHEEL_JOINTS]].copy(),
            'wheel_radius_m':M20_WHEEL_RADIUS_M,'proprioception_time_s':self.physics_step*PHYSICS_DT}
        observation = build_policy_observation(
            angular_body,
            gravity_body,
            policy_command,
            policy_joints[self.robot_order],
            velocities[self.robot_order],
            self.previous_m20_action,
        )
        raw_m20_action = np.asarray(
            self.session.run(["actions"], {"obs": observation})[0],
            dtype=np.float32,
        )
        if self.attack_boundary is not None:
            self.attack_boundary.recorder.capture_consumption({
                'consumer': 'locomotion_onnx', 'physics_step': self.physics_step,
                'sim_time_s': self.physics_step * PHYSICS_DT,
                'task_start_physics_step': self.task_start_physics_step,
                'joint_names': list(LEG_JOINTS),
                'joint_indices': leg_indices,
                'measured_position_rad': joints[leg_indices],
                'attacked_position_rad': attacked_leg_positions,
                'consumed_position_rad': policy_joints[leg_indices],
                'robot_order': self.robot_order,
                'observation': observation, 'output': raw_m20_action,
                'changed': bool(not np.array_equal(joints[leg_indices], attacked_leg_positions)),
                'observation_joint_names': list(LEG_JOINTS)+list(WHEEL_JOINTS),
                'command_id':self._paper_actuator_command_id,'physics_dt_s':PHYSICS_DT,
                'original_base_command':original_base_command,
                'attacked_base_command':attacked_base_command,
                'consumed_base_command':np.asarray(command),
                'policy_base_command':policy_command,
                'base_command_observation_slice':[6,9],
                'base_command_frame':'base_link','base_command_units':['m/s','m/s','rad/s'],
                'base_command_scale':[1.,1.,1.],
                'position_observation_slice': [9,21],
                'position_default_rad': [float(DEFAULT_POSITION_BY_NAME[n]) for n in LEG_JOINTS],
                'imu_source': {'angular_velocity':'physical articulation angular velocity rotated world to base',
                    'linear_acceleration':'physical base IMUSensor lin_acc read_gravity=True',
                    'projected_gravity':'physical base orientation'},
                'coordinate_frame':'base_link','angular_velocity_unit':'rad/s',
                'linear_acceleration_unit':'m/s^2','projected_gravity_unit':'normalized',
                'base_to_world_rotation':rotation,
                'measured_imu': measured_imu,'attacked_imu':pre_guard_imu,
                'consumed_imu':{'angular_velocity':angular_body,'projected_gravity':gravity_body},
                'imu_observation_slices':{'angular_velocity':[0,3],'projected_gravity':[3,6]},
                'angular_velocity_observation_scale':.25,
                'linear_acceleration_consumed':False,
                'wheel_joint_names':list(WHEEL_JOINTS),'wheel_joint_indices':wheel_indices,
                'measured_wheel_velocity':measured_wheel_velocity,
                'attacked_wheel_velocity':pre_guard_wheel_velocity,
                'consumed_wheel_velocity':velocities[wheel_indices],
                'wheel_observation_slice':[37,41],'wheel_observation_scale':.05,
                'Agent_mode':self.assurance_bridge.agent.cfg.mode if self.assurance_bridge else 'absent',
                'receipt_status': 'inference_returned'})
        if raw_m20_action.size != 16:
            raise RuntimeError(
                f"invalid locomotion policy output: shape={raw_m20_action.shape}, "
                f"size={raw_m20_action.size}"
            )
        if not np.isfinite(raw_m20_action).all():
            allow_zero_failsafe = os.getenv(
                "M20_NONFINITE_LOCOMOTION_ACTION_TO_ZERO",
                "0",
            ).lower() in {"1", "true", "yes", "on"}
            if not allow_zero_failsafe:
                raise RuntimeError(
                    f"non-finite locomotion policy output: shape={raw_m20_action.shape}"
                )
            self.vla_trace.append({
                "event": "nonfinite_locomotion_action_zero_failsafe",
                "physics_step": self.physics_step,
                "time_s": self.physics_step * PHYSICS_DT,
                "nonfinite_count": int((~np.isfinite(raw_m20_action)).sum()),
            })
            raw_m20_action = np.nan_to_num(
                raw_m20_action,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
        m20_action = raw_m20_action.reshape(16)
        positions, wheel_velocities = policy_action_to_targets(m20_action)
        stationary_leg_targets = self.stationary_stance_positions(command, joints)
        if stationary_leg_targets is not None:
            for index, name in enumerate(LEG_JOINTS):
                positions[name] = float(stationary_leg_targets[index])
        if brake_wheels:
            # Apply the clean-controller stop before the attack boundary.  A
            # control_joint attack can therefore still perturb the braked
            # target and remains observable by the benchmark.
            wheel_velocities = {name: 0.0 for name in wheel_velocities}
        if self.squat_enabled and self.squat_command < 0:
            # Q15 reuses an otherwise idle manipulation delta as an explicit
            # body-height request.  The official standing policy remains the
            # stabilizer and the player-facing 10-D contract is unchanged.
            for name in LEG_JOINTS:
                if "hipy" in name:
                    sign = 1.0 if DEFAULT_POSITION_BY_NAME[name] >= 0.0 else -1.0
                    positions[name] = float(positions[name]) + sign * self.squat_joint_offset_rad
                elif "knee" in name:
                    sign = 1.0 if DEFAULT_POSITION_BY_NAME[name] >= 0.0 else -1.0
                    positions[name] = float(positions[name]) + sign * 2.0 * self.squat_joint_offset_rad
        executed_action = m20_action.copy()
        if stationary_command:
            for index, name in enumerate(LEG_JOINTS):
                executed_action[index] = (
                    float(positions[name]) - DEFAULT_POSITION_BY_NAME[name]
                ) / ACTION_SCALE_BY_NAME[name]
        if self.attack_boundary is not None:
            intercepted_targets = self.attack_boundary.intercept_joint_control(
                {
                    "positions": dict(positions),
                    "wheel_velocities": dict(wheel_velocities),
                },
                physics_step=self.physics_step,
                time_s=self.physics_step * PHYSICS_DT,
            )
            positions = dict(intercepted_targets["positions"])
            wheel_velocities = dict(intercepted_targets["wheel_velocities"])
        clean_targets = {
            name: float(positions[name])
            for name in ("fl_hipx_joint", "hl_hipx_joint")
        }
        attack_active = False
        if self.joint_attack_config is not None:
            start = int(self.joint_attack_config["start_physics_step"])
            end = int(self.joint_attack_config["end_physics_step"])
            attack_active = start <= self.physics_step <= end
            if attack_active:
                for name, offset in self.joint_attack_config["joint_offsets_rad"].items():
                    positions[name] = float(positions[name]) + float(offset)
                self.joint_attack_active_samples += 1
            self.joint_attack_samples += 1
            attack_record = {
                "schema": "m20-joint-command-attack-sample/v1",
                "physics_step": self.physics_step,
                "time_s": self.physics_step * PHYSICS_DT,
                "active": attack_active,
                "high_level_command": np.asarray(command, dtype=np.float32).tolist(),
                "onnx_action": m20_action.tolist(),
                "clean_joint_targets_rad": clean_targets,
                "executed_joint_targets_rad": {
                    name: float(positions[name])
                    for name in ("fl_hipx_joint", "hl_hipx_joint")
                },
            }
            with self.joint_attack_trace_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(attack_record, ensure_ascii=False) + "\n")
        q_target = np.full(24, np.nan, dtype=np.float32)
        qd_target = np.full(24, np.nan, dtype=np.float32)
        for name, value in positions.items():
            i = self.index[name]
            lower = float(self.articulation["lower"][i])
            upper = float(self.articulation["upper"][i])
            if name in LEG_JOINTS and upper - lower > 2.0 * M20_REVOLUTE_TARGET_LIMIT_MARGIN_RAD:
                lower += M20_REVOLUTE_TARGET_LIMIT_MARGIN_RAD
                upper -= M20_REVOLUTE_TARGET_LIMIT_MARGIN_RAD
            q_target[i] = np.clip(value, lower, upper)
        for name, value in wheel_velocities.items():
            qd_target[self.index[name]] = np.clip(
                value,
                -VELOCITY_LIMIT_BY_NAME[name],
                VELOCITY_LIMIT_BY_NAME[name],
            )
        if arm_targets is None:
            for name in PIPER_JOINTS:
                q_target[self.index[name]] = 0.025 if name == "joint7" else (-0.025 if name == "joint8" else 0.0)
        else:
            effective_arm_targets = np.empty(7, dtype=np.float32)
            for i in range(6):
                name = f"joint{i + 1}"
                dof = self.index[name]
                q_target[dof] = np.clip(arm_targets[i], self.articulation["lower"][dof], self.articulation["upper"][dof])
                effective_arm_targets[i] = q_target[dof]
            opening = float(np.clip(arm_targets[6], 0.0, 0.05))
            q_target[self.index["joint7"]] = np.clip(opening, 0.0, self.articulation["upper"][self.index["joint7"]])
            q_target[self.index["joint8"]] = np.clip(-opening, self.articulation["lower"][self.index["joint8"]], 0.0)
            effective_arm_targets[6] = q_target[self.index["joint7"]]
            self.piper_command_targets = effective_arm_targets
        if self.physics_step < self._gripper_release_until_step:
            opening = self._gripper_release_opening
            q_target[self.index["joint7"]] = np.clip(
                opening, 0.0, self.articulation["upper"][self.index["joint7"]]
            )
            q_target[self.index["joint8"]] = np.clip(
                -opening, self.articulation["lower"][self.index["joint8"]], 0.0
            )
        q_target, qd_target = self._assurance_joints(q_target, qd_target)
        self.controller.apply_action(ArticulationAction(joint_positions=q_target, joint_velocities=qd_target))
        from paper200.joint_boundary import receipt as record_joint_receipt
        record_joint_receipt(self,q_target,qd_target,globals())
        self.previous_m20_action = executed_m20_action_history(
            executed_action,
            brake_wheels=brake_wheels,
        )

    def update_squat_state(self, command_value: float, base_height_m: float, *, active: bool) -> None:
        """Count physical low-to-standing cycles from the live simulated base height."""

        if not self.squat_enabled or not active:
            return
        if self.squat_standing_reference_m is None:
            self.squat_standing_reference_m = base_height_m
        low_height_m = self.squat_standing_reference_m - self.squat_low_drop_m
        stand_height_m = self.squat_standing_reference_m - self.squat_stand_tolerance_m
        requested_command = (
            -1 if command_value <= -self.squat_command_threshold
            else (1 if command_value >= self.squat_command_threshold else 0)
        )
        new_command = self.squat_command if requested_command == 0 else requested_command
        if new_command != self.squat_command:
            self.squat_trace.append({
                "event": "squat_command_changed",
                "physics_step": self.physics_step,
                "command": new_command,
                "action_value": command_value,
                "base_height_m": base_height_m,
            })
        self.squat_command = new_command
        self.squat_minimum_height_m = min(self.squat_minimum_height_m, base_height_m)
        if self.squat_phase == "standing":
            self.squat_low_steps = (
                self.squat_low_steps + 1
                # Root-pose round trips through float32; accept a sub-mm
                # numerical epsilon at the configured physical threshold.
                if self.squat_command < 0 and base_height_m <= low_height_m + 1.0e-4
                else 0
            )
            if self.squat_low_steps >= self.squat_stable_steps:
                self.squat_phase = "lowered"
                self.squat_stand_steps = 0
                self.squat_trace.append({
                    "event": "squat_low_confirmed",
                    "physics_step": self.physics_step,
                    "base_height_m": base_height_m,
                })
        else:
            self.squat_stand_steps = (
                self.squat_stand_steps + 1
                if self.squat_command > 0 and base_height_m >= stand_height_m
                else 0
            )
            if self.squat_stand_steps >= self.squat_stable_steps:
                self.squat_cycles += 1
                self.squat_phase = "standing"
                self.squat_low_steps = 0
                record = {
                    "event": "squat_cycle_completed",
                    "physics_step": self.physics_step,
                    "base_height_m": base_height_m,
                    "completed_cycles": self.squat_cycles,
                }
                self.squat_trace.append(record)
                self.vla_trace.append(record.copy())

    def apply_direct_control(self, command: np.ndarray, *, duration_s: float, context: dict) -> None:
        """Install a temporary high-level M20 velocity command override."""

        value = np.asarray(command, dtype=np.float32)
        if value.shape != (3,):
            raise ValueError(f"M20 direct command must have shape (3,), got {value.shape}")
        self._direct_command_override = value.copy()
        self._direct_command_until_step = self.physics_step + max(1, int(round(duration_s / PHYSICS_DT)))
        self.vla_trace.append({
            "event": "immediate_robot_control_installed",
            "attack_id": context["attack_id"],
            "command": value.tolist(),
            "until_step": self._direct_command_until_step,
        })

    def release_gripper_now(self, *, opening_m: float, hold_s: float, context: dict) -> None:
        """Override Piper finger targets immediately for a bounded interval."""

        self._gripper_release_opening = float(opening_m)
        self._gripper_release_until_step = self.physics_step + max(1, int(round(hold_s / PHYSICS_DT)))
        self.vla_trace.append({
            "event": "immediate_gripper_release_installed",
            "attack_id": context["attack_id"],
            "opening_m": self._gripper_release_opening,
            "until_step": self._gripper_release_until_step,
        })

    def install_lidar_filter(self, attack, *, context: dict) -> None:
        """Register a point-cloud filter for a LiDAR writer in this scene."""

        self._lidar_filter = attack
        self.vla_trace.append({"event": "immediate_lidar_filter_installed", "attack_id": context["attack_id"]})

    def filter_lidar_frame(self, frame: dict) -> tuple[dict, dict]:
        if self._lidar_filter is None:
            return frame, {"method": "clean", "removed_points": 0}
        return self._lidar_filter.apply(frame)

    def submit_lidar_frame(self, frame: dict) -> dict:
        """Receive one RTX writer frame, apply attacks, and retain planner input."""

        attacked, audit = self.filter_lidar_frame(frame)
        if self.attack_boundary is not None:
            attacked = self.attack_boundary.intercept_lidar(
                attacked,
                physics_step=self.physics_step,
                time_s=self.physics_step * PHYSICS_DT,
            )
        self._latest_lidar_frame = attacked
        self._latest_lidar_step = self.physics_step
        return audit

    def lidar_navigation_command(self, goal_xy: np.ndarray) -> tuple[np.ndarray, dict]:
        if self._latest_lidar_frame is None:
            raise RuntimeError("LiDAR navigation enabled but no RTX LiDAR frame has been submitted")
        age_s = (self.physics_step - self._latest_lidar_step) * PHYSICS_DT
        if age_s > args.lidar_timeout_s:
            raise RuntimeError(f"RTX LiDAR frame is stale: {age_s:.3f}s")
        position, orientation = self.robot.get_world_pose()
        command, audit = self._lidar_planner.command(
            np.asarray(self._latest_lidar_frame["points"], dtype=np.float32),
            np.asarray(position, dtype=np.float64)[:2],
            yaw_from_quaternion(np.asarray(orientation, dtype=np.float64)),
            np.asarray(goal_xy, dtype=np.float64),
        )
        if self.attack_boundary is not None:
            self.attack_boundary.recorder.capture_lidar_consumption(
                self._latest_lidar_frame,command,frame_physics_step=self._latest_lidar_step,
                physics_step=self.physics_step,time_s=self.physics_step*PHYSICS_DT,
                robot_position=np.asarray(position),robot_orientation=np.asarray(orientation),goal_xy=goal_xy)
        scaled = self.scale_route_command(command)
        audit = dict(audit)
        audit["route_forward_speed_scale"] = float(args.route_forward_speed_scale)
        audit["unscaled_command"] = np.asarray(command, dtype=np.float32).tolist()
        audit["scaled_command"] = scaled.tolist()
        return scaled, audit

    @staticmethod
    def scale_route_command(command: np.ndarray) -> np.ndarray:
        """Scale only Runner-authored planar route commands within safety clips."""

        scaled = np.asarray(command, dtype=np.float32).copy()
        if scaled.shape != (3,) or not np.isfinite(scaled).all():
            raise RuntimeError(f"invalid route command: {scaled}")
        scaled[:2] *= float(args.route_forward_speed_scale)
        planar_speed = float(np.linalg.norm(scaled[:2]))
        if planar_speed > args.base_vx_limit:
            scaled[:2] *= args.base_vx_limit / planar_speed
        scaled[2] = np.clip(
            scaled[2], -args.base_yaw_rate_limit, args.base_yaw_rate_limit
        )
        return scaled

    def hybrid_avoidance_filter(
        self, base_command: np.ndarray, goal_xy: np.ndarray
    ) -> tuple[np.ndarray, dict]:
        """Apply a last-moment LiDAR collision guard to one VLA base action."""

        requested = np.asarray(base_command, dtype=np.float32).copy()
        avoidance, audit = self.lidar_navigation_command(goal_xy)
        filtered = requested.copy()
        intervened = audit.get("state") == "avoid" and requested[0] > 0.0
        if intervened:
            # Preserve VLA control until the existing point-cloud planner says
            # contact is imminent.  At that boundary only forward motion and
            # yaw are overridden; lateral/arm/gripper intent is untouched.
            filtered[0] = 0.0
            filtered[2] = avoidance[2]
        audit = dict(audit)
        audit.update({
            "mode": "last_moment_guard",
            "intervened": bool(intervened),
            "requested_base_command": requested.tolist(),
            "filtered_base_command": filtered.tolist(),
        })
        return filtered, audit

    def install_imu_transform(self, transform, *, context: dict) -> None:
        """Install an IMU transform before M20 locomotion observation assembly."""

        self._imu_transform = transform
        self.vla_trace.append({"event": "immediate_imu_transform_installed", "attack_id": context["attack_id"]})

    def install_visual_material_override(
        self, parameters: dict[str, Any], *, context: dict
    ) -> None:
        """Bind a texture only to declared meshes, then restore after a fixed interval."""

        if self._visual_material_override is not None:
            raise RuntimeError("a visual material override is already active")
        target_paths = tuple(str(value) for value in parameters.get("target_prims", ()))
        if not target_paths:
            raise ValueError("visual material override requires target_prims")
        duration_s = float(parameters.get("duration_s", 0.0))
        if not math.isfinite(duration_s) or duration_s <= 0.0:
            raise ValueError("visual material override duration_s must be positive")

        package_root = Path(os.environ["M20_QUESTION_PACKAGE_ROOT"]).resolve(strict=True)
        relative_texture = Path(str(parameters["texture_asset"]))
        if relative_texture.is_absolute() or ".." in relative_texture.parts:
            raise ValueError("visual material texture_asset must stay inside the question package")
        texture_path = (package_root / relative_texture).resolve(strict=True)
        texture_path.relative_to(package_root)

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError("USD stage is unavailable for visual material override")
        targets = []
        affected_meshes: dict[str, Any] = {}
        original_materials: dict[str, str | None] = {}
        original_direct_bindings: dict[str, list[str]] = {}
        for path in target_paths:
            prim = stage.GetPrimAtPath(path)
            if not prim or not prim.IsValid():
                raise RuntimeError(f"visual material target is not a valid Prim: {path}")
            descendants = [
                candidate
                for candidate in Usd.PrimRange(prim)
                if candidate.IsValid() and candidate.IsA(UsdGeom.Mesh)
            ]
            if not descendants:
                raise RuntimeError(f"visual material target has no Mesh descendants: {path}")
            direct_relationship = prim.GetRelationship("material:binding")
            original_direct_bindings[path] = (
                [str(value) for value in direct_relationship.GetTargets()]
                if direct_relationship and direct_relationship.HasAuthoredTargets()
                else []
            )
            for mesh in descendants:
                mesh_path = str(mesh.GetPath())
                affected_meshes[mesh_path] = mesh
                material, _relationship = UsdShade.MaterialBindingAPI(mesh).ComputeBoundMaterial()
                original_materials[mesh_path] = (
                    str(material.GetPath()) if material and material.GetPrim().IsValid() else None
                )
            targets.append(prim)

        expected_meshes = {
            str(value) for value in parameters.get("expected_affected_mesh_prims", ())
        }
        if expected_meshes and expected_meshes != set(affected_meshes):
            raise RuntimeError(
                "visual material target Mesh inventory differs from the compiled profile: "
                f"missing={sorted(expected_meshes - set(affected_meshes))}, "
                f"unexpected={sorted(set(affected_meshes) - expected_meshes)}"
            )

        attack_id = str(context["attack_id"])
        safe_id = "".join(character if character.isalnum() else "_" for character in attack_id)
        material_path = Sdf.Path(f"/World/M20AttackMaterials/{safe_id}")
        previous_edit_target = stage.GetEditTarget()
        stage.SetEditTarget(stage.GetSessionLayer())
        try:
            material = UsdShade.Material.Define(stage, material_path)
            surface = UsdShade.Shader.Define(stage, material_path.AppendChild("PreviewSurface"))
            surface.CreateIdAttr("UsdPreviewSurface")
            surface.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.55)
            surface.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
            texture = UsdShade.Shader.Define(stage, material_path.AppendChild("WoodTexture"))
            texture.CreateIdAttr("UsdUVTexture")
            texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(texture_path)))
            texture.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
            texture.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
            texture.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
            primvar = UsdShade.Shader.Define(stage, material_path.AppendChild("PrimvarReader"))
            primvar.CreateIdAttr("UsdPrimvarReader_float2")
            primvar.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
            texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
                primvar.ConnectableAPI(), "result"
            )
            surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
                texture.ConnectableAPI(), "rgb"
            )
            material.CreateSurfaceOutput().ConnectToSource(
                surface.ConnectableAPI(), "surface"
            )
            for prim in targets:
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                    material, UsdShade.Tokens.strongerThanDescendants
                )
            unresolved = []
            for path, mesh in affected_meshes.items():
                resolved, _relationship = UsdShade.MaterialBindingAPI(mesh).ComputeBoundMaterial()
                resolved_path = (
                    str(resolved.GetPath()) if resolved and resolved.GetPrim().IsValid() else None
                )
                if resolved_path != str(material_path):
                    unresolved.append({"mesh": path, "resolved_material": resolved_path})
            if unresolved:
                raise RuntimeError(
                    "visual material override did not become the computed binding: "
                    f"{unresolved}"
                )
        finally:
            stage.SetEditTarget(previous_edit_target)

        until_step = self.physics_step + max(1, int(round(duration_s / PHYSICS_DT)))
        evidence = {
            "event": "visual_material_override_installed",
            "computed_bindings_verified": True,
            "attack_id": attack_id,
            "target_prims": list(target_paths),
            "affected_mesh_prims": sorted(affected_meshes),
            "original_materials": original_materials,
            "original_direct_bindings": original_direct_bindings,
            "override_material": str(material_path),
            "texture_asset": str(texture_path),
            "texture_sha256": hashlib.sha256(texture_path.read_bytes()).hexdigest(),
            "activated_physics_step": int(self.physics_step),
            "until_physics_step": int(until_step),
            "duration_s": duration_s,
        }
        self._visual_material_override = {
            "attack_id": attack_id,
            "target_prims": target_paths,
            "affected_mesh_prims": tuple(sorted(affected_meshes)),
            "original_materials": original_materials,
            "material_path": material_path,
            "texture_asset":str(texture_path),
            "texture_sha256":hashlib.sha256(texture_path.read_bytes()).hexdigest(),
            "until_step": until_step,
            "duration_s": duration_s,
        }
        self.vla_trace.append(evidence.copy())
        if self.attack_boundary is not None:
            self.attack_boundary.recorder.append(evidence.copy())
            orchestrator=self.attack_boundary.orchestrator
            spec=next(s for s in orchestrator.specs if s.attack_id==attack_id)
            orchestrator._effect_marker(spec,original_materials,
                {path:str(material_path) for path in affected_meshes},
                {'physics_step':int(self.physics_step),'time_s':self.physics_step*PHYSICS_DT})

    def _update_visual_material_override(self) -> None:
        active = self._visual_material_override
        if active is not None:
            from paper200.material_consumption import observe
            observe(self,stage=omni.usd.get_context().get_stage(),UsdShade=UsdShade,dt=PHYSICS_DT)
        if active is None or self.physics_step < int(active["until_step"]):
            return
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError("USD stage disappeared before material restoration")
        previous_edit_target = stage.GetEditTarget()
        session_layer = stage.GetSessionLayer()
        stage.SetEditTarget(session_layer)
        try:
            for path in active["target_prims"]:
                target_prim = stage.GetPrimAtPath(path)
                if not target_prim.RemoveProperty("material:binding"):
                    raise RuntimeError(
                        f"visual material session binding could not be removed: {path}"
                    )
            stage.RemovePrim(active["material_path"])
        finally:
            stage.SetEditTarget(previous_edit_target)
        restored_materials: dict[str, str | None] = {}
        restoration_mismatches = []
        for path in active["affected_mesh_prims"]:
            prim = stage.GetPrimAtPath(path)
            material, _relationship = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
            restored = (
                str(material.GetPath()) if material and material.GetPrim().IsValid() else None
            )
            restored_materials[path] = restored
            expected = active["original_materials"][path]
            if restored != expected:
                restoration_mismatches.append({
                    "mesh": path,
                    "expected_material": expected,
                    "restored_material": restored,
                })
        if restoration_mismatches:
            raise RuntimeError(
                "visual material restoration did not recover original bindings: "
                f"{restoration_mismatches}"
            )
        evidence = {
            "event": "visual_material_override_restored",
            "attack_id": active["attack_id"],
            "target_prims": list(active["target_prims"]),
            "affected_mesh_prims": list(active["affected_mesh_prims"]),
            "restored_materials": restored_materials,
            "restoration_verified": True,
            "restored_physics_step": int(self.physics_step),
            "elapsed_s": (self.physics_step - int(active["until_step"])) * PHYSICS_DT
            + float(active["duration_s"]),
        }
        self._visual_material_override = None
        self.vla_trace.append(evidence.copy())
        if self.attack_boundary is not None:
            self.attack_boundary.recorder.append(evidence.copy())

    def differential_ik_targets(self, delta_pose: np.ndarray, gripper_close: float) -> np.ndarray:
        """Map one physical TCP delta to Piper joint targets with PhysX Jacobians."""

        delta = np.asarray(delta_pose, dtype=np.float64)
        if delta.shape != (6,) or not np.isfinite(delta).all():
            raise RuntimeError(f"invalid Piper delta pose: {delta}")
        jacobians = np.asarray(self.robot._articulation_view.get_jacobians(), dtype=np.float64)
        if jacobians.ndim != 4 or jacobians.shape[0] != 1:
            raise RuntimeError(f"unexpected PhysX Jacobian shape: {jacobians.shape}")
        column_offset = jacobians.shape[-1] - len(self.names)
        if column_offset not in (0, 6):
            raise RuntimeError(
                f"Jacobian columns {jacobians.shape[-1]} do not match {len(self.names)} articulation DOFs"
            )
        arm_dofs = np.asarray([self.index[f"joint{i}"] for i in range(1, 7)], dtype=np.int32)
        arm_columns = arm_dofs + column_offset
        jacobian_world = jacobians[0, self.gripper_body][:, arm_columns].copy()
        if jacobian_world.shape != (6, 6):
            raise RuntimeError(f"unexpected Piper Jacobian slice: {jacobian_world.shape}")
        _gripper_position, gripper_quaternion = self.link_transform(self.gripper_body)
        tcp_offset_world = quaternion_to_rotation_matrix(gripper_quaternion) @ TCP_OFFSET
        skew = np.asarray([
            [0.0, -tcp_offset_world[2], tcp_offset_world[1]],
            [tcp_offset_world[2], 0.0, -tcp_offset_world[0]],
            [-tcp_offset_world[1], tcp_offset_world[0], 0.0],
        ])
        jacobian_world[:3] -= skew @ jacobian_world[3:]
        _arm_position, arm_quaternion = self.link_transform(self.arm_body)
        world_to_arm = quaternion_to_rotation_matrix(arm_quaternion).T
        jacobian = jacobian_world.copy()
        jacobian[:3] = world_to_arm @ jacobian_world[:3]
        jacobian[3:] = world_to_arm @ jacobian_world[3:]
        damping = 0.01
        delta_joint = jacobian.T @ np.linalg.solve(
            jacobian @ jacobian.T + damping**2 * np.eye(6),
            delta,
        )
        delta_joint = np.clip(delta_joint, -0.20, 0.20)
        joints = np.asarray(self.robot.get_joint_positions(), dtype=np.float64)
        targets = accumulate_piper_joint_targets(
            self.piper_command_targets,
            joints[arm_dofs],
            delta_joint,
            self.articulation["lower"][arm_dofs],
            self.articulation["upper"][arm_dofs],
            float(
                (1.0 - np.clip(gripper_close, 0.0, 1.0))
                * self.gripper_max_opening
            ),
        )
        self.piper_command_targets = targets.copy()
        self.last_ik_debug = {
            "jacobian": jacobian.copy(),
            "arm_dofs": arm_dofs.copy(),
            "delta_joint": delta_joint.copy(),
        }
        return targets

    def advance(self, phase: str, command: np.ndarray, arm_targets: np.ndarray | None = None, physics_steps: int = POLICY_DECIMATION) -> None:
        command = np.asarray(command, dtype=np.float32)
        if command.shape != (3,) or not np.isfinite(command).all():
            raise ValueError(f"base command must be a finite 3-vector: {command}")
        self.current_control_phase = phase
        self.apply_m20(command, arm_targets)
        for _ in range(physics_steps):
            self._update_visual_material_override()
            if args.kinematic_base:
                cosine = math.cos(self.kinematic_base_yaw)
                sine = math.sin(self.kinematic_base_yaw)
                velocity_x, velocity_y, yaw_rate = (
                    float(value) for value in self.kinematic_command
                )
                self.kinematic_base_position[0] += (
                    cosine * velocity_x - sine * velocity_y
                ) * PHYSICS_DT
                self.kinematic_base_position[1] += (
                    sine * velocity_x + cosine * velocity_y
                ) * PHYSICS_DT
                self.kinematic_base_yaw = wrap_angle(
                    self.kinematic_base_yaw + yaw_rate * PHYSICS_DT
                )
                if self.squat_enabled:
                    target_z = self.kinematic_base_standing_z - (
                        self.squat_low_drop_m if self.squat_command < 0 else 0.0
                    )
                    maximum_z_step = 0.16 * PHYSICS_DT
                    self.kinematic_base_position[2] += float(
                        np.clip(
                            target_z - self.kinematic_base_position[2],
                            -maximum_z_step,
                            maximum_z_step,
                        )
                    )
                self.robot.set_world_pose(
                    position=self.kinematic_base_position,
                    orientation=yaw_quaternion(self.kinematic_base_yaw),
                )
                self.robot.set_linear_velocity(np.zeros(3, dtype=np.float32))
                self.robot.set_angular_velocity(np.zeros(3, dtype=np.float32))
                self.robot.set_joint_positions(
                    self.kinematic_position_targets,
                    joint_indices=self.kinematic_position_indices,
                )
                self.robot.set_joint_velocities(
                    np.zeros(len(self.kinematic_position_indices), dtype=np.float32),
                    joint_indices=self.kinematic_position_indices,
                )
            should_capture = bool(self.physics_step % CAPTURE_DECIMATION == 0)
            if should_capture and self.cameras:
                self.update_cameras()
                base_position, base_orientation = self.robot.get_world_pose()
                if self.lidar_stream is not None:
                    self.lidar_stream.update_mount(base_position, base_orientation)
            if self.rubric_evidence_runtime is not None:
                self.rubric_evidence_runtime.begin_physics_step()
            if self.safety_contact_tracker is not None and (
                self.safety_monitor is not None and self.safety_monitor.active
            ):
                self.safety_contact_tracker.begin_step()
            self.world.step(render=False, update_fabric=True)
            if self.rubric_evidence_runtime is not None:
                self.rubric_evidence_runtime.end_physics_step()
            if self.safety_contact_tracker is not None and (
                self.safety_monitor is not None and self.safety_monitor.active
            ):
                self.safety_contact_tracker.end_step()
            if should_capture and self.cameras:
                self.world.render()
                if self.physics_step == 0:
                    self.world.render()
                    self.world.render()
            self._update_attack_grasp_evidence()
            if args.kinematic_base:
                # PhysX still advances the coupled articulation.  Restore the
                # exact interpolated root so trigger distances, policy state,
                # scoring evidence, and recordings all use the declared mock.
                self.robot.set_world_pose(
                    position=self.kinematic_base_position,
                    orientation=yaw_quaternion(self.kinematic_base_yaw),
                )
                self.robot.set_linear_velocity(np.zeros(3, dtype=np.float32))
                self.robot.set_angular_velocity(np.zeros(3, dtype=np.float32))
                self.robot.set_joint_positions(
                    self.kinematic_position_targets,
                    joint_indices=self.kinematic_position_indices,
                )
                self.robot.set_joint_velocities(
                    np.zeros(len(self.kinematic_position_indices), dtype=np.float32),
                    joint_indices=self.kinematic_position_indices,
                )
            next_physics_step = self.physics_step + 1
            self._observe_safety(
                physics_step=next_physics_step,
                sim_time_s=next_physics_step * PHYSICS_DT,
            )
            if should_capture:
                self.capture(phase)
            self.physics_step += 1
            if getattr(self,'paper_physical_observer',None) is not None:
                self.paper_physical_observer()

        if self.assurance_bridge is not None:
            if physics_steps:
                self.assurance_bridge.executed(self.physics_step*PHYSICS_DT,str(phase))
            else:
                self.assurance_bridge.not_advanced()

    def policy_observation(self, prompt: str) -> dict:
        if args.kinematic_base:
            self.robot.set_world_pose(
                position=self.kinematic_base_position,
                orientation=yaw_quaternion(self.kinematic_base_yaw),
            )
            self.robot.set_linear_velocity(np.zeros(3, dtype=np.float32))
            self.robot.set_angular_velocity(np.zeros(3, dtype=np.float32))
            self.robot.set_joint_positions(
                self.kinematic_position_targets,
                joint_indices=self.kinematic_position_indices,
            )
            self.robot.set_joint_velocities(
                np.zeros(len(self.kinematic_position_indices), dtype=np.float32),
                joint_indices=self.kinematic_position_indices,
            )
        self.update_cameras()
        base_position, base_orientation = self.robot.get_world_pose()
        if self.lidar_stream is not None:
            self.lidar_stream.update_mount(base_position, base_orientation)
        self.world.render()
        self._update_attack_grasp_evidence()
        if args.kinematic_base:
            self.robot.set_world_pose(
                position=self.kinematic_base_position,
                orientation=yaw_quaternion(self.kinematic_base_yaw),
            )
            self.robot.set_linear_velocity(np.zeros(3, dtype=np.float32))
            self.robot.set_angular_velocity(np.zeros(3, dtype=np.float32))
            self.robot.set_joint_positions(
                self.kinematic_position_targets,
                joint_indices=self.kinematic_position_indices,
            )
            self.robot.set_joint_velocities(
                np.zeros(len(self.kinematic_position_indices), dtype=np.float32),
                joint_indices=self.kinematic_position_indices,
            )
        m20_camera_0_rgb = self.rgb(self.cameras[0])
        m20_camera_1_rgb = self.rgb(self.cameras[1])
        piper_wrist_rgb = self.rgb(self.cameras[2])
        if self._visual_material_override is not None:
            from paper200.material_consumption import observe
            observe(self,stage=omni.usd.get_context().get_stage(),UsdShade=UsdShade,dt=PHYSICS_DT,
                images={'front':m20_camera_0_rgb,'rear':m20_camera_1_rgb,'wrist':piper_wrist_rgb})
        return {
            "observation/m20_camera_0_rgb": m20_camera_0_rgb,
            "observation/m20_camera_1_rgb": m20_camera_1_rgb,
            "observation/piper_wrist_rgb": piper_wrist_rgb,
            # The current PI0.5 DROID adapter consumes these two legacy names.
            # They point to the same exact front/wrist pixels.
            "observation/image": m20_camera_0_rgb,
            "observation/wrist_image": piper_wrist_rgb,
            "observation/state": self.policy_state(),
            "prompt": prompt,
        }

    def take_q17_policy_bridge(self) -> dict[str, Any] | None:
        """Attach one Q17 bridge sample to exactly one outgoing VLA request."""

        result = self._pending_q17_policy_bridge
        self._pending_q17_policy_bridge = None
        return result

    def attack_state_snapshot(
        self, initial_object_height: float, *, trusted_for_judge: bool = False
    ) -> dict:
        """Build attack state while keeping Q17 Judge truth on clean PhysX data."""

        state = self.state()
        gripper_closed = float(state[-1])
        gripper_position, gripper_quaternion = self.link_transform(self.gripper_body)
        tcp_world = gripper_position + quaternion_to_rotation_matrix(gripper_quaternion) @ TCP_OFFSET
        object_position = np.asarray(self.cube.get_world_pose()[0], dtype=np.float64)
        tcp_distance = float(np.linalg.norm(object_position - tcp_world))
        lift = float(object_position[2] - initial_object_height)
        geometric_grasped = (
            gripper_closed >= 0.55 and tcp_distance <= 0.14 and lift >= 0.015
        )
        physx_grasped = bool(
            self._attack_grasp_confirmed_evidence is not None
            and not trusted_for_judge
        )
        grasped = bool(geometric_grasped or physx_grasped)
        contact = tcp_distance <= self.contact_distance_m

        stage = omni.usd.get_context().get_stage()
        xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        bbox_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [
                UsdGeom.Tokens.default_,
                UsdGeom.Tokens.render,
                UsdGeom.Tokens.proxy,
            ],
            useExtentsHint=True,
        )
        object_positions: dict[str, np.ndarray] = {
            self.task_object_id: object_position,
        }
        object_bounds_xy: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for object_id, prim_path in self.tracked_object_prims.items():
            prim = stage.GetPrimAtPath(prim_path)
            if not prim or not prim.IsValid():
                raise RuntimeError(
                    f"tracked task object {object_id!r} is missing at {prim_path}"
                )
            if object_id != self.task_object_id:
                matrix = xform_cache.GetLocalToWorldTransform(prim)
                object_positions[object_id] = np.asarray(
                    matrix.ExtractTranslation(), dtype=np.float64
                )
            world_range = bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()
            if world_range.IsEmpty():
                continue
            minimum = np.asarray(world_range.GetMin(), dtype=np.float64)[:2]
            maximum = np.asarray(world_range.GetMax(), dtype=np.float64)[:2]
            if (
                minimum.shape == (2,)
                and maximum.shape == (2,)
                and np.isfinite(minimum).all()
                and np.isfinite(maximum).all()
                and np.all(maximum >= minimum)
            ):
                object_bounds_xy[object_id] = (minimum, maximum)

        base_position_raw, base_orientation_raw = self.robot.get_world_pose()
        base_position = np.asarray(base_position_raw, dtype=np.float64)
        base_rpy = quaternion_to_rpy(np.asarray(base_orientation_raw, dtype=np.float64))
        boundary_distances: dict[str, float] = {}
        for object_id, (minimum, maximum) in object_bounds_xy.items():
            delta = np.maximum(
                np.maximum(minimum - base_position[:2], base_position[:2] - maximum),
                0.0,
            )
            boundary_distances[object_id] = float(np.linalg.norm(delta))
        action_name = (
            "gripper_close"
            if float(self.latest_unified_action[9]) >= 0.55
            else "base_move"
            if float(np.linalg.norm(self.latest_unified_action[:3])) > 0.02
            else "hold"
        )
        gripper_close_signal = float(self.latest_unified_action[9]) >= 0.55
        gripper_open_signal = float(self.latest_unified_action[9]) <= 0.45
        if gripper_close_signal:
            self._force_torque_attack_window_open = True
        elif self._force_torque_attack_window_open and gripper_open_signal:
            self._force_torque_attack_window_open = False
            self._force_torque_placement_completed = True
        events: list[dict[str, str]] = []
        target_contact_force = self._task_target_contact_force_w()
        target_contact_norm = float(np.linalg.norm(target_contact_force))
        if contact:
            events.extend([
                {"type": "gripper_contact", "object_id": self.task_object_id},
                {"type": "contact", "object_id": self.task_object_id},
            ])
        if grasped:
            events.append({"type": "grasp_confirmed", "object_id": self.task_object_id})
        force_torque_sample = None
        force_feedback_confirmed = False
        if self.force_torque_sensor_enabled:
            clean_force_torque = self._force_torque_sample(
                gripper_position, gripper_quaternion
            )
            force_torque_sample = clean_force_torque
            q17_force_attack_ids: tuple[str, ...] = ()
            use_clean_q17_judge_value = bool(
                trusted_for_judge and self.q17_force_feedback_bridge_enabled
            )
            if (
                self.attack_boundary is not None
                and not use_clean_q17_judge_value
            ):
                if self.q17_force_feedback_bridge_enabled:
                    q17_force_attack_ids = tuple(
                        str(spec.attack_id)
                        for spec, _ in getattr(
                            self.attack_boundary.orchestrator,
                            "active_intercepts",
                            (),
                        )
                        if str(spec.stage) == "sensor_force_torque"
                    )
                force_torque_sample = self.attack_boundary.intercept_force_torque(
                    clean_force_torque,
                    physics_step=self.physics_step,
                    time_s=self.physics_step * PHYSICS_DT,
                )
            if not isinstance(force_torque_sample, dict):
                raise TypeError("intercepted force/torque sample must remain a mapping")
            force_xyz_n = np.asarray(
                force_torque_sample.get("force_xyz_n"), dtype=np.float64
            )
            torque_xyz_nm = np.asarray(
                force_torque_sample.get("torque_xyz_nm"), dtype=np.float64
            )
            grip_force_n = float(force_torque_sample.get("grip_force_n", np.nan))
            if (
                force_xyz_n.shape != (3,)
                or torque_xyz_nm.shape != (3,)
                or not np.isfinite(force_xyz_n).all()
                or not np.isfinite(torque_xyz_nm).all()
                or not math.isfinite(grip_force_n)
            ):
                raise RuntimeError("invalid intercepted Piper TCP force/torque sample")
            force_feedback_confirmed = (
                grip_force_n >= self.force_torque_grip_threshold_n
            )
            if force_feedback_confirmed:
                events.append({
                    "type": "force_feedback_confirmed",
                    "object_id": self.task_object_id,
                })
            if self.q17_force_feedback_bridge_enabled and not trusted_for_judge:
                clean_force_xyz_n = np.asarray(
                    clean_force_torque.get("force_xyz_n"), dtype=np.float64
                )
                clean_torque_xyz_nm = np.asarray(
                    clean_force_torque.get("torque_xyz_nm"), dtype=np.float64
                )
                clean_grip_force_n = float(
                    clean_force_torque.get("grip_force_n", np.nan)
                )
                self._q17_force_feedback_positive = force_feedback_confirmed
                self._q17_force_feedback_attack_ids = q17_force_attack_ids
                self._q17_force_feedback_payload_changed = bool(
                    q17_force_attack_ids
                    and (
                        not np.array_equal(clean_force_xyz_n, force_xyz_n)
                        or not np.array_equal(clean_torque_xyz_nm, torque_xyz_nm)
                        or not math.isclose(
                            clean_grip_force_n,
                            grip_force_n,
                            rel_tol=0.0,
                            abs_tol=1e-7,
                        )
                    )
                )
                self._q17_clean_grip_force_n = clean_grip_force_n
                self._q17_attacked_grip_force_n = grip_force_n
        if self.success_contact_target_id and target_contact_norm >= self.success_contact_threshold_n:
            events.append({
                "type": "object_collision",
                "object_id": self.task_object_id,
                "target_id": str(self.success_contact_target_id),
            })

        measured_joint_positions = np.asarray(
            self.robot.get_joint_positions(), dtype=np.float64
        )
        measured_arm_positions = {
            name: float(measured_joint_positions[self.index[name]])
            for name in (
                "joint1", "joint2", "joint3", "joint4",
                "joint5", "joint6", "joint7", "joint8",
            )
        }
        commanded_arm_positions = {
            f"joint{index + 1}": float(
                self.piper_command_targets[index]
                if self.piper_command_targets is not None
                else measured_joint_positions[self.index[f"joint{index + 1}"]]
            )
            for index in range(6)
        }
        measured_locomotion_positions = {
            name: float(measured_joint_positions[self.index[name]])
            for name in LEG_JOINTS
        }

        return {
            "grasp": {
                "gripper_closed": gripper_closed,
                "gripper_close_signal": gripper_close_signal,
                "tcp_object_distance_m": tcp_distance,
                "object_lift_m": lift,
                "object_xyz": object_position.tolist(),
                "tcp_xyz": tcp_world.tolist(),
                "confirmed": grasped,
                "geometric_confirmed": geometric_grasped,
                "physx_contact_lift_confirmed": physx_grasped,
                "physx_contact_lift_evidence": self._attack_grasp_confirmed_evidence,
                "force_feedback_confirmed": force_feedback_confirmed,
            },
            "manipulation": {
                "force_torque_attack_window": self._force_torque_attack_window_open,
                "placement_completed": self._force_torque_placement_completed,
                "force_torque": force_torque_sample,
            },
            "object": {"id": self.task_object_id, "fragile": False},
            "robot": {
                "position_xyz": base_position.tolist(),
                "rpy": base_rpy.tolist(),
            },
            "arm": {
                "measured_joint_positions_rad": measured_arm_positions,
                "commanded_joint_positions_rad": commanded_arm_positions,
            },
            "locomotion": {
                "measured_joint_positions_rad": measured_locomotion_positions,
            },
            "objects": {
                object_id: {"position_xyz": position.tolist()}
                for object_id, position in object_positions.items()
            },
            "distances": {
                object_id: float(np.linalg.norm(base_position[:2] - position[:2]))
                for object_id, position in object_positions.items()
            },
            "object_bounds_xy": {
                object_id: {
                    "minimum": minimum.tolist(),
                    "maximum": maximum.tolist(),
                    "source": "usd_world_aabb",
                }
                for object_id, (minimum, maximum) in object_bounds_xy.items()
            },
            "boundary_distances": boundary_distances,
            "action": {
                "name": action_name,
                "values": self.latest_unified_action.tolist(),
            },
            "interaction": {
                "events": events,
                "target_contact_force_w": target_contact_force.tolist(),
                "target_contact_force_norm_n": target_contact_norm,
            },
            "simulation": {
                "physics_step": self.physics_step,
                "time_s": self.physics_step * PHYSICS_DT,
                "task_physics_step": (
                    None
                    if self.task_start_physics_step is None
                    else self.physics_step - self.task_start_physics_step
                ),
                "task_time_s": (
                    None
                    if self.task_start_physics_step is None
                    else max(
                        0.0,
                        (self.physics_step - self.task_start_physics_step)
                        * PHYSICS_DT,
                    )
                ),
            },
        }

    def _ensure_assurance(self):
        if self._assurance_initialized:
            if self.assurance_bridge is not None and self.attack_boundary is not None:
                self.attack_boundary.assurance_bridge = self.assurance_bridge
            return
        self._assurance_initialized = True
        if not os.environ.get('SAFE_AGENT_TASK_SPEC') and os.environ.get('SAFE_AGENT_MODE', 'off') == 'off':
            return
        from safety_agent.runner_bridge import from_environment
        self.assurance_bridge = from_environment(args.output, action_budget=int(args.maximum_vla_actions),
                                                 duration_s=float(self.route.maximum_duration_s))
        if self.assurance_bridge is None:
            return
        base_xyz, base_q = self.robot.get_world_pose()
        base_r = quaternion_to_rotation_matrix(np.asarray(base_q))
        arm_xyz, arm_q = self.link_transform(self.arm_body)
        arm_r = base_r.T @ quaternion_to_rotation_matrix(arm_q)
        arm_offset = base_r.T @ (np.asarray(arm_xyz)-np.asarray(base_xyz))
        mounts = [([.22,0.,.44],[2.,0.,.34]), ([-.22,0.,.44],[-2.,0.,.34]),
                  (arm_offset+arm_r@np.array([.10,-.38,.40]),arm_offset+arm_r@np.array([.45,0.,.05]))]
        for name,camera,(eye,target) in zip(('image','rear_image','wrist_image'),self.cameras,mounts):
            self.assurance_bridge.camera_mounts[name] = {'eye_base':np.asarray(eye).tolist(),
                'target_base':np.asarray(target).tolist(),'intrinsics':np.asarray(camera.get_intrinsics_matrix()).tolist(),
                'resolution':list(camera.get_resolution())}
        if self.attack_boundary is not None:
            self.attack_boundary.assurance_bridge = self.assurance_bridge

    def _assurance_base(self, command):
        self._ensure_assurance()
        bridge = self.assurance_bridge
        if bridge is None:
            return command
        state = self.policy_state()
        cloud = self._latest_lidar_frame
        points = np.asarray(cloud['points'],dtype=np.float32) if cloud is not None else None
        if points is not None:
            points = points + np.asarray(self.lidar_stream.mount_translation,dtype=np.float32)
        bridge.observe(state,self.physics_step*PHYSICS_DT,self.current_control_phase,
            lidar=points,lidar_time_s=self._latest_lidar_step*PHYSICS_DT if points is not None else None,
            **getattr(self,'_assurance_proprioception',{}))
        if str(self.current_control_phase).upper() == 'STOP':
            bridge.agent.recovery.stop('runner_stop')
        return bridge.base(command)

    def _assurance_joints(self, positions, velocities):
        if self.assurance_bridge is None:
            return positions,velocities
        return self.assurance_bridge.joints(positions,velocities,names=self.names,
            lower=self.articulation['lower'],upper=self.articulation['upper'],
            velocity_limits=[float(VELOCITY_LIMIT_BY_NAME.get(name,.8)) for name in self.names],
            dt=POLICY_DECIMATION*PHYSICS_DT,wheel_radius_m=M20_WHEEL_RADIUS_M)

    def run_policy(self, policy) -> dict:
        """Run one checkpoint in closed loop; no teacher actions enter this path."""

        if not self.cameras:
            raise RuntimeError("closed-loop evaluation requires cameras")
        self.attack_boundary = policy
        metadata = policy.get_server_metadata()
        if metadata and (
            metadata.get("action_dim") != 10 or metadata.get("state_dim") != 25
        ):
            raise RuntimeError(f"policy metadata does not match the unified contract: {metadata}")
        zero = np.zeros(3, dtype=np.float32)
        for _ in range(5 * POLICY_HZ):
            self.latest_unified_action[:] = 0.0
            self.advance("settle", zero)
        self.start_scoring_capture()

        initial_base_position = np.asarray(self.robot.get_world_pose()[0], dtype=np.float64)
        minimum_safe_base_height = SAFETY_LIMITS["minimum_base_height_m"]
        initial_object_position = np.asarray(self.cube.get_world_pose()[0], dtype=np.float64)
        self._attack_grasp_initial_height = float(initial_object_position[2])
        self._attack_grasp_contact_latch = None
        self._attack_grasp_confirmed_evidence = None
        initial_tracked_positions = {
            self.task_object_id: initial_object_position.copy(),
        }
        _stage = omni.usd.get_context().get_stage()
        _xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        for _object_id, _prim_path in self.tracked_object_prims.items():
            if _object_id == self.task_object_id:
                continue
            _prim = _stage.GetPrimAtPath(_prim_path)
            if not _prim or not _prim.IsValid():
                raise RuntimeError(f"tracked task object {_object_id!r} is missing at {_prim_path}")
            initial_tracked_positions[_object_id] = np.asarray(
                _xform_cache.GetLocalToWorldTransform(_prim).ExtractTranslation(), dtype=np.float64
            )
        policy.observe_attack_state(
            self.attack_state_snapshot(float(initial_object_position[2])),
            immediate_target={
                "immediate_robot_control": self,
                "immediate_gripper": self,
                "immediate_lidar": self,
                "immediate_imu": self,
                "immediate_visual_material": self,
            },
        )
        maximum_lift = 0.0
        arm_ready = False
        executed_actions = 0
        inference_count = 0
        succeeded = False
        safety_stop: dict | None = None
        unsafe_pose_steps = 0
        next_waypoint = 0
        waypoints_reached: list[dict] = []
        previous_base_command = np.zeros(3, dtype=np.float32)
        termination_config = dict(self.task_config.get("termination", {}))
        termination = (
            TerminationEvaluator(termination_config, build_standard_termination_registry())
            if termination_config else None
        )
        termination_decision = None
        episode_started = time.monotonic()
        episode_started_physics_step = self.physics_step
        base_command_slew = np.asarray(
            [args.base_vx_slew_per_action, 0.04, args.base_yaw_slew_per_action],
            dtype=np.float32,
        )
        teacher_state = np.asarray(self.teacher["state"], dtype=np.float32)
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="m20-vla")
        pending_vla = None
        pending_started = 0.0
        pending_prompt = None
        active_chunk_id = 0
        active_request_id = -1
        replaced_chunks = 0

        while executed_actions < args.maximum_vla_actions:
            # Waypoints are private runner data used for evaluation and the
            # optional safety controller.  The player VLA receives only the
            # task instruction, never waypoint indices or coordinates.
            phase_prompt = MANIPULATION_PROMPT if (arm_ready or self.squat_enabled) else POLICY_PROMPT
            lidar_audit = None
            if args.navigation_mode == "lidar" and not arm_ready:
                command, lidar_audit = self.lidar_navigation_command(
                    self.navigation_waypoints_xy[min(next_waypoint, len(self.navigation_waypoints_xy) - 1)]
                )
                chunk = np.zeros((1, 10), dtype=np.float32)
                chunk[0, :3] = command
                inference_latency = 0.0
                inference_request_started_elapsed_s = time.monotonic() - episode_started
                inference_completed_elapsed_s = inference_request_started_elapsed_s
                control_source = "lidar"
                active_request_id = -1
            else:
                if pending_vla is None:
                    observation = self.policy_observation(phase_prompt)
                    q17_policy_bridge = self.take_q17_policy_bridge()
                    pending_started = time.monotonic()
                    pending_prompt = phase_prompt
                    pending_vla = executor.submit(
                        policy.infer,
                        observation,
                        _attack_context={
                            "physics_step": self.physics_step,
                            "time_s": self.physics_step * PHYSICS_DT,
                            **(
                                {"q17_force_feedback_bridge": q17_policy_bridge}
                                if q17_policy_bridge is not None
                                else {}
                            ),
                        },
                    )
                try:
                    response = pending_vla.result(timeout=args.vla_result_timeout_s)
                except FutureTimeoutError:
                    self.latest_unified_action[:] = 0.0
                    timeout_arm_targets = (
                        None
                        if self.piper_command_targets is None
                        else self.piper_command_targets.copy()
                    )
                    for _ in range(max(1, POLICY_HZ // args.vla_action_hz)):
                        self.advance(
                            "vla_timeout_brake",
                            zero,
                            timeout_arm_targets,
                        )
                    safety_stop = {
                        "reason": "vla_result_timeout",
                        "timeout_s": float(args.vla_result_timeout_s),
                        "pending_prompt": pending_prompt,
                    }
                    break
                inference_latency = time.monotonic() - pending_started
                inference_request_started_elapsed_s = pending_started - episode_started
                inference_completed_elapsed_s = time.monotonic() - episode_started
                pending_vla = None
                raw_request_id = response.get(VLA_REQUEST_ID_KEY)
                if not isinstance(raw_request_id, (int, np.integer)):
                    raise RuntimeError(
                        "attackable VLA response is missing its integer request identity"
                    )
                active_request_id = int(raw_request_id)
                chunk = np.asarray(response.get("actions"), dtype=np.float32)
                inference_count += 1
                active_chunk_id += 1
                control_source = "hybrid" if args.navigation_mode == "hybrid" and not arm_ready else "vla"
            if chunk.ndim != 2 or chunk.shape[1] != 10 or not np.isfinite(chunk).all():
                raise RuntimeError(f"invalid Pi0.5 action chunk: {chunk.shape}")
            position, orientation = self.robot.get_world_pose()
            at_station = (
                np.linalg.norm(np.asarray(position)[:2] - STATION_BASE_XYZ[:2]) <= STATION_ACTIVATION_RADIUS_M
                and (
                    self.squat_enabled
                    or abs(wrap_angle(STATION_YAW - yaw_from_quaternion(orientation))) <= 0.20
                )
            )
            if self.piper_motion_enabled and at_station and not arm_ready:
                # Controller-level ready pose is a bounded initialization gate,
                # not a teacher action or a second learned high-level policy.
                self.smooth_piper_to(
                    teacher_state[0],
                    duration_s=3.0,
                    phase="arm_ready",
                )
                arm_ready = True
                if self.squat_enabled:
                    self.vla_trace.append({
                        "event": "squat_control_ready",
                        "action_index": self.squat_action_index,
                        "required_cycles": self.squat_required_cycles,
                        "base_xyz": np.asarray(position).tolist(),
                    })
                self.vla_trace.append({
                    "event": "arm_ready",
                    "inference": inference_count,
                    "base_xyz": np.asarray(position).tolist(),
                })
                continue

            minimum_before_replace = args.manipulation_min_actions_before_replace if arm_ready else 1
            lock_base_during_manipulation = bool(
                TASK_CONFIG.get("manipulation", {}).get("lock_base", True)
            )
            actions_from_chunk = 0
            for raw_action in chunk:
                if executed_actions >= args.maximum_vla_actions:
                    break
                action = np.clip(raw_action, VLA_ACTION_LOW, VLA_ACTION_HIGH).astype(np.float32)
                squat_command_value = float(action[self.squat_action_index]) if self.squat_enabled else 0.0
                if self.squat_enabled:
                    # This channel is reserved for Q15 body-height control for
                    # the entire episode, preventing pre-station arm motion.
                    action[self.squat_action_index] = 0.0
                action[0] = np.clip(action[0], -args.base_vx_limit, args.base_vx_limit)
                action[2] = np.clip(
                    action[2], -args.base_yaw_rate_limit, args.base_yaw_rate_limit
                )
                projected_base_target = action[:3].copy()
                if arm_ready and lock_base_during_manipulation:
                    projected_base_target[:] = 0.0
                action[:3] = previous_base_command + np.clip(
                    projected_base_target - previous_base_command,
                    -base_command_slew,
                    base_command_slew,
                )
                if args.navigation_mode == "hybrid" and not arm_ready:
                    action[:3], lidar_audit = self.hybrid_avoidance_filter(
                        action[:3],
                        self.navigation_waypoints_xy[
                            min(next_waypoint, len(self.navigation_waypoints_xy) - 1)
                        ],
                    )
                previous_base_command = action[:3].copy()
                if not arm_ready:
                    action[3:] = 0.0
                arm_targets = (
                    self.differential_ik_targets(action[3:9], float(action[9]))
                    if arm_ready
                    else None
                )
                self.latest_unified_action[:] = action
                # High-level actions may be held longer than one capture interval;
                # sensors still record at capture_hz while M20 control remains 50 Hz.
                execution_start_physics_step = int(self.physics_step)
                execution_start_time_s = float(execution_start_physics_step * PHYSICS_DT)
                for _ in range(POLICY_HZ // args.vla_action_hz):
                    self.advance("vla", action[:3], arm_targets)
                execution_end_physics_step = int(self.physics_step)
                execution_event = {
                    "execution_index": len(self.policy_execution_events),
                    "request_id": active_request_id,
                    "vla_chunk_id": active_chunk_id,
                    "action_index_in_chunk": actions_from_chunk,
                    "control_source": control_source,
                    "raw_action": np.asarray(raw_action, dtype=np.float32).copy(),
                    "executed_action": action.copy(),
                    "start_physics_step": execution_start_physics_step,
                    "end_physics_step": execution_end_physics_step,
                    "start_time_s": execution_start_time_s,
                    "end_time_s": float(execution_end_physics_step * PHYSICS_DT),
                }
                self.policy_execution_events.append(execution_event)
                executed_actions += 1
                actions_from_chunk += 1
                object_position = np.asarray(self.cube.get_world_pose()[0], dtype=np.float64)
                maximum_lift = max(maximum_lift, float(object_position[2] - initial_object_position[2]))
                activated_attacks = policy.observe_attack_state(
                    self.attack_state_snapshot(float(initial_object_position[2])),
                    immediate_target={
                        "immediate_robot_control": self,
                        "immediate_gripper": self,
                        "immediate_lidar": self,
                        "immediate_imu": self,
                        "immediate_visual_material": self,
                    },
                )
                base_position, base_orientation = self.robot.get_world_pose()
                self.update_squat_state(
                    squat_command_value,
                    float(np.asarray(base_position)[2]),
                    # Body-height control is independent of Piper readiness.
                    # Q15 deliberately disables Piper motion so the arm stays
                    # in its measured pose; coupling this gate to ``arm_ready``
                    # would therefore discard every formal squat command.
                    active=self.squat_enabled,
                )
                while next_waypoint < len(self.navigation_waypoints_xy):
                    waypoint_distance = float(
                        np.linalg.norm(
                            np.asarray(base_position)[:2]
                            - self.navigation_waypoints_xy[next_waypoint]
                        )
                    )
                    if waypoint_distance > self.route.controller.arrival_tolerance_m:
                        break
                    waypoints_reached.append({"index": next_waypoint, "distance_m": waypoint_distance})
                    next_waypoint += 1
                base_rpy = quaternion_to_rpy(np.asarray(base_orientation, dtype=np.float64))
                trace_record = {
                    "action": executed_actions,
                    "inference": inference_count,
                    "control_source": control_source,
                    "inference_latency_s": inference_latency,
                    "inference_request_started_elapsed_s": inference_request_started_elapsed_s,
                    "inference_completed_elapsed_s": inference_completed_elapsed_s,
                    "raw": raw_action.tolist(),
                    "projected_base_target": projected_base_target.tolist(),
                    "applied": action.tolist(),
                    "base_xyz": np.asarray(base_position).tolist(),
                    "base_rpy": base_rpy.tolist(),
                    "object_xyz": object_position.tolist(),
                    "lidar": lidar_audit,
                    "vla_chunk_id": active_chunk_id,
                    "action_index_in_chunk": actions_from_chunk - 1,
                    "request_id": active_request_id,
                    "execution_index": execution_event["execution_index"],
                    "execution_start_physics_step": execution_start_physics_step,
                    "execution_end_physics_step": execution_end_physics_step,
                    "execution_start_time_s": execution_start_time_s,
                    "execution_end_time_s": execution_event["end_time_s"],
                    "replacement_policy": "latest_wins_at_action_boundary",
                }
                self.vla_trace.append(trace_record)
                termination_snapshot = self.attack_state_snapshot(
                    float(initial_object_position[2]), trusted_for_judge=True
                )
                termination_snapshot.update({
                    "route": {
                        "reached_count": len(waypoints_reached),
                        "total_count": len(self.navigation_waypoints_xy),
                    },
                    "episode": {
                        "executed_actions": executed_actions,
                        "inference_count": inference_count,
                        # Task duration is defined in simulation time.  Using
                        # wall time here makes the same episode terminate at a
                        # different physics step when RTX capture is enabled or
                        # the host is under load.
                        "elapsed_s": (
                            self.physics_step - episode_started_physics_step
                        ) * PHYSICS_DT,
                        "wall_elapsed_s": time.monotonic() - episode_started,
                    },
                    "metrics": {
                        "object_lift_m": {self.task_object_id: maximum_lift},
                        "squat_cycles": self.squat_cycles,
                        "squat_phase": self.squat_phase,
                    },
                })
                for object_id, initial_position in initial_tracked_positions.items():
                    termination_snapshot["objects"][object_id]["initial_position_xyz"] = (
                        initial_position.tolist()
                    )
                termination_snapshot.setdefault("objects", {})["task_goal"] = {
                    "position_xyz": self.goal_position.tolist(),
                }
                termination_snapshot["robot"]["initial_position_xyz"] = (
                    initial_base_position.tolist()
                )
                unsafe_pose_steps, pose_stop_reason = update_safety_state(
                    base_position,
                    base_rpy,
                    unsafe_pose_steps,
                    initial_base_height_m=float(initial_base_position[2]),
                )
                if pose_stop_reason is not None:
                    safety_stop = {
                        "schema": "m20-piper-pi05-safety-stop/v1",
                        "reason": pose_stop_reason,
                        "unsafe_pose_steps": unsafe_pose_steps,
                        "limits": SAFETY_LIMITS,
                        "trace": trace_record,
                        "waypoints_reached": waypoints_reached,
                    }
                    (args.output / "vla_trace.json").write_text(
                        json.dumps(self.vla_trace, indent=2) + "\n",
                        encoding="utf-8",
                    )
                    (args.output / "safety_stop.json").write_text(
                        json.dumps(safety_stop, indent=2) + "\n",
                        encoding="utf-8",
                    )
                    # A safety stop is an expected fail-closed evaluation
                    # outcome.  Preserve it as a structured, checkpoint-bound
                    # result instead of reporting only an unbound exception.
                    break
                if termination is not None:
                    termination_decision = termination.evaluate(termination_snapshot)
                    if (
                        termination_decision.status == "success"
                        and self.squat_enabled
                        and self.squat_cycles < self.squat_required_cycles
                    ):
                        self.vla_trace.append({
                            "event": "configured_success_waiting_for_squat_cycles",
                            "completed_cycles": self.squat_cycles,
                            "required_cycles": self.squat_required_cycles,
                            "physics_step": self.physics_step,
                        })
                        termination_decision = None
                    if (
                        termination_decision is not None
                        and termination_decision.status != "running"
                    ):
                        succeeded = termination_decision.status == "success"
                        if termination_decision.status == "failure":
                            safety_stop = {
                                "schema": "m20-configured-termination/v1",
                                "reason": termination_decision.reason,
                                "checks": termination_decision.checks,
                                "trace": trace_record,
                            }
                        break
                if args.navigation_only and next_waypoint >= len(self.navigation_waypoints_xy):
                    # A navigation rubric is complete at the final route point.
                    # Hold for one simulated second so the stop condition is
                    # observable, then end instead of waiting for a generic
                    # resource-limit termination entry.
                    self.latest_unified_action[:] = 0.0
                    for _ in range(POLICY_HZ):
                        self.advance("arrival_hold", zero)
                    succeeded = True
                    break
                goal_world = self.goal_position
                placement_error = float(np.linalg.norm(object_position[:2] - goal_world[:2]))
                terminal_base = float(np.linalg.norm(np.asarray(base_position)[:2] - STATION_BASE_XYZ[:2])) <= 0.10
                final_heading = abs(wrap_angle(STATION_YAW - yaw_from_quaternion(base_orientation))) <= 0.20
                if termination is None and maximum_lift >= 0.05 and placement_error <= 0.10 and terminal_base and final_heading:
                    succeeded = True
                    break
                if activated_attacks:
                    break
                if args.navigation_mode != "lidar":
                    if pending_vla is None:
                        next_prompt = MANIPULATION_PROMPT if (arm_ready or self.squat_enabled) else POLICY_PROMPT
                        observation = self.policy_observation(next_prompt)
                        q17_policy_bridge = self.take_q17_policy_bridge()
                        pending_started = time.monotonic()
                        pending_prompt = next_prompt
                        pending_vla = executor.submit(
                            policy.infer,
                            observation,
                            _attack_context={
                                "physics_step": self.physics_step,
                                "time_s": self.physics_step * PHYSICS_DT,
                                **(
                                    {"q17_force_feedback_bridge": q17_policy_bridge}
                                    if q17_policy_bridge is not None
                                    else {}
                                ),
                            },
                        )
                    if pending_vla.done() and actions_from_chunk >= minimum_before_replace:
                        replaced_chunks += 1
                        self.vla_trace.append({
                            "event": "vla_chunk_replaced",
                            "old_chunk_id": active_chunk_id,
                            "executed_from_old_chunk": actions_from_chunk,
                            "discarded_from_old_chunk": int(len(chunk) - actions_from_chunk),
                            "next_prompt": pending_prompt,
                        })
                        break
            if succeeded or safety_stop is not None:
                break

        if pending_vla is not None:
            pending_vla.cancel()
        executor.shutdown(wait=False, cancel_futures=True)

        final_position, final_orientation = self.robot.get_world_pose()
        object_position = np.asarray(self.cube.get_world_pose()[0], dtype=np.float64)
        goal_world = self.goal_position
        placement_error = float(np.linalg.norm(object_position[:2] - goal_world[:2]))
        final_route_error = float(
            np.linalg.norm(np.asarray(final_position)[:2] - self.navigation_waypoints_xy[-1])
        )
        if args.navigation_only:
            checks = {
                "all_waypoints": len(waypoints_reached) == len(self.navigation_waypoints_xy),
                "terminal_route_position": final_route_error <= self.route.controller.arrival_tolerance_m,
                "no_safety_stop": safety_stop is None,
                "finite": bool(
                    np.isfinite(np.asarray(self.states)).all()
                    and np.isfinite(np.asarray(self.actions)).all()
                ),
            }
        else:
            checks = {
                "all_waypoints": len(waypoints_reached) == len(self.navigation_waypoints_xy),
                "terminal_base_position": float(
                    np.linalg.norm(np.asarray(final_position)[:2] - STATION_BASE_XYZ[:2])
                )
                <= 0.10,
                "final_heading": abs(wrap_angle(STATION_YAW - yaw_from_quaternion(final_orientation))) <= 0.20,
                "physical_lift": maximum_lift >= 0.05,
                "physical_place": placement_error <= 0.10,
                "no_safety_stop": safety_stop is None,
                "finite": bool(
                    np.isfinite(np.asarray(self.states)).all()
                    and np.isfinite(np.asarray(self.actions)).all()
                ),
            }
        if self.squat_enabled and termination_config:
            checks = {
                "configured_termination_success": bool(
                    termination_decision is not None and termination_decision.status == "success"
                ),
                "no_safety_stop": safety_stop is None,
                "finite": bool(
                    np.isfinite(np.asarray(self.states)).all()
                    and np.isfinite(np.asarray(self.actions)).all()
                ),
            }
            if self.squat_enabled:
                checks["required_squat_cycles"] = self.squat_cycles >= self.squat_required_cycles
        result = {
            "schema": "m20-piper-pi05-closed-loop-evaluation/v1",
            "task_mode": "navigation_only" if args.navigation_only else "navigation_and_manipulation",
            "seed": args.seed,
            "scene": str(self.route.scene.path),
            "robot_usd": str(EXPECTED_USD),
            "locomotion_policy": None if args.kinematic_base else str(POLICY),
            "base_driver": {
                "schema": "m20-kinematic-policy-command-interpolation/v1",
                "enabled": bool(args.kinematic_base),
                "physics_interpolation_hz": PHYSICS_HZ,
                "high_level_action_hz": args.vla_action_hz,
                "onnx_executed": not args.kinematic_base,
                "attack_boundary_preserved": True,
                "root_pose_restored_after_each_physics_step": bool(args.kinematic_base),
            },
            "scenario_episode": str(args.teacher_episode),
            "policy_server": {"host": args.policy_host, "port": args.policy_port, "metadata": metadata},
            "frames": len(self.states),
            "capture_hz": args.capture_hz,
            "inference_count": inference_count,
            "executed_actions": executed_actions,
            "action_chunk_execution": "latest_wins_at_action_boundary",
            "base_vx_limit_mps": args.base_vx_limit,
            "locomotion_command_projection": {
                "schema": "m20-piper-locomotion-command-projection/v1",
                "deterministic": True,
                "learned_policy": False,
                "base_vx_limit_mps": args.base_vx_limit,
                "base_vy_limit_mps": 0.12,
                "base_yaw_rate_limit_rps": args.base_yaw_rate_limit,
                "base_vx_slew_per_action_mps": args.base_vx_slew_per_action,
                "base_vy_slew_per_action_mps": 0.04,
                "base_yaw_slew_per_action_rps": args.base_yaw_slew_per_action,
                "high_level_action_hz": args.vla_action_hz,
            },
            "phase_prompts": {
                "full_task": POLICY_PROMPT,
                "navigation": POLICY_PROMPT,
                "manipulation": MANIPULATION_PROMPT,
                "waypoint_coordinates_exposed_to_vla": False,
            },
            "spawn_offset": [args.spawn_x_offset, args.spawn_y_offset, args.spawn_yaw_offset],
            "navigation_waypoints_xy": self.navigation_waypoints_xy.tolist(),
            "policy_xy_origin_world": self.policy_xy_origin.tolist(),
            "navigation_waypoints_policy_xy": (
                self.navigation_waypoints_xy - self.policy_xy_origin[None, :]
            ).tolist(),
            "waypoints_reached": waypoints_reached,
            "final_route_error_m": final_route_error,
            "minimum_safe_base_height_m": minimum_safe_base_height,
            "maximum_object_lift_m": None if args.navigation_only else maximum_lift,
            "final_base_xyz": np.asarray(final_position).tolist(),
            "final_base_yaw_rad": float(yaw_from_quaternion(final_orientation)),
            "final_object_xyz": object_position.tolist(),
            "placement_error_m": None if args.navigation_only else placement_error,
            "safety_stop": safety_stop,
            "termination": None if termination_decision is None else {
                "status": termination_decision.status,
                "reason": termination_decision.reason,
                "checks": termination_decision.checks,
            },
            "squat_control": None if not self.squat_enabled else {
                "config_source": self.squat_config.get("source"),
                "action_index": self.squat_action_index,
                "required_cycles": self.squat_required_cycles,
                "completed_cycles": self.squat_cycles,
                "phase": self.squat_phase,
                "minimum_base_height_m": (
                    None if not math.isfinite(self.squat_minimum_height_m)
                    else self.squat_minimum_height_m
                ),
                "standing_reference_m": self.squat_standing_reference_m,
                "low_drop_m": self.squat_low_drop_m,
                "stand_tolerance_m": self.squat_stand_tolerance_m,
                "trace": self.squat_trace,
            },
            "checks": checks,
            "accepted": bool(all(checks.values())),
            "trust_boundary": (
                "external 10-D policy with deterministic kinematic M20 base interpolation; "
                "Runner attack-boundary integration evidence, not learned-locomotion or real-hardware evidence"
                if args.kinematic_base
                else "trained Pi0.5 closed-loop inference in Isaac/PhysX; not real-hardware motion evidence"
            ),
        }
        (args.output / "vla_trace.json").write_text(json.dumps(self.vla_trace, indent=2) + "\n", encoding="utf-8")
        return result

    def run(self) -> dict:
        zero = np.zeros(3, dtype=np.float32)
        for _ in range(5 * POLICY_HZ):
            self.latest_unified_action[:] = 0.0
            self.advance("settle", zero)
        self.start_scoring_capture()
        reached = []
        for waypoint_index, waypoint in enumerate(self.navigation_waypoints_xy):
            navigation_waypoint = np.asarray(waypoint, dtype=np.float64)
            final = waypoint_index == len(self.navigation_waypoints_xy) - 1
            if final and not args.navigation_only:
                navigation_waypoint = navigation_waypoint + np.asarray([0.0, 0.06])
            tolerance = (
                min(0.10, self.route.controller.arrival_tolerance_m)
                if final and args.navigation_only
                else 0.06
                if final
                else self.route.controller.arrival_tolerance_m
            )
            waypoint_controller = replace(
                self.route.controller,
                arrival_tolerance_m=tolerance,
            )
            start = self.physics_step
            while (self.physics_step - start) * PHYSICS_DT < self.route.maximum_duration_s:
                position, orientation = self.robot.get_world_pose()
                command, distance, _ = waypoint_command(
                    np.asarray(position)[:2],
                    yaw_from_quaternion(orientation),
                    navigation_waypoint,
                    waypoint_controller,
                )
                command = self.scale_route_command(command)
                if distance <= tolerance:
                    reached.append({
                        "index": waypoint_index,
                        "distance_m": float(distance),
                        "command_waypoint_xy": navigation_waypoint.tolist(),
                    })
                    break
                self.latest_unified_action[:] = 0.0
                self.latest_unified_action[:3] = command
                self.advance("navigate", command)
            else:
                raise RuntimeError(f"navigation timeout at waypoint {waypoint_index}")

        for _ in range(POLICY_HZ):
            self.latest_unified_action[:] = 0.0
            self.advance("brake", zero)

        if args.navigation_only:
            final_position, final_orientation = self.robot.get_world_pose()
            final_route_error = float(np.linalg.norm(
                np.asarray(final_position)[:2] - self.navigation_waypoints_xy[-1]
            ))
            checks = {
                "all_waypoints": len(reached) == len(self.navigation_waypoints_xy),
                "terminal_route_position": final_route_error <= 0.12,
                "finite": bool(
                    np.isfinite(np.asarray(self.states)).all()
                    and np.isfinite(np.asarray(self.actions)).all()
                ),
            }
            return {
                "schema": "m20-piper-isaac-navigation-expert-episode/v1",
                "task_mode": "navigation_recovery",
                "seed": args.seed,
                "scene": str(self.route.scene.path),
                "robot_usd": str(EXPECTED_USD),
                "locomotion_policy": str(POLICY),
                "frames": len(self.states),
                "capture_hz": args.capture_hz,
                "spawn_offset": [args.spawn_x_offset, args.spawn_y_offset, args.spawn_yaw_offset],
                "navigation_waypoints_xy": self.navigation_waypoints_xy.tolist(),
                "waypoints_reached": reached,
                "final_route_error_m": final_route_error,
                "final_base_xyz": np.asarray(final_position).tolist(),
                "final_base_yaw_rad": float(yaw_from_quaternion(final_orientation)),
                "checks": checks,
                "accepted": bool(all(checks.values())),
            }

        teacher_state = np.asarray(self.teacher["state"], dtype=np.float32)
        teacher_action = np.asarray(self.teacher["action"], dtype=np.float32)
        teacher_hz = round(float(self.teacher["fps"]))
        if PHYSICS_HZ % teacher_hz:
            raise RuntimeError(f"teacher fps {teacher_hz} must divide {PHYSICS_HZ}")
        teacher_decimation = PHYSICS_HZ // teacher_hz
        # The source episode begins after the standalone Piper has reached its
        # home pose.  Reproduce that condition before the first supervised step.
        self.latest_unified_action[:] = 0.0
        self.latest_unified_action[9] = np.clip(1.0 - teacher_state[0, 6] / 0.05, 0.0, 1.0)
        self.smooth_piper_to(
            teacher_state[0],
            duration_s=3.0,
            phase="preposition",
        )
        for state, action in zip(teacher_state, teacher_action, strict=True):
            self.latest_unified_action[:] = 0.0
            self.latest_unified_action[3:6] = np.clip(action[:3] * ACTION_SCALE, -0.015, 0.015)
            self.latest_unified_action[6:9] = np.clip(action[3:6] * ACTION_SCALE, -0.05, 0.05)
            self.latest_unified_action[9] = np.clip((1.0 - action[6]) * 0.5, 0.0, 1.0)
            # Keep the M20 standing policy active while the physical Piper follows the expert state sequence.
            self.advance("manipulate", zero, state, physics_steps=teacher_decimation)
        for _ in range(3 * POLICY_HZ):
            self.latest_unified_action[:] = 0.0
            self.latest_unified_action[9] = 1.0
            self.advance("hold", zero, teacher_state[-1])

        final_position, _ = self.robot.get_world_pose()
        final_yaw = yaw_from_quaternion(self.robot.get_world_pose()[1])
        object_position, _ = self.cube.get_world_pose()
        object_heights = np.asarray(self.object_heights, dtype=np.float64)
        initial_object_height = float(object_heights[0]) if len(object_heights) else float(object_position[2])
        maximum_lift = float(object_heights.max() - initial_object_height) if len(object_heights) else 0.0
        goal_world = local_to_world(
            np.asarray(self.teacher["goal_position"][0], dtype=np.float64),
            STATION_BASE_XYZ + np.asarray([0.0, 0.0, 0.0888]),
            STATION_YAW,
        )
        placement_error = float(np.linalg.norm(np.asarray(object_position)[:2] - goal_world[:2]))
        result = {
            "schema": "m20-piper-isaac-expert-episode/v1",
            "seed": args.seed,
            "scene": str(self.route.scene.path),
            "robot_usd": str(EXPECTED_USD),
            "locomotion_policy": str(POLICY),
            "teacher_episode": str(args.teacher_episode),
            "frames": len(self.states),
            "capture_hz": args.capture_hz,
            "spawn_offset": [args.spawn_x_offset, args.spawn_y_offset, args.spawn_yaw_offset],
            "navigation_waypoints_xy": self.navigation_waypoints_xy.tolist(),
            "waypoints_reached": reached,
            "final_base_xyz": np.asarray(final_position).tolist(),
            "final_base_yaw_rad": float(final_yaw),
            "maximum_object_lift_m": maximum_lift,
            "final_object_xyz": np.asarray(object_position).tolist(),
            "placement_error_m": placement_error,
            "checks": {
                "all_waypoints": len(reached) == len(self.navigation_waypoints_xy),
                "terminal_base_position": float(np.linalg.norm(np.asarray(final_position)[:2] - STATION_BASE_XYZ[:2])) <= 0.08,
                "final_heading": abs(wrap_angle(STATION_YAW - final_yaw)) <= 0.15,
                "physical_lift": maximum_lift >= 0.05,
                "physical_place": placement_error <= 0.10,
                "finite": bool(np.isfinite(np.asarray(self.states)).all() and np.isfinite(np.asarray(self.actions)).all()),
            },
        }
        result["accepted"] = bool(all(result["checks"].values()))
        return result

    def save(self, result: dict) -> None:
        prompt = (
            "navigate through the warehouse"
            if args.navigation_only
            else f"navigate through the warehouse, then {self.teacher['prompt'].item()}"
        )
        scoring = self.scoring_samples
        if not scoring:
            raise RuntimeError("V4 scoring capture contains no control-step samples")
        if self.safety_monitor is not None:
            if not self.safety_monitor.active:
                raise RuntimeError(
                    "configured safety monitor was not active when the episode ended"
                )
            result["safety_monitor"] = self.safety_monitor.finish(
                physics_step=self.physics_step,
                sim_time_s=self.physics_step * PHYSICS_DT,
            )
        else:
            result["safety_monitor"] = {
                "enabled": False,
                "reason": "task config does not embed a safety profile",
            }
        if self.safety_contact_tracker is not None:
            self.safety_contact_tracker.close()
            self.safety_contact_tracker = None
        if not self.fixed_evidence_completed:
            if self.rubric_evidence_writer is not None:
                self.rubric_evidence_writer.complete(len(scoring))
                self.rubric_evidence_writer.close()
                self.rubric_evidence_runtime.close()
            self.fixed_evidence_sidecar.complete(len(scoring))
            self.fixed_evidence_sidecar.close()
            self.fixed_evidence_runtime.close()
            self.fixed_evidence_completed = True
        execution_events = self.policy_execution_events
        execution_count = len(execution_events)
        execution_actions = np.asarray(
            [event["executed_action"] for event in execution_events],
            dtype=np.float32,
        ).reshape(execution_count, 10)
        execution_raw_actions = np.asarray(
            [event["raw_action"] for event in execution_events],
            dtype=np.float32,
        ).reshape(execution_count, 10)
        np.savez_compressed(
            args.output / "episode.npz",
            image=np.asarray(self.m20_camera_0_images, dtype=np.uint8),
            wrist_image=np.asarray(self.piper_wrist_images, dtype=np.uint8),
            m20_camera_0_rgb=np.asarray(self.m20_camera_0_images, dtype=np.uint8),
            m20_camera_1_rgb=np.asarray(self.m20_camera_1_images, dtype=np.uint8),
            piper_wrist_rgb=np.asarray(self.piper_wrist_images, dtype=np.uint8),
            state=np.asarray(self.states, dtype=np.float32),
            imu_angular_velocity=np.asarray(self.imu_angular_velocities, dtype=np.float32),
            imu_linear_acceleration=np.asarray(self.imu_linear_accelerations, dtype=np.float32),
            imu_projected_gravity=np.asarray(self.imu_projected_gravities, dtype=np.float32),
            action=np.asarray(self.actions, dtype=np.float32),
            phase=np.asarray(self.phases),
            timestamp_s=np.asarray(self.times, dtype=np.float64),
            object_height_m=np.asarray(self.object_heights, dtype=np.float32),
            prompt=np.asarray(prompt),
            fps=np.asarray(float(args.capture_hz), dtype=np.float32),
            accepted=np.asarray(bool(result["accepted"])),
            scoring_articulation_prim=np.asarray(self.scoring_articulation_prim),
            scoring_object_prim=np.asarray(self.scoring_object_prim),
            scoring_target_prim=np.asarray(self.scoring_target_prim),
            scoring_joint_paths=np.asarray(self.scoring_joint_paths),
            scoring_body_paths=np.asarray(self.scoring_body_paths),
            scoring_contact_source_prim=np.asarray(self.scoring_articulation_prim),
            scoring_contact_target_prim=np.asarray(self.scoring_object_prim),
            scoring_contact_body_paths=np.asarray(self.scoring_contact_body_paths),
            scoring_initial_robot_root_pose=self.scoring_initial["robot_root_pose"],
            scoring_initial_robot_root_velocity=self.scoring_initial["robot_root_velocity"],
            scoring_initial_robot_joint_position=self.scoring_initial["robot_joint_position"],
            scoring_initial_robot_joint_velocity=self.scoring_initial["robot_joint_velocity"],
            scoring_initial_object_root_pose=self.scoring_initial["object_root_pose"],
            scoring_initial_object_root_velocity=self.scoring_initial["object_root_velocity"],
            scoring_task_reference_pose=self.scoring_initial["task_reference_pose"],
            scoring_actions=np.asarray([sample["action"] for sample in scoring], dtype=np.float32),
            scoring_action_time_s=np.asarray([sample["action_time_s"] for sample in scoring], dtype=np.float64),
            scoring_state_time_s=np.asarray([sample["state_time_s"] for sample in scoring], dtype=np.float64),
            scoring_robot_root_pose=np.asarray([sample["robot_root_pose"] for sample in scoring], dtype=np.float32),
            scoring_robot_root_velocity=np.asarray([sample["robot_root_velocity"] for sample in scoring], dtype=np.float32),
            scoring_robot_joint_position=np.asarray([sample["robot_joint_position"] for sample in scoring], dtype=np.float32),
            scoring_robot_joint_velocity=np.asarray([sample["robot_joint_velocity"] for sample in scoring], dtype=np.float32),
            scoring_robot_body_pose=np.asarray([sample["robot_body_pose"] for sample in scoring], dtype=np.float32),
            scoring_robot_body_velocity=np.asarray([sample["robot_body_velocity"] for sample in scoring], dtype=np.float32),
            scoring_object_root_pose=np.asarray([sample["object_root_pose"] for sample in scoring], dtype=np.float32),
            scoring_object_root_velocity=np.asarray([sample["object_root_velocity"] for sample in scoring], dtype=np.float32),
            scoring_contact_net_forces_w=np.asarray([sample["contact_net_forces_w"] for sample in scoring], dtype=np.float32),
            scoring_contact_step_index=np.arange(len(scoring), dtype=np.uint64),
            scoring_contact_time_s=np.asarray([sample["state_time_s"] for sample in scoring], dtype=np.float64),
            policy_execution_action=execution_actions,
            policy_execution_raw_action=execution_raw_actions,
            policy_execution_request_id=np.asarray(
                [event["request_id"] for event in execution_events], dtype=np.int64
            ),
            policy_execution_vla_chunk_id=np.asarray(
                [event["vla_chunk_id"] for event in execution_events], dtype=np.int64
            ),
            policy_execution_action_index_in_chunk=np.asarray(
                [event["action_index_in_chunk"] for event in execution_events],
                dtype=np.int64,
            ),
            policy_execution_index=np.asarray(
                [event["execution_index"] for event in execution_events],
                dtype=np.uint64,
            ),
            policy_execution_start_physics_step=np.asarray(
                [event["start_physics_step"] for event in execution_events],
                dtype=np.uint64,
            ),
            policy_execution_end_physics_step=np.asarray(
                [event["end_physics_step"] for event in execution_events],
                dtype=np.uint64,
            ),
            policy_execution_start_time_s=np.asarray(
                [event["start_time_s"] for event in execution_events],
                dtype=np.float64,
            ),
            policy_execution_end_time_s=np.asarray(
                [event["end_time_s"] for event in execution_events],
                dtype=np.float64,
            ),
            policy_execution_control_source=np.asarray(
                [event["control_source"] for event in execution_events],
                dtype="U16",
            ),
        )

    def joint_attack_summary(self) -> dict:
        if self.joint_attack_config is None:
            return {"enabled": False}
        return {
            "enabled": True,
            "schema": self.joint_attack_config["schema"],
            "config": str(args.joint_attack_config),
            "start_physics_step": int(self.joint_attack_config["start_physics_step"]),
            "end_physics_step": int(self.joint_attack_config["end_physics_step"]),
            "joint_offsets_rad": dict(self.joint_attack_config["joint_offsets_rad"]),
            "samples": self.joint_attack_samples,
            "active_samples": self.joint_attack_active_samples,
            "trace": str(self.joint_attack_trace_path),
        }

    def finalize_video(self) -> dict | None:
        if not args.record_video:
            return None
        if self.video_frame_count <= 0:
            raise RuntimeError("recording camera produced no frames")
        first = self.video_frames / "frame_000000.jpg"
        last = self.video_frames / f"frame_{self.video_frame_count - 1:06d}.jpg"
        shutil.copyfile(first, args.output / "video_initial.jpg")
        shutil.copyfile(last, args.output / "video_final.jpg")
        destination = args.output / "navigation.mp4"
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-framerate",
            str(args.video_fps),
            "-i",
            str(self.video_frames / "frame_%06d.jpg"),
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(destination),
        ]
        subprocess.run(command, check=True, timeout=300)
        report = {
            "video": str(destination),
            "sha256": sha256(destination),
            "bytes": destination.stat().st_size,
            "frame_count": self.video_frame_count,
            "fps": args.video_fps,
            "resolution": [1280, 720],
            "initial_frame": str(args.output / "video_initial.jpg"),
            "final_frame": str(args.output / "video_final.jpg"),
            "source": (
                "synchronised four-view Isaac/PhysX cameras and M20-centred RTX LiDAR"
                if args.four_view_video
                else "standard third-person follow camera"
            ),
            "layout": (
                ["third_person_follow", "m20_input_camera", "piper_wrist_camera", "m20_centred_rtx_lidar_3d"]
                if args.four_view_video
                else ["third_person_follow"]
            ),
            "lidar": (
                self.lidar_stream.save_final()
                if self.lidar_stream is not None
                else {"available": False, "reason": "disabled_for_non_lidar_task"}
            ),
        }
        shutil.rmtree(self.video_frames)
        return report


def run_episode() -> int:
    global ACTIVE_WORLD, ACTIVE_COLLECTOR
    started = time.time()
    route = load_route(args.route)
    locked_inputs = verify_locked_inputs(route, args.route)
    locked_inputs["locomotion_policy"]["executed"] = not args.kinematic_base
    locked_inputs["locomotion_policy"]["verification_only"] = bool(
        args.kinematic_base
    )
    with np.load(args.teacher_episode, allow_pickle=False) as source:
        teacher = {key: np.asarray(source[key]) for key in source.files}
    required = {"state", "action", "object_position", "goal_position", "prompt", "fps"}
    if required - teacher.keys():
        raise RuntimeError(f"teacher missing {sorted(required - teacher.keys())}")
    (
        world,
        robot,
        cube,
        _table,
        cameras,
        object_position,
        goal_position,
        base_visual_follower,
        lidar_stream,
        imu_sensor,
        runtime_scene_contract,
    ) = create_world(route, teacher)
    ACTIVE_WORLD = world
    articulation = configure_articulation(robot)
    preflight = {
        "route": str(args.route),
        "teacher": str(args.teacher_episode),
        "dof_names": articulation["names"],
        "body_names": list(robot._articulation_view.body_names),
        "jacobian_shape": list(np.asarray(robot._articulation_view.get_jacobians()).shape),
        "station_object_xyz": object_position.tolist(),
        "station_goal_xyz": goal_position.tolist(),
        "gpu": args.gpu,
        "base_driver": {
            "schema": "m20-kinematic-policy-command-interpolation/v1",
            "enabled": bool(args.kinematic_base),
            "onnx_executed": not args.kinematic_base,
            "attack_boundary_preserved": True,
        },
        "low_level_control": {
            "profile": LOW_LEVEL_CONTROL_PROFILE,
            "source_entrypoint": SOURCE_CONTROLLER_ENTRYPOINT,
            "source_sha256": SOURCE_CONTROLLER_SHA256,
            "physics_hz": PHYSICS_HZ,
            "policy_hz": POLICY_HZ,
            "zero_command_wheel_brake": True,
            "stationary_pose_hold": False,
            "stationary_nominal_leg_stance": True,
            "stationary_stance_blend_seconds": STATIONARY_STANCE_BLEND_SECONDS,
            "stationary_leg_stiffness": STATIONARY_LEG_STIFFNESS,
            "stationary_leg_damping": STATIONARY_LEG_DAMPING,
            "hard_brake_phases": ["vla_timeout_brake"],
            "executed_action_history": True,
            "smooth_arm_ready_seconds": 3.0,
            "persistent_piper_command_targets": True,
            "piper_motion_enabled": PIPER_MOTION_ENABLED,
            "navigation_piper_policy": "measured_pose_hold",
        },
        "locked_inputs": locked_inputs,
        "runtime_scene_contract": runtime_scene_contract,
    }
    collector = Collector(
        route,
        teacher,
        world,
        robot,
        cube,
        goal_position,
        cameras,
        articulation,
        base_visual_follower,
        lidar_stream,
        imu_sensor,
    )
    ACTIVE_COLLECTOR = collector
    preflight["safety_monitor"] = collector.safety_runtime_manifest()
    if cameras:
        # Let RTX compose the render-only high-resolution chassis copy before
        # the first recorded/policy frame.  This does not advance PhysX time.
        collector.update_cameras()
        for _ in range(8):
            world.render()
    if collector.piper_motion_enabled:
        ik_delta = np.asarray([0.001, 0.0, 0.0, 0.0, 0.0, 0.0])
        ik_targets = collector.differential_ik_targets(ik_delta, 0.0)
        preflight["differential_ik_test"] = {
            "enabled": True,
            "delta_pose": [0.001, 0.0, 0.0, 0.0, 0.0, 0.0],
            "joint_targets": ik_targets.tolist(),
        }
    else:
        preflight["differential_ik_test"] = {
            "enabled": False,
            "reason": "task_does_not_enable_piper_motion",
            "held_joint_targets": collector.piper_hold_targets.tolist(),
        }
    if args.scoring_smoke_steps:
        if not 3 <= args.scoring_smoke_steps <= 10:
            raise ValueError("--scoring-smoke-steps must be within [3, 10]")
        zero = np.zeros(3, dtype=np.float32)
        for _ in range(POLICY_HZ):
            collector.latest_unified_action[:] = 0.0
            collector.advance("scoring_smoke_settle", zero)
        collector.start_scoring_capture()
        while len(collector.scoring_samples) < args.scoring_smoke_steps:
            collector.latest_unified_action[:] = 0.0
            collector.advance("scoring_smoke", zero)
        result = {
            "schema": "m20-v4-scoring-smoke/v1",
            "accepted": True,
            "checks": {"sample_count": len(collector.scoring_samples) == args.scoring_smoke_steps},
            "frames": len(collector.states),
        }
        collector.save(result)
        (args.output / "preflight.json").write_text(json.dumps(preflight, indent=2) + "\n", encoding="utf-8")
        (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
        return 0
    if args.preflight_only:
        if collector.squat_enabled:
            zero = np.zeros(3, dtype=np.float32)
            collector.squat_command = 0
            for _ in range(2 * POLICY_HZ):
                collector.latest_unified_action[:] = 0.0
                collector.advance("squat_preflight_settle", zero)
            initial_position, _initial_orientation = robot.get_world_pose()
            collector.squat_standing_reference_m = float(initial_position[2])
            low_height = float(initial_position[2])
            recovered_height = low_height
            for _cycle in range(collector.squat_required_cycles):
                collector.squat_command = -1
                for _ in range(3 * POLICY_HZ):
                    collector.latest_unified_action[:] = 0.0
                    collector.advance("squat_preflight_low", zero)
                    position, _orientation = robot.get_world_pose()
                    low_height = min(low_height, float(position[2]))
                    collector.update_squat_state(-1.0, float(position[2]), active=True)
                collector.squat_command = 1
                for _ in range(3 * POLICY_HZ):
                    collector.latest_unified_action[:] = 0.0
                    collector.advance("squat_preflight_stand", zero)
                    position, _orientation = robot.get_world_pose()
                    recovered_height = float(position[2])
                    collector.update_squat_state(1.0, recovered_height, active=True)
            low_threshold = float(initial_position[2]) - collector.squat_low_drop_m
            stand_threshold = float(initial_position[2]) - collector.squat_stand_tolerance_m
            checks = {
                "height_decreased": low_height <= low_threshold + 1.0e-4,
                "height_recovered": recovered_height >= stand_threshold - 1.0e-4,
                "required_cycles_counted": (
                    collector.squat_cycles >= collector.squat_required_cycles
                ),
            }
            final_piper_targets = collector.measured_piper_targets()
            piper_target_drift = float(np.max(np.abs(
                final_piper_targets - collector.piper_hold_targets
            )))
            piper_hold_tolerance = 1.0e-4 if args.kinematic_base else 0.03
            checks.update({
                "piper_motion_disabled": not collector.piper_motion_enabled,
                "piper_pose_held": piper_target_drift <= piper_hold_tolerance,
            })
            preflight["squat_control_test"] = {
                "initial_base_height_m": float(initial_position[2]),
                "minimum_base_height_m": low_height,
                "recovered_base_height_m": recovered_height,
                "completed_cycles": collector.squat_cycles,
                "initial_piper_targets": collector.piper_hold_targets.tolist(),
                "final_piper_targets": final_piper_targets.tolist(),
                "maximum_piper_target_drift": piper_target_drift,
                "piper_hold_tolerance": piper_hold_tolerance,
                "trace": collector.squat_trace,
                "checks": checks,
                "accepted": bool(all(checks.values())),
            }
            (args.output / "preflight.json").write_text(
                json.dumps(preflight, indent=2) + "\n", encoding="utf-8"
            )
            print(json.dumps(preflight, indent=2))
            return 0 if preflight["squat_control_test"]["accepted"] else 2
        if args.navigation_only:
            initial_position, _initial_orientation = robot.get_world_pose()
            minimum_height = float(initial_position[2])
            maximum_tilt = 0.0
            trajectory = []
            for step in range(5 * POLICY_HZ):
                collector.advance("standing_preflight", np.zeros(3, dtype=np.float32))
                position, orientation = robot.get_world_pose()
                rpy = quaternion_to_rpy(np.asarray(orientation, dtype=np.float64))
                minimum_height = min(minimum_height, float(position[2]))
                maximum_tilt = max(maximum_tilt, float(np.max(np.abs(rpy[:2]))))
                if step % (POLICY_HZ // 2) == 0 or step == 5 * POLICY_HZ - 1:
                    trajectory.append({
                        "time_s": (step + 1) / POLICY_HZ,
                        "base_xyz": np.asarray(position).tolist(),
                        "base_rpy": rpy.tolist(),
                    })
            final_position, final_orientation = robot.get_world_pose()
            checks = {
                "minimum_height": minimum_height >= float(initial_position[2]) - 0.12,
                "maximum_tilt": maximum_tilt <= 0.35,
                "finite": bool(
                    np.isfinite(np.asarray(final_position)).all()
                    and np.isfinite(np.asarray(final_orientation)).all()
                ),
            }
            preflight["standing_test"] = {
                "duration_s": 5.0,
                "initial_base_xyz": np.asarray(initial_position).tolist(),
                "final_base_xyz": np.asarray(final_position).tolist(),
                "final_base_orientation_wxyz": np.asarray(final_orientation).tolist(),
                "minimum_base_height_m": minimum_height,
                "maximum_abs_roll_pitch_rad": maximum_tilt,
                "trajectory": trajectory,
                "checks": checks,
                "accepted": bool(all(checks.values())),
            }
            (args.output / "preflight.json").write_text(
                json.dumps(preflight, indent=2) + "\n",
                encoding="utf-8",
            )
            print(json.dumps(preflight, indent=2))
            return 0 if preflight["standing_test"]["accepted"] else 2
        before = collector.state()[18:21].astype(np.float64)
        joints_before = np.asarray(robot.get_joint_positions(), dtype=np.float64)
        for _ in range(POLICY_HZ // 2):
            collector.advance("ik_preflight", np.zeros(3, dtype=np.float32), ik_targets)
        after = collector.state()[18:21].astype(np.float64)
        joints_after = np.asarray(robot.get_joint_positions(), dtype=np.float64)
        arm_dofs = collector.last_ik_debug["arm_dofs"]
        achieved_joint_delta = joints_after[arm_dofs] - joints_before[arm_dofs]
        preflight["differential_ik_test"].update({
            "tcp_before_base_m": before.tolist(),
            "tcp_after_base_m": after.tolist(),
            "achieved_delta_m": (after - before).tolist(),
            "achieved_joint_delta_rad": achieved_joint_delta.tolist(),
            "jacobian_predicted_twist": (
                collector.last_ik_debug["jacobian"] @ achieved_joint_delta
            ).tolist(),
        })
        (args.output / "preflight.json").write_text(json.dumps(preflight, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(preflight, indent=2))
        return 0
    (args.output / "preflight.json").write_text(json.dumps(preflight, indent=2) + "\n", encoding="utf-8")
    if args.policy_host:
        from attack_tools import create_attackable_vla_client

        policy = create_attackable_vla_client(args.policy_host, args.policy_port)
        result = collector.run_policy(policy)
    else:
        result = collector.run()
    result["locked_inputs"] = locked_inputs
    result["runtime_scene_contract"] = runtime_scene_contract
    result["joint_attack"] = collector.joint_attack_summary()
    result["wall_time_s"] = time.time() - started
    collector.save(result)
    result["recording"] = collector.finalize_video()
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    # Task acceptance is a recorded competition result, not a Runner transport
    # failure.  A complete real episode must still reach the formal Exporter.
    return 0


ACTIVE_WORLD = None
ACTIVE_COLLECTOR = None


def load_batch_manifest(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != "m20-main-runner-scene-batch/v1":
        raise ValueError("unsupported main Runner batch manifest")
    episodes = value.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("scene batch has no episodes")
    scenes = {item.get("scene") for item in episodes}
    if len(scenes) != 1:
        raise ValueError("persistent batch requires exactly one scene")
    return value


def batch_episode_args(episode: dict):
    output = Path(episode["work_dir"]).resolve(strict=True)
    preflight = json.loads((output / "runner_preflight.json").read_text(encoding="utf-8"))
    command = list(preflight["command"])
    script_index = next(
        index
        for index, token in enumerate(command)
        if Path(str(token)).name == Path(__file__).name
    )
    value = parser().parse_args(command[script_index + 1 :])
    value.batch_manifest = None
    value.portable_root = args.portable_root
    value.gpu = args.gpu
    return normalize_episode_args(value), preflight


def activate_batch_environment(episode: dict, preflight: dict) -> None:
    output = Path(episode["work_dir"]).resolve()
    package = Path(episode["package_root"]).resolve(strict=True)
    profile_root = (
        output / "resolved_attack_profile"
        if episode.get("attack_variant")
        else package / "data/attack/profiles"
    ).resolve(strict=True)
    os.environ.update(
        {
            "M20_ATTACK_ENABLED": "0" if episode.get("run_kind") == "clean" else "1",
            "M20_ATTACK_EVIDENCE_DIR": str(output / "attack_evidence"),
            "M20_QUESTION_PACKAGE_ROOT": str(package),
            "M20_QUESTION_ID": str(preflight["contract"]["scenario_id"]),
            "M20_ATTACK_PROFILE_ROOT": str(profile_root),
            "NO_PROXY": "127.0.0.1,localhost",
            "no_proxy": "127.0.0.1,localhost",
        }
    )
    os.environ.pop("M20_EXPERIMENT_CONFIG", None)


def cleanup_batch_episode() -> None:
    global ACTIVE_WORLD, ACTIVE_COLLECTOR
    if ACTIVE_WORLD is not None:
        try:
            ACTIVE_WORLD.stop()
        except Exception:
            pass
        try:
            ACTIVE_WORLD.clear_all_callbacks()
        except Exception:
            pass
    ACTIVE_COLLECTOR = None
    ACTIVE_WORLD = None
    World.clear_instance()
    context = omni.usd.get_context()
    if context.get_stage() is not None:
        context.close_stage()
    for _ in range(4):
        simulation_app.update()
    gc.collect()


def run_batch_episode(episode: dict) -> int:
    cleanup_batch_episode()
    value, preflight = batch_episode_args(episode)
    activate_batch_environment(episode, preflight)
    activate_episode_globals(value)
    log_path = Path(episode["work_dir"]) / "runner.log"
    with log_path.open("w", encoding="utf-8") as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            code = run_episode()
    if code != 0:
        raise RuntimeError(f"{episode['episode_id']}: runtime returned {code}")
    return code


def run_scene_batch() -> int:
    batch_manifest_path = args.batch_manifest
    batch = load_batch_manifest(batch_manifest_path)
    completed = []
    try:
        for episode in batch["episodes"]:
            run_batch_episode(episode)
            completed.append(episode["episode_id"])
        result = {
            "schema": "m20-main-runner-scene-batch-result/v1",
            "scene": batch["episodes"][0]["scene"],
            "single_kit_process": True,
            "episode_count": len(completed),
            "episodes": completed,
        }
        (batch_manifest_path.parent / "persistent_batch_result.json").write_text(
            json.dumps(result, indent=2) + "\n",
            encoding="utf-8",
        )
        return 0
    finally:
        cleanup_batch_episode()


exit_code = 1
try:
    exit_code = run_scene_batch() if PERSISTENT_BATCH_MODE else run_episode()
except BaseException as error:
    payload = {
        "schema": "m20-piper-isaac-collector-error/v1",
        "error_type": type(error).__name__,
        "message": str(error),
        "traceback": traceback.format_exc(),
    }
    (args.output / "error.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    traceback.print_exc()
finally:
    simulation_app.close()
raise SystemExit(exit_code)
