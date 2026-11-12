"""Simulator-independent contracts and control math for M20_Piper navigation.

This module deliberately has no Isaac Sim imports.  It is used by the simulator
entry point and by fast unit tests, so joint ordering and observation semantics
can be checked without starting Kit.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

ROBOT_ORDER: tuple[str, ...] = (
    "fl_hipx_joint",
    "fl_hipy_joint",
    "fl_knee_joint",
    "fl_wheel_joint",
    "fr_hipx_joint",
    "fr_hipy_joint",
    "fr_knee_joint",
    "fr_wheel_joint",
    "hl_hipx_joint",
    "hl_hipy_joint",
    "hl_knee_joint",
    "hl_wheel_joint",
    "hr_hipx_joint",
    "hr_hipy_joint",
    "hr_knee_joint",
    "hr_wheel_joint",
)

POLICY_ORDER: tuple[str, ...] = (
    "fl_hipx_joint",
    "fl_hipy_joint",
    "fl_knee_joint",
    "fr_hipx_joint",
    "fr_hipy_joint",
    "fr_knee_joint",
    "hl_hipx_joint",
    "hl_hipy_joint",
    "hl_knee_joint",
    "hr_hipx_joint",
    "hr_hipy_joint",
    "hr_knee_joint",
    "fl_wheel_joint",
    "fr_wheel_joint",
    "hl_wheel_joint",
    "hr_wheel_joint",
)

M20_JOINTS = frozenset(ROBOT_ORDER)
PIPER_JOINTS: tuple[str, ...] = tuple(f"joint{index}" for index in range(1, 9))
EXPECTED_DOF_NAMES = frozenset((*ROBOT_ORDER, *PIPER_JOINTS))
LEG_JOINTS: tuple[str, ...] = POLICY_ORDER[:12]
WHEEL_JOINTS: tuple[str, ...] = POLICY_ORDER[12:]
# All four delivered M20 wheel collision cylinders use radius=0.09 m in
# M20_high_precision_compat.urdf.  Encoder linear-speed attacks use this value
# only to convert the specified m/s bias into the rad/s values consumed by ONNX.
M20_WHEEL_RADIUS_M = 0.09

DEFAULT_POSITION_BY_NAME: Mapping[str, float] = {
    "fl_hipx_joint": 0.0,
    "fl_hipy_joint": -0.3,
    "fl_knee_joint": 0.6,
    "fl_wheel_joint": 0.0,
    "fr_hipx_joint": 0.0,
    "fr_hipy_joint": -0.3,
    "fr_knee_joint": 0.6,
    "fr_wheel_joint": 0.0,
    "hl_hipx_joint": 0.0,
    "hl_hipy_joint": 0.3,
    "hl_knee_joint": -0.6,
    "hl_wheel_joint": 0.0,
    "hr_hipx_joint": 0.0,
    "hr_hipy_joint": 0.3,
    "hr_knee_joint": -0.6,
    "hr_wheel_joint": 0.0,
}

ACTION_SCALE_BY_NAME: Mapping[str, float] = {
    **{name: (0.125 if "hipx" in name else 0.25) for name in LEG_JOINTS},
    **{name: 5.0 for name in WHEEL_JOINTS},
}

KP_BY_NAME: Mapping[str, float] = {
    **{name: 80.0 for name in LEG_JOINTS},
    **{name: 0.0 for name in WHEEL_JOINTS},
    **{name: 100.0 for name in PIPER_JOINTS},
}
KD_BY_NAME: Mapping[str, float] = {
    **{name: 2.0 for name in LEG_JOINTS},
    **{name: 0.6 for name in WHEEL_JOINTS},
    **{name: 1.0 for name in PIPER_JOINTS},
}
EFFORT_LIMIT_BY_NAME: Mapping[str, float] = {
    **{name: 76.4 for name in LEG_JOINTS},
    **{name: 21.6 for name in WHEEL_JOINTS},
    **{name: 100.0 for name in PIPER_JOINTS[:6]},
    **{name: 10.0 for name in PIPER_JOINTS[6:]},
}
VELOCITY_LIMIT_BY_NAME: Mapping[str, float] = {
    **{name: 22.4 for name in LEG_JOINTS},
    **{name: 79.3 for name in WHEEL_JOINTS},
    **{name: 5.0 for name in PIPER_JOINTS[:5]},
    PIPER_JOINTS[5]: 3.0,
    **{name: 1.0 for name in PIPER_JOINTS[6:]},
}

ACCEPTANCE_DEFAULTS: Mapping[str, float] = {
    "standing_seconds": 5.0,
    "minimum_base_height_m": 0.34,
    "maximum_standing_tilt_rad": 0.35,
    "maximum_navigation_tilt_rad": 0.60,
    "maximum_piper_error_rad": 0.10,
    "maximum_mount_translation_drift_m": 0.001,
    "maximum_mount_rotation_drift_rad": math.radians(0.1),
    "maximum_braked_planar_speed_mps": 0.05,
}

# On the high-precision combination asset the official policy's positive
# longitudinal command moves the base along its USD +X axis.  Navigation uses
# the same body/world convention, so no sign change is required.  This differs
# from the legacy low-resolution combination and is guarded by dynamic tests.
NAVIGATION_TO_POLICY_COMMAND_SCALE = np.asarray([1.0, 1.0, 1.0], dtype=np.float32)

ARMATURE_BY_NAME: Mapping[str, float] = {
    **{name: 0.0 for name in LEG_JOINTS},
    **{name: 0.00243216 for name in WHEEL_JOINTS},
}


@dataclass(frozen=True)
class ControllerConfig:
    forward_speed_mps: float
    heading_gain: float
    maximum_yaw_rate_rps: float
    rotate_in_place_threshold_rad: float
    slowdown_distance_m: float
    arrival_tolerance_m: float
    settle_seconds: float
    brake_seconds: float


@dataclass(frozen=True)
class ObstacleAabb:
    name: str
    minimum_xy: tuple[float, float]
    maximum_xy: tuple[float, float]


@dataclass(frozen=True)
class SceneConfig:
    path: Path
    sha256: str
    bundle_manifest: Path | None
    bundle_manifest_sha256: str | None
    default_prim: str
    source_physics_scene: str | None
    minimum_collision_prim_count: int
    camera_mode: str
    robot_radius_m: float
    obstacle_aabbs: tuple[ObstacleAabb, ...]


@dataclass(frozen=True)
class RouteConfig:
    schema: str
    name: str
    spawn_xyz: tuple[float, float, float]
    spawn_yaw_rad: float
    waypoints_xy: tuple[tuple[float, float], ...]
    maximum_duration_s: float
    controller: ControllerConfig
    scene: SceneConfig | None
    source_path: Path


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], context: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"{context} keys mismatch: missing={missing}, extra={extra}")


def _finite_number(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context} must be finite")
    return result


def _bounded(value: Any, low: float, high: float, context: str, *, low_open: bool = False) -> float:
    result = _finite_number(value, context)
    lower_ok = result > low if low_open else result >= low
    if not lower_ok or result > high:
        left = "(" if low_open else "["
        raise ValueError(f"{context} must be in {left}{low}, {high}]")
    return result


def _sha256_file(path: Path) -> str:
    """Return an observed digest for evidence, never as a package gate."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _project_local_path(value: Any, route_directory: Path, context: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} must be a non-empty path string")
    path = (route_directory / value).resolve(strict=True)
    package_root = (
        route_directory.parents[1]
        if route_directory.name == "env"
        and route_directory.parent.name == "data"
        and len(route_directory.parents) >= 2
        else route_directory
    )
    unified_release_root = (
        package_root.parent.parent
        if package_root.parent.name == "task"
        and (package_root.parent.parent / "UNIFIED_RELEASE.json").is_file()
        else None
    )
    portable_shared_root = (
        (package_root.parent / "_shared_assets").resolve()
        if (package_root.parent / "_shared_assets").is_dir()
        else None
    )
    unified_assets_root = (
        (unified_release_root / "assets").resolve(strict=True)
        if unified_release_root is not None
        else None
    )
    # Resolve roots from the package/release layout itself.  This keeps the
    # containment check intact after the release is moved to another machine.
    allowed_roots = (PROJECT_ROOT, package_root.resolve(strict=True))
    # Manipulation-only tasks do not carry a route file.  The trusted adapter
    # writes a runtime-only bootstrap route in the episode directory and
    # supplies the immutable question package root through this environment
    # variable.  Permit only that package/release boundary; arbitrary absolute
    # paths remain rejected.
    question_root_text = os.environ.get("M20_QUESTION_PACKAGE_ROOT")
    if question_root_text:
        question_root = Path(question_root_text).expanduser().resolve(strict=True)
        allowed_roots += (question_root,)
        if (
            question_root.parent.name == "task"
            and (question_root.parent.parent / "UNIFIED_RELEASE.json").is_file()
        ):
            allowed_roots += (
                (question_root.parent.parent / "assets").resolve(strict=True),
            )
    if unified_assets_root is not None:
        allowed_roots += (unified_assets_root,)
    if portable_shared_root is not None:
        allowed_roots += (portable_shared_root,)
    if not any(path == root or root in path.parents for root in allowed_roots):
        raise ValueError(f"{context} must stay inside an approved M20 scene root")
    if not path.is_file():
        raise ValueError(f"{context} must identify a file: {path}")
    return path


def point_obstacle_clearance(
    position_xy: Sequence[float], obstacle: ObstacleAabb, robot_radius_m: float
) -> float:
    """Return point-to-AABB distance minus a circular robot radius."""

    position = np.asarray(position_xy, dtype=np.float64)
    if position.shape != (2,) or not np.isfinite(position).all():
        raise ValueError("position_xy must be a finite two-vector")
    if not math.isfinite(robot_radius_m) or robot_radius_m < 0.0:
        raise ValueError("robot_radius_m must be non-negative and finite")
    minimum = np.asarray(obstacle.minimum_xy, dtype=np.float64)
    maximum = np.asarray(obstacle.maximum_xy, dtype=np.float64)
    outside = np.maximum(np.maximum(minimum - position, position - maximum), 0.0)
    return float(np.linalg.norm(outside) - robot_radius_m)


def minimum_route_obstacle_clearance(
    points_xy: Sequence[Sequence[float]],
    obstacles: Sequence[ObstacleAabb],
    robot_radius_m: float,
    sample_spacing_m: float = 0.05,
) -> tuple[float, str, int]:
    """Sample a polyline and return its minimum inflated-obstacle clearance."""

    if len(points_xy) < 2 or not obstacles:
        raise ValueError("route clearance needs at least two points and one obstacle")
    if not math.isfinite(sample_spacing_m) or sample_spacing_m <= 0.0:
        raise ValueError("sample_spacing_m must be positive and finite")
    points = [np.asarray(point, dtype=np.float64) for point in points_xy]
    if any(point.shape != (2,) or not np.isfinite(point).all() for point in points):
        raise ValueError("route points must be finite two-vectors")
    minimum_clearance = float("inf")
    nearest_name = ""
    sample_count = 0
    for start, end in zip(points, points[1:]):
        segment_length = float(np.linalg.norm(end - start))
        steps = max(1, math.ceil(segment_length / sample_spacing_m))
        for alpha in np.linspace(0.0, 1.0, steps + 1):
            position = start + float(alpha) * (end - start)
            for obstacle in obstacles:
                clearance = point_obstacle_clearance(position, obstacle, robot_radius_m)
                if clearance < minimum_clearance:
                    minimum_clearance = clearance
                    nearest_name = obstacle.name
            sample_count += 1
    return minimum_clearance, nearest_name, sample_count


def load_route(path: str | Path) -> RouteConfig:
    """Load and strictly validate a route JSON document."""

    source = Path(path).expanduser().resolve(strict=True)
    with source.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("route root must be an object")
    schema = payload.get("schema")
    if schema == "m20-piper-flat-navigation-route/v1":
        expected_route_keys = {"schema", "name", "spawn", "waypoints", "maximum_duration_s", "controller"}
    elif schema == "m20-piper-known-map-scene-route/v1":
        expected_route_keys = {
            "schema", "name", "scene", "spawn", "waypoints", "maximum_duration_s", "controller"
        }
    else:
        raise ValueError("unsupported route schema")
    _require_exact_keys(payload, expected_route_keys, "route")
    if not isinstance(payload["name"], str) or not payload["name"].strip():
        raise ValueError("route.name must be a non-empty string")

    spawn = payload["spawn"]
    if not isinstance(spawn, dict):
        raise ValueError("route.spawn must be an object")
    _require_exact_keys(spawn, {"xyz", "yaw_rad"}, "route.spawn")
    if not isinstance(spawn["xyz"], list) or len(spawn["xyz"]) != 3:
        raise ValueError("route.spawn.xyz must contain exactly three numbers")
    spawn_xyz = tuple(_finite_number(v, f"route.spawn.xyz[{i}]") for i, v in enumerate(spawn["xyz"]))
    if any(abs(value) > 100.0 for value in spawn_xyz[:2]):
        raise ValueError("route.spawn x/y must stay within 100 m of the origin")
    if not 0.1 <= spawn_xyz[2] <= 2.0:
        raise ValueError("route.spawn z must be within [0.1, 2.0] m")
    spawn_yaw = _bounded(spawn["yaw_rad"], -math.pi, math.pi, "route.spawn.yaw_rad")

    scene: SceneConfig | None = None
    if schema == "m20-piper-known-map-scene-route/v1":
        scene_payload = payload["scene"]
        if not isinstance(scene_payload, dict):
            raise ValueError("route.scene must be an object")
        required_scene_keys = {
                "path",
                "default_prim",
                "source_physics_scene",
                "minimum_collision_prim_count",
                "camera_mode",
                "robot_radius_m",
                "obstacle_aabbs_xy_m",
        }
        optional_scene_keys = {
            "sha256",
            "bundle_manifest",
            "bundle_manifest_sha256",
        }
        actual_scene_keys = set(scene_payload)
        missing_scene_keys = sorted(required_scene_keys - actual_scene_keys)
        extra_scene_keys = sorted(
            actual_scene_keys - required_scene_keys - optional_scene_keys
        )
        if missing_scene_keys or extra_scene_keys:
            raise ValueError(
                "route.scene keys mismatch: "
                f"missing={missing_scene_keys}, extra={extra_scene_keys}"
            )
        default_prim = scene_payload["default_prim"]
        source_physics_scene = scene_payload["source_physics_scene"]
        if not isinstance(default_prim, str) or not default_prim.startswith("/"):
            raise ValueError("route.scene.default_prim must be an absolute prim path")
        if source_physics_scene is not None and (
            not isinstance(source_physics_scene, str) or not source_physics_scene.startswith("/")
        ):
            raise ValueError("route.scene.source_physics_scene must be null or an absolute prim path")
        minimum_collision_count = scene_payload["minimum_collision_prim_count"]
        if isinstance(minimum_collision_count, bool) or not isinstance(minimum_collision_count, int) or minimum_collision_count <= 0:
            raise ValueError("route.scene.minimum_collision_prim_count must be a positive integer")
        if scene_payload["camera_mode"] != "chase":
            raise ValueError("route.scene.camera_mode must be 'chase'")
        raw_obstacles = scene_payload["obstacle_aabbs_xy_m"]
        if not isinstance(raw_obstacles, list) or not raw_obstacles:
            raise ValueError("route.scene.obstacle_aabbs_xy_m must be a non-empty array")
        obstacles: list[ObstacleAabb] = []
        obstacle_names: set[str] = set()
        for index, raw_obstacle in enumerate(raw_obstacles):
            if not isinstance(raw_obstacle, dict):
                raise ValueError(f"route.scene.obstacle_aabbs_xy_m[{index}] must be an object")
            _require_exact_keys(raw_obstacle, {"name", "minimum", "maximum"}, f"route.scene obstacle {index}")
            name = raw_obstacle["name"]
            if not isinstance(name, str) or not name or name in obstacle_names:
                raise ValueError(f"route.scene obstacle {index} must have a unique non-empty name")
            if not isinstance(raw_obstacle["minimum"], list) or len(raw_obstacle["minimum"]) != 2:
                raise ValueError(f"route.scene obstacle {index}.minimum must have two values")
            if not isinstance(raw_obstacle["maximum"], list) or len(raw_obstacle["maximum"]) != 2:
                raise ValueError(f"route.scene obstacle {index}.maximum must have two values")
            minimum_xy = tuple(
                _finite_number(value, f"route.scene obstacle {index}.minimum[{axis}]")
                for axis, value in enumerate(raw_obstacle["minimum"])
            )
            maximum_xy = tuple(
                _finite_number(value, f"route.scene obstacle {index}.maximum[{axis}]")
                for axis, value in enumerate(raw_obstacle["maximum"])
            )
            if any(minimum_xy[axis] >= maximum_xy[axis] for axis in range(2)):
                raise ValueError(f"route.scene obstacle {index} has invalid bounds")
            obstacles.append(ObstacleAabb(name, minimum_xy, maximum_xy))
            obstacle_names.add(name)
        scene_path = _project_local_path(
            scene_payload["path"], source.parent, "route.scene.path"
        )
        bundle_manifest_value = scene_payload.get("bundle_manifest")
        bundle_manifest = (
            _project_local_path(
                bundle_manifest_value, source.parent, "route.scene.bundle_manifest"
            )
            if bundle_manifest_value
            else None
        )
        scene = SceneConfig(
            path=scene_path,
            # Authored SHA fields are intentionally ignored.  The observed
            # digest is retained only for HDF5 evidence metadata.
            sha256=_sha256_file(scene_path),
            bundle_manifest=bundle_manifest,
            bundle_manifest_sha256=None,
            default_prim=default_prim,
            source_physics_scene=source_physics_scene,
            minimum_collision_prim_count=minimum_collision_count,
            camera_mode="chase",
            # A zero value validates the authored route centreline against the
            # obstacle AABBs without adding a synthetic circular envelope.
            # Live robot collision remains entirely governed by PhysX shapes.
            robot_radius_m=_bounded(scene_payload["robot_radius_m"], 0.0, 2.0, "route.scene.robot_radius_m"),
            obstacle_aabbs=tuple(obstacles),
        )

    raw_waypoints = payload["waypoints"]
    if not isinstance(raw_waypoints, list) or not raw_waypoints:
        raise ValueError("route.waypoints must be a non-empty array")
    if len(raw_waypoints) > 100:
        raise ValueError("route.waypoints cannot contain more than 100 entries")
    waypoints: list[tuple[float, float]] = []
    for index, waypoint in enumerate(raw_waypoints):
        if not isinstance(waypoint, list) or len(waypoint) != 2:
            raise ValueError(f"route.waypoints[{index}] must contain exactly two numbers")
        xy = tuple(_finite_number(v, f"route.waypoints[{index}][{axis}]") for axis, v in enumerate(waypoint))
        if any(abs(value) > 100.0 for value in xy):
            raise ValueError(f"route.waypoints[{index}] is outside the 100 m workspace")
        waypoints.append(xy)

    controller_payload = payload["controller"]
    if not isinstance(controller_payload, dict):
        raise ValueError("route.controller must be an object")
    controller_keys = {
        "forward_speed_mps",
        "heading_gain",
        "maximum_yaw_rate_rps",
        "rotate_in_place_threshold_rad",
        "slowdown_distance_m",
        "arrival_tolerance_m",
        "settle_seconds",
        "brake_seconds",
    }
    _require_exact_keys(controller_payload, controller_keys, "route.controller")
    controller = ControllerConfig(
        forward_speed_mps=_bounded(
            controller_payload["forward_speed_mps"], 0.0, 1.0, "forward_speed_mps", low_open=True
        ),
        heading_gain=_bounded(controller_payload["heading_gain"], 0.0, 10.0, "heading_gain", low_open=True),
        maximum_yaw_rate_rps=_bounded(
            controller_payload["maximum_yaw_rate_rps"], 0.0, 3.0, "maximum_yaw_rate_rps", low_open=True
        ),
        rotate_in_place_threshold_rad=_bounded(
            controller_payload["rotate_in_place_threshold_rad"],
            0.0,
            math.pi,
            "rotate_in_place_threshold_rad",
            low_open=True,
        ),
        slowdown_distance_m=_bounded(
            controller_payload["slowdown_distance_m"], 0.01, 5.0, "slowdown_distance_m"
        ),
        arrival_tolerance_m=_bounded(
            controller_payload["arrival_tolerance_m"], 0.0, 1.0, "arrival_tolerance_m", low_open=True
        ),
        settle_seconds=_bounded(controller_payload["settle_seconds"], 0.0, 30.0, "settle_seconds"),
        brake_seconds=_bounded(controller_payload["brake_seconds"], 0.0, 30.0, "brake_seconds"),
    )
    if controller.arrival_tolerance_m > controller.slowdown_distance_m:
        raise ValueError("arrival_tolerance_m cannot exceed slowdown_distance_m")

    # B-board Q02 has two long navigation legs and an explicit 900-second budget.
    # This validates a ceiling; each task still uses its own configured duration.
    maximum_duration = _bounded(payload["maximum_duration_s"], 1.0, 900.0, "maximum_duration_s")
    skip_route_clearance = os.getenv("M20_SKIP_ROUTE_CLEARANCE_CHECK", "0").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if scene is not None and not skip_route_clearance:
        clearance, nearest_obstacle, _ = minimum_route_obstacle_clearance(
            [spawn_xyz[:2], *waypoints], scene.obstacle_aabbs, scene.robot_radius_m
        )
        if clearance < 0.0:
            # Serialized obstacle boxes are advisory route metadata. The live
            # PhysX shapes remain authoritative after a scene/station revision.
            print(
                "ROBOLAB_PACKAGE_DIAGNOSTIC="
                f"route intersects serialized inflated obstacle {nearest_obstacle}: "
                f"clearance={clearance:.6f} m; continuing with live scene physics",
                file=sys.stderr,
            )
    return RouteConfig(
        schema=payload["schema"],
        name=payload["name"].strip(),
        spawn_xyz=spawn_xyz,
        spawn_yaw_rad=spawn_yaw,
        waypoints_xy=tuple(waypoints),
        maximum_duration_s=maximum_duration,
        controller=controller,
        scene=scene,
        source_path=source,
    )


def wrap_angle(angle_rad: float) -> float:
    """Wrap an angle to [-pi, pi)."""

    return (float(angle_rad) + math.pi) % (2.0 * math.pi) - math.pi


def quaternion_to_rotation_matrix(quaternion_wxyz: Sequence[float]) -> np.ndarray:
    quaternion = np.asarray(quaternion_wxyz, dtype=np.float64)
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
        raise ValueError("quaternion must be a finite wxyz vector")
    norm = float(np.linalg.norm(quaternion))
    if norm < 1.0e-12:
        raise ValueError("quaternion norm is zero")
    w, x, y, z = quaternion / norm
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def yaw_from_quaternion(quaternion_wxyz: Sequence[float]) -> float:
    rotation = quaternion_to_rotation_matrix(quaternion_wxyz)
    return math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))


def projected_gravity(quaternion_wxyz: Sequence[float]) -> np.ndarray:
    """Return unit gravity in the robot base frame."""

    rotation_world_from_body = quaternion_to_rotation_matrix(quaternion_wxyz)
    return (rotation_world_from_body.T @ np.asarray([0.0, 0.0, -1.0], dtype=np.float64)).astype(np.float32)


def base_tilt_rad(quaternion_wxyz: Sequence[float]) -> float:
    rotation = quaternion_to_rotation_matrix(quaternion_wxyz)
    return math.acos(float(np.clip(rotation[2, 2], -1.0, 1.0)))


def waypoint_command(
    position_xy: Sequence[float],
    yaw_rad: float,
    waypoint_xy: Sequence[float],
    config: ControllerConfig,
) -> tuple[np.ndarray, float, float]:
    """Compute [vx, vy, wz], distance, and heading error for one waypoint."""

    position = np.asarray(position_xy, dtype=np.float64)
    waypoint = np.asarray(waypoint_xy, dtype=np.float64)
    if position.shape != (2,) or waypoint.shape != (2,) or not np.isfinite(position).all() or not np.isfinite(waypoint).all():
        raise ValueError("position_xy and waypoint_xy must be finite 2-vectors")
    delta = waypoint - position
    distance = float(np.linalg.norm(delta))
    desired_heading = float(yaw_rad) if distance < 1.0e-12 else math.atan2(float(delta[1]), float(delta[0]))
    heading_error = wrap_angle(desired_heading - float(yaw_rad))
    yaw_rate = float(np.clip(config.heading_gain * heading_error, -config.maximum_yaw_rate_rps, config.maximum_yaw_rate_rps))
    if distance <= config.arrival_tolerance_m or abs(heading_error) >= config.rotate_in_place_threshold_rad:
        forward_speed = 0.0
    else:
        forward_speed = config.forward_speed_mps * min(1.0, distance / config.slowdown_distance_m)
    return np.asarray([forward_speed, 0.0, yaw_rate], dtype=np.float32), distance, heading_error


def navigation_command_to_policy(
    command_vx_vy_wz: Sequence[float],
    scale_vx_vy_wz: Sequence[float] = NAVIGATION_TO_POLICY_COMMAND_SCALE,
) -> np.ndarray:
    """Map a USD/world navigation command into a policy's command frame."""

    command = _vector(command_vx_vy_wz, 3, "command_vx_vy_wz")
    scale = _vector(scale_vx_vy_wz, 3, "scale_vx_vy_wz")
    if np.any(np.abs(scale) < 1.0e-6) or np.any(np.abs(scale) > 1.0):
        raise ValueError("scale_vx_vy_wz entries must have magnitude within [1e-6, 1]")
    return command * scale


def _vector(value: Sequence[float] | np.ndarray, length: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if result.shape != (length,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite vector with shape ({length},)")
    return result


def build_policy_observation(
    base_angular_velocity_xyz: Sequence[float],
    gravity_body_xyz: Sequence[float],
    command_vx_vy_wz: Sequence[float],
    joint_positions_robot_order: Sequence[float],
    joint_velocities_robot_order: Sequence[float],
    previous_action_policy_order: Sequence[float],
) -> np.ndarray:
    """Build the official M20 ONNX observation with shape ``[1, 57]``."""

    angular_velocity = _vector(base_angular_velocity_xyz, 3, "base_angular_velocity_xyz") * 0.25
    gravity = _vector(gravity_body_xyz, 3, "gravity_body_xyz")
    command = _vector(command_vx_vy_wz, 3, "command_vx_vy_wz")
    position_robot = _vector(joint_positions_robot_order, 16, "joint_positions_robot_order")
    velocity_robot = _vector(joint_velocities_robot_order, 16, "joint_velocities_robot_order")
    previous_action = _vector(previous_action_policy_order, 16, "previous_action_policy_order")

    robot_index = {name: index for index, name in enumerate(ROBOT_ORDER)}
    position_policy = np.asarray([position_robot[robot_index[name]] for name in POLICY_ORDER], dtype=np.float32)
    velocity_policy = np.asarray([velocity_robot[robot_index[name]] for name in POLICY_ORDER], dtype=np.float32)
    default_policy = np.asarray([DEFAULT_POSITION_BY_NAME[name] for name in POLICY_ORDER], dtype=np.float32)
    position_policy[12:] = 0.0
    relative_position = position_policy - default_policy
    observation = np.concatenate(
        (
            angular_velocity,
            gravity,
            command,
            relative_position,
            velocity_policy * 0.05,
            previous_action,
        )
    ).astype(np.float32, copy=False)
    if observation.shape != (57,) or not np.isfinite(observation).all():
        raise RuntimeError("constructed policy observation is not finite [57]")
    return observation.reshape(1, 57)


def policy_action_to_targets(action_policy_order: Sequence[float]) -> tuple[dict[str, float], dict[str, float]]:
    """Map official policy output to 12 position and four velocity targets."""

    action = _vector(action_policy_order, 16, "action_policy_order")
    position_targets = {
        name: DEFAULT_POSITION_BY_NAME[name] + float(action[index]) * ACTION_SCALE_BY_NAME[name]
        for index, name in enumerate(LEG_JOINTS)
    }
    velocity_targets = {
        name: float(action[index + 12]) * ACTION_SCALE_BY_NAME[name]
        for index, name in enumerate(WHEEL_JOINTS)
    }
    return position_targets, velocity_targets


def yaw_quaternion(yaw_rad: float) -> np.ndarray:
    half = 0.5 * float(yaw_rad)
    return np.asarray([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=np.float64)


def named_array(names: Sequence[str], values_by_name: Mapping[str, float]) -> np.ndarray:
    missing = sorted(set(names) - set(values_by_name))
    if missing:
        raise ValueError(f"values missing names: {missing}")
    result = np.asarray([values_by_name[name] for name in names], dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError("named array contains non-finite values")
    return result
