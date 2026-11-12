"""Typed attack handlers used by the config-driven M20 attack pipeline."""
from __future__ import annotations
from typing import Any, Mapping
import math
import hashlib
import numpy as np
from .immediate_attack_handlers import IMMEDIATE_HANDLERS, IMUTransform, ObjectPointDropout

def picture_red_to_green(picture: np.ndarray, parameters: Mapping[str, Any],
                         _context: Mapping[str, Any]) -> np.ndarray:
    image = np.asarray(picture)
    if image.ndim != 3 or image.shape[-1] < 3:
        raise ValueError(f"picture must be HxWxC, got {image.shape}")
    result, margin = image.copy(), int(parameters.get("red_margin", 30))
    red, green, blue = [result[..., i].astype(np.int16) for i in range(3)]
    mask = (red >= green + margin) & (red >= blue + margin)
    old_red = result[..., 0].copy()
    result[..., 0][mask], result[..., 1][mask] = result[..., 1][mask], old_red[mask]
    return result


def picture_region_patch(picture: np.ndarray, parameters: Mapping[str, Any],
                         _context: Mapping[str, Any]) -> np.ndarray:
    """Overlay a deterministic bounded patch on an RGB observation."""
    image = np.asarray(picture)
    if image.ndim != 3 or image.shape[-1] < 3:
        raise ValueError(f"picture must be HxWxC, got {image.shape}")
    box = np.asarray(parameters.get("bbox_norm", [0.35, 0.35, 0.65, 0.65]), dtype=float)
    if box.shape != (4,) or not np.isfinite(box).all() or np.any(box < 0) or np.any(box > 1):
        raise ValueError("bbox_norm must contain four values in [0, 1]")
    x0, y0, x1, y1 = box
    if x0 >= x1 or y0 >= y1:
        raise ValueError("bbox_norm must have positive width and height")
    height, width = image.shape[:2]
    left, right = int(x0 * width), max(int(x1 * width), int(x0 * width) + 1)
    top, bottom = int(y0 * height), max(int(y1 * height), int(y0 * height) + 1)
    result = image.copy()
    color_a = np.asarray(parameters.get("color_a_rgb", [0, 255, 0]), dtype=np.uint8)
    color_b = np.asarray(parameters.get("color_b_rgb", [15, 90, 15]), dtype=np.uint8)
    if color_a.shape != (3,) or color_b.shape != (3,):
        raise ValueError("patch colors must be RGB triples")
    tile = max(1, int(parameters.get("tile_px", 8)))
    yy, xx = np.indices((bottom - top, right - left))
    checker = ((xx // tile + yy // tile) % 2).astype(bool)
    patch = np.where(checker[..., None], color_a, color_b)
    result[top:bottom, left:right, :3] = patch
    return result

def state_offset(state: np.ndarray, parameters: Mapping[str, Any],
                 _context: Mapping[str, Any]) -> np.ndarray:
    result = np.asarray(state).copy()
    indices, offsets = list(parameters.get("indices", [])), list(parameters.get("offsets", []))
    if len(indices) != len(offsets):
        raise ValueError("state indices and offsets must have equal lengths")
    result[indices] += np.asarray(offsets, dtype=result.dtype)
    return result


def state_set_values(state: np.ndarray, parameters: Mapping[str, Any],
                     _context: Mapping[str, Any]) -> np.ndarray:
    result = np.asarray(state).copy()
    indices, values = list(parameters.get("indices", [])), list(parameters.get("values", []))
    if len(indices) != len(values):
        raise ValueError("state indices and values must have equal lengths")
    result[indices] = np.asarray(values, dtype=result.dtype)
    return result

def replace_prompt(prompt: str, parameters: Mapping[str, Any],
                   _context: Mapping[str, Any]) -> str:
    if not isinstance(prompt, str):
        raise TypeError(f"prompt must be str, got {type(prompt).__name__}")
    return str(parameters["replacement"])


def replace_prompt_sequence(prompt: str, parameters: Mapping[str, Any],
                            context: Mapping[str, Any]) -> str:
    """Select the authored prompt-injection round by attack-relative time."""
    if not isinstance(prompt, str):
        raise TypeError(f"prompt must be str, got {type(prompt).__name__}")
    raw_steps = parameters.get("steps")
    if not isinstance(raw_steps, list) or len(raw_steps) < 2:
        raise ValueError("prompt sequence requires at least two timed steps")
    steps: list[tuple[float, str]] = []
    for item in raw_steps:
        if not isinstance(item, Mapping):
            raise TypeError("prompt sequence step must be a mapping")
        at_s = float(item["at_s"])
        replacement = str(item["replacement"])
        if not math.isfinite(at_s) or at_s < 0.0 or not replacement:
            raise ValueError("invalid prompt sequence step")
        steps.append((at_s, replacement))
    if steps != sorted(steps, key=lambda item: item[0]) or steps[0][0] != 0.0:
        raise ValueError("prompt sequence must be ordered and start at zero")
    if parameters.get('delivery') != 'once_per_scheduled_request' and str(parameters.get("after_sequence", "hold_last")) != "hold_last":
        raise ValueError("prompt sequence only supports after_sequence='hold_last'")
    runtime = context.get("runtime", {}) if isinstance(context, Mapping) else {}
    if parameters.get('delivery') == 'once_per_scheduled_request':
        index = runtime.get('prompt_round_index')
        if type(index) is not int or not 0 <= index < len(steps):
            raise ValueError('scheduled prompt requires a real request round index')
        return steps[index][1]
    snapshot = context.get("snapshot", {}) if isinstance(context, Mapping) else {}
    attack_runtime = snapshot.get("_attack_runtime", {}) if isinstance(snapshot, Mapping) else {}
    now_s = float(runtime.get("time_s", 0.0))
    activated_s = float(attack_runtime.get("activated_s", now_s))
    elapsed_s = max(0.0, now_s - activated_s)
    selected = steps[0][1]
    for at_s, replacement in steps:
        if elapsed_s < at_s:
            break
        selected = replacement
    return selected


def append_target_coordinates(prompt: str, parameters: Mapping[str, Any],
                              _context: Mapping[str, Any]) -> str:
    if not isinstance(prompt, str):
        raise TypeError(f"prompt must be str, got {type(prompt).__name__}")
    xyz = np.asarray(parameters["target_xyz"], dtype=float)
    if xyz.shape != (3,) or not np.isfinite(xyz).all():
        raise ValueError("target_xyz must be a finite xyz vector")
    label = str(parameters.get("target_label", "target"))
    return f"{prompt} Use {label} coordinates x={xyz[0]:.3f}, y={xyz[1]:.3f}, z={xyz[2]:.3f}."

def scale_actions(actions: np.ndarray, parameters: Mapping[str, Any],
                  _context: Mapping[str, Any]) -> np.ndarray:
    result, factor = np.asarray(actions).copy(), float(parameters["factor"])
    if not 0.1 <= factor <= 5.0:
        raise ValueError("action factor must be within [0.1, 5.0]")
    indices = list(parameters.get("indices", range(result.shape[-1])))
    result[..., indices] *= factor
    return result


def affine_actions(actions: np.ndarray, parameters: Mapping[str, Any],
                   _context: Mapping[str, Any]) -> np.ndarray:
    result = np.asarray(actions).copy()
    indices = list(parameters.get("indices", range(result.shape[-1])))
    scale = np.asarray(parameters.get("scale", 1.0), dtype=float)
    offset = np.asarray(parameters.get("offset", 0.0), dtype=float)
    if not np.isfinite(scale).all() or not np.isfinite(offset).all():
        raise ValueError("action affine parameters must be finite")
    if np.any(np.abs(scale) > 5.0) or np.any(np.abs(offset) > 5.0):
        raise ValueError("action affine parameters exceed the bounded range")
    result[..., indices] = result[..., indices] * scale + offset
    return result


def gripper_directional_scale(actions: np.ndarray, parameters: Mapping[str, Any],
                              _context: Mapping[str, Any]) -> np.ndarray:
    """Scale only the requested gripper-command direction."""
    result = np.asarray(actions).copy()
    index = int(parameters.get("index", 9))
    scale = float(parameters["scale"])
    direction = str(parameters.get("direction", "close"))
    if not 0.0 <= scale <= 1.0 or direction not in {"open", "close"}:
        raise ValueError("invalid directional gripper scale")
    value = result[..., index]
    mask = value > 0.5 if direction == "close" else value < 0.5
    neutral = 0.0 if direction == "close" else 1.0
    result[..., index] = np.where(mask, neutral + (value - neutral) * scale, value)
    return result


def gripper_opening_command_offset(actions: np.ndarray,
                                   parameters: Mapping[str, Any],
                                   _context: Mapping[str, Any]) -> np.ndarray:
    """Apply a physical opening-width delta to a normalized closed fraction."""
    result = np.asarray(actions).copy()
    index = int(parameters.get("index", 9))
    opening_offset = float(parameters["opening_offset_m"])
    maximum_opening = float(parameters["calibrated_maximum_opening_m"])
    if not math.isfinite(opening_offset) or not math.isfinite(maximum_opening) or maximum_opening <= 0.0:
        raise ValueError("invalid gripper command opening calibration")
    result[..., index] = np.clip(
        result[..., index] - opening_offset / maximum_opening, 0.0, 1.0
    )
    return result


def base_velocity_toward_object(command: np.ndarray,
                                parameters: Mapping[str, Any],
                                context: Mapping[str, Any]) -> np.ndarray:
    """Replace planar velocity with a fixed speed toward a registered object."""
    result = np.asarray(command).copy()
    if result.shape[-1] < 2:
        raise ValueError("base command must expose planar x/y components")
    snapshot = context.get("snapshot", {}) if isinstance(context, Mapping) else {}
    robot = snapshot.get("robot", {}) if isinstance(snapshot, Mapping) else {}
    objects = snapshot.get("objects", {}) if isinstance(snapshot, Mapping) else {}
    object_id = str(parameters["object_id"])
    target = objects.get(object_id, {}) if isinstance(objects, Mapping) else {}
    robot_xyz = np.asarray(robot.get("position_xyz"), dtype=np.float64)
    target_xyz = np.asarray(target.get("position_xyz"), dtype=np.float64)
    rpy = np.asarray(robot.get("rpy"), dtype=np.float64)
    speed = float(parameters["speed_mps"])
    if (
        robot_xyz.shape != (3,) or target_xyz.shape != (3,) or rpy.shape != (3,)
        or not np.isfinite(robot_xyz).all() or not np.isfinite(target_xyz).all()
        or not np.isfinite(rpy).all() or not math.isfinite(speed) or speed < 0.0
    ):
        raise ValueError("toward-object base command requires finite poses and speed")
    direction = target_xyz[:2] - robot_xyz[:2]
    norm = float(np.linalg.norm(direction))
    if norm <= 1.0e-9:
        result[..., :2] = 0.0
        return result
    world_velocity = direction * (speed / norm)
    cosine, sine = math.cos(float(rpy[2])), math.sin(float(rpy[2]))
    result[..., 0] = cosine * world_velocity[0] + sine * world_velocity[1]
    result[..., 1] = -sine * world_velocity[0] + cosine * world_velocity[1]
    return result


def world_frame_planar_velocity_offset(actions: np.ndarray,
                                       parameters: Mapping[str, Any],
                                       context: Mapping[str, Any]) -> np.ndarray:
    """Add a world-XY velocity vector after rotating it into the base frame."""
    result = np.asarray(actions).copy()
    offset = np.asarray(parameters["offset_world_xy_mps"], dtype=np.float64)
    runtime = context.get("runtime", {}) if isinstance(context, Mapping) else {}
    yaw = float(runtime.get("base_yaw_rad", np.nan))
    if offset.shape != (2,) or not np.isfinite(offset).all() or not math.isfinite(yaw):
        raise ValueError("world-frame base offset requires finite xy and base_yaw_rad")
    cosine, sine = math.cos(yaw), math.sin(yaw)
    local = np.asarray([
        cosine * offset[0] + sine * offset[1],
        -sine * offset[0] + cosine * offset[1],
    ])
    result[..., 0:2] += local.astype(result.dtype)
    return result


def world_frame_tcp_translation_offset(actions: np.ndarray,
                                       parameters: Mapping[str, Any],
                                       context: Mapping[str, Any]) -> np.ndarray:
    """Rotate a world-frame XYZ offset into the robot/arm-base action frame."""
    result = np.asarray(actions).copy()
    indices = list(parameters.get("translation_indices", [3, 4, 5]))
    offset = np.asarray(parameters["offset_world_xyz_m"], dtype=np.float64)
    snapshot = context.get("snapshot", {}) if isinstance(context, Mapping) else {}
    robot = snapshot.get("robot", {}) if isinstance(snapshot, Mapping) else {}
    rpy = np.asarray(robot.get("rpy"), dtype=np.float64)
    if len(indices) != 3 or offset.shape != (3,) or rpy.shape != (3,):
        raise ValueError("world-frame TCP offset requires xyz offset, rpy, and three indices")
    if not np.isfinite(offset).all() or not np.isfinite(rpy).all():
        raise ValueError("world-frame TCP offset inputs must be finite")
    cosine, sine = math.cos(float(rpy[2])), math.sin(float(rpy[2]))
    local = np.asarray([
        cosine * offset[0] + sine * offset[1],
        -sine * offset[0] + cosine * offset[1],
        offset[2],
    ])
    result[..., indices] += local.astype(result.dtype)
    return result


def translation_direction_offset(actions: np.ndarray,
                                 parameters: Mapping[str, Any],
                                 _context: Mapping[str, Any]) -> np.ndarray:
    """Extend the current Cartesian translation command along its own direction."""
    result = np.asarray(actions).copy()
    indices = list(parameters.get("translation_indices", [3, 4, 5]))
    offset_m = float(parameters["offset_m"])
    vector = np.asarray(result[..., indices], dtype=np.float64)
    norm = np.linalg.norm(vector, axis=-1, keepdims=True)
    if len(indices) != 3 or not math.isfinite(offset_m):
        raise ValueError("translation-direction offset requires three indices and finite offset")
    safe_direction = np.divide(vector, norm, out=np.zeros_like(vector), where=norm > 1.0e-9)
    result[..., indices] += (safe_direction * offset_m).astype(result.dtype)
    return result


def state_hold_trigger_arm_positions(state: np.ndarray,
                                     parameters: Mapping[str, Any],
                                     context: Mapping[str, Any]) -> np.ndarray:
    """Hold policy arm-position feedback at trigger-time measured joint angles."""
    result = np.asarray(state).copy()
    indices = list(parameters.get("indices", [12, 13, 14, 15, 16, 17]))
    names = list(parameters.get("joint_names", [f"joint{i}" for i in range(1, 7)]))
    snapshot = context.get("snapshot", {}) if isinstance(context, Mapping) else {}
    measured = snapshot.get("arm", {}).get("measured_joint_positions_rad", {})
    if len(indices) != len(names) or not isinstance(measured, Mapping):
        raise ValueError("invalid trigger-time arm state hold parameters")
    values = np.asarray([measured[name] for name in names], dtype=result.dtype)
    if not np.isfinite(values).all():
        raise ValueError("trigger-time arm positions must be finite")
    result[..., indices] = values
    return result


def gripper_opening_feedback_offset(state: np.ndarray,
                                    parameters: Mapping[str, Any],
                                    _context: Mapping[str, Any]) -> np.ndarray:
    """Convert a physical opening-width bias to the policy's closed fraction."""
    result = np.asarray(state).copy()
    index = int(parameters.get("index", 24))
    opening_offset = float(parameters["opening_offset_m"])
    maximum_opening = float(parameters["calibrated_maximum_opening_m"])
    if not math.isfinite(opening_offset) or not math.isfinite(maximum_opening) or maximum_opening <= 0.0:
        raise ValueError("invalid gripper opening feedback calibration")
    result[..., index] = np.clip(
        result[..., index] - opening_offset / maximum_opening, 0.0, 1.0
    )
    return result


def joint_encoder_position_feedback(sample: Mapping[str, Any],
                                    parameters: Mapping[str, Any],
                                    context: Mapping[str, Any]) -> dict[str, Any]:
    """Attack one explicitly registered low-level M20 joint position channel."""
    if not isinstance(sample, Mapping):
        raise TypeError("joint encoder sample must be a mapping")
    names = tuple(str(name) for name in sample.get("joint_names", ()))
    positions = np.asarray(sample.get("position_rad"), dtype=np.float64)
    if not names or positions.shape != (len(names),) or not np.isfinite(positions).all():
        raise ValueError("joint encoder sample requires aligned finite positions")
    joint_name = str(parameters["joint_name"])
    if joint_name not in names:
        raise KeyError(f"registered joint is absent from encoder sample: {joint_name}")
    index = names.index(joint_name)
    operation = str(parameters.get("operation", "offset"))
    result_positions = positions.copy()
    if operation == "offset":
        offset = float(parameters["offset_rad"])
        if not math.isfinite(offset) or abs(offset) > math.pi:
            raise ValueError("joint encoder position offset must be finite and within pi")
        result_positions[index] += offset
    elif operation == "hold_trigger":
        snapshot = context.get("snapshot", {}) if isinstance(context, Mapping) else {}
        locomotion = snapshot.get("locomotion", {}) if isinstance(snapshot, Mapping) else {}
        measured = locomotion.get("measured_joint_positions_rad", {}) if isinstance(locomotion, Mapping) else {}
        if not isinstance(measured, Mapping) or joint_name not in measured:
            raise KeyError(f"trigger snapshot is missing registered joint {joint_name}")
        value = float(measured[joint_name])
        if not math.isfinite(value):
            raise ValueError("trigger-time registered joint position must be finite")
        result_positions[index] = value
    else:
        raise ValueError(f"unsupported joint encoder operation: {operation!r}")
    result = dict(sample)
    result["position_rad"] = result_positions
    return result

def lidar_object_dropout_frame(frame: Mapping[str, Any], parameters: Mapping[str, Any],
                               _context: Mapping[str, Any]) -> dict[str, Any]:
    attacked, _audit = ObjectPointDropout.from_parameters(parameters).apply(frame)
    return attacked


def _lidar_points(frame: Mapping[str, Any]) -> np.ndarray:
    points = np.asarray(frame["points"], dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError(f"LiDAR points must be finite Nx3, got {points.shape}")
    return points


def _range_pair(parameters: Mapping[str, Any], name: str, default: list[float]) -> tuple[float, float]:
    value = np.asarray(parameters.get(name, default), dtype=np.float64)
    if value.shape != (2,) or not np.isfinite(value).all() or value[0] > value[1]:
        raise ValueError(f"{name} must be an ordered pair of finite numbers")
    return float(value[0]), float(value[1])


def _append_lidar_points(frame: Mapping[str, Any], points: np.ndarray,
                         phantom: np.ndarray, parameters: Mapping[str, Any]) -> dict[str, Any]:
    """Append returns while preserving every per-point array's leading dimension."""
    result = dict(frame)
    result["points"] = np.ascontiguousarray(np.concatenate((points, phantom), axis=0))
    for key, value in frame.items():
        if key == "points":
            continue
        array = np.asarray(value)
        if array.ndim == 0 or len(array) != len(points):
            continue
        shape = (len(phantom), *array.shape[1:])
        if key == "object_ids":
            fill_value = int(parameters.get("phantom_object_id", -1))
        elif np.issubdtype(array.dtype, np.floating):
            fill_value = float(parameters.get("phantom_float_fill", 0.0))
        else:
            fill_value = int(parameters.get("phantom_integer_fill", 0))
        padding = np.full(shape, fill_value, dtype=array.dtype)
        result[key] = np.ascontiguousarray(np.concatenate((array, padding), axis=0))
    return result


def lidar_phantom_add_frame(frame: Mapping[str, Any], parameters: Mapping[str, Any],
                            context: Mapping[str, Any]) -> dict[str, Any]:
    """Append deterministic phantom returns in the M20 LiDAR sensor frame."""
    points = _lidar_points(frame)
    count = int(parameters.get("point_count", parameters.get("count", parameters.get("phantom_count", 180))))
    if not 1 <= count <= 100_000:
        raise ValueError("LiDAR phantom count must be within [1, 100000]")
    x_range = _range_pair(parameters, "x_range_m", [0.35, 0.85])
    y_range = _range_pair(parameters, "y_range_m", [-0.2, 0.2])
    z_range = _range_pair(parameters, "z_range_m", [-0.30, -0.26])
    runtime = dict(context.get("runtime", {}))
    step = int(runtime.get("physics_step", 0)) // int(parameters.get("seed_step_divisor", 1))
    seed = int(parameters.get("seed", 0))
    sampling = str(parameters.get("sampling", "seeded_uniform"))
    if str(parameters.get("geometry", "")) == "wall":
        sampling = "dense_wall"
    if sampling == "wall_grid":
        # Dense vertical wall in the LiDAR sensor frame.  x_range controls
        # thickness, y_range width, and z_range height.
        rows = int(parameters.get("rows", 64))
        columns = int(parameters.get("columns", 128))
        layers = int(parameters.get("layers", 1))
        if not (2 <= rows <= 1024 and 2 <= columns <= 2048 and 1 <= layers <= 64):
            raise ValueError("wall_grid rows/columns/layers exceed bounded ranges")
        generated = rows * columns * layers
        if generated > 100_000:
            raise ValueError("wall_grid may generate at most 100000 points")
        xx, yy, zz = np.meshgrid(
            np.linspace(*x_range, layers),
            np.linspace(*y_range, columns),
            np.linspace(*z_range, rows),
            indexing="ij",
        )
        phantom = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel())).astype(np.float32)
        count = generated
    elif sampling == "dense_wall":
        x_m = float(parameters.get("x_m", 1.2))
        thickness = float(parameters.get("thickness_m", 0.08))
        noise = float(parameters.get("noise_std_m", 0.01))
        if thickness <= 0 or noise < 0:
            raise ValueError("dense wall thickness must be positive and noise non-negative")
        rng = np.random.default_rng(np.random.SeedSequence([seed, step]))
        phantom = np.column_stack((
            rng.uniform(x_m - thickness / 2.0, x_m + thickness / 2.0, count),
            rng.uniform(*y_range, count), rng.uniform(*z_range, count),
        )).astype(np.float32)
        if noise:
            phantom += rng.normal(0.0, noise, phantom.shape).astype(np.float32)
    elif sampling == "seeded_uniform":
        rng = np.random.default_rng(np.random.SeedSequence([seed, step]))
        phantom = np.column_stack((
            rng.uniform(*x_range, count), rng.uniform(*y_range, count), rng.uniform(*z_range, count)
        )).astype(np.float32)
    elif sampling == "linspace":
        phantom = np.column_stack((
            np.linspace(*x_range, count), np.linspace(*y_range, count), np.linspace(*z_range, count)
        )).astype(np.float32)
    else:
        raise ValueError("LiDAR phantom sampling must be 'wall_grid', 'dense_wall', 'seeded_uniform', or 'linspace'")
    result = _append_lidar_points(frame, points, phantom, parameters)
    result["attack_metadata"] = {
        "kind": "phantom_wall" if sampling in {"wall_grid", "dense_wall"} else "phantom_add",
        "sampling": sampling, "clean_point_count": len(points),
        "added_point_count": count, "attacked_point_count": len(points) + count,
        "clean_sha256": hashlib.sha256(np.ascontiguousarray(points).tobytes()).hexdigest(),
        "attacked_sha256": hashlib.sha256(np.ascontiguousarray(result["points"]).tobytes()).hexdigest(),
        "wall_aabb_sensor_frame": {
            "min": phantom.min(axis=0).astype(float).tolist(),
            "max": phantom.max(axis=0).astype(float).tolist(),
        },
        "handler": "lidar_phantom_add_frame",
        "frame_index": context.get("runtime", {}).get("physics_step"),
        "elapsed_attack_s": context.get("runtime", {}).get("elapsed_attack_s"),
    }
    return result


def _lidar_selection(frame: Mapping[str, Any], parameters: Mapping[str, Any], count: int) -> tuple[np.ndarray, str]:
    target = parameters.get("target_prim_path")
    if target:
        object_ids, stable_id_map = frame.get("object_ids"), frame.get("stable_id_map")
        if object_ids is None or stable_id_map is None:
            raise ValueError("targeted LiDAR drift requires object_ids and stable_id_map")
        target = str(target).rstrip("/")
        descendants = bool(parameters.get("include_descendants", True))
        matching = {
            stable_id for stable_id, path in stable_id_map.items()
            if str(path) == target or (descendants and str(path).startswith(target + "/"))
        }
        if not matching:
            return np.zeros(count, dtype=bool), "target_not_observed_in_this_frame"
        ids = np.asarray(object_ids)
        if ids.shape != (count,):
            raise ValueError("object_ids must align with LiDAR points")
        return np.isin(ids, list(matching)), "stable_object_id"
    points = _lidar_points(frame)
    for key, coordinates in (("sensor_aabb", points), ("world_aabb", None)):
        bounds = parameters.get(key)
        if bounds is None:
            continue
        minimum = np.asarray(bounds["min"], dtype=np.float64)
        maximum = np.asarray(bounds["max"], dtype=np.float64)
        if minimum.shape != (3,) or maximum.shape != (3,) or np.any(minimum > maximum):
            raise ValueError(f"{key} min/max must be ordered xyz vectors")
        if coordinates is None:
            transform = np.asarray(frame.get("sensor_to_world", np.eye(4)), dtype=np.float64)
            if transform.shape != (4, 4):
                raise ValueError("sensor_to_world must be 4x4")
            coordinates = (transform @ np.column_stack((points, np.ones(count))).T).T[:, :3]
        return np.all((coordinates >= minimum) & (coordinates <= maximum), axis=1), key
    return np.ones(count, dtype=bool), "all_points"


def lidar_point_drift_frame(frame: Mapping[str, Any], parameters: Mapping[str, Any],
                            _context: Mapping[str, Any]) -> dict[str, Any]:
    """Apply range, angular, and Cartesian drift before navigation consumes a frame."""
    points = _lidar_points(frame)
    selected, selection_method = _lidar_selection(frame, parameters, len(points))
    scale = float(parameters.get("range_scale", 1.0))
    yaw = float(parameters.get("yaw_bias_rad", 0.0))
    offset = np.asarray(parameters.get("translation_xyz_m", [0.0, 0.0, 0.0]), dtype=np.float32)
    if not np.isfinite(scale) or not 0.1 <= scale <= 5.0:
        raise ValueError("LiDAR range_scale must be within [0.1, 5.0]")
    if not np.isfinite(yaw) or abs(yaw) > np.pi:
        raise ValueError("LiDAR yaw_bias_rad must be within [-pi, pi]")
    if offset.shape != (3,) or not np.isfinite(offset).all():
        raise ValueError("LiDAR translation_xyz_m must be a finite xyz vector")
    cosine, sine = np.cos(yaw), np.sin(yaw)
    rotation = np.asarray([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    attacked_points = points.copy()
    attacked_points[selected] = (points[selected] * scale) @ rotation.T + offset
    result = dict(frame)
    result["points"] = np.ascontiguousarray(attacked_points)
    result["attack_metadata"] = {
        "kind": "point_drift", "selection_method": selection_method,
        "target_prim_path": parameters.get("target_prim_path"),
        "modified_point_count": int(selected.sum()),
        "range_scale": scale, "yaw_bias_rad": yaw, "translation_xyz_m": offset.tolist(),
    }
    return result

def imu_sample_transform(sample: Mapping[str, Any], parameters: Mapping[str, Any],
                         _context: Mapping[str, Any]) -> dict[str, np.ndarray]:
    return IMUTransform(str(parameters["attack"]), parameters).apply(sample)


def imu_oscillating_bias(sample: Mapping[str, Any], parameters: Mapping[str, Any],
                         context: Mapping[str, Any]) -> dict[str, np.ndarray]:
    result = {key: np.asarray(value, dtype=np.float64).copy() for key, value in sample.items()}
    key = str(parameters.get("key", "angular_velocity"))
    if key not in result:
        raise KeyError(f"IMU sample does not contain {key!r}")
    axis = int(parameters.get("axis", 1))
    amplitude = float(parameters.get("amplitude", 0.03))
    frequency_hz = float(parameters.get("frequency_hz", 2.0))
    phase_rad = float(parameters.get("phase_rad", 0.0))
    time_s = float(context.get("runtime", {}).get("time_s", 0.0))
    if axis not in range(result[key].shape[-1]) or not all(
        math.isfinite(value) for value in (amplitude, frequency_hz, phase_rad, time_s)
    ):
        raise ValueError("invalid oscillating IMU parameters")
    result[key][axis] += amplitude * math.sin(2.0 * math.pi * frequency_hz * time_s + phase_rad)
    return result


def force_torque_set_value(
    sample: Mapping[str, Any],
    parameters: Mapping[str, Any],
    _context: Mapping[str, Any],
) -> dict[str, Any]:
    """Replace one Piper TCP force/torque sample with bounded finite values.

    ``force_xyz_n`` and ``torque_xyz_nm`` are expressed in the sample's
    declared sensor frame.  ``grip_force_n`` is the scalar feedback consumed
    by the Runner's grasp-feedback gate; it defaults to the force norm.
    """

    result = dict(sample)
    for required in ("force_xyz_n", "torque_xyz_nm", "grip_force_n"):
        if required not in sample:
            raise KeyError(f"force/torque sample does not contain {required!r}")

    force = np.asarray(
        parameters.get("force_xyz_n", sample["force_xyz_n"]), dtype=np.float64
    )
    torque = np.asarray(
        parameters.get("torque_xyz_nm", sample["torque_xyz_nm"]), dtype=np.float64
    )
    grip_force = float(parameters.get("grip_force_n", np.linalg.norm(force)))
    if force.shape != (3,) or torque.shape != (3,):
        raise ValueError("force_xyz_n and torque_xyz_nm must be xyz vectors")
    if not np.isfinite(force).all() or not np.isfinite(torque).all() or not math.isfinite(grip_force):
        raise ValueError("force/torque attack values must be finite")
    if np.any(np.abs(force) > 1000.0) or np.any(np.abs(torque) > 100.0):
        raise ValueError("force/torque attack values exceed the bounded range")
    if grip_force < 0.0 or grip_force > 1000.0:
        raise ValueError("grip_force_n must be within [0, 1000]")

    result["force_xyz_n"] = force
    result["torque_xyz_nm"] = torque
    result["grip_force_n"] = grip_force
    return result


def wheel_encoder_velocity_bias(
    sample: Mapping[str, Any],
    parameters: Mapping[str, Any],
    _context: Mapping[str, Any],
) -> dict[str, Any]:
    """Add a linear-speed-equivalent bias to all four wheel encoders."""
    if not isinstance(sample, Mapping):
        raise TypeError("wheel encoder sample must be a mapping")
    names = tuple(str(name) for name in sample.get("joint_names", ()))
    angular_velocity = np.asarray(
        sample.get("angular_velocity_rad_s"), dtype=np.float64
    )
    wheel_radius_m = float(sample.get("wheel_radius_m", np.nan))
    linear_bias_mps = float(parameters["linear_velocity_bias_mps"])
    if len(names) != 4 or angular_velocity.shape != (4,):
        raise ValueError("wheel encoder sample must contain four named velocities")
    if not np.isfinite(angular_velocity).all():
        raise ValueError("wheel encoder velocities must be finite")
    if not math.isfinite(wheel_radius_m) or wheel_radius_m <= 0.0:
        raise ValueError("wheel_radius_m must be finite and positive")
    if not math.isfinite(linear_bias_mps):
        raise ValueError("linear_velocity_bias_mps must be finite")
    result = dict(sample)
    result["angular_velocity_rad_s"] = (
        angular_velocity + linear_bias_mps / wheel_radius_m
    )
    return result

def joint_target_offset(targets: Mapping[str, Any], parameters: Mapping[str, Any],
                        _context: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(targets)
    positions = dict(result.get("positions", {}))
    for name, offset in parameters.get("joint_offsets_rad", {}).items():
        if name not in positions:
            raise KeyError(f"joint target not present: {name}")
        positions[name] = float(positions[name]) + float(offset)
    result["positions"] = positions
    return result


def arm_joint_hold_measured(
    targets: Mapping[str, Any],
    parameters: Mapping[str, Any],
    context: Mapping[str, Any],
) -> dict[str, Any]:
    """Freeze only Piper arm position targets at trigger-time measured angles."""

    result = dict(targets)
    positions = dict(result.get("positions", {}))
    snapshot = context.get("snapshot", {})
    arm = snapshot.get("arm", {}) if isinstance(snapshot, Mapping) else {}
    measured = (
        arm.get("measured_joint_positions_rad", {})
        if isinstance(arm, Mapping)
        else {}
    )
    scope = str(parameters.get("joint_scope", "arm"))
    defaults = {
        "arm": tuple(f"joint{index}" for index in range(1, 7)),
        "gripper": ("joint7", "joint8"),
    }
    if scope not in defaults:
        raise ValueError(f"unsupported measured joint hold scope: {scope!r}")
    joint_names = tuple(
        str(name)
        for name in parameters.get("joint_names", defaults[scope])
    )
    if joint_names != defaults[scope]:
        raise ValueError(f"measured joint hold must target exactly {defaults[scope]}")
    if not isinstance(measured, Mapping):
        raise TypeError("trigger snapshot arm measurements must be a mapping")
    for name in joint_names:
        if name not in positions:
            raise KeyError(f"arm joint target not present: {name}")
        if name not in measured:
            raise KeyError(f"trigger snapshot is missing measured angle: {name}")
        value = float(measured[name])
        if not math.isfinite(value):
            raise ValueError(f"non-finite trigger-time measured angle for {name}")
        positions[name] = value
    result["positions"] = positions
    return result


def arm_joint_hold_trigger_command(
    targets: Mapping[str, Any],
    parameters: Mapping[str, Any],
    context: Mapping[str, Any],
) -> dict[str, Any]:
    """Freeze Piper targets at the command values captured at trigger time."""
    result = dict(targets)
    positions = dict(result.get("positions", {}))
    snapshot = context.get("snapshot", {}) if isinstance(context, Mapping) else {}
    arm = snapshot.get("arm", {}) if isinstance(snapshot, Mapping) else {}
    commanded = arm.get("commanded_joint_positions_rad", {}) if isinstance(arm, Mapping) else {}
    names = tuple(str(name) for name in parameters.get("joint_names", ()))
    if names != tuple(f"joint{index}" for index in range(1, 7)):
        raise ValueError("trigger command hold must target Piper joint1..joint6")
    if not isinstance(commanded, Mapping):
        raise TypeError("trigger snapshot commanded arm positions must be a mapping")
    for name in names:
        if name not in positions or name not in commanded:
            raise KeyError(f"missing trigger command for {name}")
        value = float(commanded[name])
        if not math.isfinite(value):
            raise ValueError(f"non-finite trigger command for {name}")
        positions[name] = value
    result["positions"] = positions
    return result

from .bboard_visual import picture_bboard_target

STANDARD_HANDLERS = {
    "picture_bboard_target": picture_bboard_target,
    "picture_red_to_green": picture_red_to_green,
    "picture_region_patch": picture_region_patch,
    "state_offset": state_offset,
    "state_set_values": state_set_values,
    "replace_prompt": replace_prompt,
    "replace_prompt_sequence": replace_prompt_sequence,
    "append_target_coordinates": append_target_coordinates,
    "scale_actions": scale_actions,
    "affine_actions": affine_actions,
    "gripper_directional_scale": gripper_directional_scale,
    "gripper_opening_command_offset": gripper_opening_command_offset,
    "base_velocity_toward_object": base_velocity_toward_object,
    "world_frame_planar_velocity_offset": world_frame_planar_velocity_offset,
    "world_frame_tcp_translation_offset": world_frame_tcp_translation_offset,
    "translation_direction_offset": translation_direction_offset,
    "state_hold_trigger_arm_positions": state_hold_trigger_arm_positions,
    "gripper_opening_feedback_offset": gripper_opening_feedback_offset,
    "joint_encoder_position_feedback": joint_encoder_position_feedback,
    "lidar_object_dropout_frame": lidar_object_dropout_frame,
    "lidar_phantom_add_frame": lidar_phantom_add_frame,
    "lidar_point_drift_frame": lidar_point_drift_frame,
    "imu_sample_transform": imu_sample_transform,
    "imu_oscillating_bias": imu_oscillating_bias,
    "force_torque_set_value": force_torque_set_value,
    "wheel_encoder_velocity_bias": wheel_encoder_velocity_bias,
    "joint_target_offset": joint_target_offset,
    "arm_joint_hold_measured": arm_joint_hold_measured,
    "arm_joint_hold_trigger_command": arm_joint_hold_trigger_command,
    **IMMEDIATE_HANDLERS,
}

from paper200.attack_runtime import roi_dropout, world_action_offset, projected_patch
STANDARD_HANDLERS.update(paper200_roi_dropout=roi_dropout, paper200_world_action_offset=world_action_offset, paper200_projected_patch=projected_patch)

from paper200.attack_runtime import world_imu_bias
STANDARD_HANDLERS["paper200_world_imu_bias"] = world_imu_bias
