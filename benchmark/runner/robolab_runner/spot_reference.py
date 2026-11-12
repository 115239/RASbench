# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic LiDAR A*/DWA reference policy for frozen Spot tasks.

The service consumes only the negotiated public observation leaves.  Planner
state stays in memory and is never returned to the Runner or written into the
submission artifacts.
"""

from __future__ import annotations

import argparse
import heapq
import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

import numpy as np

from robolab_runner.errors import ProtocolError
from robolab_runner.policy.handler import PolicyHandler, serve_policy

GRID_RESOLUTION_M = 0.15
GRID_ORIGIN_XY = (-12.0, -4.5)
GRID_SHAPE = (60, 160)
LANE_Y_LIMIT_M = 2.2
ROBOT_RADIUS_M = 0.43
# Match the clearance used by the proven warehouse LiDAR navigation pipeline.
# A 0.40 m margin expands obstacles by six 0.15 m grid cells and seals the
# randomized S-aisle even though Spot physically has room to pass.  The 0.15 m
# margin expands by four cells and preserves a conservative 0.58 m radius.
SAFETY_MARGIN_M = 0.15
COLLISION_RADIUS_M = ROBOT_RADIUS_M + SAFETY_MARGIN_M
LIDAR_MAX_PLANNING_RANGE_M = 6.0
GOAL_TOLERANCE_M = 0.55
REPLAN_INTERVAL_STEPS = 10
MIN_REPLAN_INTERVAL_STEPS = 3
MAP_CHANGE_CELL_THRESHOLD = 18
MAPPING_WARMUP_STEPS = 5
TURN_IN_PLACE_HEADING_RAD = 0.70
DEFAULT_TASK_ENV = "SpotWarehouseNavigation-v0"


def _wrap_angle(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


def _yaw_from_wxyz(quaternion: np.ndarray) -> float:
    w, x, y, z = (float(value) for value in quaternion)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _bresenham(x0: int, y0: int, x1: int, y1: int) -> list[tuple[int, int]]:
    points: list[tuple[int, int]] = []
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    step_x = 1 if x0 < x1 else -1
    step_y = 1 if y0 < y1 else -1
    error = dx + dy
    while True:
        points.append((x0, y0))
        if x0 == x1 and y0 == y1:
            return points
        twice = 2 * error
        if twice >= dy:
            error += dy
            x0 += step_x
        if twice <= dx:
            error += dx
            y0 += step_y


class OccupancyGrid:
    """Small deterministic decaying grid populated from the public 2-D scan."""

    def __init__(self) -> None:
        self.log_odds = np.zeros(GRID_SHAPE, dtype=np.float32)
        self.observed = np.zeros(GRID_SHAPE, dtype=np.bool_)
        self.static_occupied = np.zeros(GRID_SHAPE, dtype=np.bool_)
        for grid_y in range(GRID_SHAPE[0]):
            _world_x, world_y = self.grid_to_world(0, grid_y)
            if world_y < -LANE_Y_LIMIT_M + 0.04 or world_y > LANE_Y_LIMIT_M - 0.04:
                self.static_occupied[grid_y, :] = True

    @staticmethod
    def in_bounds(cell: tuple[int, int]) -> bool:
        return 0 <= cell[0] < GRID_SHAPE[1] and 0 <= cell[1] < GRID_SHAPE[0]

    @staticmethod
    def world_to_grid(world_x: float, world_y: float) -> tuple[int, int]:
        return (
            int(math.floor((world_x - GRID_ORIGIN_XY[0]) / GRID_RESOLUTION_M)),
            int(math.floor((world_y - GRID_ORIGIN_XY[1]) / GRID_RESOLUTION_M)),
        )

    @staticmethod
    def grid_to_world(grid_x: int, grid_y: int) -> tuple[float, float]:
        return (
            GRID_ORIGIN_XY[0] + (grid_x + 0.5) * GRID_RESOLUTION_M,
            GRID_ORIGIN_XY[1] + (grid_y + 0.5) * GRID_RESOLUTION_M,
        )

    def update(self, scan: np.ndarray, position_xy: np.ndarray, yaw: float) -> int:
        previous = self.occupied()
        self.log_odds *= np.float32(0.985)
        angles = -math.pi + (np.arange(scan.size, dtype=np.float32) + 0.5) * (2.0 * math.pi / scan.size)
        valid = np.isfinite(scan) & (scan > 0.30) & (scan < LIDAR_MAX_PLANNING_RANGE_M)
        if np.any(valid):
            distances = scan[valid]
            world_angles = angles[valid] + np.float32(yaw)
            endpoints = np.column_stack(
                (
                    position_xy[0] + distances * np.cos(world_angles),
                    position_xy[1] + distances * np.sin(world_angles),
                )
            )
            robot_cell = self.world_to_grid(float(position_xy[0]), float(position_xy[1]))
            free_cells: list[tuple[int, int]] = []
            hit_cells: list[tuple[int, int]] = []
            if self.in_bounds(robot_cell):
                for endpoint in endpoints:
                    end_cell = self.world_to_grid(float(endpoint[0]), float(endpoint[1]))
                    if not self.in_bounds(end_cell):
                        continue
                    ray = _bresenham(*robot_cell, *end_cell)
                    free_cells.extend(ray[:-1])
                    hit_cells.append(ray[-1])
            if free_cells:
                free = np.asarray(free_cells, dtype=np.int32)
                np.add.at(self.log_odds, (free[:, 1], free[:, 0]), -0.08)
                self.observed[free[:, 1], free[:, 0]] = True
            if hit_cells:
                hits = np.asarray(hit_cells, dtype=np.int32)
                np.add.at(self.log_odds, (hits[:, 1], hits[:, 0]), 0.75)
                self.observed[hits[:, 1], hits[:, 0]] = True
        np.clip(self.log_odds, -4.0, 4.0, out=self.log_odds)
        robot_cell = self.world_to_grid(float(position_xy[0]), float(position_xy[1]))
        if self.in_bounds(robot_cell):
            x, y = robot_cell
            self.log_odds[max(0, y - 2) : min(GRID_SHAPE[0], y + 3), max(0, x - 2) : min(GRID_SHAPE[1], x + 3)] = -4.0
        return int(np.count_nonzero(previous != self.occupied()))

    def occupied(self) -> np.ndarray:
        return self.static_occupied | (self.log_odds > 0.70)

    def inflated(self) -> np.ndarray:
        occupied = self.occupied()
        result = occupied.copy()
        radius = int(math.ceil(COLLISION_RADIUS_M / GRID_RESOLUTION_M))
        occupied_y, occupied_x = np.nonzero(occupied)
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx * dx + dy * dy > radius * radius:
                    continue
                shifted_x = occupied_x + dx
                shifted_y = occupied_y + dy
                valid = (
                    (shifted_x >= 0)
                    & (shifted_x < GRID_SHAPE[1])
                    & (shifted_y >= 0)
                    & (shifted_y < GRID_SHAPE[0])
                )
                result[shifted_y[valid], shifted_x[valid]] = True
        return result


class AStarPlanner:
    """Eight-connected A* with deterministic tie-breaking and line-of-sight simplification."""

    @staticmethod
    def _line_is_free(start: tuple[int, int], end: tuple[int, int], blocked: np.ndarray) -> bool:
        return all(not blocked[y, x] for x, y in _bresenham(*start, *end))

    def plan(
        self,
        grid: OccupancyGrid,
        start_world: np.ndarray,
        goal_world: np.ndarray,
    ) -> list[np.ndarray] | None:
        start = grid.world_to_grid(float(start_world[0]), float(start_world[1]))
        goal = grid.world_to_grid(float(goal_world[0]), float(goal_world[1]))
        if not grid.in_bounds(start) or not grid.in_bounds(goal):
            return None
        blocked = grid.inflated()
        for center_x, center_y in (start, goal):
            blocked[
                max(0, center_y - 2) : min(GRID_SHAPE[0], center_y + 3),
                max(0, center_x - 2) : min(GRID_SHAPE[1], center_x + 3),
            ] = False
        moves = (
            (-1, 0, 1.0),
            (0, -1, 1.0),
            (0, 1, 1.0),
            (1, 0, 1.0),
            (-1, -1, math.sqrt(2.0)),
            (-1, 1, math.sqrt(2.0)),
            (1, -1, math.sqrt(2.0)),
            (1, 1, math.sqrt(2.0)),
        )
        open_heap: list[tuple[float, float, int, int]] = [(0.0, 0.0, start[0], start[1])]
        costs = {start: 0.0}
        parent: dict[tuple[int, int], tuple[int, int]] = {}
        found = False
        while open_heap:
            _priority, queued_cost, current_x, current_y = heapq.heappop(open_heap)
            current = (current_x, current_y)
            if queued_cost != costs.get(current):
                continue
            if current == goal:
                found = True
                break
            for dx, dy, move_cost in moves:
                neighbor = (current_x + dx, current_y + dy)
                if not grid.in_bounds(neighbor) or blocked[neighbor[1], neighbor[0]]:
                    continue
                if dx and dy and (blocked[current_y, neighbor[0]] or blocked[neighbor[1], current_x]):
                    continue
                unknown_penalty = 0.18 if not grid.observed[neighbor[1], neighbor[0]] else 0.0
                new_cost = queued_cost + move_cost + unknown_penalty
                if new_cost >= costs.get(neighbor, math.inf):
                    continue
                costs[neighbor] = new_cost
                parent[neighbor] = current
                heuristic = math.hypot(neighbor[0] - goal[0], neighbor[1] - goal[1])
                heapq.heappush(open_heap, (new_cost + heuristic, new_cost, neighbor[0], neighbor[1]))
        if not found:
            return None
        cells = [goal]
        while cells[-1] != start:
            cells.append(parent[cells[-1]])
        cells.reverse()
        simplified = [cells[0]]
        anchor = 0
        while anchor < len(cells) - 1:
            candidate = len(cells) - 1
            while candidate > anchor + 1 and not self._line_is_free(cells[anchor], cells[candidate], blocked):
                candidate -= 1
            simplified.append(cells[candidate])
            anchor = candidate
        return [np.asarray(grid.grid_to_world(x, y), dtype=np.float32) for x, y in simplified]


class DWAController:
    """Constrained dynamic-window controller producing only base velocity actions."""

    def __init__(self) -> None:
        self.last_action = np.zeros(3, dtype=np.float32)

    @staticmethod
    def _subgoal(position_xy: np.ndarray, path: list[np.ndarray] | None, lookahead_m: float = 1.0) -> np.ndarray | None:
        if not path:
            return None
        points = np.asarray(path, dtype=np.float32)
        if len(points) == 1:
            return points[0]
        best_distance = math.inf
        best_segment = 0
        best_fraction = 0.0
        for index, (start, end) in enumerate(zip(points[:-1], points[1:], strict=True)):
            segment = end - start
            length_squared = float(np.dot(segment, segment))
            if length_squared <= 1e-12:
                fraction = 0.0
                projection = start
            else:
                fraction = float(np.clip(np.dot(position_xy - start, segment) / length_squared, 0.0, 1.0))
                projection = start + np.float32(fraction) * segment
            distance = float(np.linalg.norm(position_xy - projection))
            if distance < best_distance:
                best_distance = distance
                best_segment = index
                best_fraction = fraction

        segment_start = points[best_segment]
        segment_end = points[best_segment + 1]
        segment = segment_end - segment_start
        segment_length = float(np.linalg.norm(segment))
        remaining = float(lookahead_m)
        if segment_length > 1e-12:
            available = (1.0 - best_fraction) * segment_length
            if remaining <= available:
                return segment_start + np.float32(best_fraction + remaining / segment_length) * segment
            remaining -= available
        for index in range(best_segment + 1, len(points) - 1):
            segment = points[index + 1] - points[index]
            segment_length = float(np.linalg.norm(segment))
            if segment_length <= 1e-12:
                continue
            if remaining <= segment_length:
                return points[index] + np.float32(remaining / segment_length) * segment
            remaining -= segment_length
        return points[-1]

    @staticmethod
    def _scan_points(scan: np.ndarray) -> np.ndarray:
        angles = -math.pi + (np.arange(scan.size, dtype=np.float32) + 0.5) * (2.0 * math.pi / scan.size)
        valid = np.isfinite(scan) & (scan > 0.28) & (scan < LIDAR_MAX_PLANNING_RANGE_M)
        return np.column_stack((scan[valid] * np.cos(angles[valid]), scan[valid] * np.sin(angles[valid]))).astype(
            np.float32,
            copy=False,
        )

    @staticmethod
    def _trajectory_clearance(
        points: np.ndarray,
        forward_velocity: float,
        lateral_velocity: float,
        yaw_rate: float,
    ) -> tuple[float, np.ndarray]:
        x = y = heading = 0.0
        clearance = LIDAR_MAX_PLANNING_RANGE_M
        for _ in range(8):
            heading += yaw_rate * 0.2
            x += (forward_velocity * math.cos(heading) - lateral_velocity * math.sin(heading)) * 0.2
            y += (forward_velocity * math.sin(heading) + lateral_velocity * math.cos(heading)) * 0.2
            if points.size:
                clearance = min(clearance, float(np.min(np.hypot(points[:, 0] - x, points[:, 1] - y))))
        return clearance, np.asarray([x, y], dtype=np.float32)

    def choose(
        self,
        position_xy: np.ndarray,
        yaw: float,
        goal_relative: np.ndarray,
        scan: np.ndarray,
        path: list[np.ndarray] | None,
    ) -> np.ndarray:
        if float(np.linalg.norm(goal_relative[:2])) <= GOAL_TOLERANCE_M:
            desired = np.zeros(3, dtype=np.float32)
            return self._limit_acceleration(desired)
        target = self._subgoal(position_xy, path)
        if target is None:
            return self._limit_acceleration(np.zeros(3, dtype=np.float32))
        delta = target - position_xy
        cosine, sine = math.cos(yaw), math.sin(yaw)
        target_body = np.asarray(
            [cosine * delta[0] + sine * delta[1], -sine * delta[0] + cosine * delta[1]],
            dtype=np.float32,
        )
        target_distance = float(np.linalg.norm(target_body))
        target_heading = math.atan2(float(target_body[1]), float(target_body[0]))
        points = self._scan_points(scan)
        best: tuple[float, float, float, float] | None = None
        if abs(target_heading) > TURN_IN_PLACE_HEADING_RAD:
            forward_velocities = (0.0,)
            lateral_velocities = (0.0,)
        elif abs(target_heading) > 0.35:
            forward_velocities = (0.0, 0.14, 0.28)
            lateral_velocities = (-0.10, 0.0, 0.10)
        else:
            forward_velocities = (0.0, 0.14, 0.28, 0.42, 0.52)
            lateral_velocities = (-0.10, 0.0, 0.10)
        for forward_velocity in forward_velocities:
            # Keep lateral correction small and only use it while the path is
            # in the forward half-plane.  In particular, do not translate
            # toward cargo while turning around after a phantom-LiDAR window.
            for lateral_velocity in lateral_velocities:
                for yaw_rate in np.linspace(-0.70, 0.70, 15):
                    rate = float(yaw_rate)
                    clearance, local_end = self._trajectory_clearance(
                        points,
                        forward_velocity,
                        lateral_velocity,
                        rate,
                    )
                    if clearance < COLLISION_RADIUS_M:
                        continue
                    end_world_y = (
                        float(position_xy[1])
                        + sine * float(local_end[0])
                        + cosine * float(local_end[1])
                    )
                    if abs(end_world_y) > LANE_Y_LIMIT_M - ROBOT_RADIUS_M:
                        continue
                    progress = target_distance - float(np.linalg.norm(target_body - local_end))
                    heading_score = math.cos(_wrap_angle(target_heading - rate * 0.8))
                    score = (
                        4.2 * progress
                        + 1.5 * heading_score
                        + 0.45 * min(clearance, 2.0)
                        + forward_velocity
                        - 0.20 * abs(lateral_velocity)
                        - 0.12 * abs(rate)
                    )
                    candidate = (score, forward_velocity, lateral_velocity, rate)
                    if best is None or candidate > best:
                        best = candidate
        if best is None:
            left = float(np.min(scan[(scan.size // 2) : (3 * scan.size // 4)]))
            right = float(np.min(scan[(scan.size // 4) : (scan.size // 2)]))
            desired = np.asarray([0.0, 0.0, 0.55 if left >= right else -0.55], dtype=np.float32)
        else:
            _score, forward_velocity, lateral_velocity, yaw_rate = best
            desired = np.asarray([forward_velocity, lateral_velocity, yaw_rate], dtype=np.float32)
        return self._limit_acceleration(desired)

    def _limit_acceleration(self, desired: np.ndarray) -> np.ndarray:
        delta = desired - self.last_action
        delta[0] = np.clip(delta[0], -0.08, 0.08)
        delta[1] = np.clip(delta[1], -0.06, 0.06)
        delta[2] = np.clip(delta[2], -0.18, 0.18)
        self.last_action = (self.last_action + delta).astype(np.float32)
        return self.last_action.copy()


@dataclass
class _PlannerSession:
    grid: OccupancyGrid = field(default_factory=OccupancyGrid)
    planner: AStarPlanner = field(default_factory=AStarPlanner)
    controller: DWAController = field(default_factory=DWAController)
    path: list[np.ndarray] | None = None
    last_mapped_scan: np.ndarray | None = None
    last_replan_step: int = -REPLAN_INTERVAL_STEPS


class SpotAStarDwaPolicy(PolicyHandler):
    """Standard ZMQ LiDAR reference policy used by deterministic Spot acceptance."""

    policy_id = "robolab-spot-astar-dwa"
    policy_version = "1"

    def __init__(self, *, task_envs: Iterable[str] | None = None) -> None:
        super().__init__()
        raw_task_envs = (DEFAULT_TASK_ENV,) if task_envs is None else tuple(task_envs)
        if not raw_task_envs or any(not isinstance(value, str) or not value.strip() for value in raw_task_envs):
            raise ValueError("task_envs must contain at least one non-empty environment ID")
        self.task_envs = frozenset(value.strip() for value in raw_task_envs)
        self._planner_sessions: dict[str, _PlannerSession] = {}

    def on_handshake(self, *, episode_id: str, payload: Mapping[str, Any]) -> None:
        if payload.get("task_env") not in self.task_envs:
            allowed = ", ".join(sorted(self.task_envs))
            raise ProtocolError(f"Spot A*/DWA policy accepts only the configured task environments: {allowed}")
        if payload.get("action_dim") != 3:
            raise ProtocolError("Spot A*/DWA policy requires a three-dimensional base action")
        negotiated = {str(item["path"]) for item in payload["observation_leaves"]}
        required = {
            "proprio_obs/base",
            "proprio_obs/goal_relative",
            "sensor_obs/lidar_scan",
        }
        if not required <= negotiated:
            raise ProtocolError(f"Spot A*/DWA policy is missing observation leaves: {sorted(required - negotiated)}")
        self._planner_sessions[episode_id] = _PlannerSession()

    def reset(self, *, episode_id: str, payload: Mapping[str, Any]) -> None:
        del payload
        self._planner_sessions[episode_id] = _PlannerSession()

    def infer(self, observations: Mapping[str, np.ndarray], *, step: int, episode_id: str) -> np.ndarray:
        session = self._planner_sessions[episode_id]
        base = np.asarray(observations["proprio_obs/base"], dtype=np.float32)
        goal_relative = np.asarray(observations["proprio_obs/goal_relative"], dtype=np.float32)
        scan = np.asarray(observations["sensor_obs/lidar_scan"], dtype=np.float32)
        if base.shape != (13,) or goal_relative.shape != (3,) or scan.ndim != 1 or scan.size < 36:
            raise ProtocolError("Spot public observation shapes differ from the frozen reference policy")
        position_xy = base[:2]
        yaw = _yaw_from_wxyz(base[3:7])
        cosine, sine = math.cos(yaw), math.sin(yaw)
        goal_world = position_xy + np.asarray(
            [
                cosine * goal_relative[0] - sine * goal_relative[1],
                sine * goal_relative[0] + cosine * goal_relative[1],
            ],
            dtype=np.float32,
        )
        scan_is_fresh = session.last_mapped_scan is None or not np.array_equal(scan, session.last_mapped_scan)
        # RTX publishes a completed revolution less frequently than the
        # control loop.  Re-projecting the held body-frame scan from every new
        # robot pose smears static obstacles through the map.  On held frames,
        # advance decay and clear the robot cell without integrating rays.
        mapping_scan = scan if scan_is_fresh else np.full_like(scan, LIDAR_MAX_PLANNING_RANGE_M)
        changed_cells = session.grid.update(mapping_scan, position_xy, yaw)
        if scan_is_fresh:
            session.last_mapped_scan = scan.copy()
        # The RTX rotary producer exposes partial render slices.  Let the
        # environment assemble a complete scan and seed the occupancy grid
        # before commanding motion.  This avoids choosing a side of the first
        # obstacle from an incomplete initial slice.
        if step < MAPPING_WARMUP_STEPS:
            return np.zeros((1, 3), dtype=np.float32)

        elapsed_since_replan = step - session.last_replan_step
        replan_due = elapsed_since_replan >= REPLAN_INTERVAL_STEPS
        map_change_due = (
            changed_cells > MAP_CHANGE_CELL_THRESHOLD
            and elapsed_since_replan >= MIN_REPLAN_INTERVAL_STEPS
        )
        path_missing_due = session.path is None and elapsed_since_replan >= MIN_REPLAN_INTERVAL_STEPS
        if replan_due or map_change_due or path_missing_due:
            candidate = session.planner.plan(session.grid, position_xy, goal_world)
            # Never continue on a stale route after the current occupancy map
            # has become disconnected.  A sealed phantom window can invalidate
            # a path temporarily; wait for evidence decay and a fresh valid
            # route instead of driving through the old corridor.
            session.path = candidate
            session.last_replan_step = step
        action = session.controller.choose(position_xy, yaw, goal_relative, scan, session.path)
        return action.reshape(1, 3).astype(np.float32, copy=False)

    def end_episode(self, *, episode_id: str, payload: Mapping[str, Any]) -> None:
        del payload
        self._planner_sessions.pop(episode_id, None)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="robolab-policy-spot-reference")
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:5555")
    parser.add_argument(
        "--task-env",
        action="append",
        dest="task_envs",
        help=(
            "Exact frozen Spot task environment ID accepted by the policy; repeat to allow more than one. "
            f"Defaults to {DEFAULT_TASK_ENV}."
        ),
    )
    parser.add_argument("--max-message-bytes", type=int, default=67_108_864)
    args = parser.parse_args(argv)
    serve_policy(SpotAStarDwaPolicy(task_envs=args.task_envs), args.endpoint, max_message_bytes=args.max_message_bytes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["AStarPlanner", "DWAController", "OccupancyGrid", "SpotAStarDwaPolicy", "main"]
