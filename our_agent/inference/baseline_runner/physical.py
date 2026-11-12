"""Continuous control inside the official Collector, including its attack boundary."""
from pathlib import Path
import json
import math
import os
import pwd
import sys
import time
from urllib.parse import unquote
import numpy as np
from common import ROOT, VLN, read, write
from mission import Mission, Budget, Stop, validate_response

def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi

def aabb_gap(a, b):
    return float(np.linalg.norm(np.maximum(0., np.maximum(a[:2] - b[3:5], b[:2] - a[3:5]))))

class Backend:
    def __init__(self, c, policy, runtime, cfg):
        from pxr import Usd, UsdGeom, PhysicsSchemaTools
        from omni.physx import get_physx_scene_query_interface, get_physx_simulation_interface
        self.c, self.policy, self.r, self.cfg = c, policy, runtime, cfg
        self.out = runtime["args"].output
        self.task = cfg["task"]
        self.dt = 1. / cfg.get("action_hz", 10)
        self.stage = c.final_stage
        self.bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy, UsdGeom.Tokens.guide], useExtentsHint=False, ignoreVisibility=True)
        self.query = get_physx_scene_query_interface()
        self.phase, self.epoch = "BOOT", 0
        self.arm_indices = np.asarray([c.index[f"joint{i}"] for i in range(1, 7)])
        self.stowed = np.asarray(c.robot.get_joint_positions())[self.arm_indices].copy()
        self.arm = np.r_[self.stowed, c.gripper_max_opening].astype(np.float32)
        self.initial_pose = self.pose()
        self.initial_object_z = float(c.cube.get_world_pose()[0][2])
        self.collision = None
        self.request_count = 0
        self.grasp_count = 0
        self.carry_reference = None
        self.carrying = False
        self.max_lift = 0.
        self.grid = None
        planner_config = c._lidar_planner.config
        old_tolerance = planner_config.goal_tolerance_m
        planner_config.goal_tolerance_m = .025
        write(self.out / "NAV_CONTROL_BINDING.json", {"lidar_goal_tolerance_before_m": old_tolerance,
              "lidar_goal_tolerance_m": .025, "dock_position_tolerance_m": .04,
              "early_stop_radius_m": .18, "early_stop_requires_exact_rubric": True,
              "obstacle_stop_distance_m": planner_config.stop_distance_m,
              "obstacle_side_clearance_m": planner_config.side_clearance_m,
              "frozen_rubric_thresholds_unchanged": True})
        self.events = (self.out / "PHASE_EVENTS.jsonl").open("x", buffering=1)
        self.trace = (self.out / "PHYSICAL_TRACE.jsonl").open("x", buffering=1)
        self.contact_log = (self.out / "CONTACT_EVENTS.jsonl").open("x", buffering=1)
        self.budget = Budget(self.now(), self.task["duration_s"], self.task["maximum_actions"])
        self.c.attack_boundary = policy
        self.c._ensure_assurance()
        self.c._attack_grasp_initial_height = self.initial_object_z
        self.c._attack_grasp_contact_latch = None
        self.c._attack_grasp_confirmed_evidence = None
        c.start_scoring_capture()
        root = "/M20PiperVLA/M20_Piper"
        target = c.scoring_object_prim
        def contacts(headers, data):
            try:
                for h in headers:
                    paths = [str(PhysicsSchemaTools.intToSdfPath(v)) for v in (h.collider0, h.collider1)]
                    robot = [p.startswith(root + "/") for p in paths]
                    if sum(robot) != 1:
                        continue
                    body = paths[robot.index(True)]
                    other = paths[robot.index(False)]
                    points = [data[i] for i in range(int(h.contact_data_offset), int(h.contact_data_offset) + int(h.num_contact_data))]
                    actual = any(float(p.separation) <= 0. or np.linalg.norm(p.impulse) > 1e-8 for p in points)
                    if not actual:
                        continue
                    # M20's authored lower-leg support cylinders live below the
                    # *_knee links. They contact the floor during the normal gait.
                    support = "wheel" in body.lower() or any(f"/{leg}_knee/" in body for leg in ("fr", "fl", "hr", "hl"))
                    bounds = self.bounds(other)
                    floor = bounds[5] <= .12 and support
                    selected = other == target or other.startswith(target + "/")
                    finger = any(f"/{link}/" in body or body.endswith("/" + link) for link in ("link7", "link8"))
                    intentional = selected and (self.phase == "PUSH" or (self.carrying or self.phase in ("PICK", "PLACE", "BUTTON", "OPEN", "HANDOFF")) and finger)
                    row = {"time_s": self.now(), "phase": self.phase, "paths": paths, "allowed": floor or intentional}
                    self.contact_log.write(json.dumps(row) + "\n")
                    if not row["allowed"]:
                        self.collision = row
            except Exception as e:
                self.collision = {"callback_error": str(e), "paths": paths}
        self.subscription = get_physx_simulation_interface().subscribe_contact_report_events(contacts)
        if self.subscription is None:
            raise Stop("CONTACT_SUBSCRIPTION_UNAVAILABLE")

    def now(self):
        return float(self.c.world.current_time)

    def pose(self):
        xyz, quat = self.c.robot.get_world_pose()
        return np.r_[np.asarray(xyz)[:2], self.r["yaw_from_quaternion"](quat)]

    def bounds(self, path):
        prim = self.stage.GetPrimAtPath(path)
        if not prim:
            raise Stop("MISSING_TARGET_PRIM:" + path)
        self.bbox.Clear()
        box = self.bbox.ComputeWorldBound(prim).ComputeAlignedRange()
        if box.IsEmpty():
            raise Stop("EMPTY_TARGET_GEOMETRY:" + path)
        return np.r_[box.GetMin(), box.GetMax()].astype(float)

    def event(self, row):
        self.phase = row["phase"]
        row.update(world_identity=id(self.c.world), collector_identity=id(self.c),
                   robot_identity=id(self.c.robot), object_identity=id(self.c.cube))
        self.events.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")

    def observe_attacks(self):
        self.policy.observe_attack_state(self.c.attack_state_snapshot(self.initial_object_z), immediate_target={
            "immediate_robot_control": self.c, "immediate_gripper": self.c,
            "immediate_lidar": self.c, "immediate_imu": self.c})

    def safe(self):
        state = np.asarray(self.c.state())
        if not np.isfinite(state).all():
            raise Stop("NONFINITE_ROBOT_STATE")
        limits = self.r["SAFETY_LIMITS"]
        if state[2] < limits["minimum_base_height_m"] or abs(state[3]) > limits["maximum_roll_rad"] or abs(state[4]) > limits["maximum_pitch_rad"]:
            raise Stop("HEIGHT_OR_TILT_SAFETY_STOP")
        if self.collision:
            raise Stop("PHYSICAL_COLLISION:" + json.dumps(self.collision))
        if (self.out / "STOP").exists() or (Path(self.cfg["output"]) / "STOP").exists():
            raise Stop("USER_STOP")
        return state

    def step(self, action=None, arm=None, request_id=-1, source="controller", budgeted=True):
        dt = self.dt if budgeted else .1
        if budgeted:
            self.budget.check(self.now(), dt)
            self.safe()
        action = np.zeros(10, np.float32) if action is None else np.asarray(action, np.float32)
        if action.shape != (10,) or not np.isfinite(action).all():
            raise Stop("INVALID_ACTION")
        self.observe_attacks()
        self.c.latest_unified_action[:] = action
        start_step = self.c.physics_step
        start = self.now()
        assurance_receipt_ids = []
        for _ in range(round(dt * 50)):
            self.c.advance(self.phase, action[:3], self.arm if arm is None else arm, physics_steps=4)
            bridge = getattr(self.c,'assurance_bridge',None)
            if bridge is not None and bridge.latest_receipt is not None:
                assurance_receipt_ids.append(bridge.latest_receipt['command_id'])
                if self.cfg.get('paper200'):
                    from paper200.execution_boundary import record_actuator_link
                    record_actuator_link(self,request_id,bridge.latest_receipt)
            if budgeted:
                self.safe()
        if abs(self.now() - start - dt) > 1e-6:
            raise Stop("CONTROL_CLOCK_DRIFT")
        if budgeted:
            self.budget.actions += 1
            if getattr(self.c,'assurance_bridge',None) is not None:
                self.c.assurance_bridge.account_task_action()
        self.c.policy_execution_events.append({"execution_index": len(self.c.policy_execution_events), "request_id": request_id,
            "vla_chunk_id": self.request_count, "action_index_in_chunk": 0, "control_source": source,
            "raw_action": action.copy(), "requested_action": action.copy(), "start_physics_step": start_step,
            "end_physics_step": self.c.physics_step, "start_time_s": start_step * self.r["PHYSICS_DT"],
            "end_time_s": self.c.physics_step * self.r["PHYSICS_DT"],
            "action_semantics":"controller_request; actual_actuator_targets_in_assurance_receipts",
            "assurance_receipt_ids":assurance_receipt_ids})
        self.max_lift = max(self.max_lift, float(self.c.cube.get_world_pose()[0][2]) - self.initial_object_z)
        self.trace.write(json.dumps({"time_s": self.now(), "physics_step": self.c.physics_step,
            "phase": self.phase, "epoch": self.epoch, "pose": self.pose().tolist(), "action": action.tolist(),
            "arm_targets": (self.arm if arm is None else arm).tolist(), "request_id": request_id}) + "\n")

    def barrier(self, epoch):
        self.epoch = epoch
        # A model response is synchronous; only one action is ever executed from it.
        # Preserve active attack overrides rather than clearing the attack pipeline.
        self.c.latest_unified_action[:] = 0
        self.step()

    def initialize(self):
        # Retain the official run_policy's five-second motor-controlled settle,
        # but include it in the continuous recorder and formal episode budget.
        start = self.now(); samples = []
        for _ in range(math.ceil(5. / self.dt)):
            self.step()
            state = self.safe()
            samples.append({"time_s": self.now(), "pose": self.pose().tolist(),
                            "linear_speed_mps": float(np.linalg.norm(state[6:9])), "yaw_rate_rps": abs(float(state[11]))})
        write(self.out / "INITIAL_HOLD.json", {"duration_s": self.now() - start, "samples": samples})

    def infer(self, phase, epoch, carrying, model_phase="ARM_APPROACH", phase_step=0):
        self.budget.check(self.now(), self.dt)
        self.safe()
        self.observe_attacks()
        prompts = {"PLACE": "Keep the base stopped. Place the object at the requested destination, open the gripper and withdraw the arm.",
                   "PUSH": "Push the requested object in the requested direction.",
                   "BUTTON": "Press the requested button.", "OPEN": "Open the requested handle."}
        from m20_piper.local_align.contract import Phase, PROMPTS
        suffix = "" if phase.startswith("NAV_") else prompts.get(phase, PROMPTS.get(Phase(model_phase), ""))
        self.policy.assurance_phase_instruction = suffix
        obs = self.c.policy_observation(self.task["instruction"] + (" " + suffix if suffix else ""))
        obs["mission"] = {"phase": phase, "epoch": epoch, "scene": self.task["scene"], "carrying": carrying,
                          "time_s": self.now(), "phase_step": phase_step, "model_phase": model_phase}
        context = {"physics_step": self.c.physics_step,
            "time_s": self.c.physics_step * self.r["PHYSICS_DT"], "mission_phase": phase, "phase_epoch": epoch}
        bridge = self.c.take_q17_policy_bridge()
        if bridge is not None:
            context["q17_force_feedback_bridge"] = bridge
        response = self.policy.infer(obs, _attack_context=context)
        validate_response(response, phase, epoch, obs["mission"]["time_s"], self.cfg["model_bindings"])
        self.request_count += 1
        return response

    def select(self, phase, epoch, carrying):
        bridge = getattr(self.c,'assurance_bridge',None)
        attempts = (3+math.ceil(2*bridge.agent.cfg.recovery_attempt_s/self.dt)
                    if bridge is not None and bridge.agent.cfg.mode=='on' else 3)
        for _ in range(attempts):
            response = self.infer(phase, epoch, carrying)
            selection = response["selection"]
            if selection["decision"] == "ACCEPT":
                write(self.out / f"SELECTION_{epoch}.json", selection)
                return selection["goal_id"]
            if selection["decision"] != "WAIT_FOR_FRESH_FRAMES":
                raise Stop("MODEL_ABSTAIN:" + json.dumps(selection))
            wait_actions = np.asarray(response['actions'])
            recovery_wait = response.get('safe_agent',{}).get('state')=='RECOVERING'
            if np.any(wait_actions[:,:9] if recovery_wait else wait_actions):
                raise Stop("NONZERO_ACTION_WHILE_WAITING_FOR_FRESH_FRAMES")
            # These actual zero-action holds execute the model's WAIT response.
            # Keep its request identity so an attack active at t=0 has a real
            # raw-command/execution pair, rather than an empty evidence array.
            self.step(action=wait_actions[0],request_id=response[self.r["VLA_REQUEST_ID_KEY"]],source="gr00t_navigation_wait")
        raise Stop("MODEL_DID_NOT_SELECT")

    def free_box(self, xy):
        import carb
        hit = []
        def report(h):
            p = str(h.collision)
            prim = self.stage.GetPrimAtPath(p)
            if prim and "PhysxTriggerAPI" in prim.GetAppliedSchemas():
                return True
            if p.startswith("/M20PiperVLA/"):
                return True
            if self.carrying and (p == self.c.scoring_object_prim or p.startswith(self.c.scoring_object_prim + "/")):
                return True
            hit.append(p)
            return False
        self.query.overlap_box(carb.Float3(.48, .48, .51), carb.Float3(float(xy[0]), float(xy[1]), .65), carb.Float4(0., 0., 0., 1.), report, False)
        self.last_box_hits = hit
        return not hit

    def plan(self, goal):
        from all_scences.geometry import Grid
        entry = self.cfg["goals"].get(goal)
        if entry is None:
            raise Stop("UNREGISTERED_MODEL_GOAL:" + str(goal))
        start = self.pose()
        if goal == "START_RETURN":
            target = self.initial_pose.copy()
            box = None
        else:
            box = self.bounds(entry["prim"])
            center = (box[:2] + box[3:5]) / 2
            target = np.r_[center, 0.]
        # Query current cooked geometry; never use the adjusted training scene map.
        anchors = [start[:2], target[:2]]
        attack_regions = []
        if self.phase == "NAV_OUTBOUND" and self.cfg.get("visit_distance_trigger"):
            from pxr import UsdGeom
            for region in entry.get("original_attack_distance_regions", []):
                prim = self.stage.GetPrimAtPath(region["prim"])
                if not prim: raise Stop("MISSING_ORIGINAL_ATTACK_REGION:" + region["prim"])
                center = np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0).ExtractTranslation())[:2]
                attack_regions.append({**region, "center_xy": center.tolist()})
                anchors.append(center)
        for region in entry.get("via_regions", []):
            anchors.append(np.clip(anchors[-1], np.asarray(region["min_xy_m"]) + .1, np.asarray(region["max_xy_m"]) - .1))
        lo = np.min(anchors, axis=0) - 5.
        hi = np.max(anchors, axis=0) + 5.
        resolution = .20
        # Center a cell on the measured departure pose; snapping an endpoint by
        # half a cell must not invent a collision or move the robot.
        lo = start[:2] - (np.ceil((start[:2] - lo) / resolution) + .5) * resolution
        shape = np.ceil((hi - lo) / resolution).astype(int)
        if np.prod(shape) > 200000:
            raise Stop("MAP_QUERY_BUDGET_EXCEEDED")
        occupied = np.ones((shape[1], shape[0]), bool)
        for y in range(shape[1]):
            for x in range(shape[0]):
                occupied[y, x] = not self.free_box(lo + (np.array([x, y]) + .5) * resolution)
        grid = Grid(occupied, lo, resolution, clearance_margin=.15)
        start_free = self.free_box(start[:2])
        write(self.out / f"MAP_START_{self.epoch}.json", {"pose": start.tolist(), "free": start_free,
              "hits": self.last_box_hits, "cell": list(map(int, grid.cell(start))),
              "cell_center": grid.point(grid.cell(start))})
        component = grid.component(grid.cell(start))
        criterion = entry.get("criterion", {})
        if criterion.get("predicate") == "position_in_xy_region":
            region = criterion["params"]
            options = [np.asarray(grid.point((x, y))) for y, x in np.argwhere(component)]
            options = [xy for xy in options if np.all(xy >= region["min_xy_m"]) and np.all(xy <= region["max_xy_m"])]
            if not options:
                raise Stop("NO_REACHABLE_FORMAL_REGION")
            xy = min(options, key=lambda xy: np.linalg.norm(xy - start[:2]))
            target = np.r_[xy, start[2]]
        elif box is not None:
            candidates = []
            object_dock = (self.cfg.get("dock_target_mode") == "object"
                           and self.phase == "NAV_OUTBOUND" and "PICK" in self.task["stages"]
                           and self.task["question"] in entry.get("task_questions", []))
            focus = np.asarray(self.c.cube.get_world_pose()[0])[:2] if object_dock else target[:2]
            for y, x in np.argwhere(component):
                xy = np.asarray(grid.point((x, y)))
                edge = np.linalg.norm(np.maximum(0., np.maximum(box[:2] - xy, xy - box[3:5])))
                if .49 <= edge <= .90:
                    yaw = math.atan2(focus[1] - xy[1], focus[0] - xy[0])
                    if criterion.get("predicate") == "aabb_horizontal_distance_below" and self.rubric_gap(entry, np.r_[xy, yaw]) > float(criterion["params"]["max_distance_m"]) - .03:
                        continue
                    cost = (10 * float(np.linalg.norm(xy - focus)) + .1 * float(np.linalg.norm(xy - start[:2]))
                            if object_dock else float(np.linalg.norm(xy - start[:2])) + 2 * abs(edge - .65))
                    candidates.append((cost, np.r_[xy, yaw]))
            if not candidates:
                raise Stop("NO_COLLISION_FREE_RUBRIC_DOCK")
            target = min(candidates, key=lambda v: v[0])[1]
        path = []; waypoint = start; protected = set(); attack_checkpoints = {}
        for region in entry.get("via_regions", []):
            candidates = [np.asarray(grid.point((x, y))) for y, x in np.argwhere(component)]
            candidates = [xy for xy in candidates if np.all(xy >= region["min_xy_m"]) and np.all(xy <= region["max_xy_m"])]
            if not candidates:
                raise Stop("NO_REACHABLE_INTERMEDIATE_REGION:" + json.dumps(region))
            via = min(candidates, key=lambda xy: np.linalg.norm(xy - waypoint[:2]) + np.linalg.norm(xy - target[:2]))
            path += grid.route(waypoint, via); waypoint = via
            protected.add(len(path) - 1)
        for region in attack_regions:
            center = np.asarray(region["center_xy"])
            if np.linalg.norm(waypoint[:2] - center) <= region["radius_m"] - .02:
                continue
            options = [np.asarray(grid.point((x, y))) for y, x in np.argwhere(component)]
            options = [xy for xy in options if np.linalg.norm(xy - center) <= region["radius_m"] - .02]
            if not options:
                raise Stop("NO_COLLISION_FREE_ORIGINAL_ATTACK_REGION:" + region["object_id"])
            via = min(options, key=lambda xy: np.linalg.norm(xy - waypoint[:2]) + np.linalg.norm(xy - target[:2]))
            path += grid.route(waypoint, via); waypoint = via
            protected.add(len(path) - 1)
            attack_checkpoints[len(path) - 1] = region
        path += grid.route(waypoint, target)
        if not self.free_box(target[:2]):
            raise Stop("EXACT_DESTINATION_OCCUPIED:" + json.dumps(self.last_box_hits))
        # START_RETURN is a measured pose and need not be a grid-cell center.
        # Keep its exact endpoint so local control and arrival test agree.
        path[0] = start[:2].tolist()
        if len(path) == 1:
            path.append(target[:2].tolist())
        else:
            path[-1] = target[:2].tolist()
        simplified = [path[0]]
        self.route_attack_checkpoints = {}
        i = 0
        while i < len(path) - 1:
            j = min(i + 8, len(path) - 1)
            j = min([j] + [p for p in protected if i < p <= j])
            while j > i + 1 and not grid.segment_free(path[i], path[j]):
                j -= 1
            simplified.append(path[j])
            if j in attack_checkpoints:
                self.route_attack_checkpoints[len(simplified) - 1] = attack_checkpoints[j]
            i = j
        self.grid = grid
        write(self.out / f"ROUTE_{self.epoch}.json", {"goal": goal, "target": target.tolist(), "path": simplified,
            "origin": lo.tolist(), "resolution": resolution, "collision_source": "LIVE_COOKED_PHYSX",
            "cells": int(np.prod(shape)), "footprint_half_extents": [.48, .48, .51],
            "original_attack_distance_regions": attack_regions,
            "dock_target_mode": self.cfg.get("dock_target_mode", "shelf"),
            "attack_region_checkpoints": self.route_attack_checkpoints,
            "attack_region_route_authority": "EXPLICIT_TEST_RUNNER_OPTION; no trigger flag is forced"})
        return simplified, target, entry

    def navigate(self, phase, epoch, carrying):
        self.carrying = carrying
        goal = self.select(phase, epoch, carrying)
        if self.task["question"] == "Q08":
            raise Stop("STAIR_GOAL_NOT_IN_CURRENT_MODEL_VOCABULARY")
        path, target, entry = self.plan(goal)
        start = self.pose(); last = start.copy(); distance = 0.; best = float("inf"); progress = self.now()
        index = 0; hold = []; drop_steps = 0
        tracking_fault_seconds = 0.
        stalled_forward_seconds = 0.
        recovery_count = 0
        last_model_refresh = self.now()
        assurance_recovery_id = None
        refresh_seconds = self.cfg.get("nav_refresh_seconds", 0.)
        while True:
            bridge = getattr(self.c,'assurance_bridge',None)
            if bridge is not None and bridge.agent.cfg.mode=='on' and not bridge.agent.cfg.visual_only:
                recovery = bridge.agent.recovery
                if recovery.state=='STOP':raise Stop('SAFE_AGENT:'+str(recovery.stop_reason))
                rid = (recovery.phase,recovery.attempts[recovery.phase])
                if recovery.operation=='REPLAN' and recovery.state=='RECOVERING' and rid!=assurance_recovery_id:
                    assurance_recovery_id = rid
                    self.grid = None
                    path,target,entry = self.plan(goal)
                    index=0; best=float('inf'); progress=self.now(); hold=[]
                    bridge.agent.log({'type':'recovery_operation','operation':'REPLAN','phase':phase,
                                      'attempt':rid[1],'time_s':self.now()})
            if refresh_seconds and self.now() - last_model_refresh >= refresh_seconds - 1e-7:
                refreshed_goal = self.select(phase, epoch, carrying)
                last_model_refresh = self.now()
                self.event({"phase": phase, "epoch": epoch, "time_s": self.now(),
                    "status": "NAV_MODEL_REFRESH", "previous_goal": goal, "goal_id": refreshed_goal})
                if refreshed_goal != goal:
                    goal = refreshed_goal
                    path, target, entry = self.plan(goal)
                    index = 0; best = float("inf"); progress = self.now(); hold = []
            pose = self.pose()
            distance += float(np.linalg.norm(pose[:2] - last[:2])); last = pose.copy()
            if carrying:
                if self.grasped():
                    drop_steps = 0
                else:
                    drop_steps += 1
                if drop_steps >= 3:
                    self.event({"phase": phase, "epoch": epoch, "time_s": self.now(), "status": "PAYLOAD_DROPPED"})
                    carrying = self.handoff(epoch, False, 10.)
                    self.carrying = carrying
            while index < len(path) - 1 and np.linalg.norm(pose[:2] - path[index]) < .22:
                region = getattr(self, "route_attack_checkpoints", {}).get(index)
                if region:
                    actual_distance = float(np.linalg.norm(pose[:2] - region["center_xy"]))
                    if actual_distance > region["radius_m"] - .01:
                        break
                    self.event({"phase": phase, "epoch": epoch, "time_s": self.now(),
                        "status": "ORIGINAL_ATTACK_REGION_VISITED", "region": region,
                        "measured_distance_m": actual_distance, "pose": pose.tolist(),
                        "attack_application": "REQUIRES_ORIGINAL_HANDLER_AND_EVIDENCE_CHECK"})
                index += 1; best = float("inf"); progress = self.now()
            remaining = float(np.linalg.norm(pose[:2] - target[:2]))
            waypoint_distance = float(np.linalg.norm(pose[:2] - path[index]))
            if waypoint_distance < best - .03:
                best = waypoint_distance; progress = self.now()
            if self.now() - progress > 35.:
                raise Stop("NO_PHYSICAL_PROGRESS")
            action = np.zeros(10, np.float32)
            if remaining < .04 or remaining < .18 and self._rubric_arrived(entry, goal):
                yaw = wrap(target[2] - pose[2])
                defer_heading = (phase == "NAV_OUTBOUND" and self.cfg.get("dock_heading_mode") == "operation_alignment"
                                 and "PICK" in self.task["stages"] and self._rubric_arrived(entry, goal))
                if abs(yaw) > .12 and not defer_heading:
                    # Reuse the attack-aware LiDAR filter even for final heading control.
                    _, audit = self.c.lidar_navigation_command(pose[:2] + [math.cos(target[2]), math.sin(target[2])])
                    if audit.get("state") == "avoid":
                        raise Stop("DOCK_ROTATION_BLOCKED")
                    action[2] = np.clip(yaw, -.18, .18)
                    hold = []
                else:
                    hold.append((self.now(), pose.copy()))
                    if len(hold) >= max(2, math.ceil(2. / self.dt)):
                        excursion = max(float(np.linalg.norm(p[:2] - hold[0][1][:2])) for _, p in hold)
                        if excursion <= .01 and self._rubric_arrived(entry, goal):
                            return {"status": "COMPLETE", "goal_id": goal, "carrying": carrying, "traveled_m": distance,
                                    "net_displacement_m": float(np.linalg.norm(pose[:2] - start[:2])), "end_pose": pose.tolist(),
                                    "stop_excursion_m": excursion, "heading_error_rad": yaw,
                                    "heading_alignment_deferred_to_pick": defer_heading}
                        if not self._rubric_arrived(entry, goal) and remaining < .04:
                            raise Stop("RUBRIC_DISTANCE_NOT_REACHED")
                        hold = hold[-max(1, math.ceil(2. / self.dt) - 1):]
            else:
                command, audit = self.c.lidar_navigation_command(np.asarray(path[index]))
                action[:3] = command
                action[0] = np.clip(action[0], 0., .16)
                # Decelerate at the actual dock. A loose waypoint tolerance is
                # insufficient for the frozen scorer's AABB distance threshold.
                if index == len(path) - 1:
                    action[0] = min(action[0], max(.025, remaining * .5))
                action[2] = np.clip(action[2], -.3, .3)
                if abs(audit.get("heading_error_rad", 0)) > .38:
                    action[0] = 0.
                hold = []
            before = self.pose()
            self.step(action)
            after = self.pose()
            forward = float(np.dot(after[:2] - before[:2], [math.cos(before[2]), math.sin(before[2])]))
            tracking_fault_seconds = tracking_fault_seconds + self.dt if action[0] > .04 and forward < -.02 * self.dt else 0.
            if tracking_fault_seconds >= 2.:
                raise Stop("LOCOMOTION_MOVED_OPPOSITE_COMMAND")
            actual_motion = float(np.linalg.norm(after[:2] - before[:2]))
            stalled_forward_seconds = (stalled_forward_seconds + self.dt
                if action[0] > .04 and actual_motion < .03 * self.dt else 0.)
            if (stalled_forward_seconds >= 5. and self.now() - progress >= 10.
                    and recovery_count < self.cfg.get("nav_stall_recoveries", 0)):
                # Restore the existing nominal standing posture through real
                # motors. The attack-aware actuator and all safety/budget
                # checks remain active; no robot pose or policy state is reset.
                recovery_count += 1
                self.event({"phase": phase, "epoch": epoch, "time_s": self.now(),
                    "status": "LOCOMOTION_STANCE_RECOVERY", "attempt": recovery_count,
                    "pose_before": self.pose().tolist(), "commanded_forward_mps": float(action[0])})
                for _ in range(math.ceil(2. / self.dt)):
                    self.step()
                self.event({"phase": phase, "epoch": epoch, "time_s": self.now(),
                    "status": "LOCOMOTION_STANCE_RECOVERY_FINISHED", "attempt": recovery_count,
                    "pose_after": self.pose().tolist()})
                stalled_forward_seconds = 0.
                hold = []

    def _rubric_arrived(self, entry, goal):
        if goal == "START_RETURN":
            return np.linalg.norm(self.pose()[:2] - self.initial_pose[:2]) <= .25
        if entry.get("criterion", {}).get("predicate") == "aabb_horizontal_distance_below":
            return self.rubric_gap(entry) <= float(entry["criterion"]["params"]["max_distance_m"])
        if entry.get("criterion", {}).get("predicate") == "position_in_xy_region":
            xy = self.pose()[:2]; region = entry["criterion"]["params"]
            return bool(np.all(xy >= region["min_xy_m"]) and np.all(xy <= region["max_xy_m"]))
        gap = aabb_gap(self.bounds("/M20PiperVLA/M20_Piper"), self.bounds(entry["prim"]))
        return gap <= float(entry.get("distance_m", .3))

    def rubric_gap(self, entry, candidate=None):
        from robolab_action_predicates.aabb import world_aabb_from_local
        item = entry["criterion"]
        geometries = {v["entity_id"]: v["local_aabb_m"] for v in entry["geometries"]}
        inputs = item["inputs"]
        xyz, quat = self.c.robot.get_world_pose()
        if candidate is not None:
            xyz = np.r_[candidate[:2], xyz[2]]
            quat = [math.cos(candidate[2] / 2), 0., 0., math.sin(candidate[2] / 2)]
        subject = world_aabb_from_local(geometries[inputs["subject_geometry"]["geometry_ref"]], np.r_[xyz, quat])[0]
        reference_input = inputs["reference_pose"]
        if "value" in reference_input:
            reference_pose = reference_input["value"]
        else:
            # Factory/kitchen rules also reference live articulated or rigid
            # objects. A frozen literal pose is not available for those rules.
            from pxr import Usd, UsdGeom
            prim_path = unquote(reference_input["path"].split("/")[-2])
            prim = self.stage.GetPrimAtPath(prim_path)
            if not prim: raise Stop("MISSING_SCORER_REFERENCE_PRIM:" + prim_path)
            transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            q = transform.ExtractRotationQuat()
            reference_pose = [*transform.ExtractTranslation(), q.GetReal(), *q.GetImaginary()]
        reference = world_aabb_from_local(geometries[inputs["reference_geometry"]["geometry_ref"]], reference_pose)[0]
        return aabb_gap(subject, reference)

    def grasped(self):
        forces = np.linalg.norm(self.c._contact_net_forces_w(), axis=-1)
        values = {p.rsplit("/", 1)[-1]: float(f) for p, f in zip(self.c.scoring_contact_body_paths, forces)}
        lift = float(self.c.cube.get_world_pose()[0][2]) - self.initial_object_z
        ok = values.get("link7", 0.) > .05 and values.get("link8", 0.) > .05 and lift >= .05
        return bool(ok)

    def manipulate(self, phase, epoch, seconds):
        started = self.now(); calls = 0; grasp_stable = 0; ever_grasped = False
        self.phase = phase
        schedule = [(6., "LOCAL_ALIGN"), (7., "SETTLE"), (18., "ARM_APPROACH"), (23., "CONTACT_CLOSE"), (28., "LIFT_VERIFY"), (float("inf"), "HOLD")]
        previous_model_phase = None; phase_step = 0
        phase_started = started
        recovered_time = 0.
        last_recovery = None
        while self.now() - started + self.dt <= seconds + 1e-7:
            bridge = getattr(self.c,'assurance_bridge',None)
            if bridge is not None and bridge.agent.cfg.mode=='on' and not bridge.agent.cfg.visual_only:
                recovery = bridge.agent.recovery
                if recovery.state=='STOP':raise Stop('SAFE_AGENT:'+str(recovery.stop_reason))
                rid = (recovery.phase,recovery.attempts[recovery.phase])
                if recovery.state=='RECOVERING' and rid!=last_recovery:
                    last_recovery=rid
                    if recovery.operation in {'REGRASP','REALIGN'}:
                        recovered_time=self.now()-started; previous_model_phase=None
                    bridge.agent.log({'type':'recovery_operation','operation':recovery.operation,
                                      'phase':phase,'attempt':rid[1],'time_s':self.now()})
            elapsed = self.now() - started - recovered_time
            model_phase = next(p for t, p in schedule if elapsed < t) if phase == "PICK" else "ARM_APPROACH"
            if phase == "PICK" and elapsed >= 30. and ever_grasped:
                if not self.grasped():
                    raise Stop("OBJECT_LOST_DURING_LOADED_PHASE")
                model_phase = next(p for t, p in [(44., "RETRACT_LOADED"), (47., "CARRY_VERIFY"), (57., "BASE_RETREAT"), (float("inf"), "RETURN_READY_HOLD")] if elapsed < t)
            if previous_model_phase != model_phase:
                phase_started = self.now(); previous_model_phase = model_phase
            # Preserve the trained 10 Hz phase clock independently of action hold rate.
            phase_step = round((self.now() - phase_started) * 10)
            response = self.infer(phase, epoch, ever_grasped, model_phase, phase_step)
            raw = np.asarray(response["actions"], np.float32)
            if raw.ndim != 2 or raw.shape[1] != 10 or not np.isfinite(raw).all():
                raise Stop("GRASP_ACTION_ABI")
            action = np.clip(raw[0], self.r["VLA_ACTION_LOW"], self.r["VLA_ACTION_HIGH"])
            action[:3] = 0.
            if model_phase not in ('LOCAL_ALIGN','SETTLE'):
                action = self.prepare_ee_command(action, response[self.r['VLA_REQUEST_ID_KEY']], model_phase)
            if model_phase == "BASE_RETREAT":
                state = self.safe()
                action[0] = np.clip(raw[0, 0], -.05, 0.)
                action[2] = np.clip(raw[0, 2], -.08, .08)
                if action[0] < 0.:
                    # Public rear camera enters the policy; real PhysX rays add
                    # a conservative local stopping check before reverse motion.
                    xyz, _ = self.c.robot.get_world_pose()
                    yaw = self.pose()[2]
                    backwards = np.array([-math.cos(yaw), -math.sin(yaw), 0.])
                    sideways = np.array([-math.sin(yaw), math.cos(yaw), 0.])
                    for offset in [-.30, 0., .30]:
                        origin = np.asarray(xyz) + backwards * .50 + sideways * offset
                        origin[2] = max(.20, float(xyz[2]) - .15)
                        hit = self.query.raycast_closest(tuple(origin), tuple(backwards), .65)
                        if hit.get("hit", False):
                            raise Stop("REAR_CORRIDOR_BLOCKED")
                from paper200.execution_boundary import apply_ordinary_ik
                apply_ordinary_ik(self,action)
            elif model_phase == "LOCAL_ALIGN":
                action[0] = np.clip(raw[0, 0], -.08, .08)
                action[2] = np.clip(raw[0, 2], -.12, .12)
                action[3:] = 0.
                if action[0] < 0:
                    # This v1 does not issue blind backwards local motion.
                    action[0] = 0.
                action[:3], _ = self.c.hybrid_avoidance_filter(action[:3], self.pose()[:2] + np.array([math.cos(self.pose()[2]), math.sin(self.pose()[2])]))
            elif model_phase != "SETTLE":
                from paper200.execution_boundary import apply_ordinary_ik
                apply_ordinary_ik(self,action)
            self.step(action, request_id=response[self.r["VLA_REQUEST_ID_KEY"]], source="gr00t")
            calls += 1; phase_step += 1
            grasp_stable = grasp_stable + 1 if self.grasped() else 0
            ever_grasped |= grasp_stable >= max(2, math.ceil(.5 / self.dt))
        # Placement is left to the real scorer; an empty-handed attempt cannot establish it.
        return {"status": "WINDOW_EXHAUSTED", "simulation_seconds": self.now() - started,
                "inference_calls": calls, "executed_model_actions": calls, "grasp_success": ever_grasped,
                "carrying": grasp_stable >= max(2, math.ceil(.3 / self.dt)), "physical_success": False,
                "place_capability": "UNVALIDATED_SAME_GR00T_WEIGHT" if phase == "PLACE" else None}

    def handoff(self, epoch, carrying, seconds):
        self.carrying = carrying
        previous_phase = self.phase; self.phase = "HANDOFF"
        target = np.asarray(self.cfg["loaded_carry_joints"] if carrying else self.stowed, dtype=np.float32)
        if target.shape != (6,):
            raise Stop("CARRY_POSTURE_ABI")
        end = self.now() + seconds; stable = 0
        while self.now() + self.dt <= end + 1e-7:
            measured = np.asarray(self.c.robot.get_joint_positions())[self.arm_indices]
            self.arm[:6] += np.clip(target - self.arm[:6], -.25 * self.dt, .25 * self.dt)
            if not carrying:
                self.arm[-1] = self.c.gripper_max_opening
            self.c.piper_command_targets = self.arm.copy()
            self.step()
            state = self.safe()
            ready = np.max(abs(measured - target)) < .06 and np.linalg.norm(state[6:9]) < .03 and abs(state[11]) < .05
            stable = stable + 1 if ready else 0
            if stable >= max(2, math.ceil(.5 / self.dt)):
                self.phase = previous_phase
                self.carrying = bool(carrying and self.grasped())
                return self.carrying
        raise Stop("HANDOFF_TIMEOUT")

    def stop(self, epoch):
        self.epoch = epoch
        self.phase = "STOP"
        if self.cfg.get('paper200'):
            from paper200.execution_boundary import cancel_at_stop
            cancel_at_stop(self)
        # Always emit zero velocity through the existing attack-aware actuator interface.
        self.c.latest_unified_action[:] = 0.
        # Apply motor targets even when the episode has no remaining simulation time.
        self.c.advance("STOP", np.zeros(3, np.float32), self.arm, physics_steps=0)
        if self.budget.remaining(self.now()) >= .1:
            self.step(budgeted=False)

def run_continuous(c, policy, runtime):
    sys.path.insert(0, str(VLN))
    sys.path.insert(0, str(ROOT / "vendor"))
    cfg = read(os.environ["THREE_SCENES_RUN_CONFIG"])
    write(runtime["args"].output / "PROCESS_IDENTITY.json", {"uid": os.getuid(), "gid": os.getgid(),
          "user": pwd.getpwuid(os.getuid()).pw_name, "groups": os.getgroups(), "home": os.environ.get("HOME")})
    from return_delivery.physics_clock import install_explicit_physics_clock
    install_explicit_physics_clock(c, runtime, runtime["args"].output)
    backend = Backend(c, policy, runtime, cfg)
    try:
        result = Mission(backend, cfg["task"], cfg["pick_seconds"], cfg["place_seconds"]).run()
        result.update(accepted=False, inference_count=backend.request_count, executed_actions=backend.budget.actions,
                      frames=len(c.states), maximum_object_lift_m=backend.max_lift,
                      final_base_xyz=np.asarray(c.robot.get_world_pose()[0]).tolist(),
                      physical_collision=backend.collision, checks={"flow_completed": result["flow_completed"]},
                      model_bindings=cfg["model_bindings"], same_collector_identity=id(c), same_world_identity=id(c.world),
                      attack_boundary_preserved=True, no_intermediate_reset=True)
        write(runtime["args"].output / "FULL_CHAIN_RESULT.json", result)
        write(runtime["args"].output / "vla_trace.json", c.vla_trace)
        return result
    finally:
        if getattr(c,'assurance_bridge',None) is not None:
            c.assurance_bridge.close()
        backend.events.close(); backend.trace.close(); backend.contact_log.close()
