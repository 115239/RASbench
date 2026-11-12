"""Owned GPU execution over the unmodified official export pipeline."""
import csv
from contextlib import contextmanager, ExitStack
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import traceback
from urllib.parse import unquote
from common import *
from catalog import build, walk

def model_binding(path):
    path = Path(path).resolve(strict=True)
    names = ["config.json", "processor_config.json", "statistics.json", "embodiment_id.json", "model.safetensors.index.json"]
    index = path / "model.safetensors.index.json"
    if index.exists():
        names += sorted(set(read(index)["weight_map"].values()))
    else:
        names += ["model.safetensors"]
    for optional in ["MODEL_BINDING.json", "GOALS.json", "SHA256SUMS", "q18_binding.json", "market_unified_binding.json", "q18_statistics.json", "CALIBRATION.json", "INFERENCE_PROFILE.json"]:
        if (path / optional).exists(): names.append(optional)
    files = {n: binding(path / n) for n in names}
    digest = hashlib.sha256(json.dumps({n: v["sha256"] for n, v in files.items()}, sort_keys=True).encode()).hexdigest()
    return {"path": str(path), "sha256": digest, "files": files}

def make_goals(catalog):
    goals = {"START_RETURN": {"measured_spawn": True}}
    names = read(VLN / "config/goalspec_codec.json")["goal_ids"][:8]
    pairs = {f"Q{17 + i:02d}": n for i, n in enumerate(names)}
    for s in ["factory", "kitchen"]:
        config = read(VLN / f"all_scences/config/{s}.json")
        for q, t in config["tasks"].items(): pairs[q] = t["outbound_goal_audit_only"]
    pairs["Q08"] = "FACTORY_YELLOW_STAIRS"
    for q, name in pairs.items():
        task = catalog["questions"][q]
        nav_items = [i for i in task["items"] if i["stage"] == "NAV_OUTBOUND"]
        first = nav_items[-1]
        rule = read(task["rule_package"]["path"])
        criterion = next((v for v in walk(first) if v.get("predicate") == "aabb_horizontal_distance_below"), first)
        reference = criterion.get("inputs", {}).get("reference_geometry", {}).get("geometry_ref")
        references = [unquote(reference)] if reference else []
        if references:
            goals[name] = {"prim": references[0], "distance_m": first.get("params", {}).get("max_distance_m", .3),
                           "criterion": {k: v for k, v in criterion.items() if k in ("predicate", "inputs", "params")},
                           "geometries": rule["trajectory_context"]["geometries"]}
        else:
            # Region-based factory items have no geometry_ref. Bind the semantic
            # destination to its authored scene anchor instead of dropping the goal.
            scene_config = read(VLN / f"all_scences/config/{task['scene']}.json")
            anchor = scene_config.get("targets", {}).get(name, {}).get("source_prim")
            if not anchor:
                paths = list(read(Path(IMAGE_PACKAGE) / "task" / q / "data/task/task.json").get("tracked_objects", {}).values())
                anchor = next((p for p in paths if "starting" not in p.lower()), None)
            if anchor:
                goals[name] = {"prim": anchor, "distance_m": .3,
                               "criterion": {k: v for k, v in criterion.items() if k in ("predicate", "inputs", "params")}}
        if name in goals:
            goals[name]["via_regions"] = [v["params"] for i in nav_items[:-1] for v in walk(i)
                                          if v.get("predicate") == "position_in_xy_region"]
            original_task = read(task["task"]["path"])
            trigger_regions = []
            for profile in task["attack_profiles"]:
                for condition in walk(profile["profile"]):
                    if condition.get("type") != "distance_to_object" or condition.get("op") != "lte" or condition.get("axes") != "xy":
                        continue
                    object_id = condition["object_id"]
                    prim = original_task.get("tracked_objects", {}).get(object_id)
                    if prim:
                        region = {"object_id": object_id, "prim": prim, "radius_m": float(condition["distance_m"])}
                        if region not in trigger_regions: trigger_regions.append(region)
            goals[name]["original_attack_distance_regions"] = trigger_regions
    goals.update({
        "GREEN_TAKEOUT": {"prim": "/World/Market/StaticColliders/Fixtures/Z13A/z13_order_rack", "distance_m": .3},
        "BLUE_DELIVERY_RACK": {"prim": "/World/Market/StaticColliders/Fixtures/Z12/z12_rack_east", "distance_m": .3},
        "KITCHEN_DINING_PLATTER": {"prim": "/World_01/dining_table", "distance_m": .3},
        "KITCHEN_SINK": {"prim": "/World_01/kitchen_cabinet_and_island/body/E_body_146/E_SM_05_CF_BX_02_01_01_004_139", "distance_m": .3},
    })
    # Attach the exact scorer geometry to shared delivery goals as well.
    for q, name in [("Q18", "GREEN_TAKEOUT"), ("Q21", "BLUE_DELIVERY_RACK"), ("Q09", "KITCHEN_DINING_PLATTER"), ("Q10", "KITCHEN_SINK")]:
        rule = read(catalog["questions"][q]["rule_package"]["path"])
        for item in rule["items"]:
            reference = item.get("inputs", {}).get("reference_geometry", {}).get("geometry_ref")
            if reference and unquote(reference) == goals[name]["prim"]:
                goals[name].update(criterion={k: v for k, v in item.items() if k in ("predicate", "inputs", "params")},
                                   geometries=rule["trajectory_context"]["geometries"])
    for name, entry in goals.items():
        entry["task_questions"] = [q for q, goal in pairs.items() if goal == name]
    return goals

