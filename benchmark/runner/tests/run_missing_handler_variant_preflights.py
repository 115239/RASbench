"""Preflight every Q1-Q200 profile that uses one of the seven synced handlers."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


HANDLERS = {
    "replace_prompt_sequence",
    "world_frame_tcp_translation_offset",
    "joint_encoder_position_feedback",
    "base_velocity_toward_object",
    "gripper_opening_command_offset",
    "arm_joint_hold_trigger_command",
    "translation_direction_offset",
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--runner-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    package_root = args.package_root.resolve(strict=True)
    runner_root = args.runner_root.resolve(strict=True)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=False)

    jobs = []
    for question in sorted((package_root / "task").glob("Q*"), key=lambda path: int(path.name[1:])):
        for profile_path in sorted((question / "data" / "attack" / "profiles").glob("*.json")):
            profile = load_json(profile_path)
            attacks = profile.get("attack", {}).get("attacks", [])
            matched = sorted({str(attack.get("handler")) for attack in attacks} & HANDLERS)
            if not matched:
                continue
            variant = str(profile.get("acceptance", {}).get("variant_id", ""))
            if not variant:
                raise RuntimeError(f"{profile_path}: missing acceptance.variant_id")
            jobs.append((question.name, question, variant, matched))

    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(runner_root)
    entry = runner_root / "robolab_runner" / "m20_question_runner.py"

    def run(job):
        qid, question, variant, matched = job
        output = output_root / f"{qid}_{variant}"
        command = [
            str(args.python),
            str(entry),
            "--package-root", str(question),
            "--output-dir", str(output),
            "--attack-variant", variant,
            "--policy-endpoint", "ws://127.0.0.1:8802",
            "--preflight-only",
        ]
        completed = subprocess.run(
            command,
            cwd=runner_root,
            env=environment,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=120,
        )
        return {
            "question_id": qid,
            "variant": variant,
            "handlers": matched,
            "status": "PASS" if completed.returncode == 0 else "FAIL",
            "returncode": completed.returncode,
            "stderr_tail": completed.stderr.strip().splitlines()[-20:],
        }

    results = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {executor.submit(run, job): (job[0], job[2]) for job in jobs}
        for future in as_completed(futures):
            qid, variant = futures[future]
            try:
                results.append(future.result())
            except Exception as error:
                results.append({
                    "question_id": qid,
                    "variant": variant,
                    "status": "HARNESS_ERROR",
                    "error": repr(error),
                })

    results.sort(key=lambda row: (int(row["question_id"][1:]), row["variant"]))
    counts = {}
    handler_counts = {}
    for result in results:
        counts[result["status"]] = counts.get(result["status"], 0) + 1
        for handler in result.get("handlers", []):
            handler_counts[handler] = handler_counts.get(handler, 0) + 1
    report = {
        "schema": "q200-synced-handler-preflight-matrix/v1",
        "scope": "all attack profiles using one of the seven synced handlers",
        "jobs": len(jobs),
        "counts": counts,
        "handler_counts": handler_counts,
        "results": results,
    }
    report_path = output_root.parent / f"{output_root.name}.summary.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "report": str(report_path),
        "jobs": len(jobs),
        "counts": counts,
        "handler_counts": handler_counts,
    }, ensure_ascii=False))
    return 0 if counts.get("FAIL", 0) == 0 and counts.get("HARNESS_ERROR", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
