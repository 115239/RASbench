"""Paths and JSON provenance for the isolated continuous runner."""
from pathlib import Path
import hashlib
import json
import os
import time

ROOT = Path(__file__).resolve().parent
BASE = Path(os.environ.get('PAPER_BASELINE_ROOT', str(ROOT.parents[1])))
WS = BASE
SCORE = BASE / "rules"
VLN = BASE / "code/navigation"
RUBRIC = BASE / "rules/rubrics.xlsx"
IMAGE = "embodied-competition-runner:0915"
IMAGE_PACKAGE = "/opt/safety_embodiment/question/unified_q01_q24_20260813"
IMAGE_RUNNER = os.environ.get('PAPER_RUNNER_ROOT', "/opt/safety_embodiment/competition_runner/runner")
MODEL_PYTHON = Path("/opt/groot-env/bin/python")
GRASP_CODE = BASE / "code/grasp"
GRASP_MODEL = BASE / "models/vla"
NAV_OUT = NAV_RETURN = BASE / "models/vln"
LOCOMOTION = BASE / "models/locomotion/policy.onnx"

def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()

def binding(path):
    p = Path(path).resolve(strict=True)
    return {"path": str(p), "sha256": sha(p), "bytes": p.stat().st_size}

def write(path, value):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + f".{os.getpid()}.{time.time_ns()}.tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, p)

def question(value):
    n = int(str(value).upper().removeprefix("Q"))
    if not 1 <= n <= 24:
        raise ValueError("question must be Q01 through Q24")
    return f"Q{n:02d}"

def scene(q):
    return "factory" if int(q[1:]) <= 8 else "kitchen" if int(q[1:]) <= 16 else "market"
