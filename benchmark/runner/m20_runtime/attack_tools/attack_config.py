"""Load and validate the attack experiment configuration envelope."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
from .attack_orchestrator import AttackSpec

def load_attack_config(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve(strict=True)
    config = json.loads(resolved.read_text(encoding="utf-8"))
    if config.get("schema") not in {"m20-attack-experiment/v2", "m20-attack-experiment/v3"}:
        raise ValueError("unsupported M20 attack config schema")
    for section in ("scene", "robot", "interfaces", "attack", "logging"):
        if section not in config:
            raise ValueError(f"missing config section: {section}")
    if config["robot"].get("type") != "M20":
        raise ValueError("this runtime requires robot.type=M20")
    for item in config["attack"].get("attacks", []):
        AttackSpec.from_dict(item)
    return config
