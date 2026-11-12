"""Unified JSONL evidence for VLA, sensor, state, and control interception."""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib, json, threading, time
from pathlib import Path
from typing import Any, Mapping
import numpy as np

def summarize(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        result = {"type": "ndarray", "shape": list(value.shape), "dtype": str(value.dtype),
                  "sha256": hashlib.sha256(value.tobytes()).hexdigest()}
        if value.ndim < 3 and value.size <= 256:
            result["values"] = value.tolist()
        return result
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): summarize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [summarize(v) for v in value] if len(value) <= 256 else {"type": "sequence", "length": len(value)}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)

class UnifiedAttackLogger:
    def __init__(self, path: str | Path, *, question_id: str | None = None,
                 scenario_id: str | None = None) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.question_id, self.scenario_id = question_id, scenario_id
        self._sequence, self._lock = 0, threading.Lock()

    def append(self, record: Mapping[str, Any]) -> None:
        with self._lock:
            self._sequence += 1
            envelope = {"schema": "m20-unified-attack-audit/v1", "sequence": self._sequence,
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "monotonic_ns": time.monotonic_ns(), "question_id": self.question_id,
                        "scenario_id": self.scenario_id, **dict(record)}
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(summarize(envelope), ensure_ascii=False) + "\n")
