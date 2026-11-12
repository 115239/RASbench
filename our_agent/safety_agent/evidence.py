from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
from .contracts import digest


class EvidenceWriter:
    def __init__(self, directory, identity):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.previous = "0" * 64
        self.stream = (self.directory / "events.jsonl").open("x", encoding="utf-8")
        self.identity = identity
        self.append({"type": "identity", **identity})

    def append(self, event):
        event = {**event, "previous_sha256": self.previous}
        event["sha256"] = digest(event)
        self.stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
        self.stream.flush()
        self.previous = event["sha256"]

    def report(self, value):
        result = {**self.identity, **value, "last_event_sha256": self.previous,
                  "formal_success": None, "physical_acceptance": "NOT_EVALUATED"}
        target = self.directory / "defense_report.json"
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        os.replace(temporary, target)

    def close(self):
        if not self.stream.closed:
            self.stream.flush()
            os.fsync(self.stream.fileno())
            self.stream.close()


def verify_chain(path):
    previous = "0" * 64
    count = 0
    for line in Path(path).read_text().splitlines():
        item = json.loads(line)
        claimed = item.pop("sha256")
        if item["previous_sha256"] != previous or digest(item) != claimed:
            raise ValueError("evidence chain mismatch at record " + str(count))
        previous = claimed
        count += 1
    return {"records": count, "last_sha256": previous}
