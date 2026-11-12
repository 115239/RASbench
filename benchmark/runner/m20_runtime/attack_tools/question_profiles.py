"""Resolve a development attack profile by question/scenario ID."""
from __future__ import annotations
import json, os
from pathlib import Path
from typing import Any

INDEX_SCHEMA = "m20-question-attack-profile-index/v1"

def load_question_profile(question_id: str, *, root: str | Path | None = None,
                          require_digest: bool = False) -> tuple[dict[str, Any], Path]:
    profile_root = Path(root or os.getenv("M20_ATTACK_PROFILE_ROOT") or
                        Path(__file__).resolve().parents[1] / "build" / "question_profiles").expanduser().resolve(strict=True)
    index = json.loads((profile_root / "index.json").read_text(encoding="utf-8"))
    if index.get("schema") != INDEX_SCHEMA:
        raise ValueError("unsupported profile index schema")
    try:
        binding = index["questions"][question_id]
    except KeyError as error:
        raise KeyError(f"question ID is not registered: {question_id}") from error
    relative = Path(str(binding["profile"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("unsafe profile path")
    profile_path = (profile_root / relative).resolve(strict=True)
    profile_path.relative_to(profile_root)
    # SHA fields in the index are legacy metadata only.  They are deliberately
    # neither required nor compared, even when an older caller passes
    # require_digest=True.
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    if profile.get("schema") not in {
        "m20-attack-experiment/v2",
        "m20-attack-experiment/v3",
        "embodied-attack-runtime/v1",
    }:
        raise ValueError("unsupported attack profile schema")
    if str(profile.get("question_id", question_id)) != question_id:
        raise ValueError("profile question_id differs")
    return profile, profile_path
