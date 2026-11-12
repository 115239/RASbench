from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import h5py

from safety_tools.profile import canonical_profile_sha256


MODULE_PATH = Path(__file__).resolve().parents[1] / "robolab_runner/safety_hdf5_enricher.py"
SPEC = importlib.util.spec_from_file_location("safety_hdf5_enricher_under_test", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load {MODULE_PATH}")
SAFETY_HDF5_ENRICHER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SAFETY_HDF5_ENRICHER)
ROOT = SAFETY_HDF5_ENRICHER.ROOT
append_safety_extension = SAFETY_HDF5_ENRICHER.append_safety_extension


class SafetyHdf5EnricherTests(unittest.TestCase):
    def test_appends_complete_validated_extension(self):
        profile = {
            "schema": "m20-safety-profile/v1",
            "profile_id": "test-s09",
            "rule_set_version": "test/v1",
            "activation": {
                "start": "task_time_zero",
                "end": "episode_end",
                "sampling": "every_physics_step",
            },
            "response_mode": "record_only",
            "rules": {
                "S09": {
                    "enabled": True,
                    "parameters": {
                        "metrics": [
                            {
                                "criterion": "base_world",
                                "body_name": "base_link",
                                "components": ["x", "y"],
                                "maximum_speed_mps": 0.5,
                            }
                        ],
                    },
                }
            },
        }
        digest = canonical_profile_sha256(profile)
        summary = {
            "schema": "m20-safety-summary/v1",
            "complete": True,
            "profile_sha256": digest,
            "violated_rules": [],
            "unknown_rules": [],
        }
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            submission = root / "submission.hdf5"
            with h5py.File(submission, "w") as hdf5_file:
                hdf5_file.create_group("data")
            (root / "resolved_safety_profile.json").write_text(
                json.dumps(profile, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            (root / "safety_summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            (root / "safety_evidence.jsonl").write_text("", encoding="utf-8")

            report = append_safety_extension(
                submission,
                root,
                expected_profile_sha256=digest,
            )

            self.assertEqual(report["status"], "VALID")
            self.assertEqual(report["event_count"], 0)
            with h5py.File(submission, "r") as hdf5_file:
                self.assertIn(ROOT, hdf5_file)
                self.assertTrue(bool(hdf5_file[ROOT].attrs["complete"]))


if __name__ == "__main__":
    unittest.main()
