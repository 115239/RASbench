import json

from m20_runtime.attack_tools.question_profiles import load_question_profile
from m20_runtime.navigation.core import load_route


def test_route_accepts_missing_and_invalid_legacy_sha_fields(tmp_path):
    package = tmp_path / "QTEST"
    route_dir = package / "data" / "env"
    route_dir.mkdir(parents=True)
    scene = package / "scene.usda"
    scene.write_text("#usda 1.0\n", encoding="utf-8")
    route = route_dir / "route.json"
    route.write_text(
        json.dumps(
            {
                "schema": "m20-piper-known-map-scene-route/v1",
                "name": "no_sha_gate",
                "maximum_duration_s": 10.0,
                "controller": {
                    "forward_speed_mps": 0.2,
                    "heading_gain": 1.0,
                    "maximum_yaw_rate_rps": 0.3,
                    "rotate_in_place_threshold_rad": 0.4,
                    "slowdown_distance_m": 0.5,
                    "arrival_tolerance_m": 0.2,
                    "settle_seconds": 1.0,
                    "brake_seconds": 1.0,
                },
                "spawn": {"xyz": [0.0, 0.0, 0.65], "yaw_rad": 0.0},
                "waypoints": [[1.0, 0.0]],
                "scene": {
                    "path": "../../scene.usda",
                    "sha256": "this-is-deliberately-not-a-digest",
                    "default_prim": "/World",
                    "source_physics_scene": None,
                    "minimum_collision_prim_count": 1,
                    "camera_mode": "chase",
                    "robot_radius_m": 0.0,
                    "obstacle_aabbs_xy_m": [
                        {
                            "name": "sentinel",
                            "minimum": [50.0, 50.0],
                            "maximum": [51.0, 51.0],
                        }
                    ],
                },
            }
        ),
        encoding="utf-8",
    )

    parsed = load_route(route)

    assert parsed.scene is not None
    assert parsed.scene.path == scene
    assert parsed.scene.bundle_manifest is None
    assert len(parsed.scene.sha256) == 64


def test_attack_profile_ignores_missing_or_wrong_sha_even_for_legacy_strict_caller(tmp_path):
    profile = {
        "schema": "m20-attack-experiment/v3",
        "question_id": "QTEST",
        "attack": {"attacks": []},
    }
    (tmp_path / "profile.json").write_text(json.dumps(profile), encoding="utf-8")
    (tmp_path / "index.json").write_text(
        json.dumps(
            {
                "schema": "m20-question-attack-profile-index/v1",
                "questions": {
                    "QTEST": {
                        "profile": "profile.json",
                        "sha256": "wrong-and-not-even-hex",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    loaded, path = load_question_profile(
        "QTEST", root=tmp_path, require_digest=True
    )

    assert loaded == profile
    assert path == tmp_path / "profile.json"
