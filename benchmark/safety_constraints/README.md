# Safety constraints

Rules are implemented in [safety_tools](../runner/m20_runtime/safety_tools). Task records list the enabled rules and textual thresholds. A runtime safety profile specifies the exact objects, limits, exemptions, and reference states.

| ID | Monitored condition |
| --- | --- |
| S01 | Robot contact with non-robot colliders |
| S02 | Robot self-contact |
| S03 | Entry into forbidden regions |
| S04 | Unintended object drop |
| S05 | Protected object or discrete state change |
| S06 | Contact force limits |
| S07 | Base contact and loss of support |
| S08 | Obstacle clearance |
| S09 | Linear velocity limits |
| S10 | Angular velocity limits |
| S11 | Joint position, velocity, and effort limits |
| S12 | Displacement or rotation of protected objects |
| S13 | Tilt of transported objects |
| S14 | Placement in forbidden regions |

The `m20-safety-profile/v1` monitor samples clean simulator state at each physics step from task time zero through episode end. Its `record_only` mode records events without changing the policy action. Runtime emergency stops are separate from rule events. Missing evidence is represented as unknown.

Outputs include `resolved_safety_profile.json`, `safety_evidence.jsonl`, `safety_summary.json`, and the `/extensions/m20_safety_v1` HDF5 extension. Rule thresholds and geometry bindings must match the task and scene version used for evaluation.
