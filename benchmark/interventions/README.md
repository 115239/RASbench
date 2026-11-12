# Interventions

The benchmark addresses instruction, RGB, LiDAR, joint-state, IMU, and motion/manipulation-command interfaces.

- [Task source](../tasks/task_source_zh.md): task-specific intervention descriptions.
- [Attack tools](../runner/m20_runtime/attack_tools): intervention implementations.
- [Collection integration](../evaluation/paper200/attack_runtime.py): runtime intervention integration.
- [Activation evidence](../evaluation/paper200/attack_proof.py): evidence checks.

An intervention specification binds its interface, target, trigger, parameters, duration, and activation evidence. Registered interventions and successfully activated interventions are separate populations. Unreached triggers must remain visible in coverage statistics.
