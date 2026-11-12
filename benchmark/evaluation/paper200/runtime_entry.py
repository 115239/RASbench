"""Use the same current physical Collector and assurance hooks for every system."""
from pathlib import Path
import json
import os
import sys

# Executed by file path: prevent this package's queue.py/common.py from
# shadowing the standard library or the baseline's explicitly selected modules.
sys.path=[p for p in sys.path if Path(p or '.').resolve()!=Path(__file__).resolve().parent]

cfg=json.loads(Path(os.environ['THREE_SCENES_RUN_CONFIG']).read_text())
root=Path(cfg['collection_root'])
# Periodic faulthandler dumps race Python 3.11 frame destruction and can crash
# an otherwise running collector. Use retained physics_clock.json for progress
# and external process inspection for diagnostics; do not add a dump timer.

from paper200.handler_probe import prepare as prepare_handler_probe
prepare_handler_probe(cfg)
from paper200.control_q1 import prepare_route as prepare_q1_control_route
prepare_q1_control_route(cfg)
from paper200.control_visual import prepare as prepare_visual_control
prepare_visual_control(cfg)
from paper200.control_direction import prepare as prepare_direction_control
prepare_direction_control(cfg)
if 'scene_key' in cfg:
    from paper200.runtime_cache import prepare as prepare_cache
    prepare_cache(root,Path(cfg['output']),cfg['scene_key'])
runtime_root=Path(cfg.get('runtime_snapshot',str(root/'runtime')))
sys.path.insert(0,str(runtime_root/'baseline_runner'))
source_path=runtime_root/'runner/m20_runtime/runners/m20_fourview_runner.py'
source=source_path.read_text()
from paper200.attack_runtime import install
install()
source=source.replace('from rubric_evidence_io import RubricEvidenceWriter',
    'from paper200.physics_evidence import RubricEvidenceWriter',1)
source=source.replace('from rubric_evidence import RubricEvidenceRuntime',
    'from paper200.physics_evidence import RubricEvidenceRuntime',1)
# The inherited boundary intercepted leg targets before Piper targets existed.
# Move it to the complete transaction, retaining the same common limit clamps
# and the subsequent Agent/actuator receipt boundary for all systems.
joint_block='''        if self.attack_boundary is not None:
            intercepted_targets = self.attack_boundary.intercept_joint_control(
                {
                    "positions": dict(positions),
                    "wheel_velocities": dict(wheel_velocities),
                },
                physics_step=self.physics_step,
                time_s=self.physics_step * PHYSICS_DT,
            )
            positions = dict(intercepted_targets["positions"])
            wheel_velocities = dict(intercepted_targets["wheel_velocities"])
'''
if source.count(joint_block)!=2:raise RuntimeError('joint attack boundary hook drift')
source=source.replace(joint_block,'')
joint_apply='        q_target, qd_target = self._assurance_joints(q_target, qd_target)'
if source.count(joint_apply)!=2:raise RuntimeError('complete joint boundary hook drift')
source=source.replace(joint_apply,'        from paper200.joint_boundary import intercept as intercept_all_joints\n        q_target, qd_target = intercept_all_joints(self, q_target, qd_target, globals())\n'+joint_apply)
source=source.replace('''        self.session = None if args.kinematic_base else ort.InferenceSession(
            str(POLICY), providers=["CPUExecutionProvider"]
        )''','''        ort_options=ort.SessionOptions()
        ort_options.intra_op_num_threads=2
        ort_options.inter_op_num_threads=1
        self.session = None if args.kinematic_base else ort.InferenceSession(
            str(POLICY), sess_options=ort_options, providers=["CPUExecutionProvider"]
        )''',1)
# Current question wrapper emits the legacy flag. Mission stages already carry
# this meaning; expose it to argparse for protocol compatibility.
source=source.replace('    AppLauncher.add_app_launcher_args(result)',
    '    result.add_argument("--manipulation-only", action="store_true")\n    AppLauncher.add_app_launcher_args(result)',1)
source=source.replace('"--/renderer/multiGpu/enabled=false',
    '"--/rtx-transient/stableIds/enabled=true --/plugins/carb.tasking.plugin/threadCount=32 --/app/extensions/fsWatcherEnabled=false --/renderer/multiGpu/enabled=false',1)
source=source.replace('simulation_app = AppLauncher(args).app',
    'from paper200.native_runtime import configure as configure_native, verify as verify_native\nconfigure_native(args)\nsimulation_app = AppLauncher(args).app\nverify_native(args.output)\nfrom paper200.cpu_threads import configure as configure_cpu_threads\nconfigure_cpu_threads(args.output)',1)
video_start=source.index('            if args.four_view_video:\n',source.index('    def capture(')) if '    def capture(' in source else source.index('            if args.four_view_video:\n')
video_end=source.index('            frame.save(',video_start)
source=source[:video_start]+'            from paper200.review_video import compose\n            frame = compose(self)\n'+source[video_end:]
# The four existing render products retain their physical camera poses. This
# prevents an occluded follow camera from hiding all review evidence.
source=source.replace('"standard third-person follow camera"',
    '"synchronised follow/front/wrist/rear review mosaic; image-space attacks retained in model wire evidence"')
source=source.replace('else ["third_person_follow"]',
    'else ["follow_camera", "front_camera_before_image_attack", "wrist_camera_before_image_attack", "rear_camera_before_image_attack"]')
source=source.replace('''            raise RuntimeError("declared source PhysicsScene is missing")''','''            available = [p for p in stage.Traverse() if p.IsA(UsdPhysics.Scene)]
            if len(available) > 1:
                raise RuntimeError("ambiguous source PhysicsScene")
            source_physics = available[0] if available else None
            (args.output / "physics_scene_binding.json").write_text(json.dumps({
                "declared": route.scene.source_physics_scene,
                "resolved": str(source_physics.GetPath()) if source_physics else None,
                "reason": "resolve actual composed source scene before creating common runtime PhysX scene"}))''',1)
source=source.replace('        source_physics.SetActive(False)',
    '        if source_physics is not None: source_physics.SetActive(False)',1)
anchor='    def run_policy(self, policy) -> dict:\n'
setup_anchor='    # Fix drifting ordinary furniture while preserving every registered task\n'
if source.count(setup_anchor)!=1:raise RuntimeError('scene binding hook drift')
source=source.replace(setup_anchor,'    from paper200.scene_adapters import materialize_visual_targets\n    materialize_visual_targets(stage)\n'+setup_anchor,1)
# The unused training prop must not fall forever in navigation-only tasks.
source=source.replace('    object_contact_api = PhysxSchema.PhysxContactReportAPI.Apply',
    '''    if TASK_CONFIG.get('execution_mode') == 'navigation_only' and not scene_object_prim:
        UsdPhysics.RigidBodyAPI(stage.GetPrimAtPath(scoring_object_prim)).CreateKinematicEnabledAttr().Set(True)
        UsdPhysics.CollisionAPI(stage.GetPrimAtPath(scoring_object_prim)).CreateCollisionEnabledAttr().Set(False)
    object_contact_api = PhysxSchema.PhysxContactReportAPI.Apply''',1)
if source.count(anchor)!=1:raise RuntimeError('Collector interface drift')
source=source.replace(anchor,anchor+'        from paper200.rollout import run\n        return run(self, policy, globals())\n\n',1)
# Sensor-derived kinematics permit the pi05 adapter to emit the SAME attacked
# 10-D end-effector command contract, with no hidden joint_delta bypass.
anchor='    def policy_observation(self, prompt: str) -> dict:\n'
source=source.replace(anchor,'''    def policy_observation(self, prompt: str) -> dict:
        result = self._paper_original_observation(prompt)
        from paper200.rollout import arm_jacobian
        result['observation/arm_jacobian'] = arm_jacobian(self, globals())
        return result

    def _paper_original_observation(self, prompt: str) -> dict:
''',1)
# Keep the base HDF5 camera/state protocol at 10 Hz; independently sample
# rubric evidence after EVERY physics step, including short contact events.
old='''        if self.rubric_evidence_runtime is not None:
            rows, contacts = self.rubric_evidence_runtime.sample()
            self.rubric_evidence_writer.step(len(self.scoring_samples), state_time_s, rows, contacts)
'''
if source.count(old)!=1:raise RuntimeError('rubric sample hook drift')
source=source.replace(old,'',1)
anchor='''                self.rubric_evidence_runtime.end_physics_step()
'''
source=source.replace(anchor,anchor+'''                if self.scoring_active:
                    rows, contacts = self.rubric_evidence_runtime.sample()
                    self.rubric_evidence_writer.step(self.rubric_evidence_writer.count,
                        (self.physics_step + 1) * PHYSICS_DT, rows, contacts)
                    if getattr(self, 'paper_inventory_writer', None) is not None:
                        self.paper_inventory_writer.step((self.physics_step + 1) * PHYSICS_DT)
''',1)
source=source.replace('self.rubric_evidence_writer.complete(len(scoring))',
                      'self.rubric_evidence_writer.complete(self.rubric_evidence_writer.count)')
# Per-process Kit caches avoid concurrent instances rewriting shared state.
if '--portable-root' not in sys.argv:
    sys.argv+=['--portable-root',str(Path(cfg['output'])/'kit')]
from paper200.resident_scene import execute_collector
execute_collector(source,source_path)
