"""Attach continuous control to the repaired runner shipped in this image."""
from pathlib import Path
import sys
from common import IMAGE_RUNNER, BASE, read, sha, write
source_path = Path(IMAGE_RUNNER) / 'm20_runtime/runners/m20_fourview_runner.py'
source = source_path.read_text()
robot = read(BASE/'ROBOT_MANIFEST.json')
robot_sha = next(item['sha256'] for item in robot['files'] if item['path'] == robot['entrypoint'])
hash_anchor = 'EXPECTED_USD_SHA256 = "7ca91879191f18f71184f7511c39250d4b52dbec503335eaf33e412b35f42d18"'
if source.count(hash_anchor) != 1 and ('EXPECTED_USD_SHA256 = "'+robot_sha+'"') not in source:
    raise RuntimeError('ROBOT_HASH_INTERFACE_DRIFT')
source = source.replace(hash_anchor, 'EXPECTED_USD_SHA256 = '+repr(robot_sha), 1)
anchor = '    def run_policy(self, policy) -> dict:\n'
if source.count(anchor) != 1:
    raise RuntimeError('RUNNER_INTERFACE_DRIFT')
source = source.replace(anchor, anchor + '        from physical import run_continuous\n        return run_continuous(self, policy, globals())\n\n', 1)
begin = source.index('    def policy_observation(self, prompt: str)')
end = source.index('    def take_q17_policy_bridge', begin)
section = source[begin:end].replace('        self.world.step(render=True)', '        self.world.render()').replace('        self.physics_step += 1\n', '')
source = source[:begin] + section + source[end:]
trigger_anchor = "    add_reference_to_stage(str(EXPECTED_USD), robot_path)\n    configure_velocity_limits(stage, robot_path)\n"
if source.count(trigger_anchor) != 1:
    raise RuntimeError('TRIGGER_BINDING_INTERFACE_DRIFT')
source = source.replace(trigger_anchor, trigger_anchor + "    from runtime_contacts import bind_contacts\n    bind_contacts(stage, robot_path, args.output)\n" + "    from trigger_semantics import audit_authored_triggers\n    audit_authored_triggers(stage, args.output)\n", 1)
sys.path.insert(0, str(Path(IMAGE_RUNNER) / 'm20_runtime'))
exec(compile(source, str(source_path), 'exec'), {'__file__': str(source_path), '__name__': '__main__'})
