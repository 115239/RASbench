"""Independent input snapshots; no mutations to the source benchmark or old runs."""
from pathlib import Path
import ast
from dataclasses import asdict
import json
import re
import shutil
import subprocess
import sys
from .common import PROJECT, SOURCE, BASE, AGENT, SYSTEMS, digest, read, write, sha, now
from .queue import initialize


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('source interface drift: '+old[:120])
    return text.replace(old, new, 1)


def method(text, cls, name):
    tree = ast.parse(text)
    node = next(n for c in tree.body if isinstance(c, ast.ClassDef) and c.name == cls
                for n in c.body if isinstance(n, ast.FunctionDef) and n.name == name)
    return '\n'.join(text.splitlines()[node.lineno-1:node.end_lineno])+'\n'


def integrate_agent(runner, agent):
    """Graft only assurance hooks onto the CURRENT main Runner and attack handlers."""
    rel = Path('m20_runtime/runners/m20_fourview_runner.py')
    src = runner/rel
    overlay = agent/'integrations/safety_embodiment/competition_runner/runner'/rel
    text, old = src.read_text(), overlay.read_text()
    cls = next(n.name for n in ast.parse(text).body if isinstance(n,ast.ClassDef)
               and any(isinstance(m,ast.FunctionDef) and m.name=='run_policy' for m in n.body))
    hooks = '\n'.join(method(old,cls,n) for n in ['_ensure_assurance','_assurance_base','_assurance_joints'])
    text = replace_once(text,'    def run_policy(self, policy) -> dict:\n',hooks+'\n    def run_policy(self, policy) -> dict:\n')
    text = replace_once(text,'        self.attack_boundary = None\n',
                        '        self.attack_boundary = None\n        self.assurance_bridge = None\n        self._assurance_initialized = False\n')
    text = replace_once(text,'        stationary_command = is_zero_base_command(command)',
                        '        command = self._assurance_base(command)\n        stationary_command = is_zero_base_command(command)')
    # Both dynamic and explicit kinematic branches submit through the same final guard.
    text = replace_once(text,'        self.controller.apply_action(ArticulationAction(joint_positions=q_target, joint_velocities=qd_target))',
                        '        q_target, qd_target = self._assurance_joints(q_target, qd_target)\n        self.controller.apply_action(ArticulationAction(joint_positions=q_target, joint_velocities=qd_target))')
    text = replace_once(text,'        self.kinematic_position_targets = q_target[',
                        '        q_target, qd_target = self._assurance_joints(q_target, qd_target)\n        self.kinematic_position_targets = q_target[')
    start = old.index('        if self.assurance_bridge is not None:\n            wheel_indices')
    end = old.index('        observation = build_policy_observation(',start)
    text = replace_once(text,'        observation = build_policy_observation(',old[start:end]+'        observation = build_policy_observation(')
    receipt = '''        if self.assurance_bridge is not None:
            if physics_steps:
                self.assurance_bridge.executed(self.physics_step*PHYSICS_DT,str(phase))
            else:
                self.assurance_bridge.not_advanced()

'''
    text = replace_once(text,'    def policy_observation(self, prompt: str) -> dict:\n',receipt+'    def policy_observation(self, prompt: str) -> dict:\n')
    # Same camera correction for all three systems, before model observation.
    text = replace_once(text,'    def update_cameras(self) -> None:\n        if not self.cameras:\n            return\n',
        '    def update_cameras(self) -> None:\n        if not self.cameras:\n            return\n        for camera in self.cameras:\n            camera.set_clipping_range(0.01, 1000.0)\n')
    ast.parse(text); src.write_text(text)
    client = runner/'m20_runtime/attack_tools/attackable_vla_client.py'
    overlay_client = agent/'integrations/safety_embodiment/competition_runner/runner/m20_runtime/attack_tools/attackable_vla_client.py'
    text, old = client.read_text(),overlay_client.read_text()
    # Preserve new joint-encoder intercept and handler fixes; use only Agent-aware transactions.
    for cls,name in [('AttackableVLAClient','infer'),('EvidenceRecorder','capture_vla_transaction')]:
        text = replace_once(text,method(text,cls,name),method(old,cls,name))
    ast.parse(text);client.write_text(text)
    entry=runner/'robolab_runner/m20_question_runner.py'
    text=entry.read_text()
    text=replace_once(text,'    args = parser.parse_args()','    parser.add_argument("--experiment-seed", type=int)\n    args = parser.parse_args()')
    text=replace_once(text,'    task = load_json(package_task_path)','    task = load_json(package_task_path)\n    if args.experiment_seed is not None:\n        task["seed"] = args.experiment_seed')
    ast.parse(text);entry.write_text(text)
    write(runner/'PAPER200_INTEGRATION.json',{'source':'current main Runner','agent':'isolated paper200 assurance hooks',
          'preserved':'current attack handlers, sensor_joint_encoder, rubric evidence and native HDF5 export',
          'camera_near_clip_m':0.01,'created':now()})


def primitives(value):
    text = json.dumps(value, ensure_ascii=False).replace('\\_','_')
    return sorted(set(p.rstrip('.') for p in re.findall(r'/World(?:_01)?/[A-Za-z0-9_./\-\u4e00-\u9fff]+',text)))


def stages(task, manifest):
    instruction = task['instruction']
    kind = manifest['task_type']
    nav = task.get('execution_mode') not in ('manipulation_only','grasp_only')
    result = ['NAV_OUTBOUND'] if nav or kind=='navigation' else []
    if kind != 'navigation':
        operation = 'PICK'
        if kind=='articulated_interaction':
            operation = ('PUSH' if any(w in instruction for w in ['推动','推向','推开']) else
                         'BUTTON' if any(w in instruction for w in ['按下','按钮']) else 'OPEN')
        result.append(operation)
        if kind=='mobile_manipulation':
            result += ['HANDOFF','NAV_RETURN' if any(w in instruction for w in ['返回','回到']) else 'NAV_DELIVERY']
        if kind=='pick_place' or any(w in instruction for w in ['放入','放到','放在','放回','放置','放下','放进']):
            result.append('PLACE')
    elif any(w in instruction for w in ['返回','回到']):
        result.append('NAV_RETURN')
    return result


def prepare(root):
    root = Path(root).resolve()
    if (root/'manifests/protocol.json').exists():
        raise FileExistsError('completed snapshot is immutable; choose a new version')
    if root.exists() and not (root/'PREPARING.json').exists():
        raise FileExistsError('refusing an unrelated existing directory')
    root.mkdir(parents=True,exist_ok=True)
    write(root/'manifests/pi05_transfer.json',{'created':now(),**read(PROJECT/'experiment/config/pi05_transfer.json')})
    write(root/'PREPARING.json',{'created':now(),'source':str(SOURCE),'status':'IN_PROGRESS'})
    benchmark = root/'benchmark'
    benchmark.mkdir(exist_ok=True)
    ignore = shutil.ignore_patterns('__pycache__','.pytest_cache','*.pyc','.git')
    shutil.copytree(SOURCE/'task',benchmark/'task',ignore=ignore,dirs_exist_ok=True)
    shutil.copy2(SOURCE/'UNIFIED_RELEASE.json',benchmark/'UNIFIED_RELEASE.json')
    if not (benchmark/'assets').exists():
        subprocess.run(['cp','-a','--reflink=auto',str(SOURCE/'assets'),str(benchmark/'assets')],check=True)
    runtime = root/'runtime'
    shutil.copytree(BASE/'competition_runner/runner',runtime/'runner',ignore=ignore,dirs_exist_ok=True)
    shutil.copytree(AGENT,runtime/'agent',ignore=ignore,dirs_exist_ok=True)
    shutil.copytree(BASE/'baseline/code',runtime/'baseline_code',ignore=ignore,dirs_exist_ok=True)
    # Baseline's compact overlay modules resolve model/framework dependencies via PAPER_BASELINE_ROOT.
    overlay = runtime/'agent/integrations/safety_embodiment/baseline/code/runner'
    shutil.copytree(overlay,runtime/'baseline_runner',ignore=ignore,dirs_exist_ok=True)
    integrate_agent(runtime/'runner',runtime/'agent')
    sys.path.insert(0,str(runtime/'agent'))
    from safety_agent.task_spec import TaskSpec, public_requirements, infer_stages, clean_text
    # Select only public columns. The attack column is never sent to the Agent.
    rows = {}
    for line in (PROJECT/'questions/论文Bench赛题_Q1-Q200.md').read_text().splitlines():
        match = re.match(r'\|Q(\d+)(?:<br\s*/?>|\s)*\|',line)
        if match:
            cells = re.split(r'(?<!\\)\|',line)[1:-1]
            if len(cells) < 5:
                raise ValueError('invalid public question row')
            rows['Q'+match[1]] = {'assets':clean_text(cells[2]),'constraints':clean_text(cells[4])}
    questions = []
    files = {}
    for n in range(1,201):
        q='Q%d'%n; package=benchmark/'task'/q
        task=read(package/'data/task/task.json'); manifest=read(package/'package_manifest.json')
        rubric=read(package/'data/task/rubric.json'); registry=read(package/'data/task/object_registry.json')
        contract=read(next((package/'contracts').glob('*/contract.json')))
        variants=manifest['attack_variant_ids']
        if set(variants)!=set(contract['attack_variants']) or len(variants)!=len(set(variants)):
            raise ValueError(q+': attack variant mismatch')
        duration=next(c['params']['seconds'] for c in task['termination']['failure']['conditions'] if c['evaluator']=='maximum_duration')
        task_stages=stages(task,manifest)
        # Record exact semantic objects used by the rubric, not surrogate cubes.
        refs=primitives(rubric)
        objects=sorted(set(refs)|set(registry['objects'].values()))
        plan={'schema':'bboard-rubric-evidence-plan/v1','task_id':task['task_id'],'rubric_id':q,
              'rubric_sha256':sha(package/'data/task/rubric.json'),'rubric_prim_references':refs,
              'objects':[{'prim':p,'runtime_prim':p} for p in objects], 'contacts':[]}
        task['rubric_evidence']=plan
        write(package/'data/task/task.json',task)
        public={'question':'Q%03d'%n,'scene':manifest['scene_id'],'instruction':task['instruction'],**rows[q]}
        spec=TaskSpec(root.name,public['question'],public['scene'],public['instruction'],public['assets'],
             public['constraints'],infer_stages(public['instruction']),tuple(primitives(public['assets'])),
             tuple(sorted(set(re.findall(r'\bS\d{2}\b',public['constraints'])))),(),digest(public),
             action_budget=task['maximum_vla_actions'],duration_s=duration,requirements=public_requirements(public))
        write(root/'manifests/public'/q/'task.json',asdict(spec))
        row={'question':q,'task_id':task['task_id'],'scene':manifest['scene_id'],'task_type':manifest['task_type'],
             'stages':task_stages,'variants':variants,'duration_s':duration,'action_budget':task['maximum_vla_actions'],
             'rubric_sha256':plan['rubric_sha256'],'rubric_prims':refs,
             'unresolved_rubric_text':bool(re.search('待确认|待评测|待补|待绑定',json.dumps(rubric,ensure_ascii=False)))}
        questions.append(row)
    for p in sorted(benchmark.rglob('*')):
        if p.is_file(): files[str(p.relative_to(root))]=sha(p)
    for p in sorted(runtime.rglob('*')):
        if p.is_file(): files[str(p.relative_to(root))]=sha(p)
    model_roots={'navigation':BASE/'baseline/models/vln','manipulation':BASE/'baseline/models/vla',
                 'pi05':PROJECT/'model/pi05_base','locomotion':BASE/'baseline/models/locomotion/policy.onnx'}
    models={}
    for role,path in model_roots.items():
        if path.is_file(): selected=[path]
        elif role=='pi05': selected=sorted(p for p in path.rglob('*') if p.is_file())
        else:
            selected=[p for p in path.iterdir() if p.is_file() and p.suffix in ('.json','.py','.safetensors')]
        models[role]={'path':str(path),'files':{str(p.relative_to(path) if path.is_dir() else p.name):sha(p) for p in selected}}
    write(root/'manifests/models.json',models)
    write(root/'manifests/files.json',files)
    write(root/'manifests/questions.json',questions)
    protocol={'schema':'paper200-collection/v1','version':root.name,'created':now(),'systems':list(SYSTEMS),
              'main_seed':82026001,'pilot_seed':72026001,'main_count':sum(1+len(q['variants']) for q in questions)*3,
              'source':str(SOURCE),'rubric_authority':'benchmark/task/Q*/data/task/rubric.json',
              'capture_hz':10,'action_hz':2,'physics_hz':200,'score_status':'NOT_SCORED',
              'source_snapshot_sha256':digest(files),'model_manifest_sha256':sha(root/'manifests/models.json'),
              'readiness':'PILOT_REQUIRED','assisted_question_list':'NOT_YET_FROZEN'}
    write(root/'manifests/protocol.json',protocol)
    initialize(root,questions,root.name)
    write(root/'reports/preparation.json',{'snapshot_complete':True,'questions':len(questions),
          'main_runs':protocol['main_count'],'physics_validated':False,'rubric_binding_issues':[q['question'] for q in questions if q['unresolved_rubric_text']]})
    return protocol
