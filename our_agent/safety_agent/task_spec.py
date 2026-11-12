"""Public task contracts. No attack descriptions or scoring predicates cross this boundary."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
import hashlib
import html
import json
import re


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(p)


def clean_text(text):
    text = re.sub(r'<br\s*/?>', '\n', text, flags=re.I)
    text = re.sub(r'\\([_*.+\-()[\]{}|])', r'\1', text)
    return html.unescape(text).replace('```Plain Text', '').replace('```', '').replace('**', '').replace('`', '').strip()


def public_rows(directory):
    """Select columns BEFORE interpreting text; never return attack/rubric columns."""
    seen = set()
    for path in sorted(Path(directory).glob('paper_*.md')):
        scene = None
        for line in path.read_text(encoding='utf-8').splitlines():
            if line.startswith('# '):
                heading = clean_text(line[2:])
                if heading and not any(s in heading for s in ('paper', '论文Bench')):
                    scene = re.sub(r'[（(].*', '', heading).strip()
            match = re.match(r'\|Q(\d+)', line)
            if not match:
                continue
            number = int(match[1])
            if number in seen:
                raise ValueError('duplicate question: ' + str(number))
            seen.add(number)
            cells = re.split(r'(?<!\\)\|', line)[1:-1]
            if len(cells) < 5 or not scene:
                raise ValueError('invalid public row: ' + str(number))
            yield {'question': f'Q{number:03d}', 'scene': scene,
                   'instruction': clean_text(cells[1]), 'assets': clean_text(cells[2]),
                   'constraints': clean_text(cells[4])}


def infer_stages(instruction):
    stages = []
    if any(s in instruction for s in ('导航', '前往', '找到', '移动至', '移动到', '走到', '停下', '停靠')):
        stages.append('NAVIGATE')
    if any(s in instruction for s in ('抓', '拿', '取下', '夹取', '提起', '抬起', '拾取')):
        stages.append('PICK')
    for words, name in [(('推动', '推开', '推向'), 'PUSH'), (('按下', '按钮'), 'BUTTON'),
                        (('拉开', '打开', '开门'), 'OPEN')]:
        if any(s in instruction for s in words) and name not in stages:
            stages.append(name)
    if 'PICK' in stages and any(s in instruction for s in ('移动', '放入', '放到', '放在', '返回')):
        stages.append('TRANSPORT')
    if any(s in instruction for s in ('放入', '放到', '放进', '放置', '横放', '放在', '放下', '放回')):
        stages.append('PLACE')
    if any(s in instruction for s in ('返回', '回到')):
        stages.append('RETURN')
    return tuple(stages or ['OBSERVE'])


def public_requirements(row):
    """Lossless public clauses plus conservative, explicitly sourced numeric limits.

    Free text stays authoritative; no unprovided default or geometric region is invented.
    """
    clauses = [v.strip() for v in row['constraints'].splitlines() if v.strip()]
    by_rule = {}
    for clause in clauses:
        for rule in re.findall(r'\bS\d{2}\b', clause, re.I):
            by_rule.setdefault(rule.upper(), []).append(clause)
    limits = {}
    for rule, key, unit in [('S08', 'clearance_m', r'(?:米|m\b)'),
                            ('S09', 'base_speed_mps', r'm/s'),
                            ('S10', 'yaw_rate_rps', r'rad/s')]:
        values = [float(v) for text in by_rule.get(rule, [])
                  for v in re.findall(r'(\d+(?:\.\d+)?)\s*'+unit, text)]
        if values:
            limits[key] = max(values) if rule == 'S08' else min(values)
    sentences = [v.strip() for v in re.split(r'[。；\n]', row['instruction']) if v.strip()]
    return {'goals': sentences, 'stage_order_source': 'public_instruction_lexical_parser',
            'spatial_relations': [v for v in sentences if any(w in v for w in ('左', '右', '上', '下', '前', '后', '旁', '内', '外'))],
            'legal_contact': by_rule.get('S01', []),
            'placement': by_rule.get('S14', []) + [v for v in sentences if '放' in v],
            'rule_clauses': by_rule, 'numeric_limits': limits,
            'geometry_authority': 'PUBLIC_DESCRIPTION_ONLY; no simulator object poses'}


@dataclass(frozen=True)
class TaskSpec:
    benchmark_id: str
    question: str
    scene: str
    instruction: str
    assets: str
    constraints: str
    stages: tuple[str, ...]
    public_prims: tuple[str, ...]
    enabled_rules: tuple[str, ...]
    unresolved: tuple[str, ...]
    public_row_sha256: str
    action_budget: int | None = None
    duration_s: float | None = None
    schema: str = 'safe-agent-task/v2'
    requirements: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.schema != 'safe-agent-task/v2' or not re.fullmatch(r'Q\d{3}', self.question):
            raise ValueError('invalid versioned task identity')
        if not self.benchmark_id or not self.instruction or not self.scene:
            raise ValueError('empty public task identity')
        if self.action_budget is not None and (type(self.action_budget) is not int or self.action_budget <= 0):
            raise ValueError('invalid action budget')
        if self.duration_s is not None and (not 0 < self.duration_s < float('inf')):
            raise ValueError('invalid duration')

    @property
    def identity(self):
        return self.benchmark_id + ':' + self.question

    def bind(self, *, action_budget, duration_s):
        # Binding is supplied by the runner at episode creation, never by observations.
        if self.action_budget is not None and self.action_budget != action_budget:
            raise ValueError('task action budget mismatch')
        if self.duration_s is not None and self.duration_s != duration_s:
            raise ValueError('task duration mismatch')
        return replace(self, action_budget=action_budget, duration_s=duration_s)

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text())
        if set(data) != set(cls.__dataclass_fields__):
            raise ValueError('TaskSpec requires exactly the public allowlisted fields')
        for key in ('stages', 'public_prims', 'enabled_rules', 'unresolved'):
            data[key] = tuple(data[key])
        return cls(**data)


def compile_questions(directory, output):
    rows = list(public_rows(directory))
    if {r['question'] for r in rows} != {f'Q{n:03d}' for n in range(1, 201)}:
        raise ValueError('expected every question Q001 through Q200 exactly once')
    identity = hashlib.sha256(json.dumps(rows, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    benchmark = 'paper200-public-' + identity[:16]
    out = Path(output) / benchmark
    tasks = []
    for row in rows:
        rules = tuple(sorted(set(re.findall(r'\bS\d{2}\b', row['constraints']))))
        prims = tuple(dict.fromkeys(re.findall(r'/World[^\s（）()，,。；;]+', row['assets'])))
        issues = ['RUNTIME_BUDGET_BINDING_REQUIRED']
        if not row['constraints']:
            issues.append('PUBLIC_CONSTRAINTS_MISSING')
        if 'default' in row['constraints']:
            issues.append('RULE_DEFAULTS_REQUIRE_FROZEN_BENCHMARK')
        spec = TaskSpec(benchmark, row['question'], row['scene'], row['instruction'],
                        row['assets'], row['constraints'], infer_stages(row['instruction']),
                        prims, rules, tuple(issues),
                        hashlib.sha256(json.dumps(row, ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
                        requirements=public_requirements(row))
        path = out / row['question'] / 'task.json'
        write_json(path, asdict(spec))
        tasks.append({'question': spec.question, 'path': str(path.relative_to(out)),
                      'sha256': sha256(path), 'unresolved': issues, 'scene': spec.scene})
    manifest = {'schema': 'safe-agent-public-manifest/v1', 'benchmark_id': benchmark,
                'source_files': {p.name: sha256(p) for p in sorted(Path(directory).glob('paper_*.md'))},
                'task_count': len(tasks), 'tasks': tasks,
                'runtime_input_columns': ['instruction', 'assets', 'constraints'],
                'physical_acceptance': 'NOT_EVALUATED'}
    write_json(out / 'manifest.json', manifest)
    return out / 'manifest.json'
