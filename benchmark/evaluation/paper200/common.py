from pathlib import Path
import hashlib
import json
import os
import time

PROJECT = Path(os.environ.get('PAPER200_PROJECT_ROOT',str(Path(__file__).resolve().parents[2])))
SOURCE = PROJECT / 'mc/tibao/unified_q1_q200_20260924'
BASE = PROJECT / 'mc/runner_paper_200/safety_embodiment'
AGENT = PROJECT / 'mc/runner_paper_200/safe_agent/paper200'
NATIVE = BASE / 'competition_runner/native'
SYSTEMS = ('groot_agent_on', 'groot_agent_off', 'pi05_agent_off')
MAIN_SEED = 82026001
PILOT_SEED = 72026001


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.%s.tmp' % os.getpid())
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n',encoding='utf-8')
    temporary.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def now():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def question(value):
    n = int(str(value).upper().removeprefix('Q'))
    if not 1 <= n <= 200:
        raise ValueError('question outside Q1-Q200')
    return 'Q%d' % n


def environment(root, gpu=None):
    env = dict(os.environ)
    env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
               PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='4',
               OPENBLAS_NUM_THREADS='4', ACCEPT_EULA='Y', OMNI_KIT_ACCEPT_EULA='Y', PRIVACY_CONSENT='Y',
               PYTHONUTF8='1',NO_ALBUMENTATIONS_UPDATE='1')
    if gpu is not None:
        env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    return env
