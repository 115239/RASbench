"""Own experiment subprocess groups, including when their parent is interrupted.

The persistent ledger prevents a retry from racing an orphaned simulator after a
hard worker crash. It is evaluation infrastructure, never an Agent observation.
"""
from pathlib import Path
from contextlib import contextmanager
import os
import signal
import socket
import subprocess
import time
import uuid
import json
from .task_spec import write_json


@contextmanager
def interruptible():
    """Give callers a chance to flush their failure record on SIGTERM/Ctrl-C."""
    def interrupted(signum, _frame):
        raise SystemExit(128 + signum)
    handlers = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        yield
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


def group_members(pgid):
    """Linux process-group membership, excluding zombies that cannot execute."""
    members = []
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            fields = (path/'stat').read_text().rsplit(')', 1)[1].split()
            if int(fields[2]) == pgid and fields[0] not in {'Z', 'X'}:
                members.append(int(path.name))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return members


def active_owned_processes(output):
    active = []
    for path in (Path(output)/'owned_processes').glob('*.json'):
        record = json.loads(path.read_text())
        if record.get('stopped') is True:
            continue
        if record['host'] != socket.gethostname() or group_members(record['pgid']):
            active.append(record)
    return active


class OwnedProcesses:
    def __init__(self, output, role, grace_s=5.):
        self.output, self.role = Path(output), role
        self.grace_s = grace_s
        self.children = []
        self.handlers = {}

    def __enter__(self):
        def interrupted(signum, _frame):
            raise SystemExit(128 + signum)
        for sig in (signal.SIGTERM, signal.SIGINT):
            self.handlers[sig] = signal.signal(sig, interrupted)
        return self

    def start(self, argv, **kwargs):
        if 'start_new_session' in kwargs:
            raise ValueError('process group ownership is managed here')
        process = subprocess.Popen(argv, start_new_session=True, **kwargs)
        record = {'host': socket.gethostname(), 'owner_pid': os.getpid(),
                  'role': self.role, 'pid': process.pid, 'pgid': process.pid,
                  'argv': list(map(str, argv)), 'stopped': False}
        path = self.output/'owned_processes'/f'{self.role}-{process.pid}-{uuid.uuid4().hex}.json'
        # Register the handle before writing, so an I/O failure still cleans it up.
        self.children.append((process, path, record))
        write_json(path, record)
        return process

    def __exit__(self, exc_type, exc, tb):
        for sig in self.handlers:
            signal.signal(sig, signal.SIG_IGN)
        try:
            for process, _, _ in self.children:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            deadline = time.monotonic() + self.grace_s
            while any(group_members(p.pid) for p, _, _ in self.children) and time.monotonic() < deadline:
                time.sleep(.05)
            for process, path, record in self.children:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=5)
                record.update(stopped=not bool(group_members(process.pid)), returncode=process.returncode)
                write_json(path, record)
        finally:
            for sig, handler in self.handlers.items():
                signal.signal(sig, handler)
        return False
