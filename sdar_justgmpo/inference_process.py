"""Run inference in its own process group, bounded by actual token progress."""
import os
from pathlib import Path
import signal
import subprocess
import time


def inference_env(gpus):
    env = dict(os.environ)
    for key in list(env):
        if key in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'LOCAL_WORLD_SIZE', 'GROUP_RANK',
                   'ROLE_RANK', 'ROLE_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT') or key.startswith('TORCHELASTIC_'):
            env.pop(key)
    env['CUDA_VISIBLE_DEVICES'] = gpus
    env['SDAR_FULL_VOCAB'] = '1'
    return env


class TokenProgress:
    """Publish at most once a second, and only after receiving new tokens."""
    def __init__(self, path):
        self.path = Path(path)
        self.tokens, self.updated, self.logged = 0, float('-inf'), float('-inf')

    def __call__(self, count):
        if count <= 0:
            return
        self.tokens += count
        now = time.monotonic()
        if now - self.updated >= 1:
            self.path.write_text(str(self.tokens))
            self.updated = now
        if now - self.logged >= 60:
            print(f'Generated {self.tokens} tokens', flush=True)
            self.logged = now


def stop_process_group(proc):
    """Reap the leader and stop its engine children, including on cancellation."""
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        proc.wait(timeout=5)
        return
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        proc.poll()
        try:
            os.killpg(proc.pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    proc.wait(timeout=5)


def run_inference(command, *, env, log, progress, timeout):
    """Startup and shutdown also count toward the no-progress deadline."""
    progress = Path(progress)
    progress.unlink(missing_ok=True)
    with Path(log).open('a') as file:
        proc = subprocess.Popen(command, env=env, stdout=file, stderr=subprocess.STDOUT,
                                start_new_session=True)
        def terminate(signum, frame):
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            raise SystemExit(128 + signum)
        previous = signal.signal(signal.SIGTERM, terminate)
        updated, stamp = time.monotonic(), None
        try:
            while proc.poll() is None:
                if progress.exists():
                    current = progress.stat().st_mtime_ns
                    if current != stamp:
                        updated, stamp = time.monotonic(), current
                if time.monotonic() - updated > timeout:
                    raise TimeoutError(f'No inference token progress for {timeout:g}s; see {log}')
                try:
                    proc.wait(timeout=min(1, timeout))
                except subprocess.TimeoutExpired:
                    pass
            if proc.returncode:
                raise RuntimeError(f'Inference exited {proc.returncode}; see {log}')
        finally:
            try:
                stop_process_group(proc)
            finally:
                signal.signal(signal.SIGTERM, previous)
