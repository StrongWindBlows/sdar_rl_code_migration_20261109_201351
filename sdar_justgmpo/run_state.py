"""Same-directory recovery and checkpoint-committed W&B history."""
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import shutil
import tempfile

from checkpointing import read_metadata, capture_rng, restore_rng


@contextmanager
def run_lock(output):
    # Outside the run directory so an unused run remains empty. Keep the inode:
    # unlinking an advisory lock can allow two processes to lock different files.
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with (output.parent / ('.' + output.name + '.training.lock')).open('a') as file:
        try:
            fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(f'Another trainer holds the run lock: {output}') from exc
        try:
            yield
        finally:
            fcntl.flock(file, fcntl.LOCK_UN)


def latest_checkpoint(output):
    output = Path(output)
    candidates = []
    for path in output.iterdir() if output.exists() else []:
        if not path.is_dir() or not (path.name.startswith('checkpoint-') or
                                    path.name == 'final' or path.name.startswith('final-step-')):
            continue
        # A published marker with damaged state is an error, not a reason to
        # silently rewind to an older model whose W&B history is already ahead.
        if (path/'resume.json').exists():
            meta = read_metadata(path)
            candidates.append((meta['completed_steps'], path.name.startswith('final'), path))
    if not candidates:
        raise ValueError(f'{output}: no complete resumable checkpoint. Legacy BF16-only '
                         'exports cannot restore FP32 weights or Adam state.')
    return max(candidates)[2]


def snapshot_logs(output, world):
    output = Path(output)
    names = ['metrics.jsonl'] + [f'rollouts-rank{rank}.jsonl' for rank in range(world)]
    offsets = {}
    for name in names:
        path = output/name
        if path.exists():
            with path.open('rb') as file:
                os.fsync(file.fileno())
                offsets[name] = os.fstat(file.fileno()).st_size
        else:
            offsets[name] = 0
    metrics = [json.loads(line) for line in (output/'metrics.jsonl').read_text().splitlines()]
    return {'protocol': 'checkpoint_committed_v1', 'offsets': offsets, 'metrics': metrics}


def validate_logs(output, metadata):
    state = metadata.get('run_state')
    if not state or state['protocol'] != 'checkpoint_committed_v1':
        raise ValueError('Checkpoint lacks the same-run W&B/log recovery protocol')
    if [m['step'] for m in state['metrics']] != list(range(1, metadata['completed_steps'] + 1)):
        raise ValueError('Checkpoint metric history is not contiguous')
    for name, offset in state['offsets'].items():
        path = Path(output)/name
        if (path.stat().st_size if path.exists() else 0) < offset:
            raise ValueError(f'Local journal is shorter than its checkpoint: {path}')
    with (Path(output)/'metrics.jsonl').open('rb') as file:
        prefix = file.read(state['offsets']['metrics.jsonl'])
    if [json.loads(line) for line in prefix.splitlines()] != state['metrics']:
        raise ValueError('Local metric history differs from its checkpoint')
    return state


def restore_logs(output, metadata):
    """Archive uncommitted bytes before truncation; safe to retry after a crash."""
    output = Path(output)
    state = validate_logs(output, metadata)
    events = output/'resume-events'
    events.mkdir(exist_ok=True)
    event = Path(tempfile.mkdtemp(prefix=f'from-{metadata["completed_steps"]}-', dir=events))
    for name, offset in state['offsets'].items():
        path = output/name
        if path.exists() and path.stat().st_size > offset:
            with path.open('rb') as src, (event/(name+'.uncommitted')).open('wb') as dst:
                src.seek(offset)
                shutil.copyfileobj(src, dst)
                dst.flush(); os.fsync(dst.fileno())
            with path.open('r+b') as file:
                file.truncate(offset)
                file.flush(); os.fsync(file.fileno())
    for name in ('completed.json', 'failed.json', 'final_train_probe.json'):
        if (output/name).exists():
            shutil.copy2(output/name, event/name)
            (output/name).unlink()
    (event/'resume.json').write_text(json.dumps({'completed_steps': metadata['completed_steps'],
                                               'wandb': metadata['wandb_parent']}, indent=2))


def resume_wandb(wandb, output, metadata):
    identity = metadata['wandb_parent']
    local = json.loads((Path(output)/'wandb.json').read_text())
    if not identity or any(identity[k] != local[k] for k in ('id', 'path')):
        raise ValueError('Checkpoint and local W&B identity do not match')
    parts = identity['path'].split('/')
    if len(parts) == 2:
        entity, (project, run_id) = None, parts
    elif len(parts) == 3:
        entity, project, run_id = parts
    else:
        raise ValueError('Invalid W&B run path')
    if run_id != identity['id']:
        raise ValueError('Invalid W&B identity')
    if os.environ.get('WANDB_MODE', 'offline') == 'offline':
        # The local checkpoint/journals restore training. Offline W&B creates a
        # new local segment; it does not claim a server-side resumed session.
        return wandb.init(entity=entity, project=project, id=run_id, dir=str(output),
                          mode='offline', config=metadata['config'], save_code=False)
    if not entity:
        raise ValueError('This checkpoint has no W&B entity; resume with WANDB_MODE=offline')
    run = wandb.init(entity=entity, project=project, id=run_id, resume='must',
                     dir=str(output), mode=os.environ.get('WANDB_MODE', 'online'), save_code=False)
    try:
        if run.id != run_id or not run.resumed:
            raise ValueError('W&B did not resume the original run')
        if run.step > metadata['completed_steps'] + 1:
            raise ValueError('W&B history is ahead of the checkpoint; refusing divergent history')
    except Exception:
        run.finish(exit_code=1)
        raise
    return run


def sync_metrics(run, metrics):
    # run.step is the next SDK history step, including on resume. Explicit
    # monotonically increasing steps make a partially uploaded batch replayable.
    rng = capture_rng()
    try:
        for metric in metrics:
            if metric['step'] >= run.step:
                run.log(metric, step=metric['step'])
    finally:
        restore_rng(rng)
