"""Complete update-boundary checkpoints for replicated data-parallel training.

Only load trusted, locally produced checkpoints: RNG/optimizer state uses pickle.
Publishing by directory rename keeps interrupted writes in hidden staging dirs.
"""
import json
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch
import torch.distributed as dist

FORMAT_VERSION = 1
TRAINING_KEYS = (
    'model', 'data', 'batch_prompts', 'group_size', 'minibatch_size', 'ppo_epochs',
    'max_tokens', 'lr', 'temperature', 'seed', 'clip', 'objective', 'block_size', 'tp_size', 'coupling_beta',
)
RUNTIME_KEYS = (
    'rollout_batch_size', 'inference_cache_fraction', 'inference_cuda_graph',
    'inference_prefix_caching', 'inference_max_prefill_tokens', 'train_tf32',
    'eval_data', 'eval_max_tokens', 'eval_batch_size', 'eval_seed',
    'eval_every', 'eval_at_start', 'inference_progress_timeout', 'inference_retries',
    'entropy_probe_size', 'entropy_probe_tokens', 'train_phase_batch_size', 'train_logit_chunk_size',
)


def capture_rng():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch_cpu': torch.get_rng_state(),
            'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch_cpu'])
    if state['torch_cuda']:
        if len(state['torch_cuda']) != torch.cuda.device_count():
            raise ValueError('Checkpoint CUDA RNG device count does not match this rank')
        torch.cuda.set_rng_state_all(state['torch_cuda'])


def read_metadata(path):
    path = Path(path)
    if path.name.startswith('.') and '-incomplete-' in path.name:
        raise ValueError(f'{path} is an unpublished checkpoint staging directory')
    marker = path/'resume.json'
    if not marker.is_file():
        raise ValueError(f'{path} is not a complete resumable checkpoint (missing resume.json). '
                         'Legacy BF16-only exports cannot restore FP32 weights or Adam state.')
    metadata = json.loads(marker.read_text())
    if metadata['format_version'] not in (FORMAT_VERSION, 2):
        raise ValueError('Unsupported checkpoint format version')
    for name, size in metadata['files'].items():
        file = path/name
        if not file.is_file() or file.stat().st_size != size:
            raise ValueError(f'Incomplete/corrupt checkpoint file: {file}')
    required = (['training_state.pt'] if metadata['format_version'] == 1 else
                ['rng_by_rank.pt']+[f'training_state-tp{i}.pt' for i in range(metadata['tp_size'])])
    if any(name not in metadata['files'] for name in required):
        raise ValueError('Missing full training state')
    return metadata


def validate_resume(metadata, world_size, tp_size):
    if metadata['world_size'] != world_size:
        raise ValueError('Resume requires the original number of training ranks')
    if metadata.get('tp_size', 1) != tp_size:
        raise ValueError('TP/DP layout must match the checkpoint')
    config = metadata['config']
    if metadata['next_data_index'] != metadata['completed_steps'] * config['batch_prompts']:
        raise ValueError('Inconsistent checkpoint data cursor')


def apply_resume_config(args, metadata, explicit_options):
    """Inherit saved settings; only training hyperparameters must stay fixed."""
    explicit_options = {key.removeprefix('no_') for key in explicit_options}
    saved = metadata['config']
    for key in TRAINING_KEYS:
        default = {'tp_size':1, 'coupling_beta':1e-4}.get(key)
        value = getattr(args, key, default)
        saved_value = saved.get(key, default)
        if key in explicit_options:
            equal = (Path(value).resolve() == Path(saved[key]).resolve()
                     if key in ('model', 'data') else value == saved_value)
            if not equal:
                raise ValueError(f'Cannot change --{key.replace("_", "-")} on strict resume '
                                 f'({value!r} != {saved_value!r})')
        setattr(args, key, saved_value)
    for key in RUNTIME_KEYS:
        if key not in explicit_options and key in saved:
            setattr(args, key, saved[key])
    if 'steps' not in explicit_options:
        args.steps = saved['steps']
    if args.steps < metadata['completed_steps']:
        raise ValueError('--steps is the target TOTAL step count and cannot be below completed_steps')


def save_full_checkpoint(path, model, optimizer, *, config, completed_steps,
                         nonzero_updates, elapsed_seconds,
                         export=None, wandb_run=None, run_state=None, parallel=None):
    """All ranks participate; rank 0 saves the common model/Adam state once.

    The caller invokes this only AFTER optimizer.step(), before the next rollout.
    Gradients need not be saved: the next update starts with zero_grad().
    """
    if parallel is not None and parallel.tp_size > 1:
        from parallel_checkpointing import save
        return save(path,model,optimizer,parallel,config=config,completed_steps=completed_steps,
                    nonzero_updates=nonzero_updates,elapsed_seconds=elapsed_seconds,
                    export=export,
                    wandb_run=wandb_run,run_state=run_state)
    distributed = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if distributed else 0
    world = dist.get_world_size() if distributed else 1
    rng = capture_rng()
    rngs = [None] * world
    if distributed:
        dist.all_gather_object(rngs, rng)
    else:
        rngs[0] = rng
    error = [None]
    if rank == 0:
        path = Path(path)
        staging = None
        try:
            if path.exists():
                raise FileExistsError(f'Refusing to overwrite checkpoint: {path}')
            path.parent.mkdir(parents=True, exist_ok=True)
            staging = Path(tempfile.mkdtemp(prefix=f'.{path.name}-incomplete-', dir=path.parent))
            with (staging/'training_state.pt').open('wb') as file:
                torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                            'rng_by_rank': rngs}, file)
                file.flush()
                os.fsync(file.fileno())
            if export is not None:
                export(staging)
            # Persist the inference export too before publishing its marker.
            for exported in staging.iterdir():
                if exported.is_file() and exported.name != 'training_state.pt':
                    with exported.open('rb') as file:
                        os.fsync(file.fileno())
            files = {p.name: p.stat().st_size for p in staging.iterdir() if p.is_file()}
            metadata = {'format_version': FORMAT_VERSION, 'world_size': world,
                        'completed_steps': completed_steps, 'nonzero_updates': nonzero_updates,
                        'next_data_index': completed_steps * config['batch_prompts'],
                        'elapsed_seconds': elapsed_seconds, 'config': config,
                        'files': files,
                        'scheduler': None, 'learning_rate_policy': 'constant',
                        'wandb_parent': wandb_run, 'run_state': run_state}
            with (staging/'resume.json').open('w') as file:
                json.dump(metadata, file, indent=2)
                file.flush()
                os.fsync(file.fileno())
            fd = os.open(staging, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
            staging.rename(path)
            fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        except Exception as exc:
            error[0] = f'{type(exc).__name__}: {exc}'
            # Leave staging for diagnosis; it is never a valid resume target.
    if distributed:
        dist.broadcast_object_list(error, src=0)
    # Checkpoint I/O (including an export callback) must not perturb RNG streams.
    restore_rng(rng)
    if error[0]:
        raise RuntimeError(f'Checkpoint was not published: {error[0]}')


def restore_full_checkpoint(path, model, optimizer, rank=0, parallel=None):
    metadata = read_metadata(path)
    if metadata['format_version'] == 2:
        if parallel is None:
            raise ValueError('TP checkpoint requires a matching Parallel context')
        from parallel_checkpointing import restore
        return restore(path,model,optimizer,parallel)
    if not 0 <= rank < metadata['world_size']:
        raise ValueError('Invalid checkpoint rank')
    state = torch.load(Path(path)/'training_state.pt', map_location='cpu',
                       weights_only=False, mmap=True)
    model.load_state_dict(state['model'], strict=True)
    optimizer.load_state_dict(state['optimizer'])
    # Return RNG for restoration AFTER engine/model initialization by the caller.
    return state['rng_by_rank'][rank]
