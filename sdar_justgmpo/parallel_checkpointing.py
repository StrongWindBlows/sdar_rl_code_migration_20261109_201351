"""Atomic TP-sharded master/Adam checkpoints with a complete BF16 export."""
import json
import os
from pathlib import Path
import tempfile

import torch
import torch.distributed as dist


def _check_errors(error, world):
    errors = [None]*world
    dist.all_gather_object(errors, error)
    if any(errors):
        raise RuntimeError(f'TP checkpoint was not published: {errors}')


def save(path, model, optimizer, parallel, *, config, completed_steps,
         nonzero_updates, elapsed_seconds,
         export=None, wandb_run=None, run_state=None):
    from checkpointing import capture_rng, restore_rng
    rng = capture_rng()
    rngs = [None]*parallel.world
    dist.all_gather_object(rngs, rng)
    staging, error = None, None
    path = Path(path)
    if parallel.rank == 0:
        try:
            if path.exists():
                raise FileExistsError(f'Refusing to overwrite checkpoint: {path}')
            path.parent.mkdir(parents=True,exist_ok=True)
            staging = tempfile.mkdtemp(prefix=f'.{path.name}-incomplete-',dir=path.parent)
        except Exception as exc:
            error = repr(exc)
    _check_errors(error, parallel.world)
    value = [staging]
    dist.broadcast_object_list(value,src=0)
    staging = Path(value[0])
    try:
        if parallel.dp_rank == 0:
            with (staging/f'training_state-tp{parallel.tp_rank}.pt').open('wb') as file:
                torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict()},file)
                file.flush(); os.fsync(file.fileno())
        if parallel.rank == 0:
            torch.save(rngs,staging/'rng_by_rank.pt')
    except Exception as exc:
        error = repr(exc)
    _check_errors(error, parallel.world)
    try:
        if parallel.dp_rank == 0 and export is not None:
            state = parallel.full_state(model)
            if parallel.tp_rank == 0:
                export(staging,state)
            del state
    except Exception as exc:
        error = repr(exc)
    _check_errors(error, parallel.world)
    if parallel.rank == 0:
        try:
            files = {p.name:p.stat().st_size for p in staging.iterdir() if p.is_file()}
            for name in files:
                with (staging/name).open('rb') as file:
                    os.fsync(file.fileno())
            metadata = dict(format_version=2,world_size=parallel.world,tp_size=parallel.tp_size,
                dp_size=parallel.dp_size,completed_steps=completed_steps,nonzero_updates=nonzero_updates,
                next_data_index=completed_steps*config['batch_prompts'],elapsed_seconds=elapsed_seconds,
                config=config,files=files,scheduler=None,
                learning_rate_policy='constant',wandb_parent=wandb_run,run_state=run_state)
            with (staging/'resume.json').open('w') as file:
                json.dump(metadata,file,indent=2)
                file.flush(); os.fsync(file.fileno())
            fd = os.open(staging,os.O_RDONLY|os.O_DIRECTORY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
            staging.rename(path)
            fd = os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        except Exception as exc:
            error = repr(exc)
    restore_rng(rng)
    _check_errors(error,parallel.world)


def restore(path,model,optimizer,parallel):
    from checkpointing import read_metadata
    metadata = read_metadata(path)
    if (metadata['tp_size'],metadata['dp_size']) != (parallel.tp_size,parallel.dp_size):
        raise ValueError('TP/DP layout must match the checkpoint')
    state = torch.load(Path(path)/f'training_state-tp{parallel.tp_rank}.pt',
                       map_location='cpu',weights_only=False,mmap=True)
    model.load_state_dict(state['model'],strict=True)
    optimizer.load_state_dict(state['optimizer'])
    return torch.load(Path(path)/'rng_by_rank.pt',map_location='cpu',weights_only=False)[parallel.rank]
