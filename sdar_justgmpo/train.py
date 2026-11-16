"""Sample -> freeze old log-probs -> shuffle -> several GMPO updates -> resample."""
import os

# Reuse the audit trainer's one-visible-device-per-worker LMDeploy setup.
local_rank = int(os.environ.get('LOCAL_RANK', 0))
visible = os.environ.get('CUDA_VISIBLE_DEVICES', '0,1,2,3,4,5,6,7').split(',')
os.environ.setdefault('SDAR_VISIBLE_DEVICES', ','.join(visible))
os.environ.setdefault('SDAR_LOCAL_RANK', str(local_rank))
os.environ['CUDA_VISIBLE_DEVICES'] = visible[local_rank]
os.environ['LOCAL_RANK'] = '0'

import argparse
import json
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from transformers import AutoTokenizer, GenerationConfig

from algorithm import Response, group_advantages, cache_old_logps, backward_response, response_logps
from data import load_math
from likelihood import SDARPolicy
from vendor.sdar.configuration_sdar import SDARConfig
from inference_files import generation_config_path
from inference_process import inference_env, run_inference
from reward import Reward, correctness
from runtime import save_checkpoint, format_prompt
from parallel import Parallel, WORK_TIMEOUT
from checkpointing import (RUNTIME_KEYS, read_metadata, apply_resume_config, validate_resume,
                           save_full_checkpoint, restore_full_checkpoint, restore_rng)
from run_state import (run_lock, latest_checkpoint, snapshot_logs, validate_logs,
                       restore_logs, resume_wandb, sync_metrics)

ROOT = Path(__file__).resolve().parent


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', help='Local SDAR model directory (required for a new run)')
    p.add_argument('--data', help='MATH JSONL or prepared JSON (required for a new run)')
    p.add_argument('--output', default=str(ROOT/'runs/sdar4b-justgmpo-math'))
    p.add_argument('--steps', type=int, default=50, help='Total rollout rounds, not optimizer updates')
    p.add_argument('--batch-prompts', type=int, default=128, help='Global prompts per rollout round')
    p.add_argument('--group-size', type=int, default=8)
    p.add_argument('--minibatch-size', type=int, default=128, help='Global responses per optimizer update')
    p.add_argument('--ppo-epochs', type=int, default=1)
    p.add_argument('--max-tokens', type=int, default=3000)
    p.add_argument('--rollout-batch-size', type=int, default=32, help='Engine concurrent responses per replica')
    p.add_argument('--inference-cache-fraction', type=float, default=0.8,
                   help='Fraction of free GPU memory reserved for the KV cache')
    p.add_argument('--inference-cuda-graph', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--inference-prefix-caching', action=argparse.BooleanOptionalAction, default=False)
    p.add_argument('--inference-max-prefill-tokens', type=int, default=4096)
    p.add_argument('--inference-progress-timeout', type=float, default=600,
                   help='Restart inference after this many seconds without new tokens')
    p.add_argument('--inference-retries', type=int, default=1)
    p.add_argument('--train-tf32', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--block-size', type=int, choices=[1,2,4,8,16,32,64,128], default=None,
                   help='Shared learner/rollout block size; default: model config (legacy fallback 4)')
    p.add_argument('--tp-size',type=int,default=1,help='Learner and rollout tensor-parallel size')
    p.add_argument('--dp-size',type=int,default=None,help='Optional check of WORLD_SIZE / tp-size')
    p.add_argument('--objective', choices=['gmpo','grpo'], default='gmpo')
    p.add_argument('--clip', type=float, default=0.4, help='Symmetric LOG-ratio clipping radius')
    p.add_argument('--lr', type=float, default=1e-6)
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--seed', type=int, default=1234)
    p.add_argument('--save-every', type=int, default=10, help='Checkpoint interval in rollout rounds')
    p.add_argument('--eval-every', type=int, default=0, help='Evaluation interval; 0 follows save-every')
    p.add_argument('--eval-at-start', action=argparse.BooleanOptionalAction, default=False)
    p.add_argument('--eval-data', help='Evaluate this test set; omit to disable')
    p.add_argument('--eval-max-tokens', type=int, default=3000)
    p.add_argument('--eval-batch-size', type=int, default=32)
    p.add_argument('--eval-seed', type=int, default=113)
    p.add_argument('--entropy-probe-size', type=int, default=0,
                   help='Fixed baseline responses per DP replica; 0 disables the probe')
    p.add_argument('--entropy-probe-tokens', type=int, default=128)
    p.add_argument('--resume', default=None)
    p.add_argument('--wandb-project', default='sdar-justgmpo')
    p.add_argument('--wandb-name', default=None)
    return p.parse_args()


def make_responses(rows, rollout, tokenizer, reward, group_size, seed, temperature):
    prompts = [format_prompt(tokenizer, row['question']) for row in rows]
    samples = rollout.sample(prompts, group_size, seed, temperature)
    texts = [tokenizer.decode(tokens, skip_special_tokens=True) for tokens, _ in samples]
    answers = [row['ground_truth_answer'] for row in rows for _ in range(group_size)]
    rewards, timeouts = reward.score(texts, answers, [done for _, done in samples])
    advantages = group_advantages(torch.tensor(rewards).reshape(-1, group_size)).flatten().tolist()
    records = []
    for i, (sample, value, advantage) in enumerate(zip(samples, rewards, advantages)):
        tokens, finished = sample
        prompt_ids = tokenizer.encode(prompts[i//group_size], add_special_tokens=False)
        records.append(Response(prompt_ids, tokens, value, finished, advantage))
    return records, timeouts


def minibatches(responses, size, seed):
    """A single permutation per epoch; groups were normalized before slicing."""
    indices = np.random.default_rng(seed).permutation(len(responses))
    for start in range(0, len(indices), size):
        yield [responses[i] for i in indices[start:start+size]]


def evaluate_checkpoint(path, args, parallel, evaluation_output=None):
    """All ranks wait on CPU while rank 0 evaluates the saved BF16 export."""
    if not args.eval_data:
        return
    torch.cuda.empty_cache()
    if parallel.world > 1:
        dist.barrier(group=parallel.control_group)
    error = [None]
    if parallel.rank == 0:
        output = Path(evaluation_output) if evaluation_output is not None else Path(path)/'eval'
        try:
            try:
                metrics = json.loads((output/'metrics.json').read_text())
            except (FileNotFoundError, json.JSONDecodeError):
                output.mkdir(parents=True, exist_ok=True)
                env = inference_env(parallel.gpu_group())
                command = [sys.executable, str(ROOT/'eval.py'), '--model', str(path),
                    '--data', args.eval_data, '--output', str(output),
                    '--tp-size', str(args.tp_size), '--block-size', str(args.block_size),
                    '--max-tokens', str(args.eval_max_tokens),
                    '--batch-size', str(args.eval_batch_size), '--seed', str(args.eval_seed),
                    '--inference-cache-fraction', str(args.inference_cache_fraction),
                    '--inference-max-prefill-tokens', str(args.inference_max_prefill_tokens),
                    '--inference-cuda-graph' if args.inference_cuda_graph else '--no-inference-cuda-graph',
                    '--inference-prefix-caching' if args.inference_prefix_caching else '--no-inference-prefix-caching']
                print(f'Evaluating {path}; log: {output / "eval.log"}', flush=True)
                for attempt in range(args.inference_retries + 1):
                    # Retry only incomplete evaluations, with the same questions and seed.
                    for name in ('predictions.jsonl', 'metrics.json'):
                        (output/name).unlink(missing_ok=True)
                    if attempt and '--inference-cuda-graph' in command:
                        command[command.index('--inference-cuda-graph')] = '--no-inference-cuda-graph'
                    try:
                        run_inference(command, env=env, log=output/'eval.log', progress=output/'progress',
                                      timeout=args.inference_progress_timeout)
                        break
                    except (RuntimeError, TimeoutError) as exc:
                        if attempt == args.inference_retries:
                            raise
                        print(f'{exc}; retrying evaluation with CUDA graph disabled', flush=True)
                metrics = json.loads((output/'metrics.json').read_text())
            summary = {key: metrics[key] for key in
                       ('accuracy', 'correct', 'total', 'mean_tokens', 'truncated_rate', 'grading_timeouts')}
            print(json.dumps(dict(checkpoint=str(path), eval=summary)), flush=True)
        except Exception as exc:
            error[0] = f'Evaluation failed for {path}: {exc}; see {output / "eval.log"}'
    if parallel.world > 1:
        dist.broadcast_object_list(error, src=0, group=parallel.control_group)
    if error[0]:
        raise RuntimeError(error[0])


def probe_entropy(model, responses, temperature, parallel):
    values = [response_logps(model, response, temperature)[1] for response in responses]
    entropy, tokens, count = parallel.sum_metrics([
        sum(values), sum(len(r.tokens) for r in responses), len(responses)])
    return {'probe/entropy': entropy/tokens, 'probe/tokens': tokens, 'probe/responses': count}


def train(args):
    output = Path(args.output).resolve()
    if args.resume == 'auto':
        args.resume = str(latest_checkpoint(output))
    metadata = read_metadata(args.resume) if args.resume else None
    if metadata:
        explicit = {x.split('=')[0][2:].replace('-', '_') for x in sys.argv[1:] if x.startswith('--')}
        apply_resume_config(args, metadata, explicit)
    if not args.model or not args.data:
        raise ValueError('New training runs require --model and --data')
    if not Path(args.model).is_dir():
        raise FileNotFoundError(args.model)
    if args.eval_data and not Path(args.eval_data).is_file():
        raise FileNotFoundError(args.eval_data)
    rows = load_math(args.data, training=True)
    rows = [rows[i] for i in np.random.default_rng(112).permutation(len(rows))]
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    assert args.tp_size > 0 and world % args.tp_size == 0
    dp_size = world // args.tp_size
    if args.dp_size is not None and args.dp_size != dp_size:
        raise ValueError('WORLD_SIZE must equal --tp-size * --dp-size')
    args.dp_size = dp_size
    assert args.batch_prompts > 0 and args.minibatch_size > 0
    assert args.batch_prompts % dp_size == args.minibatch_size % dp_size == 0
    assert (args.batch_prompts * args.group_size) % args.minibatch_size == 0
    assert args.group_size > 1 and args.steps > 0 and args.ppo_epochs > 0
    assert args.temperature > 0 and args.clip > 0 and args.max_tokens > 0 and args.save_every > 0
    assert args.eval_max_tokens > 0 and args.eval_batch_size > 0
    assert args.rollout_batch_size > 0 and 0 < args.inference_cache_fraction < 1
    assert args.inference_max_prefill_tokens > 0
    assert args.eval_every >= 0 and args.inference_progress_timeout > 0 and args.inference_retries >= 0
    assert args.entropy_probe_size >= 0 and args.entropy_probe_tokens > 0
    if metadata:
        validate_resume(metadata, world, args.tp_size)
    torch.cuda.set_device(0)
    if world > 1:
        dist.init_process_group('nccl', timeout=WORK_TIMEOUT,device_id=torch.device('cuda',0))
    parallel = Parallel(args.tp_size)
    sum_metrics = parallel.sum_metrics
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = args.train_tf32
    init_path = args.resume or args.model
    model_config = SDARConfig.from_pretrained(init_path)
    if args.block_size is not None:
        model_config.block_size = args.block_size
    args.block_size = model_config.block_size
    assert args.block_size in (1,2,4,8,16,32,64,128), 'Block size must divide the 128-token phase padding'
    updates_per_round = args.batch_prompts * args.group_size // args.minibatch_size * args.ppo_epochs
    config = vars(args) | dict(world_size=world, dataset_rows=len(rows), total_steps=args.steps,
        algorithm=args.objective.upper(), policy_updates_per_batch=updates_per_round,
        kl_beta=0.0, master_dtype='float32', rollout_order='sequential',
        top_k=0, top_p=1.0, eos_policy='include_first_eos_then_stop',
        truncated_reward=0.0, loss_normalization='mean_responses(mean_valid_tokens)',
        step_unit='rollout_round', microbatch_size_per_device=1)
    if rank == 0:
        print(f'Loaded {len(rows)} training examples from {args.data}', flush=True)
    start = metadata['completed_steps'] if metadata else 0
    nonzero_updates = metadata['nonzero_updates'] if metadata else 0
    prior_elapsed = metadata['elapsed_seconds'] if metadata else 0.0
    run = None
    if rank == 0:
        import wandb
        if metadata:
            validate_logs(output, metadata)
            run = resume_wandb(wandb, output, metadata)
            run.config.update({key: config[key] for key in RUNTIME_KEYS}, allow_val_change=True)
            (output/'config.json').write_text(json.dumps(config, indent=2))
            sync_metrics(run, metadata['run_state']['metrics'])
            if start < args.steps:
                restore_logs(output, metadata)
        else:
            output.mkdir(parents=True, exist_ok=True)
            if (output/'config.json').exists():
                raise FileExistsError('Use a new output directory or --resume auto')
            run = wandb.init(project=args.wandb_project, name=args.wandb_name or output.name,
                             config=config, dir=str(output), mode=os.environ.get('WANDB_MODE','offline'))
            (output/'config.json').write_text(json.dumps(config, indent=2))
            (output/'wandb.json').write_text(json.dumps(dict(id=run.id,url=run.url,path=run.path), indent=2))
            source = output/'source'; source.mkdir()
            for file in ROOT.glob('*.py'):
                shutil.copy2(file, source/file.name)
            shutil.copy2(ROOT/'run_train.sh', source/'run_train.sh')
            shutil.copytree(ROOT/'vendor', source/'vendor', ignore=shutil.ignore_patterns('__pycache__'))
        (output/'metrics.jsonl').touch(exist_ok=True)
    if world > 1:
        dist.barrier()

    def finish(path):
        if rank == 0:
            result = dict(completed_rounds=args.steps, optimizer_steps=args.steps*updates_per_round,
                          nonzero_updates=nonzero_updates, final_model=str(path), status='completed')
            (output/'completed.json').write_text(json.dumps(result, indent=2))
            run.summary.update(result)
            run.finish()

    if metadata:
        evaluate_checkpoint(args.resume, args, parallel)
    elif args.eval_at_start:
        evaluate_checkpoint(args.model, args, parallel, output/'eval-step0')
    if start == args.steps:
        finish(args.resume)
        if world > 1:
            dist.destroy_process_group()
        return
    assert model_config.attention_dropout == 0, 'Two-pass likelihood requires deterministic forwards'
    model = (SDARPolicy(model_config).float() if metadata else
             SDARPolicy.from_pretrained(args.model, config=model_config, torch_dtype=torch.float32))
    model = parallel.shard_model(model).cuda()
    if metadata:
        # state_dict restores tensors, not generation_config. Keep future
        # exports consistent with the config loaded by the rollout engine.
        model.generation_config = GenerationConfig.from_pretrained(
            generation_config_path(init_path).parent, local_files_only=True)
    model.train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    tokenizer = AutoTokenizer.from_pretrained(init_path, local_files_only=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9,0.999),
                                   eps=1e-8, weight_decay=0.0, foreach=False)
    rng = restore_full_checkpoint(args.resume, model, optimizer, rank,parallel=parallel) if metadata else None
    if args.tp_size > 1:
        from parallel_rollout import ParallelRollout
        rollout = ParallelRollout(parallel,tokenizer,args)
    else:
        from rollout import Rollout
        rollout = Rollout(init_path, tokenizer, args.max_tokens, args.rollout_batch_size,
                          block_size=model_config.block_size,
                          cache_fraction=args.inference_cache_fraction,cuda_graph=args.inference_cuda_graph,
                          prefix_caching=args.inference_prefix_caching,
                          max_prefill_tokens=args.inference_max_prefill_tokens)
    reward = Reward() if parallel.tp_rank == 0 else None
    # Exclude worker import/startup time from the reference's one-second grading deadline.
    if reward is not None:
        reward.pool.apply(correctness, (r'\boxed{1}', '1'))
    if rng is not None:
        restore_rng(rng)
    started = time.time()

    def checkpoint(round_number, path):
        state = snapshot_logs(output, world) if rank == 0 else None
        save_full_checkpoint(path, model, optimizer, config=config, completed_steps=round_number,
            nonzero_updates=nonzero_updates, elapsed_seconds=prior_elapsed+time.time()-started,
            export=lambda directory,state=None: save_checkpoint(model, tokenizer, directory,
                state=state,tokenizer_source=args.model),
            wandb_run=dict(id=run.id,url=run.url,path=run.path) if rank == 0 else None,
            run_state=state,parallel=parallel)
        optimizer.zero_grad(set_to_none=True)
        if rank == 0:
            sync_metrics(run, state['metrics'])

    def abort(round_number, error, full_path=None):
        path = full_path or output/f'checkpoint-{round_number}'
        if not path.exists():
            checkpoint(round_number, path)
        if rank == 0:
            (output/'failed.json').write_text(json.dumps(dict(status='stopped',
                completed_rounds=round_number, checkpoint=str(path), error=str(error)), indent=2))
            run.finish(exit_code=1)
        if reward is not None:
            reward.close()
        rollout.close()
        if world > 1:
            dist.destroy_process_group()
        raise RuntimeError(f'{error}; saved training state at {path}')

    probe = None
    eval_every = args.eval_every or args.save_every
    final_path = output/('final' if not (output/'final').exists() else f'final-step-{args.steps}')
    for step in range(start, args.steps):
        tick = time.time()
        batch = [rows[i % len(rows)] for i in range(step*args.batch_prompts, (step+1)*args.batch_prompts)]
        rollout.update_weights(model)
        payload = None
        if parallel.tp_rank == 0:
            try:
                recs,timeouts = make_responses(batch[parallel.dp_rank::dp_size], rollout, tokenizer, reward,
                    args.group_size,args.seed+step*100000+parallel.dp_rank*1000,args.temperature)
                payload = (None,recs,timeouts)
            except Exception as exc:
                payload = (repr(exc),None,None)
        error,recs,timeouts = parallel.broadcast(payload)
        errors = parallel.failures(error)
        if errors:
            abort(step, f'Rollout or grading failed: {errors}')
        rollout_seconds = time.time()-tick
        if args.entropy_probe_size and probe is None:
            probe_file = output/f'entropy-probe-dp{parallel.dp_rank}.json'
            if parallel.tp_rank == 0 and not probe_file.exists():
                probe_file.write_text(json.dumps([dict(prompt=r.prompt,
                    tokens=r.tokens[:args.entropy_probe_tokens], reward=0.0, finished=False)
                    for r in recs[::args.group_size][:args.entropy_probe_size]]))
            if world > 1:
                dist.barrier(group=parallel.control_group)
            probe = [Response(**record) for record in json.loads(probe_file.read_text())]
            if step == 0:
                baseline = probe_entropy(model, probe, args.temperature, parallel)
                if rank == 0:
                    (output/'entropy-probe-step0.json').write_text(json.dumps(baseline, indent=2))
                    print(json.dumps(dict(step=0, **baseline)), flush=True)
        if parallel.tp_rank == 0:
            with (output/f'rollouts-rank{rank}.jsonl').open('a') as file:
                for i, row in enumerate(batch[parallel.dp_rank::dp_size]):
                    group = recs[i*args.group_size:(i+1)*args.group_size]
                    file.write(json.dumps(dict(step=step+1,question=row['question'],answer=row['ground_truth_answer'],
                        prompt_ids=group[0].prompt,responses=[r.tokens for r in group],
                        finished=[r.finished for r in group],rewards=[r.reward for r in group],
                        advantages=[r.advantage for r in group]))+'\n')
        print(f'Rank {rank}, round {step+1}: sampled {len(recs)} responses; caching old log-probs', flush=True)
        cache_old_logps(model, recs, args.temperature)
        updates = []
        local_batch = args.minibatch_size // dp_size
        for epoch in range(args.ppo_epochs):
            for mb in minibatches(recs, local_batch, args.seed+step*100000+epoch*1000+parallel.dp_rank):
                optimizer.zero_grad(set_to_none=True)
                stats = [backward_response(model, r, len(mb), args.temperature, args.clip, args.objective)
                         for r in mb]
                parallel.reduce_gradients(model)
                norm = parallel.clip_grad_norm(model)
                optimizer.step()
                nonzero_updates += int(norm.item() > 0)
                keys = ['loss','entropy_sum','tokens','clipped','log_ratio_sum','log_ratio_square_sum']
                loss, entropy, tokens, clipped, logsum, log2sum = sum_metrics(
                    [sum(s[k] for s in stats) for k in keys])
                maximum = torch.tensor(max(s['log_ratio_abs_max'] for s in stats), device='cuda')
                if dp_size > 1:
                    dist.all_reduce(maximum, op=dist.ReduceOp.MAX,group=parallel.dp_group)
                entry = dict(optimizer_step=step*updates_per_round+len(updates)+1,
                    loss=loss/args.minibatch_size, grad_norm=norm.item(),
                    active_token_entropy=entropy/max(tokens,1), active_tokens=tokens,
                    clip_fraction=clipped/max(tokens,1), log_ratio_mean=logsum/max(tokens,1),
                    log_ratio_rms=(log2sum/max(tokens,1))**0.5, log_ratio_abs_max=maximum.item())
                updates.append(entry)
                if rank == 0:
                    print(json.dumps(dict(round=step+1, **entry)), flush=True)
        correct, count, length, done, active, grading_timeouts = sum_metrics([
            sum(r.reward for r in recs),len(recs),sum(len(r.tokens) for r in recs),
            sum(r.finished for r in recs),sum(r.advantage != 0 for r in recs),timeouts])
        metrics = {'step':step+1,'optimizer_steps':(step+1)*updates_per_round,
            'rollout/sample_accuracy':correct/count,'rollout/mean_length':length/count,
            'rollout/eos_fraction':done/count,'rollout/active_fraction':active/count,
            'reward/timeouts':grading_timeouts,'train/updates':updates,
            'train/grad_norm':updates[-1]['grad_norm'],'train/clip_fraction':updates[-1]['clip_fraction'],
            'time/rollout_seconds_rank0':rollout_seconds,'time/step_seconds':time.time()-tick,
            'time/total_seconds':prior_elapsed+time.time()-started,
            'system/gpu_peak_gib':torch.cuda.max_memory_allocated()/2**30}
        if probe is not None:
            metrics.update(probe_entropy(model, probe, args.temperature, parallel))
        if rank == 0:
            print(json.dumps(metrics), flush=True)
            with (output/'metrics.jsonl').open('a') as file:
                file.write(json.dumps(metrics)+'\n')
        round_number = step+1
        full_path = None
        if round_number % args.save_every == 0 or round_number == args.steps:
            full_path = final_path if round_number == args.steps else output/f'checkpoint-{round_number}'
            checkpoint(round_number, full_path)
        if args.eval_data and (round_number % eval_every == 0 or full_path is not None):
            path = full_path or output/f'inference-step-{round_number}'
            if full_path is None:
                if rank == 0 and path.exists():
                    archive = output/'resume-events'
                    archive.mkdir(exist_ok=True)
                    path.rename(archive/f'{path.name}-{time.time_ns()}')
                if args.tp_size > 1:
                    rollout.update_weights(model, path)
                else:
                    if rank == 0:
                        save_checkpoint(model, tokenizer, path, tokenizer_source=args.model)
                    if world > 1:
                        dist.barrier(group=parallel.control_group)
            try:
                evaluate_checkpoint(path, args, parallel)
            except RuntimeError as exc:
                abort(round_number, exc, full_path)
        if world > 1:
            dist.barrier()
    finish(final_path)
    if reward is not None:
        reward.close()
    rollout.close()
    if world > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    args = arguments()
    if int(os.environ.get('RANK', 0)) == 0:
        with run_lock(args.output):
            train(args)
    else:
        train(args)
