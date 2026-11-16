"""Real TP2/DP2 long-trajectory stress test, independent of training rewards.

Force EOS-free engine output to the requested length, use explicit nonzero
test advantages, and exercise two AdamW steps plus rollout with resident Adam
state. This verifies execution/memory, not on-policy probabilities or quality.
Run with torchrun --standalone --nproc_per_node=4.
"""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

# Match train.py's one visible GPU per learner, keeping each TP pair for rollout.
if 'RANK' in os.environ:
    local_rank = int(os.environ['LOCAL_RANK'])
    visible = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    os.environ['SDAR_VISIBLE_DEVICES'] = ','.join(visible)
    os.environ['SDAR_LOCAL_RANK'] = str(local_rank)
    os.environ['CUDA_VISIBLE_DEVICES'] = visible[local_rank]
    os.environ['LOCAL_RANK'] = '0'

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.distributed as dist
from transformers import AutoTokenizer
import algorithm
from algorithm import Response, cache_old_logps, backward_response
from likelihood import SDARPolicy, phase_inputs
from parallel import Parallel
from parallel_rollout import ParallelRollout
from runtime import format_prompt
from vendor.sdar.configuration_sdar import SDARConfig
# Import at module scope so LMDeploy spawned workers inherit the layout patch.
from rollout import Rollout, GenerationConfig


def worker(request_path):
    request = json.loads(Path(request_path).read_text())
    tokenizer = AutoTokenizer.from_pretrained(request['model'], local_files_only=True)
    engine = Rollout(request['model'], tokenizer, request['length'], 1,
                     block_size=request['block_size'], tp_size=2)
    try:
        config = GenerationConfig(max_new_tokens=request['length'], top_k=0, top_p=1.,
            temperature=1., do_sample=True, ignore_eos=False,
            bad_token_ids=sorted(engine.eos_ids), min_new_tokens=request['length'],
            random_seed=request['seed'], skip_special_tokens=False,
            include_stop_str_in_output=True)
        started = time.time()
        output = engine.pipe([request['prompt']], gen_config=[config], use_tqdm=False)[0]
        tokens = output.token_ids[:request['length']]
        assert output.finish_reason != 'error'
        assert len(tokens) == request['length'], (len(tokens), request['length'])
        assert not (set(tokens) & engine.eos_ids)
        result = dict(tokens=tokens, finish_reason=output.finish_reason,
                      engine_seconds=time.time()-started, forced_eos_suppression=True)
        Path(request['response']).write_text(json.dumps(result)+'\n')
    finally:
        engine.close()


def sample_long(rollout, prompt, length, seed, stage_name):
    parallel = rollout.parallel
    payload = None
    if parallel.tp_rank == 0:
        try:
            output = Path(rollout.args.output)
            response = output/f'{stage_name}-dp{parallel.dp_rank}.json'
            request = rollout.directory/f'{stage_name}-request-dp{parallel.dp_rank}.json'
            request.write_text(json.dumps(dict(model=str(rollout.model_path), prompt=prompt,
                length=length, block_size=rollout.args.block_size, seed=seed,
                response=str(response))))
            env = dict(os.environ)
            for key in list(env):
                if key in ('RANK','LOCAL_RANK','WORLD_SIZE','LOCAL_WORLD_SIZE','GROUP_RANK',
                           'ROLE_RANK','ROLE_WORLD_SIZE','MASTER_ADDR','MASTER_PORT') or key.startswith('TORCHELASTIC_'):
                    env.pop(key)
            env['CUDA_VISIBLE_DEVICES'] = parallel.gpu_group()
            env['SDAR_FULL_VOCAB'] = '1'
            with (output/f'{stage_name}-engine-dp{parallel.dp_rank}.log').open('w') as log:
                proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                    '--rollout-worker', str(request)], env=env, stdout=log,
                    stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    code = proc.wait(timeout=3600)
                    if code:
                        raise RuntimeError(f'Long rollout worker exited {code}')
                except BaseException:
                    try:
                        os.killpg(proc.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    proc.wait()
                    raise
            payload = (None, json.loads(response.read_text()))
        except Exception as exc:
            payload = (repr(exc), None)
    error, result = parallel.broadcast(payload)
    if error:
        raise RuntimeError(error)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--length', type=int, default=3000)
    parser.add_argument('--block-size', type=int, default=32)
    args = parser.parse_args()
    args.output = args.output.resolve()
    torch.cuda.set_device(0)
    torch.set_num_threads(4)
    torch.manual_seed(1234)
    torch.backends.cuda.matmul.allow_tf32 = False
    dist.init_process_group('nccl', timeout=timedelta(hours=2), device_id=torch.device('cuda',0))
    parallel = Parallel(2)
    assert parallel.dp_size == 2 and args.length > 128
    if parallel.rank == 0:
        args.output.mkdir(parents=True, exist_ok=False)
    dist.barrier()
    started = time.time()

    def stage(message):
        if parallel.rank == 0:
            print(f'[{time.time()-started:.1f}s] {message}', flush=True)

    def record(name, **extra):
        torch.cuda.synchronize()
        entry = dict(stage=name, rank=parallel.rank,
            elapsed_seconds=time.time()-started,
            allocated_gib=torch.cuda.memory_allocated()/2**30,
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30, **extra)
        with (args.output/f'metrics-rank{parallel.rank}.jsonl').open('a') as file:
            file.write(json.dumps(entry)+'\n')
        return entry

    config = SDARConfig.from_pretrained(args.model)
    config.block_size = args.block_size
    stage('Loading FP32 master weights and sharding TP=2 / DP=2')
    model = SDARPolicy.from_pretrained(args.model, config=config, torch_dtype=torch.float32)
    model = parallel.shard_model(model).cuda().train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6, betas=(.9,.999),
                                 eps=1e-8, weight_decay=0., foreach=False)
    record('loaded')
    rollout = ParallelRollout(parallel, tokenizer, args)
    questions = ['Explain in detail why there are infinitely many prime numbers, including several proofs.',
                 'Explain the binomial theorem in detail, with worked examples and a combinatorial proof.']
    prompt = format_prompt(tokenizer, questions[parallel.dp_rank])
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    stage(f'Exporting and generating exactly {args.length} tokens on each TP replica')
    rollout.update_weights(model)
    sampled = sample_long(rollout, prompt, args.length, 2100+parallel.dp_rank, 'initial')
    response = Response(prompt_ids, sampled['tokens'], reward=0., finished=False,
                        advantage=1. if parallel.dp_rank == 0 else -1.)
    phases = sum(1 for _ in phase_inputs(response, args.block_size, config.mask_token_id, 'cpu'))
    assert len(response.tokens) == args.length and phases == args.block_size
    record('rollout', tokens=len(response.tokens), prompt_tokens=len(prompt_ids), phases=phases)

    # Print progress through actual production phase iteration; leave production
    # code unchanged so existing checkpoint source fingerprints remain valid.
    original_phase_logps = algorithm.phase_logps
    def progress_phases(*a, **kw):
        for index, item in enumerate(original_phase_logps(*a, **kw), 1):
            yield item
            if index % 8 == 0:
                stage(f'phase {index}/{phases}; allocated={torch.cuda.memory_allocated()/2**30:.2f} GiB')
    algorithm.phase_logps = progress_phases
    stage('Caching old log-probabilities for full-length trajectories')
    cache_old_logps(model, [response])
    frozen = response.old_logps.clone()
    assert torch.isfinite(frozen).all()
    record('old_cached')
    updates = []
    probe = dict(model.named_parameters())['model.layers.0.self_attn.q_proj.weight']
    for step in (1,2):
        stage(f'Long backward + AdamW step {step}; optimizer state already exists={bool(optimizer.state)}')
        tick = time.time()
        optimizer.zero_grad(set_to_none=True)
        before = probe.detach().flatten()[:4096].clone()
        stats = backward_response(model, response, minibatch_size=1)
        assert stats['tokens'] == args.length
        parallel.reduce_gradients(model)
        norm = parallel.clip_grad_norm(model)
        assert norm.item() > 0 and torch.isfinite(norm)
        optimizer.step()
        delta = (probe.detach().flatten()[:4096]-before).abs().max().item()
        assert delta > 0
        torch.testing.assert_close(response.old_logps, frozen, atol=0, rtol=0)
        counters = {int(s['step'].item()) for s in optimizer.state.values()}
        assert counters == {step}
        entry = record(f'update_{step}', stats=stats, grad_norm=norm.item(),
            sampled_parameter_change=delta, seconds=time.time()-tick, optimizer_step=step)
        updates.append(entry)
        stage(f'Completed AdamW step {step}: grad_norm={norm.item():.6f}; peak={entry["peak_allocated_gib"]:.2f} GiB')

    # Production train.py retains gradients and moments during the next export
    # and rollout. Keep them resident here to test that memory combination too.
    stage('Generating another full-length response with Adam moments and gradients resident')
    rollout.update_weights(model)
    next_sample = sample_long(rollout, prompt, args.length, 3100+parallel.dp_rank, 'after_updates')
    assert len(next_sample['tokens']) == args.length
    final = record('complete', prompt_tokens=len(prompt_ids), response_tokens=args.length,
                   phases=phases, old_logps_unchanged=True, updates=updates,
                   next_rollout_tokens=len(next_sample['tokens']))
    all_ranks = [None]*parallel.world
    dist.all_gather_object(all_ranks, final)
    rollout.close()
    if parallel.rank == 0:
        result = dict(status='passed', model=args.model, tp_size=2, dp_size=2,
            block_size=args.block_size, tokens_per_response=args.length,
            rollout_concurrency_per_replica=1, learner_microbatch_per_replica=1,
            optimizer='AdamW FP32 master/moments, BF16 compute', optimizer_steps=2,
            actual_engine_generated_tokens=True, eos_suppressed_for_stress_test=True,
            advantages='explicit +1/-1 by DP replica; not reward-derived',
            post_update_rollout_with_adam_and_gradients_resident=True,
            production_source_changed=False, ranks=all_ranks,
            elapsed_seconds=time.time()-started)
        (args.output/'results.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps({k:v for k,v in result.items() if k!='ranks'}), flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--rollout-worker':
        worker(sys.argv[2])
    else:
        main()
