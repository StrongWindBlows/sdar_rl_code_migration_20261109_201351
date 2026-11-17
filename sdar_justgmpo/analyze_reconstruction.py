"""Measure sequential vs block-full-mask likelihoods on saved rollout tokens."""
import argparse
import json
import math
from pathlib import Path
import random
from statistics import mean, variance
import time

import torch

from algorithm import Response, response_logps
from likelihood import SDARPolicy, reconstruction_phases
from vendor.sdar.configuration_sdar import SDARConfig


def read_rollouts(paths, step):
    for path in paths:
        with Path(path).open() as file:
            for line_index, line in enumerate(file):
                row = json.loads(line)
                if row['step'] != step:
                    continue
                for i, tokens in enumerate(row['responses']):
                    source = dict(source=str(path), line=line_index + 1, step=step, sample=i)
                    yield source, Response(row['prompt_ids'], tokens, row['rewards'][i],
                                           row['finished'][i], row['advantages'][i])


@torch.no_grad()
def score_response(model, response, temperature=1.0, phase_batch_size=4, logit_chunk_size=128):
    device = next(model.parameters()).device
    arrays = [torch.empty(len(response.tokens), device=device) for _ in range(4)]
    seq, full, seq_entropy, full_entropy = arrays
    forwards = 0
    for phase in reconstruction_phases(model, response, temperature, phase_batch_size, logit_chunk_size):
        seq[phase.offsets] = phase.logps
        seq_entropy[phase.offsets] = phase.entropy
        full[phase.full_mask_offsets] = phase.full_mask_logps
        full_entropy[phase.full_mask_offsets] = phase.full_mask_entropy
        forwards += 1
    arrays = [a.double().cpu() for a in arrays]
    seq, full, seq_entropy, full_entropy = arrays
    gap = seq - full
    p, size = len(response.prompt), model.config.block_size
    blocks = (p + len(response.tokens) - 1) // size - p // size + 1
    return dict(tokens=len(response.tokens), reward=response.reward, finished=response.finished,
        advantage=response.advantage, sequential_logp=seq.sum().item(),
        full_mask_logp=full.sum().item(), logp_gap=gap.sum().item(),
        gap_per_token=gap.mean().item(), sequential_entropy=seq_entropy.mean().item(),
        full_mask_entropy=full_entropy.mean().item(), backbone_forwards=forwards,
        sequential_projected_tokens=len(seq), projected_tokens=2 * len(seq) - blocks,
        token_logps=dict(sequential=seq.tolist(), full_mask=full.tolist(),
                        sequential_entropy=seq_entropy.tolist(), full_mask_entropy=full_entropy.tolist()))


def summarize(records):
    def group(rows):
        if not rows:
            return dict(responses=0)
        gaps = [r['logp_gap'] for r in rows]
        return dict(responses=len(rows), tokens=sum(r['tokens'] for r in rows),
            mean_logp_gap=mean(gaps), logp_gap_variance=variance(gaps) if len(gaps) > 1 else None,
            mean_gap_per_token=mean(r['gap_per_token'] for r in rows),
            token_weighted_gap=sum(gaps) / sum(r['tokens'] for r in rows),
            negative_gap_fraction=mean(g < 0 for g in gaps))

    def correlation(key):
        x, y = [r[key] for r in records], [r['gap_per_token'] for r in records]
        mx, my = mean(x), mean(y)
        dx, dy = [a - mx for a in x], [b - my for b in y]
        denominator = math.sqrt(sum(a*a for a in dx) * sum(b*b for b in dy))
        return sum(a*b for a, b in zip(dx, dy)) / denominator if denominator else None

    return dict(all=group(records), correct=group([r for r in records if r['reward'] == 1]),
        incorrect=group([r for r in records if r['reward'] != 1]),
        gap_per_token_reward_correlation=correlation('reward'),
        gap_per_token_length_correlation=correlation('tokens'))


def measure(function):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    result = function()
    torch.cuda.synchronize()
    return result, dict(seconds=time.perf_counter() - start,
        peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True, help='Weights used to sample this round')
    parser.add_argument('--rollouts', type=Path, nargs='+', required=True)
    parser.add_argument('--step', type=int, required=True, help='Rollout round; sampled before its update')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=32, help='Uniformly sampled responses')
    parser.add_argument('--seed', type=int, default=113)
    parser.add_argument('--block-size', type=int, default=32)
    parser.add_argument('--phase-batch-size', type=int, default=4)
    parser.add_argument('--logit-chunk-size', type=int, default=128)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--benchmark', action='store_true', help='Also time ordinary sequential rescoring')
    args = parser.parse_args()
    if min(args.limit, args.phase_batch_size, args.logit_chunk_size, args.temperature) <= 0:
        parser.error('limit, batch size, chunk size and temperature must be positive')
    if args.block_size not in (1, 2, 4, 8, 16, 32, 64, 128):
        parser.error('block size must divide 128')
    samples = list(read_rollouts(args.rollouts, args.step))
    if not samples:
        parser.error('No saved responses for this step')
    samples = random.Random(args.seed).sample(samples, min(args.limit, len(samples)))
    args.output.mkdir(parents=True, exist_ok=False)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(rollouts=[str(p) for p in args.rollouts], device=torch.cuda.get_device_name())
    config['note'] = ('Saved trajectories rescored under the supplied BF16 export. The likelihood gap is not '
                      'an exact KL for confidence-based decoding. Timings measure inference rescoring, not training.')
    (args.output/'config.json').write_text(json.dumps(config, indent=2) + '\n')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    cfg = SDARConfig.from_pretrained(args.model)
    cfg.block_size = args.block_size
    # SDAR selects packed Flex Attention by module mode, even under no_grad.
    model = SDARPolicy.from_pretrained(args.model, config=cfg,
        torch_dtype=torch.bfloat16).cuda().train()
    from transformers import GenerationConfig
    from inference_files import generation_config_path
    eos = GenerationConfig.from_pretrained(generation_config_path(args.model).parent).eos_token_id
    eos = {eos} if isinstance(eos, int) else set(eos)
    for _, response in samples:
        end = next((i + 1 for i, token in enumerate(response.tokens) if token in eos), len(response.tokens))
        response.tokens = response.tokens[:end]

    def full(response):
        return score_response(model, response, args.temperature, args.phase_batch_size, args.logit_chunk_size)

    def baseline(response):
        return response_logps(model, response, args.temperature, args.phase_batch_size)

    # Warm both paths on each selected shape before comparing their elapsed times.
    records = []
    with (args.output/'responses.jsonl').open('w') as file:
        for source, response in samples:
            if args.benchmark:
                baseline(response)
                full(response)
                (reference, _), sequential_cost = measure(lambda: baseline(response))
            result, cost = measure(lambda: full(response))
            record = source | result | dict(cost=cost)
            if args.benchmark:
                record['sequential_cost'] = sequential_cost
                record['sequential_logp_max_error'] = max(abs(a-b) for a, b in zip(
                    reference.cpu().tolist(), result['token_logps']['sequential']))
            file.write(json.dumps(record) + '\n')
            file.flush()
            records.append(record)
            print(json.dumps({k: record[k] for k in ('step', 'sample', 'tokens', 'logp_gap', 'cost')}), flush=True)
    (args.output/'summary.json').write_text(json.dumps(summarize(records), indent=2) + '\n')


if __name__ == '__main__':
    main()
