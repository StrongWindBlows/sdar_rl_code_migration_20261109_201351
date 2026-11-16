"""Greedy MATH evaluation for a local SDAR model or exported checkpoint."""
import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import time

os.environ['HF_HOME'] = os.environ.get('SDAR_CACHE_DIR',
    str(Path(__file__).resolve().parents[2]/'.cache/huggingface'))
os.environ['HF_MODULES_CACHE'] = str(Path(os.environ['HF_HOME'])/'modules')
os.environ.setdefault('NCCL_P2P_DISABLE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import torch
from transformers import AutoTokenizer

from data import load_math
from inference_files import inference_model_path
from reward import Reward, correctness
# Install the sampling adapter in spawned engine workers as well as the parent.
from rollout import Rollout
from runtime import format_prompt
from inference_process import TokenProgress


def evaluate(rows, rollout, tokenizer, reward, predictions, *, batch_size=1, seed=113, progress=None):
    correct = tokens = truncated = timeouts = 0
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start+batch_size]
        prompts = [format_prompt(tokenizer, row['question']) for row in batch]
        options = dict(do_sample=False)
        if progress is not None:
            options['progress'] = progress
        samples = rollout.sample(prompts, 1, seed+start, **options)
        texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids, _ in samples]
        scores, grading_timeouts = reward.score(texts,
            [row['ground_truth_answer'] for row in batch], [done for _, done in samples])
        timeouts += grading_timeouts
        for index, (row, (ids, done), text, score) in enumerate(zip(batch, samples, texts, scores), start):
            predictions.write(json.dumps(dict(index=index, **row, response=text,
                token_ids=ids, num_tokens=len(ids), finished=done, correct=bool(score)),
                ensure_ascii=False)+'\n')
            correct += int(score)
            tokens += len(ids)
            truncated += int(not done)
        predictions.flush()
        print(f'Evaluated {start+len(batch)}/{len(rows)}, correct={correct}', flush=True)
    return dict(total=len(rows), correct=correct, accuracy=correct/len(rows),
        mean_tokens=tokens/len(rows), truncated=truncated, truncated_rate=truncated/len(rows),
        grading_timeouts=timeouts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, help='Base model or exported checkpoint directory')
    parser.add_argument('--data', required=True, help='Test JSONL or prepared JSON; no length filtering')
    parser.add_argument('--output', required=True, help='Directory for predictions and metrics')
    parser.add_argument('--tp-size', type=int, default=2)
    parser.add_argument('--block-size', type=int, choices=[1, 2, 4, 8, 16, 32, 64, 128],
                        help='Default: model config; use 32 for the base b32 model')
    parser.add_argument('--max-tokens', type=int, default=3000)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--inference-cache-fraction', type=float, default=0.8,
                        help='Fraction of free GPU memory reserved for the KV cache')
    parser.add_argument('--inference-cuda-graph', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--inference-prefix-caching', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--inference-max-prefill-tokens', type=int, default=4096)
    parser.add_argument('--seed', type=int, default=113)
    parser.add_argument('--limit', type=int, help='Evaluate only the first N rows for a smoke test')
    args = parser.parse_args()
    if min(args.tp_size, args.max_tokens, args.batch_size, args.inference_max_prefill_tokens) <= 0 or (args.limit is not None and args.limit <= 0):
        parser.error('sizes and --limit must be positive')
    if not 0 < args.inference_cache_fraction < 1:
        parser.error('--inference-cache-fraction must be between 0 and 1')
    rows = load_math(args.data)
    if args.limit is not None:
        rows = rows[:args.limit]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output/'predictions.jsonl').exists() or (output/'metrics.json').exists():
        raise FileExistsError(f'Evaluation results already exist in {output}')
    config = vars(args) | dict(do_sample=False, decoding_order='sequential')
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    with ExitStack() as stack:
        model_path = stack.enter_context(inference_model_path(args.model))
        config['block_size'] = args.block_size or json.loads((model_path/'config.json').read_text()).get('block_size')
        if config['block_size'] is None:
            parser.error('Model config has no block_size; pass --block-size (32 for the b32 base model)')
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        rollout = Rollout(model_path, tokenizer, args.max_tokens, args.batch_size,
                          block_size=config['block_size'], tp_size=args.tp_size,
                          cache_fraction=args.inference_cache_fraction,cuda_graph=args.inference_cuda_graph,
                          prefix_caching=args.inference_prefix_caching,
                          max_prefill_tokens=args.inference_max_prefill_tokens)
        stack.callback(rollout.close)
        reward = Reward()
        stack.callback(reward.close)
        reward.pool.apply(correctness, (r'\boxed{1}', '1'))
        (output/'config.json').write_text(json.dumps(config, indent=2)+'\n')
        started = time.monotonic()
        with (output/'predictions.jsonl').open('x') as predictions:
            metrics = evaluate(rows, rollout, tokenizer, reward, predictions,
                               batch_size=args.batch_size, seed=args.seed,
                               progress=TokenProgress(output/'progress'))
        metrics['elapsed_seconds'] = time.monotonic()-started
    (output/'metrics.json').write_text(json.dumps(config | metrics, indent=2)+'\n')
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == '__main__':
    main()
