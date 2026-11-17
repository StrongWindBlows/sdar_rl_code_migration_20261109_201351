"""Compare greedy sequential, fixed-group and confidence-based decoding."""
import argparse
from contextlib import ExitStack
import csv
import gzip
import json
import os
from pathlib import Path
import sys
import time


def decoding_settings(block_size, parallel_tokens, thresholds):
    settings = [dict(name='sequential', strategy='sequential', steps=block_size)]
    for size in dict.fromkeys(parallel_tokens):
        if size < 1 or block_size % size:
            raise ValueError('parallel token counts must divide the block size')
        if size != 1:
            settings.append(dict(name=f'fixed-{size}', strategy='sequential', steps=block_size // size))
    for threshold in dict.fromkeys(thresholds):
        if not 0 < threshold <= 1:
            raise ValueError('confidence thresholds must be in (0, 1]')
        settings.append(dict(name=f'dynamic-{threshold}', strategy='low_confidence_dynamic',
                             steps=block_size, threshold=threshold))
    return settings


def counted_result(result, prompt_length):
    """Check actual forwards and committed IDs before publishing their counts."""
    result = dict(result)
    events = result.pop('events')
    commits = {e['block_start'] + p: token for e in events
               for p, token in zip(e['positions'], e['tokens'])}
    if [commits[prompt_length + i] for i in range(len(result['token_ids']))] != result['token_ids']:
        raise ValueError('Committed token IDs do not match the returned response')
    if result['forward_calls'] != len(events) or len(events) != sum(result['forward_types'].values()):
        raise ValueError('Forward counters disagree')
    result.pop('state')
    result.pop('eligible_blocks')
    result['denoise_forwards'] = sum(e['phase'] == 'denoise' for e in events)
    return result, events


def summarize(records, seconds):
    tokens = sum(len(r['token_ids']) for r in records)
    counted = all(r['forward_calls'] is not None for r in records)
    forwards = sum(r['forward_calls'] for r in records) if counted else None
    denoise = sum(r['denoise_forwards'] for r in records) if counted else None
    return dict(total=len(records), correct=sum(r['reward'] for r in records),
        accuracy=sum(r['reward'] for r in records) / len(records),
        tokens=tokens, mean_tokens=tokens / len(records),
        truncated=sum(not r['finished'] for r in records),
        grading_timeouts=sum(r['grading'] == 'timeout' for r in records),
        forward_calls=forwards, denoise_forwards=denoise,
        tpf_denoise=tokens / denoise if denoise else None,
        tpf_all=tokens / forwards if forwards else None,
        generation_seconds=seconds, tokens_per_second=tokens / seconds,
        timing_scope='counted generation with event hooks' if counted else 'native rollout generation')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--parallel-tokens', type=int, nargs='*', default=[2, 4, 8])
    parser.add_argument('--thresholds', type=float, nargs='*', default=[0.9, 0.95, 0.99])
    parser.add_argument('--mode', choices=['count', 'timing'], default='count',
                        help='count audits actual forwards; timing uses native rollout without event hooks')
    parser.add_argument('--block-size', type=int, default=32)
    parser.add_argument('--max-tokens', type=int, default=3000)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cache-fraction', type=float, default=0.15)
    parser.add_argument('--cuda-graph', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--seed', type=int, default=113)
    parser.add_argument('--start-index', type=int, default=0)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--save-events', action='store_true')
    args = parser.parse_args()
    if min(args.block_size, args.max_tokens, args.batch_size) <= 0 or not 0 < args.cache_fraction < 1:
        parser.error('sizes must be positive and cache fraction must be in (0, 1)')
    if args.start_index < 0 or (args.limit is not None and args.limit <= 0):
        parser.error('start index must be nonnegative and limit must be positive')
    if args.save_events and args.mode != 'count':
        parser.error('--save-events requires --mode count')
    try:
        settings = decoding_settings(args.block_size, args.parallel_tokens, args.thresholds)
    except ValueError as error:
        parser.error(str(error))

    os.environ['HF_HOME'] = os.environ.get('SDAR_CACHE_DIR',
        str(Path(__file__).resolve().parents[2] / '.cache/huggingface'))
    os.environ['HF_MODULES_CACHE'] = str(Path(os.environ['HF_HOME']) / 'modules')
    os.environ['SDAR_FULL_VOCAB'] = '1'
    os.environ.setdefault('NCCL_P2P_DISABLE', '1')
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    from data import load_math
    from inference_files import inference_model_path
    # GA owns the audited forward counter; import it only for this entry point.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'ga'))
    from engine import Backend
    from protocol import Grader
    from rollout import Rollout
    from lmdeploy.pytorch.config import UnmaskingStrategy
    import torch

    rows = load_math(args.data)
    stop = args.start_index + args.limit if args.limit is not None else len(rows)
    rows = rows[args.start_index:stop]
    if not rows:
        parser.error('No questions selected')
    args.output.mkdir(parents=True, exist_ok=False)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(tp_size=1, do_sample=False, settings=settings,
        device=torch.cuda.get_device_name(), note='Compare speed only within the same mode and hardware settings.')
    (args.output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    with ExitStack() as stack:
        path = stack.enter_context(inference_model_path(args.model))
        if args.mode == 'count':
            backend = Backend(dict(model=str(path), block_length=args.block_size,
                max_new_tokens=args.max_tokens, batch_size=args.batch_size,
                cache_fraction=args.cache_fraction, eager=not args.cuda_graph,
                seed=args.seed, trace=False, top_k_record=1), settings[0])
            tokenizer = backend.tokenizer
            prompts = [backend.prompt(row['question']) for row in rows]
        else:
            from transformers import AutoTokenizer
            from runtime import format_prompt
            tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
            backend = Rollout(path, tokenizer, args.max_tokens, args.batch_size,
                block_size=args.block_size, tp_size=1, cache_fraction=args.cache_fraction,
                cuda_graph=args.cuda_graph)
            prompts = [format_prompt(tokenizer, row['question']) for row in rows]
        stack.callback(backend.close)
        grader = Grader()
        stack.callback(grader.close)
        comparison = []
        for setting in settings:
            directory = args.output / setting['name']
            directory.mkdir()
            if args.mode == 'count':
                backend.set_setting(setting)
                backend.generate([dict(id='warmup', prompt_ids=prompts[0], max_new_tokens=min(32, args.max_tokens))])
            else:
                for strategy in (backend.pipe.engine.model_agent_strategy,
                                 backend.pipe.engine.executor.model_agent.agent_strategy):
                    cfg = strategy.unmasking_processor.dllm_config
                    cfg.unmasking_strategy = UnmaskingStrategy.from_str(setting['strategy'])
                    cfg.denoising_steps = setting['steps']
                    cfg.confidence_threshold = setting.get('threshold', 0.9)
                backend.max_tokens = min(32, args.max_tokens)
                backend.sample(prompts[:1], 1, args.seed, do_sample=False)
                backend.max_tokens = args.max_tokens
            records, seconds = [], 0.0
            events_file = stack.enter_context(gzip.open(directory / 'events.jsonl.gz', 'wt')) if args.save_events else None
            with (directory / 'predictions.jsonl').open('w') as file:
                for start in range(0, len(rows), args.batch_size):
                    batch = rows[start:start + args.batch_size]
                    batch_prompts = prompts[start:start + len(batch)]
                    torch.cuda.synchronize()
                    begin = time.perf_counter()
                    if args.mode == 'count':
                        outputs = backend.generate([dict(id=str(args.start_index + start + i), prompt_ids=p)
                                                    for i, p in enumerate(batch_prompts)])
                    else:
                        outputs = backend.sample(batch_prompts, 1, args.seed + start, do_sample=False)
                    torch.cuda.synchronize()
                    seconds += time.perf_counter() - begin
                    for i, (row, result) in enumerate(zip(batch, outputs)):
                        if args.mode == 'count':
                            result, events = counted_result(result, len(batch_prompts[i]))
                            if events_file:
                                for event in events:
                                    events_file.write(json.dumps(event) + '\n')
                        else:
                            ids, finished = result
                            result = dict(token_ids=ids, finished=finished,
                                text=tokenizer.decode(ids, skip_special_tokens=True),
                                forward_calls=None, denoise_forwards=None)
                        grading = grader.score(result['text'], row['ground_truth_answer'], result['finished'])
                        record = dict(index=args.start_index + start + i, **row, **result, **grading)
                        file.write(json.dumps(record, ensure_ascii=False) + '\n')
                        records.append(record)
                    file.flush()
                    print(f"{setting['name']}: {len(records)}/{len(rows)}", flush=True)
            metrics = setting | summarize(records, seconds)
            (directory / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
            comparison.append(metrics)
            print(json.dumps(metrics), flush=True)
        (args.output / 'summary.json').write_text(json.dumps(comparison, indent=2) + '\n')
        fields = ['name', 'accuracy', 'tpf_denoise', 'tpf_all', 'forward_calls',
                  'generation_seconds', 'tokens_per_second', 'mean_tokens', 'truncated']
        with (args.output / 'comparison.csv').open('w') as file:
            writer = csv.DictWriter(file, fieldnames=fields, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(comparison)


if __name__ == '__main__':
    main()
