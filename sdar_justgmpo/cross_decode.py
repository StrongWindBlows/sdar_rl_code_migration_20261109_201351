"""Swap the first divergent block between native greedy SDAR decoders."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
from datetime import datetime
from itertools import zip_longest
import json
import multiprocessing as mp
import os
from pathlib import Path
import random
import subprocess
import sys

from branch_rl import ROOT, RUNS, configure_environment, read, write
from data import load_math
from eval_decoding import counted_result, decoding_settings, summarize as decoding_summary

GA = ROOT / 'ga'
MODES = ('sequential', 'dynamic')
DIRECTIONS = (('sequential', 'dynamic'), ('dynamic', 'sequential'))


def first_divergent_block(sequential, dynamic, prompt_length, block_size):
    for offset, (left, right) in enumerate(zip_longest(sequential, dynamic)):
        if left != right:
            return (prompt_length + offset) // block_size * block_size
    return None


def generate_path(backend, grader, job, answer):
    raw, = backend.generate([job])
    result, events = counted_result(raw, len(job['prompt_ids']))
    grade = grader.score(result['text'], answer, result['finished'])
    record = {key: result[key] for key in ('token_ids', 'finished', 'forward_calls', 'denoise_forwards')}
    return dict(record, reward=grade['reward'], grading=grade['grading'],
                seconds=backend.last_seconds), events


def evaluate_question(backend, grader, row, settings, block_size):
    paths, traces = {}, {}
    prompt = backend.prompt(row['question'])
    answer = row['ground_truth_answer']
    for mode in MODES:
        backend.set_setting(settings[mode])
        job = dict(id=f"{row['index']}:{mode}", prompt_ids=prompt)
        paths[mode], traces[mode] = generate_path(backend, grader, job, answer)

    sequential, dynamic = (paths[mode]['token_ids'] for mode in MODES)
    block_start = first_divergent_block(sequential, dynamic, len(prompt), block_size)
    if block_start is not None:
        first = max(0, block_start - len(prompt))
        end = block_start + block_size - len(prompt)
        for donor, receiver in DIRECTIONS:
            name = f'{donor}_to_{receiver}'
            backend.set_setting(settings[receiver])
            job = dict(id=f"{row['index']}:{name}", prompt_ids=prompt,
                       block_override=(block_start, settings[donor]))
            crossed, traces[name] = generate_path(backend, grader, job, answer)
            tokens = crossed['token_ids']
            if tokens[:first] != paths[receiver]['token_ids'][:first]:
                raise RuntimeError(f"{job['id']} changed the prefix before block {block_start}")
            crossed['matches_donor'] = tokens[first:end] == paths[donor]['token_ids'][first:end]
            paths[name] = crossed

        for name, path in paths.items():
            events = [event for event in traces[name] if event['block_start'] == block_start]
            path['block_forwards'] = len(events)
            path['block_denoise_forwards'] = sum(event['phase'] == 'denoise' for event in events)
    return dict(index=row['index'], prompt_length=len(prompt), block_start=block_start, paths=paths)


def worker(config, gpu_index, gpu_uuid, rows, output):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_uuid
    sys.path.insert(0, str(GA))
    from engine import Backend
    from inference_files import inference_model_path
    from protocol import Grader

    block = config['block_size']
    settings = dict(zip(MODES, decoding_settings(block, [], [config['threshold']])))
    with ExitStack() as stack, output.open('a') as file:
        model = stack.enter_context(inference_model_path(ROOT / config['model']))
        backend = Backend(dict(model=str(model), block_length=block, max_new_tokens=config['max_tokens'],
                               batch_size=1, cache_fraction=config['cache_fraction'], eager=True,
                               seed=config['seed'], trace=False, top_k_record=1), settings['sequential'])
        stack.callback(backend.close)
        grader = Grader()
        stack.callback(grader.close)
        print(f'GPU {gpu_index} ready: {len(rows)} questions', flush=True)
        for i, row in enumerate(rows, 1):
            result = evaluate_question(backend, grader, row, settings, block)
            file.write(json.dumps(result, separators=(',', ':')) + '\n')
            file.flush()
            if i == 1 or i % 8 == 0 or i == len(rows):
                print(f'GPU {gpu_index}: {i}/{len(rows)} questions', flush=True)


def path_metrics(paths):
    return dict(decoding_summary(paths, sum(path['seconds'] for path in paths)),
                block_forwards=sum(path.get('block_forwards', 0) for path in paths),
                block_denoise_forwards=sum(path.get('block_denoise_forwards', 0) for path in paths))


def summarize(rows, seed):
    sys.path.insert(0, str(GA))
    from protocol import paired_stats

    divergent = [row for row in rows if row['block_start'] is not None]
    groups = {'all': rows,
              'full_block': [row for row in divergent if row['block_start'] >= row['prompt_length']],
              'mixed_prompt_block': [row for row in divergent if row['block_start'] < row['prompt_length']]}
    results = {}
    for group, selected in groups.items():
        if not selected:
            continue
        comparison = {}
        for donor, receiver in DIRECTIONS:
            name = f'{donor}_to_{receiver}'
            baseline = {row['index']: row['paths'][receiver] for row in selected}
            crossed = {row['index']: row['paths'].get(name, row['paths'][receiver]) for row in selected}
            paired = paired_stats({i: p['reward'] for i, p in baseline.items()},
                                  {i: p['reward'] for i, p in crossed.items()}, seed)
            comparison[name] = dict(paired, baseline=path_metrics(list(baseline.values())),
                                    crossed=path_metrics(list(crossed.values())),
                                    donor_mismatches=sum(p.get('matches_donor') is False for p in crossed.values()))
        results[group] = comparison
    return dict(questions=len(rows), divergent_questions=len(divergent), groups=results)


def read_shards(shards):
    return [json.loads(line) for shard in shards if shard.exists() for line in shard.read_text().splitlines()]


def run(config, output):
    output.mkdir(parents=True, exist_ok=True)
    config_file = output / 'config.json'
    if config_file.exists() and read(config_file) != config:
        raise ValueError('Use the same configuration to resume this output directory')
    write(config_file, config)
    questions = output / 'questions.json'
    if not questions.exists():
        rows = [dict(row, index=i) for i, row in enumerate(load_math(ROOT / config['data'], training=True))]
        write(questions, random.Random(config['seed']).sample(rows, config['questions']))
    rows = read(questions)
    configure_environment()
    devices = dict(line.split(', ', 1) for line in subprocess.check_output(
        ['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader,nounits'], text=True).splitlines())
    gpus = config['gpus']
    shards = [output / f'gpu-{gpu}.jsonl' for gpu in gpus]
    completed = {row['index'] for row in read_shards(shards)}
    remaining = [row for row in rows if row['index'] not in completed]
    workers = min(len(gpus), len(remaining))
    if workers:
        print(f'{len(remaining)}/{len(rows)} questions remaining on {workers} GPUs: {output}', flush=True)
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn'),
                                 max_tasks_per_child=1) as pool:
            jobs = [pool.submit(worker, config, gpu, devices[str(gpu)], remaining[i::workers], shards[i])
                    for i, gpu in enumerate(gpus[:workers])]
            for job in jobs:
                job.result()
    records = sorted(read_shards(shards), key=lambda row: row['index'])
    write(output / 'summary.json', summarize(records, config['seed']))
    print(f"Completed: {output / 'summary.json'}", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    run(read(args.config), args.output or RUNS / f'cross-decode-{datetime.now():%Y%m%d-%H%M%S}')
