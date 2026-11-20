"""One RLOO step and full-trajectory policy search under native Dynamic decoding."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from math import ceil, exp, log, sqrt
import multiprocessing as mp
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory, mkdtemp

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from branch_rl import (ROOT, RUNS, compare_dynamic, configure_environment, dynamic_metrics, generate,
                       make_splits, read, write)
from branch_rollout import worker
from inference_process import inference_env
from run_state import run_lock


def candidate_tensor(name, base, delta, step, logit_scale):
    weight = base.float().add(delta, alpha=step)
    if name == 'lm_head.weight':
        weight.mul_(exp(logit_scale))
    return weight.bfloat16()


def export_candidate(reference, delta, coordinates, destination):
    if coordinates == (0.0, 0.0):
        return reference
    if destination.is_dir():
        return destination
    with TemporaryDirectory(dir=destination.parent, prefix='.candidate-') as directory:
        path = Path(directory) / 'model'
        path.mkdir()
        for file in reference.iterdir():
            if file.is_file() and not file.name.startswith(('training_state', 'rng_by_rank', 'resume.json')):
                if file.suffix == '.safetensors':
                    with safe_open(file, framework='pt', device='cpu') as base:
                        state = {name: candidate_tensor(name, base.get_tensor(name), delta[name], *coordinates)
                                 for name in base.keys()}
                    save_file(state, path / file.name, metadata={'format': 'pt'})
                    del state
                else:
                    shutil.copy2(file, path / file.name)
        path.rename(destination)
    return destination


def train_direction(config, devices, reference, rows, output):
    direction = output / 'direction.pt'
    if direction.exists():
        return direction
    training_data = output / 'direction-questions.json'
    write(training_data, rows)
    training = Path(mkdtemp(dir=output, prefix='rloo-'))
    env = inference_env(','.join(devices))
    env.pop('SDAR_VISIBLE_DEVICES', None)
    env.pop('SDAR_LOCAL_RANK', None)
    tp = config['learner_tp_size']
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
               f'--nproc_per_node={len(devices)}', str(ROOT / 'sdar_justgmpo/train.py'),
               '--model', str(reference), '--data', str(training_data), '--output', str(training),
               '--objective', 'rloo', '--steps', '1', '--direction-output', str(direction),
               '--batch-prompts', str(len(rows)), '--group-size', str(config['group_size']),
               '--minibatch-size', str(len(rows) * config['group_size']),
               '--temperature', str(config['temperature']), '--lr', str(config['lr']),
               '--tp-size', str(tp), '--dp-size', str(len(devices) // tp),
               '--block-size', str(config['block_size']), '--max-tokens', str(config['max_tokens']),
               '--seed', str(config['seed']),
               '--rollout-batch-size', str(config['learner_rollout_batch_size']),
               '--train-phase-batch-size', str(config['train_phase_batch_size']),
               '--inference-cache-fraction', str(config['learner_cache_fraction']),
               '--inference-cuda-graph' if config['learner_cuda_graph'] else '--no-inference-cuda-graph']
    for saved in sorted(output.glob('rloo-*/.rollout-*')):
        if all((saved / f'response-dp{i}.json').is_file() for i in range(len(devices) // tp)):
            print(f'Reusing saved sequential rollouts: {saved}', flush=True)
            command.extend(['--direction-replay', str(saved)])
            break
    log_path = training / 'train.log'
    print(f'RLOO on TP{tp} × DP{len(devices) // tp}: {log_path}', flush=True)
    with log_path.open('w') as log_file:
        subprocess.run(command, env=env, stdout=log_file, stderr=subprocess.STDOUT, check=True)
    return direction


def objectives(rows, rho):
    return np.array([np.mean([r['reward'] for r in rows]),
                     np.mean([r['num_tokens'] - rho * r['denoise_forwards'] for r in rows])])


def smoothed_gradients(perturbations, returns):
    """Two antithetic score estimators in standard normal coordinates."""
    differences = returns[:, 0, :] - returns[:, 1, :]
    return differences.T @ perturbations / (2 * len(perturbations))


def common_direction(gradients):
    norms = np.linalg.norm(gradients, axis=1)
    if np.any(norms == 0):
        return None
    direction = (gradients / norms[:, None]).sum(axis=0)
    if np.any(gradients @ direction <= 0):
        return None
    return direction / np.linalg.norm(direction)


def explore(config, devices, reference, delta, rows, rho, scale, output):
    search = output / 'search'
    search.mkdir(exist_ok=True)
    perturbations = np.random.default_rng(config['seed']).standard_normal((config['search_pairs'], 2))
    settings = [(f'{i}-{label}', tuple(sign * u * scale))
                for i, u in enumerate(perturbations) for sign, label in ((1, 'plus'), (-1, 'minus'))]
    pending = [(name, coords) for name, coords in settings if not (search / f'{name}.json').exists()]
    for offset in range(0, len(pending), len(devices)):
        wave = pending[offset:offset + len(devices)]
        with TemporaryDirectory(dir=search, prefix='.weights-') as directory:
            models = [export_candidate(reference, delta, coords, Path(directory) / name)
                      for name, coords in wave]
            print(f'Dynamic search: {len(models)} models × {len(rows)} questions', flush=True)
            with ProcessPoolExecutor(max_workers=len(models), mp_context=mp.get_context('spawn'),
                                     max_tasks_per_child=1) as pool:
                jobs = {pool.submit(worker, config, gpu, model, rows, 'dynamic', False): name
                        for ((name, _), gpu, model) in zip(wave, devices, models)}
                for job in as_completed(jobs):
                    write(search / f'{jobs[job]}.json', job.result())
    results = [read(search / f'{name}.json') for name, _ in settings]
    returns = np.array([objectives(result, rho) for result in results]).reshape(-1, 2, 2)
    gradients = smoothed_gradients(perturbations, returns)
    direction = common_direction(gradients)
    write(output / 'search.json', dict(rho=rho, perturbations=perturbations.tolist(),
          gradients=gradients.tolist(), direction=None if direction is None else direction.tolist(),
          candidates=[dict(name=name, coordinates=list(coords), metrics=dynamic_metrics(result))
                      for (name, coords), result in zip(settings, results, strict=True)]))
    return direction


def compare(reference, candidate):
    base, base_seq = reference
    model, model_seq = candidate
    result = compare_dynamic(base, model)
    expected = [r['index'] for r in base]
    if any([r['index'] for r in rows] != expected for rows in (base_seq, model_seq)):
        raise ValueError('Use the same ordered questions for both decoding modes')
    seq_base = float(np.mean([r['rewards'] for r in base_seq]))
    seq_model = float(np.mean([r['rewards'] for r in model_seq]))
    return dict(result, tpf_gain=result['candidate']['tpf_denoise'] - result['reference']['tpf_denoise'],
                sequential_gain=seq_model - seq_base, sequential_reference=seq_base,
                sequential_candidate=seq_model,
                sequential_grading_timeouts=sum(r['timeouts'] for r in base_seq + model_seq))


def thresholds(config):
    return dict(accuracy_gain=config['minimum_dynamic_gain'], tpf_gain=config['minimum_tpf_gain'],
                forward_saving=config['minimum_forward_saving'],
                sequential_gain=-config['maximum_sequential_loss'])


def check_thresholds(metrics, limits):
    checks = {key: metrics[key] >= limit for key, limit in limits.items()}
    for key in ('accuracy_gain', 'tpf_gain'):
        checks[key] = metrics[key] > limits[key]
    return checks


def population_interval(values, bounds, population, tail):
    """One tail of Theorem 4.3 (Bardenet–Maillard) at each endpoint."""
    values = np.asarray(values, dtype=np.float64)
    low, high = bounds
    n = len(values)
    if not 0 < n <= population or not np.all((low <= values) & (values <= high)):
        raise ValueError('Values must belong to the bounded confirmation population')
    center = float(values.mean())
    if n == population:
        return center, center
    correction = 1 - (n - 1) / population if n <= population / 2 else (1 - n / population) * (1 + 1 / n)
    level = log(5 / tail)
    radius = float(values.std()) * sqrt(2 * correction * level / n)
    radius += (7 / 3 + 3 / sqrt(2)) * (high - low) * level / n
    return max(low, center - radius), min(high, center + radius)


def acceptance(config, reference, candidate, population, rho):
    report = compare(reference, candidate)
    base, base_seq = reference
    model, model_seq = candidate
    rr, ll, dd, ff = (np.array([[row[key] for row in rows] for rows in (base, model)])
                      for key in ('reward', 'num_tokens', 'denoise_forwards', 'forward_calls'))
    # Dynamic commits at least one token per masked forward, with block padding and cache passes.
    tokens, block = config['max_tokens'], config['block_size']
    max_denoise = tokens + 2 * block
    max_forward = max_denoise + ceil((tokens + 4096) / block) + 4
    tail = config['alpha'] / 14  # Thirteen finite-population tails and one sampled-sequence tail.

    def interval(values, low, high):
        return population_interval(values, (low, high), population, tail)

    accuracy, _ = interval(rr[1] - rr[0], -1, 1)
    saved, _ = interval(ff[0] - ff[1], 1 - max_forward, max_forward - 1)
    base_cost = interval(ff[0], 1, max_forward)
    denoise = [interval(counts, 1, max_denoise) for counts in dd]
    residual = ll - rho * dd
    base_residual = interval(residual[0], -rho * max_denoise, tokens)
    rho_gap = [h / d for h in base_residual for d in denoise[0]]
    dd_gap = interval(dd[1] - dd[0], 1 - max_denoise, max_denoise - 1)
    span = tokens + rho * (max_denoise - 1)
    gap, _ = interval(residual[1] - residual[0], -span, span)
    # TPF_1 - TPF_0 = (delta residual - (TPF_0 - rho) * delta denoise) / mean denoise_1.
    tpf_numerator = gap - max(x * y for x in (min(rho_gap), max(rho_gap)) for y in dd_gap)
    # Hoeffding also covers independent within-prompt rollout randomness. Uniform sampling
    # without replacement has a smaller exponential moment than its with-replacement analogue.
    sequential = report['sequential_gain'] - 2 * sqrt(log(1 / tail) / (2 * len(base_seq)))
    bounds = dict(accuracy_gain=accuracy, tpf_gain=min(tpf_numerator / d for d in denoise[1]),
                  forward_saving=min(saved / c for c in base_cost), sequential_gain=sequential)
    limits = thresholds(config)
    passes = check_thresholds(bounds, limits)
    return dict(report, simultaneous_lower_bounds=bounds, thresholds=limits, passes=passes,
                accepted=all(passes.values()), inference=dict(alpha=config['alpha'], population=population,
                dynamic='paired empirical Bernstein–Serfling', sequential='paired question Hoeffding'))


def evaluate(config, devices, path, rows, cache):
    return tuple(generate(config, devices, path, rows, mode, cache / mode)
                 for mode in ('dynamic', 'sequential'))


def run(config, output):
    devices = dict(line.split(', ', 1) for line in subprocess.check_output(
        ['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader,nounits'], text=True).splitlines())
    gpus = [devices[str(i)] for i in config['gpus']]
    torch.set_num_threads(4)
    output.mkdir(exist_ok=True, parents=True)
    if (output / 'config.json').exists() and read(output / 'config.json') != config:
        raise ValueError('Use the original configuration when continuing this experiment')
    write(output / 'config.json', config)
    split_path = output / 'splits.json'
    direction_size = config['direction_questions']
    if not split_path.exists():
        previous = [path for path in RUNS.glob('*/splits.json') if path != split_path]
        split_config = dict(config, train_questions=direction_size + config['search_questions'])
        splits, fresh = make_splits(split_config, exclude_splits=previous, with_population=True)
        write(output / 'population.json', dict(fresh=fresh, confirmation=fresh - len(splits['train']) - len(splits['dev'])))
        write(split_path, splits)
    splits = read(split_path)
    reference = Path(config['reference_model'])
    train_rows = splits['train'][:direction_size]
    weights = train_direction(config, gpus, reference, train_rows, output)
    delta = torch.load(weights, weights_only=True, mmap=True, map_location='cpu')
    calibration = generate(config, gpus, reference, train_rows, 'dynamic', output / 'calibration')
    rho = dynamic_metrics(calibration)['tpf_denoise']
    scale = np.array([config['sigma_update'], config['sigma_logit']])
    direction = explore(config, gpus, reference, delta, splits['train'][direction_size:], rho, scale, output)

    dev_rows = splits['dev']
    baseline = evaluate(config, gpus, reference, dev_rows, output / 'dev' / 'reference')
    development = {}
    models = {}

    def develop(name, coordinates):
        path = export_candidate(reference, delta, coordinates, output / name)
        metrics = compare(baseline, evaluate(config, gpus, path, dev_rows, output / 'dev' / name))
        development[name] = dict(metrics, coordinates=list(coordinates))
        models[name] = path
        write(output / 'development.json', development)

    develop('sequential-only', (1.0, 0.0))
    joint = []
    if direction is not None:
        for i, step in enumerate(config['step_sizes'], 1):
            name = f'joint-{i}'
            develop(name, tuple(step * scale * direction))
            joint.append(name)
    limits = thresholds(config)
    eligible = [name for name in joint if all(development[name][key] > limits[key]
                                               for key in ('accuracy_gain', 'tpf_gain'))]

    def rank(name):
        result = development[name]
        return (all(check_thresholds(result, limits).values()),
                result['accuracy_gain'], result['tpf_gain'], result['forward_saving'])

    choice = max(eligible, key=rank, default=None)
    diagnostic = choice or max(joint, key=rank, default='sequential-only')
    scale_only = development[diagnostic]['coordinates'][1] if joint else config['sigma_logit']
    develop('scale-only', (0.0, scale_only))
    write(output / 'selection.json', dict(candidate=choice, math500_candidate=diagnostic, thresholds=limits))
    print(f'Development selected: {choice}; MATH-500 comparison: {diagnostic}', flush=True)

    accepted = False
    if choice is not None:
        rows = splits['holdout']
        cache = output / 'holdout'
        base = evaluate(config, gpus, reference, rows, cache / 'reference')
        model = evaluate(config, gpus, models[choice], rows, cache / choice)
        population = read(output / 'population.json')['confirmation']
        holdout = acceptance(config, base, model, population, rho)
        accepted = holdout['accepted']
        write(output / 'holdout.json', dict(holdout, candidate_path=str(models[choice])))
    else:
        write(output / 'holdout.json', dict(status='skipped', reason='no joint candidate met the development thresholds'))
    cache = output / 'math500'
    base = evaluate(config, gpus, reference, splits['math500'], cache / 'reference')
    model = evaluate(config, gpus, models[diagnostic], splits['math500'], cache / diagnostic)
    write(output / 'math500.json', dict(compare(base, model), candidate_path=str(models[diagnostic])))
    selected = models[choice] if accepted else reference
    write(output / 'completed.json', dict(accepted=accepted, selected_model=str(selected),
                                          math500_candidate=str(models[diagnostic])))
    print(f'Finished. Selected model: {selected}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    config = read(args.config)
    for key in ('reference_model', 'train_data', 'math500_data'):
        config[key] = str((ROOT / config[key]).resolve())
    configure_environment()
    output = (args.output or RUNS / datetime.now().strftime('joint-rl-%Y%m%d-%H%M%S')).resolve()
    with run_lock(output):
        run(config, output)
