"""Train and evaluate a one-decision Dynamic decoding intervention."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import json
import multiprocessing as mp
import os
from pathlib import Path
import random
import subprocess

import numpy as np
import torch

from branch_rollout import worker
from data import load_math

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'sdar_justgmpo/runs'


def read(path):
    return json.loads(path.read_text())


def write(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def make_splits(config):
    """Use questions absent from earlier saved RL rollouts for the three new splits."""
    history = load_math(config['math500_data'])
    seen = {' '.join(row['question'].split()) for row in history}
    for path in RUNS.glob('*/rollouts-rank*.jsonl'):
        with path.open() as file:
            for line in file:
                seen.add(' '.join(json.loads(line)['question'].split()))
    fresh = []
    for row in load_math(config['train_data'], training=True):
        key = ' '.join(row['question'].split())
        if key not in seen:
            fresh.append(row)
            seen.add(key)
    random.Random(config['seed']).shuffle(fresh)
    splits = {}
    cursor = 0
    for name in ('train', 'dev', 'holdout'):
        end = cursor + config[f'{name}_questions']
        splits[name] = [dict(row, index=i) for i, row in enumerate(fresh[cursor:end], cursor)]
        if len(splits[name]) != end - cursor:
            raise ValueError(f'Insufficient fresh questions for {name}')
        cursor = end
    splits['math500'] = [dict(row, index=i) for i, row in enumerate(history, cursor)]
    return splits


def generate(config, devices, model, rows, mode, cache):
    """Publish each completed shard so an interrupted run can continue."""
    cache.mkdir(parents=True, exist_ok=True)
    count = min(len(devices), len(rows))
    jobs = [(gpu, rows[len(rows) * i // count:len(rows) * (i + 1) // count], cache / f'shard-{i}.json')
            for i, gpu in enumerate(devices[:count])]
    pending = [(gpu, shard, path) for gpu, shard, path in jobs if not path.exists()]
    if pending:
        print(f'{cache.name}: {mode}, {len(rows)} questions on {len(pending)} GPUs', flush=True)
        with ProcessPoolExecutor(max_workers=len(pending), mp_context=mp.get_context('spawn')) as pool:
            tasks = {pool.submit(worker, config, gpu, model, shard, mode, True): path
                     for gpu, shard, path in pending}
            for task in as_completed(tasks):
                write(tasks[task], task.result())
    return sorted((row for _, _, path in jobs for row in read(path)), key=lambda row: row['index'])


def branch_loss(probability, reward_delta, forward_delta, threshold, tau, forward_price):
    gate = torch.sigmoid((probability - threshold) / tau)
    return -(gate * (reward_delta - forward_price * forward_delta)).mean()


def features(config, weight, records, path):
    from likelihood import projected_token_probability

    if path.exists():
        return torch.load(path, map_location='cpu', weights_only=True)
    triggered = [r for r in records if r['branch'] is not None]
    branches = [r['branch'] for r in triggered]
    if not branches:
        raise ValueError(f'No questions reached a marginal commit state for {path.name}')
    bank = dict(
        indices=torch.tensor([r['index'] for r in triggered]),
        hidden=torch.tensor([r['hidden'] for r in branches], dtype=torch.bfloat16),
        tokens=torch.tensor([r['token_id'] for r in branches]),
        reward=torch.tensor([r['delta_reward'] for r in branches], dtype=torch.float32),
        forwards=torch.tensor([r['delta_forwards'] for r in branches], dtype=torch.float32),
        reference=torch.tensor([r['reference_probability'] for r in branches]))
    with torch.no_grad():
        probabilities, top1 = projected_token_probability(weight, bank['hidden'].cuda(), bank['tokens'].cuda())
    error = (probabilities.cpu() - bank['reference']).abs()
    stats = dict(states=len(branches), top1_mismatches=int(sum(top1.cpu() != bank['tokens'])),
                 mean_probability_error=float(error.mean()), max_probability_error=float(error.max()))
    print(f'Partial-block probability comparison: {stats}', flush=True)
    if stats['top1_mismatches'] or stats['max_probability_error'] > config['probability_tolerance']:
        raise ValueError(f'Partial-block probabilities do not match rollout within tolerance: {stats}')
    write(path.with_suffix('.json'), stats)
    staging = path.with_suffix('.tmp')
    torch.save(bank, staging)
    staging.replace(path)
    return bank


def head_weight(model_path):
    from safetensors import safe_open

    model_path = Path(model_path)
    name = 'lm_head.weight'
    index = read(model_path / 'model.safetensors.index.json')
    with safe_open(model_path / index['weight_map'][name], framework='pt', device='cpu') as shard:
        return shard.get_tensor(name)


def train_head(config, records, output):
    from likelihood import SDARPolicy, projected_token_probability
    from runtime import save_checkpoint
    from transformers import AutoTokenizer

    model_path = config['reference_model']
    model = SDARPolicy.from_pretrained(model_path, torch_dtype=torch.bfloat16).cuda().requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    bank = features(config, model.lm_head.weight, records, output / 'states.pt')
    batch = {k: bank[k].cuda() for k in ('hidden', 'tokens', 'reward', 'forwards')}
    head = torch.nn.Parameter(model.lm_head.weight.detach().float())
    optimizer = torch.optim.Adam([head], lr=config['lr'])
    options = [config[k] for k in ('threshold', 'tau', 'forward_price')]
    snapshots = config['save_steps']
    metrics = []
    for step in range(1, snapshots[-1] + 1):
        optimizer.zero_grad(set_to_none=True)
        probability, _ = projected_token_probability(head, batch['hidden'], batch['tokens'])
        loss = branch_loss(probability, batch['reward'], batch['forwards'], *options)
        loss.backward()
        optimizer.step()
        metrics.append(dict(step=step, loss_before_update=loss.item()))
        if step in snapshots:
            with torch.no_grad():
                model.lm_head.weight.copy_(head)
            path = output / f'step-{step}'
            save_checkpoint(model, tokenizer, path, tokenizer_source=model_path)
            print(f'Head step {step}: loss {loss.item():.6f}, saved {path}', flush=True)
    write(output / 'optimization.json', metrics)


def dynamic_metrics(rows):
    tokens = sum(r['num_tokens'] for r in rows)
    forwards = sum(r['forward_calls'] for r in rows)
    return dict(questions=len(rows), correct=int(sum(r['reward'] for r in rows)),
                accuracy=float(np.mean([r['reward'] for r in rows])), mean_forwards=forwards / len(rows),
                total_forwards=forwards, tpf_denoise=tokens / sum(r['denoise_forwards'] for r in rows),
                mean_tokens=tokens / len(rows), truncated=sum(not r['finished'] for r in rows),
                grading_timeouts=sum(r['grading'] == 'timeout' for r in rows))


def paired_dynamic(reference, candidate):
    if [r['index'] for r in reference] != [r['index'] for r in candidate]:
        raise ValueError('Evaluation requires the same questions in the same order')
    reward = np.array([c['reward'] - b['reward'] for b, c in zip(reference, candidate, strict=True)])
    forwards = np.array([[b['forward_calls'], c['forward_calls']]
                         for b, c in zip(reference, candidate, strict=True)])
    return reward, forwards


def compare_dynamic(reference, candidate):
    reward, forwards = paired_dynamic(reference, candidate)
    baseline_cost, candidate_cost = forwards.mean(0)
    return dict(reference=dynamic_metrics(reference), candidate=dynamic_metrics(candidate),
                accuracy_gain=float(reward.mean()), forward_saving=float(1 - candidate_cost / baseline_cost),
                mean_forwards_saved=float(baseline_cost - candidate_cost),
                repair=int(sum(reward > 0)), harm=int(sum(reward < 0)))


def acceptance(config, reference, candidate, sequential_reference, sequential_candidate):
    reward, forwards = paired_dynamic(reference, candidate)
    indices = [r['index'] for r in reference]
    if any([r['index'] for r in rows] != indices for rows in (sequential_reference, sequential_candidate)):
        raise ValueError('Sequential and Dynamic comparisons must use the same held-out questions')
    def sequential_accuracy(rows):
        return np.array([np.mean(r['rewards']) for r in rows])
    seq_base, seq_candidate = map(sequential_accuracy, (sequential_reference, sequential_candidate))
    seq_delta = seq_candidate - seq_base
    rng = np.random.default_rng(config['seed'])
    draws = rng.integers(len(reference), size=(config['bootstrap_samples'], len(reference)))
    resampled_costs = forwards[draws].sum(1)
    lower = config['alpha'] / 3
    bounds = dict(dynamic_gain=float(np.quantile(reward[draws].mean(1), lower)),
                  forward_saving=float(np.quantile(1 - resampled_costs[:, 1] / resampled_costs[:, 0], lower)),
                  sequential_gain=float(np.quantile(seq_delta[draws].mean(1), lower)))
    thresholds = dict(dynamic_gain=config['minimum_dynamic_gain'],
                      forward_saving=config['minimum_forward_saving'],
                      sequential_gain=-config['maximum_sequential_loss'])
    passes = {key: bounds[key] > limit for key, limit in thresholds.items()}
    return dict(compare_dynamic(reference, candidate),
                sequential_reference=float(seq_base.mean()), sequential_candidate=float(seq_candidate.mean()),
                sequential_gain=float(seq_delta.mean()), sequential_grading_timeouts=sum(
                    r['timeouts'] for rows in (sequential_reference, sequential_candidate) for r in rows),
                simultaneous_lower_bounds=bounds, thresholds=thresholds, accepted=all(passes.values()), passes=passes,
                inference=dict(method='paired question bootstrap with Bonferroni correction', alpha=config['alpha'],
                               samples=config['bootstrap_samples']))


@torch.no_grad()
def local_gain(config, bank, rows, path):
    from likelihood import projected_token_probability

    indices = np.array([r['index'] for r in rows])
    hidden, tokens = bank['hidden'].cuda(), bank['tokens'].cuda()
    threshold, tau = config['threshold'], config['tau']
    probabilities = []
    for model in (config['reference_model'], path):
        weight = head_weight(model).cuda()
        probability, _ = projected_token_probability(weight, hidden, tokens)
        probabilities.append(probability.cpu())
        del weight
    old, new = probabilities
    gate_shift = torch.sigmoid((new - threshold) / tau) - torch.sigmoid((old - threshold) / tau)
    offsets = np.searchsorted(indices, bank['indices'].numpy())
    accuracy, savings = np.zeros((2, len(rows)))
    accuracy[offsets] = (gate_shift * bank['reward']).numpy()
    savings[offsets] = (-gate_shift * bank['forwards']).numpy()
    utility = accuracy + config['forward_price'] * savings
    draws = np.random.default_rng(config['seed']).integers(len(rows), size=(config['bootstrap_samples'], len(rows)))
    return dict(questions=len(rows), triggered=len(tokens), accuracy_gain=float(accuracy.mean()),
                forwards_saved=float(savings.mean()), utility_gain=float(utility.mean()),
                utility_ci95=np.quantile(utility[draws].mean(1), [.025, .975]).tolist())


def configure_environment():
    os.environ['HF_HOME'] = os.environ.get('SDAR_CACHE_DIR', str(ROOT.parent / '.cache/huggingface'))
    os.environ['HF_MODULES_CACHE'] = str(Path(os.environ['HF_HOME']) / 'modules')
    os.environ.setdefault('NCCL_P2P_DISABLE', '1')
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    os.environ['SDAR_FULL_VOCAB'] = '1'


def run(config, output):
    devices = dict(line.split(', ', 1) for line in subprocess.check_output(
        ['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader,nounits'], text=True).splitlines())
    gpus = [devices[str(i)] for i in config['gpus']]
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus[0]
    torch.set_num_threads(4)
    torch.manual_seed(config['seed'])
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'config.json').exists() and read(output / 'config.json') != config:
        raise ValueError('Use the original configuration to continue this run')
    write(output / 'config.json', config)
    if not (output / 'splits.json').exists():
        write(output / 'splits.json', make_splits(config))
    splits = read(output / 'splits.json')
    reference = config['reference_model']
    records = generate(config, gpus, reference, splits['train'], 'collect', output / 'branches')
    branches = [r['branch'] for r in records if r['branch'] is not None]
    write(output / 'collection.json', dict(questions=len(records), paired=len(branches),
        original_replays_verified=sum(r['original_action_verified'] for r in branches),
        reward_changes=sum(r['delta_reward'] != 0 for r in branches),
        forward_changes=sum(r['delta_forwards'] != 0 for r in branches),
        replay_overhead=sum(r['replay_overhead'] for r in branches),
        grading_timeouts=sum(r['grading'] == 'timeout' for r in records)
                        + sum(r['alternate_grading'] == 'timeout' for r in branches)))
    if not (output / 'optimization.json').exists():
        train_head(config, records, output)
        torch.cuda.empty_cache()
    dev_base = generate(config, gpus, reference, splits['dev'], 'dynamic', output / 'dev-reference')
    development = {}
    for step in config['save_steps']:
        path = output / f'step-{step}'
        rows = generate(config, gpus, str(path), splits['dev'], 'dynamic', output / f'dev-step-{step}')
        development[step] = compare_dynamic(dev_base, rows)
    def rank(step):
        metrics = development[step]
        quality, saving = metrics['accuracy_gain'], metrics['forward_saving']
        meets = quality >= config['minimum_dynamic_gain'] and saving >= config['minimum_forward_saving']
        return meets, quality + config['forward_price'] * metrics['mean_forwards_saved']
    chosen = max(development, key=rank)
    candidate = output / f'step-{chosen}'
    write(output / 'selection.json', dict(chosen_step=chosen, development=development))
    print(f'Candidate selected on development questions: {candidate}', flush=True)
    holdout = splits['holdout']
    holdout_base = generate(config, gpus, reference, holdout, 'collect', output / 'holdout-branches')
    bank = features(config, head_weight(reference).cuda(), holdout_base, output / 'holdout-states.pt')
    gain = local_gain(config, bank, holdout_base, candidate)
    torch.cuda.empty_cache()
    dynamic = generate(config, gpus, str(candidate), holdout, 'dynamic', output / 'holdout-dynamic')
    sequential = [generate(config, gpus, str(model), holdout, 'sequential', output / f'holdout-seq-{name}')
                  for name, model in [('reference', reference), ('candidate', candidate)]]
    report = acceptance(config, holdout_base, dynamic, *sequential)
    report.update(candidate_path=str(candidate), local_branch=gain)
    write(output / 'holdout.json', report)
    print(f"Holdout acceptance: {report['accepted']}, checks: {report['passes']}", flush=True)
    if splits['math500']:
        history = [generate(config, gpus, str(model), splits['math500'], 'dynamic', output / f'math500-{name}')
                   for name, model in [('reference', reference), ('candidate', candidate)]]
        write(output / 'math500.json', compare_dynamic(*history))
    write(output / 'completed.json', dict(candidate=str(candidate), accepted=report['accepted']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    settings = read(args.config)
    for key in ('reference_model', 'train_data', 'math500_data'):
        settings[key] = str((ROOT / settings[key]).resolve())
    configure_environment()
    directory = args.output or RUNS / datetime.now().strftime('branch-rl-%Y%m%d-%H%M%S')
    run(settings, directory.resolve())
