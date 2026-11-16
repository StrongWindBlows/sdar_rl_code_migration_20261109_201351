"""Experiment definitions and analysis, independent of the inference engine."""
from collections import Counter
import gzip
import hashlib
import itertools
import json
import math
import multiprocessing as mp
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
INSTRUCTION = r'(Please put the final answer in \boxed{} tag, i.e. $\boxed{answer here}$)'


def stable_seed(value, seed=20261110):
    return int.from_bytes(hashlib.sha256(f'{seed}:{value}'.encode()).digest()[:8], 'big')


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def read_jsonl(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_data(directory, scope):
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    rows = read_jsonl(directory / 'test.jsonl')
    dev, evaluation, pilot = map(set, (manifest['dev_ids'], manifest['eval_ids'], manifest['pilot_ids']))
    assert len(dev) == 20 and len(evaluation) == 80 and dev.isdisjoint(evaluation)
    assert len(pilot) == 10 and pilot <= dev, 'Pilot must be entirely in the development set'
    assert dev | evaluation == {row['unique_id'] for row in rows}
    assert file_hash(directory / 'test.jsonl') == manifest['sample_sha256']
    selected = {'pilot': pilot, 'dev': dev, 'eval': evaluation}[scope]
    rounds = {item['unique_id']: item['intervention_round'] for item in manifest['items']}
    return [dict(row, intervention_round=rounds[row['unique_id']]) for row in rows if row['unique_id'] in selected]


def action_pool(state, positions='confidence', seed=20261110):
    """Two binary pools; token-identical actions are evaluated only once."""
    current, historical = state['current_ids'], state['historical_ids']
    confidence = state['confidence']
    selected = list(state['positions'])
    rank = lambda i: (-(historical[i] != current[i]) if positions == 'disagreement' else 0,
                      -confidence[i], i)
    if positions == 'disagreement' and len(selected) == 4:
        selected = sorted(sorted(state['masked_positions'], key=rank)[:4])
    mutable = sorted(sorted(selected, key=rank)[:4])
    base = tuple(current[i] for i in selected)
    actions = {}

    def add(values, family):
        values = tuple(values)
        if values not in actions:
            key = json.dumps([selected, values], separators=(',', ':'))
            actions[values] = {'id': hashlib.sha256(key.encode()).hexdigest()[:12],
                              'positions': selected, 'tokens': list(values), 'families': []}
        if family not in actions[values]['families']:
            actions[values]['families'].append(family)

    for family in ('current_only', 'cross_round'):
        alternatives = [(current[i], state['topk_ids'][i][1] if family == 'current_only' else historical[i])
                        for i in mutable]
        for values in itertools.product(*alternatives):
            proposal = dict(zip(selected, base))
            proposal.update(zip(mutable, values))
            add([proposal[i] for i in selected], family)
    add(base, 'Current')
    parent = [historical[i] if i in mutable else current[i] for i in selected]
    add(parent, 'Historical')
    for value, action in actions.items():
        if 'cross_round' in action['families'] and value not in (base, tuple(parent)):
            action['families'].append('hybrid')
    pools = {family: [a['id'] for a in actions.values() if family in a['families']]
             for family in ('current_only', 'cross_round')}
    current_id = actions[base]['id']
    size = min(map(len, pools.values()))
    rng = random.Random(stable_seed(state['id'], seed))
    matched = {family: [current_id] + rng.sample([a for a in ids if a != current_id], size - 1)
               for family, ids in pools.items()}
    differences = sum(current[i] != historical[i] for i in mutable)
    return {'actions': list(actions.values()), 'pools': pools, 'matched': matched,
            'mutable_positions': mutable, 'differences': differences,
            'novel_hybrids': max(2 ** differences - 2, 0)}


def forward_budget(state, max_new_tokens, continuation_steps, block=32):
    """Worst-case continuation work, including block cache writes; source forward is already paid."""
    per_round = block // continuation_steps
    remaining = len(state['masked_positions']) - len(state['positions'])
    later_tokens = max(0, max_new_tokens - (state['block_start'] - state['prompt_length']) - block)
    return math.ceil(remaining / per_round) + 1 + math.ceil(later_tokens / block) * (continuation_steps + 1)


def trace_difference(expected, actual):
    keys = ('block_start', 'round', 'input_ids', 'mask_before', 'mask_after', 'positions', 'tokens')
    for index, (left, right) in enumerate(zip(expected, actual)):
        for key in keys:
            if left[key] != right[key]:
                return {'event': index, 'field': key, 'expected': left[key], 'actual': right[key]}
    if len(expected) != len(actual):
        return {'field': 'event_count', 'expected': len(expected), 'actual': len(actual)}
    return None


def restoration_difference(baseline, restored, state):
    suffix = [event for event in baseline['events'] if event['forward'] >= state['source_forward']]
    difference = trace_difference(suffix, restored['events'])
    if difference:
        return difference
    for key in ('token_ids', 'finish_reason'):
        if baseline[key] != restored[key]:
            return {'field': key, 'expected': baseline[key], 'actual': restored[key]}
    expected = baseline['forward_calls'] - state['source_forward']
    if expected != restored['future_forwards']:
        return {'field': 'future_forwards', 'expected': expected, 'actual': restored['future_forwards']}
    return None


def divergence_summary(events):
    complete = {e['block_start'] for e in events if e.get('phase') == 'cache' and e['full_generation_block']}
    selected = [e for e in events if e.get('d_selected') is not None and e['round'] in (2, 3)
                and e['block_start'] in complete and e['positions']]

    def summarize(group):
        width = max((len(e['positions']) for e in group), default=0)
        counts, mutable = Counter(e['d_selected'] for e in group), Counter(e['d_mutable'] for e in group)
        below = sum(n for d, n in mutable.items() if d < 2) / len(group) if group else None
        return {'states': len(group), 'd_selected': {d: counts[d] for d in range(width + 1)},
                'd_mutable': {d: mutable[d] for d in range(min(4, width) + 1)},
                'fraction_below_two': below, 'fraction_at_least_two': 1 - below if below is not None else None}

    return {**summarize(selected), 'by_round': {str(r): summarize([e for e in selected if e['round'] == r]) for r in (2, 3)}}


def paired_stats(left, right, seed=20261110):
    """right - left; resampling unit is the question."""
    common = sorted(set(left) & set(right))
    differences = [right[key] - left[key] for key in common]
    if not common:
        return {'n': 0}
    gains, losses = sum(d > 0 for d in differences), sum(d < 0 for d in differences)
    discordant = gains + losses
    p = min(1.0, 2 * sum(math.comb(discordant, k) for k in range(min(gains, losses) + 1)) / 2 ** discordant) if discordant else 1.0
    rng = random.Random(seed)
    samples = sorted(sum(rng.choices(differences, k=len(common))) / len(common) for _ in range(2000))
    return {'n': len(common), 'difference': sum(differences) / len(common), 'gains': gains,
            'losses': losses, 'bootstrap_95': [samples[50], samples[1949]], 'mcnemar_exact_p': p}


def oracle_summary(records):
    states = {}
    for row in records:
        states.setdefault(row['state_id'], []).append(row)
    detail, timeouts = [], []
    for state_id, rows in states.items():
        if any(row.get('grading') == 'timeout' for row in rows):
            timeouts.append(state_id)
            continue
        rewards = {r['action']['id']: r['reward'] for r in rows}
        pool = rows[0]['pool']
        best = lambda ids: max(rewards[key] for key in ids)
        parents = {name: max(r['reward'] for r in rows if name in r['action']['families'])
                   for name in ('Current', 'Historical')}
        hybrid = [r['reward'] for r in rows if 'hybrid' in r['action']['families']]
        detail.append({'state_id': state_id, 'question_id': rows[0]['question_id'], **parents,
                       'unique_actions': len(rows), 'differences': pool['differences'],
                       'hybrid_rescue': bool(hybrid and max(hybrid) > max(parents.values())),
                       'best_current_only': best(pool['pools']['current_only']),
                       'best_cross_round': best(pool['pools']['cross_round']),
                       'matched_current_only': best(pool['matched']['current_only']),
                       'matched_cross_round': best(pool['matched']['cross_round']),
                       'cross_is_current_subset': set(pool['pools']['cross_round']) <= set(pool['pools']['current_only'])})
    return {'states': len(detail), 'states_with_grading_timeouts': timeouts,
            'hybrid_rescues': sum(r['hybrid_rescue'] for r in detail),
            'matched_comparison': paired_stats({r['question_id']: r['matched_current_only'] for r in detail},
                                               {r['question_id']: r['matched_cross_round'] for r in detail}),
            'details': detail}


def _init_grader():
    sys.path.insert(0, str(ROOT / 'sdar_justgmpo'))
    from reward import correctness
    global _correctness
    _correctness = correctness


def _grade(value):
    return _correctness(*value)


class Grader:
    def __init__(self, timeout=1.0):
        self.timeout = timeout
        self._start()

    def _start(self):
        self.pool = mp.get_context('spawn').Pool(2, initializer=_init_grader)
        self.pool.map(_grade, [('no answer', '0', True)] * 2)  # Import time is outside grading timeouts.

    def score(self, text, answer, finished):
        try:
            metadata, reward = self.pool.apply_async(_grade, ((text, answer, finished),)).get(self.timeout)
            return {'reward': reward, 'grading': 'ok' if finished else 'truncated', **metadata}
        except mp.TimeoutError:
            self.close()
            self._start()  # A timed-out symbolic calculation must not block subsequent answers.
            return {'reward': 0.0, 'grading': 'timeout'}

    def close(self):
        self.pool.terminate()
        self.pool.join()
