"""Paired position/value interventions, with no model or GPU dependency."""
from collections import Counter
import hashlib
import itertools
import json
import random

from protocol import stable_seed

FORWARD_CAP = 512
MATCHED_SEEDS = list(range(20261111, 20261143))


def make_pool(case, eos_ids=(), vocab_size=None):
    state = case['snapshot']
    current, historical = state['current_ids'], state['historical_ids']
    selected, masked = set(state['positions']), set(state['masked_positions'])
    confidence = state['confidence']
    mask_id = state['input_ids'][state['masked_positions'][0]]
    legal = lambda token: token != mask_id and token >= 0 and (vocab_size is None or token < vocab_size)
    risk = lambda i: (-(current[i] != historical[i]), confidence[i], i)
    protected = {i for i in masked if confidence[i] >= .95 and current[i] == historical[i]}
    legacy = {a['name']: a for a in case['actions']}
    first = legacy['Delay1']['deferred_position']
    assert first in selected - protected
    movable = [first] + sorted(selected - protected - {first}, key=risk)[:2]
    incoming = [legacy['Swap1']['replacement_position']]
    remaining = masked - selected - set(incoming)
    if remaining:
        incoming += sorted(remaining, key=lambda i: (-confidence[i], i))[:1]
    active = set(movable + incoming)
    fixed = selected - set(movable)
    editable = active - protected
    positions = {'P0': sorted(selected), 'P1': sorted(selected - {first}),
                 'P3': sorted(selected - {first} | {incoming[0]})}
    if len(movable) >= 2:
        removed = set(movable[:2])
        positions['P2'] = sorted(selected - removed)
        if len(incoming) == 2:
            positions['P4'] = sorted(selected - removed | set(incoming))
    positions = dict(sorted(positions.items()))
    values = {'V0': list(current)}
    rejected = []

    def value_plan(name, changes, origin):
        values[name] = list(current)
        for i, token in changes:
            if i in editable and legal(token):
                values[name][i] = token
            else:
                rejected.append({'plan': name, 'position': i, 'token': token, 'origin': origin})

    top2 = lambda i: state['topk_ids'][i][1]
    value_plan('V1', [(movable[-1], top2(movable[-1]))], 'top2')
    new_editable = [i for i in incoming if i in editable]
    pair = [first] + new_editable[:1]
    value_plan('V2', [(i, top2(i)) for i in pair], 'top2')
    differing = sorted([i for i in editable if historical[i] != current[i] and legal(historical[i])])
    adjacent = [(i, j) for i, j in itertools.combinations(differing, 2) if j == i + 1]
    history_positions = list(adjacent[0]) if adjacent else sorted(differing, key=risk)[:1]
    value_plan('V3', [(i, historical[i]) for i in history_positions], 'previous_top1')
    actions, matrix = {}, {}
    for p, subset in positions.items():
        assert fixed <= set(subset) <= masked and 1 <= len(subset) <= len(selected)
        for v, vector in values.items():
            tokens = [vector[i] for i in subset]
            key = json.dumps([subset, tokens], separators=(',', ':'))
            action_id = hashlib.sha256(key.encode()).hexdigest()[:12]
            nominal = 'G0' if p == 'P0' and v == 'V0' else 'G1' if v == 'V0' else 'G2' if p == 'P0' else 'G3'
            changed = [i for i in subset if vector[i] != current[i]]
            actual = ('G3' if changed else 'G1') if set(subset) != selected else ('G2' if changed else 'G0')
            if action_id not in actions:
                actions[action_id] = {'id': action_id, 'positions': subset, 'tokens': tokens,
                    'group': actual, 'changed_values': changed, 'cells': [],
                    'contains_eos': bool(set(tokens) & set(eos_ids))}
            actions[action_id]['cells'].append({'position_plan': p, 'value_plan': v, 'nominal_group': nominal})
            matrix[f'{p}/{v}'] = action_id
    for p, name in (('P0', 'Current'), ('P1', 'Delay1'), ('P3', 'Swap1')):
        a = actions[matrix[f'{p}/V0']]
        assert a['positions'] == legacy[name]['positions'] and a['tokens'] == legacy[name]['tokens']
    return {'anchors': sorted(selected & protected), 'protected_values': sorted(protected),
        'movable': movable, 'incoming': incoming, 'active': sorted(active), 'fixed': sorted(fixed),
        'background': sorted(masked - fixed - active), 'position_plans': positions, 'value_plans': values,
        'history_mode': 'adjacent_pair' if adjacent else 'single' if history_positions else 'none',
        'history_positions': history_positions, 'rejected_values': rejected,
        'actions': list(actions.values()), 'matrix': matrix, 'current_id': matrix['P0/V0']}


def audit_path(result, events, case, action):
    state = case['snapshot']
    steps, q = int(case['setting'][1:]), len(state['positions'])
    debt = q - len(action['positions'])
    source = events[0]
    assert source['forward'] == state['source_forward']
    assert source['input_ids'] == state['input_ids'] and source['mask_before'] == state['mask']
    assert source['positions'] == action['positions'] and source['tokens'] == action['tokens']
    assert all(source['current_ids'][i] == state['current_ids'][i] for i in state['masked_positions'])
    assert all(source['topk_ids'][i][:2] == state['topk_ids'][i][:2] for i in state['masked_positions'])
    expected_input, expected_mask = list(state['input_ids']), list(state['mask'])
    for i, value in zip(action['positions'], action['tokens']):
        expected_input[i], expected_mask[i] = value, 1
    assert source['mask_after'] == expected_mask
    assert events[1]['input_ids'] == expected_input and events[1]['mask_before'] == expected_mask
    assert result['restore_forwards'] == state['source_forward']
    assert result['forward_calls'] == result['restore_forwards'] + result['future_forwards'] <= FORWARD_CAP
    assert sum(result['forward_types'].values()) == result['forward_calls']
    checked = 0
    for e in events:
        if e['phase'] == 'denoise':
            assert e['round'] <= steps
        assert all(e['mask_after'][i] != 0 for i, flag in enumerate(e['mask_before']) if flag == 1)
        assert all(e['mask_before'][i] == 0 for i in e['positions'])
        if e is not source:
            assert e['tokens'] == [e['current_ids'][i] for i in e['positions']]
        if e['full_generation_block'] and e['phase'] == 'denoise':
            expected_count = q
            if e is source:
                expected_count -= debt
            elif e['block_start'] == state['block_start'] and e['round'] == steps:
                expected_count += debt
            assert len(e['positions']) == expected_count
    for left, right in zip(events, events[1:]):
        if left['block_start'] != right['block_start']:
            continue
        next_input = list(left['input_ids'])
        for i, value in zip(left['positions'], left['tokens']):
            next_input[i] = value
        assert right['input_ids'] == next_input and right['mask_before'] == left['mask_after']
        checked += 1
    block = [e for e in events if e['block_start'] == state['block_start'] and e['phase'] == 'denoise']
    assert block[-1]['round'] == steps and 0 not in block[-1]['mask_after']
    return {'passed': True, 'source_distribution_matches': True, 'checked_transitions': checked,
        'source_block_final_round': steps, 'final_round_commits': len(block[-1]['positions']),
        'source_non_top1_values': len(action['changed_values']), 'total_forwards': result['forward_calls']}


def analyze_state(case, results):
    pool = case['pool']
    observed = {r['action_id']: r for r in results}
    assert set(observed) == {a['id'] for a in pool['actions']}
    summary = {'question_id': case['question_id'], 'setting': case['setting'],
        'state_id': case['snapshot']['id'], 'logical_cells': len(pool['matrix']),
        'unique_actions': len(observed), 'group_sizes': dict(Counter(a['group'] for a in pool['actions'])),
        'anchors': len(pool['anchors']), 'active_positions': len(pool['active']),
        'history_mode': pool['history_mode'], 'grading_timeouts': [a for a, r in observed.items() if r['grading'] == 'timeout']}
    if summary['grading_timeouts']:
        summary['eligible'] = False
        return summary
    current = pool['current_id']
    reward = lambda a: observed[a]['reward']
    groups = {g: [a['id'] for a in pool['actions'] if a['group'] == g] for g in ('G0', 'G1', 'G2', 'G3')}
    best = {g: max([reward(current)] + [reward(a) for a in ids]) for g, ids in groups.items()}
    singles = max(best['G0'], best['G1'], best['G2'])
    pure, joint = sorted(groups['G1'] + groups['G2']), sorted(groups['G3'])
    count = min(len(pure), len(joint))
    matched = []
    for seed in MATCHED_SEEDS:
        rng = random.Random(stable_seed(summary['state_id'], seed))
        a, b = [current] + rng.sample(pure, count), [current] + rng.sample(joint, count)
        matched.append({'seed': seed, 'singles_ids': a, 'joint_ids': b,
            'singles_reward': max(map(reward, a)), 'joint_reward': max(map(reward, b)),
            'singles_forwards': sum(observed[i]['forward_calls'] for i in a),
            'joint_forwards': sum(observed[i]['forward_calls'] for i in b)})
    interactions = []
    s0 = set(case['snapshot']['positions'])
    for p, subset in pool['position_plans'].items():
        if p == 'P0':
            continue
        for v, vector in pool['value_plans'].items():
            if v == 'V0':
                continue
            cells = [pool['matrix'][f'{x}/{y}'] for x, y in (('P0', 'V0'), (p, 'V0'), ('P0', v), (p, v))]
            scores = list(map(reward, cells))
            changed = {i for i in subset if vector[i] != case['snapshot']['current_ids'][i]}
            common = sorted(changed & s0)
            new = sorted(changed - s0)
            interactions.append({'position_plan': p, 'value_plan': v, 'action_ids': cells,
                'rewards_00_10_01_11': scores, 'interaction': scores[3] - scores[1] - scores[2] + scores[0],
                'true_joint': bool(changed), 'changed_shared_positions': common,
                'changed_new_positions': new, 'only_newly_enabled': bool(new) and not common,
                'paired_rescue': scores == [0, 0, 0, 1]})
    summary.update(eligible=True, best=best, singles_best=singles,
        all_best=max(r['reward'] for r in results), joint_exclusive=best['G3'] > singles,
        joint_exclusive_action_ids=[i for i in joint if reward(i) > singles],
        matched_extra_candidates=count, matched=matched, interactions=interactions,
        successful_actions=[a for a, r in observed.items() if r['reward']])
    return summary
