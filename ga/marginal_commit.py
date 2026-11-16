"""One-step commit-count interventions on natural Dynamic states."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import gzip
import json
import multiprocessing as mp
from pathlib import Path
import random
import shutil
import sys

import numpy as np

from degeneration import all_rows, events_for, repeat_runs
from early import snapshot
from protocol import (Grader, file_hash, read_jsonl, stable_seed,
                      trace_difference, write_json)
from threshold_rules import factor_count

GA = Path(__file__).resolve().parent
OLD = GA / 'runs/degeneration-20261112'
LANDSCAPE = GA / 'runs/threshold-landscape-20261112'
OUT = GA / 'runs/marginal-commit-20261112'
CODE = ('marginal_commit.py', 'marginal_commit.json', 'test_marginal_commit.py')


def read(path):
    return json.loads(Path(path).read_text())


def sources():
    return {r['question_id']: r for r in all_rows(OLD) if r['setting'] == 'Dynamic'}


def select_sources(events, qid, prompt_length, study):
    chosen = []
    for window, (lo, hi) in enumerate(study['windows']):
        candidates = []
        for e in events:
            m, k0 = e['mask_before'].count(0), len(e['positions'])
            counts = {max(1, min(m, k0+d)) for d in study['deltas']}
            if (e['phase'] == 'denoise' and e['full_generation_block'] and m >= 2
                    and e['forward'] < study['forward_cap'] and len(counts) >= 2
                    and lo <= e['block_start']-prompt_length < hi):
                candidates.append(e)
        if candidates:
            rng = random.Random(stable_seed(f'T3:{qid}:{window}', study['seed']))
            chosen.append((window, rng.choice(candidates)))
    return chosen


def make_actions(state, study):
    import torch
    p, masked, original = state['confidence'], state['masked_positions'], state['positions']
    rank = lambda i: (-p[i], i)
    order = sorted(original, key=rank) + sorted(set(masked)-set(original), key=rank)
    assert min(p[i] for i in original) >= max((p[i] for i in masked if i not in original), default=-1)
    k0, m = len(original), len(masked)
    policies = {f'D{d:+d}' if d else 'Current': max(1, min(m, k0+d)) for d in study['deltas']}
    for value in study['control_thresholds']:
        boundary = torch.tensor(value, dtype=getattr(torch, study['score_dtype'])).item()
        policies[f'F{round(value*100):03d}'] = max(1, sum(p[i] >= boundary for i in masked))
    policies['Factor1'] = factor_count([p[i] for i in masked], study['control_factor'])[0]
    actions = []
    for k in sorted(set(policies.values())):
        selected = sorted(order[:k])
        actions.append({'id': f'k{k:02d}', 'k': k, 'positions': selected,
                        'tokens': [state['current_ids'][i] for i in selected],
                        'policies': [name for name, count in policies.items() if count == k]})
    assert next(a for a in actions if a['k'] == k0)['positions'] == original
    return actions, policies


def features(state, action):
    """Current masked distributions and geometry only; no history or future outcomes."""
    m = state['masked_positions']
    selected, original = set(action['positions']), set(state['positions'])
    added, removed = sorted(selected-original), sorted(original-selected)
    p = np.asarray(state['confidence'])
    entropy = np.asarray(state['entropy'])
    margin = np.array([v[0]-v[1] for v in state['topk_probs']])
    k, k0 = action['k'], len(original)
    value = {'k': k, 'delta_k': k-k0, 'k0': k0, 'masked': len(m),
             'fraction': k/len(m), 'round': state['round'],
             'progress': (state['block_start']-state['prompt_length'])/8192,
             'forward_progress': state['source_forward']/512}
    groups = {'masked': m, 'selected': sorted(selected), 'added': added, 'removed': removed}
    for name, positions in groups.items():
        for metric, data in [('p', p), ('entropy', entropy)]:
            values = data[positions] if positions else np.array([0.])
            value[f'{name}_{metric}_mean'] = float(values.mean())
            value[f'{name}_{metric}_min'] = float(values.min())
    ranked = np.sort(p[m])[::-1]
    committed = [i for i, flag in enumerate(state['mask']) if flag != 0]
    complete = set(committed) | selected
    run = longest = 0
    for i in range(32):
        run = run+1 if i in complete else 0
        longest = max(longest, run)
    value.update(boundary_gap=float(ranked[k-1]-ranked[k]) if k < len(m) else 0.,
                 added_error_mass=float(sum(1-p[i] for i in added)),
                 selected_margin_mean=float(margin[sorted(selected)].mean()),
                 added_margin_min=float(margin[added].min()) if added else 0.,
                 entropy_spread=float(entropy[m].std()),
                 selected_span=(max(selected)-min(selected))/31,
                 completed_run=longest/32,
                 added_anchor_distance=float(np.mean([min((abs(i-j) for j in committed), default=32)
                                                      for i in added]))/32 if added else 0.)
    for q in (10, 25, 50, 75):
        value[f'p_q{q}'] = float(np.percentile(p[m], q))
    return value


def feature_groups(names):
    progress = ['k', 'delta_k', 'k0', 'masked', 'fraction', 'round', 'progress', 'forward_progress']
    confidence = [n for n in names if '_p_' in n]
    entropy = [n for n in names if '_entropy_' in n]
    return {'Progress': progress, 'Confidence': progress+confidence, 'Entropy': progress+entropy,
            'Simple': progress+confidence+entropy, 'Structure': list(names)}


def prepare(out):
    from run import fingerprint
    study = read(GA/'marginal_commit.json')
    original = read(OLD/'source_fingerprint.json')
    config = dict(read(OLD/'manifest.json')['config'], gpus=study['gpus'], trace=False,
                  max_new_tokens=study['max_new_tokens'], batch_size=1)
    assert fingerprint(read(GA/'runs/pilot8192-20261110/manifest.json')['config']) == original
    light = {r['question_id']: r for path in (LANDSCAPE/'E0').glob('gpu*.jsonl')
             for r in read_jsonl(path) if r['rule']['name'] == 'F090'}
    data, rows = read(OLD/'data.json'), sources()
    gpu_of = read(LANDSCAPE/'manifest.json')['gpu_of']
    cases, missing = [], []
    for qid in sorted(rows):
        row, events = rows[qid], events_for(rows[qid])
        native = read_jsonl(light[qid]['trace_path'])
        assert trace_difference(events, native) is None
        prompt_length = len(row['state']['prompt_ids'])
        selected = select_sources(events, qid, prompt_length, study)
        for window in set(range(len(study['windows'])))-{w for w, _ in selected}:
            missing.append({'question_id': qid, 'window': window})
        for window, event in selected:
            state = snapshot(row, events, event['forward'])
            state['confidence'] = native[event['forward']-1]['confidence']
            state['entropy'] = event['entropy']
            actions, policies = make_actions(state, study)
            cases.append({'id': f's{len(cases):03d}', 'question_id': qid, 'window': window,
                          'gpu': gpu_of[qid],
                          'snapshot': state, 'actions': actions, 'policies': policies,
                          'features': {a['id']: features(state, a) for a in actions}})
    pilot = read(Path(config['data'])/'manifest.json')['pilot_ids']
    names = sorted(cases[0]['features'][cases[0]['actions'][0]['id']])
    assert not (out/'manifest.json').exists()
    out.mkdir(parents=True, exist_ok=True)
    write_json(out/'data.json', data)
    write_json(out/'cases.json', cases)
    write_json(out/'missing.json', missing)
    write_json(out/'original_fingerprint.json', original)
    write_json(out/'manifest.json', {'date': '2026-11-12', 'study': study, 'engine': config,
        'pilot_ids': pilot, 'folds': read(OLD/'folds.json'), 'feature_groups': feature_groups(names),
        'code': {name: file_hash(GA/name) for name in CODE}, 'cases_sha256': file_hash(out/'cases.json'),
        'data_sha256': file_hash(out/'data.json'),
        'source_hashes': {str(Path(r['trace_path'])): file_hash(r['trace_path']) for r in rows.values()},
        'dependencies': {name: file_hash(GA/name) for name in ('engine.py', 'protocol.py', 'early.py',
                         'degeneration.py', 'threshold_study.py', 'threshold_rules.py', 'joint50.py')},
        'protocol': {
            'sampling': 'Seeded uniform event per fixed token window; full blocks, >=2 masked, >=2 distinct main counts. No reward/EOS-length/repetition filtering.',
            'action': 'Nested native-confidence order; original Current set preserved, ties by position. All token values fixed to current top-1. Single intervention then Dynamic .90.',
            'controls': 'Current-step count recommendations .85/.95/factor=1 under the common nested ordering, then .90 continuation; not full-policy baseline generations.',
            'budget': '512 total actual forwards per path, including prefix replay and source forward; max8192 generated tokens. No fixed block-round limit.',
            'primary': 'Window0: one intervention per question. Other windows diagnostic; aggregate states within question before bootstrap.',
            'prediction': 'Five folds by question. Fixed depth2/minleaf12 regression trees for delta reward and delta forwards; no hyperparameter search. Max predicted delta reward >=0, then min predicted delta forwards; Current has known relative values (0,0).',
            'evaluation': 'Only original development50. Prior confirmation50 already used, final400 reserved.',
            'timing': 'Compact in-memory events, source statistics reused. Disk I/O and grading outside generation timer; offline counterfactual cost reported separately.'}})
    (out/'source').mkdir()
    for name in CODE:
        shutil.copy2(GA/name, out/'source'/name)
    (out/'PLAN.md').write_text('# Experiment T3：Marginal Commit Utility\n\n'
        '固定开发50题；Dynamic .90 自然状态，每题三个固定生成区间各抽一个状态。'
        '主动作 k0-2/k0/k0+2/k0+4，并补齐 .85/.95/factor1 建议数量；合法裁剪、去重。'
        '固定当前 top-1，只干预一次，随后回到 Dynamic .90；每条路径512实际整题forward、8192tokens。\n\n'
        '先 check，再原10题 pilot，恢复验收通过后运行全部50题；不根据pilot收益修改候选。'
        'Window0为主结果，其余区间为阶段诊断。按题五折浅树消融；按题bootstrap。'
        '报告正确/损害/修复、forward、当前块剩余步数、TPF、重复和截断；两个错误路径之间的节省不记有效质量收益。\n\n'
        '本轮不访问已确认50题与最终400题。所有参数和源码指纹见 manifest.json。\n')
    print(json.dumps({'questions': len(data), 'states': len(cases),
        'windows': dict(Counter(c['window'] for c in cases)), 'actions': sum(len(c['actions']) for c in cases),
        'pilot_states': sum(c['question_id'] in pilot for c in cases)}, ensure_ascii=False))


def backend_class():
    from threshold_study import backend_class as study_backend

    class MarginalBackend(study_backend()):
        def _record(self, track, base, input_ids, before, after, proposed, output, is_decoding):
            super()._record(track, base, input_ids, before, after, proposed, output, is_decoding)
            track.events[-1]['current_ids'] = proposed
    return MarginalBackend


def capped_reference(row, events, cap):
    selected = [e for e in events if e['forward'] <= cap]
    if row['forward_calls'] <= cap:
        return dict(row, events=selected)
    prompt = len(row['state']['prompt_ids'])
    end = max([prompt]+[e['block_start']+32 for e in selected if 0 not in e['mask_after']])
    return dict(row, events=selected, token_ids=row['token_ids'][:end-prompt],
                finish_reason='forward_budget', finished=False, forward_calls=cap)


def block_metrics(events, state, reason):
    block = [e for e in events if e['block_start'] == state['block_start']]
    complete = next((e for e in block if 0 not in e['mask_after']), None)
    return {'complete': complete is not None,
            'extra_denoise': sum(e['phase'] == 'denoise' and e['forward'] > state['source_forward']
                                  for e in block if complete is None or e['forward'] <= complete['forward'])
                             if complete else None,
            'completion_forward': complete['forward'] if complete else None,
            'censored_reason': None if complete else reason}


def audit_path(result, case, action, reference_events, cap):
    state, suffix = case['snapshot'], result['events']
    source = suffix[0]
    assert source['forward'] == state['source_forward'] == result['restore_forwards']
    assert source['input_ids'] == state['input_ids'] and source['mask_before'] == state['mask']
    assert source['positions'] == action['positions'] and source['tokens'] == action['tokens']
    assert all(source['current_ids'][i] == state['current_ids'][i] for i in state['masked_positions'])
    assert source['confidence'] == state['confidence']
    assert result['forward_calls'] <= cap and sum(result['forward_types'].values()) == result['forward_calls']
    prefix = [e for e in reference_events if e['forward'] < state['source_forward']]
    events = prefix+suffix
    assert len(events) == result['forward_calls']
    commits, previous = dict(enumerate(state['prompt_ids'])), None
    counts = Counter()
    denoise = 0
    for e in events:
        if previous is not None and e['block_start'] == previous['block_start']:
            expected = list(previous['input_ids'])
            for i, token in zip(previous['positions'], previous['tokens']):
                expected[i] = token
            assert e['input_ids'] == expected and e['mask_before'] == previous['mask_after']
        masked = [i for i, flag in enumerate(e['mask_before']) if flag == 0]
        assert e['positions'] == [i for i in masked if e['mask_after'][i] == 1]
        assert all(e['mask_after'][i] == e['mask_before'][i] or
                   (e['mask_before'][i], e['mask_after'][i]) == (1, 2)
                   for i in range(32) if i not in e['positions'])
        for i, token in zip(e['positions'], e['tokens']):
            absolute = e['block_start']+i
            assert absolute not in commits
            commits[absolute] = token
        for i, flag in enumerate(e['mask_before']):
            if flag != 0:
                assert commits[e['block_start']+i] == e['input_ids'][i]
        if masked:
            denoise += 1
        if e['forward'] > state['source_forward'] and masked:
            p, boundary = e['confidence'], e['effective_threshold']
            expected = [i for i in masked if p[i] >= boundary]
            assert e['positions'] == expected if expected else (
                len(e['positions']) == 1 and p[e['positions'][0]] == max(p[i] for i in masked))
            assert all(token == e['current_ids'][i] for i, token in zip(e['positions'], e['tokens']))
        kind = e['compute_kind'] if e['compute_kind'] == 'prefill' else ('decode' if masked else 'cache')
        counts[('restore_' if e['forward'] <= state['source_forward'] else '')+kind] += 1
        previous = e
    assert dict(counts) == result['forward_types']
    assert [commits[len(state['prompt_ids'])+i] for i in range(len(result['token_ids']))] == result['token_ids']
    return {'passed': True, 'events': len(events), 'denoise_forwards': denoise,
            'block': block_metrics(suffix, state, result['finish_reason'])}


def compare_current(result, row, events, state, cap):
    reference = capped_reference(row, events, cap)
    expected = [e for e in reference['events'] if e['forward'] >= state['source_forward']]
    difference = trace_difference(expected, result['events'])
    assert difference is None, difference
    assert result['token_ids'] == reference['token_ids']
    assert result['finish_reason'] == reference['finish_reason']
    assert result['forward_calls'] == reference['forward_calls']


def worker(gpu, out, stage):
    from joint50 import environment, score
    environment(gpu)
    Backend = backend_class()
    from lmdeploy.utils import get_logger
    get_logger('lmdeploy').setLevel('ERROR')
    out = Path(out)
    manifest = read(out/'manifest.json')
    cases = [c for c in read(out/'cases.json') if c['gpu'] == gpu]
    if stage == 'pilot':
        cases = [c for c in cases if c['question_id'] in manifest['pilot_ids']]
    if stage == 'check':
        cases = cases[:1]
    if not cases:
        return
    directory = out/('checks' if stage == 'check' else 'branches')
    directory.mkdir(exist_ok=True)
    path = directory/f'gpu{gpu}.jsonl'
    old = read_jsonl(path) if path.exists() else []
    done = {(r['case_id'], r['action']['id'], r['repeat']) for r in old}
    data = {q['unique_id']: q for q in read(out/'data.json')}
    rows, traces = sources(), {}
    backend = Backend(manifest['engine'])
    grader = Grader(manifest['engine']['grade_timeout'])
    try:
        q = data[cases[0]['question_id']]
        backend.generate([{'id': f'warm-{gpu}', 'question': q['problem'], 'max_new_tokens': 96}])
        for case in cases:
            qid, state = case['question_id'], case['snapshot']
            if qid not in traces:
                traces[qid] = events_for(rows[qid])
            current = next(a for a in case['actions'] if 'Current' in a['policies'])
            others = [a for a in case['actions'] if a is not current]
            random.Random(stable_seed(case['id'], manifest['study']['seed'])).shuffle(others)
            jobs = [(current, 0)]+[(a, 0) for a in others]
            cap = manifest['study']['forward_cap']
            if stage == 'check':
                cap = min(cap, state['source_forward']+32)
                ordered = sorted(others, key=lambda a: a['k'])
                jobs = [(current, 0), (ordered[0], 0), (ordered[-1], 0), (current, 1)]
                jobs = list({(a['id'], repeat): (a, repeat) for a, repeat in jobs}.values())
            for action, repeat in jobs:
                if (case['id'], action['id'], repeat) in done:
                    continue
                assert action is current or (case['id'], current['id'], 0) in done
                name = f"{case['id']}-{action['id']}-{repeat}"
                job = {'id': name, 'snapshot': state, 'action': action,
                       'future_budget': cap-state['source_forward']}
                result = backend.generate([job])[0]
                seconds = backend.last_seconds
                audit = audit_path(result, case, action, traces[qid], cap)
                if action is current:
                    compare_current(result, rows[qid], traces[qid], state, cap)
                trace = directory/f'{name}.jsonl.gz'
                with gzip.open(trace, 'wt', compresslevel=1) as f:
                    for event in result.pop('events'):
                        f.write(json.dumps(event, separators=(',', ':'))+'\n')
                result.pop('state')
                result.update(score(grader, result, data[qid]['answer']))
                result.update(case_id=case['id'], question_id=qid, window=case['window'],
                    action=action, repeat=repeat, gpu=gpu, stage=stage, seconds=seconds,
                    trace_path=str(trace), forward_cap=cap, validation=audit,
                    source_forward=state['source_forward'],
                    exact_repeat=bool(repeat_runs(result['token_ids'])),
                    tpf_denoise=len(result['token_ids'])/audit['denoise_forwards'])
                with path.open('a') as f:
                    f.write(json.dumps(result, ensure_ascii=False, separators=(',', ':'))+'\n')
                done.add((case['id'], action['id'], repeat))
                print(json.dumps({'stage': stage, 'gpu': gpu, 'state': case['id'], 'k': action['k'],
                    'reward': result['reward'], 'forward': result['forward_calls']}), flush=True)
    finally:
        grader.close()
        backend.close()


def parallel(out, stage):
    manifest = read(out/'manifest.json')
    for name, digest in manifest['code'].items():
        assert file_hash(GA/name) == digest, name
    for name, digest in manifest['dependencies'].items():
        assert file_hash(GA/name) == digest, name
    assert file_hash(out/'cases.json') == manifest['cases_sha256']
    if stage == 'run':
        pilot = read(out/'pilot.json')
        assert pilot['complete'] and pilot['all_current_restored']
    if stage in ('pilot', 'run'):
        assert len(list((out/'checks').glob('gpu*.jsonl'))) == len(manifest['study']['gpus'])
    with ProcessPoolExecutor(len(manifest['study']['gpus']), mp_context=mp.get_context('spawn')) as pool:
        futures = [pool.submit(worker, gpu, str(out), stage) for gpu in manifest['study']['gpus']]
        for future in futures:
            future.result()
    if stage == 'pilot':
        expected = {(c['id'], a['id']) for c in read(out/'cases.json')
                    if c['question_id'] in manifest['pilot_ids'] for a in c['actions']}
        actual = {(r['case_id'], r['action']['id']) for r in records(out)}
        assert expected <= actual
        write_json(out/'pilot.json', {'complete': True, 'all_current_restored': True, 'actions': len(expected)})


def records(out, directory='branches'):
    return [r for p in sorted((out/directory).glob('gpu*.jsonl')) for r in read_jsonl(p)]


def outcome(current, alternative):
    if alternative['reward'] > current['reward']:
        return 'repair'
    if alternative['reward'] < current['reward']:
        return 'harm'
    if current['reward'] == 0:
        return 'both_wrong'
    return 'safe_save' if alternative['forward_calls'] < current['forward_calls'] else 'correct_no_save'


def choose_action(case, predictions):
    """Current has exact relative value (0,0); all other values are predictions."""
    k0 = len(case['snapshot']['positions'])
    return max(case['actions'], key=lambda a: (*predictions[a['id']][:1],
        -predictions[a['id']][1], -abs(a['k']-k0), -a['k']))['id']


def forecast(cases, rows, manifest, out):
    sys.path.insert(0, str(GA/'cache/analysis'))
    from sklearn.tree import DecisionTreeRegressor, export_text
    lookup = {(r['case_id'], r['action']['id']): r for r in rows}
    selected, models, predictions = {}, {}, []
    for name, columns in manifest['feature_groups'].items():
        selected[name], models[name] = {}, []
        for fold in range(5):
            train = [c for c in cases if manifest['folds'][c['question_id']] != fold]
            test = [c for c in cases if manifest['folds'][c['question_id']] == fold]
            assert {c['question_id'] for c in train}.isdisjoint(c['question_id'] for c in test)
            states_per_q = Counter(c['question_id'] for c in train)
            x, target, weight = [], [], []
            for case in train:
                current = lookup[case['id'], f"k{len(case['snapshot']['positions']):02d}"]
                alternatives = [a for a in case['actions'] if a['k'] != current['action']['k']]
                for a in alternatives:
                    row = lookup[case['id'], a['id']]
                    x.append([case['features'][a['id']][f] for f in columns])
                    target.append([row['reward']-current['reward'],
                                   (row['forward_calls']-current['forward_calls'])/manifest['study']['forward_cap']])
                    weight.append(1/states_per_q[case['question_id']]/len(alternatives))
            learners = []
            for head in range(2):
                tree = DecisionTreeRegressor(max_depth=manifest['study']['tree_depth'],
                    min_samples_leaf=manifest['study']['tree_min_leaf'], random_state=manifest['study']['seed'])
                tree.fit(x, np.asarray(target)[:, head], sample_weight=weight)
                learners.append(tree)
            models[name].append({'fold': fold, 'train_ids': sorted({c['question_id'] for c in train}),
                'test_ids': sorted({c['question_id'] for c in test}),
                'reward_tree': export_text(learners[0], feature_names=columns),
                'cost_tree': export_text(learners[1], feature_names=columns)})
            for case in test:
                estimate = {}
                for a in case['actions']:
                    if 'Current' in a['policies']:
                        estimate[a['id']] = [0., 0.]
                    else:
                        values = [[case['features'][a['id']][f] for f in columns]]
                        estimate[a['id']] = [float(tree.predict(values)[0]) for tree in learners]
                selected[name][case['id']] = choose_action(case, estimate)
                predictions.append({'model': name, 'fold': fold, 'case_id': case['id'],
                                    'predictions': estimate, 'chosen': selected[name][case['id']]})
    write_json(out/'prediction_models.json', models)
    write_json(out/'predictions.json', predictions)
    return selected


def compare_policy(cases, selected, lookup, study):
    by_q = defaultdict(list)
    for case in cases:
        current = lookup[case['id'], f"k{len(case['snapshot']['positions']):02d}"]
        row = lookup[case['id'], selected[case['id']]]
        by_q[case['question_id']].append([current['reward'], row['reward'],
            current['forward_calls'], row['forward_calls'], current['seconds'], row['seconds'],
            outcome(current, row) == 'repair', outcome(current, row) == 'harm',
            outcome(current, row) == 'safe_save', row['exact_repeat'], row['finish_reason'] == 'forward_budget',
            len(row['token_ids']), row['validation']['denoise_forwards']])
    values = np.asarray([np.mean(v, axis=0) for q, v in sorted(by_q.items())])
    avg = values.mean(axis=0)
    rng = np.random.default_rng(study['seed'])
    sample = values[rng.integers(0, len(values), size=(study['bootstrap_samples'], len(values)))].mean(axis=1)
    return {'questions': len(values), 'states': len(cases), 'reward': float(avg[1]),
        'current_reward': float(avg[0]), 'delta_reward': float(avg[1]-avg[0]),
        'delta_reward_ci95': np.quantile(sample[:,1]-sample[:,0], [.025,.975]).tolist(),
        'forward_mean': float(avg[3]), 'current_forward_mean': float(avg[2]),
        'forward_change': float(avg[3]/avg[2]-1),
        'forward_change_ci95': np.quantile(sample[:,3]/sample[:,2]-1, [.025,.975]).tolist(),
        'seconds_mean': float(avg[5]), 'repair': float(avg[6]), 'harm': float(avg[7]),
        'safe_save': float(avg[8]), 'repeat': float(avg[9]), 'budget_exhaustion': float(avg[10]),
        'tokens_mean': float(avg[11]), 'denoise_mean': float(avg[12]), 'tpf_denoise': float(avg[11]/avg[12])}


def report(out):
    manifest, cases, rows = read(out/'manifest.json'), read(out/'cases.json'), records(out)
    lookup = {(r['case_id'], r['action']['id']): r for r in rows}
    expected = {(c['id'], a['id']) for c in cases for a in c['actions']}
    assert len(rows) == len(lookup) and set(lookup) == expected
    policies = {p: {c['id']: f'k{c["policies"][p]:02d}' for c in cases} for p in cases[0]['policies']}
    for pool in ('MainOracle', 'AllOracle'):
        policies[pool] = {}
        for case in cases:
            actions = [a for a in case['actions'] if pool == 'AllOracle' or
                       any(p.startswith('D') or p == 'Current' for p in a['policies'])]
            best = max(actions, key=lambda a: (lookup[case['id'],a['id']]['reward'],
                -lookup[case['id'],a['id']]['forward_calls'], -abs(a['k']-len(case['snapshot']['positions']))))
            policies[pool][case['id']] = best['id']
    policies.update({f'Tree{name}': choice for name, choice in forecast(cases, rows, manifest, out).items()})
    summaries = {}
    for scope, subset in [('primary', [c for c in cases if c['window'] == 0]), ('all_states', cases)]+[
            (f'window{w}', [c for c in cases if c['window'] == w]) for w in range(1,3)]:
        summaries[scope] = {name: compare_policy(subset, choice, lookup, manifest['study'])
                            for name, choice in policies.items()}
    interventions = []
    for case in cases:
        current = lookup[case['id'], f"k{len(case['snapshot']['positions']):02d}"]
        for action in case['actions']:
            row = lookup[case['id'], action['id']]
            cb, rb = current['validation']['block'], row['validation']['block']
            interventions.append({'case_id': case['id'], 'question_id': case['question_id'],
                'window': case['window'], 'action_id': action['id'], 'k': action['k'],
                'delta_k': action['k']-current['action']['k'], 'outcome': outcome(current,row),
                'delta_reward': row['reward']-current['reward'],
                'delta_forward': row['forward_calls']-current['forward_calls'],
                'delta_block_steps': rb['extra_denoise']-cb['extra_denoise'] if rb['complete'] and cb['complete'] else None,
                'block_complete': rb['complete'], 'contains_eos': any(t in read(
                    Path(manifest['engine']['model'])/'generation_config.json')['eos_token_id'] for t in action['tokens']),
                'features': case['features'][action['id']]})
    checks = records(out, 'checks')
    costs = {'formal_paths': len(rows), 'formal_forwards': sum(r['forward_calls'] for r in rows),
        'formal_restore_forwards': sum(r['restore_forwards'] for r in rows),
        'formal_request_seconds': sum(r['seconds'] for r in rows), 'check_paths': len(checks),
        'check_forwards': sum(r['forward_calls'] for r in checks),
        'warmup': 'One max96-token request per worker invocation, excluded from listed formal/check forwards.',
        'timing': 'Sum of request seconds, compact recording; excludes disk, grading, model loading. Not eight-GPU wall time.'}
    write_json(out/'interventions.json', interventions)
    write_json(out/'selected_actions.json', policies)
    write_json(out/'summary.json', {'coverage': {'questions': 50, 'states': len(cases),
        'windows': dict(Counter(c['window'] for c in cases)), 'actions': len(rows)},
        'scopes': summaries, 'costs': costs,
        'outcome_questions': {label: sorted({r['question_id'] for r in interventions if r['outcome'] == label})
                              for label in ('repair','harm','safe_save')},
        'grading': dict(Counter(r['grading'] for r in rows))})
    plot(out, summaries['primary'], interventions)
    primary = summaries['primary']
    lines = ['# Experiment T3：Marginal Commit Utility Study', '',
        '日期：2026-11-12。原有50道开发题，Dynamic .90自然状态，当前top-1固定；单步改变提交数量后继续Dynamic .90。', '',
        f'共{len(cases)}个状态、{len(rows)}个唯一动作。512次整题forward上限包含恢复重放、源forward、去噪和缓存写入；8192-token上限。', '',
        '## 主结果：每题最早区间一个状态', '',
        'Current、固定数量偏移和简单数量规则是可事前执行的单次干预；Oracle为事后上界。浅树使用按题五折的折外选择；同题全部区间都在同一折。', '',
        '| 规则 | 正确 / 50 | 平均整题forward | 相对Current | 新增 / 损失正确 | 去噪TPF |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for name, r in primary.items():
        lines.append(f"| {name} | {r['reward']*50:.0f}/50 | {r['forward_mean']:.1f} | {r['forward_change']:.1%} | {r['repair']*50:.0f} / {r['harm']*50:.0f} | {r['tpf_denoise']:.2f} |")
    lines += ['', '## 口径与解释', '',
        '- .85/.95/factor1只决定源状态当前一步的k，采用同一嵌套排序，随后仍回到.90；不是完整策略的复现成绩。',
        '- 两个错误路径之间的计算减少不记作保持质量的收益；完整标签及当前块步数变化见interventions.json。',
        '- 其他区间和合并结果见summary.json。合并结果先在题内平均状态，再对题求平均和bootstrap；不能当成实际多次干预策略的正确率。',
        '- 浅树固定depth=2、min_leaf=12，分别预测奖励差和forward差。优先最大预测奖励差，再比较预测成本；Current相对值固定为(0,0)。未按测试折结果调参。',
        '- 本轮仅使用原开发50题；已确认50题和最终400题未进入本轮。预测结果仍是开发数据上的分组交叉验证，不能代替独立确认。', '',
        '## 计算与验收', '',
        f"- 正式{len(rows)}条路径，{costs['formal_forwards']:,}次forward，其中恢复及源forward为{costs['formal_restore_forwards']:,}次。",
        f"- 短GPU验收{len(checks)}条、{costs['check_forwards']:,}次forward；不计入正式样本。每次worker的warmup另计。",
        f"- 正式请求秒数合计{costs['formal_request_seconds']:.2f}；不是八卡墙钟时间或独占峰值性能。",
        '- 每个状态的Current均复现原路径，全部动作检查源输入、固定top-1、嵌套集合、下一轮输入、未来.90规则、最终token重建和计数。独立复查见audit.json。', '',
        '![主结果与提交效用](overview.png)', '',
        '完整数据：[summary.json](summary.json)、[interventions.json](interventions.json)、[predictions.json](predictions.json)、[prediction_models.json](prediction_models.json)。', '']
    (out/'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps({'primary': {n: [r['reward']*50, r['forward_mean']] for n,r in primary.items()},
                      'costs': costs}, ensure_ascii=False))


def plot(out, primary, interventions):
    sys.path.insert(0, str(GA/'cache/analysis'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12,4.4), constrained_layout=True)
    for name, r in primary.items():
        axes[0].scatter(r['forward_mean'], r['reward']*100)
        axes[0].annotate(name, (r['forward_mean'],r['reward']*100), xytext=(4,4), textcoords='offset points', fontsize=7)
    axes[0].set(xlabel='Mean total forwards', ylabel='Accuracy (%)', title='One intervention per question: first window')
    colors = {'repair':'tab:green', 'harm':'tab:red', 'safe_save':'tab:blue',
              'both_wrong':'gray', 'correct_no_save':'orange'}
    for label, color in colors.items():
        group = [r for r in interventions if r['delta_k'] > 0 and r['outcome'] == label]
        axes[1].scatter([r['features']['added_p_min'] for r in group],
            [r['delta_forward'] for r in group], c=color, label=label, s=15, alpha=.6)
    axes[1].axhline(0,color='black',lw=.7)
    axes[1].set(xlabel='Minimum confidence of added positions', ylabel='Delta total forwards',
                title='Advancing commits: measured quality and cost')
    axes[1].legend(fontsize=7)
    fig.savefig(out/'overview.png',dpi=180)
    fig.savefig(out/'overview.pdf')
    plt.close(fig)


def audit(out):
    from run import fingerprint
    manifest, cases = read(out/'manifest.json'), {c['id']: c for c in read(out/'cases.json')}
    assert fingerprint(read(GA/'runs/pilot8192-20261110/manifest.json')['config']) == read(out/'original_fingerprint.json')
    rows, traces = sources(), {}
    records_by_stage = {name: records(out,name) for name in ('branches','checks')}
    checked = Counter()
    for stage, paths in records_by_stage.items():
        for row in paths:
            case = cases[row['case_id']]
            qid = case['question_id']
            if qid not in traces:
                traces[qid] = events_for(rows[qid])
            result = dict(row, events=read_jsonl(row['trace_path']))
            value = audit_path(result,case,row['action'],traces[qid],row['forward_cap'])
            assert value == row['validation']
            if 'Current' in row['action']['policies']:
                compare_current(result, rows[qid], traces[qid], case['snapshot'], row['forward_cap'])
                checked['current_restorations'] += 1
            checked[stage] += 1
    assert checked['branches'] == sum(len(c['actions']) for c in cases.values())
    write_json(out/'audit.json', {'passed': True, **checked, 'original_sources_model_data_unchanged': True})
    print(json.dumps(dict(checked)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare','check','pilot','run','report','audit'))
    parser.add_argument('--out', type=Path, default=OUT)
    args = parser.parse_args()
    if args.stage in ('check','pilot','run'):
        parallel(args.out, args.stage)
    else:
        globals()[args.stage](args.out)


if __name__ == '__main__':
    main()
