"""One early commit/swap/delay/reopen intervention under a strict forward cap."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys

from early import repetition, snapshot
from protocol import (Grader, file_hash, load_data, paired_stats, read_jsonl,
                      restoration_difference, write_json)
from run import fingerprint

GA = Path(__file__).resolve().parent
POLICIES = ('Current', 'Swap1', 'Delay1', 'Reopen1', 'Reopen2')
FORWARD_CAP, EARLY_TOKENS, CONFIDENCE = 512, 256, 0.9


def risk(event, position):
    """Only masked, pre-commit predictions are used as evidence."""
    historical = event['historical_ids']
    changed = historical is not None and historical[position] != event['current_ids'][position]
    return int(changed), 1 - event['topk_probs'][position][0], -position


def select_source(events, prompt_length, steps):
    for event in events:
        if (event['phase'] != 'denoise' or not event['full_generation_block']
                or not 2 <= event['round'] < steps
                or event['block_start'] - prompt_length >= EARLY_TOKENS):
            continue
        if any(risk(event, i)[0] or event['topk_probs'][i][0] < CONFIDENCE for i in event['positions']):
            return event
    return None


def make_actions(source, preceding):
    selected = source['positions']
    risky = max(selected, key=lambda i: risk(source, i))
    available = [i for i, flag in enumerate(source['mask_before']) if flag == 0 and i not in selected]
    alternative = min(available, key=lambda i: (risk(source, i)[:2], i))
    committed = {}
    for event in preceding:
        if event['block_start'] == source['block_start']:
            for i in event['positions']:
                committed[i] = {'position': i, 'forward': event['forward'],
                    'probability': event['topk_probs'][i][0], 'changed': bool(risk(event, i)[0])}
    suspects = sorted(committed, key=lambda i: (-committed[i]['changed'], committed[i]['probability'], i))[:2]
    assert len(suspects) == 2
    actions = []
    for policy in POLICIES:
        positions = list(selected)
        if policy in ('Swap1', 'Delay1'):
            positions.remove(risky)
        if policy == 'Swap1':
            positions.append(alternative)
        positions.sort()
        reopen = suspects[:int(policy[-1])] if policy.startswith('Reopen') else []
        actions.append({'name': policy, 'positions': positions,
            'tokens': [source['current_ids'][i] for i in positions], 'reopen': reopen,
            'deferred_position': risky if policy in ('Swap1', 'Delay1') else None,
            'replacement_position': alternative if policy == 'Swap1' else None,
            'reopen_evidence': [committed[i] for i in reopen]})
    return actions


def capped_reference(row, events):
    events = [e for e in events if e['forward'] <= FORWARD_CAP]
    if row['forward_calls'] <= FORWARD_CAP:
        return dict(row, events=events)
    prompt_length = len(row['state']['prompt_ids'])
    valid_end = max([prompt_length] + [e['block_start'] + 32 for e in events if 0 not in e['mask_after']])
    return dict(row, events=events, token_ids=row['token_ids'][:valid_end - prompt_length],
                finish_reason='forward_budget', forward_calls=FORWARD_CAP, finished=False)


def action_backend():
    from engine import Backend

    class ActionBackend(Backend):
        def _install(self):
            super()._install()
            original_post = self.agent.agent_strategy.post_sampling
            processor = self.agent.agent_strategy.unmasking_processor
            original_static = processor.low_confidence_static

            def static(logits, tokens, mask):
                track, base = self.active
                state = track.job.get('snapshot')
                compensate = track.job.get('policy') in ('Delay1', 'Reopen1', 'Reopen2')
                if (compensate and track.source_reached and base == state['block_start']
                        and track.rounds[base] + 1 == self.setting['steps']):
                    # Keep the original number of rounds: the final round pays the commit debt.
                    self.final_fill = bool((mask == 0).any())
                    return mask.masked_fill(mask == 0, 1)
                return original_static(logits, tokens, mask)

            self._replace(processor, 'low_confidence_static', static)

            def post_sampling(inputs, logits, next_ids, extra):
                assert len(inputs.model_metas) == 1
                track = self.tracks[inputs.model_metas[0]['ga_session']]
                base = int((inputs.history_lengths + inputs.seq_length - self.block)[0])
                self.active, self.final_fill = (track, base), False
                at_source = track.source_pending and len(track.events) == len(track.job['snapshot']['replay'])
                previous_end = track.valid_end
                writer, self.writer = self.writer, None
                try:
                    output, extra = original_post(inputs, logits, next_ids, extra)
                finally:
                    self.writer = writer
                event = track.events[-1]
                event['final_round_fill'] = self.final_fill
                if at_source:
                    state, reopen = track.job['snapshot'], track.job['reopen']
                    event.update(policy=track.job['policy'], reopened_positions=reopen)
                    mask_id = state['input_ids'][state['masked_positions'][0]]
                    for i in reopen:
                        assert state['mask'][i] == 1 and int(extra.dllm_mask[i]) == 1
                        extra.dllm_mask[i], output[i] = 0, mask_id
                        event['mask_after'][i] = 0
                        track.token_map.pop(base + i)
                    if reopen:
                        track.valid_end = previous_end
                if writer:
                    writer.write(json.dumps(event, separators=(',', ':')) + '\n')
                return output, extra

            self._replace(self.agent.agent_strategy, 'post_sampling', post_sampling)

    return ActionBackend


def audit(result, events, case, action):
    state, steps = case['snapshot'], int(case['setting'][1:])
    source = events[0]
    assert source['forward'] == state['source_forward']
    assert source['input_ids'] == state['input_ids'] and source['mask_before'] == state['mask']
    assert source['positions'] == action['positions'] and source['tokens'] == action['tokens']
    expected_input, expected_mask = list(state['input_ids']), list(state['mask'])
    for i, token in zip(action['positions'], action['tokens']):
        expected_input[i], expected_mask[i] = token, 1
    mask_id = state['input_ids'][state['masked_positions'][0]]
    for i in action['reopen']:
        expected_input[i], expected_mask[i] = mask_id, 0
    assert source['mask_after'] == expected_mask
    assert events[1]['input_ids'] == expected_input and events[1]['mask_before'] == expected_mask
    assert result['restore_forwards'] == state['source_forward']
    assert result['forward_calls'] == result['restore_forwards'] + result['future_forwards'] <= FORWARD_CAP
    assert sum(result['forward_types'].values()) == result['forward_calls']
    transitions = 0
    for event in events:
        if event['phase'] == 'denoise':
            assert event['round'] <= steps
        for i, token in zip(event['positions'], event['tokens']):
            assert event['mask_before'][i] == 0 and token == event['current_ids'][i]
        if event['block_start'] == state['block_start'] and event['phase'] == 'denoise':
            assert len(event['positions']) == (len(action['positions']) if event is source else
                sum(flag == 0 for flag in event['mask_before']) if event['round'] == steps else 32 // steps)
        for i, flag in enumerate(event['mask_before']):
            if flag == 1 and not (event is source and i in action['reopen']):
                assert event['mask_after'][i] != 0
    for left, right in zip(events, events[1:]):
        if left['block_start'] != right['block_start']:
            continue
        expected = list(left['input_ids'])
        for i, token in zip(left['positions'], left['tokens']):
            expected[i] = token
        for i in left.get('reopened_positions', []):
            expected[i] = mask_id
        assert right['input_ids'] == expected and right['mask_before'] == left['mask_after']
        transitions += 1
    block_events = [e for e in events if e['block_start'] == state['block_start'] and e['phase'] == 'denoise']
    assert block_events[-1]['round'] == steps and 0 not in block_events[-1]['mask_after']
    return {'passed': True, 'source_action_and_next_input': True, 'checked_block_transitions': transitions,
            'source_block_final_round': block_events[-1]['round'],
            'total_forward_cap': FORWARD_CAP, 'total_forwards': result['forward_calls']}


def worker(gpu, config, cases, run):
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu), PYTHONDONTWRITEBYTECODE='1',
        HF_HOME=str(GA / 'cache/hf'), HF_MODULES_CACHE=str(GA / 'cache/hf_modules'),
        TRITON_CACHE_DIR=str(GA / 'cache/triton'), TORCHINDUCTOR_CACHE_DIR=str(GA / 'cache/inductor'),
        SDAR_FULL_VOCAB='1', TOKENIZERS_PARALLELISM='false')
    sys.dont_write_bytecode = True
    backend_type = action_backend()
    from lmdeploy.utils import get_logger
    get_logger('lmdeploy').setLevel('ERROR')
    backend = backend_type(config, next(s for s in config['settings'] if s['name'] == cases[0]['setting']))
    grader = Grader(config['grade_timeout'])
    work, out = Path(run) / 'pilot', Path(run) / 'action_value'
    rows = {(r['setting'], r['question_id']): r for r in json.loads((work / f'exp0-gpu{gpu}.json').read_text())}
    questions = {r['unique_id']: r for r in load_data(config['data'], 'pilot')}
    traces = {}
    for setting in ('T8', 'T4'):
        traces[setting] = defaultdict(list)
        for event in read_jsonl(work / f'exp0-{setting}-gpu{gpu}.trace.jsonl.gz'):
            traces[setting][event['request_id']].append(event)
    records = []
    try:
        for case in cases:
            q, setting, state = case['question_id'], case['setting'], case['snapshot']
            backend.set_setting(next(s for s in config['settings'] if s['name'] == setting))
            reference = capped_reference(rows[setting, q], traces[setting][q])
            source_ok = False
            for action in case['actions']:
                name = f"{setting}-{q.replace('/', '_').replace('.json', '')}-{action['name']}"
                job = {'id': name, 'snapshot': state, 'action': action, 'policy': action['name'],
                       'reopen': action['reopen'], 'future_budget': FORWARD_CAP - state['source_forward']}
                assert action['name'] == 'Current' or source_ok
                result = backend.generate([job], out / f'{name}.trace.jsonl.gz')[0]
                seconds = backend.last_seconds
                events = result.pop('events')
                result.pop('state')
                checks = audit(result, events, case, action)
                if action['name'] == 'Current':
                    difference = restoration_difference(reference, dict(result, events=events), state)
                    assert difference is None, difference
                    checks['restoration_difference'] = difference
                    source_ok = True
                result.update(question_id=q, setting=setting, policy=action['name'], gpu=gpu,
                    action=action, source_forward=state['source_forward'], source_block=state['block_start'],
                    source_round=state['round'], forward_cap=FORWARD_CAP, validation=checks,
                    inference_seconds=seconds, repetition=repetition(result['token_ids'], max_period=256))
                result.update(grader.score(result['text'], questions[q]['answer'], result['finished']))
                records.append(result)
                write_json(out / f'results-gpu{gpu}.json', records)
                print(json.dumps({k: result[k] for k in ('question_id', 'setting', 'policy',
                    'reward', 'finish_reason', 'forward_calls')}) , flush=True)
    finally:
        grader.close()
        backend.close()
    return {'gpu': gpu, 'continuations': len(records)}


def prepare(run):
    out, work = run / 'action_value', run / 'pilot'
    out.mkdir(exist_ok=True)
    assert not (out / 'manifest.json').exists(), 'Use the saved manifest for running or reporting.'
    parent = json.loads((run / 'manifest.json').read_text())
    config = parent['config']
    assert fingerprint(config)['digest'] == parent['fingerprint']['digest']
    pilot = {r['unique_id'] for r in load_data(config['data'], 'pilot')}
    cases, missing = [], []
    for gpu in config['gpus']:
        rows = json.loads((work / f'exp0-gpu{gpu}.json').read_text())
        for setting in ('T8', 'T4'):
            grouped = defaultdict(list)
            for event in read_jsonl(work / f'exp0-{setting}-gpu{gpu}.trace.jsonl.gz'):
                grouped[event['request_id']].append(event)
            for row in rows:
                if row['setting'] != setting:
                    continue
                q, events = row['question_id'], grouped[row['question_id']]
                assert q in pilot
                event = select_source(events, len(row['state']['prompt_ids']), int(setting[1:]))
                if event is None:
                    missing.append({'question_id': q, 'setting': setting})
                    continue
                state = snapshot(row, events, event['forward'])
                actions = make_actions(event, [e for e in events if e['forward'] < event['forward']])
                cases.append({'question_id': q, 'gpu': gpu, 'setting': setting, 'snapshot': state, 'actions': actions})
    write_json(out / 'cases.json', cases)
    sources = list(work.glob('exp0-gpu*.json')) + list(work.glob('exp0-T[48]-gpu*.trace.jsonl.gz'))
    manifest = {'parent_fingerprint': parent['fingerprint']['digest'], 'config': config,
        'code': {p.name: file_hash(p) for p in (Path(__file__), GA / 'early.py')},
        'source_files': {str(p.relative_to(run)): file_hash(p) for p in sources},
        'cases_sha256': file_hash(out / 'cases.json'), 'scope': 'original 10 development pilot questions',
        'source_rule': {'max_early_generated_positions': EARLY_TOKENS, 'minimum_round': 2,
            'exclude_final_round': True, 'commit_confidence_threshold': CONFIDENCE,
            'trigger': 'First eligible full block round with a selected top-1 flip or confidence below threshold.'},
        'actions': {'Current': 'Original positions and current top-1.',
            'Swap1': 'Swap the riskiest selected position for the least risky unselected masked position.',
            'Delay1': 'Omit the riskiest selected position this round; later rounds may commit it.',
            'Reopen1': 'Mask the riskiest previously committed active-block position after Current.',
            'Reopen2': 'Mask the two riskiest previously committed active-block positions after Current.'},
        'risk_order': 'Top-1 changed since preceding masked proposal, then lower commit confidence, then smaller position. Previously committed positions use their stored pre-commit evidence only.',
        'budget': {'total_forward_cap': FORWARD_CAP, 'includes_replay_prefill_cache': True,
            'block_rounds': 'Keep T8/T4; fill all remaining holes in the final source-block round.',
            'max_new_tokens': config['max_new_tokens'], 'scoring_model_calls': 0,
            'early_eos': True, 'one_intervention_per_answer': True},
        'reporting': 'Each policy separately; best of the tested actions is diagnostic only, not a deployed-policy result.',
        'planned_continuations': len(cases) * len(POLICIES), 'missing_sources': missing}
    write_json(out / 'manifest.json', manifest)
    print(json.dumps({'prepared': str(out), 'states': len(cases), 'missing': missing,
                      'continuations': manifest['planned_continuations']}, ensure_ascii=False))


def report(run):
    out = run / 'action_value'
    manifest = json.loads((out / 'manifest.json').read_text())
    rows = [r for p in out.glob('results-gpu*.json') for r in json.loads(p.read_text())]
    summary = {'completed': len(rows), 'planned': manifest['planned_continuations'],
        'total_forwards': sum(r['forward_calls'] for r in rows),
        'restore_forwards': sum(r['restore_forwards'] for r in rows),
        'inference_seconds': sum(r['inference_seconds'] for r in rows), 'settings': {}}
    for setting in ('T8', 'T4'):
        group = [r for r in rows if r['setting'] == setting]
        current = {r['question_id']: r for r in group if r['policy'] == 'Current'}
        stats = {}
        for policy in POLICIES:
            results = [r for r in group if r['policy'] == policy]
            if not results:
                continue
            stats[policy] = {'n': len(results), 'correct': sum(r['reward'] for r in results),
                'finish_reasons': dict(Counter(r['finish_reason'] for r in results)),
                'mean_forwards': sum(r['forward_calls'] for r in results) / len(results),
                'mean_generated_tokens': sum(len(r['token_ids']) for r in results) / len(results),
                'repetitions': sum(r['repetition'] is not None for r in results),
                'changed_full_path': sum(r['token_ids'] != current[r['question_id']]['token_ids'] for r in results),
                'paired': paired_stats({q: r['reward'] for q, r in current.items()},
                                       {r['question_id']: r['reward'] for r in results}),
                'new_correct': [r['question_id'] for r in results if r['reward'] > current[r['question_id']]['reward']],
                'new_wrong': [r['question_id'] for r in results if r['reward'] < current[r['question_id']]['reward']]}
        diagnostic = {q: max(r['reward'] for r in group if r['question_id'] == q) for q in current}
        summary['settings'][setting] = {'policies': stats, 'diagnostic_best_of_five_correct': sum(diagnostic.values()),
            'diagnostic_rescued_questions': [q for q, reward in diagnostic.items() if reward > current[q]['reward']]}
    summary['all_validated'] = all(r['validation']['passed'] for r in rows)
    summary['grade_statuses'] = dict(Counter(r['grading'] for r in rows))
    write_json(out / 'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--stage', choices=('prepare', 'run', 'report'), required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    if args.stage == 'prepare':
        prepare(run)
    elif args.stage == 'report':
        report(run)
    else:
        out = run / 'action_value'
        manifest = json.loads((out / 'manifest.json').read_text())
        assert not list(out.glob('results-gpu*.json'))
        assert all(file_hash(GA / name) == digest for name, digest in manifest['code'].items())
        assert file_hash(out / 'cases.json') == manifest['cases_sha256']
        assert fingerprint(manifest['config'])['digest'] == manifest['parent_fingerprint']
        cases = json.loads((out / 'cases.json').read_text())
        gpus = sorted({case['gpu'] for case in cases})
        with ProcessPoolExecutor(max_workers=len(gpus), mp_context=mp.get_context('spawn')) as executor:
            futures = [executor.submit(worker, gpu, dict(manifest['config']),
                [c for c in cases if c['gpu'] == gpu], run) for gpu in gpus]
            for future in futures:
                print(future.result(), flush=True)
        report(run)


if __name__ == '__main__':
    main()
