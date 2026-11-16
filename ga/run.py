"""One entry point for checks, baseline traces, restoration, and paired interventions."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import hashlib
import importlib.metadata
import json
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo

from protocol import (ROOT, Grader, action_pool, divergence_summary, file_hash, forward_budget,
                      load_data, oracle_summary, paired_stats, read_jsonl,
                      restoration_difference, trace_difference, write_json)

GA = Path(__file__).resolve().parent


def settings(config):
    return {item['name']: item for item in config['settings']}


def branch_directory(work, config):
    return Path(work) / f"{config['source']}-{config['positions']}"


def baseline_rows(work, source):
    return [row for path in sorted(Path(work).glob('exp0-gpu*.json'))
            for row in json.loads(path.read_text()) if row['setting'] == source]


def gpu_check(backend, config):
    jobs = [{'id': f'check/{i}', 'question': question, 'force_length': True, 'intervention_round': 2 + i % 2}
            for i, question in enumerate([
                'Explain in detail how to compute 37 times 24 and verify the result.',
                'Compute 123 + 456. Explain the calculation and then verify each digit.',
                'Derive the sum of the first ten positive integers in two ways.',
                'Explain the difference between the area and perimeter of a rectangle with sides 7 and 11.'])]
    checks = {}
    for setting in config['settings']:
        print(f"Checking {setting['name']}", flush=True)
        backend.set_setting(setting)
        outputs = backend.generate(jobs[:2])
        events = [event for result in outputs for event in result['events']
                  if event['full_generation_block'] and event['positions']]
        if setting['strategy'] == 'low_confidence_static':
            expected = config['block_length'] // setting['steps']
            assert events and all(len(event['positions']) == expected for event in events)
        else:
            assert events and all(len(event['positions']) >= 1 for event in events)
        checks[setting['name']] = {'passed': True, 'events': len(events)}
    backend.set_setting(settings(config)['T8'])
    print('Checking recording, restoration, injection and budgets', flush=True)
    baseline = backend.generate(jobs[:1])[0]
    backend.config['trace'] = False
    plain = backend.generate(jobs[:1])[0]
    backend.config['trace'] = True
    recording_difference = trace_difference(baseline['events'], plain['events'])
    recording_ok = baseline['token_ids'] == plain['token_ids'] and recording_difference is None
    batched = backend.generate(jobs)[0]
    batch_difference = trace_difference(baseline['events'], batched['events'])
    batch_ok = baseline['token_ids'] == batched['token_ids'] and batch_difference is None
    state = baseline['state']
    assert state is not None, 'Short check must reach a complete nonterminal block'
    action = {'positions': state['positions'], 'tokens': [state['current_ids'][i] for i in state['positions']]}
    restore_jobs = [dict(jobs[0], id=f'restore/{i}', snapshot=state, action=action) for i in range(2)]
    restored = backend.generate(restore_jobs)
    restoration = [restoration_difference(baseline, result, state) for result in restored]
    reversed_results = backend.generate(restore_jobs[::-1])
    restoration += [restoration_difference(baseline, result, state) for result in reversed_results]
    second = backend.generate(jobs[1:2])[0]
    second_state = second['state']
    assert second_state is not None
    second_action = {'positions': second_state['positions'],
                     'tokens': [second_state['current_ids'][i] for i in second_state['positions']]}
    second_restore = backend.generate([dict(jobs[1], snapshot=second_state, action=second_action)])[0]
    restoration.append(restoration_difference(second, second_restore, second_state))
    alternative = dict(action, tokens=list(action['tokens']))
    alternative['tokens'][0] = state['topk_ids'][action['positions'][0]][1]
    limited = backend.generate([dict(jobs[0], id='budget/check', snapshot=state,
                                     action=alternative, future_budget=1)])[0]
    injection_ok = limited['events'][0]['tokens'] == alternative['tokens']
    budget_ok = limited['future_forwards'] == 1 and limited['finish_reason'] == 'forward_budget'
    native = backend.compare_native(jobs[0], baseline)
    return {'passed': recording_ok and batch_ok and not any(restoration) and native['passed'] and injection_ok and budget_ok,
            'settings': checks, 'recording_invariant': recording_ok, 'recording_difference': recording_difference,
            'request_order_invariant': batch_ok, 'request_order_difference': batch_difference,
            'execution_batch_size': config['batch_size'], 'max_observed_batch': backend.max_observed_batch,
            'restoration_differences': restoration, 'action_injection': injection_ok,
            'strict_forward_budget': budget_ok, 'native': native}


def worker(gpu, config, rows, stage, work):
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu), PYTHONDONTWRITEBYTECODE='1',
        HF_HOME=str(GA / 'cache/hf'), HF_MODULES_CACHE=str(GA / 'cache/hf_modules'),
        TRITON_CACHE_DIR=str(GA / 'cache/triton'), TORCHINDUCTOR_CACHE_DIR=str(GA / 'cache/inductor'),
        SDAR_FULL_VOCAB='1', TOKENIZERS_PARALLELISM='false')
    sys.dont_write_bytecode = True
    from engine import Backend
    from lmdeploy.utils import get_logger
    get_logger('lmdeploy').setLevel('ERROR')
    work = Path(work)
    source = config['source']
    if stage == 'check':
        config = dict(config, max_new_tokens=config['check_tokens'], trace=True)
    backend = Backend(config, settings(config)[source])
    grader = Grader(config['grade_timeout']) if stage in ('check', 'exp0', 'exp1') else None
    try:
        if stage == 'check':
            result = gpu_check(backend, config)
            grade_cases = [grader.score(r'\boxed{5}', '5', True), grader.score(r'\boxed{6}', '5', True),
                           grader.score(r'\boxed{5}', '5', False)]
            result['grading'] = grade_cases
            result['passed'] &= [case['reward'] for case in grade_cases] == [1.0, 0.0, 0.0]
            write_json(work / f'check-gpu{gpu}.json', result)
            assert result['passed'], json.dumps(result, ensure_ascii=False)
            return {'gpu': gpu, 'passed': True}
        if stage == 'exp0':
            records = []
            for setting in config['settings']:
                backend.set_setting(setting)
                backend.generate([{'id': 'warmup', 'question': 'What is 2 + 3?', 'max_new_tokens': 16}])
                prefix = 'exp0' if config['trace'] else 'throughput'
                path = work / f"{prefix}-{setting['name']}-gpu{gpu}.trace.jsonl.gz" if config['trace'] else None
                calls_before = backend.batch_forward_calls
                outputs = backend.generate([{'id': row['unique_id'], 'question': row['problem'],
                                             'intervention_round': row['intervention_round']} for row in rows], path)
                elapsed = backend.last_seconds
                for row, result in zip(rows, outputs):
                    events = result.pop('events')
                    result.update(question_id=row['unique_id'], setting=setting['name'], gpu=gpu, subject=row['subject'],
                                  level=row['level'], divergence=divergence_summary(events),
                                  batch_wall_seconds=elapsed, batch_questions=len(rows),
                                  batched_forward_calls=backend.batch_forward_calls - calls_before)
                    result.update(grader.score(result['text'], row['answer'], result['finished']))
                    records.append(result)
                write_json(work / f'{prefix}-gpu{gpu}.json', records)
                print(f"GPU {gpu}: {setting['name']} finished {len(rows)} questions in {elapsed:.2f}s", flush=True)
            return {'gpu': gpu, 'answers': len(records)}

        branch = branch_directory(work, config)
        branch.mkdir(exist_ok=True)
        baselines = {row['question_id']: row for row in baseline_rows(work, source)}
        selected = [baselines[row['unique_id']] for row in rows if baselines[row['unique_id']]['state'] is not None]
        if stage == 'verify-resume':
            events = {}
            for path in sorted(work.glob(f'exp0-{source}-gpu*.trace.jsonl.gz')):
                for event in read_jsonl(path):
                    events.setdefault(event['request_id'], []).append(event)
            jobs = []
            for row in selected:
                state = row['state']
                action = {'positions': state['positions'], 'tokens': [state['current_ids'][i] for i in state['positions']]}
                jobs.append({'id': f"verify/{row['question_id']}", 'snapshot': state, 'action': action})
            outputs = backend.generate(jobs, branch / f'verify-gpu{gpu}.trace.jsonl.gz')
            records = []
            for row, result in zip(selected, outputs):
                baseline = dict(row, events=events[row['question_id']])
                difference = restoration_difference(baseline, result, row['state'])
                records.append({'question_id': row['question_id'], 'state_id': row['state']['id'],
                                'passed': difference is None, 'difference': difference,
                                'expected_future_forwards': row['forward_calls'] - row['state']['source_forward'],
                                'future_forwards': result['future_forwards'], 'restore_forwards': result['restore_forwards']})
            write_json(branch / f'verify-gpu{gpu}.json', records)
            assert all(record['passed'] for record in records), f'Restoration failed: {branch}'
            return {'gpu': gpu, 'verified_states': len(records)}

        if stage == 'exp1':
            continuation = settings(config)[config['continuation']]
            backend.set_setting(continuation)
            questions = {row['unique_id']: row for row in rows}
            jobs, metadata = [], []
            for row in selected:
                state = row['state']
                pool = action_pool(state, config['positions'], config['seed'])
                budget = forward_budget(state, config['max_new_tokens'], continuation['steps'], config['block_length'])
                for action in pool['actions']:
                    jobs.append({'id': f"{state['id']}/{action['id']}", 'snapshot': state,
                                 'action': action, 'future_budget': budget})
                    metadata.append({'question_id': row['question_id'], 'state_id': state['id'], 'action': action,
                                     'pool': {key: value for key, value in pool.items() if key != 'actions'},
                                     'future_budget': budget, 'source': source, 'continuation': config['continuation'],
                                     'positions_mode': config['positions']})
            outputs = backend.generate(jobs, branch / f'exp1-gpu{gpu}.trace.jsonl.gz')
            records = []
            for information, result in zip(metadata, outputs):
                result.pop('events')
                result.pop('state')
                result.update(information)
                result.update(grader.score(result['text'], questions[information['question_id']]['answer'], result['finished']))
                records.append(result)
            write_json(branch / f'exp1-gpu{gpu}.json', records)
            return {'gpu': gpu, 'states': len(selected), 'continuations': len(records)}
    finally:
        if grader:
            grader.close()
        backend.close()


def report(work, config):
    work = Path(work)
    summary = {'scope': work.name, 'baselines': {}, 'paired': {}}
    all_rows = [row for path in sorted(work.glob('exp0-gpu*.json')) for row in json.loads(path.read_text())]
    for setting in settings(config):
        rows = [row for row in all_rows if row['setting'] == setting]
        if not rows:
            continue
        counts, mutable = {}, {}
        for row in rows:
            for target, key in ((counts, 'd_selected'), (mutable, 'd_mutable')):
                for d, n in row['divergence'][key].items():
                    target[d] = target.get(d, 0) + n
        total = sum(mutable.values())
        summary['baselines'][setting] = {'n': len(rows), 'accuracy': sum(row['reward'] for row in rows) / len(rows),
            'truncation_rate': sum(not row['finished'] for row in rows) / len(rows),
            'grading_timeouts': sum(row['grading'] == 'timeout' for row in rows),
            'mean_tokens': sum(len(row['token_ids']) for row in rows) / len(rows),
            'mean_forward_calls': sum(row['forward_calls'] for row in rows) / len(rows),
            'selected_state_coverage': sum(row['state'] is not None for row in rows) / len(rows),
            'd_selected': counts, 'd_mutable': mutable,
            'fraction_below_two': sum(n for d, n in mutable.items() if int(d) < 2) / total if total else None,
            'per_question_divergence': {row['question_id']: row['divergence'] for row in rows}}
        summary['baselines'][setting]['worker_inference_seconds'] = {row['gpu']: row['batch_wall_seconds'] for row in rows}
        summary['baselines'][setting]['by_round'] = {}
        for round_id in ('2', '3'):
            parts = [row['divergence']['by_round'][round_id] for row in rows]
            histograms = {key: {} for key in ('d_selected', 'd_mutable')}
            for part in parts:
                for key, histogram in histograms.items():
                    for d, n in part[key].items():
                        histogram[d] = histogram.get(d, 0) + n
            count = sum(part['states'] for part in parts)
            below = sum(n for d, n in histograms['d_mutable'].items() if int(d) < 2) / count if count else None
            summary['baselines'][setting]['by_round'][round_id] = {'states': count, **histograms,
                'fraction_below_two': below, 'fraction_at_least_two': 1 - below if below is not None else None}
    summary['throughput'] = {}
    throughput = [row for path in sorted(work.glob('throughput-gpu*.json')) for row in json.loads(path.read_text())]
    for setting in settings(config):
        rows = [row for row in throughput if row['setting'] == setting]
        if rows:
            seconds = {row['gpu']: row['batch_wall_seconds'] for row in rows}
            summary['throughput'][setting] = {'worker_inference_seconds': seconds,
                'tokens': sum(len(row['token_ids']) for row in rows),
                'per_gpu_tokens_per_second': {gpu: sum(len(row['token_ids']) for row in rows if row['gpu'] == gpu) / duration
                                              for gpu, duration in seconds.items()}}
    for left, right in (('T32', 'T8'), ('T16', 'T8'), ('T32', 'T4'), ('T16', 'T4'), ('T8', 'T4')):
        a = {row['question_id']: row['reward'] for row in all_rows if row['setting'] == left and row['grading'] != 'timeout'}
        b = {row['question_id']: row['reward'] for row in all_rows if row['setting'] == right and row['grading'] != 'timeout'}
        if a and b:
            summary['paired'][f'{right}-{left}'] = paired_stats(a, b, config['seed'])
    branch = branch_directory(work, config)
    verification = [row for path in sorted(branch.glob('verify-gpu*.json')) for row in json.loads(path.read_text())]
    summary['restoration'] = {'states': len(verification), 'passed': sum(row['passed'] for row in verification)}
    oracle = [row for path in sorted(branch.glob('exp1-gpu*.json')) for row in json.loads(path.read_text())]
    if oracle:
        summary['oracle'] = oracle_summary(oracle)
    write_json(work / 'summary.json', summary)
    lines = [f'# SDAR b32 — {work.name}', '', '| Setting | n | Accuracy | Truncated | Mean forwards | P(d_U<2) |',
             '|---|---:|---:|---:|---:|---:|']
    for name, value in summary['baselines'].items():
        fraction = value['fraction_below_two']
        lines.append(f"| {name} | {value['n']} | {value['accuracy']:.1%} | {value['truncation_rate']:.1%} | "
                     f"{value['mean_forward_calls']:.1f} | {fraction:.1%} |" if fraction is not None else
                     f"| {name} | {value['n']} | {value['accuracy']:.1%} | {value['truncation_rate']:.1%} | {value['mean_forward_calls']:.1f} | n/a |")
    lines += ['', f"Restoration: {summary['restoration']['passed']}/{len(verification)} states passed."]
    if oracle:
        lines += [f"Oracle: {summary['oracle']['states']} states; {summary['oracle']['hybrid_rescues']} hybrid rescues."]
    lines += ['', 'Exploratory development results do not establish quality equivalence. See summary.json for paired intervals, pool comparisons and grading timeouts.']
    (work / 'summary.md').write_text('\n'.join(lines) + '\n')
    return summary


def fingerprint(config):
    code_paths = list((ROOT / 'DiRL/lmdeploy').rglob('*.py')) + list((ROOT / 'sdar_justgmpo').glob('*.py'))
    code_paths += list((ROOT / 'sdar_justgmpo/vendor').rglob('*.py')) + [GA / name for name in ('run.py', 'engine.py', 'protocol.py')]
    code = {str(path.relative_to(ROOT)): file_hash(path) for path in sorted(code_paths)}
    model = {path.name: file_hash(path) for path in sorted(Path(config['model']).iterdir()) if path.is_file()}
    data = {name: file_hash(Path(config['data']) / name) for name in ('test.jsonl', 'manifest.json')}
    # Workpoint/position choices are versioned per branch until the final freeze.
    shared = {key: value for key, value in config.items() if key not in ('scope', 'source', 'positions', 'trace')}
    value = {'config': shared, 'code': code, 'model': model, 'data': data}
    value['digest'] = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, choices=('check', 'exp0', 'verify-resume', 'exp1', 'report', 'freeze'))
    parser.add_argument('--config', type=Path, default=GA / 'config.json')
    parser.add_argument('--run-id', default=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d-%H%M%S'))
    parser.add_argument('--scope', choices=('pilot', 'dev', 'eval'))
    parser.add_argument('--source', choices=('T8', 'T4'))
    parser.add_argument('--positions', choices=('confidence', 'disagreement'))
    parser.add_argument('--gpus', nargs='+', type=int)
    parser.add_argument('--max-new-tokens', type=int)
    parser.add_argument('--no-trace', action='store_true')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    for key in ('scope', 'source', 'positions', 'gpus', 'max_new_tokens'):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    if args.no_trace:
        config['trace'] = False
    assert config['block_length'] == 32 and config['top_k_record'] >= 2
    rows = load_data(config['data'], config['scope'])
    run = GA / 'runs' / args.run_id
    run.mkdir(parents=True, exist_ok=True)
    signature = fingerprint(config)
    manifest_path = run / 'manifest.json'
    if manifest_path.exists():
        assert json.loads(manifest_path.read_text())['fingerprint']['digest'] == signature['digest'], 'Code, model, data or shared parameters changed; use a new run ID'
    else:
        write_json(manifest_path, {'config': config, 'fingerprint': signature,
            'environment': {'python': sys.version, 'packages': {name: importlib.metadata.version(name)
                              for name in ('torch', 'transformers', 'triton')}}})
    scientific = {key: value for key, value in config.items() if key not in ('scope', 'trace')}
    if config['scope'] == 'eval':
        frozen = json.loads((run / 'frozen.json').read_text())
        assert frozen['fingerprint'] == signature['digest'] and frozen['config'] == scientific, 'Evaluation requires matching frozen configuration'
    work = run / config['scope']
    work.mkdir(exist_ok=True)
    if args.stage == 'report':
        report(work, config)
        print(work / 'summary.md')
        return
    if args.stage == 'freeze':
        assert config['scope'] != 'eval'
        summary = report(work, config)
        assert summary.get('oracle', {}).get('states', 0) > 0, 'Complete development interventions before freezing'
        write_json(run / 'frozen.json', {'config': scientific, 'fingerprint': signature['digest'], 'development_scope': config['scope']})
        print(run / 'frozen.json')
        return
    if args.stage == 'check':
        subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(GA), '-p', 'test_protocol.py', '-v'], check=True,
                       env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
    else:
        check_files = list((run / 'pilot').glob('check-gpu*.json')) + list((run / 'dev').glob('check-gpu*.json'))
        assert check_files and all(json.loads(path.read_text())['passed'] for path in check_files), 'Run --stage check successfully first'
    if args.stage in ('verify-resume', 'exp1'):
        assert config['trace'], 'Branches require recorded baseline states'
        assert baseline_rows(work, config['source']), 'Run Experiment 0 for this scope first'
    if args.stage == 'exp1':
        verification = [row for path in branch_directory(work, config).glob('verify-gpu*.json') for row in json.loads(path.read_text())]
        expected = {row['state']['id'] for row in baseline_rows(work, config['source']) if row['state'] is not None}
        assert expected and expected == {row['state_id'] for row in verification if row['passed']}, 'Every selected source state must pass restoration first'
    if args.stage == 'check':
        gpus = config['gpus'][:1]
        output_directory = work
    else:
        gpus = config['gpus'][:min(len(rows), len(config['gpus']))]
        output_directory = branch_directory(work, config) if args.stage in ('verify-resume', 'exp1') else work
    output_directory.mkdir(exist_ok=True)
    label = 'verify' if args.stage == 'verify-resume' else args.stage
    if args.stage == 'exp0' and not config['trace']:
        label = 'throughput'
    assert not list(output_directory.glob(f'{label}-gpu*.json')), 'This stage already has results; use a new run ID for reruns'
    with ProcessPoolExecutor(max_workers=len(gpus), mp_context=mp.get_context('spawn')) as executor:
        futures = [executor.submit(worker, gpu, dict(config), rows[index::len(gpus)], args.stage, work)
                   for index, gpu in enumerate(gpus)]
        for future in futures:
            print(json.dumps(future.result(), ensure_ascii=False), flush=True)
    if args.stage != 'check':
        report(work, config)
    print(f'Completed {args.stage}: {work}', flush=True)


if __name__ == '__main__':
    main()
