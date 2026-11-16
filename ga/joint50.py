"""Expand the frozen joint oracle to 50 development questions, reusing pilot paths."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
import os
from pathlib import Path
import random
import sys

from action_value import make_actions, select_source
from early import extract_boxed_answer, repetition, snapshot
from joint_protocol import FORWARD_CAP, MATCHED_SEEDS, analyze_state, audit_path, make_pool
from protocol import (Grader, file_hash, paired_stats, read_jsonl, restoration_difference,
                      trace_difference, write_json)
from run import fingerprint

GA = Path(__file__).resolve().parent
OLD = GA / 'runs/joint-oracle-20261111'
OUT = GA / 'runs/joint50-20261111'
CODE = ('joint50.py', 'test_joint50.py', 'joint.py', 'joint_protocol.py', 'action_value.py', 'early.py')
SEED = 20261113


def select_fifty(rows, old):
    """Keep the 20 original development IDs; stratify 30 from the old evaluation pool."""
    groups = defaultdict(list)
    for row in rows:
        if row['unique_id'] in old['eval_ids']:
            groups[row['subject'], row['level']].append(row['unique_id'])
    quotas = {key: len(ids) * 30 // 80 for key, ids in groups.items()}
    order = sorted(groups, key=lambda key: (-(len(groups[key]) * 30 % 80), key))
    for key in order[:30 - sum(quotas.values())]:
        quotas[key] += 1
    rng = random.Random(SEED)
    promoted = {q for key in sorted(groups) for q in rng.sample(sorted(groups[key]), quotas[key])}
    selected = set(old['dev_ids']) | promoted
    pilot = set(old['pilot_ids'])
    assert len(selected) == 50 and len(promoted) == 30 and pilot <= selected
    data = [dict(row, cohort='pilot10' if row['unique_id'] in pilot else 'new40')
            for row in rows if row['unique_id'] in selected]
    selection = {'seed': SEED, 'selection': 'Original dev20 plus proportional subject-by-level sample of 30 old eval IDs; largest remainders, sorted IDs before sampling.',
        'selected_ids': [r['unique_id'] for r in data], 'pilot10_ids': sorted(pilot),
        'new40_ids': sorted(selected - pilot), 'promoted_eval30_ids': sorted(promoted),
        'remaining_eval50_ids': sorted(set(old['eval_ids']) - promoted),
        'final400': 'Original complement of mechanism100; untouched.',
        'usage': 'All selected 50 are development data; promoted IDs are no longer held-out evaluation data.'}
    return data, selection


def prepare(out):
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / 'manifest.json').exists()
    parent = json.loads((OLD / 'manifest.json').read_text())
    assert all(file_hash(GA / name) == digest for name, digest in parent['code'].items())
    config = dict(parent['config'], scope='development50')
    assert fingerprint(config)['digest'] == parent['parent_fingerprint']
    data_dir = Path(config['data'])
    split = json.loads((data_dir / 'manifest.json').read_text())
    data, selection = select_fifty(read_jsonl(data_dir / 'test.jsonl'), split)
    old_cases = json.loads((OLD / 'cases.json').read_text())
    original_gpus = {c['question_id']: c['gpu'] for c in old_cases}
    fresh = [r for r in data if r['cohort'] == 'new40']
    assignment = {r['unique_id']: config['gpus'][i % 4] for i, r in enumerate(fresh)}
    for row in data:
        row['gpu'] = original_gpus[row['unique_id']] if row['cohort'] == 'pilot10' else assignment[row['unique_id']]
    write_json(out / 'data.json', data)
    write_json(out / 'selection.json', selection)
    sources = [OLD / 'manifest.json', OLD / 'cases.json'] + sorted(OLD.glob('run-gpu*.json'))
    write_json(out / 'manifest.json', {'config': config, 'parent': str(OLD),
        'parent_fingerprint': parent['parent_fingerprint'], 'questions': 50,
        'code': {name: file_hash(GA / name) for name in CODE},
        'data_sha256': file_hash(out / 'data.json'), 'selection_sha256': file_hash(out / 'selection.json'),
        'source_files': {str(p.relative_to(OLD)): file_hash(p) for p in sources},
        'protocol': {'source': 'Unchanged first eligible early trigger, within 256 generated positions, p<0.9 or top1 flip.',
            'candidates': 'Unchanged joint_protocol.make_pool; max 5 position plans x 4 value plans.',
            'forward_cap': FORWARD_CAP, 'same_block_rounds': True, 'matched_seeds': MATCHED_SEEDS,
            'grading': 'Original EOS/boxed equivalence; 1s timeout, CPU-only 10s recheck for timeouts; unresolved judgments remain excluded.',
            'reuse': 'Pilot paths reused only after fresh uninterrupted baselines, source snapshots and actual pools match.'}})
    print(json.dumps({'questions': len(data), 'pilot': 10, 'new': 40, 'promoted_eval': 30,
                      'remaining_eval': 50, 'gpus': dict(Counter(r['gpu'] for r in data))}))


def environment(gpu):
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu), PYTHONDONTWRITEBYTECODE='1',
        HF_HOME=str(GA / 'cache/hf'), HF_MODULES_CACHE=str(GA / 'cache/hf_modules'),
        TRITON_CACHE_DIR=str(GA / 'cache/triton'), TORCHINDUCTOR_CACHE_DIR=str(GA / 'cache/inductor'),
        SDAR_FULL_VOCAB='1', TOKENIZERS_PARALLELISM='false')
    sys.dont_write_bytecode = True


def score(grader, result, answer):
    value = grader.score(result['text'], answer, result['finished'])
    if value['grading'] == 'timeout':
        grader.timeout = 10
        try:
            value = dict(grader.score(result['text'], answer, result['finished']), initial_grading_timeout=True)
        finally:
            grader.timeout = 1
    return value


def baseline_worker(gpu, config, data, out):
    environment(gpu)
    from engine import Backend
    from lmdeploy.utils import get_logger
    get_logger('lmdeploy').setLevel('ERROR')
    out = Path(out)
    settings = {s['name']: s for s in config['settings']}
    backend = Backend(config, settings['T8'])
    grader = Grader(config['grade_timeout'])
    model = json.loads((Path(config['model']) / 'config.json').read_text())
    generation = json.loads((Path(config['model']) / 'generation_config.json').read_text())
    old_cases = {(c['setting'], c['question_id']): c for c in json.loads((OLD / 'cases.json').read_text())}
    old_results = {(r['setting'], r['question_id']): r for r in json.loads((OLD / f'run-gpu{gpu}.json').read_text()) if r['group'] == 'G0'}
    path = out / f'baseline-gpu{gpu}.json'
    records = json.loads(path.read_text()) if path.exists() else []
    completed = {(r['setting'], r['question_id']) for r in records}
    case_path = out / f'cases-gpu{gpu}.json'
    cases = json.loads(case_path.read_text()) if case_path.exists() else []
    try:
        for setting in ('T8', 'T4'):
            backend.set_setting(settings[setting])
            for row in data:
                q = row['unique_id']
                if (setting, q) in completed:
                    continue
                name = f"baseline-{setting}-{q.replace('/', '_').replace('.json', '')}"
                trace = out / f'{name}.trace.jsonl.gz'
                result = backend.generate([{'id': name, 'question': row['problem'], 'future_budget': FORWARD_CAP}], trace)[0]
                elapsed, events = backend.last_seconds, result.pop('events')
                result.update(question_id=q, setting=setting, cohort=row['cohort'], gpu=gpu,
                    trace_path=str(trace), inference_seconds=elapsed, state={'prompt_ids': backend.prompt(row['problem'])})
                event = select_source(events, len(result['state']['prompt_ids']), settings[setting]['steps'])
                if event is not None:
                    state = snapshot(result, events, event['forward'])
                    case = {'question_id': q, 'setting': setting, 'gpu': gpu, 'cohort': row['cohort'],
                        'snapshot': state, 'actions': make_actions(event, [e for e in events if e['forward'] < event['forward']])}
                    case['pool'] = make_pool(case, generation['eos_token_id'], model['vocab_size'])
                    if row['cohort'] == 'pilot10':
                        old = old_cases[setting, q]
                        for field in ('input_ids', 'mask', 'positions', 'prefix_ids', 'replay', 'source_forward'):
                            assert state[field] == old['snapshot'][field], (setting, q, field)
                        assert case['pool'] == old['pool'], (setting, q, 'pool')
                    cases.append(case)
                    result['source_available'] = True
                else:
                    result['source_available'] = False
                if row['cohort'] == 'pilot10':
                    old = old_results[setting, q]
                    assert all(result[k] == old[k] for k in ('token_ids', 'finish_reason', 'forward_calls'))
                    assert trace_difference(read_jsonl(OLD / f"{old['request_id']}.trace.jsonl.gz"), events) is None
                    result['matches_old_pilot_current'] = True
                result.update(score(grader, result, row['answer']))
                records.append(result)
                write_json(path, records)
                write_json(case_path, cases)
                print(json.dumps({'stage': 'baseline', 'gpu': gpu, 'setting': setting, 'question': q,
                    'reward': result['reward'], 'source': result['source_available'], 'forwards': result['forward_calls']}), flush=True)
    finally:
        grader.close()
        backend.close()
    return {'gpu': gpu, 'baselines': len(records), 'states': len(cases)}


def branch_worker(gpu, config, cases, data, out):
    environment(gpu)
    from joint import backend_type
    Backend = backend_type()
    from lmdeploy.utils import get_logger
    get_logger('lmdeploy').setLevel('ERROR')
    out = Path(out)
    settings = {s['name']: s for s in config['settings']}
    backend = Backend(config, settings['T8'])
    grader = Grader(config['grade_timeout'])
    baselines = {(r['setting'], r['question_id']): r for r in json.loads((out / f'baseline-gpu{gpu}.json').read_text())}
    original = {(r['setting'], r['state_id'], r['action_id']): r for r in json.loads((OLD / f'run-gpu{gpu}.json').read_text())}
    questions = {r['unique_id']: r for r in data}
    path = out / f'run-gpu{gpu}.json'
    records = json.loads(path.read_text()) if path.exists() else []
    completed = {(r['setting'], r['state_id'], r['action_id']) for r in records}
    try:
        for case in cases:
            setting, q, state, pool = case['setting'], case['question_id'], case['snapshot'], case['pool']
            backend.set_setting(settings[setting])
            baseline = baselines[setting, q]
            baseline_events = read_jsonl(baseline['trace_path'])
            control_ok = (setting, state['id'], pool['current_id']) in completed
            for action in pool['actions']:
                key = setting, state['id'], action['id']
                if key in completed:
                    continue
                if case['cohort'] == 'pilot10':
                    result = dict(original[key], cohort='pilot10', origin='reused_pilot', new_forward_calls=0,
                        new_inference_seconds=0., trace_path=str(OLD / f"{original[key]['request_id']}.trace.jsonl.gz"))
                    assert result['action'] == action
                    result['validation'] = dict(result['validation'], fresh_baseline_matches_old=True)
                else:
                    assert action['id'] == pool['current_id'] or control_ok
                    name = f"run-{setting}-{q.replace('/', '_').replace('.json', '')}-{action['id']}"
                    trace = out / f'{name}.trace.jsonl.gz'
                    job = {'id': name, 'snapshot': state, 'action': action,
                           'future_budget': FORWARD_CAP - state['source_forward']}
                    result = backend.generate([job], trace)[0]
                    elapsed, events = backend.last_seconds, result.pop('events')
                    result.pop('state')
                    checks = audit_path(result, events, case, action)
                    if action['id'] == pool['current_id']:
                        difference = restoration_difference(dict(baseline, events=baseline_events), dict(result, events=events), state)
                        assert difference is None, difference
                        checks['current_restoration'] = True
                        control_ok = True
                    result.update(question_id=q, setting=setting, gpu=gpu, state_id=state['id'], action_id=action['id'],
                        group=action['group'], cohort='new40', origin='new', action=action, validation=checks,
                        trace_path=str(trace), inference_seconds=elapsed, new_inference_seconds=elapsed,
                        new_forward_calls=result['forward_calls'], boxed_answer=extract_boxed_answer(result['text']),
                        repetition=repetition(result['token_ids'], max_period=256))
                    result.update(score(grader, result, questions[q]['answer']))
                    if action['id'] == pool['current_id']:
                        assert result['reward'] == baseline['reward']
                records.append(result)
                completed.add(key)
                write_json(path, records)
                print(json.dumps({'stage': 'branch', 'gpu': gpu, 'setting': setting, 'question': q,
                    'group': action['group'], 'reward': result['reward'], 'origin': result['origin'],
                    'forwards': result['forward_calls']}), flush=True)
    finally:
        grader.close()
        backend.close()
    return {'gpu': gpu, 'actions': len(records), 'new_forwards': sum(r['new_forward_calls'] for r in records)}


def cohort_statistics(data, baselines, states, cohort):
    ids = {r['unique_id'] for r in data if cohort == 'all50' or r['cohort'] == cohort}
    output = {}
    for setting in ('T8', 'T4'):
        baseline = {r['question_id']: r for r in baselines if r['setting'] == setting and r['question_id'] in ids}
        found = {s['question_id']: s for s in states if s['setting'] == setting and s['question_id'] in ids}
        assert set(baseline) == ids and all(s['eligible'] for s in found.values())
        current = {q: r['reward'] for q, r in baseline.items()}
        values = {q: found[q]['best'] if q in found else {g: current[q] for g in ('G0', 'G1', 'G2', 'G3')} for q in ids}
        all_best = {q: found[q]['all_best'] if q in found else current[q] for q in ids}
        singles = {q: found[q]['singles_best'] if q in found else current[q] for q in ids}
        matched = []
        for j, seed in enumerate(MATCHED_SEEDS):
            matched.append({'seed': seed,
                'singles_correct': sum(found[q]['matched'][j]['singles_reward'] if q in found else current[q] for q in ids),
                'joint_correct': sum(found[q]['matched'][j]['joint_reward'] if q in found else current[q] for q in ids)})
        output[setting] = {'questions': len(ids), 'source_states': len(found),
            'current_correct': sum(current.values()), 'oracle_correct': {g: sum(v[g] for v in values.values()) for g in ('G0', 'G1', 'G2', 'G3')},
            'singles_union_correct': sum(singles.values()), 'all_groups_correct': sum(all_best.values()),
            'joint_exclusive_questions': sorted(q for q, s in found.items() if s['joint_exclusive']),
            'paired_current_vs_all': paired_stats(current, all_best),
            'paired_singles_vs_all': paired_stats(singles, all_best), 'matched': matched,
            'missing_source_questions': sorted(ids - set(found))}
    return output


def report(out):
    data = json.loads((out / 'data.json').read_text())
    cases = json.loads((out / 'cases.json').read_text())
    rows = [r for p in sorted(out.glob('run-gpu*.json')) for r in json.loads(p.read_text())]
    baselines = [r for p in sorted(out.glob('baseline-gpu*.json')) for r in json.loads(p.read_text())]
    grouped = defaultdict(list)
    for r in rows:
        grouped[r['setting'], r['state_id']].append(r)
    assert len(rows) == sum(len(c['pool']['actions']) for c in cases)
    states = [analyze_state(c, grouped[c['setting'], c['snapshot']['id']]) for c in cases]
    summary = {'questions': 50, 'source_states': len(cases), 'unique_actions': len(rows),
        'reused_actions': sum(r['origin'] == 'reused_pilot' for r in rows),
        'new_actions': sum(r['origin'] == 'new' for r in rows),
        'new_branch_forwards': sum(r['new_forward_calls'] for r in rows),
        'baseline_forwards': sum(r['forward_calls'] for r in baselines),
        'represented_branch_forwards': sum(r['forward_calls'] for r in rows),
        'new_branch_seconds': sum(r['new_inference_seconds'] for r in rows),
        'baseline_seconds': sum(r['inference_seconds'] for r in baselines),
        'grading': dict(Counter(r['grading'] for r in rows)), 'states': states,
        'cohorts': {name: cohort_statistics(data, baselines, states, name) for name in ('all50', 'new40', 'pilot10')}}
    write_json(out / 'summary.json', summary)
    lines = ['# 联合位置与词值 Oracle：50 题扩量', '',
        '各池保留 Current，结果为事后 oracle。缺少合法源状态的题目保留 Current，分母不减少。', '',
        '| 数据范围 | 设置 | Current | 纯位置 | 纯词值 | 联合 | 单因素并集 | 全池 | 联合独有成功 |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for cohort, settings in summary['cohorts'].items():
        for setting, s in settings.items():
            n = s['questions']
            values = ' | '.join(f"{s['oracle_correct'][g]:g}/{n}" for g in ('G0', 'G1', 'G2', 'G3'))
            lines.append(f"| {cohort} | {setting} | {values} | {s['singles_union_correct']:g}/{n} | {s['all_groups_correct']:g}/{n} | {len(s['joint_exclusive_questions'])} |")
    lines += ['', f"源状态 {len(cases)}/100，唯一动作 {len(rows)}；复用 {summary['reused_actions']}，新增 {summary['new_actions']}。", '',
        f"本轮新增模型 forward：基线 {summary['baseline_forwards']} + 新分支 {summary['new_branch_forwards']}。旧复用路径成本不重复计入本轮新增。", '',
        '抽样及评估集用途变更见 [selection.json](selection.json)，全部配对与分层结果见 [summary.json](summary.json)。']
    (out / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k not in ('states', 'cohorts')}, ensure_ascii=False))
    for name, settings in summary['cohorts'].items():
        for setting, value in settings.items():
            print(name, setting, json.dumps({k: v for k, v in value.items() if k != 'matched'}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('prepare', 'baseline', 'run', 'report'), required=True)
    parser.add_argument('--out', type=Path, default=OUT)
    args = parser.parse_args()
    out = args.out.resolve()
    if args.stage == 'prepare':
        prepare(out)
        return
    if args.stage == 'report':
        report(out)
        return
    manifest = json.loads((out / 'manifest.json').read_text())
    assert all(file_hash(GA / name) == digest for name, digest in manifest['code'].items())
    assert file_hash(out / 'data.json') == manifest['data_sha256']
    data = json.loads((out / 'data.json').read_text())
    if args.stage == 'run':
        assert file_hash(out / 'cases.json') == manifest['cases_sha256']
        cases = json.loads((out / 'cases.json').read_text())
    with ProcessPoolExecutor(max_workers=4, mp_context=mp.get_context('spawn')) as executor:
        futures = []
        for gpu in manifest['config']['gpus']:
            rows = [r for r in data if r['gpu'] == gpu]
            if args.stage == 'baseline':
                futures.append(executor.submit(baseline_worker, gpu, dict(manifest['config']), rows, out))
            else:
                futures.append(executor.submit(branch_worker, gpu, dict(manifest['config']),
                    [c for c in cases if c['gpu'] == gpu], rows, out))
        outcomes = [f.result() for f in futures]
    if args.stage == 'baseline':
        cases = [c for p in sorted(out.glob('cases-gpu*.json')) for c in json.loads(p.read_text())]
        write_json(out / 'cases.json', cases)
        manifest.update(cases_sha256=file_hash(out / 'cases.json'), source_states=len(cases),
            unique_actions=sum(len(c['pool']['actions']) for c in cases),
            baseline_files={p.name: file_hash(p) for p in out.glob('baseline-gpu*.json')})
        write_json(out / 'manifest.json', manifest)
        print(json.dumps({'states': len(cases), 'actions': manifest['unique_actions'], 'workers': outcomes}))
    else:
        report(out)


if __name__ == '__main__':
    main()
