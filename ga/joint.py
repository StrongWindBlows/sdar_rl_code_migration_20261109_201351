"""Run the paired, single-step position/value oracle on the saved pilot states."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import sys

from action_value import capped_reference
from early import extract_boxed_answer, repetition
from joint_protocol import FORWARD_CAP, MATCHED_SEEDS, analyze_state, audit_path, make_pool
from protocol import (Grader, file_hash, load_data, paired_stats, read_jsonl,
                      restoration_difference, write_json)
from run import fingerprint

GA = Path(__file__).resolve().parent
CODE = ('joint.py', 'joint_protocol.py', 'test_joint.py', 'action_value.py', 'early.py')


def backend_type():
    from engine import Backend

    class JointBackend(Backend):
        def _install(self):
            super()._install()
            original_post = self.agent.agent_strategy.post_sampling
            processor = self.agent.agent_strategy.unmasking_processor
            original_static = processor.low_confidence_static

            def static(logits, tokens, mask):
                track, base = self.active
                state = track.job['snapshot']
                debt = len(state['positions']) - len(track.job['action']['positions'])
                if (debt and track.source_reached and base == state['block_start']
                        and track.rounds[base] + 1 == self.setting['steps']):
                    return mask.masked_fill(mask == 0, 1)
                return original_static(logits, tokens, mask)

            def post_sampling(inputs, logits, next_ids, extra):
                assert len(inputs.model_metas) == 1
                track = self.tracks[inputs.model_metas[0]['ga_session']]
                base = int((inputs.history_lengths + inputs.seq_length - self.block)[0])
                self.active = track, base
                return original_post(inputs, logits, next_ids, extra)

            self._replace(processor, 'low_confidence_static', static)
            self._replace(self.agent.agent_strategy, 'post_sampling', post_sampling)

    return JointBackend


def prepare(parent, out):
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / 'manifest.json').exists(), 'The prepared experiment already exists.'
    old = json.loads((parent / 'manifest.json').read_text())
    config = old['config']
    assert fingerprint(config)['digest'] == old['fingerprint']['digest']
    config_model = json.loads((Path(config['model']) / 'config.json').read_text())
    generation = json.loads((Path(config['model']) / 'generation_config.json').read_text())
    cases = json.loads((parent / 'action_value/cases.json').read_text())
    pilot = {r['unique_id'] for r in load_data(config['data'], 'pilot')}
    assert len(cases) == 20 and {c['question_id'] for c in cases} == pilot
    for case in cases:
        case['pool'] = make_pool(case, generation['eos_token_id'], config_model['vocab_size'])
    write_json(out / 'cases.json', cases)
    shutil.copyfile(GA / 'JOINT_ORACLE_PLAN.md', out / 'PLAN.md')
    source_paths = [parent / 'action_value/cases.json', parent / 'action_value/manifest.json']
    source_paths += sorted((parent / 'pilot').glob('exp0-gpu*.json'))
    source_paths += sorted((parent / 'pilot').glob('exp0-T[48]-gpu*.trace.jsonl.gz'))
    source_paths += sorted((parent / 'action_value').glob('results-gpu*.json'))
    manifest = {'parent': str(parent), 'parent_fingerprint': old['fingerprint']['digest'], 'config': config,
        'code': {name: file_hash(GA / name) for name in CODE},
        'source_files': {str(p.relative_to(parent)): file_hash(p) for p in source_paths},
        'cases_sha256': file_hash(out / 'cases.json'), 'plan_sha256': file_hash(out / 'PLAN.md'),
        'forward_cap': FORWARD_CAP, 'matched_seeds': MATCHED_SEEDS,
        'scope': 'Original 10 development pilot questions; T8/T4 separately.',
        'states': len(cases), 'logical_cells': sum(len(c['pool']['matrix']) for c in cases),
        'unique_actions': sum(len(c['pool']['actions']) for c in cases),
        'group_sizes': dict(Counter(a['group'] for c in cases for a in c['pool']['actions'])),
        'states_with_true_joint': sum(any(a['group'] == 'G3' for a in c['pool']['actions']) for c in cases)}
    write_json(out / 'manifest.json', manifest)
    print(json.dumps({k: manifest[k] for k in ('states', 'logical_cells', 'unique_actions', 'group_sizes', 'states_with_true_joint')}, ensure_ascii=False))


def worker(gpu, config, cases, parent, out, stage):
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu), PYTHONDONTWRITEBYTECODE='1',
        HF_HOME=str(GA / 'cache/hf'), HF_MODULES_CACHE=str(GA / 'cache/hf_modules'),
        TRITON_CACHE_DIR=str(GA / 'cache/triton'), TORCHINDUCTOR_CACHE_DIR=str(GA / 'cache/inductor'),
        SDAR_FULL_VOCAB='1', TOKENIZERS_PARALLELISM='false')
    sys.dont_write_bytecode = True
    parent, out = Path(parent), Path(out)
    Backend = backend_type()
    from lmdeploy.utils import get_logger
    get_logger('lmdeploy').setLevel('ERROR')
    settings = {s['name']: s for s in config['settings']}
    backend = Backend(config, settings[cases[0]['setting']])
    grader = Grader(config['grade_timeout']) if stage == 'run' else None
    rows = {(r['setting'], r['question_id']): r for r in json.loads((parent / f'pilot/exp0-gpu{gpu}.json').read_text())}
    legacy = {(r['setting'], r['question_id'], r['policy']): r
              for r in json.loads((parent / f'action_value/results-gpu{gpu}.json').read_text())}
    questions = {r['unique_id']: r for r in load_data(config['data'], 'pilot')}
    traces = {}
    for setting in {c['setting'] for c in cases}:
        traces[setting] = defaultdict(list)
        for e in read_jsonl(parent / f'pilot/exp0-{setting}-gpu{gpu}.trace.jsonl.gz'):
            traces[setting][e['request_id']].append(e)
    path = out / f'{stage}-gpu{gpu}.json'
    records = json.loads(path.read_text()) if path.exists() else []
    completed = {(r['state_id'], r['action_id']) for r in records}
    try:
        for case in cases:
            q, setting, state, pool = case['question_id'], case['setting'], case['snapshot'], case['pool']
            backend.set_setting(settings[setting])
            reference = capped_reference(rows[setting, q], traces[setting][q])
            actions = pool['actions']
            if stage == 'check':
                current = next(a for a in actions if a['id'] == pool['current_id'])
                allele = next(a for a in actions if a['group'] == 'G2')
                joint = next(a for a in actions if a['group'] == 'G3' and len(a['positions']) == len(state['positions']) - 2)
                actions = [current, allele, joint, current]
            control_ok = (state['id'], pool['current_id']) in completed
            for index, action in enumerate(actions):
                if stage == 'run' and (state['id'], action['id']) in completed:
                    continue
                assert action['id'] == pool['current_id'] or control_ok
                name = f"{stage}-{setting}-{q.replace('/', '_').replace('.json', '')}-{action['id']}"
                if stage == 'check':
                    name += f'-{index}'
                job = {'id': name, 'snapshot': state, 'action': action,
                       'future_budget': FORWARD_CAP - state['source_forward']}
                result = backend.generate([job], out / f'{name}.trace.jsonl.gz')[0]
                seconds = backend.last_seconds
                events = result.pop('events')
                result.pop('state')
                checks = audit_path(result, events, case, action)
                if action['id'] == pool['current_id']:
                    diff = restoration_difference(reference, dict(result, events=events), state)
                    assert diff is None, diff
                    checks['current_restoration'] = True
                    control_ok = True
                for p, old_name in (('P1', 'Delay1'), ('P3', 'Swap1')):
                    if action['id'] == pool['matrix'][f'{p}/V0']:
                        old = legacy[setting, q, old_name]
                        old_events = read_jsonl(parent / f"action_value/{old['request_id']}.trace.jsonl.gz")
                        diff = restoration_difference(dict(old, events=old_events), dict(result, events=events), state)
                        assert diff is None, (old_name, diff)
                        checks['matches_legacy_' + old_name] = True
                result.update(question_id=q, setting=setting, gpu=gpu, state_id=state['id'],
                    action_id=action['id'], group=action['group'], action=action, validation=checks,
                    inference_seconds=seconds, boxed_answer=extract_boxed_answer(result['text']),
                    repetition=repetition(result['token_ids'], max_period=256))
                if grader:
                    result.update(grader.score(result['text'], questions[q]['answer'], result['finished']))
                records.append(result)
                completed.add((state['id'], action['id']))
                write_json(path, records)
                print(json.dumps({'stage': stage, 'gpu': gpu, 'setting': setting, 'question': q,
                    'action': action['id'], 'group': action['group'], 'reward': result.get('reward'),
                    'forwards': result['forward_calls']}), flush=True)
    finally:
        if grader:
            grader.close()
        backend.close()
    return {'gpu': gpu, 'stage': stage, 'paths': len(records), 'passed': True,
            'forward_calls': sum(r['forward_calls'] for r in records)}


def report(out):
    manifest = json.loads((out / 'manifest.json').read_text())
    cases = json.loads((out / 'cases.json').read_text())
    rows = [r for p in sorted(out.glob('run-gpu*.json')) for r in json.loads(p.read_text())]
    grouped = defaultdict(list)
    for r in rows:
        grouped[r['setting'], r['state_id']].append(r)
    states = [analyze_state(c, grouped[c['setting'], c['snapshot']['id']]) for c in cases
              if len(grouped[c['setting'], c['snapshot']['id']]) == len(c['pool']['actions'])]
    summary = {'completed_actions': len(rows), 'planned_actions': manifest['unique_actions'],
        'completed_states': len(states), 'planned_states': len(cases),
        'total_forwards': sum(r['forward_calls'] for r in rows),
        'restore_forwards': sum(r['restore_forwards'] for r in rows),
        'inference_seconds': sum(r['inference_seconds'] for r in rows),
        'grading': dict(Counter(r['grading'] for r in rows)), 'all_validated': all(r['validation']['passed'] for r in rows),
        'states': states, 'settings': {}}
    for setting in ('T8', 'T4'):
        eligible = [s for s in states if s['setting'] == setting and s['eligible']]
        current = {s['question_id']: s['best']['G0'] for s in eligible}
        summary['settings'][setting] = {'states': len(eligible),
            'oracle_correct': {g: sum(s['best'][g] for s in eligible) for g in ('G0', 'G1', 'G2', 'G3')},
            'singles_union_correct': sum(s['singles_best'] for s in eligible),
            'all_groups_correct': sum(s['all_best'] for s in eligible),
            'joint_exclusive_questions': [s['question_id'] for s in eligible if s['joint_exclusive']],
            'paired_current_vs_all': paired_stats(current, {s['question_id']: s['all_best'] for s in eligible}),
            'matched': [{'seed': seed,
                'singles_correct': sum(s['matched'][j]['singles_reward'] for s in eligible),
                'joint_correct': sum(s['matched'][j]['joint_reward'] for s in eligible),
                'singles_forwards': sum(s['matched'][j]['singles_forwards'] for s in eligible),
                'joint_forwards': sum(s['matched'][j]['joint_forwards'] for s in eligible)}
                for j, seed in enumerate(MATCHED_SEEDS)]}
    write_json(out / 'summary.json', summary)
    lines = ['# Experiment 2-Joint：单步联合 Oracle', '',
        f"完成 {len(rows)}/{manifest['unique_actions']} 个唯一动作、{len(states)}/20 个状态。", '',
        '| 设置 | Current | Timing-only + Current | Allele-only + Current | Joint + Current | 单因素并集 | 全池 | 联合独有成功 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name, s in summary['settings'].items():
        values = ' | '.join(f"{s['oracle_correct'][g]:g}/{s['states']}" for g in ('G0', 'G1', 'G2', 'G3'))
        lines.append(f"| {name} | {values} | {s['singles_union_correct']:g}/{s['states']} | {s['all_groups_correct']:g}/{s['states']} | {len(s['joint_exclusive_questions'])} |")
    lines += ['', '各池均保留 Current；这是事后 oracle，不是在线策略成绩。', '',
        f"实际 forward 合计 {summary['total_forwards']}，包含重放 {summary['restore_forwards']}。判分状态：{summary['grading']}。", '',
        '完整矩阵、交互值和 32 个相同候选数量对照见 [summary.json](summary.json)。']
    (out / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k not in ('states', 'settings')}, ensure_ascii=False))
    for name, s in summary['settings'].items():
        print(name, json.dumps({k: v for k, v in s.items() if k != 'matched'}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, default=GA / 'runs/pilot8192-20261110')
    parser.add_argument('--out', type=Path, default=GA / 'runs/joint-oracle-20261111')
    parser.add_argument('--stage', choices=('prepare', 'check', 'run', 'report'), required=True)
    args = parser.parse_args()
    parent, out = args.parent.resolve(), args.out.resolve()
    if args.stage == 'prepare':
        prepare(parent, out)
        return
    if args.stage == 'report':
        report(out)
        return
    manifest = json.loads((out / 'manifest.json').read_text())
    assert all(file_hash(GA / name) == digest for name, digest in manifest['code'].items())
    assert file_hash(out / 'cases.json') == manifest['cases_sha256']
    assert fingerprint(manifest['config'])['digest'] == manifest['parent_fingerprint']
    cases = json.loads((out / 'cases.json').read_text())
    if args.stage == 'check':
        cases = [next(c for c in cases if c['setting'] == setting) for setting in ('T8', 'T4')]
    else:
        check = json.loads((out / 'check.json').read_text())
        assert check and all(r['passed'] for r in check)
    gpus = sorted({c['gpu'] for c in cases})
    with ProcessPoolExecutor(max_workers=len(gpus), mp_context=mp.get_context('spawn')) as executor:
        futures = [executor.submit(worker, gpu, dict(manifest['config']),
            [c for c in cases if c['gpu'] == gpu], parent, out, args.stage) for gpu in gpus]
        outcomes = [f.result() for f in futures]
    if args.stage == 'check':
        write_json(out / 'check.json', outcomes)
    else:
        report(out)


if __name__ == '__main__':
    main()
