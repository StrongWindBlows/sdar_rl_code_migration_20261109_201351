"""R1: one causal repetition trigger, one local perturbation, explicit rewind cost."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path
import random
import shutil
import time

from degeneration import all_rows, events_for, repeat_runs
from early import CORE, snapshot
from protocol import Grader, file_hash, read_jsonl, stable_seed, trace_difference, write_json

GA = Path(__file__).resolve().parent
OLD = GA / 'runs/degeneration-20261112'
OUT = GA / 'runs/repetition-recovery-20261112'
T8 = {'name': 'T8', 'steps': 8, 'strategy': 'low_confidence_static'}
D95 = {'name': 'D95', 'steps': 32, 'strategy': 'low_confidence_dynamic', 'threshold': .95}
CODE = ('repetition_recovery.py', 'repetition_recovery.json', 'test_repetition_recovery.py', 'R1_PLAN.md')


def read(path):
    return json.loads(Path(path).read_text())


class Detector:
    """Only committed tokens in a contiguous prefix can confirm repetition."""
    def __init__(self, prompt_length):
        self.prompt_length = prompt_length
        self.known, self.prefix, self.found = {}, [], None

    def observe(self, event):
        for i, token in zip(event['positions'], event['tokens']):
            self.known[event['block_start'] + i] = token
        previous = len(self.prefix)
        while self.prompt_length + len(self.prefix) in self.known:
            self.prefix.append(self.known[self.prompt_length + len(self.prefix)])
        if self.found is None and len(self.prefix) > previous:
            runs = repeat_runs(self.prefix)
            if runs:
                pattern = runs[0]
                start, period = pattern['motif_start'], pattern['period']
                self.found = dict(pattern, forward=event['forward'], observed_tokens=len(self.prefix),
                                  motif=self.prefix[start:start + period])
        return self.found


def target_position(event, pattern, prompt_length, rollback):
    lower = pattern['onset'] if rollback else pattern['observed_tokens']
    candidates = []
    for i in event['positions']:
        pos = event['block_start'] + i - prompt_length
        expected = pattern['motif'][(pos-pattern['motif_start']) % pattern['period']]
        if pos >= lower and event['current_ids'][i] == expected:
            candidates.append(i)
    return min(candidates) if candidates else None


def draw(logits, temperature, seed):
    import torch
    p = (logits.detach().float().cpu() / temperature).softmax(-1)
    generator = torch.Generator().manual_seed(seed)
    token = int(torch.multinomial(p, 1, generator=generator))
    return token, float(p[token])


def break_token(logits, current):
    alternative = logits.float().clone()
    alternative[current] = float('-inf')
    return int(alternative.argmax())


def groups(study):
    result = [{'id': name, 'group': name, 'rollback': name in ('G5', 'GR')}
              for name in ('G0', 'G1', 'G3', 'G5', 'GR')]
    for group in ('G2', 'G4'):
        for temperature in study['temperatures']:
            for seed in study['sampling_seeds']:
                result.append({'id': f'{group}-T{temperature:g}-s{seed}', 'group': group,
                    'rollback': group == 'G4', 'temperature': temperature, 'seed': seed})
    return result


def capped_tokens(row, events, cap):
    if row['forward_calls'] <= cap:
        return row['token_ids'], row['finish_reason']
    prompt = len(row['state']['prompt_ids'])
    end = max([prompt] + [e['block_start']+32 for e in events[:cap] if 0 not in e['mask_after']])
    return row['token_ids'][:end-prompt], 'forward_budget'


def make_case(row, events, pattern, cap):
    prompt = len(row['state']['prompt_ids'])
    boundary = (prompt+pattern['onset'])//32*32
    local = next(e for e in events if e['forward'] == pattern['forward']+1)
    rewind = next(e for e in events if e['block_start'] == boundary)
    tokens, reason = capped_tokens(row, events, cap)
    return {'question_id': row['question_id'], 'cohort': row['cohort'], 'pattern': pattern,
        'rollback_block': boundary, 'local': snapshot(row, events, local['forward']),
        'rewind': snapshot(row, events, rewind['forward']),
        'original_events': [{k: e[k] for k in (*CORE, 'forward', 'phase', 'compute_kind')} for e in events[:cap]],
        'reference_tokens': tokens, 'reference_reason': reason,
        'trace_path': row['trace_path'], 'original_forward_calls': row['forward_calls']}


def prepare(out, config_path=GA/'repetition_recovery.json'):
    from run import fingerprint
    study = read(config_path)
    parent = read(GA / 'runs/pilot8192-20261110/manifest.json')
    current = fingerprint(dict(parent['config'], model=study['model']))
    previous = parent['fingerprint']
    assert current['data'] == previous['data']
    model_changes = {p: {'old': previous['model'].get(p), 'current': current['model'].get(p)}
                     for p in set(current['model']) | set(previous['model'])
                     if current['model'].get(p) != previous['model'].get(p)}
    assert set(model_changes) <= {'model-00002-of-00002.safetensors'}, model_changes
    source_changes = {p: {'old': previous['code'].get(p), 'current': current['code'].get(p)}
                      for p in set(current['code']) | set(previous['code'])
                      if current['code'].get(p) != previous['code'].get(p)}
    # Training has evolved independently; inference, grading, and the old adapter stay fixed.
    assert all(p.startswith('sdar_justgmpo/') and '/vendor/' not in p
               and Path(p).name != 'reward.py' for p in source_changes), source_changes
    preflight = read(GA / 'runs/r1-preflight-20261112/preflight.json')
    selected = {r['question_id'] for r in preflight['cases']}
    rows = {r['question_id']: r for r in all_rows(OLD) if r['setting'] == 'T8'}
    cases = []
    for q in sorted(selected):
        row, detector = rows[q], Detector(len(rows[q]['state']['prompt_ids']))
        events = events_for(row)
        for e in events:
            if detector.observe(e):
                break
        pattern = detector.found
        assert pattern['forward'] == next(c['trigger_forward'] for c in preflight['cases'] if c['question_id'] == q)
        cases.append(make_case(row, events, pattern, study['forward_cap']))
    engine = dict(parent['config'], model=study['model'], trace=True, max_new_tokens=study['max_new_tokens'], batch_size=1)
    assert not (out / 'manifest.json').exists()
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / 'cases.json', cases)
    write_json(out / 'data.json', read(OLD / 'data.json'))
    write_json(out / 'manifest.json', {'study': study, 'engine': engine, 'groups': groups(study),
        'code': {name: file_hash(GA/name) for name in CODE},
        'dependencies': {name: file_hash(GA/name) for name in ('engine.py', 'early.py', 'protocol.py', 'degeneration.py')},
        'cases_sha256': file_hash(out/'cases.json'), 'original_fingerprint': current,
        'preexisting_training_changes': source_changes,
        'model_changes': model_changes,
        'model_decision': ('Local weights differ from history; exact Current validation is required.' if model_changes
                           else 'All model files match the historical official checkpoint; exact Current validation is still required.'),
        'study_config': {'path': str(config_path), 'sha256': file_hash(config_path)},
        'model_provenance': read(GA/'cache/r1-model-provenance.json'),
        'preflight_sha256': file_hash(GA/'runs/r1-preflight-20261112/preflight.json'),
        'source_hashes': {c['trace_path']: file_hash(c['trace_path']) for c in cases},
        'protocol': 'R1_PLAN.md; one intervention, one local position; full-vocabulary categorical sampling; all sunk and replay forwards counted.'})
    (out/'source').mkdir()
    for name in CODE:
        shutil.copy2(GA/name, out/'source'/name)
    shutil.copy2(config_path, out/'source/study.json')
    print(json.dumps({'questions': len(cases), 'paths': len(cases)*len(groups(study))}), flush=True)


def collect_current(gpu, out):
    from joint50 import environment, score
    environment(gpu)
    from engine import Backend
    from lmdeploy.utils import get_logger
    get_logger('lmdeploy').setLevel('ERROR')
    out = Path(out)
    manifest = read(out/'manifest.json')
    study = manifest['study']
    cases = read(out/'cases.json')
    data = {r['unique_id']: r for r in read(out/'data.json')}
    backend, grader = Backend(manifest['engine'], T8), Grader(1)
    path = out/f'natural-gpu{gpu}.jsonl'
    done = {r['question_id'] for r in read_jsonl(path)} if path.exists() else set()
    try:
        for index, case in enumerate(cases):
            q = case['question_id']
            if study['gpus'][index % len(study['gpus'])] != gpu or q in done:
                continue
            trace = out/f'natural-{q.replace("/", "_")}.jsonl.gz'
            result = backend.generate([{'id': q, 'question': data[q]['problem'],
                'future_budget': study['forward_cap']}], trace)[0]
            events = result.pop('events')
            result['state'] = {'prompt_ids': backend.prompt(data[q]['problem'])}
            detector = Detector(len(result['state']['prompt_ids']))
            for e in events:
                if detector.observe(e):
                    break
            length = min(len(events), len(case['original_events']))
            result.update(question_id=q, cohort=case['cohort'], gpu=gpu, pattern=detector.found,
                trace_path=str(trace), inference_seconds=backend.last_seconds,
                old_trace_difference=trace_difference(case['original_events'][:length], events[:length]))
            result.update(score(grader, result, data[q]['answer']))
            with path.open('a') as f:
                f.write(json.dumps(result,ensure_ascii=False)+'\n')
            print(json.dumps({'q': q, 'reward': result['reward'], 'forward': result['forward_calls'],
                              'repeat': detector.found is not None, 'old_difference': result['old_trace_difference']}), flush=True)
    finally:
        grader.close(); backend.close()


def refresh(out):
    manifest = read(out/'manifest.json')
    rows = [r for p in out.glob('natural-gpu*.jsonl') for r in read_jsonl(p)]
    assert len(rows) == 14
    archive = out/'old-state-attempt'
    archive.mkdir(exist_ok=True)
    for name in ('cases.json', 'manifest.json', 'pilot.log'):
        shutil.copy2(out/name, archive/name)
    cases, missing = [], []
    cap = manifest['study']['forward_cap']
    for r in sorted(rows,key=lambda r:r['question_id']):
        pattern = r['pattern']
        if pattern is None or pattern['forward'] >= r['forward_calls']:
            missing.append({'question_id': r['question_id'], 'reason': 'no nonterminal repetition trigger'})
            continue
        case = make_case(r, read_jsonl(r['trace_path']), pattern, cap)
        if cap-pattern['forward'] < case['rewind']['source_forward']+1:
            missing.append({'question_id':r['question_id'], 'reason':'insufficient rollback rebuild budget'})
            continue
        cases.append(case)
    write_json(out/'cases.json', cases)
    write_json(out/'missing.json', missing)
    manifest['cases_sha256'] = file_hash(out/'cases.json')
    manifest['current_state_refresh'] = {'reason':'Old Current failed exact native replay under user-selected local weights',
        'natural_paths':14,'eligible_repetition_states':len(cases),'missing':missing,
        'physical_forwards':sum(r['forward_calls'] for r in rows)}
    pilot = [q for q in manifest['study']['pilot_ids'] if any(c['question_id'] == q for c in cases)]
    ordered = sorted(cases, key=lambda c:c['pattern']['forward'])
    for c in ordered[:1]+ordered[-1:]+ordered:
        if len(pilot) >= min(3,len(cases)):
            break
        if c['question_id'] not in pilot:
            pilot.append(c['question_id'])
    manifest['study']['pilot_ids'] = pilot
    manifest['current_state_refresh']['pilot_rule'] = 'Retain original pilot when eligible; fill with earliest then latest eligible confirmations, using pre-intervention information only.'
    manifest['source_hashes'] = {r['trace_path']:file_hash(r['trace_path']) for r in rows}
    for name in CODE:
        manifest['code'][name] = file_hash(GA/name)
        shutil.copy2(GA/name,out/'source'/name)
    write_json(out/'manifest.json',manifest)
    print(json.dumps(manifest['current_state_refresh']),flush=True)


def backend_class():
    from engine import Backend

    class RecoveryBackend(Backend):
        def _install(self):
            super()._install()
            strategy = self.agent.agent_strategy
            native_post = type(strategy).post_sampling

            def post(inputs, logits, next_ids, extra):
                assert len(inputs.model_metas) == 1
                track = self.tracks[inputs.model_metas[0]['ga_session']]
                job, state = track.job, track.job['snapshot']
                group, pattern = job['arm']['group'], job['pattern']
                active = len(track.events) >= len(state['replay'])
                at_source = active and not track.source_reached
                if active and group == 'G1':
                    self.set_setting(D95)
                base = int((inputs.history_lengths+inputs.seq_length-self.block)[0])
                before = extra.dllm_mask.view(-1, self.block).clone()
                ids = strategy.slice_outputs(inputs.input_ids.flatten(), inputs.seq_length).view(-1, self.block)
                proposed = next_ids.view(-1, self.block).clone()
                output, extra = native_post(strategy, inputs, logits, next_ids, extra)
                output, after = output.view(-1, self.block), extra.dllm_mask.view(-1, self.block)
                if not active:
                    replay = state['replay'][len(track.events)]
                    assert before[0].tolist() == replay['mask_before'] and ids[0].tolist() == replay['input_ids']
                    assert after[0].tolist() == replay['mask_after']
                    assert [int(output[0,i]) for i in replay['positions']] == replay['tokens']
                    after[0].copy_(after.new_tensor(replay['mask_after']))
                    for i, token in zip(replay['positions'], replay['tokens']):
                        output[0, i] = token
                if at_source:
                    assert ids[0].tolist() == state['input_ids'] and before[0].tolist() == state['mask']
                    track.search_steps, track.perturbation = 0, None
                    track.sample_seed = stable_seed(job['question_id'], job['arm'].get('seed', 0))
                mutation = None
                if active and group in ('G2', 'G3', 'G4', 'G5') and track.perturbation is None:
                    if bool((before[0] == 0).any()):
                        track.search_steps += 1
                        eligible = (base == state['block_start']) if job['arm']['rollback'] else track.search_steps <= job['search_steps']
                        event = {'block_start': base, 'positions': [i for i in range(32) if before[0,i] == 0 and after[0,i] == 1],
                                 'current_ids': proposed[0].tolist()}
                        position = target_position(event, pattern, len(track.prompt_ids), job['arm']['rollback']) if eligible else None
                        if position is not None:
                            sampling_started = time.monotonic()
                            values = logits.view(-1, logits.shape[-1])[position]
                            old = int(proposed[0,position])
                            p = values.float().softmax(-1)
                            if group in ('G2', 'G4'):
                                token, sampling_p = draw(values, job['arm']['temperature'], track.sample_seed)
                            else:
                                token = break_token(values, old)
                                sampling_p = float(p[token])
                            output[0,position] = token
                            mutation = {'position': position, 'old_token': old, 'new_token': token,
                                'changed': token != old, 'original_p': float(p[old]), 'new_p': float(p[token]),
                                'sampling_p': sampling_p, 'seed': track.sample_seed,
                                'temperature': job['arm'].get('temperature'), 'physical_forward': track.forward_calls,
                                'total_forward': track.forward_calls+job['sunk_forward'],
                                'sampling_seconds': time.monotonic()-sampling_started}
                            track.perturbation = mutation
                self._record(track, base, ids[0].tolist(), before[0].tolist(), after[0].tolist(),
                             proposed[0].tolist(), output[0].tolist(), inputs.is_decoding)
                track.events[-1].update(total_forward=track.forward_calls+job['sunk_forward'], mutation=mutation,
                                       elapsed_seconds=time.monotonic()-self.request_started)
                if self.writer:
                    self.writer.write(json.dumps(track.events[-1], separators=(',', ':'))+'\n')
                if at_source:
                    track.source_reached = True
                extra.dllm_mask = after.flatten()
                return output.flatten(), extra

            self._replace(strategy, 'post_sampling', post)

        def _record(self, track, *args):
            writer, self.writer = self.writer, None
            try:
                super()._record(track, *args)
            finally:
                self.writer = writer
            if not track.job['arm']['rollback']:
                if not hasattr(track, 'detector'):
                    track.detector = Detector(len(track.prompt_ids))
                if track.detector.found is None:
                    started = time.monotonic()
                    found = track.detector.observe(track.events[-1])
                    track.events[-1]['detector_seconds'] = time.monotonic()-started
                    if found:
                        assert found == track.job['pattern'], (found, track.job['pattern'])

        async def _one(self, job):
            self.set_setting(T8)
            self.request_started = time.monotonic()
            result = await super()._one(job)
            # The inherited result filters replay events; the disk trace retains every paid call.
            return result

    return RecoveryBackend


def audit(result, events, case, arm, cap):
    state = case['rewind' if arm['rollback'] else 'local']
    sunk = case['pattern']['forward'] if arm['rollback'] else 0
    assert len(events) == result['forward_calls']
    assert sum(result['forward_types'].values()) == result['forward_calls']
    assert result['forward_calls']+sunk <= cap
    assert trace_difference(state['replay'], events[:len(state['replay'])]) is None
    source = events[len(state['replay'])]
    assert source['block_start'] == state['block_start'] and source['mask_before'] == state['mask']
    known, denoise, mutations = dict(enumerate(state['prompt_ids'])), 0, []
    for index, e in enumerate(events):
        assert e['forward'] == index+1 and e['total_forward'] == e['forward']+sunk
        denoise += e['phase'] == 'denoise'
        if e['mutation']:
            mutations.append(e['mutation'])
            i = e['mutation']['position']
            assert i in e['positions'] and e['tokens'][e['positions'].index(i)] == e['mutation']['new_token']
        for i, token in zip(e['positions'], e['tokens']):
            absolute = e['block_start']+i
            assert absolute not in known and e['mask_before'][i] == 0 and e['mask_after'][i] == 1
            known[absolute] = token
        if index and e['block_start'] == events[index-1]['block_start']:
            last = events[index-1]
            expected = list(last['input_ids'])
            for i, token in zip(last['positions'], last['tokens']):
                expected[i] = token
            assert e['input_ids'] == expected and e['mask_before'] == last['mask_after']
    assert len(mutations) <= 1
    p = len(state['prompt_ids'])
    assert result['token_ids'] == [known[p+i] for i in range(len(result['token_ids']))]
    if arm['group'] in ('G0', 'GR'):
        assert trace_difference(case['original_events'][:len(events)], events) is None
        assert result['token_ids'] == case['reference_tokens'][:len(result['token_ids'])]
        if arm['group'] == 'G0':
            assert result['token_ids'] == case['reference_tokens'] and result['finish_reason'] == case['reference_reason']
    original_denoise = sum(e['phase'] == 'denoise' for e in case['original_events'][:sunk])
    return {'passed': True, 'events': len(events), 'mutation': mutations[0] if mutations else None,
            'denoise_total': denoise+original_denoise,
            'prefix_rebuild': state['source_forward']-1 if arm['rollback'] else 0}


def worker(gpu, out, phase):
    from joint50 import environment, score
    environment(gpu)
    backend_type = backend_class()
    from lmdeploy.utils import get_logger
    get_logger('lmdeploy').setLevel('ERROR')
    out = Path(out)
    manifest = read(out/'manifest.json')
    study, engine = manifest['study'], manifest['engine']
    for name, digest in manifest['code'].items():
        assert file_hash(GA/name) == digest, name
    data = {r['unique_id']: r for r in read(out/'data.json')}
    cases = read(out/'cases.json')
    if phase == 'pilot':
        cases = [c for c in cases if c['question_id'] in study['pilot_ids']]
    else:
        cases = [c for i, c in enumerate(cases) if study['gpus'][i % len(study['gpus'])] == gpu]
    backend = backend_type(engine, T8)
    grader = Grader(engine['grade_timeout'])
    record_path = out/f'results-{phase}-gpu{gpu}.jsonl'
    paths = set(out.glob('results-pilot-*.jsonl')) | ({record_path} if record_path.exists() else set())
    done = {(r['question_id'], r['arm']['id']) for path in paths for r in read_jsonl(path)}
    completed = 0
    try:
        for case in cases:
            q, pattern = case['question_id'], case['pattern']
            arms = list(manifest['groups'])
            random.Random(stable_seed(q, study['seed'])).shuffle(arms)
            # Always establish restoration controls before the intervention arms.
            arms.sort(key=lambda a: 0 if a['group'] in ('G0', 'GR') else 1)
            for arm in arms:
                if (q, arm['id']) in done:
                    continue
                state = case['rewind' if arm['rollback'] else 'local']
                sunk = pattern['forward'] if arm['rollback'] else 0
                name = q.replace('/', '_').replace('.json', '')+'-'+arm['id']
                trace = out/'traces'/f'{name}.jsonl.gz'
                trace.parent.mkdir(exist_ok=True)
                job = {'id': name, 'question_id': q, 'snapshot': state, 'arm': arm, 'pattern': pattern,
                       'sunk_forward': sunk, 'search_steps': study['search_steps'],
                       'future_budget': study['forward_cap']-sunk-state['source_forward']}
                result = backend.generate([job], trace)[0]
                elapsed = backend.last_seconds
                result.pop('events'); result.pop('state')
                events = read_jsonl(trace)
                checks = audit(result, events, case, arm, study['forward_cap'])
                suffix_start = case['rollback_block']-state['prompt_length'] if arm['rollback'] else pattern['observed_tokens']
                suffix_runs = repeat_runs(result['token_ids'][suffix_start:])
                result.update(question_id=q, cohort=case['cohort'], arm=arm, gpu=gpu,
                    trace_path=str(trace), total_forward=result['forward_calls']+sunk, sunk_forward=sunk,
                    prefix_rebuild_forward=checks['prefix_rebuild'], denoise_forward=checks['denoise_total'],
                    tpf_denoise=len(result['token_ids'])/checks['denoise_total'],
                    tpf_total=len(result['token_ids'])/(result['forward_calls']+sunk),
                    validation=checks, inference_seconds=elapsed, suffix_start=suffix_start,
                    detector_seconds=sum(e.get('detector_seconds', 0.) for e in events),
                    trigger_seconds=events[pattern['forward']-1]['elapsed_seconds'] if not arm['rollback'] else None,
                    suffix_repetitions=suffix_runs, escape=bool(result['finished'] and not suffix_runs))
                result.update(score(grader, result, data[q]['answer']))
                with record_path.open('a') as handle:
                    handle.write(json.dumps(result, ensure_ascii=False)+'\n')
                completed += 1
                print(json.dumps({'gpu': gpu, 'q': q, 'arm': arm['id'], 'reward': result['reward'],
                    'escape': result['escape'], 'total_forward': result['total_forward'],
                    'changed': checks['mutation']['changed'] if checks['mutation'] else None}), flush=True)
    finally:
        grader.close()
        backend.close()
    return {'gpu': gpu, 'completed': completed}


def report(out):
    import numpy as np
    rows = [r for p in sorted(out.glob('results-*.jsonl')) for r in read_jsonl(p)]
    assert len(rows) == len({(r['question_id'], r['arm']['id']) for r in rows})
    grouped = defaultdict(list)
    trigger_seconds = {r['question_id']: r['trigger_seconds'] for r in rows if r['arm']['group'] == 'G0'}
    for r in rows:
        arm = r['arm']
        name = arm['group'] + (f"-T{arm['temperature']:g}" if 'temperature' in arm else '')
        grouped[name].append(r)
    summaries = {}
    for name, group in sorted(grouped.items()):
        questions = sorted({r['question_id'] for r in group})
        byq = {q: [r for r in group if r['question_id'] == q] for q in questions}
        question_rewards = np.array([np.mean([r['reward'] for r in byq[q]]) for q in questions])
        rng = np.random.default_rng(20261112)
        interval = np.quantile(rng.choice(question_rewards, (2000, len(questions))).mean(1), [.025,.975])
        summaries[name] = {'questions': len(questions), 'paths': len(group),
            'reward_rate_question_bootstrap95': interval.tolist(),
            'correct_expected_questions': sum(np.mean([r['reward'] for r in v]) for v in byq.values()),
            'escape_expected_questions': sum(np.mean([r['escape'] for r in v]) for v in byq.values()),
            'mean_forward': float(np.mean([r['total_forward'] for r in group])),
            'mean_tpf_denoise': float(np.mean([r['tpf_denoise'] for r in group])),
            'mean_tpf_total': float(np.mean([r['tpf_total'] for r in group])),
            'aggregate_tpf_denoise': sum(len(r['token_ids']) for r in group)/sum(r['denoise_forward'] for r in group),
            'aggregate_tpf_total': sum(len(r['token_ids']) for r in group)/sum(r['total_forward'] for r in group),
            'mean_prefix_rebuild_forward': float(np.mean([r['prefix_rebuild_forward'] for r in group])),
            'mean_inference_seconds': float(np.mean([r['inference_seconds'] for r in group])),
            'mean_strategy_seconds_estimate': float(np.mean([r['inference_seconds']+
                (trigger_seconds[r['question_id']] if r['arm']['rollback'] else 0.) for r in group])),
            'forward_budget': sum(r['finish_reason'] == 'forward_budget' for r in group),
            'changed': sum(bool(r['validation']['mutation'] and r['validation']['mutation']['changed']) for r in group),
            'per_question_correct': {q: sum(r['reward'] for r in v) for q,v in byq.items()},
            'per_seed': {str(seed): {'correct': sum(r['reward'] for r in group if r['arm'].get('seed') == seed),
                                    'escape': sum(r['escape'] for r in group if r['arm'].get('seed') == seed)}
                         for seed in sorted({r['arm']['seed'] for r in group if 'seed' in r['arm']})}}
    summary = {'paths': len(rows), 'questions': len({r['question_id'] for r in rows}),
        'new_physical_forwards': sum(r['forward_calls'] for r in rows),
        'strategy_forwards': sum(r['total_forward'] for r in rows),
        'inference_seconds_sum': sum(r['inference_seconds'] for r in rows),
        'grading_timeouts': sum(r['grading'] == 'timeout' for r in rows), 'groups': summaries}
    write_json(out/'summary.json', summary)
    lines = ['# R1：高置信度重复恢复（自动指标）', '',
        f"原14题在当前本地权重下重跑后，{summary['questions']}题触发重复，完成{len(rows)}条分支。每条路径512整题forward、8192 tokens。随机组按种子平均，表中正确数为期望题数；数学复核与最终解释见ANALYSIS.md。", '',
        '| 组别 | 路径 | 正确期望题数 | 逃逸期望题数 | 平均总F | 平均TPF去噪 / 全部 | 实际换词 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name,s in summaries.items():
        lines.append(f"| {name} | {s['paths']} | {s['correct_expected_questions']:.2f} | {s['escape_expected_questions']:.2f} | {s['mean_forward']:.1f} | {s['mean_tpf_denoise']:.2f} / {s['mean_tpf_total']:.2f} | {s['changed']} |")
    lines += ['', f"已完成{len(rows)}条路径，新增物理forward合计{summary['new_physical_forwards']}，策略口径{summary['strategy_forwards']}；判分超时{summary['grading_timeouts']}。", '',
              '退出重复、原reward和有效数学恢复须分开解释；本表不是14道未筛选题的总体正确率。详细逐题与种子结果见summary.json，数学复核和最终结论另行补充。', '']
    (out/'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=('prepare', 'collect', 'refresh', 'pilot', 'run', 'report'))
    p.add_argument('--out', type=Path, default=OUT)
    p.add_argument('--gpu', type=int, default=6)
    p.add_argument('--config', type=Path, default=GA/'repetition_recovery.json')
    args = p.parse_args()
    if args.stage == 'prepare':
        prepare(args.out, args.config)
    elif args.stage == 'collect':
        cards = read(args.out/'manifest.json')['study']['gpus']
        with ProcessPoolExecutor(len(cards), mp_context=mp.get_context('spawn')) as pool:
            for f in [pool.submit(collect_current,gpu,args.out) for gpu in cards]:
                f.result()
    elif args.stage == 'refresh':
        refresh(args.out)
    elif args.stage == 'pilot':
        worker(args.gpu, args.out, 'pilot')
        rows = [r for path in args.out.glob('results-pilot-*.jsonl') for r in read_jsonl(path)]
        manifest = read(args.out/'manifest.json')
        pilot_count = sum(c['question_id'] in manifest['study']['pilot_ids'] for c in read(args.out/'cases.json'))
        assert len(rows) == pilot_count*len(manifest['groups'])
        write_json(args.out/'pilot_verification.json', {'passed': all(r['validation']['passed'] for r in rows), 'paths': len(rows)})
    elif args.stage == 'run':
        assert read(args.out/'pilot_verification.json')['passed']
        cards = read(args.out/'manifest.json')['study']['gpus']
        with ProcessPoolExecutor(len(cards), mp_context=mp.get_context('spawn')) as pool:
            futures = [pool.submit(worker, gpu, args.out, 'main') for gpu in cards]
            for future in futures:
                print(future.result(), flush=True)
    else:
        report(args.out)


if __name__ == '__main__':
    main()
