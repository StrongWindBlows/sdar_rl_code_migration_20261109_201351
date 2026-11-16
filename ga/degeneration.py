"""Natural SDAR trajectories and causal, question-held-out degeneration forecasts."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import gzip
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import sys
import time

import numpy as np

from protocol import Grader, file_hash, paired_stats, read_jsonl, trace_difference, write_json

GA = Path(__file__).resolve().parent
PILOT = GA / 'runs/pilot8192-20261110'
FIFTY = GA / 'runs/joint50-20261111'
DEFAULT_OUT = GA / 'runs/degeneration-20261112'
SEED = 20261112


def read_json(path):
    return json.loads(Path(path).read_text())


def source_rows():
    rows = {}
    for path in sorted((PILOT / 'pilot').glob('exp0-gpu*.json')):
        for row in read_json(path):
            if row['setting'] in ('T8', 'Dynamic'):
                trace = path.parent / f"exp0-{row['setting']}-gpu{row['gpu']}.trace.jsonl.gz"
                rows[row['setting'], row['question_id']] = dict(row,
                    trace_path=str(trace), source_file=str(path), origin='pilot8192')
    for path in sorted(FIFTY.glob('baseline-gpu*.json')):
        for row in read_json(path):
            key = row['setting'], row['question_id']
            if row['setting'] == 'T8' and key not in rows:
                rows[key] = dict(row, source_file=str(path), origin='joint50')
    return rows


def events_for(row):
    return [e for e in read_jsonl(row['trace_path']) if e['request_id'] == row['request_id']]


def prepare(out):
    from run import fingerprint
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / 'manifest.json').exists()
    old = read_json(PILOT / 'manifest.json')
    check = fingerprint(old['config'])
    assert check == old['fingerprint'], 'Original code/model/data fingerprint changed'
    config = dict(old['config'], gpus=list(range(5)), scope='degeneration50')
    config['settings'] = [s for s in config['settings'] if s['name'] in ('T8', 'Dynamic')]
    data = read_json(FIFTY / 'data.json')
    sources = source_rows()
    reused, jobs = [], []
    for row in data:
        q = row['unique_id']
        for setting in ('T8', 'Dynamic'):
            old_row = sources.get((setting, q))
            if old_row and old_row['finish_reason'] != 'forward_budget':
                reused.append(dict(old_row, cohort=row['cohort'], new_forward_calls=0))
            else:
                jobs.append({'question_id': q, 'setting': setting, 'cohort': row['cohort'],
                             'prefix_reference': old_row})
    # Long T8 paths avoid the busy GPU 3; each GPU receives Dynamic work.
    for setting, cards in [('T8', [4, 0, 1, 2]), ('Dynamic', list(range(5)))]:
        for i, job in enumerate(j for j in jobs if j['setting'] == setting):
            job['gpu'] = cards[i % len(cards)]
    ids = sorted(r['unique_id'] for r in data)
    np.random.default_rng(SEED).shuffle(ids)
    folds = {q: i % 5 for i, q in enumerate(ids)}
    write_json(out / 'data.json', data)
    write_json(out / 'jobs.json', jobs)
    write_json(out / 'reused.json', reused)
    write_json(out / 'folds.json', folds)
    write_json(out / 'source_fingerprint.json', check)
    paths = {Path(r['source_file']) for r in sources.values()}
    paths |= {Path(r['trace_path']) for r in sources.values()}
    paths |= {FIFTY / 'selection.json', FIFTY / 'data.json', PILOT / 'manifest.json'}
    write_json(out / 'manifest.json', {'date': '2026-11-12', 'config': config,
        'new_paths': len(jobs), 'reused_paths': len(reused), 'seed': SEED,
        'source_hashes': {str(p): file_hash(p) for p in sorted(paths)},
        'adapter_sha256': file_hash(GA / 'engine.py'),
        'protocol': {'period': 256, 'copies': 4, 'min_tokens': 32,
            'horizons_blocks': [1, 2, 4], 'strict_gap_tokens': 32,
            'cv': 'Five question folds: test=f, calibration=(f+1)%5, train=other three.',
            'logistic_C': 1.0, 'trajectory_false_alarm_targets': [0.05, 0.10],
            'gate': 'History/proposal model must improve strict event recall over all simple controls at observed <=10% clean-trajectory false alarms, cover >=3 independent questions including >=2 new40, before considering intervention.',
            'scope': 'Existing development50 only; evaluation50 and final400 untouched.'}})
    (out / 'source').mkdir()
    for name in ('degeneration.py', 'test_degeneration.py', 'DEGENERATION_PLAN.md'):
        shutil.copy2(GA / name, out / 'source' / name)
    print(json.dumps({'new': len(jobs), 'reuse': len(reused), 'gpus': dict(Counter(j['gpu'] for j in jobs))}))


def collect_worker(gpu, out):
    from joint50 import environment, score
    environment(gpu)
    from engine import Backend
    from lmdeploy.utils import get_logger
    get_logger('lmdeploy').setLevel('ERROR')
    out = Path(out)
    config = read_json(out / 'manifest.json')['config']
    settings = {s['name']: s for s in config['settings']}
    data = {r['unique_id']: r for r in read_json(out / 'data.json')}
    jobs = [j for j in read_json(out / 'jobs.json') if j['gpu'] == gpu]
    path = out / f'collected-gpu{gpu}.json'
    results = read_json(path) if path.exists() else []
    done = {(r['setting'], r['question_id']) for r in results}
    backend = Backend(config, settings['T8'])
    grader = Grader(config['grade_timeout'])
    sources = source_rows()
    checks = []
    try:
        for setting in ('T8', 'Dynamic'):
            backend.set_setting(settings[setting])
            reference = next(r for (s, _), r in sources.items() if s == setting and r['origin'] == 'pilot8192')
            q = reference['question_id']
            trace = out / f'check-{setting}-gpu{gpu}.trace.jsonl.gz'
            result = backend.generate([{'id': f'check-{setting}-{gpu}', 'question': data[q]['problem'],
                                        'max_new_tokens': 128}], trace)[0]
            expected = events_for(reference)
            assert result['token_ids'] == reference['token_ids'][:len(result['token_ids'])]
            assert trace_difference(expected[:len(result['events'])], result['events']) is None
            checks.append({'setting': setting, 'question_id': q, 'passed': True,
                'forward_calls': result['forward_calls'], 'seconds': backend.last_seconds})
            write_json(out / f'checks-gpu{gpu}.json', checks)
            for job in [j for j in jobs if j['setting'] == setting]:
                q = job['question_id']
                if (setting, q) in done:
                    continue
                name = f"natural-{setting}-{q.replace('/', '_').replace('.json', '')}"
                trace = out / f'{name}.trace.jsonl.gz'
                result = backend.generate([{'id': name, 'question': data[q]['problem']}], trace)[0]
                elapsed, events = backend.last_seconds, result.pop('events')
                reference = job['prefix_reference']
                if reference:
                    old_events = events_for(reference)
                    assert trace_difference(old_events, events[:len(old_events)]) is None, (setting, q, 'prefix trace')
                    assert result['token_ids'][:len(reference['token_ids'])] == reference['token_ids']
                result['state'] = {'prompt_ids': backend.prompt(data[q]['problem'])}
                result.update(question_id=q, setting=setting, cohort=job['cohort'], gpu=gpu,
                    trace_path=str(trace), origin='new', source_file=str(path), inference_seconds=elapsed,
                    new_forward_calls=result['forward_calls'], prefix_512_match=bool(reference))
                result.update(score(grader, result, data[q]['answer']))
                results.append(result)
                write_json(path, results)
                print(json.dumps({'gpu': gpu, 'setting': setting, 'q': q, 'reward': result['reward'],
                    'forward': result['forward_calls'], 'finish': result['finish_reason']}), flush=True)
    finally:
        grader.close()
        backend.close()
    return {'gpu': gpu, 'paths': len(results)}


def repeat_runs(ids, max_period=256, min_tokens=32):
    """Exact tandem repeats; all coordinates are in token order, not commit order."""
    ids = np.asarray(ids)
    runs = []
    for p in range(1, min(max_period, len(ids) // 4) + 1):
        equal = ids[p:] == ids[:-p]
        changes = np.diff(np.r_[False, equal, False].astype(np.int8))
        for start, end in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)):
            if end - start + p >= max(4 * p, min_tokens):
                runs.append({'motif_start': int(start), 'onset': int(start + p), 'period': p,
                    'end_exclusive': int(end + p), 'confirmation_end': int(start + max(4 * p, min_tokens))})
    return sorted(runs, key=lambda r: (r['onset'], r['period']))


def align_event(run, commits, prompt_length, events):
    r = dict(run)
    start, onset, p = r['motif_start'], r['onset'], r['period']
    r['onset_commit'] = commits[prompt_length + onset][0]
    r['first_repeat_commit'] = min(commits[prompt_length + i][0] for i in range(onset, onset + p))
    r['confirmation_forward'] = max(commits[prompt_length + i][0] for i in range(start, r['confirmation_end']))
    r['onset_block'] = (prompt_length + onset) // 32
    return r


def repetition_features(ids):
    ids = list(ids)[-256:]
    if not ids:
        return [0., 0., 1., 0.]
    def repeated(n):
        words = [tuple(ids[i:i+n]) for i in range(len(ids) - n + 1)]
        return 1 - len(set(words)) / len(words) if words else 0.
    tail = 1
    for value in reversed(ids[:-1]):
        if value != ids[-1]:
            break
        tail += 1
    return [repeated(2), repeated(4), len(set(ids)) / len(ids), min(tail, 32) / 32]


NAMES = ['log_length', 'log_forward', 'mask_fraction', 'commit_fraction', 'round_fraction', 'remaining_length',
    'text_rep2', 'text_rep4', 'text_diversity', 'text_tail_run',
    'p_min', 'p_q25', 'p_mean', 'entropy_mean', 'entropy_q75', 'margin_min', 'margin_mean',
    'selected_p_min', 'selected_p_mean', 'selected_entropy', 'selected_margin_min',
    'has_history', 'flip_fraction', 'selected_flip', 'history_support', 'p_change_abs', 'p_change', 'stable_rounds',
    'proposal_rep2', 'proposal_rep4', 'proposal_diversity', 'proposal_tail_run', 'proposal_max_fraction']


class FeatureStream:
    """Causal state: observe a forward, compute features, then apply its commits."""
    def __init__(self, prompt_ids, steps):
        self.prompt_length = len(prompt_ids)
        self.known = dict(enumerate(prompt_ids))
        self.steps = steps
        self.previous = None
        self.base = None
        self.stable = np.zeros(32)

    def observe(self, e):
        base = e['block_start']
        if base != self.base:
            self.base, self.previous = base, None
            self.stable = np.zeros(32)
            self.prefix = [self.known[i] for i in range(self.prompt_length, base)]
            self.text = repetition_features(self.prefix)
        masked = np.asarray(e['mask_before']) == 0
        if e['phase'] != 'denoise' or not masked.any():
            self.apply(e)
            return None
        ids = np.asarray(e['current_ids'])
        probabilities = np.asarray(e['topk_probs'])
        p, margin = probabilities[:, 0], probabilities[:, 0] - probabilities[:, 1]
        entropy = np.asarray(e['entropy'])
        selected = np.asarray(e['positions'], dtype=int)
        assert len(selected), 'Natural static/dynamic decode must make progress'
        generation_end = max(0, base + 32 - self.prompt_length)
        x = [np.log1p(generation_end), np.log1p(e['forward']), masked.mean(), len(selected)/32,
             e['round']/self.steps, max(0, 8192-generation_end)/8192, *self.text,
             p[masked].min(), np.quantile(p[masked], .25), p[masked].mean(), entropy[masked].mean(),
             np.quantile(entropy[masked], .75), margin[masked].min(), margin[masked].mean(),
             p[selected].min(), p[selected].mean(), entropy[selected].mean(), margin[selected].min()]
        if self.previous is None:
            history = [0.] + [float('nan')]*6
            self.stable[:] = 1
        else:
            previous_ids, previous_p = self.previous
            flip = ids != previous_ids
            self.stable = np.where(flip, 1, self.stable + 1)
            now = np.asarray(e['history_probability_now'])
            history = [1., flip[masked].mean(), flip[selected].mean(), now[masked].mean(),
                       np.abs(p[masked]-previous_p[masked]).mean(),
                       (p[masked]-previous_p[masked]).mean(), self.stable[masked].mean()/32]
        proposal = [int(ids[i]) if masked[i] else e['input_ids'][i] for i in range(32)
                    if base+i >= self.prompt_length]
        x += history + repetition_features(self.prefix[-256:] + proposal)
        x += [max(Counter(proposal).values()) / len(proposal)]
        assert len(x) == len(NAMES)
        self.previous = ids.copy(), p.copy()
        self.apply(e)
        return {'forward': e['forward'], 'generation_end': generation_end, 'block_start': base,
                'round': e['round'], 'x': [float(v) for v in x]}

    def apply(self, e):
        for i, token in zip(e['positions'], e['tokens']):
            absolute = e['block_start'] + i
            assert absolute not in self.known, ('repeat commit', absolute)
            self.known[absolute] = token


def target(feature, record, horizon=4):
    """Only pre-onset states; incomplete negative follow-up stays censored."""
    event = record['event']
    if event:
        if feature['forward'] >= event['first_repeat_commit']:
            return None
        return int(event['onset'] <= feature['generation_end'] + 32*horizon)
    if record['finish_reason'] == 'eos':
        return 0
    # A period-256 positive needs up to 1024 observed tokens for confirmation.
    return 0 if feature['generation_end'] + 32*horizon + 1024 <= record['tokens'] else None


def all_rows(out):
    rows = read_json(out / 'reused.json')
    rows += [r for p in sorted(out.glob('collected-gpu*.json')) for r in read_json(p)]
    assert len(rows) == 100 and len({(r['setting'], r['question_id']) for r in rows}) == 100
    return rows


def extract(out):
    rows = all_rows(out)
    prompts = {r['question_id']: r['state']['prompt_ids'] for p in sorted(FIFTY.glob('baseline-gpu*.json'))
               for r in read_json(p) if r['setting'] == 'T8'}
    records, feature_count, feature_seconds = [], 0, 0.
    with gzip.open(out / 'features.jsonl.gz', 'wt', compresslevel=1) as writer:
        for row in rows:
            events = events_for(row)
            prompt = prompts[row['question_id']]
            commits = {}
            for e in events:
                for i, token in zip(e['positions'], e['tokens']):
                    absolute = e['block_start'] + i
                    assert absolute not in commits
                    commits[absolute] = e['forward'], token
            assert all(commits[len(prompt)+i][1] == token for i, token in enumerate(row['token_ids']))
            runs = repeat_runs(row['token_ids'])
            event = align_event(runs[0], commits, len(prompt), events) if runs else None
            # Find the first confirmation in commit order, which need not be the first onset in token order.
            confirmations = [align_event(r, commits, len(prompt), events) for r in runs]
            record = {k: row[k] for k in ('question_id', 'setting', 'cohort', 'finish_reason', 'reward', 'forward_calls', 'origin')}
            record.update(tokens=len(row['token_ids']), event=event, trace_audit_passed=True,
                first_confirmation=min((r['confirmation_forward'] for r in confirmations), default=None),
                source_file=row['source_file'], trace_path=row['trace_path'], request_id=row['request_id'],
                sensitivity={f'p{p}_l{n}': bool(repeat_runs(row['token_ids'], p, n)) for p in (64,256) for n in (32,64)},
                post_onset_forwards=row['forward_calls']-event['first_repeat_commit'] if event else 0,
                confirmation_observed_by_512=bool(event and record['forward_calls'] and min(r['confirmation_forward'] for r in confirmations)<=512))
            records.append(record)
            stream = FeatureStream(prompt, 8 if row['setting'] == 'T8' else 32)
            begin = time.perf_counter()
            features = [f for e in events if (f := stream.observe(e)) is not None]
            feature_seconds += time.perf_counter() - begin
            for f in features:
                f.update(question_id=row['question_id'], setting=row['setting'])
                writer.write(json.dumps(f, separators=(',', ':'))+'\n')
            feature_count += len(features)
            print(json.dumps({'extract': row['setting'], 'q': row['question_id'], 'repeat': bool(event), 'features':len(features)}), flush=True)
    write_json(out / 'events.json', records)
    write_json(out / 'extraction.json', {'names': NAMES, 'rows': feature_count,
        'feature_seconds': feature_seconds, 'mean_feature_ms': 1000*feature_seconds/feature_count,
        'scope': 'CPU feature computation from already-materialized arrays; excludes logits statistics, device transfer, scoring and file I/O.'})


def split_questions(folds, test_fold):
    calibration = (test_fold + 1) % 5
    return ({q for q, f in folds.items() if f not in (test_fold, calibration)},
            {q for q, f in folds.items() if f == calibration},
            {q for q, f in folds.items() if f == test_fold})


def alarm_threshold(maxima, rate):
    """Allow at most floor(rate*n) calibration trajectories to trigger."""
    values = sorted(maxima, reverse=True)
    assert values
    return float(np.nextafter(values[int(len(values)*rate)], np.inf))


def assess_alarms(features, scores, thresholds, records, horizon=4):
    grouped = {}
    for f, score in zip(features, scores):
        grouped.setdefault((f['setting'], f['question_id']), []).append((f,float(score)))
    trials = []
    for key, items in grouped.items():
        r = records[key]
        threshold = thresholds[r['question_id']]
        alerts = [(f,s) for f,s in items if s >= threshold]
        f, score = alerts[0] if alerts else (None, None)
        e = r['event']
        gap = e['onset']-f['generation_end'] if f is not None and e else None
        pre = bool(f and e and f['forward'] < e['first_repeat_commit'])
        hit = pre and gap <= 32*horizon
        strict = hit and gap >= 32
        clean = not e and r['finish_reason'] == 'eos'
        flags = [s >= threshold for _, s in items]
        episodes = sum(flag and (i==0 or not flags[i-1]) for i,flag in enumerate(flags))
        trials.append({'setting':r['setting'], 'question_id':r['question_id'], 'cohort':r['cohort'],
            'event':bool(e), 'clean':clean, 'censored_without_event':not e and not clean,
            'threshold':threshold, 'alarm_forward':f['forward'] if f else None, 'alarm_score':score,
            'alarm_generation_end':f['generation_end'] if f else None,
            'predicted_within_horizon':hit, 'strict_hit':strict, 'token_gap':gap,
            'forward_lead':e['first_repeat_commit']-f['forward'] if f and e else None,
            'late_detection':bool(f and e and not pre), 'early_outside_horizon':bool(pre and not hit),
            'false_alarm':bool(clean and f), 'alarm_episodes':episodes})
    summary = {}
    for setting in ('T8','Dynamic'):
        rows = [r for r in trials if r['setting']==setting]
        positives = [r for r in rows if r['event']]
        negatives = [r for r in rows if r['clean']]
        eligible_alarms = [r for r in rows if not r['censored_without_event'] and r['alarm_forward'] is not None]
        hits = [r for r in positives if r['strict_hit']]
        summary[setting] = {'event_n':len(positives), 'clean_n':len(negatives),
            'strict_hits':len(hits), 'strict_new40':sum(r['cohort']=='new40' for r in hits),
            'strict_recall':len(hits)/len(positives) if positives else None,
            'pre_onset_hits':sum(r['predicted_within_horizon'] for r in positives),
            'false_alarm_n':sum(r['false_alarm'] for r in negatives),
            'false_alarm_rate':sum(r['false_alarm'] for r in negatives)/len(negatives) if negatives else None,
            'strict_precision':len(hits)/len(eligible_alarms) if eligible_alarms else None,
            'late_detections':sum(r['late_detection'] for r in positives),
            'mean_clean_alarm_episodes':sum(r['alarm_episodes'] for r in negatives)/len(negatives) if negatives else None,
            'median_token_lead':float(np.median([r['token_gap'] for r in hits])) if hits else None,
            'median_forward_lead':float(np.median([r['forward_lead'] for r in hits])) if hits else None}
    return summary,trials


def forecast(out):
    sys.path.insert(0,str(GA / 'cache/analysis'))
    from sklearn.impute import SimpleImputer
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, brier_score_loss
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)
    import joblib
    records = {(r['setting'],r['question_id']):r for r in read_json(out / 'events.json')}
    features = read_jsonl(out / 'features.jsonl.gz')
    folds = read_json(out / 'folds.json')
    X = np.asarray([f['x'] for f in features])
    methods = {'length':[0,1], 'text':[0,1,6,7,8,9],
               'current':list(range(21)), 'history':list(range(28)), 'full':list(range(len(NAMES))),
               'full_tree':list(range(len(NAMES)))}
    output, predictions = {}, {}
    write_json(out / 'forecast_protocol.json', {'folds_sha256':file_hash(out/'folds.json'),
        'code_sha256':file_hash(__file__), 'methods':methods,
        'tree':{'type':'HistGradientBoostingClassifier','max_iter':100,'max_depth':2,'max_leaf_nodes':4,
                'min_samples_leaf':30,'max_bins':32,'learning_rate':.05,'l2_regularization':1.,
                'selection':'One nonlinear check permitted by the initial plan; fixed before its results.'},
        'normalization':'Training fold only; equal question weights followed by binary class balancing.',
        'calibration':'Whole-trajectory maxima of complete event-free T8 calibration questions; no test threshold tuning.'})
    (out / 'models').mkdir(exist_ok=True)
    for horizon in (1,2,4):
        labels = [target(f,records[f['setting'],f['question_id']],horizon) for f in features]
        y = np.asarray([-1 if value is None else value for value in labels],dtype=int)
        for method in [*methods,'low_confidence','high_entropy','low_entropy','text_repeat']:
            scores = np.zeros(len(features))
            cutoffs = {rate:{} for rate in (.05,.10)}
            fold_checks = []
            for fold in range(5):
                train, calibration, test = split_questions(folds,fold)
                assert not (train&calibration or train&test or calibration&test)
                train_idx = np.asarray([i for i,f in enumerate(features)
                    if f['setting']=='T8' and f['question_id'] in train and y[i]>=0])
                evaluate_idx = np.asarray([i for i,f in enumerate(features) if f['question_id'] in calibration|test])
                if method in methods:
                    cols = methods[method]
                    counts = Counter(features[i]['question_id'] for i in train_idx)
                    weights = np.asarray([1/counts[features[i]['question_id']] for i in train_idx])
                    # Equal question contribution before class balancing; training only.
                    for label in (0,1):
                        hit = y[train_idx]==label
                        assert hit.any(), (fold,horizon,label)
                        weights[hit] /= weights[hit].sum()
                    weights *= len(weights)/weights.sum()
                    learner = (HistGradientBoostingClassifier(max_iter=100,max_depth=2,max_leaf_nodes=4,
                        min_samples_leaf=30,max_bins=32,learning_rate=.05,l2_regularization=1.,
                        early_stopping=False,random_state=SEED) if method=='full_tree' else
                        LogisticRegression(C=1.,max_iter=1000,random_state=SEED))
                    model = make_pipeline(SimpleImputer(strategy='constant',fill_value=0.),StandardScaler(),learner)
                    model.fit(X[train_idx][:,cols],y[train_idx],
                              **{f'{type(learner).__name__.lower()}__sample_weight':weights})
                    values = model.predict_proba(X[evaluate_idx][:,cols])[:,1]
                    if horizon==4:
                        joblib.dump({'model':model,'columns':cols,'names':NAMES,'train':sorted(train),
                                     'calibration':sorted(calibration),'test':sorted(test)},
                                    out / 'models' / f'{method}-fold{fold}.joblib')
                else:
                    col,sign = {'low_confidence':(17,-1),'high_entropy':(13,1),
                                'low_entropy':(13,-1),'text_repeat':(7,1)}[method]
                    values = sign*X[evaluate_idx,col]
                local = dict(zip(evaluate_idx,values))
                maxima = []
                for q in sorted(calibration):
                    r = records['T8',q]
                    if r['event'] is None and r['finish_reason']=='eos':
                        maxima.append(max(local[i] for i in evaluate_idx
                            if features[i]['setting']=='T8' and features[i]['question_id']==q))
                for rate in cutoffs:
                    threshold = alarm_threshold(maxima,rate)
                    cutoffs[rate].update({q:threshold for q in test})
                for i,value in local.items():
                    if features[i]['question_id'] in test:
                        scores[i] = value
                fold_checks.append({'fold':fold,'training_questions':len(train),'calibration_questions':len(calibration),
                    'test_questions':len(test),'calibration_clean_n':len(maxima),
                    'training_states':len(train_idx),'positive_training_states':int((y[train_idx]==1).sum()),
                    'thresholds':{str(rate):cutoffs[rate][next(iter(test))] for rate in cutoffs}})
            entry = {'folds':fold_checks,'rates':{}}
            for rate,thresholds in cutoffs.items():
                summary,trials = assess_alarms(features,scores,thresholds,records,horizon)
                entry['rates'][str(rate)] = summary
                if horizon==4:
                    predictions[f'{method}@{rate}'] = trials
            valid = np.asarray([i for i,f in enumerate(features) if f['setting']=='T8' and y[i]>=0])
            counts = Counter(features[i]['question_id'] for i in valid)
            weights = np.asarray([1/counts[features[i]['question_id']] for i in valid])
            entry['T8_state_AP_question_weighted'] = float(average_precision_score(y[valid],scores[valid],sample_weight=weights))
            if method in methods:
                entry['T8_state_brier_question_weighted'] = float(brier_score_loss(y[valid],scores[valid],sample_weight=weights))
            output[f'h{horizon}/{method}'] = entry
            if horizon==4:
                np.save(out / f'scores-{method}.npy',scores)
            print(json.dumps({'horizon':horizon,'method':method,'metrics':entry['rates']['0.1']['T8']}),flush=True)
    write_json(out / 'forecast.json',output)
    write_json(out / 'alarms.json',predictions)
    controls = ['length','text','current','low_confidence','high_entropy','low_entropy','text_repeat']
    gate = {}
    for method in ('history','full','full_tree'):
        metric = output[f'h4/{method}']['rates']['0.1']['T8']
        best = max(output[f'h4/{control}']['rates']['0.1']['T8']['strict_hits'] for control in controls)
        passed = (metric['false_alarm_rate']<=.1 and metric['strict_hits']>=3 and metric['strict_new40']>=2
                  and metric['strict_hits']>best)
        gate[method] = {'passed':passed,'metric':metric,'best_control_hits':best}
    write_json(out / 'gate.json', {'models':gate,'passed':any(g['passed'] for g in gate.values()),
        'interpretation':'Development screening only. Models with high false alarms cannot pass by raising recall; positive screening still requires cost measurement and paired intervention.'})


def summarize(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = all_rows(out)
    records = read_json(out/'events.json')
    records_by_key = {(r['setting'],r['question_id']):r for r in records}
    features = read_jsonl(out/'features.jsonl.gz')
    alarms = read_json(out/'alarms.json')
    scores = read_json(out/'forecast.json')
    baselines = {}
    for setting in ('T8','Dynamic'):
        a = [r for r in rows if r['setting']==setting]
        e = [r for r in records if r['setting']==setting and r['event']]
        baselines[setting] = {'n':len(a),'correct':sum(r['reward'] for r in a),
            'exact_repetition':len(e),'finishes':dict(Counter(r['finish_reason'] for r in a)),
            'forward_total':sum(r['forward_calls'] for r in a),
            'forward_mean':float(np.mean([r['forward_calls'] for r in a])),
            'forward_median':float(np.median([r['forward_calls'] for r in a])),
            'forward_p90':float(np.quantile([r['forward_calls'] for r in a],.9)),
            'tokens_mean':float(np.mean([len(r['token_ids']) for r in a])),
            'post_onset_fraction':sum(r['post_onset_forwards'] for r in e)/sum(r['forward_calls'] for r in a),
            'cap512_correct':sum(r['reward'] if r['finish_reason']=='eos' and r['forward_calls']<=512 else 0 for r in a),
            'cap512_forwards':sum(min(r['forward_calls'],512) for r in a),
            'onsets':[r['event']['onset'] for r in e],
            'sensitivity':{key:sum(r['sensitivity'][key] for r in records if r['setting']==setting)
                           for key in ('p64_l32','p64_l64','p256_l32','p256_l64')},
            'cohorts':{cohort:{'n':sum(r['cohort']==cohort for r in a),
                'correct':sum(r['reward'] for r in a if r['cohort']==cohort),
                'exact_repetition':sum(r['cohort']==cohort for r in e)} for cohort in ('pilot10','new40')}}
    phase_values = {}
    for f in features:
        r = records_by_key[f['setting'],f['question_id']]
        if r['event'] is None or f['setting']!='T8':
            continue
        gap = r['event']['onset']-f['generation_end']
        phase = ('post' if f['forward']>=r['event']['first_repeat_commit'] else
                 'early' if gap>128 else 'pre32' if gap>=32 else 'near')
        phase_values.setdefault((f['question_id'],phase),[]).append([f['x'][18],f['x'][13]])
    phases = {phase:[np.mean(values,axis=0).tolist() for (q,p),values in phase_values.items() if p==phase]
              for phase in ('early','pre32','near','post')}
    phase_stats = {phase:{'n':len(values),'selected_p_mean':float(np.mean([v[0] for v in values])),
                         'masked_entropy_mean':float(np.mean([v[1] for v in values]))}
                   for phase,values in phases.items()}
    rng = np.random.default_rng(SEED)
    def interval(values):
        values = np.asarray(values,float)
        estimates = values[rng.integers(0,len(values),(2000,len(values)))].mean(axis=1)
        return np.quantile(estimates,[.025,.975]).tolist()
    uncertainty = {}
    for name, trials in alarms.items():
        a = [r for r in trials if r['setting']=='T8']
        uncertainty[name] = {'strict_recall_bootstrap95':interval([r['strict_hit'] for r in a if r['event']]),
                            'clean_false_alarm_bootstrap95':interval([r['false_alarm'] for r in a if r['clean']])}
    checks = [r for p in out.glob('checks-gpu*.json') for r in read_json(p)]
    summary = {'baselines':baselines,'phase_features_question_weighted':phase_stats,
        'paired_accuracy_dynamic_minus_t8':paired_stats(
            {r['question_id']:r['reward'] for r in rows if r['setting']=='T8'},
            {r['question_id']:r['reward'] for r in rows if r['setting']=='Dynamic'}),
        'prediction_uncertainty':uncertainty,
        'uncertainty_scope':'Question bootstrap of fixed held-out predictions; excludes uncertainty from refitting models and development choices.',
        'new_forward_calls':sum(r['new_forward_calls'] for r in rows),
        'new_inference_seconds':sum(r.get('inference_seconds',0) for r in rows if r['origin']=='new'),
        'check_forward_calls':sum(r['forward_calls'] for r in checks),
        'check_inference_seconds':sum(r['seconds'] for r in checks),'check_paths':len(checks),
        'new_prefix_512_checks':sum(r.get('prefix_512_match',False) for r in rows if r['origin']=='new')}
    write_json(out/'summary.json',summary)
    fig,axes = plt.subplots(2,2,figsize=(12,8),layout='constrained')
    by_key = {(r['setting'],r['question_id']):r for r in rows}
    ids = sorted(q for setting,q in by_key if setting=='T8')
    for repeat,color in ((False,'#6a8da8'),(True,'#d75b42')):
        qs = [q for q in ids if bool(records_by_key['T8',q]['event'])==repeat]
        axes[0,0].scatter([by_key['T8',q]['forward_calls'] for q in qs],
                          [by_key['Dynamic',q]['forward_calls'] for q in qs],s=35,alpha=.8,
                          color=color,label='T8 exact loop' if repeat else 'Other T8 paths')
    axes[0,0].plot([20,2600],[20,2600],':',color='gray')
    axes[0,0].set(xscale='log',yscale='log',xlabel='T8 whole-answer forwards',ylabel='Dynamic whole-answer forwards',
                  title='Same 50 development questions')
    axes[0,0].legend(fontsize=8)
    axes[0,1].hist([baselines['T8']['onsets'],baselines['Dynamic']['onsets']],bins=np.arange(0,897,128),
                   label=['T8 (14 events)','Dynamic (2 events)'],color=['#d75b42','#6a8da8'])
    axes[0,1].set(xlabel='Exact repetition onset (generated token)',ylabel='Questions',title='Retrospective exact-loop onset')
    axes[0,1].legend(fontsize=8)
    names = ['text','current','history','full','full_tree','high_entropy']
    metrics = [scores[f'h4/{name}']['rates']['0.1']['T8'] for name in names]
    x = np.arange(len(names))
    axes[1,0].bar(x-.18,[m['strict_recall']*100 for m in metrics],.36,label='Strict early recall (n=14)',color='#398477')
    axes[1,0].bar(x+.18,[m['false_alarm_rate']*100 for m in metrics],.36,label='False alarm rate (n=35)',color='#e1a64a')
    axes[1,0].set(xticks=x,xticklabels=['Text','Current','+History','Full','Tree','Entropy'],ylabel='Percent',
                  title='T8: 4-block forecast, held-out questions')
    axes[1,0].legend(fontsize=8)
    axes[1,1].boxplot([[v[0] for v in phases[p]] for p in ('early','pre32','near','post')],
                      tick_labels=['>128 before','32-128 before','Near onset','After onset'])
    axes[1,1].set(ylabel='Mean confidence of selected tokens',title='T8 loop cases: one mean per question/phase')
    axes[1,1].tick_params(axis='x',labelrotation=12)
    for ax in axes.flat:
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y',alpha=.15)
    fig.savefig(out/'overview.png',dpi=180)
    fig.savefig(out/'overview.pdf')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'collect', 'extract', 'forecast', 'summarize'])
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if args.stage == 'prepare':
        prepare(args.out)
    elif args.stage == 'collect':
        with ProcessPoolExecutor(5, mp_context=mp.get_context('spawn')) as pool:
            futures = [pool.submit(collect_worker, gpu, str(args.out)) for gpu in range(5)]
            print([f.result() for f in futures])
    elif args.stage == 'extract':
        extract(args.out)
    elif args.stage=='forecast':
        forecast(args.out)
    else:
        summarize(args.out)


if __name__ == '__main__':
    main()
