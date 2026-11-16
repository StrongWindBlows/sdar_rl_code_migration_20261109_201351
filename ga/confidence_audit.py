"""CPU-only test of confidence sufficiency using the completed T3 interventions."""
import argparse
from collections import Counter, defaultdict, deque
import gzip
import json
from pathlib import Path
import shutil
import sys

import numpy as np

from marginal_commit import (compare_policy, features as t3_features, make_actions,
                            outcome, records, sources)
from protocol import file_hash, read_jsonl, write_json

GA = Path(__file__).resolve().parent
PARENT = GA / 'runs/marginal-commit-20261112'
OUT = GA / 'runs/confidence-sufficiency-20261112'
SETTINGS = dict(seed=20261112, tree_depth=2, tree_min_leaf=12,
                logistic_C=1., ridge_alpha=10., bootstrap_samples=2000)


def read(path):
    return json.loads(Path(path).read_text())


def source_history(cases, manifest):
    """Stream each original trace once; capture at most three strictly earlier rounds."""
    natural = sources()
    wanted, histories = defaultdict(dict), {}
    for case in cases:
        row, state = natural[case['question_id']], case['snapshot']
        wanted[row['trace_path']][row['request_id'], state['source_forward']] = case
    for filename, targets in wanted.items():
        assert file_hash(filename) == manifest['source_hashes'][filename]
        previous = defaultdict(lambda: deque(maxlen=3))
        with gzip.open(filename, 'rt') as handle:
            for line in handle:
                event = json.loads(line)
                key = event['request_id'], event['block_start']
                case = targets.get((event['request_id'], event['forward']))
                if case is not None:
                    state = case['snapshot']
                    assert event['block_start'] == state['block_start']
                    for source, saved in [('input_ids', 'input_ids'), ('mask_before', 'mask'),
                                          ('current_ids', 'current_ids'), ('topk_probs', 'topk_probs')]:
                        assert event[source] == state[saved], (case['id'], source)
                    history = list(previous[key])
                    assert state['historical_ids'] == (history[-1]['ids'] if history else None)
                    assert all(h['forward'] < state['source_forward'] for h in history)
                    histories[case['id']] = history
                if event['phase'] == 'denoise':
                    previous[key].append(dict(forward=event['forward'],
                        ids=event['current_ids'], probability=[p[0] for p in event['topk_probs']]))
    assert len(histories) == len(cases)
    return histories


def feature_layers(state, action, history):
    """C1 sees the full masked confidence multiset; C2 adds no future observations."""
    original = set(state['positions'])
    selected = set(action['positions'])
    groups = dict(masked=state['masked_positions'], selected=sorted(selected),
                  added=sorted(selected-original), removed=sorted(original-selected))
    p = np.asarray(state['confidence'])
    c0 = dict(k=action['k'], k0=len(original), delta_k=action['k']-len(original),
              masked=len(groups['masked']), round=state['round'],
              progress=(state['block_start']-state['prompt_length'])/8192,
              forward_progress=state['source_forward']/512)
    for group, positions in groups.items():
        values = p[positions] if positions else np.array([0.])
        c0.update({f'{group}_p_min': float(values.min()), f'{group}_p_mean': float(values.mean())})
    c1 = dict(c0)
    ranked = np.sort(p[groups['masked']])[::-1]
    c1.update({f'rank_p_{i:02d}': float(v) for i, v in enumerate(np.pad(ranked, (0, 32-len(ranked))))})
    for group, positions in groups.items():
        values = p[positions] if positions else np.array([0.])
        c1.update({f'{group}_p_q{q}': float(np.percentile(values, q)) for q in (25, 50, 75)})
        c1[f'{group}_uncertainty_mass'] = float(sum(1-p[i] for i in positions))
    k = action['k']
    c1['boundary_gap'] = float(ranked[k-1]-ranked[k]) if k < len(ranked) else 0.
    margin = np.asarray([v[0]-v[1] for v in state['topk_probs']])
    entropy = np.asarray(state['entropy'])
    shape, temporal = {}, dict(history_available=int(bool(history)), history_depth=len(history))
    for group, positions in groups.items():
        for name, values in [('margin', margin), ('entropy', entropy)]:
            part = values[positions] if positions else np.array([0.])
            shape.update({f'{group}_{name}_mean': float(part.mean()),
                          f'{group}_{name}_min': float(part.min())})
        if history and positions:
            now = np.asarray(state['current_ids'])[positions]
            prior = np.asarray([h['ids'] for h in history])[:, positions]
            temporal[f'{group}_last_flip'] = float(np.mean(now != prior[-1]))
            temporal[f'{group}_history_agreement'] = float(np.mean(prior == now))
            temporal[f'{group}_p_change'] = float(np.mean(
                np.asarray(state['topk_probs'])[positions, 0]-np.asarray(history[-1]['probability'])[positions]))
        else:
            temporal.update({f'{group}_{name}': 0. for name in ('last_flip', 'history_agreement', 'p_change')})
    shape['masked_entropy_std'] = float(entropy[groups['masked']].std())
    visible = {i for i, flag in enumerate(state['mask']) if flag != 0}
    complete = visible | selected
    longest = run = 0
    for i in range(32):
        run = run+1 if i in complete else 0
        longest = max(run, longest)
    spatial = dict(completed_run=longest/32,
        selected_span=(max(selected)-min(selected))/31,
        selected_adjacent_pairs=sum(i+1 in selected for i in selected)/31,
        added_anchor_distance=float(np.mean([min((abs(i-j) for j in visible), default=32)
            for i in groups['added']]))/32 if groups['added'] else 0.,
        added_adjacent_to_visible=float(np.mean([i-1 in visible or i+1 in visible
            for i in groups['added']])) if groups['added'] else 0.)
    return {'C0': c0, 'C1': c1, 'C1_margin_entropy': c1 | shape,
            'C1_history': c1 | temporal, 'C1_position': c1 | spatial,
            'C2': c1 | shape | temporal | spatial}


def prepare(out):
    manifest, cases = read(PARENT/'manifest.json'), read(PARENT/'cases.json')
    completion = read(PARENT/'completion.json')
    for name, digest in completion['result_sha256'].items():
        assert file_hash(PARENT/name) == digest, name
    rows = records(PARENT)
    lookup = {(r['case_id'], r['action']['id']): r for r in rows}
    assert len(lookup) == len(rows) == 499
    assert set(lookup) == {(c['id'], a['id']) for c in cases for a in c['actions']}
    histories, observations = source_history(cases, manifest), []
    old_labels = {(r['case_id'], r['action_id']): r for r in read(PARENT/'interventions.json')}
    for case in cases:
        state = case['snapshot']
        actions, policies = make_actions(state, manifest['study'])
        assert actions == case['actions'] and policies == case['policies']
        current = lookup[case['id'], f'k{len(state["positions"]):02d}']
        for action in case['actions']:
            result = lookup[case['id'], action['id']]
            label = old_labels[case['id'], action['id']]
            assert t3_features(state, action) == case['features'][action['id']]
            assert outcome(current, result) == label['outcome']
            assert result['reward']-current['reward'] == label['delta_reward']
            assert result['forward_calls']-current['forward_calls'] == label['delta_forward']
            observations.append(dict(case_id=case['id'], question_id=case['question_id'],
                window=case['window'], action_id=action['id'], delta_k=action['k']-len(state['positions']),
                reward=result['reward'], current_reward=current['reward'],
                delta_reward=label['delta_reward'], delta_forward=label['delta_forward'], outcome=label['outcome'],
                features=feature_layers(state, action, histories[case['id']])))
    out.mkdir(parents=True, exist_ok=True)
    assert not (out/'protocol.json').exists()
    protocol = dict(date='2026-11-12', settings=SETTINGS, folds=manifest['folds'],
        data_scope='Existing development50 only; no new GPU forward and no independent confirmation.',
        timing='Fixed after inspecting T3 and before fitting this T4 comparison; exploratory, not fresh preregistration.',
        primary='C2 versus C1. Window0 for one-action-per-question outcomes; all windows for secondary diagnostics.',
        inputs={str(path): file_hash(path) for path in [PARENT/'cases.json', PARENT/'interventions.json',
            PARENT/'manifest.json', PARENT/'completion.json', *sorted((PARENT/'branches').glob('gpu*.jsonl'))]},
        source_trace_hashes=manifest['source_hashes'],
        code={name: file_hash(GA/name) for name in ['confidence_audit.py', 'test_confidence_audit.py']},
        feature_groups={name: list(values) for name, values in observations[0]['features'].items()},
        history='Last up to three strictly earlier denoise forwards in the same block, queried only at currently masked positions. Missing history encoded explicitly.',
        models='Fixed depth2/minleaf12 regression trees for harm, repair, cost; standardized L2 logistic harm/repair heads (C=1) and ridge cost (alpha=10). Equal capacity across feature groups; no tuning.',
        targets='Harm indicator, repair indicator, delta total forward /512. Harm probability is unconditional action harm, not conditioned on future baseline correctness.',
        training='Only non-Current unique actions. Every question has weight1, divided equally over its states and actions. All its rows stay in the same inherited fold.',
        decision='Always include Current=(0 reward change,0 forward change). Maximize predicted repair probability minus harm probability, then minimize predicted cost; ties prefer smallest absolute delta_k. No test-dependent safety threshold.',
        matching=dict(confidence_caliper=.05, exact=['window','k0','actual_delta_k'],
            strict_masked_caliper=4, strict_round_caliper=2,
            rule='Across questions, all candidate pairs from visible features only; greedy nearest disjoint states, no outcome-based selection. Loose version drops masked/round calipers and is descriptive only.'),
        uncertainty='Question-cluster bootstrap, 2000 draws; predictive proper scores and paired policy outcomes. No significance claims from individual matched examples or secondary feature ablations.')
    write_json(out/'protocol.json', protocol)
    write_json(out/'observations.json', observations)
    write_json(out/'history.json', histories)
    write_json(out/'input_audit.json', dict(passed=True, questions=50, states=len(cases), actions=len(rows),
        parent_version_verified=True, labels_recomputed=True, native_actions_recomputed=True,
        strictly_prior_history=True, history_missing_states=sum(not h for h in histories.values()),
        source_traces_verified=len(manifest['source_hashes']), new_gpu_forwards=0))
    (out/'source').mkdir()
    for name in protocol['code']:
        shutil.copy2(GA/name, out/'source'/name)
    (out/'PLAN.md').write_text('# Experiment T4：Confidence Sufficiency\n\n'
        '仅复用T3最终499条反事实路径、137个状态、50道开发题。先核对版本、标签、原生BF16动作及历史来源，再比较C0置信度摘要、C1完整置信度排序和C2额外信息。\n\n'
        '主比较C2对C1；同题五折隔离，固定浅树与正则化线性模型，不调参。匹配规则仅使用源状态特征。模型输出检验与单步动作选择分开报告。最早区间为主结果，其余区间和单类特征消融仅诊断。\n\n'
        '本次规则在已有T3结果之后、T4拟合之前冻结，仍属于反复使用开发题的探索；不视为独立验证。详细规则、数据与代码指纹见protocol.json。\n')
    print(json.dumps(read(out/'input_audit.json')))


def observation_weights(rows):
    """Equal mass per question, then per state, then per unique alternative."""
    state_counts = Counter(r['case_id'] for r in rows)
    states = defaultdict(set)
    for row in rows:
        states[row['question_id']].add(row['case_id'])
    return np.asarray([1/len(states[r['question_id']])/state_counts[r['case_id']] for r in rows])


def fit_predict(train, test, columns, group, family, settings):
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeRegressor, export_text
    x = np.array([[r['features'][group][f] for f in columns] for r in train])
    z = np.array([[r['features'][group][f] for f in columns] for r in test])
    weights = observation_weights(train)
    weights *= len(train)/weights.sum()
    y = np.array([[r['outcome']=='harm', r['outcome']=='repair', r['delta_forward']/512] for r in train], dtype=float)
    scaling = {}
    if family == 'Linear':
        scaler = StandardScaler().fit(x, sample_weight=weights)
        x, z = scaler.transform(x), scaler.transform(z)
        scaling = dict(mean=scaler.mean_.tolist(), scale=scaler.scale_.tolist())
    predictions, model = [], dict(columns=columns, scaling=scaling, heads=[])
    for head in range(3):
        if family == 'Tree':
            learner = DecisionTreeRegressor(max_depth=settings['tree_depth'],
                min_samples_leaf=settings['tree_min_leaf'], random_state=settings['seed'])
        elif head == 2:
            learner = Ridge(alpha=settings['ridge_alpha'])
        elif len(np.unique(y[:,head])) == 1:
            predictions.append(np.full(len(test), y[0,head]))
            model['heads'].append(dict(constant=float(y[0,head])))
            continue
        else:
            learner = LogisticRegression(C=settings['logistic_C'], max_iter=2000, random_state=settings['seed'])
        learner.fit(x, y[:,head], sample_weight=weights)
        predictions.append(learner.predict_proba(z)[:,1] if family=='Linear' and head < 2 else learner.predict(z))
        if family == 'Tree':
            model['heads'].append(dict(tree=export_text(learner, feature_names=columns)))
        else:
            model['heads'].append(dict(coefficients=learner.coef_.tolist(), intercept=np.asarray(learner.intercept_).tolist(),
                                       iterations=np.asarray(getattr(learner, 'n_iter_', [])).tolist()))
    return np.array(predictions).T, model


def select_actions(cases, predictions):
    selected = {}
    for case in cases:
        k0 = len(case['snapshot']['positions'])
        def value(action):
            pred = predictions[case['id'], action['id']]
            return pred[1]-pred[0], -pred[2], -abs(action['k']-k0), -action['k']
        selected[case['id']] = max(case['actions'], key=value)['id']
    return selected


def proper_scores(rows, predictions):
    from sklearn.metrics import roc_auc_score
    weight = observation_weights(rows)
    weight /= weight.sum()
    target = np.array([[r['outcome']=='harm', r['outcome']=='repair', r['delta_forward']/512] for r in rows])
    pred = np.array([predictions[r['case_id'],r['action_id']] for r in rows])
    result = dict(questions=len({r['question_id'] for r in rows}), actions=len(rows),
        harm_brier=float(weight @ (pred[:,0]-target[:,0])**2),
        repair_brier=float(weight @ (pred[:,1]-target[:,1])**2),
        delta_reward_mse=float(weight @ ((pred[:,1]-pred[:,0])-(target[:,1]-target[:,0]))**2),
        forward_mae=float(weight @ np.abs(pred[:,2]-target[:,2])*512),
        harm_actions=int(target[:,0].sum()), repair_actions=int(target[:,1].sum()))
    result['harm_auc'] = float(roc_auc_score(target[:,0],pred[:,0],sample_weight=weight)) if len(set(target[:,0]))==2 else None
    result['per_question'] = {}
    for qid in sorted({r['question_id'] for r in rows}):
        index = [i for i,r in enumerate(rows) if r['question_id']==qid]
        w = weight[index]/weight[index].sum()
        result['per_question'][qid] = dict(harm_brier=float(w@(pred[index,0]-target[index,0])**2),
            repair_brier=float(w@(pred[index,1]-target[index,1])**2),
            forward_mae=float(w@np.abs(pred[index,2]-target[index,2])*512))
    return result


def bootstrap_pair(values, seed=20261112):
    values = np.asarray(values)
    rng = np.random.default_rng(seed)
    boot = values[rng.integers(len(values), size=(SETTINGS['bootstrap_samples'], len(values)))].mean(axis=1)
    return dict(mean=float(values.mean()), ci95=np.quantile(boot,[.025,.975]).tolist())


def confidence_matches(observations, strict):
    rows = [r for r in observations if r['delta_k']>0]
    edges = []
    for i,left in enumerate(rows):
        a=left['features']['C0']
        for right in rows[i+1:]:
            b=right['features']['C0']
            if left['question_id']==right['question_id'] or left['window']!=right['window']:
                continue
            if a['k0']!=b['k0'] or a['delta_k']!=b['delta_k']:
                continue
            gaps=[abs(a[f]-b[f]) for f in ('added_p_min','added_p_mean')]
            if max(gaps)>.05 or (strict and (abs(a['masked']-b['masked'])>4 or abs(a['round']-b['round'])>2)):
                continue
            edges.append((sum(gaps),left['case_id'],right['case_id'],left['action_id'],right['action_id'],left,right))
    used, selected = set(), []
    for *_,left,right in sorted(edges, key=lambda e:e[:5]):
        if left['case_id'] in used or right['case_id'] in used:
            continue
        used.update((left['case_id'],right['case_id']))
        selected.append(dict(left={k:v for k,v in left.items() if k!='features'} | dict(confidence=left['features']['C0']),
                             right={k:v for k,v in right.items() if k!='features'} | dict(confidence=right['features']['C0'])))
    return dict(candidate_edges=len(edges), pairs=len(selected), states=len(used),
        questions=len({r[side]['question_id'] for r in selected for side in ('left','right')}),
        unequal_delta_reward=sum(p['left']['delta_reward']!=p['right']['delta_reward'] for p in selected),
        unequal_delta_forward=sum(p['left']['delta_forward']!=p['right']['delta_forward'] for p in selected),
        pair_details=selected)


def analyze(out):
    sys.path.insert(0,str(GA/'cache/analysis'))
    protocol, cases = read(out/'protocol.json'), read(PARENT/'cases.json')
    for name,digest in protocol['code'].items():
        assert file_hash(GA/name)==digest, name
    for name,digest in protocol['inputs'].items():
        assert file_hash(name)==digest, name
    obs = read(out/'observations.json')
    alternatives = [r for r in obs if r['delta_k']!=0]
    lookup = {(r['case_id'],r['action']['id']):r for r in records(PARENT)}
    selected = {name:{c['id']:f'k{c["policies"][name]:02d}' for c in cases} for name in ('Current','F085','F095','Factor1')}
    predicted, models, splits = {}, {}, []
    for fold in range(5):
        train = [r for r in alternatives if protocol['folds'][r['question_id']]!=fold]
        test = [r for r in alternatives if protocol['folds'][r['question_id']]==fold]
        ids = {label:sorted({r['question_id'] for r in rows}) for label,rows in [('train',train),('test',test)]}
        assert set(ids['train']).isdisjoint(ids['test'])
        splits.append(dict(fold=fold,**ids))
        weight=observation_weights(train)
        targets=np.array([[r['outcome']=='harm',r['outcome']=='repair',r['delta_forward']/512] for r in train])
        constant=np.average(targets,axis=0,weights=weight)
        predicted.setdefault('TrainConstant',{}).update({(r['case_id'],r['action_id']):constant.tolist() for r in test})
        for family in ('Tree','Linear'):
            for group,columns in protocol['feature_groups'].items():
                name=f'{family}_{group}'
                values, model=fit_predict(train,test,columns,group,family,protocol['settings'])
                predicted.setdefault(name,{}).update({(r['case_id'],r['action_id']):v.tolist() for r,v in zip(test,values)})
                models.setdefault(name,[]).append(dict(fold=fold,**model))
    for name,values in predicted.items():
        for case in cases:
            values[case['id'],f'k{len(case["snapshot"]["positions"]):02d}']=[0.,0.,0.]
        selected[name]=select_actions(cases,values)
    metrics, scores, comparisons = {}, {}, {}
    study=dict(SETTINGS,forward_cap=512)
    for scope,subset in [('primary',[c for c in cases if c['window']==0]),('all_states',cases)]:
        metrics[scope]={name:compare_policy(subset,choice,lookup,study) for name,choice in selected.items()}
        valid={c['id'] for c in subset}
        score_rows=[r for r in alternatives if r['case_id'] in valid and r['delta_k']>0]
        scores[scope]={name:proper_scores(score_rows,values) for name,values in predicted.items()}
        comparisons[scope]={}
        for family in ('Tree','Linear'):
            a,b=f'{family}_C2',f'{family}_C1'
            comparisons[scope][family]={}
            for metric in ('harm_brier','repair_brier','forward_mae'):
                av,bv=scores[scope][a]['per_question'],scores[scope][b]['per_question']
                comparisons[scope][family][metric]=bootstrap_pair([av[q][metric]-bv[q][metric] for q in sorted(av)])
            by_q=defaultdict(list)
            for case in subset:
                left,right=lookup[case['id'],selected[a][case['id']]],lookup[case['id'],selected[b][case['id']]]
                by_q[case['question_id']].append([left['reward']-right['reward'],left['forward_calls']-right['forward_calls']])
            values=np.array([np.mean(v,axis=0) for q,v in sorted(by_q.items())])
            comparisons[scope][family]['policy_delta_reward']=bootstrap_pair(values[:,0])
            comparisons[scope][family]['policy_delta_forward']=bootstrap_pair(values[:,1])
    matches={name:confidence_matches(obs,strict) for name,strict in [('strict',True),('loose',False)]}
    write_json(out/'matches.json',matches)
    write_json(out/'models.json',dict(splits=splits,models=models))
    write_json(out/'predictions.json',{name:[dict(case_id=k[0],action_id=k[1],harm=v[0],repair=v[1],delta_forward=v[2]*512)
        for k,v in sorted(values.items())] for name,values in predicted.items()})
    write_json(out/'selected_actions.json',selected)
    write_json(out/'scores.json',scores)
    write_json(out/'comparisons.json',comparisons)
    write_json(out/'summary.json',dict(metrics=metrics,history=read(out/'input_audit.json'),
        coverage=dict(questions=50,states=len(cases),actions=len(obs),alternatives=len(alternatives),
            increased_actions=sum(r['delta_k']>0 for r in obs),increased_harm=sum(r['delta_k']>0 and r['outcome']=='harm' for r in obs),
            increased_repair=sum(r['delta_k']>0 and r['outcome']=='repair' for r in obs))))
    report(out,metrics,scores,matches,comparisons)
    print(json.dumps({'primary':{n:[m['reward']*50,m['forward_mean']] for n,m in metrics['primary'].items()},
                      'matches':{n:{k:v for k,v in m.items() if k!='pair_details'} for n,m in matches.items()}}))


def report(out,metrics,scores,matches,comparisons):
    lines=['# Experiment T4：置信度是否足以决定提交', '',
        '日期：2026-11-12。仅复用T3原50道开发题、137个状态、499条最终路径；新增GPU forward为0。', '',
        '## 方法', '',
        '先从原始逐动作结果重算奖励差、整题forward差和动作数量，核对T3最终指纹。历史只取同块之前最多三轮，且仅在当前仍masked的位置上计算。', '',
        'C0为新增/提交/剩余位置置信度最低值与均值及数量、阶段、预算；C1加入完整32槽置信度排序（尾部补零）、分位数、累计不确定性及边界间隔；C2再加入margin/熵、历史变化、位置结构。历史缺失显式记录。', '',
        '固定浅树与正则化线性模型，无参数搜索。继承按题五折，所有状态和动作按题隔离；每题训练权重相同。预测损害概率、修复概率、forward变化。以预测奖励差优先、成本其次选择动作，始终保留Current。', '',
        '所有模型规则在本次拟合前固定，但已经看过T3结果，50题也被反复用于开发；这仍是探索性折外检验，不能替代独立验证。', '',
        '## 主结果：最早区间，每题一个状态', '',
        '| 选择器 | 正确 / 50 | 平均整题forward | 新增 / 丢失正确 |',
        '| --- | ---: | ---: | ---: |']
    for name in ['Current','F085','F095','Factor1','TrainConstant']+[f'{f}_{g}' for f in ('Tree','Linear') for g in ('C0','C1','C2')]:
        m=metrics['primary'][name]
        lines.append(f"| {name} | {m['reward']*50:.0f} | {m['forward_mean']:.2f} | {m['repair']*50:.0f} / {m['harm']*50:.0f} |")
    lines+=['','完整消融和题内平均的全部状态结果见summary.json。表中是从已知反事实路径读取折外选择结果，未执行在线选择器，也未测量线上特征开销。','','## 同置信度配对','','配对先按可见信息选定，不按奖励寻找反例。两组均跨题、同生成区间、同k0及实际Δk，新增置信度最低值/均值相差各不超过0.05；严格组另要求剩余masked数相差≤4、轮次相差≤2。按距离贪心匹配，每个状态最多出现一次。','','| 配对 | 对数 | 覆盖题数 | 奖励差不同的对数 | forward差不同的对数 |','| --- | ---: | ---: | ---: | ---: |']
    for name,m in matches.items():
        lines.append(f"| {name} | {m['pairs']} | {m['questions']} | {m['unequal_delta_reward']} | {m['unequal_delta_forward']} |")
    lines+=['','配对只是相近摘要下的结果描述。剩余状态、题目难度及完整置信度向量仍可能不同；结果不同不等于条件风险可预测，零差异也不证明等效。完整配对见matches.json。','','## 折外预测：实际增加提交的动作','','每题权重相同。Brier分数和forward MAE越低越好；预测标签稀少，必须对照训练折常数预测，不能用大量零标签获得的小误差宣称有效。','','| 模型 | 损害 Brier | 修复 Brier | forward MAE |','| --- | ---: | ---: | ---: |']
    for name in ['TrainConstant']+[f'{f}_{g}' for f in ('Tree','Linear') for g in ('C0','C1','C2')]:
        s=scores['all_states'][name]
        lines.append(f"| {name} | {s['harm_brier']:.5f} | {s['repair_brier']:.5f} | {s['forward_mae']:.2f} |")
    lines+=['','## C2 相对 C1 的主比较','','下表为全状态、按题bootstrap；负的预测误差差值表示C2更好。区间不能覆盖开发过程中的多次尝试偏差。','','| 模型 | 损害 Brier 差 [95%区间] | 修复 Brier 差 [95%区间] | forward MAE 差 [95%区间] |','| --- | --- | --- | --- |']
    def interval(value):
        return f"{value['mean']:.5f} [{value['ci95'][0]:.5f}, {value['ci95'][1]:.5f}]"
    for family in ('Tree','Linear'):
        c=comparisons['all_states'][family]
        lines.append(f"| {family} | {interval(c['harm_brier'])} | {interval(c['repair_brier'])} | {interval(c['forward_mae'])} |")
    lines+=['','C1只改进C0时，只能说明摘要丢失信息；C2改进C1才涉及置信度之外的信息。判断算法价值还需动作选择改善正确率—总forward关系，不能只靠单一误差或AUC。','','![置信度信息层次的折外核验](overview.png)','','输入核验见input_audit.json；完整协议与代码指纹见protocol.json；预测、模型与折划分分别见predictions.json、models.json。解释与最终判断见ANALYSIS.md。','']
    (out/'REPORT.md').write_text('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    for family,color in [('Tree','tab:blue'),('Linear','tab:orange')]:
        names=[f'{family}_{g}' for g in ('C0','C1','C2')]
        axes[0].plot(range(3),[scores['all_states'][n]['harm_brier'] for n in names],marker='o',label=family,color=color)
        for name in names:
            m=metrics['primary'][name]
            axes[1].scatter(m['forward_mean'],m['reward']*100,color=color)
            axes[1].annotate(name,(m['forward_mean'],m['reward']*100),xytext=(3,4),textcoords='offset points',fontsize=7)
    axes[0].axhline(scores['all_states']['TrainConstant']['harm_brier'],color='gray',ls='--',label='Train constant')
    axes[0].set(xticks=range(3),xticklabels=['C0 summary','C1 confidence','C2 extra'],ylabel='Question-weighted harm Brier',title='Held-out predictions: increased commits')
    axes[0].legend(fontsize=8)
    m=metrics['primary']['Current']
    axes[1].scatter(m['forward_mean'],m['reward']*100,color='black',marker='*',s=90)
    axes[1].annotate('Current',(m['forward_mean'],m['reward']*100),xytext=(4,4),textcoords='offset points',fontsize=8)
    axes[1].set(xlabel='Mean total forwards',ylabel='Accuracy (%)',title='One intervention per question: first window')
    fig.savefig(out/'overview.png',dpi=180)
    fig.savefig(out/'overview.pdf')
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','analyze'])
    parser.add_argument('--out',type=Path,default=OUT)
    args=parser.parse_args()
    (prepare if args.stage=='prepare' else analyze)(args.out)


if __name__=='__main__':
    main()
