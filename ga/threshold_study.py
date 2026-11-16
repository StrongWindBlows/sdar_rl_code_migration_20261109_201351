"""Run the fixed-B32 threshold landscape, one complete trajectory per policy/question."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import gzip
import json
import multiprocessing as mp
from pathlib import Path
import random
import shutil
import subprocess
import urllib.request

import numpy as np

from degeneration import all_rows as old_rows, events_for, repeat_runs
from protocol import Grader, file_hash, paired_stats, read_jsonl, trace_difference, write_json
from threshold_rules import calibrate, factor_count, threshold

GA = Path(__file__).resolve().parent
OUT = GA / 'runs/threshold-landscape-20261112'
OLD = GA / 'runs/degeneration-20261112'
SETTING = dict(name='Dynamic',strategy='low_confidence_dynamic',steps=32,threshold=.9)
CODE = ('threshold_study.py','threshold_rules.py','test_thresholds.py','threshold_study.json')


def read(path):
    return json.loads(Path(path).read_text())


def prepare(out):
    from run import fingerprint
    config = read(GA/'threshold_study.json')
    parent = read(GA/'runs/pilot8192-20261110/manifest.json')
    original = fingerprint(parent['config'])
    assert original == parent['fingerprint']
    data = read(GA/'runs/joint50-20261111/data.json')
    ids = sorted(r['unique_id'] for r in data)
    rng = random.Random(config['seed'])
    calibration_id = rng.choice(ids)
    order = ids.copy()
    rng.shuffle(order)
    gpu_of = {q:config['gpus'][i%len(config['gpus'])] for i,q in enumerate(order)}
    engine = dict(parent['config'],gpus=config['gpus'],max_new_tokens=config['max_new_tokens'],
                  batch_size=1,trace=False,cache_fraction=config['cache_fraction'])
    assert not (out/'manifest.json').exists()
    out.mkdir(parents=True,exist_ok=True)
    write_json(out/'original_fingerprint.json',original)
    write_json(out/'data.json',data)
    write_json(out/'manifest.json',{'date':'2026-11-12','engine':engine,'study':config,
        'calibration_id':calibration_id,'gpu_of':gpu_of,
        'timing_ids':rng.sample([q for q in ids if q!=calibration_id],config['timing_questions']),
        'code':{name:file_hash(GA/name) for name in CODE},
        'data_sha256':file_hash(out/'data.json'),
        'protocol':{
            'sampling':'Greedy; native full-vocabulary softmax; same BF16 engine and stopping rules.',
            'denoise':'Any forward with generated MASK positions before the decision, including the mixed prompt block. Pure prefill and cache writes excluded.',
            'tokens':'Returned generated token IDs including the first EOS; prompt, MASK and discarded post-EOS positions excluded.',
            'calibration':'Seed-selected development question, native static T32. Q1 over committed confidences, only returned generation positions.',
            'osdt':'Threshold-only B32 adaptation. Q1, cap .98, slack .05, clip [.75,.98]; missing block/step -> .90. Engine >= and min-one shared by all rules.',
            'cad':'Threshold-only B32 adaptation: .95 -> .80 using generated block-start / 8192, as in sequence-progress pseudocode; no future EOS length.',
            'factor':'Published largest-k rule (k+1)*(1-p[k]) < phi, min-one. Fixed .33/.50/.67 support controls precede E2.',
            'comparison':'E0 reports all 50; E1/E2 main comparisons use the same 49 non-calibration questions for every rule.',
            'timing':'Full generation with compact in-memory events; disk I/O, grading and loading outside request timing. Same-question same-GPU randomized policy order.',
            'validation':'All 50 are development. Remaining evaluation50 and final400 untouched. E3 is outside this run.'}})
    (out/'source').mkdir()
    for name in CODE:
        shutil.copy2(GA/name,out/'source'/name)
    sources = {'osdt.py':'https://raw.githubusercontent.com/juchengshen/osdt/main/osdt/llada_generate_osdt.py',
               'cad-paper.html':'https://arxiv.org/html/2512.07173v4'}
    upstream = {}
    for name,url in sources.items():
        path = out/'source'/name
        path.write_bytes(urllib.request.urlopen(url,timeout=30).read())
        upstream[name] = {'url':url,'sha256':file_hash(path)}
    write_json(out/'upstream.json',upstream)
    print(json.dumps({'questions':len(data),'calibration':calibration_id,'gpus':config['gpus']}))


def backend_class():
    import torch
    from engine import Backend, MASKED, UNMASKED

    class StudyBackend(Backend):
        def __init__(self,config,table=None):
            self.rule = dict(name='F090',kind='fixed',threshold=.9)
            self.table = table or {}
            super().__init__(config,SETTING)

        def set_rule(self,rule):
            self.rule = rule
            setting = dict(SETTING)
            if rule['kind']=='calibration':
                setting.update(strategy='low_confidence_static',steps=32)
            self.set_setting(setting)

        def _install(self):
            super()._install()
            processors = {id(s.unmasking_processor):s.unmasking_processor
                          for s in (self.core.model_agent_strategy,self.agent.agent_strategy)}
            for processor in processors.values():
                native_scores = processor._get_scores
                def get_scores(logits,tokens,native=native_scores):
                    scores = native(logits,tokens)
                    self.confidence = scores.tolist()
                    if self.decision.get('threshold') is not None:
                        self.decision['effective_threshold'] = torch.tensor(
                            self.decision['threshold'],dtype=scores.dtype).item()
                    return scores
                self._replace(processor,'_get_scores',get_scores)
                native_dynamic = processor.low_confidence_dynamic
                def dynamic(logits,tokens,mask,proc=processor,native=native_dynamic):
                    flags = (mask==MASKED).tolist()
                    if not any(flags):
                        return native(logits,tokens,mask)
                    track,base = self.context
                    step = track.rounds.get(base,0)+1
                    block = (base-len(track.prompt_ids)//32*32)//32
                    generated = max(0,base-len(track.prompt_ids))
                    if self.rule['kind']=='factor':
                        scores = proc._get_scores(logits,tokens)
                        k,forced = factor_count([p for p,m in zip(self.confidence,flags) if m],self.rule['factor'])
                        filtered = scores.masked_fill(mask!=MASKED,float('-inf'))
                        indices = filtered.topk(k).indices
                        result = mask.clone()
                        result[indices] = UNMASKED
                        self.decision.update(forced=bool(forced),factor_k=k)
                        return result
                    value,fallback,clipped = threshold(self.rule,block,step,generated,
                        self.config['max_new_tokens'],self.table)
                    self.decision.update(threshold=value,table_fallback=fallback,threshold_clipped=clipped)
                    proc.dllm_config.confidence_threshold = value
                    return native(logits,tokens,mask)
                self._replace(processor,'low_confidence_dynamic',dynamic)
            original = self.agent.agent_strategy.post_sampling
            def post(inputs,logits,ids,extra):
                assert len(inputs.model_metas)==1
                track = self.tracks[inputs.model_metas[0]['ga_session']]
                base = int((inputs.history_lengths+inputs.seq_length-self.block).item())
                self.context = track,base
                self.confidence,self.decision = None,{}
                return original(inputs,logits,ids,extra)
            self._replace(self.agent.agent_strategy,'post_sampling',post)

        def _record(self,track,base,input_ids,before,after,proposed,output,is_decoding):
            super()._record(track,base,input_ids,before,after,proposed,output,is_decoding)
            event = track.events[-1]
            if event['phase']=='denoise':
                event.update(self.decision,confidence=self.confidence)
                if 'effective_threshold' in event:
                    event['forced'] = not any(p>=event['effective_threshold']
                        for p,m in zip(self.confidence,before) if m==MASKED)
    return StudyBackend


def audit_path(result,prompt_length,rule):
    commits,rounds = {},Counter()
    denoise = mixed = fallback = clipped = forced = 0
    ks = Counter()
    for event in result['events']:
        before,after = event['mask_before'],event['mask_after']
        masked = [i for i,m in enumerate(before) if m==0]
        selected = event['positions']
        assert selected == [i for i in masked if after[i]==1]
        assert all(before[i]==after[i] or (before[i],after[i])==(1,2)
                   for i in range(32) if i not in selected)
        if masked:
            denoise += 1
            mixed += not event['full_generation_block']
            rounds[event['block_start']] += 1
            assert event['round']==rounds[event['block_start']] and selected
            p = np.asarray(event['confidence'])
            if rule['kind']=='factor':
                k,_ = factor_count(p[masked],rule['factor'])
                assert len(selected)==k
                assert min(p[selected])>=max((p[i] for i in masked if i not in selected),default=-1)
            elif rule['kind']=='calibration':
                assert len(selected)==1 and p[selected[0]]==p[masked].max()
            else:
                expected = [i for i in masked if p[i]>=event['effective_threshold']]
                if expected:
                    assert selected==expected
                else:
                    assert len(selected)==1 and p[selected[0]]==p[masked].max()
            ks[len(selected)] += 1
            fallback += event.get('table_fallback',False)
            clipped += event.get('threshold_clipped',False)
            forced += event.get('forced',False)
        else:
            assert not selected
        for i,token in zip(selected,event['tokens']):
            position = event['block_start']+i
            assert position not in commits
            commits[position] = token
    ids = result['token_ids']
    assert [commits[prompt_length+i] for i in range(len(ids))]==ids
    assert len(result['events'])==result['forward_calls']==sum(result['forward_types'].values())
    return {'passed':True,'denoise_forwards':denoise,'mixed_denoise_forwards':int(mixed),
        'commit_count_histogram':dict(ks),'block_rounds':list(rounds.values()),
        'fallback_rounds':int(fallback),'clipped_rounds':int(clipped),'forced_rounds':int(forced)}


def measure(result,prompt_length,rule,caps):
    audit = audit_path(result,prompt_length,rule)
    runs = repeat_runs(result['token_ids'])
    budget = {}
    for cap in caps:
        completed = prompt_length
        for e in result['events']:
            if e['forward']>cap:
                break
            if 0 not in e['mask_after']:
                completed = max(completed,e['block_start']+32)
        tokens = min(len(result['token_ids']),completed-prompt_length)
        ended = result['finish_reason']=='eos' and result['forward_calls']<=cap
        budget[str(cap)] = {'tokens':tokens,'forwards':min(cap,result['forward_calls']),
            'eos':ended,'budget_exhausted':result['forward_calls']>cap,
            'repeat':bool(repeat_runs(result['token_ids'][:tokens]))}
    return dict(audit,exact_repeat=bool(runs),repeat_event=runs[0] if runs else None,budget_views=budget)


def worker(gpu,out,stage,rules=None,question_ids=None,repeats=1):
    from joint50 import environment,score
    environment(gpu)
    Backend = backend_class()
    from lmdeploy.utils import get_logger
    get_logger('lmdeploy').setLevel('ERROR')
    out = Path(out)
    manifest = read(out/'manifest.json')
    study,config = manifest['study'],manifest['engine']
    rules = rules or study[stage]
    data = [r for r in read(out/'data.json') if manifest['gpu_of'][r['unique_id']]==gpu
            and (question_ids is None or r['unique_id'] in question_ids)]
    table = read(out/'calibration.json')['table'] if (out/'calibration.json').exists() else {}
    backend = Backend(config,table)
    grader = Grader(config['grade_timeout'])
    directory = out/stage
    directory.mkdir(exist_ok=True)
    result_path = directory/f'gpu{gpu}.jsonl'
    old = read_jsonl(result_path) if result_path.exists() else []
    done = {(r['question_id'],r['rule']['name'],r['repeat']) for r in old}
    try:
        q = data[0] if data else read(out/'data.json')[0]
        backend.generate([{'id':f'warm-{gpu}','question':q['problem'],'max_new_tokens':96}])
        rng = random.Random(study['seed']+gpu)
        rng.shuffle(data)
        for question in data:
            jobs = [(rule,i) for rule in rules for i in range(repeats)]
            rng.shuffle(jobs)
            for rule,repeat in jobs:
                qid = question['unique_id']
                if (qid,rule['name'],repeat) in done:
                    continue
                backend.set_rule(rule)
                name = f"{rule['name']}-{qid.replace('/','_')}-{repeat}"
                result = backend.generate([{'id':name,'question':question['problem']}])[0]
                seconds = backend.last_seconds
                prompt_length = len(backend.prompt(question['problem']))
                metrics = measure(result,prompt_length,rule,study['budget_views'])
                trace = directory/f'{name}.jsonl.gz'
                with gzip.open(trace,'wt',compresslevel=1) as f:
                    for event in result.pop('events'):
                        f.write(json.dumps(event,separators=(',',':'))+'\n')
                result.update(score(grader,result,question['answer']))
                for b in metrics['budget_views'].values():
                    b['reward'] = result['reward'] if b['eos'] else 0.
                result.update(question_id=qid,rule=rule,repeat=repeat,gpu=gpu,stage=stage,
                    cohort=question['cohort'],seconds=seconds,metrics=metrics,trace_path=str(trace),prompt_length=prompt_length)
                with result_path.open('a') as f:
                    f.write(json.dumps(result,ensure_ascii=False,separators=(',',':'))+'\n')
                print(json.dumps({'stage':stage,'gpu':gpu,'q':qid,'rule':rule['name'],
                    'reward':result['reward'],'forward':result['forward_calls'],'seconds':round(seconds,3)}),flush=True)
    finally:
        grader.close()
        backend.close()


def check_worker(gpu,out):
    from joint50 import environment
    environment(gpu)
    Backend = backend_class()
    from lmdeploy.utils import get_logger
    get_logger('lmdeploy').setLevel('ERROR')
    out = Path(out)
    manifest = read(out/'manifest.json')
    config = manifest['engine']
    backend = Backend(config,{'block':[.87],'step_block':[[.87]]})
    reference = next(r for r in old_rows(OLD) if r['setting']=='Dynamic'
                     and r['question_id']==manifest['calibration_id'])
    data = {r['unique_id']:r for r in read(out/'data.json')}
    rules = [manifest['study']['E0'][3],manifest['study']['E1'][0],
             manifest['study']['E2'][0],manifest['study']['E1'][3],manifest['study']['E0'][3]]
    checks = []
    try:
        for i,rule in enumerate(rules):
            backend.set_rule(rule)
            result = backend.generate([{'id':f'check-{gpu}-{i}',
                'question':data[reference['question_id']]['problem'],'max_new_tokens':128}])[0]
            audit = audit_path(result,len(backend.prompt(data[reference['question_id']]['problem'])),rule)
            if rule['name']=='F090':
                assert result['token_ids']==reference['token_ids'][:len(result['token_ids'])]
                assert trace_difference(events_for(reference)[:len(result['events'])],result['events']) is None
            checks.append({'rule':rule['name'],'passed':True,'forward_calls':result['forward_calls'],
                'seconds':backend.last_seconds,'audit':audit})
        write_json(out/f'checks-gpu{gpu}.json',checks)
    finally:
        backend.close()


def parallel(out,stage,rules=None,question_ids=None,repeats=1):
    manifest = read(out/'manifest.json')
    hardware = subprocess.run(['nvidia-smi','--query-gpu=index,name,memory.used,utilization.gpu',
                               '--format=csv'],capture_output=True,text=True,check=True).stdout
    (out/f'hardware-{stage}.csv').write_text(hardware)
    with ProcessPoolExecutor(len(manifest['study']['gpus']),mp_context=mp.get_context('spawn')) as pool:
        if stage=='check':
            futures = [pool.submit(check_worker,gpu,str(out)) for gpu in manifest['study']['gpus']]
        else:
            futures = [pool.submit(worker,gpu,str(out),stage,rules,question_ids,repeats)
                       for gpu in manifest['study']['gpus']]
        for future in futures:
            future.result()


def calibration(out):
    from joint50 import environment,score
    manifest = read(out/'manifest.json')
    gpu = manifest['study']['gpus'][-1]
    environment(gpu)
    backend = backend_class()(manifest['engine'])
    q = next(r for r in read(out/'data.json') if r['unique_id']==manifest['calibration_id'])
    rule = dict(name='CalibrationT32',kind='calibration')
    backend.set_rule(rule)
    grader = Grader(1.)
    try:
        result = backend.generate([{'id':'calibration','question':q['problem']}])[0]
        seconds = backend.last_seconds
        plen = len(backend.prompt(q['problem']))
        audit = audit_path(result,plen,rule)
        table = calibrate(result['events'],plen,len(result['token_ids']))
        with gzip.open(out/'calibration.trace.jsonl.gz','wt') as f:
            for event in result.pop('events'):
                f.write(json.dumps(event,separators=(',',':'))+'\n')
        result.update(score(grader,result,q['answer']))
        write_json(out/'calibration.json',{'question_id':q['unique_id'],'table':table,
            'audit':audit,'result':result,'seconds':seconds,'gpu':gpu})
    finally:
        grader.close()
        backend.close()


def rows_for(out,stages=('E0','E1','support','E2')):
    return [r for stage in stages for p in sorted((out/stage).glob('gpu*.jsonl')) for r in read_jsonl(p)]


def aggregate(rows):
    tokens = np.asarray([len(r['token_ids']) for r in rows])
    forwards = np.asarray([r['forward_calls'] for r in rows])
    denoise = np.asarray([r['metrics']['denoise_forwards'] for r in rows])
    seconds = np.asarray([r['seconds'] for r in rows])
    histogram = Counter()
    for r in rows:
        histogram.update({int(k):v for k,v in r['metrics']['commit_count_histogram'].items()})
    block_rounds = [v for r in rows for v in r['metrics']['block_rounds']]
    caps = rows[0]['metrics']['budget_views']
    return {'n':len(rows),'correct':sum(r['reward'] for r in rows),'accuracy':float(np.mean([r['reward'] for r in rows])),
        'tokens_total':int(tokens.sum()),'tokens_mean':float(tokens.mean()),
        'forward_total':int(forwards.sum()),'forward_mean':float(forwards.mean()),
        'forward_median':float(np.median(forwards)),'forward_p90':float(np.quantile(forwards,.9)),
        'denoise_total':int(denoise.sum()),'tpf_denoise':float(tokens.sum()/denoise.sum()),
        'tpf_denoise_question_mean':float((tokens/denoise).mean()),
        'tpf_total':float(tokens.sum()/forwards.sum()),'tpf_total_question_mean':float((tokens/forwards).mean()),
        'seconds_total':float(seconds.sum()),'seconds_mean':float(seconds.mean()),
        'seconds_median':float(np.median(seconds)),'seconds_p90':float(np.quantile(seconds,.9)),
        'repeat_n':sum(r['metrics']['exact_repeat'] for r in rows),
        'length_n':sum(r['finish_reason']=='length' for r in rows),
        'grading_timeouts':sum(r['grading']=='timeout' for r in rows),
        'commit_count_histogram':dict(sorted(histogram.items())),
        'block_rounds_mean':float(np.mean(block_rounds)),'block_rounds_histogram':dict(Counter(block_rounds)),
        'forced_fraction':sum(r['metrics']['forced_rounds'] for r in rows)/denoise.sum(),
        'table_fallback_fraction':sum(r['metrics']['fallback_rounds'] for r in rows)/denoise.sum(),
        'threshold_clipped_fraction':sum(r['metrics']['clipped_rounds'] for r in rows)/denoise.sum(),
        'budget_views':{cap:{'correct':sum(r['metrics']['budget_views'][cap]['reward'] for r in rows),
            'budget_exhausted_n':sum(r['metrics']['budget_views'][cap]['budget_exhausted'] for r in rows),
            'forward_total':sum(r['metrics']['budget_views'][cap]['forwards'] for r in rows),
            'repeat_n':sum(r['metrics']['budget_views'][cap]['repeat'] for r in rows)} for cap in caps}}


def frontier(metrics,cost='forward_mean'):
    return [name for name,m in metrics.items() if not any(
        other['accuracy']>=m['accuracy'] and other[cost]<=m[cost]
        and (other['accuracy']>m['accuracy'] or other[cost]<m[cost])
        for key,other in metrics.items() if key!=name)]


def report(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    manifest = read(out/'manifest.json')
    rows = rows_for(out)
    names = list(dict.fromkeys(r['name'] for stage in ('E0','E1','support','E2') for r in manifest['study'][stage]
                               if any(row['rule']['name']==r['name'] for row in rows)))
    groups = {name:[r for r in rows if r['rule']['name']==name] for name in names}
    common = {q['unique_id'] for q in read(out/'data.json')} - {manifest['calibration_id']}
    complete = {name:a for name,a in groups.items() if len(a)==50}
    all50 = {name:aggregate(a) for name,a in complete.items()}
    common49 = {name:aggregate([r for r in a if r['question_id'] in common]) for name,a in complete.items()}
    pairings = {}
    for left in complete:
        for right in complete:
            if left!=right:
                a = {r['question_id']:r['reward'] for r in complete[left] if r['question_id'] in common}
                b = {r['question_id']:r['reward'] for r in complete[right] if r['question_id'] in common}
                pairings[f'{left}->{right}'] = paired_stats(a,b)
    result = {'all50':all50,'common49':common49,'frontier_forward':frontier(common49),
        'frontier_latency':frontier(common49,'seconds_mean'),'paired_accuracy':pairings,
        'completed_paths':len(rows),'formal_forward_total':sum(r['forward_calls'] for r in rows),
        'formal_seconds_total':sum(r['seconds'] for r in rows),
        'note':'Development comparisons; latency includes adapter overhead and shared-GPU interference. Calibration excluded from all common49 comparisons.'}
    write_json(out/'summary.json',result)
    lines = ['# SDAR-B32 Threshold Landscape Study','',
        '日期：2026-11-12。固定原有 50 道开发题、原生 4B-b32、BF16、greedy、TP=1、batch=1、GPU 0–7、8192-token 上限。各配置均从 prompt 完整生成。剩余评估 50 题与最终 400 题未使用。','',
        '## 全部 50 题','',
        '| 配置 | 正确 | 精确重复 | 长度截断 | 平均 tokens | TPF denoise | TPF 全部 forward | 平均 forward | 平均秒 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name,m in all50.items():
        lines.append(f"| {name} | {m['correct']:g}/50 | {m['repeat_n']} | {m['length_n']} | {m['tokens_mean']:.1f} | {m['tpf_denoise']:.2f} | {m['tpf_total']:.2f} | {m['forward_mean']:.1f} | {m['seconds_mean']:.3f} |")
    lines += ['', '## 排除校准题的共同 49 题','',f"校准题：`{manifest['calibration_id']}`。所有规则使用相同分母；校准来源为原生 static T32 的一次完整生成。",'',
        '| 配置 | 正确 / 49 | 平均 forward | 平均秒 | 保底提交比例 | OSDT 回退比例 |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for name,m in common49.items():
        lines.append(f"| {name} | {m['correct']:g}/49 | {m['forward_mean']:.1f} | {m['seconds_mean']:.3f} | {m['forced_fraction']:.1%} | {m['table_fallback_fraction']:.1%} |")
    lines += ['',f"按实际 forward 的开发集前沿：{', '.join(result['frontier_forward'])}。",'',
        '## 定义与边界','',
        '- `TPF denoise` 分母为提交前存在生成 MASK 的实际 forward，包含混合 prompt 首块；纯 prefill 与缓存写入不计。端到端 TPF 分母包含所有实际调用。分子只使用实际返回 token，包含首个 EOS，不含其后并行预测。',
        '- 主表为总 token / 总 forward；逐题平均、P90、512/1024-forward 预算视图及逐轮提交数见 [summary.json](summary.json)。重复 token 同样会提高 TPF，因此同时报告答案质量和整题成本。',
        '- E1 均为固定 B32 下的规则对照。OSDT 使用 Q1、cap=.98、slack=.05、阈值裁剪到 [.75,.98]，表外回退 .90。CadProgress 使用 .95→.80、当前块起点生成长度/8192 的序列进度；均不等于完整上游方法复现。',
        '- E2 使用已有 factor 规则，不使用历史或额外 backbone 调用；低阈值固定 .33/.50/.67 为其支持对照。所有方法共用引擎的至少一位置保底和终止规则。',
        '- 秒数包含逐 forward 调度和轻量观测，不含加载、磁盘写入和判分；各请求同卡但阶段间外部负载可能不同。不能作为推理引擎峰值吞吐或单次 forward 等价 FLOPs 的证明。',
        '- 已按题配对；全部结果属于开发筛查。挑选最佳配置带来选择偏差，尚未使用保留题验证。',
        '', '## 文件','',
        '- [manifest.json](manifest.json)：冻结规则、数据 ID、硬件分配与来源。',
        '- [summary.json](summary.json)：完整指标、逐题配对和前沿。',
        '- [overview.png](overview.png) / [overview.pdf](overview.pdf)：质量、TPF、成本关系。',
        '- 各阶段目录保存逐请求结果与轻量逐轮轨迹；[source/](source/) 保存执行源码与上游材料。']
    (out/'DEVELOPMENT.md').write_text('\n'.join(lines)+'\n')
    if not common49:
        return
    fig,axes = plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
    colors = {'E0':'#397c9c','E1':'#d38436','support':'#8b8b8b','E2':'#3c9478'}
    stage_of = {r['name']:stage for stage in colors for r in manifest['study'][stage]}
    for name,m in common49.items():
        for ax,cost in zip(axes[:2],('forward_mean','seconds_mean')):
            ax.scatter(m[cost],m['accuracy']*100,color=colors[stage_of[name]],s=45)
            ax.annotate(name,(m[cost],m['accuracy']*100),fontsize=7,xytext=(3,3),textcoords='offset points')
    fixed = [(r['threshold'],all50[r['name']]) for r in manifest['study']['support']+manifest['study']['E0'] if r['name'] in all50]
    fixed.sort(key=lambda x:x[0])
    axes[2].plot([t for t,m in fixed],[m['tpf_denoise'] for t,m in fixed],'o-',label='Pooled denoise TPF')
    axes[2].plot([t for t,m in fixed],[m['tpf_denoise_question_mean'] for t,m in fixed],'s--',label='Question mean denoise TPF')
    axes[2].legend(fontsize=8)
    axes[0].set(xlabel='Mean actual forwards / question',ylabel='Accuracy (%)',title='Common 49 development questions')
    axes[1].set(xlabel='Mean request seconds',ylabel='Accuracy (%)',title='Recorded latency (shared GPUs)')
    axes[2].set(xlabel='Fixed threshold',ylabel='Tokens / denoising forward',title='All 50: pooled vs question mean')
    for ax in axes:
        ax.spines[['top','right']].set_visible(False)
        ax.grid(alpha=.15)
    fig.savefig(out/'overview.png',dpi=180)
    fig.savefig(out/'overview.pdf')
    plt.close(fig)
    print(json.dumps({'paths':len(rows),'frontier':result['frontier_forward']}))


def timing(out):
    manifest,summary = read(out/'manifest.json'),read(out/'summary.json')
    chosen = ['F090']
    for stage in ('E0','E1','E2'):
        candidates = [r['name'] for r in manifest['study'][stage]]
        best = min(candidates,key=lambda name:(-summary['common49'][name]['accuracy'],summary['common49'][name]['forward_mean']))
        if best not in chosen:
            chosen.append(best)
    rules = [r for stage in ('E0','E1','E2') for r in manifest['study'][stage] if r['name'] in chosen]
    write_json(out/'timing_plan.json',{'rules':rules,'ids':manifest['timing_ids'],
        'repeats':manifest['study']['timing_repeats'],'selection':'Best accuracy then fewest forwards within each family, plus F090; development selection, timing only.'})
    parallel(out,'timing',rules,manifest['timing_ids'],manifest['study']['timing_repeats'])
    rows = rows_for(out,('timing',))
    original = {(r['rule']['name'],r['question_id']):r for r in rows_for(out)}
    for row in rows:
        reference = original[row['rule']['name'],row['question_id']]
        assert row['token_ids']==reference['token_ids'] and row['forward_calls']==reference['forward_calls']
    write_json(out/'timing.json',{name:aggregate([r for r in rows if r['rule']['name']==name]) for name in chosen})


def audit(out):
    from run import fingerprint
    manifest = read(out/'manifest.json')
    original = read(GA/'runs/pilot8192-20261110/manifest.json')
    assert fingerprint(original['config'])==read(out/'original_fingerprint.json')
    assert all(file_hash(GA/name)==digest for name,digest in manifest['code'].items())
    rows = rows_for(out)
    assert len(rows)==len({(r['rule']['name'],r['question_id']) for r in rows})
    old = {r['question_id']:r for r in old_rows(OLD) if r['setting']=='Dynamic'}
    reference_n = 0
    for row in rows:
        data = read_jsonl(row['trace_path'])
        plen = row['prompt_length']
        result = dict(row,events=data)
        assert json.loads(json.dumps(audit_path(result,plen,row['rule'])))=={k:v for k,v in row['metrics'].items()
            if k not in ('exact_repeat','repeat_event','budget_views')}
        if row['rule']['name']=='F090':
            ref = old[row['question_id']]
            assert row['token_ids']==ref['token_ids'] and row['finish_reason']==ref['finish_reason']
            assert row['forward_calls']==ref['forward_calls']
            assert trace_difference(events_for(ref),data) is None
            reference_n += 1
    checks = [r for p in out.glob('checks-gpu*.json') for r in read(p)]
    write_json(out/'audit.json',{'passed':True,'formal_paths':len(rows),'short_checks':len(checks),
        'original_f090_matches':reference_n,'original_code_model_data_unchanged':True,
        'study_code_hashes_match':True,'all_paths_reconstructed':True})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('prepare','check','E0','calibrate','E1','support','E2','report','timing','audit'))
    parser.add_argument('--out',type=Path,default=OUT)
    args = parser.parse_args()
    if args.stage=='prepare': prepare(args.out)
    elif args.stage=='calibrate': calibration(args.out)
    elif args.stage=='report': report(args.out)
    elif args.stage=='timing': timing(args.out)
    elif args.stage=='audit': audit(args.out)
    else: parallel(args.out,args.stage)


if __name__=='__main__':
    main()
