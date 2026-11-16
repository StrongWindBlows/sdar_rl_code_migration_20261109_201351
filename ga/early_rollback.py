"""Twelve bounded early rollback probes: original Current, reopen one, reopen two."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import sys

from early import repetition
from protocol import (Grader, ROOT, file_hash, forward_budget, read_jsonl,
                      restoration_difference, write_json)
from run import fingerprint

GA = Path(__file__).resolve().parent


def rollback_backend():
    from engine import Backend

    class RollbackBackend(Backend):
        def _install(self):
            super()._install()
            original = self.agent.agent_strategy.post_sampling

            def post_sampling(inputs, logits, next_ids, extra):
                assert len(inputs.model_metas) == 1
                track = self.tracks[inputs.model_metas[0]['ga_session']]
                at_source = track.source_pending and len(track.events) == len(track.job['snapshot']['replay'])
                previous_end = track.valid_end
                writer, self.writer = self.writer, None
                try:
                    output, extra = original(inputs, logits, next_ids, extra)
                finally:
                    self.writer = writer
                event = track.events[-1]
                if at_source:
                    reopen = track.job['reopen']
                    event['reopened_positions'] = reopen
                    state = track.job['snapshot']
                    mask_id = state['input_ids'][state['masked_positions'][0]]
                    for i in reopen:
                        assert int(extra.dllm_mask[i]) == 1
                        extra.dllm_mask[i] = 0
                        output[i] = mask_id
                        event['mask_after'][i] = 0
                        track.token_map.pop(state['block_start'] + i)
                    if reopen:
                        track.valid_end = previous_end
                if writer:
                    writer.write(json.dumps(event, separators=(',', ':')) + '\n')
                return output, extra

            self._replace(self.agent.agent_strategy, 'post_sampling', post_sampling)

    return RollbackBackend


def worker(gpu, config, cases, run):
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu), PYTHONDONTWRITEBYTECODE='1',
        HF_HOME=str(GA/'cache/hf'), HF_MODULES_CACHE=str(GA/'cache/hf_modules'),
        TRITON_CACHE_DIR=str(GA/'cache/triton'), TORCHINDUCTOR_CACHE_DIR=str(GA/'cache/inductor'),
        SDAR_FULL_VOCAB='1', TOKENIZERS_PARALLELISM='false')
    sys.dont_write_bytecode = True
    backend = rollback_backend()(config, next(s for s in config['settings'] if s['name']=='T8'))
    grader = Grader(config['grade_timeout'])
    work, out = Path(run)/'pilot', Path(run)/'exp1_5'
    baselines = {r['question_id']:r for p in work.glob('exp0-gpu*.json')
                 for r in json.loads(p.read_text()) if r['setting']=='T8'}
    questions = {r['unique_id']:r for r in read_jsonl(Path(config['data'])/'test.jsonl')}
    baseline_events = read_jsonl(work/f'exp0-T8-gpu{gpu}.trace.jsonl.gz')
    records = []
    try:
        for case in cases:
            q, state, annotation = case['question_id'], case['snapshot'], case['annotation']
            baseline = dict(baselines[q], events=[e for e in baseline_events if e['request_id']==q])
            action = {'positions':state['positions'], 'tokens':[state['current_ids'][i] for i in state['positions']]}
            # Both reopened positions are in this active block: at most one extra T8 round.
            budget = forward_budget(state,config['max_new_tokens'],8) + math.ceil(2/4)
            for count, absolute in ((0,[]),(1,annotation['rollback_one']),(2,annotation['rollback_two'])):
                reopen = [p-state['block_start'] for p in absolute]
                name = q.replace('/','_').replace('.json','')+f'-reopen{count}'
                job = {'id':name,'snapshot':state,'action':action,'future_budget':budget,'reopen':reopen}
                result = backend.generate([job],out/f'{name}.trace.jsonl.gz')[0]
                events = result.pop('events')
                result.pop('state')
                source = events[0]
                assert source['forward']==state['source_forward']
                assert source['input_ids']==state['input_ids'] and source['mask_before']==state['mask']
                assert source['positions']==action['positions'] and source['tokens']==action['tokens']
                expected_input, expected_mask = list(state['input_ids']),list(state['mask'])
                for i,value in zip(action['positions'],action['tokens']):
                    expected_input[i],expected_mask[i] = value,1
                mask_id = state['input_ids'][state['masked_positions'][0]]
                for i in reopen:
                    expected_input[i],expected_mask[i] = mask_id,0
                assert source['mask_after']==expected_mask
                assert events[1]['input_ids']==expected_input and events[1]['mask_before']==expected_mask
                assert result['restore_forwards']==state['source_forward']
                assert result['future_forwards']<=budget
                assert result['forward_calls']==result['restore_forwards']+result['future_forwards']
                if count==0:
                    difference = restoration_difference(baseline,dict(result,events=events),state)
                    write_json(out/(name+'.verification.json'),{'question_id':q,'passed':difference is None,'difference':difference})
                    assert difference is None, difference
                probes = []
                for i in reopen:
                    observed = []
                    for event in events[1:]:
                        if event['block_start']!=state['block_start'] or event['mask_before'][i]!=0:
                            continue
                        observed.append({'forward':event['forward'],'topk_ids':event['topk_ids'][i],
                            'topk_probs':event['topk_probs'][i], 'committed':i in event['positions']})
                    assert observed and any(e['committed'] for e in observed)
                    probes.append({'absolute_position':state['block_start']+i,'observations':observed})
                result.update(question_id=q,gpu=gpu,reopened_tokens=count,reopened_absolute_positions=absolute,
                    source_forward=state['source_forward'],future_budget=budget,
                    source_and_next_input_verified=True,remasked_probes=probes,
                    repetition=repetition(result['token_ids']),inference_seconds=backend.last_seconds)
                result.update(grader.score(result['text'],questions[q]['answer'],result['finished']))
                records.append(result)
                write_json(out/f'rollback-gpu{gpu}.json',records)
                print(json.dumps({'question':q,'reopen':count,'reward':result['reward'],
                    'tokens':len(result['token_ids']),'finish':result['finish_reason']}),flush=True)
    finally:
        grader.close()
        backend.close()
    return {'gpu':gpu,'continuations':len(records)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    out = run/'exp1_5'
    parent = json.loads((run/'manifest.json').read_text())
    config = parent['config']
    assert config['batch_size']==1 and config['max_new_tokens']==8192
    assert fingerprint(config)['digest']==parent['fingerprint']['digest']
    cases = json.loads((out/'rollback_cases.json').read_text())
    assert len(cases)==4 and not list(out.glob('rollback-gpu*.json'))
    write_json(out/'rollback_manifest.json',{'parent_fingerprint':parent['fingerprint']['digest'],
        'code':{p.name:file_hash(p) for p in (Path(__file__),GA/'early.py')},
        'annotations_sha256':file_hash(out/'annotations.json'),'cases_sha256':file_hash(out/'rollback_cases.json'),
        'config':config,'variants':[0,1,2],'planned_continuations':12,
        'selection':'Manual, retrospective localization on the four failed development questions; not an online detector.',
        'action':'Apply the original Current commit, reopen selected active-block positions, then return to T8. No token values are forced.',
        'budget':'Common remaining T8 forward upper bound plus one for every variant, including control; unchanged 8192 total token cap.'})
    gpus = sorted({case['gpu'] for case in cases})
    with ProcessPoolExecutor(max_workers=len(gpus),mp_context=mp.get_context('spawn')) as executor:
        futures = [executor.submit(worker,gpu,dict(config),[c for c in cases if c['gpu']==gpu],run) for gpu in gpus]
        for future in futures:
            print(future.result(),flush=True)


if __name__=='__main__':
    main()
