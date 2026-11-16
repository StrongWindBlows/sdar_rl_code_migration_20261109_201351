"""CPU-only alignment of manually reviewed failures with existing SDAR traces."""
import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
import sys

from tokenizers import Tokenizer
from protocol import ROOT, file_hash, write_json

sys.path.insert(0, str(ROOT / 'sdar_justgmpo'))
from vendor.gmpo.math_grader import extract_boxed_answer

CORE = ('block_start', 'round', 'input_ids', 'mask_before', 'mask_after', 'positions', 'tokens')


def repetition(ids, max_period=64):
    """Earliest exact tandem run: >=4 copies and >=32 tokens; onset is copy 2."""
    candidates = []
    for period in range(1, max_period + 1):
        equal = 0
        threshold = max(4 * period, 32)
        for i in range(period, len(ids) + 1):
            if i < len(ids) and ids[i] == ids[i - period]:
                equal += 1
                continue
            length = equal + period
            if equal and length >= threshold:
                start = i - equal - period
                candidates.append({'motif_start': start, 'onset': start + period,
                    'period': period, 'end_exclusive': i, 'copies': length // period,
                    'detection_end_exclusive': start + threshold})
            equal = 0
    return min(candidates, key=lambda r: (r['onset'], r['period'])) if candidates else None


def token_rows(row, tokenizer, events):
    commits = {}
    prompt = row['state']['prompt_length']
    for event in events:
        for i, value in zip(event['positions'], event['tokens']):
            position = event['block_start'] + i
            assert position not in commits, 'Natural baseline must commit each token only once'
            commits[position] = {'forward': event['forward'], 'round': event['round'],
                'block_start': event['block_start'], 'block_position': i,
                'probability': event['topk_probs'][i][0], 'token_id': value}
    pieces = [tokenizer.decode([value], skip_special_tokens=True) for value in row['token_ids']]
    assert ''.join(pieces) == row['text'], 'Token/character alignment must be lossless'
    tokens, offset = [], 0
    for generated, (value, piece) in enumerate(zip(row['token_ids'], pieces)):
        absolute = prompt + generated
        assert commits[absolute]['token_id'] == value
        tokens.append({'generated_position': generated, 'absolute_position': absolute,
            'text': piece, 'char_start': offset, 'char_end': offset + len(piece), **commits[absolute]})
        offset += len(piece)
    return tokens


def observations(token, events, tokenizer):
    i = token['block_position']
    records = []
    for event in events:
        if event['block_start'] != token['block_start']:
            continue
        values, probabilities = event['topk_ids'][i], event['topk_probs'][i]
        rank = values.index(token['token_id']) if token['token_id'] in values else None
        records.append({'forward': event['forward'], 'round': event['round'], 'phase': event['phase'],
            'mask_before': event['mask_before'][i], 'committed_this_round': i in event['positions'],
            'top1': tokenizer.decode([event['current_ids'][i]], skip_special_tokens=False),
            'top1_probability': probabilities[0],
            'committed_value_probability_if_top5': probabilities[rank] if rank is not None else None,
            'probability_upper_bound_if_outside_top5': probabilities[-1] if rank is None else None})
    return records


def snapshot(row, events, forward):
    event = next(e for e in events if e['forward'] == forward)
    prompt = row['state']['prompt_ids']
    return {'id': f"{row['question_id']}@{event['block_start']}:{event['round']}:early",
        'question_id': row['question_id'], 'block_start': event['block_start'],
        'round': event['round'], 'source_forward': forward, 'prompt_ids': prompt,
        'prompt_length': len(prompt), 'prefix_ids': (prompt + row['token_ids'])[:event['block_start']],
        'replay': [{k: e[k] for k in CORE} for e in events if e['forward'] < forward],
        'input_ids': event['input_ids'], 'mask': event['mask_before'],
        'masked_positions': [i for i, flag in enumerate(event['mask_before']) if flag == 0],
        'positions': event['positions'], 'current_ids': event['current_ids'],
        'historical_ids': event['historical_ids'], 'topk_ids': event['topk_ids'],
        'topk_probs': event['topk_probs'], 'confidence': [p[0] for p in event['topk_probs']]}


def branch_summary(rows, baseline, eos_ids):
    start = baseline['state']['block_start'] + 32 - baseline['state']['prompt_length']
    answers = Counter(extract_boxed_answer(r['text']) for r in rows)
    return {'actions': len(rows), 'unique_full_token_sequences': len({tuple(r['token_ids']) for r in rows}),
        'unique_decoded_texts': len({r['text'] for r in rows}),
        'unique_suffixes_after_source_block': len({tuple(r['token_ids'][start:]) for r in rows}),
        'same_suffix_as_baseline': sum(r['token_ids'][start:] == baseline['token_ids'][start:] for r in rows),
        'answer_counts': [{'boxed_answer': answer, 'count': n} for answer, n in answers.items()],
        'finish_reasons': dict(Counter(r['finish_reason'] for r in rows)),
        'actions_containing_eos': sum(any(t in eos_ids for t in r['action']['tokens']) for r in rows),
        'correct': sum(r['reward'] for r in rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    work, out = args.run / 'pilot', args.run / 'exp1_5'
    annotations = json.loads((out / 'annotations.json').read_text())
    config = json.loads((args.run / 'manifest.json').read_text())['config']
    rows = [r for p in work.glob('exp0-gpu*.json') for r in json.loads(p.read_text()) if r['setting'] == 'T8']
    failed = {r['question_id']: r for r in rows if not r['reward']}
    assert set(failed) == {a['question_id'] for a in annotations}
    pilot = set(json.loads((Path(config['data']) / 'manifest.json').read_text())['pilot_ids'])
    assert set(failed) <= pilot
    tokenizer = Tokenizer.from_file(str(Path(config['model']) / 'tokenizer.json'))
    eos_ids = set(json.loads((Path(config['model']) / 'generation_config.json').read_text())['eos_token_id'])
    events = defaultdict(list)
    groups = {r['question_id']: 'correct' if r['reward'] else 'failed' for r in rows}
    readouts = {name: Counter() for name in ('correct', 'failed')}
    trace_files = sorted(work.glob('exp0-T8-gpu*.trace.jsonl.gz'))
    for path in trace_files:
        with gzip.open(path, 'rt') as handle:
            for line in handle:
                e = json.loads(line)
                if e['request_id'] in failed:
                    events[e['request_id']].append(e)
                if e['phase'] != 'denoise':
                    continue
                c = readouts[groups[e['request_id']]]
                for i in range(31):
                    if e['mask_before'][i] != 1 or e['mask_before'][i + 1] != 1:
                        continue
                    if e['input_ids'][i] == e['input_ids'][i + 1]:
                        continue
                    c['distinct_visible_adjacent_pairs'] += 1
                    c['self_matches'] += e['current_ids'][i] == e['input_ids'][i]
                    c['next_matches'] += e['current_ids'][i] == e['input_ids'][i + 1]
    branch_rows = {mode: [r for p in (work / f'T8-{mode}').glob('exp1-gpu*.json') for r in json.loads(p.read_text())]
                   for mode in ('confidence', 'disagreement')}
    cases, rollback = [], []
    for annotation in annotations:
        q = annotation['question_id']
        row, trace = failed[q], events[q]
        tokens = token_rows(row, tokenizer, trace)
        by_position = {t['absolute_position']: t for t in tokens}
        suspects = [dict(by_position[p], observations=observations(by_position[p], trace, tokenizer))
                    for p in annotation['suspect_absolute_positions']]
        first = min(t['forward'] for t in suspects)
        char_start = row['text'].index(annotation['quote'])
        char_end = char_start + len(annotation['quote'])
        span = [t for t in tokens if t['char_start'] < char_end and t['char_end'] > char_start]
        error_end = max(t['forward'] for t in span)
        repeat = repetition(row['token_ids'])
        if repeat:
            repeat['onset_token'] = tokens[repeat['onset']]
            repeat['detection_forward'] = max(t['forward'] for t in tokens[repeat['motif_start']:repeat['detection_end_exclusive']])
            repeat['motif_text'] = tokenizer.decode(row['token_ids'][repeat['motif_start']:repeat['motif_start']+repeat['period']])
        before = [t for t in tokens if t['forward'] < first]
        source = row['state']
        case = {'question_id': q, 'annotation': annotation, 'suspects': suspects,
            'first_suspect_commit_forward': first, 'error_span_fully_committed_forward': error_end,
            'error_span_generated_positions': [span[0]['generated_position'], span[-1]['generated_position']],
            'committed_before_first_suspect': before, 'committed_before_count': len(before),
            'error_block_before_first_suspect': next(e for e in trace if e['forward'] == first),
            'repetition': repeat, 'original_intervention_forward': source['source_forward'],
            'original_intervention_block_generated_start': source['block_start']-source['prompt_length'],
            'error_before_original_intervention': error_end < source['source_forward'],
            'repetition_before_original_intervention': repeat['detection_forward'] < source['source_forward'] if repeat else None,
            'branches': {mode: branch_summary([r for r in data if r['question_id']==q],row,eos_ids)
                         for mode,data in branch_rows.items() if any(r['question_id']==q for r in data)}}
        cases.append(case)
        rollback.append({'question_id': q, 'gpu': row['gpu'], 'annotation': annotation,
            'snapshot': snapshot(row, trace, annotation['rollback_forward'])})
        with (out / (q.split('/')[-2]+'_'+q.split('/')[-1].replace('.json','.tokens.jsonl'))).open('w') as handle:
            for token in tokens:
                handle.write(json.dumps(token, ensure_ascii=False)+'\n')
    write_json(out / 'analysis.json', {'scope':'four failed T8 pilot questions; existing traces only',
        'position_convention':'zero-based; generated excludes prompt; forward is one-based actual calls',
        'annotation_hash':file_hash(out/'annotations.json'), 'analysis_code_hash':file_hash(__file__),
        'readout_semantics_check':readouts, 'cases':cases})
    write_json(out / 'rollback_cases.json', rollback)
    print(json.dumps([{'question':c['question_id'],'error_commit':c['first_suspect_commit_forward'],
        'error_complete':c['error_span_fully_committed_forward'],
        'repeat':{k:c['repetition'][k] for k in ('onset','period','detection_forward')} if c['repetition'] else None,
        'intervention':c['original_intervention_forward'],'committed_before':c['committed_before_count']}
        for c in cases],indent=2))


if __name__ == '__main__':
    main()
