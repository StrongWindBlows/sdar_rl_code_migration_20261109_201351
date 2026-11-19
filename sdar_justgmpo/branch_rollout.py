"""Collect one-action interventions using the existing Dynamic 0.9 backend."""
from contextlib import ExitStack
import logging
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

import torch

from eval_decoding import counted_result
from inference_files import inference_model_path

GA = Path(__file__).resolve().parents[1] / 'ga'


def marginal_position(event, threshold, band):
    """Pick the first state's closest optional token to the native threshold."""
    mask = event['mask_before']
    masked = [i for i, flag in enumerate(mask) if flag == 0]
    if not event['full_generation_block'] or not 2 <= len(masked) < len(mask):
        return None
    probabilities = [scores[0] for scores in event['topk_probs']]
    # Inference compares BF16 scores. Excluding all maximum ties keeps the
    # mandatory top-1 commit present in both actions.
    rounded = torch.tensor(probabilities, dtype=torch.bfloat16).tolist()
    maximum = max(rounded[i] for i in masked)
    optional = [i for i in masked if rounded[i] < maximum
                and abs(probabilities[i] - threshold) <= band
                and event['current_ids'][i] == event['topk_ids'][i][0]]
    return min(optional, key=lambda i: (abs(probabilities[i] - threshold), i), default=None)


def alternate_action(event, position):
    committed = position in event['positions']
    positions = sorted(set(event['positions']) ^ {position})
    return committed, dict(positions=positions, tokens=[event['current_ids'][i] for i in positions])


def branch_targets(baseline, alternate, source_forward, committed):
    """Return commit minus defer. Replay before the shared decision is experimental work."""
    reference_cost = baseline['forward_calls'] - source_forward + 1
    alternate_cost = alternate['future_forwards'] + 1
    direction = 1 if committed else -1
    return dict(delta_reward=direction * (baseline['reward'] - alternate['reward']),
                delta_forwards=direction * (reference_cost - alternate_cost),
                native_commit=committed, reference_remaining_forwards=reference_cost,
                alternate_remaining_forwards=alternate_cost,
                replay_overhead=alternate['restore_forwards'] - 1,
                alternate_reward=alternate['reward'])


def select_branch(backend, result, job, config, verify):
    from early import snapshot
    from protocol import restoration_difference

    for event in result['events']:
        position = marginal_position(event, config['threshold'], config['band'])
        if position is not None:
            break
    else:
        return None, None
    row = dict(result, question_id=job['id'], state={'prompt_ids': job['prompt_ids']})
    state = snapshot(row, result['events'], event['forward'])
    committed, action = alternate_action(event, position)
    replay = dict(id=job['id'], snapshot=state)
    if verify:
        original = {key: event[key] for key in ('positions', 'tokens')}
        restored, = backend.generate([dict(replay, action=original)])
        difference = restoration_difference(result, restored, state)
        if difference:
            raise ValueError(f"Question {job['id']} failed native replay: {difference}")
    # Read the vector used by the actual native head at the intervention.
    # This also keeps cached-attention and dense rescoring roundoff out of the labels.
    agent = backend.agent
    get_logits = agent.get_logits
    hidden = []

    def capture(states):
        track, = backend.tracks.values()
        if track.forward_calls == state['source_forward']:
            hidden.append(states[0, position - backend.block].tolist())
        return get_logits(states)

    with patch.object(agent, 'get_logits', capture):
        alternative, = backend.generate([dict(replay, action=action)])
    source = alternative['events'][0]
    if (
        alternative['restore_forwards'] != state['source_forward']
        or source['input_ids'] != state['input_ids']
        or source['mask_before'] != state['mask']
        or any(source[k] != action[k] for k in ('positions', 'tokens'))
        or source['current_ids'][position] != state['current_ids'][position]
    ):
        raise ValueError(f"Question {job['id']} did not replay the selected decision")
    prefix = result['events'][:state['source_forward'] - 1]
    counted_result(dict(alternative, events=prefix + alternative['events']), len(job['prompt_ids']))
    vector, = hidden
    decision = dict(prefix_ids=state['prefix_ids'], input_ids=state['input_ids'], hidden=vector,
                    position=position, token_id=state['current_ids'][position],
                    reference_probability=state['confidence'][position],
                    source_forward=state['source_forward'], native_commit=committed,
                    original_action_verified=verify)
    return decision, alternative


def dynamic_worker(config, path, rows, collect, verify, gpu):
    from engine import Backend
    from protocol import Grader

    setting = dict(strategy='low_confidence_dynamic', steps=config['block_size'],
                   threshold=config['threshold'])
    with ExitStack() as stack:
        model = stack.enter_context(inference_model_path(path))
        backend = Backend(dict(model=str(model), block_length=config['block_size'],
            max_new_tokens=config['max_tokens'], batch_size=1, cache_fraction=config['cache_fraction'],
            eager=True, seed=config['seed'], trace=collect, top_k_record=1), setting)
        stack.callback(backend.close)
        logging.getLogger('lmdeploy').setLevel(logging.WARNING)
        grader = Grader()
        stack.callback(grader.close)
        output = []
        for row in rows:
            job = dict(id=str(row['index']), prompt_ids=backend.prompt(row['question']))
            result, = backend.generate([job])
            baseline, _ = counted_result(result, len(job['prompt_ids']))
            answer = row['ground_truth_answer']
            baseline |= grader.score(result['text'], answer, result['finished'])
            record = {key: baseline[key] for key in ('reward', 'finished', 'forward_calls',
                      'denoise_forwards', 'grading')}
            record.update(index=row['index'], num_tokens=len(result['token_ids']))
            if collect:
                decision, alternate = select_branch(backend, result, job, config, verify)
                if decision is not None:
                    alternate |= grader.score(alternate['text'], answer, alternate['finished'])
                    decision |= branch_targets(baseline, alternate, decision['source_forward'],
                                               decision['native_commit'])
                    decision['alternate_grading'] = alternate['grading']
                    verify = False
                record['branch'] = decision
            output.append(record)
            if len(output) % 32 == 0:
                print(f"{'collect' if collect else 'dynamic'} GPU {gpu}: {len(output)}/{len(rows)} questions",
                      flush=True)
    return output


def sequential_worker(config, path, rows):
    from transformers import AutoTokenizer
    from protocol import Grader
    from rollout import Rollout
    from runtime import format_prompt

    draws = config['sequential_draws']
    batch_size = config['sequential_batch_size']
    with ExitStack() as stack:
        model = stack.enter_context(inference_model_path(path))
        tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
        rollout = Rollout(model, tokenizer, config['max_tokens'], batch_size,
                          block_size=config['block_size'], cache_fraction=config['cache_fraction'])
        stack.callback(rollout.close)
        grader = Grader()
        stack.callback(grader.close)
        output = []
        for start in range(0, len(rows), batch_size // draws):
            batch = rows[start:start + batch_size // draws]
            prompts = [format_prompt(tokenizer, row['question']) for row in batch]
            samples = rollout.sample(prompts, draws, config['seed'] + batch[0]['index'] * draws)
            for j, row in enumerate(batch):
                group = samples[j * draws:(j + 1) * draws]
                grades = [grader.score(tokenizer.decode(ids, skip_special_tokens=True),
                                       row['ground_truth_answer'], finished) for ids, finished in group]
                output.append(dict(index=row['index'], rewards=[g['reward'] for g in grades],
                                   timeouts=sum(g['grading'] == 'timeout' for g in grades)))
    return output


def worker(config, gpu, path, rows, mode, verify):
    """One independent TP1 replica per assigned GPU."""
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    sys.path.insert(0, str(GA))
    torch.set_num_threads(4)
    torch.manual_seed(config['seed'])
    start = time.monotonic()
    if mode == 'sequential':
        result = sequential_worker(config, path, rows)
    else:
        result = dynamic_worker(config, path, rows, mode == 'collect', verify, gpu)
    print(f"{mode} {gpu} finished {len(rows)} questions in {time.monotonic()-start:.1f}s", flush=True)
    return result
