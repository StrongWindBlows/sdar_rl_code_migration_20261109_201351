"""Small, process-local adapters around the existing LMDeploy engine."""
import asyncio
from collections import Counter
from contextlib import ExitStack
from dataclasses import dataclass, field
import gzip
import json
import math
from pathlib import Path
import random
import sys
import time

import torch
from transformers import AutoTokenizer

from protocol import INSTRUCTION, ROOT, stable_seed

sys.path.insert(0, str(ROOT / 'DiRL'))
from lmdeploy import GenerationConfig, PytorchEngineConfig
from lmdeploy.messages import ResponseType
from lmdeploy.pytorch import consts
from lmdeploy.pytorch.config import UnmaskingStrategy
from lmdeploy.pytorch.engine.engine import Engine, InputsMakerAsync
from lmdeploy.pytorch.strategies.ar.sampling import ARSamplingStrategy
from lmdeploy.pytorch.strategies.dllm.model_agent import DLLMStoppingCriteria

MASKED, UNMASKED, CACHED = consts.DLLM_MASKED, consts.DLLM_UNMASKED, consts.DLLM_CACHED


@dataclass
class Track:
    job: dict
    prompt_ids: list
    session_id: int
    events: list = field(default_factory=list)
    previous: dict = field(default_factory=dict)
    rounds: dict = field(default_factory=dict)
    pending: dict = field(default_factory=dict)
    counts: Counter = field(default_factory=Counter)
    state: dict | None = None
    raw: dict = field(default_factory=dict)
    seen_states: int = 0
    forward_calls: int = 0
    future_forwards: int = 0
    restore_forwards: int = 0
    budget_exhausted: bool = False
    source_reached: bool = False

    def __post_init__(self):
        self.rng = random.Random(stable_seed(self.job['id'], self.job['state_seed']))
        self.token_map = dict(enumerate(self.prompt_ids))
        self.valid_end = len(self.prompt_ids)

    @property
    def source_pending(self):
        return self.job.get('snapshot') is not None and not self.source_reached


class Backend:
    def __init__(self, config, setting):
        self.config = config
        self.block = config['block_length']
        self.stack = ExitStack()
        self.tracks = {}
        self.session = 0
        self.writer = None
        self.batch_forward_calls = 0
        self.max_observed_batch = 0
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        self.tokenizer = AutoTokenizer.from_pretrained(config['model'], trust_remote_code=True, local_files_only=True)
        generation = json.loads((Path(config['model']) / 'generation_config.json').read_text())
        self.eos_ids = set(generation['eos_token_id'])
        self.core = Engine(config['model'], PytorchEngineConfig(
            tp=1, dtype='bfloat16', session_len=config['max_new_tokens'] + 4096,
            max_batch_size=config['batch_size'], cache_max_entry_count=config['cache_fraction'],
            max_prefill_token_num=4096, eager_mode=config['eager'], enable_prefix_caching=False,
            distributed_executor_backend='uni', dllm_block_length=self.block,
            dllm_denoising_steps=setting['steps'], dllm_unmasking_strategy=setting['strategy'],
            dllm_confidence_threshold=setting.get('threshold', 0.9)))
        self.agent = self.core.executor.model_agent
        self.agent.sampling_vocab_size = None
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self._install()
        self.set_setting(setting)

    def _replace(self, obj, name, replacement):
        original = getattr(obj, name)
        setattr(obj, name, replacement)
        self.stack.callback(setattr, obj, name, original)
        return original

    def set_setting(self, setting):
        self.setting = dict(setting)
        for strategy in (self.core.model_agent_strategy, self.agent.agent_strategy):
            cfg = strategy.unmasking_processor.dllm_config
            cfg.unmasking_strategy = UnmaskingStrategy.from_str(setting['strategy'])
            cfg.denoising_steps = setting['steps']
            cfg.confidence_threshold = setting.get('threshold', 0.9)

    def prompt(self, question):
        text = self.tokenizer.apply_chat_template(
            [{'role': 'user', 'content': question + INSTRUCTION}], tokenize=False, add_generation_prompt=True)
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _install(self):
        core, agent = self.core, self.agent
        # Expose each stop/budget boundary instead of computing past it in an internal loop.
        self._replace(core.engine_strategy, 'get_num_loops', lambda is_decoding: 1)

        async def no_prefetch(inputs_maker):
            return None, None

        self._replace(InputsMakerAsync, 'prefetch_next_inputs', no_prefetch)
        original_inputs = core.create_model_inputs

        def make_inputs(messages, is_prefill):
            inputs = original_inputs(messages, is_prefill)
            inputs.model_metas = [{'ga_session': message.session_id} for message in messages]
            return inputs

        self._replace(core, 'create_model_inputs', make_inputs)
        original_outputs = core._make_infer_outputs

        def make_outputs(batched_outputs, running, is_decoding):
            for message in running:
                if self.tracks[message.session_id].budget_exhausted:
                    # The existing engine suppresses unchanged-length responses, including a mid-block FINISH.
                    message.resp.data = None
            return original_outputs(batched_outputs, running, is_decoding)

        self._replace(core, '_make_infer_outputs', make_outputs)
        def sampling_inputs(sequences):
            out = ARSamplingStrategy.make_sampling_inputs(core.sampling_strategy, sequences)
            for name in ('temperature', 'bad_words', 'bad_mask', 'stop_words', 'stop_mask',
                         'repetition_penalty', 'top_k', 'top_p', 'min_p', 'random_seeds',
                         'random_offsets', 'all_ids', 'guided_input_ids', 'num_ignore_eos'):
                value = getattr(out, name)
                if value is not None:
                    setattr(out, name, value.repeat_interleave(self.block, dim=0))
            out.response_formats = tuple(x for x in out.response_formats for _ in range(self.block))
            out.batch_size *= self.block
            return out

        self._replace(core.sampling_strategy, 'make_sampling_inputs', sampling_inputs)
        original_forward = agent.async_forward

        async def forward(inputs, *args, **kwargs):
            self.batch_forward_calls += 1
            self.max_observed_batch = max(self.max_observed_batch, len(inputs.model_metas))
            for meta in inputs.model_metas:
                track = self.tracks[meta['ga_session']]
                track.forward_calls += 1
                if track.source_pending:
                    track.restore_forwards += 1
                    track.counts['restore_decode' if inputs.is_decoding else 'restore_prefill'] += 1
                else:
                    track.future_forwards += 1
                    track.counts['prefill' if not inputs.is_decoding else 'decode'] += 1
            return await original_forward(inputs, *args, **kwargs)

        self._replace(agent, 'async_forward', forward)
        original_sample = agent.async_sampling_logits

        async def sampling(logits, sampling_inputs, inputs):
            if self.config['trace']:
                probabilities = logits.float().softmax(-1).view(-1, self.block, logits.shape[-1])
                values, ids = probabilities.topk(self.config['top_k_record'], dim=-1)
                entropy = -(probabilities * probabilities.clamp_min(1e-30).log()).sum(-1)
                values, ids, entropy = values.tolist(), ids.tolist(), entropy.tolist()
                bases = (inputs.history_lengths + inputs.seq_length - self.block).tolist()
                for row, meta in enumerate(inputs.model_metas):
                    track = self.tracks[meta['ga_session']]
                    historical = track.previous.get(bases[row])
                    history_probability = None
                    if historical is not None:
                        history_ids = torch.tensor(historical, device=logits.device)
                        history_probability = probabilities[row].gather(-1, history_ids[:, None]).flatten().tolist()
                    track.raw = {'topk_ids': ids[row], 'topk_probs': values[row], 'entropy': entropy[row],
                                 'history_probability_now': history_probability}
            return await original_sample(logits, sampling_inputs, inputs)

        self._replace(agent, 'async_sampling_logits', sampling)
        original_post = agent.agent_strategy.post_sampling

        def post_sampling(inputs, logits, next_ids, extra):
            before = extra.dllm_mask.view(-1, self.block).clone()
            input_ids = agent.agent_strategy.slice_outputs(inputs.input_ids.flatten(), inputs.seq_length)
            input_ids = input_ids.view(-1, self.block)
            proposals = next_ids.view(-1, self.block).clone()
            out_ids, extra = original_post(inputs, logits, next_ids, extra)
            out_ids = out_ids.view(-1, self.block)
            after = extra.dllm_mask.view(-1, self.block)
            bases = (inputs.history_lengths + inputs.seq_length - self.block).tolist()
            for row, meta in enumerate(inputs.model_metas):
                track = self.tracks[meta['ga_session']]
                reached_source = False
                if track.source_pending:
                    replay = track.job['snapshot']['replay']
                    reached_source = len(track.events) == len(replay)
                    action = track.job['action'] if reached_source else replay[len(track.events)]
                    flags = before[row] if reached_source else before.new_tensor(action['mask_after'])
                    after[row].copy_(flags)
                    for position, token in zip(action['positions'], action['tokens']):
                        out_ids[row, position] = token
                        if reached_source:
                            after[row, position] = UNMASKED
                self._record(track, bases[row], input_ids[row].tolist(), before[row].tolist(),
                             after[row].tolist(), proposals[row].tolist(), out_ids[row].tolist(), inputs.is_decoding)
                if reached_source:
                    track.source_reached = True
            extra.dllm_mask = after.flatten()
            return out_ids.flatten(), extra

        self._replace(agent.agent_strategy, 'post_sampling', post_sampling)
        original_stop = DLLMStoppingCriteria.step

        def stopping(criteria, token_ids, stop_words, inputs=None, extra_inputs=None):
            stopped, stop_pos, updated = original_stop(criteria, token_ids, stop_words, inputs, extra_inputs)
            for row, meta in enumerate(inputs.model_metas):
                track = self.tracks[meta['ga_session']]
                budget = track.job.get('future_budget')
                if budget is not None and track.future_forwards >= budget and not stopped[row]:
                    stopped[row] = True
                    track.budget_exhausted = True
            return stopped, stop_pos, updated

        self._replace(DLLMStoppingCriteria, 'step', stopping)

    def _record(self, track, base, input_ids, before, after, proposed, output, is_decoding):
        masked = [i for i, flag in enumerate(before) if flag == MASKED]
        positions = [i for i in masked if after[i] == UNMASKED]
        historical = track.previous.get(base)
        if masked:
            track.rounds[base] = track.rounds.get(base, 0) + 1
        round_id = track.rounds.get(base, 0)
        phase = 'denoise' if masked else 'cache'
        if is_decoding and phase == 'cache':
            prefix = 'restore_' if track.source_pending else ''
            track.counts[prefix + 'decode'] -= 1
            track.counts[prefix + 'cache'] += 1
        snapshot = track.job.get('snapshot')
        logical = track.forward_calls
        confidence = [p[0] for p in track.raw['topk_probs']] if track.raw else [0.0] * self.block
        mutable = sorted(positions, key=lambda i: (-confidence[i], i))[:4]
        full = base >= len(track.prompt_ids)
        event = {'request_id': track.job['id'], 'session_id': track.session_id, 'forward': logical,
                 'physical_forward': track.forward_calls, 'block_start': base, 'round': round_id,
                 'restoring': track.source_pending,
                 'phase': phase, 'compute_kind': 'decode' if is_decoding else 'prefill',
                 'full_generation_block': full, 'input_ids': input_ids, 'mask_before': before,
                 'mask_after': after, 'positions': positions, 'tokens': [output[i] for i in positions],
                 'd_selected': sum(historical[i] != proposed[i] for i in positions) if historical is not None and positions else None,
                 'd_mutable': sum(historical[i] != proposed[i] for i in mutable) if historical is not None and positions else None}
        if self.config['trace']:
            event.update(track.raw)
            event['current_ids'] = proposed
            event['historical_ids'] = historical
        track.events.append(event)
        if self.writer:
            self.writer.write(json.dumps(event, separators=(',', ':')) + '\n')
        track.token_map.update((base + i, token) for i, token in enumerate(output) if after[i] != MASKED)
        if MASKED not in after:
            track.valid_end = max(track.valid_end, base + self.block)
        if (self.config['trace'] and full and historical is not None and positions
                and round_id == track.job.get('intervention_round') and not snapshot):
            track.pending[base] = {
                'id': f"{track.job['id']}@{base}:{round_id}", 'question_id': track.job['id'],
                'selection_seed': self.config['seed'],
                'block_start': base, 'prompt_length': len(track.prompt_ids), 'round': round_id,
                'commit_step': sum(bool(e['positions']) for e in track.events), 'source_forward': logical,
                'prefix_ids': [track.token_map[i] for i in range(base)], 'prompt_ids': track.prompt_ids,
                'replay': [{key: event[key] for key in ('block_start', 'round', 'input_ids', 'mask_before',
                           'mask_after', 'positions', 'tokens')} for event in track.events[:-1]],
                'input_ids': input_ids, 'mask': before, 'masked_positions': masked, 'positions': positions,
                'current_ids': proposed, 'historical_ids': historical, 'confidence': confidence,
                'topk_ids': track.raw['topk_ids'], 'topk_probs': track.raw['topk_probs']}
        # Only completed, nonterminal blocks enter the reservoir. EOS blocks never reach cache phase.
        if phase == 'cache' and base in track.pending:
            candidate = track.pending.pop(base)
            if not any(token in self.eos_ids for token in output):
                track.seen_states += 1
                if track.rng.randrange(track.seen_states) == 0:
                    track.state = candidate
        if masked:
            track.previous[base] = proposed

    async def _one(self, job):
        job = dict(job, state_seed=self.config['seed'])
        sid = self.session
        self.session += 1
        snapshot = job.get('snapshot')
        prompt = snapshot['prompt_ids'] if snapshot else job.get('prompt_ids', self.prompt(job.get('question', '')))
        track = Track(job, prompt, sid)
        self.tracks[sid] = track
        inputs = prompt
        limit = job.get('max_new_tokens', self.config['max_new_tokens'])
        generation = GenerationConfig(max_new_tokens=limit, do_sample=False,
            top_k=1, top_p=1.0, temperature=1.0, random_seed=self.config['seed'],
            stop_token_ids=sorted(self.eos_ids), ignore_eos=job.get('force_length', False))
        instance = self.core.create_instance()
        output = await instance.async_infer(sid, input_ids=inputs, gen_config=generation)
        if output.status != ResponseType.FINISH:
            raise RuntimeError(f"Inference failed for {job['id']}: {output.status}")
        await instance.async_end(sid)
        tokens = [track.token_map[i] for i in range(len(prompt), track.valid_end)][:limit]
        stop = next((i for i, token in enumerate(tokens) if token in self.eos_ids), None)
        finished = stop is not None and not job.get('force_length', False)
        if finished:
            tokens = tokens[:stop + 1]
        reason = 'eos' if finished else 'forward_budget' if track.budget_exhausted else 'length'
        result = {'request_id': job['id'], 'token_ids': tokens, 'text': self.tokenizer.decode(tokens, skip_special_tokens=True),
                  'finish_reason': reason, 'finished': finished, 'forward_calls': track.forward_calls,
                  'future_forwards': track.future_forwards, 'restore_forwards': track.restore_forwards,
                  'forward_types': dict(track.counts),
                  'events': [event for event in track.events if not snapshot or event['forward'] >= snapshot['source_forward']],
                  'state': track.state,
                  'eligible_blocks': track.seen_states}
        del self.tracks[sid]
        return result

    def generate(self, jobs, trace_path=None):
        async def batches():
            output = []
            for start in range(0, len(jobs), self.config['batch_size']):
                output.extend(await asyncio.gather(*(asyncio.wait_for(self._one(job), timeout=300)
                              for job in jobs[start:start + self.config['batch_size']])))
            return output

        started = time.monotonic()
        context = gzip.open(trace_path, 'wt', compresslevel=1) if trace_path else ExitStack()
        with context as writer:
            self.writer = writer if trace_path else None
            try:
                results = self.loop.run_until_complete(batches())
            finally:
                self.writer = None
        self.last_seconds = time.monotonic() - started
        return results

    @torch.inference_mode()
    def compare_native(self, job, result):
        """One small block-attention forward through the checkpoint's own implementation."""
        from transformers import AutoModelForCausalLM
        event = next(e for e in result['events'] if e['positions'])
        prompt = self.prompt(job['question'])
        ids = prompt[:event['block_start']] + event['input_ids']
        tokens = torch.tensor([ids], device='cuda')
        positions = torch.arange(len(ids), device='cuda')
        mask = (positions[:, None] // self.block >= positions[None, :] // self.block)[None, None]
        model = AutoModelForCausalLM.from_pretrained(self.config['model'], trust_remote_code=True,
            local_files_only=True, torch_dtype=torch.bfloat16).cuda().eval()
        logits = model(tokens, attention_mask=mask, use_cache=False).logits[0, -self.block:].float()
        logp = logits.log_softmax(-1)
        selected = event['positions']
        top1 = logits.argmax(-1).tolist()
        errors = [abs(logp[i, event['topk_ids'][i][0]].item() - math.log(event['topk_probs'][i][0]))
                  for i, flag in enumerate(event['mask_before']) if flag == MASKED]
        report = {'committed_top1_matches': all(top1[i] == event['current_ids'][i] for i in selected),
                  'masked_top1_matches': sum(top1[i] == event['current_ids'][i] for i, flag in enumerate(event['mask_before']) if flag == MASKED),
                  'masked_positions': len(errors), 'max_top1_logp_error': max(errors)}
        del model
        torch.cuda.empty_cache()
        report['passed'] = report['committed_top1_matches'] and report['max_top1_logp_error'] < 0.2
        return report

    def close(self):
        async def shutdown():
            self.core.close()
            pending = [task for task in asyncio.all_tasks() if task is not asyncio.current_task()]
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
        self.loop.run_until_complete(shutdown())
        self.stack.close()
        self.loop.close()
