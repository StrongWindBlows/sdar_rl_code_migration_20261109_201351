"""Reuse SDAR JustGRPO sampling adapters; stop normally and retain the EOS action."""
import os
import sys
from contextlib import ExitStack
from pathlib import Path

import torch
from inference_files import inference_model_path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'DiRL'))
from lmdeploy import pipeline, PytorchEngineConfig, GenerationConfig, ChatTemplateConfig
from lmdeploy.pytorch.strategies.ar.sampling import ARSamplingStrategy
from lmdeploy.pytorch.strategies.dllm.sampling import DLLMSamplingStrategy


def per_token_sampling_inputs(self, seqs):
    """LMDeploy logits are [sequence, block_position, vocab], flattened.

    Local DiRL repeats [block_position, sequence], which misaligns per-request
    seeds/stop masks. Expand each sequence consecutively to match logits.
    """
    out = ARSamplingStrategy.make_sampling_inputs(self, seqs)
    for name in ('temperature', 'bad_words', 'bad_mask', 'stop_words', 'stop_mask',
                 'repetition_penalty', 'top_k', 'top_p', 'min_p', 'random_seeds',
                 'random_offsets', 'all_ids', 'guided_input_ids', 'num_ignore_eos'):
        value = getattr(out, name)
        if value is not None:
            setattr(out, name, value.repeat_interleave(self.dllm_block_length, dim=0))
    out.response_formats = tuple(x for x in out.response_formats for _ in range(self.dllm_block_length))
    out.batch_size *= self.dllm_block_length
    return out


DLLMSamplingStrategy.make_sampling_inputs = per_token_sampling_inputs


def canonical_tokens(tokens, max_tokens, eos_ids):
    """Keep the first EOS action; exclude any trailing engine block positions."""
    tokens = tokens[:max_tokens]
    stop = next((i for i, token in enumerate(tokens) if token in eos_ids), None)
    return (tokens[:stop+1], True) if stop is not None else (tokens, False)


def raw_token_stream(all_input_ids, state, **kwargs):
    """The trainer needs token IDs, not LMDeploy's incremental text rendering.

    Full-vocab sampling may emit model padding rows unknown to the tokenizer.
    HF batch_decode handles them; LMDeploy's incremental string join raises.
    Preserve every sampled ID and let the trainer decode the complete response.
    """
    state.ids_offset = len(all_input_ids)
    return '', state


class Rollout:
    def __init__(self, model_path, tokenizer, max_tokens, batch_size, block_size=4, tp_size=1,
                 *, cache_fraction=0.8, cuda_graph=True, prefix_caching=False,
                 max_prefill_tokens=4096):
        os.environ['SDAR_FULL_VOCAB'] = '1'
        self.tp_size = tp_size
        self.tokenizer, self.max_tokens = tokenizer, max_tokens
        from transformers import GenerationConfig as HFGenerationConfig
        cfg = PytorchEngineConfig(tp=tp_size, dtype='bfloat16', session_len=max_tokens+4096,
            max_batch_size=batch_size, cache_max_entry_count=cache_fraction,
            max_prefill_token_num=max_prefill_tokens, eager_mode=not cuda_graph,
            distributed_executor_backend='uni' if tp_size == 1 else 'mp',
            enable_prefix_caching=prefix_caching, dllm_block_length=block_size,
            dllm_denoising_steps=block_size, dllm_unmasking_strategy='sequential')
        self._files = ExitStack()
        try:
            engine_path = self._files.enter_context(inference_model_path(model_path))
            eos_ids = HFGenerationConfig.from_pretrained(engine_path, local_files_only=True).eos_token_id
            # Use the same recovered config as LMDeploy, including all EOS IDs.
            self.eos_ids = {eos_ids} if isinstance(eos_ids, int) else set(eos_ids)
            self.pipe = pipeline(str(engine_path), backend_config=cfg,
                chat_template_config=ChatTemplateConfig(model_name='base'), log_level='ERROR')
        except BaseException:
            self._files.close()
            raise
        if tp_size == 1:
            self.engine_model = self.pipe.engine.executor.model_agent.patched_model.get_model()
            self.pipe.engine.executor.model_agent.sampling_vocab_size = None
        self.pipe.tokenizer.detokenize_incrementally = raw_token_stream

    @torch.inference_mode()
    def update_weights(self, policy):
        if self.tp_size != 1:
            raise RuntimeError('TP rollout loads a fresh complete export through ParallelRollout')
        # No file reload and no stale KV reuse: each request starts a new session.
        self.engine_model.load_weights((name, value.detach().to(torch.bfloat16))
                                       for name, value in policy.state_dict().items())
        torch.cuda.synchronize()

    def sample(self, prompts, group_size, seed, temperature=1.0, *, do_sample=True, progress=None):
        texts = [p for p in prompts for _ in range(group_size)]
        configs = [GenerationConfig(max_new_tokens=self.max_tokens, top_k=0, top_p=1.0,
            temperature=temperature, do_sample=do_sample, ignore_eos=False,
            skip_special_tokens=False, include_stop_str_in_output=True, random_seed=seed+i) for i in range(len(texts))]
        if progress is None:
            outputs = self.pipe(texts, gen_config=configs, use_tqdm=False)
            assert len(outputs) == len(texts), 'Complete reward groups are required'
            tokens = [o.token_ids for o in outputs]
            reasons = [o.finish_reason for o in outputs]
        else:
            # Multiplex all requests so short answers immediately free batch slots.
            tokens, reasons = [[] for _ in texts], [None for _ in texts]
            for out in self.pipe.stream_infer(texts, gen_config=configs, stream_response=True):
                tokens[out.index].extend(out.token_ids)
                reasons[out.index] = out.finish_reason
                progress(len(out.token_ids))
        assert all(tokens) and all(r in ('stop', 'length') for r in reasons), 'Rollout failed'
        return [canonical_tokens(ids, self.max_tokens, self.eos_ids) for ids in tokens]

    def close(self):
        try:
            self.pipe.close()
        finally:
            self._files.close()
