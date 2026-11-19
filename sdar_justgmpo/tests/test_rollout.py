"""Token boundaries and batch-position sampling metadata must match the policy."""
import asyncio
from contextlib import nullcontext
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
import torch
from transformers import GenerationConfig as HFGenerationConfig
from rollout import Rollout, canonical_tokens, per_token_sampling_inputs, ARSamplingStrategy, raw_token_stream
from inference_files import LEGACY_GENERATION_EXPORT
from lmdeploy.pytorch.engine.model_agent import BaseModelAgent
from lmdeploy.pytorch.strategies.base.model_inputs import make_dummy_inputs
from lmdeploy.pytorch.strategies.ar.model_agent import ARModelAgentStrategy
from lmdeploy.pytorch.strategies.dllm.model_agent import DLLMExtraInputs, DLLMModelAgentStrategy
from lmdeploy.pytorch.consts import DLLM_MASKED, DLLM_UNMASKED


class RolloutTests(unittest.TestCase):
    def test_runtime_options_reach_engine_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            HFGenerationConfig(eos_token_id=9).save_pretrained(directory)
            with patch('rollout.pipeline') as pipeline, patch(
                    'rollout.inference_model_path', return_value=nullcontext(directory)):
                engine = Rollout(directory, Mock(), max_tokens=3000, batch_size=32, tp_size=2,
                    cache_fraction=0.25, cuda_graph=False, prefix_caching=True, max_prefill_tokens=8192)
                config = pipeline.call_args.kwargs['backend_config']
                self.assertEqual(config.max_batch_size, 32)
                self.assertEqual(config.cache_max_entry_count, 0.25)
                self.assertTrue(config.eager_mode)
                self.assertTrue(config.enable_prefix_caching)
                self.assertEqual(config.max_prefill_token_num, 8192)
                engine.close()

    def test_training_samples_and_evaluation_is_greedy(self):
        engine = object.__new__(Rollout)
        engine.max_tokens, engine.eos_ids = 4, {9}
        engine.pipe = Mock(side_effect=lambda texts, **kw:
            [SimpleNamespace(token_ids=[1, 9, 7], finish_reason='stop') for _ in texts])
        self.assertEqual(engine.sample(['a', 'b'], 2, 31), [([1, 9], True)]*4)
        configs = engine.pipe.call_args.kwargs['gen_config']
        self.assertTrue(all(c.do_sample for c in configs))
        self.assertEqual([c.random_seed for c in configs], [31, 32, 33, 34])
        self.assertEqual(engine.sample(['a'], 1, 113, do_sample=False), [([1, 9], True)])
        self.assertFalse(engine.pipe.call_args.kwargs['gen_config'][0].do_sample)

    def test_streaming_keeps_prompt_order_and_reports_only_received_tokens(self):
        engine = object.__new__(Rollout)
        engine.max_tokens, engine.eos_ids = 4, {9}
        chunks = [(1, [3], None), (0, [1, 2], None), (1, [9], None),
                  (1, [], 'stop'), (0, [9, 7], None), (0, [], 'stop')]
        engine.pipe = Mock()
        engine.pipe.stream_infer.return_value = (
            SimpleNamespace(index=i, token_ids=ids, finish_reason=reason) for i, ids, reason in chunks)
        progress = Mock()
        self.assertEqual(engine.sample(['a', 'b'], 1, 31, progress=progress),
                         [([1, 2, 9], True), ([3, 9], True)])
        self.assertEqual(sum(call.args[0] for call in progress.call_args_list), 6)
        engine.pipe.assert_not_called()

    def test_streaming_rejects_incomplete_reward_groups(self):
        engine = object.__new__(Rollout)
        engine.max_tokens, engine.eos_ids = 4, {9}
        engine.pipe = Mock()
        engine.pipe.stream_infer.return_value = [
            SimpleNamespace(index=1, token_ids=[9], finish_reason='stop')]
        with self.assertRaisesRegex(AssertionError, 'Rollout failed'):
            engine.sample(['a', 'b'], 1, 31, progress=Mock())

    def test_engine_and_token_boundaries_both_use_recovered_generation_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent, model = root/'final', root/'checkpoint-75'
            parent.mkdir(); model.mkdir()
            HFGenerationConfig(eos_token_id=[151645, 151643]).save_pretrained(parent)
            HFGenerationConfig(eos_token_id=151643, _from_model_config=True).save_pretrained(model)
            (model/'tokenizer_config.json').write_text('{}')
            (model/'resume.json').write_text(json.dumps({
                'config': {'resume': str(parent)},
                'fingerprint': {'source': LEGACY_GENERATION_EXPORT}}))
            pipe = Mock(return_value=[SimpleNamespace(
                token_ids=[42, 151645, 43], finish_reason='stop')])

            def pipeline(path, **kwargs):
                cfg = HFGenerationConfig.from_pretrained(path, local_files_only=True)
                self.assertEqual(cfg.eos_token_id, [151645, 151643])
                return pipe

            with patch('rollout.pipeline', side_effect=pipeline):
                engine = Rollout(model, Mock(), max_tokens=4, batch_size=1)
                try:
                    self.assertEqual(engine.sample(['prompt'], 1, 1234), [([42, 151645], True)])
                finally:
                    engine.close()
            self.assertEqual(HFGenerationConfig.from_pretrained(model).eos_token_id, 151643)

    def test_scalar_and_list_eos_exports_keep_the_first_stop_token(self):
        for eos_ids, stop_token in ((151643, 151643), ([151645, 151643], 151645),
                                    ([151645, 151643], 151643)):
            with self.subTest(eos_ids=eos_ids, stop_token=stop_token), tempfile.TemporaryDirectory() as directory:
                HFGenerationConfig(eos_token_id=eos_ids).save_pretrained(directory)
                pipe = Mock(return_value=[SimpleNamespace(
                    token_ids=[42, stop_token, 43], finish_reason='stop')])
                with patch('rollout.pipeline', return_value=pipe), patch(
                        'rollout.inference_model_path', return_value=nullcontext(directory)):
                    engine = Rollout(directory, Mock(), max_tokens=4, batch_size=1)
                    try:
                        self.assertEqual(engine.sample(['prompt'], 1, 1234),
                                         [([42, stop_token], True)])
                    finally:
                        engine.close()

    def test_chunked_prefill_matches_unsplit_sampling_positions(self):
        # Run the actual engine orchestration on CPU with a deterministic fake
        # backbone. DLLM pads the prefill to complete blocks; the final chunk
        # must preserve every block position for both b4 and b32.
        for block_size in (1, 4, 32):
            for length in (4096+block_size, 8192, 8192+block_size):
                with self.subTest(block_size=block_size, length=length):
                    inputs = make_dummy_inputs(1, length, False)
                    inputs.input_ids = torch.arange(length)[None]
                    agent = object.__new__(BaseModelAgent)
                    if block_size == 1:
                        agent.agent_strategy = ARModelAgentStrategy()
                    else:
                        agent.agent_strategy = object.__new__(DLLMModelAgentStrategy)
                        agent.agent_strategy.block_size = block_size

                    async def forward(chunk):
                        hidden = chunk.input_ids.float()[..., None]
                        return {'hidden_states': torch.cat([hidden, -hidden], -1)}

                    agent.async_forward = forward
                    agent.get_logits = lambda hidden: hidden.flatten(0, 1)
                    agent.cache_config = SimpleNamespace(max_prefill_token_num=length)
                    whole = asyncio.run(agent._async_model_forward(inputs, False, False))['logits']
                    agent.cache_config.max_prefill_token_num = 4096
                    split = asyncio.run(agent._async_model_forward(inputs, False, False))['logits']
                    self.assertEqual(tuple(split.shape), (block_size, 2))
                    torch.testing.assert_close(split, whole, rtol=0, atol=0)

    def test_raw_stream_preserves_unknown_vocabulary_rows(self):
        state=SimpleNamespace(ids_offset=2)
        tokens=[1,2,151900,151645,3]
        previous=state.ids_offset
        _,state=raw_token_stream(tokens,state,skip_special_tokens=False)
        self.assertEqual(tokens[previous:],[151900,151645,3])
        self.assertEqual(state.ids_offset,len(tokens))

    def test_eos_action_is_kept_and_suffix_is_excluded(self):
        eos = {151643,151645}
        self.assertEqual(canonical_tokens([1,2,151645,3],4,eos),([1,2,151645],True))
        self.assertEqual(canonical_tokens([151643,3],4,eos),([151643],True))
        self.assertEqual(canonical_tokens([1,2,3,151645],3,eos),([1,2,3],False))
        self.assertEqual(canonical_tokens([1,2,3,4],4,eos),([1,2,3,4],False))

    def test_sequence_major_sampling_metadata(self):
        fields=('temperature bad_words bad_mask stop_words stop_mask repetition_penalty '
                'top_k top_p min_p random_seeds random_offsets all_ids guided_input_ids num_ignore_eos').split()
        values={k:None for k in fields}
        values.update(random_seeds=torch.tensor([11,22,33]),
                      stop_words=torch.tensor([[1,2],[3,4],[5,6]]),
                      response_formats=('a','b','c'),batch_size=3)
        for block_size in (4,32):
            with self.subTest(block_size=block_size):
                obj=SimpleNamespace(**values)
                with patch.object(ARSamplingStrategy,'make_sampling_inputs',return_value=obj):
                    out=per_token_sampling_inputs(SimpleNamespace(dllm_block_length=block_size),[])
                self.assertEqual(out.random_seeds.tolist(),[11]*block_size+[22]*block_size+[33]*block_size)
                self.assertEqual(out.response_formats,('a',)*block_size+('b',)*block_size+('c',)*block_size)
                self.assertEqual(out.stop_words.tolist(),[[1,2]]*block_size+[[3,4]]*block_size+[[5,6]]*block_size)
                self.assertEqual(out.batch_size,3*block_size)

    def test_random_draws_are_disjoint_across_positions_phases_and_blocks(self):
        block = 4
        for unmask_order in ((0, 1, 2, 3), (3, 0, 2, 1)):
            seen = set()
            for start in (8, 12):
                mask = torch.full((block,), DLLM_MASKED)
                inputs = SimpleNamespace(history_lengths=torch.tensor([start]),
                                         seq_length=torch.tensor([block]))
                for position in unmask_order:
                    offsets = DLLMExtraInputs(mask).sampling_offsets(inputs).tolist()
                    self.assertEqual(len(set(offsets)), block)
                    self.assertTrue(seen.isdisjoint(offsets))
                    seen.update(offsets)
                    mask[position] = DLLM_UNMASKED

    def test_random_draws_follow_logical_state_across_prefill_and_rebatching(self):
        strategy = object.__new__(DLLMModelAgentStrategy)
        strategy.block_size = 4
        masked, filled = DLLM_MASKED, DLLM_UNMASKED
        short = torch.tensor([filled, filled, masked, masked])
        long = torch.tensor([filled]*4 + [filled, masked, masked, masked])
        prefill = SimpleNamespace(history_lengths=torch.tensor([0, 0]), seq_length=torch.tensor([4, 8]))
        packed = DLLMExtraInputs(torch.cat((short, long)))
        offsets = strategy.slice_extra_inputs(packed, prefill.seq_length).sampling_offsets(prefill)
        decode = SimpleNamespace(history_lengths=torch.tensor([4, 0]), seq_length=torch.tensor([4, 4]))
        reordered = DLLMExtraInputs(torch.cat((long[-4:], short))).sampling_offsets(decode)
        torch.testing.assert_close(offsets.reshape(2, 4).flip(0), reordered.reshape(2, 4))


if __name__=='__main__': unittest.main()
