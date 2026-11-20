import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from algorithm import Response
from analyze_reconstruction import read_rollouts, score_response, summarize as reconstruction_summary
from cross_decode import GA, first_divergent_block
from eval_decoding import counted_result, decoding_settings, summarize as decoding_summary
from likelihood import SDARPolicy, phase_inputs, reconstruction_phases
from test_algorithm import dense_packed
from vendor.sdar import modeling_sdar as sdar
from vendor.sdar.configuration_sdar import SDARConfig


class DistributionTests(unittest.TestCase):
    def test_shared_parameter_gradient_includes_both_distributions(self):
        theta = torch.tensor([.3, -.4, .8, -.5, -.3, .9], dtype=torch.float64, requires_grad=True)
        first = theta[:2].log_softmax(0)
        conditional = theta[2:].reshape(2, 2).log_softmax(1)
        parallel = theta[2:].reshape(2, 2).mean(0).log_softmax(0)
        ls = torch.stack([first[i] + conditional[i, j] for i in range(2) for j in range(2)])
        lp = torch.stack([first[i] + parallel[j] for i in range(2) for j in range(2)])
        reward = torch.tensor([1., 0., 0., 1.], dtype=torch.float64)
        torch.testing.assert_close(ls.exp().sum(), theta.new_tensor(1.))
        torch.testing.assert_close(lp.exp().sum(), theta.new_tensor(1.))
        beta, d = .3, ls - lp
        objective = (ls.exp() * (reward - beta*d)).sum()
        exact, = torch.autograd.grad(objective, theta, retain_graph=True)
        surrogate = (ls.exp().detach() * ((reward - beta*d.detach())*ls + beta*lp)).sum()
        derived, = torch.autograd.grad(surrogate, theta)
        torch.testing.assert_close(derived, exact, atol=1e-14, rtol=1e-12)

    def test_better_coupled_objective_can_reduce_accuracy(self):
        def values(p):
            s = torch.tensor([p, 1-p, 1-p, p], dtype=torch.float64) / 2
            parallel = torch.full_like(s, .25)
            kl = (s*(s.log() - parallel.log())).sum().item()
            return p - .3*kl, p, .5
        before, after = values(.99), values(.98)
        self.assertGreater(after[0], before[0])
        self.assertLess(after[1], before[1])
        self.assertEqual(after[2], before[2])
        u = torch.tensor(math.log(99), dtype=torch.float64, requires_grad=True)
        p = u.sigmoid()
        kl = p*(2*p).log() + (1-p)*(2*(1-p)).log()
        gradient, = torch.autograd.grad(p - .3*kl, u)
        self.assertLess(gradient.item(), 0)

    def test_independent_unused_proposal_has_zero_reward_gradient(self):
        logits = torch.tensor([.7, -.2], dtype=torch.float64, requires_grad=True)
        action = torch.tensor([.3, .7], dtype=torch.float64)
        joint = action[:, None] * logits.softmax(0)[None, :]
        reward = torch.tensor([0., 1.], dtype=torch.float64)[:, None]
        gradient, = torch.autograd.grad((joint*reward).sum(), logits)
        torch.testing.assert_close(gradient, torch.zeros_like(logits), atol=1e-14, rtol=0)

    def test_eos_tail_is_marginalized_in_normalized_output_probability(self):
        # Vocabulary 0, 1, EOS=2, maximum length 2. An EOS at the first
        # position has probability q1[EOS], irrespective of the sampled tail.
        first = torch.tensor([.4, .35, .25], dtype=torch.float64)
        second = torch.tensor([.1, .2, .7], dtype=torch.float64)
        outputs = {(2,): first[2]}
        outputs.update({(i, j): first[i]*second[j] for i in (0, 1) for j in (0, 1, 2)})
        torch.testing.assert_close(sum(outputs.values()), first.new_tensor(1.))
        torch.testing.assert_close(outputs[(2,)], (first[2]*second).sum())
        wrong_total = first[2]*second[2] + sum(v for k, v in outputs.items() if len(k) == 2)
        self.assertLess(wrong_total, 1)


class ReconstructionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        kernels = dict(
            flash_rms_norm=lambda x, weight, bias, eps: x*torch.rsqrt(x.square().mean(-1, keepdim=True)+eps)*weight,
            liger_kernel_is_available=False,
            fused_flex_attention=lambda query, key, value, attention_mask, scale, **kw: (
                F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask[:, None],
                                              scale=scale, enable_gqa=True), None))
        for context in (patch.multiple(sdar, **kernels), patch('likelihood.packed_attention', dense_packed)):
            context.start()
            self.addCleanup(context.stop)
        torch.manual_seed(8)
        self.model = SDARPolicy(SDARConfig(vocab_size=32, hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1, head_dim=16,
            mask_token_id=31, block_size=4, use_cache=False, attention_dropout=0.)).double().train()

    def serial(self, response, full_mask=False):
        p, t, block = len(response.prompt), len(response.tokens), self.model.config.block_size
        n = ((p + t + 127) // 128)*128
        clean = torch.full((n,), 31, dtype=torch.long)
        clean[:p+t] = torch.tensor(response.prompt + response.tokens)
        index = torch.arange(n)
        attention = (index[:, None] // block >= index[None, :] // block)[None]
        result = []
        for j in range(p, p+t):
            start = max(p, j // block * block) if full_mask else j
            ids = clean.clone()
            ids[start:] = 31
            hidden = self.model.model(input_ids=ids[None], position_ids=index[None],
                                      attention_mask=attention, use_cache=False).last_hidden_state[0, j]
            result.append(F.linear(hidden, self.model.lm_head.weight).float().log_softmax(-1)[clean[j]])
        return torch.stack(result)

    def packed(self, response, size=4, chunk=2):
        phases = list(reconstruction_phases(self.model, response, phase_batch_size=size, logit_chunk_size=chunk))
        offsets = torch.cat([p.offsets for p in phases])
        seq = torch.cat([p.logps for p in phases])[offsets.argsort()]
        full_offsets = torch.cat([p.full_mask_offsets for p in phases])
        full = torch.cat([p.full_mask_logps for p in phases])[full_offsets.argsort()]
        self.assertEqual(sorted(offsets.tolist()), list(range(len(response.tokens))))
        self.assertEqual(full_offsets.tolist(), list(range(len(response.tokens))))
        return seq, full

    def gradient(self):
        return torch.cat([p.grad.flatten() for p in self.model.parameters() if p.grad is not None])

    def test_full_mask_and_sequential_match_independent_serial_with_eos_and_prompt_residues(self):
        for p, tokens in [(1, [2]), (3, [6, 7, 8, 2]), (4, [6, 7, 2]), (5, [6, 7, 8, 9, 10, 11, 2])]:
            response = Response([1]*p, tokens, 1.)
            expected = self.serial(response), self.serial(response, full_mask=True)
            for size in (1, 2, 4):
                actual = self.packed(response, size)
                for a, e in zip(actual, expected):
                    torch.testing.assert_close(a, e, atol=1e-6, rtol=1e-6)

    def test_shared_backbone_gradient_matches_serial_and_chunked_projection(self):
        response = Response([1]*5, [6, 7, 8, 9, 10, 11, 2], 1.)
        reference = -self.serial(response).mean() - .3*self.serial(response, full_mask=True).mean()
        reference.backward()
        expected = self.gradient().clone()
        for size, chunk in [(1, 2), (4, 2), (4, None)]:
            self.model.zero_grad(set_to_none=True)
            seq, full = self.packed(response, size, chunk)
            (-seq.mean() - .3*full.mean()).backward()
            torch.testing.assert_close(self.gradient(), expected, atol=2e-7, rtol=2e-5)

    def test_extra_targets_do_not_add_backbone_forwards_or_duplicate_first_tokens(self):
        response = Response([1]*5, [6, 7, 8, 9, 10, 11, 2], 1.)
        p, t = len(response.prompt), len(response.tokens)
        blocks = (p+t-1)//4 - p//4 + 1
        original = list(phase_inputs(response, 4, 31, 'cpu', 4))
        with patch.object(self.model.model, 'forward', wraps=self.model.model.forward) as backbone, \
             patch.object(self.model, 'token_logps', wraps=self.model.token_logps) as projection:
            self.packed(response, chunk=2)
        self.assertEqual(backbone.call_count, len(original))
        self.assertEqual(sum(call.args[1].numel() for call in projection.call_args_list), 2*t - blocks)
        self.assertEqual(sum(len(item[2]) for item in original), t)

    def test_offline_score_preserves_first_eos_and_excludes_padding(self):
        response = Response([1]*3, [6, 7, 8, 2], 1.)
        record = score_response(self.model, response, logit_chunk_size=2)
        self.assertEqual(record['tokens'], 4)
        self.assertEqual(record['projected_tokens'], 6)
        self.assertEqual(len(record['token_logps']['full_mask']), 4)
        self.assertAlmostEqual(record['logp_gap'],
            (self.serial(response) - self.serial(response, full_mask=True)).double().sum().item(), places=5)


class AnalysisAndEvaluationTests(unittest.TestCase):
    def test_saved_rollouts_filter_by_round_without_changing_advantages(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'rollouts.jsonl'
            base = dict(prompt_ids=[1, 1, 1], responses=[[6, 2], [7, 2]], rewards=[1., 0.],
                        finished=[True, True], advantages=[.7, -.7])
            path.write_text('\n'.join(json.dumps(dict(step=step, **base)) for step in (1, 2)))
            result = list(read_rollouts([path], 2))
        self.assertEqual(len(result), 2)
        self.assertEqual(result[1][0]['line'], 2)
        self.assertEqual(result[1][1].advantage, -.7)
        self.assertEqual(result[0][1].tokens, [6, 2])

    def test_analysis_uses_response_variance_and_token_weighted_gap(self):
        records = [dict(tokens=2, reward=1, logp_gap=2., gap_per_token=1.),
                   dict(tokens=6, reward=0, logp_gap=-3., gap_per_token=-.5)]
        result = reconstruction_summary(records)
        self.assertEqual(result['all']['token_weighted_gap'], -1/8)
        self.assertEqual(result['all']['logp_gap_variance'], 12.5)
        self.assertEqual(result['all']['negative_gap_fraction'], .5)
        self.assertAlmostEqual(result['gap_per_token_reward_correlation'], 1.)

    def test_settings_distinguish_fixed_grouping_from_dynamic_selection(self):
        settings = decoding_settings(32, [1, 2, 4, 4], [.9, .9])
        self.assertEqual([s['name'] for s in settings], ['sequential', 'fixed-2', 'fixed-4', 'dynamic-0.9'])
        self.assertEqual(settings[2]['steps'], 8)
        self.assertEqual(settings[2]['strategy'], 'sequential')
        self.assertEqual(settings[3]['strategy'], 'low_confidence_dynamic')
        with self.assertRaises(ValueError):
            decoding_settings(32, [3], [])

    def test_first_divergent_block_includes_prompt_offset(self):
        self.assertIsNone(first_divergent_block([4, 5, 6], [4, 5, 6], 30, 32))
        self.assertEqual(first_divergent_block([4, 5, 6], [4, 0, 6], 30, 32), 0)
        self.assertEqual(first_divergent_block([4, 5, 6], [4, 5, 0], 30, 32), 32)

    def test_block_override_restores_the_receiver_after_each_forward(self):
        with patch.object(sys, 'path', [str(GA), *sys.path]):
            from engine import Backend

        sequential, dynamic = decoding_settings(32, [], [0.9])
        backend = Backend.__new__(Backend)
        backend.setting = sequential
        override = (64, dynamic)
        with patch.object(backend, 'set_setting', side_effect=lambda s: setattr(backend, 'setting', s)):
            for start, expected in [(32, sequential), (64, dynamic), (64, dynamic), (96, sequential)]:
                with backend._block_setting(override, start):
                    self.assertEqual(backend.setting, expected)
                self.assertEqual(backend.setting, sequential)

    def test_forward_audit_keeps_eos_tail_compute_but_counts_only_returned_tokens(self):
        events = [dict(block_start=0, positions=[1, 2, 3], tokens=[6, 2, 9], phase='denoise'),
                  dict(block_start=0, positions=[], tokens=[], phase='cache')]
        raw = dict(token_ids=[6, 2], forward_calls=2, forward_types=dict(denoise=1, cache=1),
                   events=events, state=None, eligible_blocks=0)
        record, _ = counted_result(raw, 1)
        self.assertEqual(record['denoise_forwards'], 1)
        with self.assertRaises(ValueError):
            counted_result(raw | dict(forward_calls=3), 1)

    def test_tpf_is_pooled_and_native_timing_never_invents_forward_counts(self):
        records = [dict(token_ids=[1]*2, reward=1., finished=True, grading='ok',
                        forward_calls=5, denoise_forwards=4),
                   dict(token_ids=[1]*6, reward=0., finished=False, grading='truncated',
                        forward_calls=3, denoise_forwards=2)]
        metrics = decoding_summary(records, 2.)
        self.assertEqual(metrics['tpf_denoise'], 8/6)
        self.assertEqual(metrics['tpf_all'], 1.)
        self.assertEqual(metrics['tokens_per_second'], 4.)
        timing = [r | dict(forward_calls=None, denoise_forwards=None) for r in records]
        metrics = decoding_summary(timing, 2.)
        self.assertIsNone(metrics['tpf_denoise'])
        self.assertEqual(metrics['timing_scope'], 'native rollout generation')


if __name__ == '__main__':
    unittest.main()
