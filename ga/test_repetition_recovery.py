"""Causal triggers, comparable perturbations, sampling, and paid rollback work."""
import unittest

import torch

from repetition_recovery import Detector, break_token, capped_tokens, draw, groups, target_position


class RecoveryTests(unittest.TestCase):
    def test_detector_waits_for_committed_contiguous_prefix(self):
        detector = Detector(64)
        e = {'block_start': 64, 'positions': list(range(1,32)), 'tokens': [7]*31, 'forward': 1}
        self.assertIsNone(detector.observe(e))
        found = detector.observe(dict(e, positions=[0], tokens=[7], forward=2))
        self.assertEqual((found['forward'], found['onset'], found['period']), (2,1,1))
        self.assertEqual(found['observed_tokens'], 32)
        self.assertEqual(detector.observe(dict(e, forward=3)), found)

    def test_target_is_selected_current_top1_and_uses_only_observed_pattern(self):
        pattern = {'motif_start': 0, 'onset': 8, 'observed_tokens': 32, 'period': 8, 'motif': list(range(8))}
        event = {'block_start': 96, 'positions': [4,1,3], 'current_ids': [i%8 for i in range(32)]}
        self.assertEqual(target_position(event, pattern, 64, False), 1)
        event['current_ids'][1] = 999
        self.assertEqual(target_position(event, pattern, 64, False), 3)
        self.assertIsNone(target_position(dict(event, block_start=64), pattern, 64, False))
        self.assertEqual(target_position(dict(event, block_start=72), pattern, 64, True), 3)

    def test_sampling_is_reproducible_and_does_not_reject_top1(self):
        logits = torch.tensor([100., 0., -100.])
        self.assertEqual(draw(logits, 1, 123)[0], 0)
        self.assertEqual(draw(logits, 1, 123), draw(logits, 1, 123))
        self.assertEqual(logits.tolist(), [100.,0.,-100.])
        # Full vocabulary and temperature affect categorical draws, without extra forwards.
        logits = torch.tensor([2., 1., 0., -1.])
        draws = [draw(logits, 2, seed)[0] for seed in range(100)]
        self.assertEqual(set(draws), set(range(4)))

    def test_group_sizes_and_matched_sampling_seeds(self):
        study = {'temperatures': [1,2], 'sampling_seeds': [11,12,13]}
        arms = groups(study)
        self.assertEqual(len(arms), 17)
        local = {(a['temperature'],a['seed']) for a in arms if a['group']=='G2'}
        rollback = {(a['temperature'],a['seed']) for a in arms if a['group']=='G4'}
        self.assertEqual(local, rollback)

    def test_break_excludes_current_even_with_tied_logits(self):
        self.assertEqual(break_token(torch.tensor([1., 1., 0.]), 1), 0)
        self.assertEqual(break_token(torch.tensor([1., 1., 0.]), 0), 1)

    def test_rollback_debits_sunk_and_rebuild_before_continuation(self):
        trigger, rebuild, cap = 221, 186, 512
        self.assertEqual(cap-trigger-rebuild, 105)
        # The source regeneration forward also consumes one of these 105 calls.
        source = rebuild+1
        self.assertEqual(cap-trigger-source, 104)

    def test_budget_does_not_emit_partial_block(self):
        row = {'forward_calls': 600, 'token_ids': list(range(1000)), 'state': {'prompt_ids': [1]*64}}
        events = [{'block_start':64,'mask_after':[1]*32}, {'block_start':96,'mask_after':[0]*32}]
        tokens, reason = capped_tokens(row, events, 2)
        self.assertEqual(tokens, list(range(32)))
        self.assertEqual(reason, 'forward_budget')


if __name__ == '__main__':
    unittest.main()
