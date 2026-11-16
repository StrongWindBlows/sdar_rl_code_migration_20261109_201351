"""Checks for causal selection and partial-block budget accounting."""
from copy import deepcopy
import unittest

from action_value import capped_reference, make_actions, select_source


def event(round_id, selected, masked, probability=0.95):
    before = [int(i not in masked) for i in range(32)]
    after = [1 if i in selected else before[i] for i in range(32)]
    return {'phase': 'denoise', 'full_generation_block': True, 'block_start': 64,
        'round': round_id, 'forward': round_id + 1, 'positions': selected,
        'mask_before': before, 'mask_after': after, 'current_ids': list(range(32)),
        'historical_ids': list(range(32)) if round_id > 1 else None,
        'topk_probs': [[probability, 1 - probability] for _ in range(32)]}


class ActionValueTests(unittest.TestCase):
    def test_first_eligible_trigger_ignores_future_and_final_round(self):
        first = event(2, [4, 5, 6, 7], range(4, 32))
        early = event(3, [8, 9, 10, 11], range(8, 32), 0.8)
        late = dict(event(4, [12, 13, 14, 15], range(12, 32), 0.1), block_start=320)
        self.assertIs(select_source([first, early, late], 64, 8), early)
        self.assertIs(select_source([first, early], 64, 8), early)
        self.assertIsNone(select_source([dict(early, round=8), late], 64, 8))

    def test_reopen_uses_commit_evidence_not_visible_position_logits(self):
        first = event(1, [0, 1, 2, 3], range(32))
        first['topk_probs'][2] = [0.5, 0.5]
        second = event(2, [4, 5, 6, 7], range(4, 32), 0.8)
        original = make_actions(second, [first])
        changed = deepcopy(second)
        for i in range(4):
            changed['current_ids'][i] = 999
            changed['topk_probs'][i] = [0.01, 0.99]
        self.assertEqual(make_actions(changed, [first]), original)
        self.assertEqual(original[3]['reopen'], [2])
        self.assertEqual(original[4]['reopen'], [2, 0])

    def test_swap_delay_and_reopen_change_distinct_decisions(self):
        first = event(1, [0, 1, 2, 3], range(32))
        second = event(2, [4, 5, 6, 7], range(4, 32), 0.8)
        second['historical_ids'][6] = 100
        second['topk_probs'][9] = [0.99, 0.01]
        current, swap, delay, reopen, _ = make_actions(second, [first])
        self.assertEqual(swap['positions'], [4, 5, 7, 9])
        self.assertEqual(delay['positions'], [4, 5, 7])
        self.assertEqual(reopen['positions'], current['positions'])
        self.assertTrue(all(second['mask_before'][i] == 1 for i in reopen['reopen']))
        self.assertEqual(second['positions'], [4, 5, 6, 7])

    def test_forward_cap_does_not_emit_an_unfinished_block(self):
        row = {'state': {'prompt_ids': [7] * 64}, 'token_ids': list(range(8192)),
               'forward_calls': 700, 'finish_reason': 'length'}
        complete = dict(event(8, [28, 29, 30, 31], range(28, 32)), forward=510, block_start=128)
        partial = dict(event(1, [0, 1, 2, 3], range(32)), forward=512, block_start=160)
        later = dict(complete, forward=520, block_start=160)
        result = capped_reference(row, [complete, partial, later])
        self.assertEqual(result['token_ids'], list(range(96)))
        self.assertEqual(result['forward_calls'], 512)
        self.assertEqual(result['finish_reason'], 'forward_budget')
        self.assertEqual(len(result['events']), 2)


if __name__ == '__main__':
    unittest.main()
