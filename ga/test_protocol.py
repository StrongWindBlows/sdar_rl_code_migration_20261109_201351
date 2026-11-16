"""Small tests of experimental contrasts and bookkeeping."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from protocol import (action_pool, divergence_summary, file_hash, forward_budget, load_data,
                      oracle_summary, paired_stats, restoration_difference, trace_difference)


def example_state(differences=4, commit_size=4):
    return {'id': 'q@64:2', 'question_id': 'q', 'block_start': 64, 'prompt_length': 40,
            'positions': list(range(commit_size)), 'masked_positions': list(range(28)),
            'current_ids': [10 + i for i in range(32)],
            'historical_ids': [100 + i if i < differences else 10 + i for i in range(32)],
            'confidence': [1 - i / 100 for i in range(32)],
            'topk_ids': [[10 + i, 200 + i] for i in range(32)]}


class ProtocolTests(unittest.TestCase):
    def test_degenerate_cross_pool(self):
        for differences, size, hybrids in ((0, 1, 0), (1, 2, 0), (2, 4, 2), (4, 16, 14)):
            pool = action_pool(example_state(differences))
            self.assertEqual(len(pool['pools']['cross_round']), size)
            self.assertEqual(sum('hybrid' in a['families'] for a in pool['actions']), hybrids)
            self.assertEqual(len(pool['matched']['current_only']), size)
            self.assertEqual(len(pool['matched']['cross_round']), size)
            self.assertTrue(all(a['positions'] == [0, 1, 2, 3] for a in pool['actions']))

    def test_shared_support_is_deduplicated(self):
        state = example_state()
        state['historical_ids'] = [values[1] for values in state['topk_ids']]
        pool = action_pool(state)
        self.assertEqual(len(pool['actions']), 16)
        self.assertEqual(set(pool['pools']['cross_round']), set(pool['pools']['current_only']))
        self.assertEqual(pool, action_pool(state))

    def test_t4_changes_only_four_coordinates(self):
        state = example_state(8, 8)
        pool = action_pool(state)
        self.assertEqual(len(pool['actions']), 31)
        for action in pool['actions']:
            self.assertEqual(action['positions'], list(range(8)))
            self.assertEqual(action['tokens'][4:], state['current_ids'][4:8])

    def test_diagnostic_position_control_is_shared(self):
        state = example_state(0)
        state['historical_ids'][7:11] = [700, 800, 900, 1000]
        pool = action_pool(state, positions='disagreement')
        self.assertTrue(all(a['positions'] == [7, 8, 9, 10] for a in pool['actions']))

    def test_budget_includes_cache_and_remaining_blocks(self):
        state = example_state()
        self.assertEqual(forward_budget(state, 56, 8), 7)  # 24 masks left: six rounds plus cache.
        self.assertEqual(forward_budget(state, 88, 8), 16)

    def test_pilot_cannot_include_evaluation_questions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [{'unique_id': str(i)} for i in range(100)]
            (root / 'test.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
            manifest = {'dev_ids': list(map(str, range(20))), 'eval_ids': list(map(str, range(20, 100))),
                        'pilot_ids': list(map(str, range(10))), 'sample_sha256': file_hash(root / 'test.jsonl'),
                        'items': [{'unique_id': str(i), 'intervention_round': 2 + i % 2} for i in range(100)]}
            (root / 'manifest.json').write_text(json.dumps(manifest))
            self.assertEqual(len(load_data(root, 'pilot')), 10)
            manifest['pilot_ids'][0] = '80'
            (root / 'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaises(AssertionError):
                load_data(root, 'pilot')

    def test_same_answer_does_not_hide_different_commit_trajectory(self):
        event = {'block_start': 64, 'round': 2, 'input_ids': [0], 'mask_before': [0],
                 'mask_after': [1], 'positions': [0], 'tokens': [42], 'forward': 5}
        baseline = {'events': [event], 'token_ids': [42], 'finish_reason': 'eos', 'forward_calls': 5}
        restored = {'events': [copy.deepcopy(event)], 'token_ids': [42], 'finish_reason': 'eos', 'future_forwards': 0}
        self.assertIsNone(restoration_difference(baseline, restored, {'source_forward': 5}))
        restored['events'][0]['positions'] = [1]
        self.assertEqual(restoration_difference(baseline, restored, {'source_forward': 5})['field'], 'positions')
        self.assertEqual(trace_difference([event], [])['field'], 'event_count')

    def test_oracle_detects_a_hybrid_rescue(self):
        pool = action_pool(example_state())
        winner = next(a['id'] for a in pool['actions'] if 'hybrid' in a['families'])
        records = [{'state_id': 's', 'question_id': 'q', 'action': action, 'pool': pool,
                    'reward': float(action['id'] == winner)} for action in pool['actions']]
        summary = oracle_summary(records)
        self.assertEqual(summary['hybrid_rescues'], 1)
        self.assertFalse(summary['details'][0]['cross_is_current_subset'])
        self.assertEqual(paired_stats({'q': 0}, {'q': 1})['difference'], 1)

    def test_divergence_uses_eligible_blocks_and_separates_rounds(self):
        event = {'block_start': 64, 'round': 2, 'phase': 'denoise', 'full_generation_block': True,
                 'positions': [0, 1, 2, 3], 'd_selected': 1, 'd_mutable': 1}
        third = dict(event, round=3, d_selected=3, d_mutable=3)
        terminal = dict(event, block_start=96, d_selected=4, d_mutable=4)
        cached = dict(event, phase='cache', positions=[], d_selected=None, d_mutable=None)
        summary = divergence_summary([event, third, terminal, cached])
        self.assertEqual(summary['states'], 2)
        self.assertEqual(summary['fraction_below_two'], .5)
        self.assertEqual(summary['by_round']['2']['d_selected'], {0: 0, 1: 1, 2: 0, 3: 0, 4: 0})
        self.assertEqual(summary['by_round']['3']['fraction_at_least_two'], 1)

    def test_grading_timeout_is_not_oracle_evidence(self):
        pool = action_pool(example_state())
        rows = [{'state_id': 's', 'question_id': 'q', 'action': action, 'pool': pool,
                 'reward': 0.0, 'grading': 'timeout' if i == 0 else 'ok'} for i, action in enumerate(pool['actions'])]
        self.assertEqual(oracle_summary(rows)['states_with_grading_timeouts'], ['s'])


if __name__ == '__main__':
    unittest.main()
