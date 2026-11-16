"""Checks for the expanded development split and the full question denominator."""
from copy import deepcopy
import unittest

from joint50 import cohort_statistics, select_fifty


class Joint50Tests(unittest.TestCase):
    def test_expansion_preserves_old_dev_and_explicitly_retires_promoted_eval(self):
        rows = [{'unique_id': f'q{i:03}', 'subject': f's{i % 3}', 'level': i % 5} for i in range(100)]
        old = {'dev_ids': [r['unique_id'] for r in rows[:20]],
               'eval_ids': [r['unique_id'] for r in rows[20:]],
               'pilot_ids': [r['unique_id'] for r in rows[:10]]}
        original = deepcopy(old)
        data, split = select_fifty(rows, old)
        self.assertEqual(select_fifty(rows, old), (data, split))
        self.assertEqual(old, original)
        self.assertEqual(len(data), 50)
        self.assertTrue(set(old['dev_ids']) <= set(split['selected_ids']))
        self.assertEqual(len(split['new40_ids']), 40)
        self.assertEqual(len(split['promoted_eval30_ids']), 30)
        self.assertTrue(set(split['remaining_eval50_ids']).isdisjoint(split['selected_ids']))
        self.assertEqual(set(split['promoted_eval30_ids']) | set(split['remaining_eval50_ids']), set(old['eval_ids']))

    def test_missing_states_use_current_and_pilot_does_not_enter_fresh_cohort(self):
        data = [{'unique_id': 'old', 'cohort': 'pilot10'}, {'unique_id': 'new', 'cohort': 'new40'}]
        baselines = [{'question_id': q, 'setting': setting, 'reward': float(q == 'old')}
                     for setting in ('T8', 'T4') for q in ('old', 'new')]
        result = cohort_statistics(data, baselines, [], 'all50')
        for s in result.values():
            self.assertEqual(s['questions'], 2)
            self.assertEqual(s['source_states'], 0)
            self.assertEqual(set(s['oracle_correct'].values()), {1.})
        fresh = cohort_statistics(data, baselines, [], 'new40')
        self.assertEqual(fresh['T8']['questions'], 1)
        self.assertEqual(fresh['T8']['all_groups_correct'], 0.)


if __name__ == '__main__':
    unittest.main()
