"""Tests for event timing, censoring, and causal features."""
import copy
import unittest

import numpy as np

from degeneration import FeatureStream, NAMES, alarm_threshold, align_event, repeat_runs, split_questions, target


def event(forward=1):
    return {'block_start': 0, 'forward': forward, 'phase': 'denoise', 'round': forward,
        'input_ids': [151669]*32, 'mask_before': [0]*32, 'positions': [0, 5], 'tokens': [100, 105],
        'current_ids': list(range(100, 132)), 'topk_probs': [[.8, .1, .05, .03, .02]]*32,
        'entropy': [1.]*32, 'history_probability_now': None}


class DegenerationTests(unittest.TestCase):
    def test_long_period_and_minimum_duration(self):
        ids = list(range(67))*5
        self.assertFalse(repeat_runs(ids, 64))
        run = repeat_runs(ids)[0]
        self.assertEqual((run['period'], run['onset'], run['confirmation_end']), (67,67,268))
        self.assertFalse(repeat_runs([1]*31))
        self.assertEqual(repeat_runs([1]*32)[0]['onset'], 1)

    def test_commit_order_is_not_token_order(self):
        run = repeat_runs(list(range(8))*4)[0]
        commits = {i: (i//4+1, i%8) for i in range(32)}
        commits[8] = (9, 0)
        aligned = align_event(run, commits, 0, [])
        self.assertEqual(aligned['onset_commit'], 9)
        self.assertEqual(aligned['first_repeat_commit'], 3)
        self.assertEqual(aligned['confirmation_forward'], 9)

    def test_censoring_and_terminal_negative(self):
        f = {'forward': 10, 'generation_end': 64}
        r = {'event': None, 'tokens': 100, 'finish_reason': 'length'}
        self.assertIsNone(target(f, r))
        self.assertEqual(target(f, dict(r, finish_reason='eos')), 0)
        self.assertEqual(target(f, dict(r, tokens=1400)), 0)
        r['event'] = {'onset': 100, 'first_repeat_commit': 30}
        self.assertEqual(target(f,r), 1)
        self.assertIsNone(target(dict(f,forward=30),r))

    def test_future_data_cannot_change_past_features(self):
        first = event()
        before = FeatureStream([],8).observe(first)
        future = event(2)
        future['current_ids'] = [999]*32
        after = FeatureStream([],8).observe(copy.deepcopy(first))
        np.testing.assert_allclose(before['x'], after['x'], equal_nan=True)
        self.assertEqual(len(after['x']), len(NAMES))

    def test_visible_readout_does_not_replace_visible_token(self):
        first, second = event(), event(2)
        second.update(positions=[1,2], tokens=[101,102], history_probability_now=[.8]*32)
        for i in (0,5):
            second['mask_before'][i] = 1
            second['input_ids'][i] = first['current_ids'][i]
        altered = copy.deepcopy(second)
        for i in (0,5):
            altered['current_ids'][i] = 9999
        a,b = FeatureStream([],8), FeatureStream([],8)
        a.observe(first); b.observe(first)
        np.testing.assert_allclose(a.observe(second)['x'], b.observe(altered)['x'], equal_nan=True)

    def test_question_split_and_whole_trajectory_threshold(self):
        folds = {str(i):i%5 for i in range(50)}
        for fold in range(5):
            train,calibration,test = split_questions(folds,fold)
            self.assertEqual((len(train),len(calibration),len(test)),(30,10,10))
            self.assertFalse(train&calibration or train&test or calibration&test)
        maxima = [.1,.2,.3,.4,.5,.6,.7,.8,.9,1.]
        for rate in (.05,.10):
            threshold = alarm_threshold(maxima,rate)
            self.assertLessEqual(sum(v>=threshold for v in maxima),int(len(maxima)*rate))


if __name__ == '__main__':
    unittest.main()
