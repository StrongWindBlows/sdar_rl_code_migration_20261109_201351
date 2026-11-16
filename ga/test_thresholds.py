"""Semantic checks for threshold schedules and fair measurement."""
import unittest

from threshold_rules import calibrate, factor_count, threshold
from threshold_study import measure


class ThresholdTests(unittest.TestCase):
    def test_step_schedule_uses_current_round_and_resets_each_block(self):
        rule = dict(kind='step', start=.95, end=.8, span=8)
        f = lambda b, s: threshold(rule,b,s,0,8192,{})[0]
        self.assertAlmostEqual(f(0,1),.95)
        self.assertAlmostEqual(f(0,8),.8)
        self.assertAlmostEqual(f(0,32),.8)
        self.assertAlmostEqual(f(1,1),.95)

    def test_sequence_progress_does_not_use_future_eos(self):
        rule = dict(kind='cad',start=.95,end=.8)
        self.assertAlmostEqual(threshold(rule,3,2,4096,8192,{})[0],.875)

    def test_osdt_missing_block_and_step_fall_back_explicitly(self):
        rule = dict(kind='osdt',mode='step_block',cap=.98,slack=.05,lower=.75,upper=.98,fallback=.9)
        table = {'step_block':[[.99,.5]]}
        self.assertEqual(threshold(rule,0,3,0,8192,table),(.9,True,False))
        self.assertEqual(threshold(rule,1,1,0,8192,table),(.9,True,False))
        self.assertEqual(threshold(rule,0,2,0,8192,table),(.75,False,True))

    def test_factor_responds_to_distribution_and_forces_progress(self):
        self.assertEqual(factor_count([.99,.98,.97,.96],.25),(4,False))
        self.assertEqual(factor_count([.99,.7,.5,.1],.25),(1,False))
        self.assertEqual(factor_count([.2,.1],.25),(1,True))

    def test_calibration_uses_committed_generated_positions_only(self):
        e = dict(phase='denoise',block_start=0,positions=[0,5,6,7],confidence=[.8]*32)
        e['confidence'][0] = .01
        e['confidence'][7] = .99
        table = calibrate([e],5,2)
        self.assertEqual(table,{'block':[.8],'step_block':[[.8]]})

    def test_mixed_prefill_counts_as_denoise_and_late_eos_is_not_budget_success(self):
        e = dict(phase='denoise',forward=1,round=1,block_start=0,
            full_generation_block=False,mask_before=[1]*30+[0,0],mask_after=[1]*32,
            positions=[30,31],tokens=[10,11],confidence=[.99]*32,
            effective_threshold=.9)
        cache = dict(e,phase='cache',forward=2,mask_before=[1]*32,mask_after=[2]*32,
                     positions=[],tokens=[])
        result = dict(events=[e,cache],token_ids=[10,11],forward_calls=2,
                      forward_types={'prefill':1,'cache':1},finish_reason='eos')
        metrics = measure(result,30,dict(kind='fixed',threshold=.9),[1,2])
        self.assertEqual(metrics['denoise_forwards'],1)
        self.assertEqual(metrics['mixed_denoise_forwards'],1)
        self.assertFalse(metrics['budget_views']['1']['eos'])
        self.assertTrue(metrics['budget_views']['2']['eos'])


if __name__ == '__main__':
    unittest.main()
