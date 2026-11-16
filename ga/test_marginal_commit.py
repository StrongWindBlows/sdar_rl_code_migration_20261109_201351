"""Semantic checks for count interventions, observable features and utility labels."""
from copy import deepcopy
import unittest

from marginal_commit import (block_metrics, capped_reference, choose_action, features,
                             make_actions, outcome, select_sources)


STUDY = {'seed': 20261112, 'windows': [[0,128],[128,256],[256,512]],
         'forward_cap': 512, 'deltas': [-2,0,2,4],
         'control_thresholds': [.85,.95], 'control_factor': 1., 'score_dtype':'bfloat16'}


def state():
    p = [.99,.98,.97,.96,.94,.92,.88,.84]+[.5]*24
    return {'masked_positions': list(range(2,32)), 'mask': [1,1]+[0]*30,
        'positions': [2,3,4,5], 'confidence': p, 'current_ids': list(range(32)),
        'topk_probs': [[v, (1-v)/2] for v in p], 'entropy': [1-v for v in p],
        'block_start': 64, 'prompt_length': 64, 'source_forward': 5, 'round': 2}


class MarginalTests(unittest.TestCase):
    def test_actions_are_nested_and_token_values_are_fixed(self):
        s = state()
        before = deepcopy(s)
        actions, policies = make_actions(s, STUDY)
        self.assertEqual(s, before)
        self.assertEqual([policies[n] for n in ('D-2','Current','D+2','D+4')], [2,4,6,8])
        for left,right in zip(actions, actions[1:]):
            self.assertLess(set(left['positions']), set(right['positions']))
        for a in actions:
            self.assertEqual(a['tokens'], [s['current_ids'][i] for i in a['positions']])

    def test_clipping_deduplicates_and_preserves_current_ties(self):
        s = state()
        s.update(masked_positions=[2,3,4], mask=[1,1,0,0,0]+[1]*27,
                 positions=[3], confidence=[.5]*32)
        actions, policies = make_actions(s, STUDY)
        self.assertEqual(policies['D-2'], 1)
        self.assertEqual(policies['D+2'], policies['D+4'])
        self.assertEqual(len(actions), len({a['k'] for a in actions}))
        self.assertEqual(next(a for a in actions if 'Current' in a['policies'])['positions'], [3])

    def test_control_threshold_uses_native_score_precision(self):
        s=state()
        s['confidence'][4]=s['confidence'][5]=0.94921875
        actions,policies=make_actions(s,STUDY)
        self.assertEqual(policies['F095'],4)
        self.assertEqual(next(a for a in actions if 'F095' in a['policies'])['positions'],[2,3,4,5])

    def test_sampling_does_not_depend_on_future_outcomes_or_other_windows(self):
        def event(base, forward):
            return dict(phase='denoise', full_generation_block=True, block_start=base,
                        forward=forward, mask_before=[0]*32, positions=[0,1,2,3])
        first = [event(64,2), event(96,5)]
        a = select_sources(first, 'q', 64, STUDY)
        b = select_sources(first+[event(320,10),event(640,20),event(96,600)], 'q', 64, STUDY)
        self.assertEqual(a[0], b[0])
        self.assertEqual(a[0][0], 0)
        self.assertTrue(all(e['forward'] < 512 for _,e in b))

    def test_features_ignore_visible_logits_and_history(self):
        s = state()
        action = make_actions(s, STUDY)[0][-1]
        original = features(s, action)
        other = deepcopy(s)
        other['historical_ids'] = [999]*32
        other['reward'], other['future_eos'] = 1, 1000
        for i in (0,1):
            other['confidence'][i] = 0.
            other['entropy'][i] = 100.
            other['topk_probs'][i] = [0.,1.]
        self.assertEqual(original, features(other, action))

    def test_failed_shorter_path_is_not_a_quality_preserving_save(self):
        current = {'reward': 0., 'forward_calls': 512}
        self.assertEqual(outcome(current, {'reward':0.,'forward_calls':30}), 'both_wrong')
        self.assertEqual(outcome(current, {'reward':1.,'forward_calls':100}), 'repair')
        current['reward'] = 1.
        self.assertEqual(outcome(current, {'reward':0.,'forward_calls':30}), 'harm')
        self.assertEqual(outcome(current, {'reward':1.,'forward_calls':100}), 'safe_save')

    def test_unfinished_block_is_censored_not_zero_steps(self):
        s = state()
        events = [{'block_start':64,'mask_after':[0]*32,'phase':'denoise','forward':5}]
        a = block_metrics(events, s, 'forward_budget')
        self.assertFalse(a['complete'])
        self.assertIsNone(a['extra_denoise'])
        events += [{'block_start':64,'mask_after':[1]*32,'phase':'denoise','forward':6}]
        self.assertEqual(block_metrics(events,s,'eos')['extra_denoise'],1)
        self.assertEqual(block_metrics([dict(events[1],forward=5)],s,'eos')['extra_denoise'],0)

    def test_budget_view_excludes_future_eos_and_partial_blocks(self):
        row = {'state':{'prompt_ids':[1]*64},'token_ids':list(range(64)),
               'forward_calls':600,'finish_reason':'eos','finished':True}
        events = [{'forward':510,'block_start':64,'mask_after':[1]*32},
                  {'forward':512,'block_start':96,'mask_after':[0]*32},
                  {'forward':600,'block_start':96,'mask_after':[1]*32}]
        result = capped_reference(row,events,512)
        self.assertEqual(result['token_ids'],list(range(32)))
        self.assertEqual(result['finish_reason'],'forward_budget')
        self.assertFalse(result['finished'])

    def test_predicted_reward_dominates_cost_and_ties_keep_current(self):
        s=state()
        actions,_=make_actions(s,STUDY)
        case={'snapshot':s,'actions':actions}
        estimates={a['id']:[0.,0.] for a in actions}
        self.assertEqual(choose_action(case,estimates),'k04')
        estimates['k08']=[-.1,-1.]
        estimates['k02']=[0.,-.2]
        self.assertEqual(choose_action(case,estimates),'k02')
        estimates['k06']=[.1,1.]
        self.assertEqual(choose_action(case,estimates),'k06')


if __name__ == '__main__':
    unittest.main()
