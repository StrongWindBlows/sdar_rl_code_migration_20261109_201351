"""Checks for observable features, grouped weights, and label-blind matching."""
from copy import deepcopy
import unittest

import numpy as np

from confidence_audit import confidence_matches, feature_layers, observation_weights, select_actions
from test_marginal_commit import STUDY, state
from marginal_commit import make_actions


class ConfidenceTests(unittest.TestCase):
    def test_features_exclude_visible_logits_and_outcomes(self):
        source=state()
        action=make_actions(source,STUDY)[0][-1]
        original=feature_layers(source,action,[])
        changed=deepcopy(source)
        changed.update(reward=1,future_reward=0)
        for i in (0,1):
            changed['confidence'][i]=.001
            changed['topk_probs'][i]=[0.,1.]
            changed['entropy'][i]=100.
        self.assertEqual(original,feature_layers(changed,action,[]))

    def test_history_cannot_change_confidence_layers(self):
        source=state()
        action=make_actions(source,STUDY)[0][-1]
        history=[dict(ids=[999]*32,probability=[.5]*32,forward=4)]
        absent,present=feature_layers(source,action,[]),feature_layers(source,action,history)
        self.assertEqual(absent['C0'],present['C0'])
        self.assertEqual(absent['C1'],present['C1'])
        self.assertEqual(absent['C2']['history_available'],0)
        self.assertEqual(present['C2']['history_available'],1)
        self.assertEqual(present['C2']['added_last_flip'],1.)

    def test_question_weight_is_independent_of_number_of_states_and_actions(self):
        rows=[dict(question_id='a',case_id='a1')]*3+[dict(question_id='a',case_id='a2')]+[dict(question_id='b',case_id='b1')]*2
        w=observation_weights(rows)
        self.assertAlmostEqual(w[:4].sum(),w[4:].sum())
        self.assertAlmostEqual(w[:3].sum(),w[3])

    def test_ranked_confidences_preserve_the_full_masked_multiset(self):
        source=state()
        action=make_actions(source,STUDY)[0][-1]
        layers=feature_layers(source,action,[])
        values=[layers['C1'][f'rank_p_{i:02d}'] for i in range(32)]
        expected=sorted([source['confidence'][i] for i in source['masked_positions']],reverse=True)+[0.,0.]
        np.testing.assert_allclose(values,expected)

    def test_matching_does_not_use_labels_and_does_not_reuse_states(self):
        def row(i,p):
            return dict(case_id=f's{i}',question_id=f'q{i}',window=0,action_id='k03',delta_k=2,
                delta_reward=0,delta_forward=0,features=dict(C0=dict(k0=1,delta_k=2,
                    masked=20,round=3,added_p_min=p,added_p_mean=p)))
        rows=[row(0,.5),row(1,.501),row(2,.52),row(3,.521)]
        first=confidence_matches(rows,True)
        other=deepcopy(rows)
        other[1]['delta_reward']=-1
        second=confidence_matches(other,True)
        pairs=lambda result:[(r['left']['case_id'],r['right']['case_id']) for r in result['pair_details']]
        self.assertEqual(pairs(first),pairs(second))
        self.assertEqual(first['states'],4)
        self.assertEqual(second['unequal_delta_reward'],1)

    def test_selection_keeps_current_for_negative_expected_reward(self):
        source=state()
        actions,_=make_actions(source,STUDY)
        case=dict(id='s',snapshot=source,actions=actions)
        prediction={('s',a['id']):[.1,0.,-1.] for a in actions}
        prediction['s','k04']=[0.,0.,0.]
        self.assertEqual(select_actions([case],prediction)['s'],'k04')
        prediction['s','k08']=[0.,.1,1.]
        self.assertEqual(select_actions([case],prediction)['s'],'k08')


if __name__=='__main__':
    unittest.main()
