import unittest

import torch

from branch_rl import acceptance, branch_loss
from branch_rollout import alternate_action, branch_targets, marginal_position


class BranchObjectiveTests(unittest.TestCase):
    def test_gradient_matches_expected_two_action_return(self):
        parameter = torch.tensor([0.3, -0.4, 0.1], dtype=torch.float64, requires_grad=True)
        probability = torch.sigmoid(parameter + 2.2)
        reward = parameter.new_tensor([[0, 1], [1, 0], [1, 1]])
        cost = parameter.new_tensor([[9, 12], [7, 3], [12, 4]])
        price, threshold, tau = .01, .9, .025
        gate = torch.sigmoid((probability - threshold) / tau)
        utility = reward - price * cost
        exact = -((1 - gate) * utility[:, 0] + gate * utility[:, 1]).mean()
        target, = torch.autograd.grad(exact, parameter, retain_graph=True)
        delta_r, delta_c = reward.diff(1).flatten(), cost.diff(1).flatten()
        actual, = torch.autograd.grad(branch_loss(probability, delta_r, delta_c, threshold, tau, price),
                                      parameter)
        torch.testing.assert_close(actual, target)
        self.assertTrue(bool((actual * parameter.new_tensor([-1, 1, -1]) > 0).all()))

    def test_marginal_edit_preserves_required_commit(self):
        event = dict(full_generation_block=True, mask_before=[1, 0, 0, 0],
                     topk_probs=[[.99], [.904], [.94], [.875]],
                     current_ids=[6, 7, 8, 9], topk_ids=[[6], [7], [8], [9]], positions=[1, 2])
        position = marginal_position(event, .9, .04)
        self.assertEqual(position, 1)
        self.assertEqual(alternate_action(event, position), (True, {'positions': [2], 'tokens': [8]}))
        event['positions'] = [2]
        self.assertEqual(alternate_action(event, 3), (False, {'positions': [2, 3], 'tokens': [8, 9]}))
        event['mask_before'] = [0, 0, 0, 0]
        self.assertIsNone(marginal_position(event, .9, .04))
        event['mask_before'] = [1, 0, 0, 0]
        event['topk_probs'] = [[.99], [.90], [.90], [.76]]
        self.assertIsNone(marginal_position(event, .9, .04))

    def test_targets_exclude_replay_and_are_always_commit_minus_defer(self):
        baseline = dict(reward=1, forward_calls=42)
        alternate = dict(reward=0, future_forwards=12, restore_forwards=15)
        for committed, sign in [(True, 1), (False, -1)]:
            target = branch_targets(baseline, alternate, 15, committed)
            self.assertEqual(target['delta_reward'], sign)
            self.assertEqual(target['delta_forwards'], sign * (28 - 13))
            self.assertEqual(target['replay_overhead'], 14)

    def test_acceptance_requires_all_three_paired_gains(self):
        cfg = dict(seed=1, bootstrap_samples=100, alpha=.05, minimum_dynamic_gain=0,
                   minimum_forward_saving=.01, maximum_sequential_loss=.02)
        def dynamic(r, c):
            return [dict(index=i, reward=r, forward_calls=c, denoise_forwards=c,
                         num_tokens=10, finished=True, grading='ok') for i in range(5)]
        def sequential(r):
            return [dict(index=i, rewards=[r] * 4, timeouts=0) for i in range(5)]
        base, candidate = dynamic(0, 10), dynamic(1, 8)
        self.assertTrue(acceptance(cfg, base, candidate, sequential(1), sequential(1))['accepted'])
        self.assertFalse(acceptance(cfg, base, candidate, sequential(1), sequential(0))['accepted'])
        self.assertFalse(acceptance(cfg, base, dynamic(1, 10), sequential(1), sequential(1))['accepted'])


if __name__ == '__main__':
    unittest.main()
