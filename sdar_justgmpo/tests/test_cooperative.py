import copy
import unittest

import torch

from algorithm import Response
from cooperative import leave_one_out, prepare_coupled, backward_coupled
import test_decode_consistency as decode_tests


class CoupledObjectiveTests(unittest.TestCase):
    def test_leave_one_out_joint_expectation_matches_complete_objective_gradient(self):
        theta = torch.tensor([.3, -.4, .8, -.5, -.3, .9], dtype=torch.float64, requires_grad=True)
        first = theta[:2].log_softmax(0)
        conditional = theta[2:].reshape(2, 2).log_softmax(1)
        parallel = theta[2:].reshape(2, 2).mean(0).log_softmax(0)
        ls = torch.stack([first[i]+conditional[i, j] for i in range(2) for j in range(2)])
        lp = torch.stack([first[i]+parallel[j] for i in range(2) for j in range(2)])
        reward = theta.new_tensor([1., 0., 0., 1.])
        beta = .3
        utility = reward - beta*(ls-lp)
        exact, = torch.autograd.grad((ls.exp()*utility).sum(), theta, retain_graph=True)
        surrogate = theta.new_zeros(())
        # Enumerate every pair of independent responses to the same prompt.
        for i in range(4):
            for j in range(4):
                advantage = leave_one_out(utility[[i, j]].detach())
                surrogate += (ls[i]+ls[j]).exp().detach() * (
                    advantage[0]*ls[i] + advantage[1]*ls[j] + beta*(lp[i]+lp[j])) / 2
        actual, = torch.autograd.grad(surrogate, theta)
        torch.testing.assert_close(actual, exact, atol=1e-14, rtol=1e-12)


class CoupledGradientTests(unittest.TestCase):
    setUp = decode_tests.ReconstructionTests.setUp
    serial = decode_tests.ReconstructionTests.serial
    gradient = decode_tests.ReconstructionTests.gradient

    def test_statistics_preserve_prompt_groups_and_use_sequence_gap_without_length_scaling(self):
        responses = [Response([1]*3, [6, 7, 2], 1.), Response([1]*3, [8, 2], 0.),
                     Response([1]*5, [6, 7, 8, 2], 0.), Response([1]*5, [9, 2], 0.)]
        beta = .03
        expected_gaps = [(self.serial(r)-self.serial(r, full_mask=True)).double().sum().item()
                         for r in responses]
        gaps = prepare_coupled(self.model, responses, 2, beta, phase_batch_size=4, logit_chunk_size=2)
        torch.testing.assert_close(torch.tensor(gaps), torch.tensor(expected_gaps), atol=1e-6, rtol=1e-6)
        utility = [r.reward-beta*d for r, d in zip(responses, expected_gaps)]
        expected = [utility[0]-utility[1], utility[1]-utility[0],
                    utility[2]-utility[3], utility[3]-utility[2]]
        torch.testing.assert_close(torch.tensor([r.advantage for r in responses]), torch.tensor(expected))
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))

    def test_streamed_update_matches_serial_coupled_gradient_and_adam(self):
        responses = [Response([1]*3, [6, 7, 8, 2], 1.), Response([1]*3, [9, 2], 0.)]
        beta, size = .03, len(responses)
        prepare_coupled(self.model, responses, size, beta, phase_batch_size=4, logit_chunk_size=2)
        initial = copy.deepcopy(self.model.state_dict())
        reference = sum(-(r.advantage*self.serial(r).sum()
                          + beta*self.serial(r, full_mask=True).sum()) / size for r in responses)
        reference.backward()
        expected = self.gradient().clone()
        torch.optim.AdamW(self.model.parameters(), lr=.001, weight_decay=0., foreach=False).step()
        expected_weights = copy.deepcopy(self.model.state_dict())
        for phases in (1, 4):
            self.model.load_state_dict(initial)
            self.model.zero_grad(set_to_none=True)
            if phases == 4:
                self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
            stats = [backward_coupled(self.model, r, size, beta, phase_batch_size=phases,
                                     logit_chunk_size=2) for r in responses]
            torch.testing.assert_close(self.gradient(), expected, atol=3e-7, rtol=3e-5)
            self.assertAlmostEqual(sum(s['loss'] for s in stats)/size, reference.item(), places=5)
            self.assertEqual(sum(s['tokens'] for s in stats), sum(len(r.tokens) for r in responses))
            torch.optim.AdamW(self.model.parameters(), lr=.001, weight_decay=0., foreach=False).step()
            for name, weight in self.model.state_dict().items():
                torch.testing.assert_close(weight, expected_weights[name], atol=2e-6, rtol=2e-5)

    def test_zero_sequential_coefficient_keeps_direct_parallel_gradient(self):
        response = Response([1]*3, [6, 7, 8, 2], 0., advantage=0.)
        (-.03*self.serial(response, full_mask=True).sum()).backward()
        expected = self.gradient().clone()
        self.model.zero_grad(set_to_none=True)
        stats = backward_coupled(self.model, response, 1, .03, phase_batch_size=4, logit_chunk_size=2)
        torch.testing.assert_close(self.gradient(), expected, atol=3e-7, rtol=3e-5)
        self.assertGreater(self.gradient().norm().item(), 0)
        self.assertEqual(stats['tokens'], len(response.tokens))


if __name__ == '__main__':
    unittest.main()
