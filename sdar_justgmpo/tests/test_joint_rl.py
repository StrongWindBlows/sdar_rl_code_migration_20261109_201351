from copy import deepcopy
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from algorithm import Response, backward_rloo
from cooperative import leave_one_out
from joint_rl import acceptance, common_direction, export_candidate, smoothed_gradients
from train import step_direction_on_cpu


class JointRLTests(unittest.TestCase):
    def test_cpu_direction_step_matches_first_adamw_step(self):
        model = torch.nn.Linear(3, 2)
        reference = deepcopy(model)
        for parameter, expected in zip(model.parameters(), reference.parameters(), strict=True):
            parameter.grad = torch.linspace(-0.1, 0.1, parameter.numel()).reshape_as(parameter)
            expected.grad = parameter.grad.clone()
        options = dict(lr=5e-6, weight_decay=0.0, foreach=False)
        torch.optim.AdamW(reference.parameters(), **options).step()
        step_direction_on_cpu(model, torch.optim.AdamW(model.parameters(), **options))
        for actual, expected in zip(model.parameters(), reference.parameters(), strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_expected_rloo_gradient_with_variable_length(self):
        logits = torch.tensor([0.3, -0.8], dtype=torch.float64, requires_grad=True)
        rewards = torch.tensor([0.0, 1.0, 0.0], dtype=logits.dtype)

        def logps(index):
            first, second = logits.sigmoid()
            return [(1 - first).log()] if index == 0 else [
                first.log(), (second if index == 1 else 1 - second).log()]

        probabilities = torch.stack([sum(logps(i)).exp() for i in range(3)])
        target, = torch.autograd.grad((probabilities * rewards).sum(), logits)
        probabilities = probabilities.detach()

        def phases(model, response, *args):
            for phase in range(len(response.tokens)):
                value = logps(response.tokens[0])[phase]
                yield None, value.view(1), value.detach().view(1) * 0

        with patch('algorithm.phase_logps', phases):
            for i, j in product(range(3), repeat=2):
                advantages = leave_one_out(rewards[[i, j]]) * probabilities[i] * probabilities[j]
                for index, advantage in zip((i, j), advantages.tolist()):
                    response = Response([], [index] * len(logps(index)), rewards[index].item(),
                                        advantage=advantage)
                    backward_rloo(None, response, 2)
        torch.testing.assert_close(logits.grad, -target)

    def test_weight_endpoints_and_logit_scale(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / 'base'
            base.mkdir()
            model = {'lm_head.weight': torch.tensor([[2.0, -4.0]], dtype=torch.bfloat16),
                     'model.layer.weight': torch.tensor([1.0], dtype=torch.bfloat16)}
            delta = {name: torch.full_like(weight, .25, dtype=torch.float32)
                     for name, weight in model.items()}
            save_file(model, base / 'model.safetensors')
            (base / 'config.json').write_text('{}')
            self.assertEqual(export_candidate(base, delta, (0.0, 0.0), root / 'zero'), base)
            candidate = export_candidate(base, delta, (1.0, np.log(2.0)), root / 'candidate')
            actual = load_file(candidate / 'model.safetensors')
            torch.testing.assert_close(actual['lm_head.weight'], torch.tensor([[4.5, -7.5]]).bfloat16())
            torch.testing.assert_close(actual['model.layer.weight'], torch.tensor([1.25]).bfloat16())
            self.assertTrue((candidate / 'config.json').is_file())

    def test_antithetic_gradients_and_common_ascent(self):
        perturbations = np.eye(2) * np.sqrt(2)
        expected = np.array([[2., 1.], [-1., 2.]])
        plus = perturbations @ expected.T
        returns = np.stack((plus + 4, 4 - plus), axis=1)
        gradients = smoothed_gradients(perturbations, returns)
        np.testing.assert_allclose(gradients, expected)
        direction = common_direction(gradients)
        self.assertTrue(np.all(gradients @ direction > 0))
        self.assertAlmostEqual(np.linalg.norm(direction), 1.)
        self.assertIsNone(common_direction(np.array([[1., 0.], [-2., 0.]])))

    def test_acceptance_requires_tpf_in_addition_to_total_forwards(self):
        config = dict(max_tokens=3000, block_size=32, alpha=.05, minimum_dynamic_gain=0.,
                      minimum_tpf_gain=0., minimum_forward_saving=.01, maximum_sequential_loss=.02)
        size = 64

        def results(reward, tokens, denoise, forwards):
            dynamic = [dict(index=i, reward=reward, num_tokens=tokens, denoise_forwards=denoise,
                            forward_calls=forwards, finished=True, grading='ok') for i in range(size)]
            sequential = [dict(index=i, rewards=[reward] * 4, timeouts=0) for i in range(size)]
            return dynamic, sequential

        base = results(0., 40, 10, 12)
        faster = acceptance(config, base, results(1., 40, 8, 9), size, 3.9)
        self.assertTrue(faster['accepted'])
        self.assertAlmostEqual(faster['simultaneous_lower_bounds']['tpf_gain'], 1.)
        shorter = acceptance(config, base, results(1., 30, 8, 9), size, 3.9)
        self.assertTrue(shorter['passes']['forward_saving'])
        self.assertFalse(shorter['passes']['tpf_gain'])
        self.assertFalse(shorter['accepted'])
