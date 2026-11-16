from io import StringIO
import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from eval import evaluate
from reward import correctness


class EvaluationTests(unittest.TestCase):
    def test_every_row_is_evaluated_once_including_partial_final_batch(self):
        rows = [dict(question=str(i), ground_truth_answer=str(i)) for i in range(3)]
        rollout = SimpleNamespace(sample=Mock(side_effect=[
            [([0, 9], True), ([1, 8], False)], [([2, 9], True)]]))
        tokenizer = SimpleNamespace(decode=lambda ids, **kw: rf'\boxed{{{ids[0]}}}')
        def score(texts, answers, finished):
            return [correctness(t, a, done)[1] for t, a, done in zip(texts, answers, finished)], 0
        predictions = StringIO()
        with patch('eval.format_prompt', side_effect=lambda t, q: q):
            metrics = evaluate(rows, rollout, tokenizer, SimpleNamespace(score=score),
                               predictions, batch_size=2)
        records = [json.loads(line) for line in predictions.getvalue().splitlines()]
        self.assertEqual([row['index'] for row in records], [0, 1, 2])
        self.assertEqual([row['correct'] for row in records], [True, False, True])
        self.assertEqual(metrics['total'], 3)
        self.assertEqual(metrics['correct'], 2)
        self.assertEqual(metrics['accuracy'], 2/3)
        self.assertEqual(metrics['truncated_rate'], 1/3)
        self.assertEqual(metrics['mean_tokens'], 2)
        self.assertTrue(all(call.kwargs == {'do_sample': False} for call in rollout.sample.call_args_list))

    def test_grading_timeouts_are_reported_as_incorrect(self):
        row = dict(question='q', ground_truth_answer='1')
        rollout = SimpleNamespace(sample=lambda *a, **kw: [([1], True)])
        tokenizer = SimpleNamespace(decode=lambda *a, **kw: r'\boxed{1}')
        reward = SimpleNamespace(score=lambda *a: ([0.], 1))
        with patch('eval.format_prompt', return_value='q'):
            metrics = evaluate([row], rollout, tokenizer, reward, StringIO())
        self.assertEqual(metrics['grading_timeouts'], 1)
        self.assertEqual(metrics['accuracy'], 0.)
