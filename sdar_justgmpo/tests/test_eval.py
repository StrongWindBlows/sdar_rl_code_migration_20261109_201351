from io import StringIO
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from eval import evaluate, merge_evaluations
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

    def test_shards_preserve_question_indices_seeds_and_weighted_metrics(self):
        rows = [dict(question=str(i),ground_truth_answer=str(i)) for i in range(5)]
        tokenizer = SimpleNamespace(decode=lambda ids,**kw: str(ids[0]))
        reward = SimpleNamespace(score=lambda texts,answers,finished: ([float(int(x)%2==0) for x in texts],0))
        def sample(prompts,group_size,seed,**kwargs):
            self.assertEqual(seed,113+int(prompts[0]))
            return [([int(p)]*(int(p)+1),int(p)!=3) for p in prompts]
        with tempfile.TemporaryDirectory() as directory, patch('eval.format_prompt',side_effect=lambda t,q:q):
            output = Path(directory)
            for shard in range(2):
                start,stop = len(rows)*shard//2,len(rows)*(shard+1)//2
                path = output/f'shard-{shard}'; path.mkdir()
                with (path/'predictions.jsonl').open('w') as predictions:
                    metrics = evaluate(rows[start:stop],SimpleNamespace(sample=sample),tokenizer,reward,
                        predictions,batch_size=2,start_index=start)
                (path/'metrics.json').write_text(json.dumps(metrics|dict(shard_index=shard,elapsed_seconds=10+shard)))
            merged = merge_evaluations(output,2)
            self.assertEqual(merged['total'],5)
            self.assertEqual(merged['correct'],3)
            self.assertEqual(merged['accuracy'],.6)
            self.assertEqual(merged['mean_tokens'],3)
            self.assertEqual(merged['truncated_rate'],.2)
            self.assertEqual(merged['elapsed_seconds'],11)
            records = [json.loads(l) for l in (output/'predictions.jsonl').read_text().splitlines()]
            self.assertEqual([r['index'] for r in records],list(range(5)))
            self.assertNotIn('shard_index',merged)
            part = output/'shard-1/predictions.jsonl'
            part.write_text(part.read_text().replace('"index": 2','"index": 1'))
            with self.assertRaisesRegex(ValueError,'exactly once'):
                merge_evaluations(output,2)
