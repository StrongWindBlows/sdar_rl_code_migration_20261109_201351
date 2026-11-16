import json
from pathlib import Path
import tempfile
import unittest

from data import load_math


class DataTests(unittest.TestCase):
    def test_raw_data_extracts_last_nested_box_and_filters_training_only(self):
        rows = [dict(problem='short', solution=r'First \boxed{1}, then \boxed{\frac{2}{3}}.'),
                dict(problem='x'*1500, solution=r'\boxed{4}')]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'math.jsonl'
            path.write_text('\n'.join(json.dumps(row) for row in rows)+'\n')
            train, test = load_math(path, training=True), load_math(path)
        self.assertEqual(train, [dict(question='short', ground_truth_answer=r'\frac{2}{3}')])
        self.assertEqual(len(test), 2)

    def test_prepared_json_preserves_answers_and_rows(self):
        rows = [dict(question='x'*1600, ground_truth_answer='0.5')]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'math.json'
            path.write_text(json.dumps(rows))
            self.assertEqual(load_math(path, training=True), rows)

    def test_missing_box_reports_the_source_row(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'math.jsonl'
            path.write_text(json.dumps(dict(problem='q', solution=r'\boxed{1}'))+'\n'+
                            json.dumps(dict(problem='bad', solution='no boxed answer'))+'\n')
            with self.assertRaisesRegex(ValueError, 'math.jsonl:2:.*ground-truth'):
                load_math(path)

    def test_unbraced_math_answers_and_empty_training_answers(self):
        rows = [dict(problem='valid', solution=r'The answer is $\boxed 9$.'),
                dict(problem='missing', solution=r'$\boxed{}$')]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'math.jsonl'
            path.write_text('\n'.join(json.dumps(row) for row in rows))
            with self.assertWarnsRegex(UserWarning, 'math.jsonl:2: skipping'):
                self.assertEqual(load_math(path, training=True),
                                 [dict(question='valid', ground_truth_answer='9')])
            with self.assertRaisesRegex(ValueError, 'math.jsonl:2:.*ground-truth'):
                load_math(path)
