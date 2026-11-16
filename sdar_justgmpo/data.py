"""Load raw MATH JSONL or the trainer's existing JSON records."""
import json
from pathlib import Path
import re
import warnings

from vendor.gmpo.math_grader import extract_boxed_answer


def load_math(path, *, training=False):
    path = Path(path)
    if path.suffix == '.jsonl':
        records = [(i, json.loads(line)) for i, line in enumerate(path.read_text().splitlines(), 1)
                   if line.strip()]
    else:
        records = enumerate(json.loads(path.read_text()), 1)
    rows = []
    for index, row in records:
        raw = 'problem' in row
        question = row['problem'] if raw else row['question']
        if raw:
            # MATH also contains unbraced LaTeX such as "\boxed 9$".
            solution = re.sub(r'\\boxed\s+([^$]+)(?=\$)', r'\\boxed{\1}', row['solution'])
            answer = extract_boxed_answer(solution)
        else:
            answer = row['ground_truth_answer']
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f'{path}:{index}: empty or invalid question')
        if training and raw and answer == '':
            warnings.warn(f'{path}:{index}: skipping training example with empty boxed answer')
            continue
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError(f'{path}:{index}: missing or invalid ground-truth answer')
        # Legacy JSON records are already prepared; preserve their order and selection.
        if training and raw and len(question) >= 1500:
            continue
        rows.append(dict(question=question, ground_truth_answer=answer))
    if not rows:
        raise ValueError(f'{path}: no examples')
    return rows
