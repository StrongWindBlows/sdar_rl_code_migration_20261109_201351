"""GMPO reference boxed reward: correct=1, incorrect/unformatted/truncated=0."""
from multiprocessing import get_context, TimeoutError

from vendor.gmpo.math_grader import boxed_reward_fn

INSTRUCTION = r'(Please put the final answer in \boxed{} tag, i.e. $\boxed{answer here}$)'


def correctness(text, answer, finished=True):
    if not finished:
        return {'formatted': False, 'truncated': True}, 0.0
    return boxed_reward_fn(text, answer, fast=False)


class Reward:
    """Same two-worker / one-second grading protocol as GMPO's MATHOracle."""
    def __init__(self):
        # Spawn keeps grading workers independent of CUDA initialized by the trainer.
        self.pool = get_context('spawn').Pool(2)

    def score(self, texts, answers, finished):
        scores, timeouts = [], 0
        for text, answer, done in zip(texts, answers, finished):
            if not done:
                scores.append(0.0)
                continue
            result = self.pool.apply_async(correctness, (text, answer))
            try:
                _, value = result.get(timeout=1)
            except TimeoutError:
                value = 0.0
                timeouts += 1
            scores.append(value)
        return scores, timeouts

    def close(self):
        self.pool.terminate()
        self.pool.join()
