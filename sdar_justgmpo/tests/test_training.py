import json
import os
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from algorithm import Response
from reward import correctness
from train import evaluate_checkpoint, minibatches, make_responses


class TrainingTests(unittest.TestCase):
    def test_minibatches_cover_every_response_once(self):
        records = [Response([1],[i],0) for i in range(16)]
        batches = list(minibatches(records,4,1234))
        self.assertEqual([len(b) for b in batches],[4,4,4,4])
        self.assertEqual(sorted(r.tokens[0] for b in batches for r in b),list(range(16)))
        self.assertNotEqual([r.tokens[0] for r in batches[0]],list(range(4)))

    def test_rewards_and_groups_match_prompt_major_sampling(self):
        rows = [dict(question='a',ground_truth_answer='1'),dict(question='b',ground_truth_answer='2')]
        samples = [([i,9],True) for i in range(6)]
        rollout = SimpleNamespace(sample=lambda *a:samples)
        tokenizer = SimpleNamespace(decode=lambda t,**kw:str(t[0]),encode=lambda p,**kw:[1 if p=='a' else 2])
        reward = SimpleNamespace(score=lambda *a:([1.,0.,0.,1.,1.,1.],0))
        with patch('train.format_prompt',side_effect=lambda tokenizer,q:q):
            records,timeouts = make_responses(rows,rollout,tokenizer,reward,3,1234,1.)
        self.assertEqual([r.prompt for r in records],[[1]]*3+[[2]]*3)
        self.assertGreater(records[0].advantage,0)
        self.assertLess(records[1].advantage,0)
        self.assertTrue(all(r.advantage==0 for r in records[3:]))
        self.assertEqual(timeouts,0)

    def test_reward_matches_gmpo_and_avoids_inherited_variable_deletion(self):
        self.assertEqual(correctness(r'\boxed{\frac{mh}{-1}}',r'\frac{hm}{m-1}')[1],0)
        self.assertEqual(correctness(r'\boxed{\frac{hm}{m-1}}',r'\frac{hm}{m-1}')[1],1)
        self.assertEqual(correctness(r'\boxed{\frac{1}{2}}','0.5')[1],1)
        self.assertEqual(correctness(r'\boxed{2}','2',finished=False)[1],0)
        self.assertEqual(correctness('unfinished reasoning','2')[1],0)


class CheckpointEvaluationTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.checkpoint = Path(directory.name)/'checkpoint-1'
        self.checkpoint.mkdir()
        (self.checkpoint/'weights.bin').write_bytes(b'checkpoint weights')
        self.output = self.checkpoint/'eval'
        self.args = SimpleNamespace(eval_data='/data/test.jsonl', eval_max_tokens=3000,
            eval_batch_size=32, eval_seed=113, tp_size=2, block_size=32,
            inference_cache_fraction=0.25,inference_cuda_graph=False,
            inference_prefix_caching=True,inference_max_prefill_tokens=8192,
            inference_retries=0, inference_progress_timeout=600)
        self.parallel = SimpleNamespace(rank=0, world=2, tp_rank=0, dp_rank=0, dp_size=1,
                                        control_group=object(), gpu_group=lambda: '4,5',
                                        failures=lambda error: [error] if error else [])
        self.metrics = dict(accuracy=0.5, correct=1, total=2, mean_tokens=32,
                            truncated_rate=0, grading_timeouts=0)
        for target in ('train.torch.cuda.empty_cache', 'train.dist.barrier',
                       'train.dist.broadcast_object_list'):
            patcher = patch(target)
            setattr(self, target.rsplit('.', 1)[-1], patcher.start())
            self.addCleanup(patcher.stop)

    def worker(self, *args, **kwargs):
        # A fresh worker must be able to write after an interrupted evaluation.
        self.assertFalse((self.output/'predictions.jsonl').exists())
        self.assertFalse((self.output/'metrics.json').exists())
        (self.output/'predictions.jsonl').write_text('{}\n')
        (self.output/'metrics.json').write_text(json.dumps(self.metrics))

    def test_saved_weights_are_evaluated_on_first_tp_group_in_clean_environment(self):
        inherited = dict(CUDA_VISIBLE_DEVICES='4', RANK='0', WORLD_SIZE='4',
                         MASTER_PORT='12345', TORCHELASTIC_RUN_ID='test')
        with patch.dict(os.environ, inherited), patch('train.run_inference',
                                                      side_effect=self.worker) as launch:
            evaluate_checkpoint(self.checkpoint, self.args, self.parallel)
            command = launch.call_args.args[0]
            flags = dict(zip(command[2::2], command[3::2]))
            self.assertEqual(flags['--model'], str(self.checkpoint))
            self.assertEqual(flags['--data'], self.args.eval_data)
            self.assertEqual(flags['--tp-size'], '2')
            self.assertEqual(flags['--block-size'], '32')
            self.assertEqual(flags['--max-tokens'], '3000')
            self.assertEqual(flags['--batch-size'], '32')
            self.assertEqual(flags['--seed'], '113')
            self.assertEqual(flags['--inference-cache-fraction'], '0.25')
            self.assertEqual(flags['--inference-max-prefill-tokens'], '8192')
            self.assertIn('--no-inference-cuda-graph', command)
            self.assertIn('--inference-prefix-caching', command)
            env = launch.call_args.kwargs['env']
            self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '4,5')
            self.assertFalse({'RANK', 'WORLD_SIZE', 'MASTER_PORT', 'TORCHELASTIC_RUN_ID'} & env.keys())
            self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'], '4')
        self.barrier.assert_called_once_with(group=self.parallel.control_group)
        self.broadcast_object_list.assert_called_once_with(
            [None], src=0, group=self.parallel.control_group)

    def test_completed_evaluation_is_reused(self):
        self.output.mkdir()
        (self.output/'metrics.json').write_text(json.dumps(self.metrics))
        with patch('train.run_inference') as launch:
            evaluate_checkpoint(self.checkpoint, self.args, self.parallel)
        launch.assert_not_called()

    def test_failed_evaluation_preserves_checkpoint_and_can_be_retried(self):
        self.output.mkdir()
        (self.output/'metrics.json').write_text('{')
        (self.output/'predictions.jsonl').write_text('interrupted result')
        with patch('train.run_inference', side_effect=RuntimeError('Inference exited 1')):
            with self.assertRaisesRegex(RuntimeError, 'Inference exited 1'):
                evaluate_checkpoint(self.checkpoint, self.args, self.parallel)
        self.assertEqual((self.checkpoint/'weights.bin').read_bytes(), b'checkpoint weights')
        with patch('train.run_inference', side_effect=self.worker):
            evaluate_checkpoint(self.checkpoint, self.args, self.parallel)
        self.assertEqual(json.loads((self.output/'metrics.json').read_text()), self.metrics)

    def test_other_ranks_receive_failure_without_starting_an_engine(self):
        self.parallel.rank = 1
        self.parallel.tp_rank = 1
        def failure(value, **kwargs):
            value[0] = 'evaluation worker failed'
        self.broadcast_object_list.side_effect = failure
        with patch('train.run_inference') as launch:
            with self.assertRaisesRegex(RuntimeError, 'evaluation worker failed'):
                evaluate_checkpoint(self.checkpoint, self.args, self.parallel)
        launch.assert_not_called()

    def test_timeout_retries_with_graph_disabled_and_same_generation_settings(self):
        self.args.inference_retries = 1
        self.args.inference_cuda_graph = True
        calls = []
        def worker(command, **kwargs):
            calls.append(list(command))
            if len(calls) == 1:
                (self.output/'predictions.jsonl').write_text('partial')
                raise TimeoutError('no progress')
            self.worker(command, **kwargs)
        with patch('train.run_inference', side_effect=worker):
            evaluate_checkpoint(self.checkpoint, self.args, self.parallel)
        self.assertIn('--inference-cuda-graph', calls[0])
        self.assertEqual([x.replace('--inference-cuda-graph', '--no-inference-cuda-graph')
                          for x in calls[0]], calls[1])

    def test_base_evaluation_writes_only_to_the_run_directory(self):
        self.output = self.checkpoint.parent/'eval-step0'
        with patch('train.run_inference', side_effect=self.worker):
            evaluate_checkpoint(self.checkpoint, self.args, self.parallel, self.output)
        self.assertFalse((self.checkpoint/'eval').exists())
        self.assertTrue((self.output/'metrics.json').is_file())

    def test_each_dp_leader_evaluates_its_shard_and_only_rank_zero_merges(self):
        root_output = self.output
        self.parallel.world, self.parallel.dp_size = 4, 2
        for rank, dp_rank, gpus in [(2,1,'6,7'), (0,0,'4,5')]:
            self.parallel.rank, self.parallel.dp_rank = rank, dp_rank
            self.parallel.gpu_group = lambda: gpus
            self.output = root_output/f'shard-{dp_rank}'
            with patch('train.run_inference',side_effect=self.worker) as launch, \
                 patch('train.merge_evaluations',return_value=self.metrics) as merge:
                evaluate_checkpoint(self.checkpoint,self.args,self.parallel)
                command = launch.call_args.args[0]
                self.assertEqual(command[command.index('--num-shards')+1],'2')
                self.assertEqual(command[command.index('--shard-index')+1],str(dp_rank))
                self.assertEqual(command[command.index('--output')+1],str(self.output))
                self.assertEqual(launch.call_args.kwargs['env']['CUDA_VISIBLE_DEVICES'],gpus)
                if rank == 0:
                    merge.assert_called_once_with(root_output,2)
                else:
                    merge.assert_not_called()

    def test_shard_failure_is_reported_before_merge(self):
        self.parallel.dp_size = 2
        self.parallel.failures = lambda error: ['DP 1 evaluation failed']
        with patch('train.run_inference',side_effect=RuntimeError('engine failed')), \
             patch('train.merge_evaluations') as merge:
            with self.assertRaisesRegex(RuntimeError,'DP 1 evaluation failed'):
                evaluate_checkpoint(self.checkpoint,self.args,self.parallel)
        merge.assert_not_called()
