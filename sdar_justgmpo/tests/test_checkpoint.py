"""Focused checks for reused full-state recovery and the offline W&B adaptation."""
import argparse
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from checkpointing import (save_full_checkpoint, restore_full_checkpoint, read_metadata,
                           apply_resume_config, validate_resume)
from run_state import resume_wandb


class CheckpointTests(unittest.TestCase):
    def test_resume_can_tune_runtime_settings_and_inherits_omitted_ones(self):
        saved = dict(model='model',data='data',batch_prompts=128,group_size=8,minibatch_size=128,
            ppo_epochs=1,max_tokens=3000,lr=5e-6,temperature=1.,seed=1234,clip=.4,objective='gmpo',
            block_size=32,tp_size=2,steps=30,rollout_batch_size=1,inference_cache_fraction=.06,
            inference_cuda_graph=True,inference_prefix_caching=False,inference_max_prefill_tokens=4096,train_tf32=False,
            eval_data='test.jsonl',eval_max_tokens=3000,eval_batch_size=1,eval_seed=113,train_phase_batch_size=1)
        metadata = dict(config=saved,completed_steps=10)
        tuning = dict(rollout_batch_size=32,inference_cache_fraction=.3,inference_cuda_graph=False,
                      inference_prefix_caching=True,inference_max_prefill_tokens=8192,eval_batch_size=32,
                      train_tf32=True,train_phase_batch_size=4)
        args = argparse.Namespace(**(saved | tuning))
        explicit = (set(tuning)-{'inference_cuda_graph'}) | {'no_inference_cuda_graph'}
        apply_resume_config(args,metadata,explicit)
        for key,value in tuning.items():
            self.assertEqual(getattr(args,key),value)
        self.assertEqual(args.lr,saved['lr'])
        self.assertEqual(args.eval_data,saved['eval_data'])
        apply_resume_config(args,metadata,set())
        for key in tuning:
            self.assertEqual(getattr(args,key),saved[key])
        # Older checkpoints need not contain newly introduced engine options.
        legacy = saved.copy()
        legacy.pop('inference_cache_fraction')
        legacy.pop('train_phase_batch_size')
        args.inference_cache_fraction = .3
        args.train_phase_batch_size = 4
        apply_resume_config(args,dict(config=legacy,completed_steps=10),set())
        self.assertEqual(args.inference_cache_fraction,.3)
        self.assertEqual(args.train_phase_batch_size,4)

    def test_real_entityless_offline_path(self):
        identity = dict(id='gjz7p1b3', path='sdar-justgmpo/gjz7p1b3')
        wb = SimpleNamespace(init=Mock())
        with tempfile.TemporaryDirectory() as directory, patch.dict('os.environ', WANDB_MODE='offline'):
            (Path(directory)/'wandb.json').write_text(json.dumps(identity))
            resume_wandb(wb, directory, dict(wandb_parent=identity, config={}))
            self.assertEqual(wb.init.call_args.kwargs['project'], 'sdar-justgmpo')
            self.assertIsNone(wb.init.call_args.kwargs['entity'])
            self.assertEqual(wb.init.call_args.kwargs['id'], identity['id'])
            with patch.dict('os.environ', WANDB_MODE='online'):
                with self.assertRaisesRegex(ValueError, 'WANDB_MODE=offline'):
                    resume_wandb(wb, directory, dict(wandb_parent=identity, config={}))

    def test_old_fingerprints_do_not_block_resume_but_layout_and_cursor_still_matter(self):
        metadata = dict(world_size=4, tp_size=2, completed_steps=3, next_data_index=24,
                        config={'batch_prompts': 8}, fingerprint={'source': 'old', 'environment': 'old'})
        validate_resume(metadata, 4, 2)
        with self.assertRaisesRegex(ValueError, 'number of training ranks'):
            validate_resume(metadata, 2, 2)
        with self.assertRaisesRegex(ValueError, 'TP/DP layout'):
            validate_resume(metadata, 4, 1)
        with self.assertRaisesRegex(ValueError, 'data cursor'):
            validate_resume(metadata | {'next_data_index': 0}, 4, 2)

    def test_fp32_weights_and_adam_continue_from_round_boundary(self):
        config = dict(model='model',data='data',batch_prompts=8,group_size=8,minibatch_size=32,
            ppo_epochs=1,max_tokens=256,lr=.001,temperature=1.,seed=1234,clip=.4,objective='gmpo',
            rollout_batch_size=8,block_size=32,steps=2)
        torch.manual_seed(71)
        model = torch.nn.Linear(4,2)
        optimizer = torch.optim.AdamW(model.parameters(),lr=.001)
        x = torch.randn(3,4)
        def update(m,o):
            o.zero_grad(); m(x).square().mean().backward(); o.step()
        update(model,optimizer)
        with tempfile.TemporaryDirectory() as directory, patch('torch.cuda.is_available',return_value=False):
            path = Path(directory)/'checkpoint-1'
            save_full_checkpoint(path,model,optimizer,config=config,completed_steps=1,
                nonzero_updates=2,elapsed_seconds=1.)
            update(model,optimizer)
            expected = copy.deepcopy(model.state_dict())
            other = torch.nn.Linear(4,2)
            other_optimizer = torch.optim.AdamW(other.parameters(),lr=.5)
            restore_full_checkpoint(path,other,other_optimizer)
            update(other,other_optimizer)
            for key,value in expected.items():
                torch.testing.assert_close(other.state_dict()[key],value,rtol=0,atol=0)
            meta = read_metadata(path)
            self.assertNotIn('fingerprint', meta)
            # Metadata from older code is accepted without comparing its fingerprints.
            meta['fingerprint'] = {'source': 'different code', 'environment': 'different versions'}
            (path/'resume.json').write_text(json.dumps(meta))
            validate_resume(read_metadata(path), 1, 1)
            self.assertEqual(meta['next_data_index'],8)
            self.assertEqual(meta['nonzero_updates'],2)
            args = argparse.Namespace(**(config | {'steps':3,'clip':.9}))
            apply_resume_config(args,meta,{'steps'})
            self.assertEqual((args.steps,args.clip,args.minibatch_size),(3,.4,32))
            self.assertEqual(args.block_size,32)
            with self.assertRaisesRegex(ValueError,'Cannot change --block-size'):
                apply_resume_config(argparse.Namespace(**(config | {'block_size':4})),meta,{'block_size'})

    def test_offline_resume_preserves_identity_without_claiming_online_resume(self):
        identity = dict(id='abc',path='entity/project/abc')
        run = SimpleNamespace(id='abc',resumed=False)
        wb = SimpleNamespace(init=Mock(return_value=run))
        with tempfile.TemporaryDirectory() as directory, patch.dict('os.environ',WANDB_MODE='offline'):
            (Path(directory)/'wandb.json').write_text(json.dumps(identity))
            self.assertIs(resume_wandb(wb,directory,dict(wandb_parent=identity,config={})),run)
            self.assertEqual(wb.init.call_args.kwargs['id'],'abc')
            self.assertEqual(wb.init.call_args.kwargs['mode'],'offline')
            self.assertNotIn('resume',wb.init.call_args.kwargs)
