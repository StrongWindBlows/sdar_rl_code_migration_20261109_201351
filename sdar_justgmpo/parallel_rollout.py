"""Two GPUs per rollout replica; release each engine before learner updates.

Weights cross the learner/engine boundary through a complete BF16 export. This
is deliberately an explicit disk handoff, not an unimplemented TP IPC update.
"""
import json
from pathlib import Path
import shutil
import sys
import tempfile

import torch
import torch.distributed as dist

from runtime import save_checkpoint
from inference_process import inference_env, run_inference


class ParallelRollout:
    def __init__(self, parallel, tokenizer, args):
        self.parallel, self.tokenizer, self.args = parallel, tokenizer, args
        directory = tempfile.mkdtemp(prefix='.rollout-',dir=args.output) if parallel.rank == 0 else None
        shared = [directory]
        dist.broadcast_object_list(shared,src=0)
        self.directory = Path(shared[0])
        self.model_path = self.directory/'model'

    def update_weights(self, model, path=None):
        self.model_path = Path(path) if path is not None else self.directory/'model'
        error = None
        try:
            if self.parallel.dp_rank == 0:
                state = self.parallel.full_state(model)
                if self.parallel.tp_rank == 0:
                    save_checkpoint(model,self.tokenizer,self.model_path,state=state,
                                    tokenizer_source=self.args.model)
                del state
        except Exception as exc:
            error = repr(exc)
        errors = [None]*self.parallel.world
        dist.all_gather_object(errors,error)
        if any(errors):
            raise RuntimeError(f'Rollout export failed: {errors}')
        torch.cuda.empty_cache()

    def sample(self, prompts, group_size, seed, temperature=1.):
        if self.parallel.tp_rank != 0:
            raise RuntimeError('Only the TP leader starts its rollout replica')
        request = self.directory/f'request-dp{self.parallel.dp_rank}.json'
        response = self.directory/f'response-dp{self.parallel.dp_rank}.json'
        progress = self.directory/f'progress-dp{self.parallel.dp_rank}'
        config = dict(model=str(self.model_path),prompts=prompts,
            group_size=group_size,seed=seed,temperature=temperature,max_tokens=self.args.max_tokens,
            batch_size=self.args.rollout_batch_size,block_size=self.args.block_size,
            cache_fraction=self.args.inference_cache_fraction,cuda_graph=self.args.inference_cuda_graph,
            prefix_caching=self.args.inference_prefix_caching,
            max_prefill_tokens=self.args.inference_max_prefill_tokens,
            tp_size=self.parallel.tp_size,output=str(response),progress=str(progress))
        env = inference_env(self.parallel.gpu_group())
        log = Path(self.args.output)/f'rollout-engine-dp{self.parallel.dp_rank}.log'
        for attempt in range(self.args.inference_retries + 1):
            response.unlink(missing_ok=True)
            config['cuda_graph'] = self.args.inference_cuda_graph and attempt == 0
            request.write_text(json.dumps(config))
            try:
                run_inference([sys.executable, str(Path(__file__).with_name('rollout_worker.py')),
                               str(request)], env=env, log=log, progress=progress,
                              timeout=self.args.inference_progress_timeout)
                return [(ids,done) for ids,done in json.loads(response.read_text())]
            except (RuntimeError, TimeoutError) as exc:
                if attempt == self.args.inference_retries:
                    raise
                print(f'DP {self.parallel.dp_rank}: {exc}; retrying with CUDA graph disabled', flush=True)

    def close(self):
        dist.barrier()
        if self.parallel.rank == 0:
            shutil.rmtree(self.directory)
