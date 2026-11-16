"""Fresh process for each TP rollout, including adapters in spawned workers."""
import json
from pathlib import Path
import sys

import torch
from transformers import AutoTokenizer
# This import must run in multiprocessing's spawned children too: it installs
# the sequence-major sampling metadata correction before engine construction.
from rollout import Rollout
from inference_process import TokenProgress


def main():
    request = json.loads(Path(sys.argv[1]).read_text())
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(request['model'],local_files_only=True)
    engine = Rollout(request['model'],tokenizer,request['max_tokens'],request['batch_size'],
                     block_size=request['block_size'],tp_size=request['tp_size'],
                     cache_fraction=request['cache_fraction'],cuda_graph=request['cuda_graph'],
                     prefix_caching=request['prefix_caching'],max_prefill_tokens=request['max_prefill_tokens'])
    try:
        result = engine.sample(request['prompts'],request['group_size'],request['seed'],request['temperature'],
                               progress=TokenProgress(request['progress']))
        Path(request['output']).write_text(json.dumps(result)+'\n')
    finally:
        engine.close()


if __name__ == '__main__':
    main()
