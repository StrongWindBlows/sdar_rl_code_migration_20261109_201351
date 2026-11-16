"""Distributed reduction and BF16 export reused from sdar_justgrpo_audit/train.py."""
import json
import shutil
from pathlib import Path
import torch
import torch.distributed as dist
from reward import INSTRUCTION
ROOT = Path(__file__).resolve().parent


def sum_metrics(values, device='cuda'):
    x = torch.tensor(values, dtype=torch.float64, device=device)
    if dist.is_initialized():
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
    return x.tolist()


def sum_gradients(model):
    """Match DDP: average the accumulated local source gradients across ranks."""
    for p in model.parameters():
        if p.grad is None:
            p.grad = torch.zeros_like(p)
        if dist.is_initialized():
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
            p.grad.div_(dist.get_world_size())


def save_checkpoint(model, tokenizer, output, state=None, tokenizer_source=None):
    output.mkdir(parents=True, exist_ok=True)
    # Inference export is BF16; master weights and Adam moments stay FP32 in training.
    if state is None:
        state = {k: v.detach().to(device='cpu', dtype=torch.bfloat16) for k,v in model.state_dict().items()}
    model.config.architectures = ['SDARForCausalLM']
    model.config.auto_map = {'AutoConfig':'configuration_sdar.SDARConfig',
        'AutoModel':'modeling_sdar.SDARForCausalLM',
        'AutoModelForCausalLM':'modeling_sdar.SDARForCausalLM'}
    # Save the unwrapped policy directly. HF model.save_pretrained would import
    # an unrelated installed DeepSpeed package that requires a CUDA toolkit.
    from huggingface_hub import save_torch_state_dict
    save_torch_state_dict(state, output, max_shard_size='5GB')
    model.config.save_pretrained(output)
    model.generation_config.save_pretrained(output)
    saved_config = json.loads((output/'config.json').read_text())
    saved_config.update(architectures=['SDARForCausalLM'], torch_dtype='bfloat16')
    (output/'config.json').write_text(json.dumps(saved_config,indent=2))
    tokenizer.save_pretrained(output)
    if tokenizer_source is not None:
        from inference_files import inference_model_path
        with inference_model_path(tokenizer_source) as source:
            for file in source.glob('tokenization_*.py'):
                shutil.copy2(file,output/file.name)
            original = json.loads((source/'tokenizer_config.json').read_text())
            saved = json.loads((output/'tokenizer_config.json').read_text())
            if 'auto_map' in original:
                saved['auto_map'] = original['auto_map']
            else:
                saved.pop('auto_map',None)
            (output/'tokenizer_config.json').write_text(json.dumps(saved,indent=2)+'\n')
    for f in (ROOT/'vendor/sdar').glob('*.py'):
        if f.name != '__init__.py': shutil.copy2(f, output/f.name)


def format_prompt(tokenizer, question):
    return tokenizer.apply_chat_template([{'role':'user','content':question+INSTRUCTION}],
        tokenize=False, add_generation_prompt=True)

