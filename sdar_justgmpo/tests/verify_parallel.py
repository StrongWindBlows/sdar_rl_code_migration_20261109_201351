"""Run with torchrun --standalone --nproc_per_node=4 for TP=2/DP=2 parity."""
import copy
from functools import partial
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import types
from types import SimpleNamespace

os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
kernel = types.ModuleType('flash_attn.ops.triton.layer_norm')
kernel.rms_norm_fn = lambda *a,**kw: None
sys.modules[kernel.__name__] = kernel

import torch
import torch.distributed as dist
import torch.nn.functional as F

from algorithm import Response, cache_old_logps, backward_response, backward_on_policy_grpo
import likelihood
from likelihood import SDARPolicy
from vendor.sdar import modeling_sdar as sdar
from vendor.sdar.configuration_sdar import SDARConfig
from parallel import Parallel
from checkpointing import save_full_checkpoint, restore_full_checkpoint, read_metadata


def dense_packed(n,block_size,device,phase_count=1):
    idx = torch.arange((phase_count+1)*n,device=device)
    q,k = idx[:,None],idx[None,:]
    qc,kc = q>=phase_count*n,k>=phase_count*n
    qb,kb = (q%n)//block_size,(k%n)//block_size
    return ((~qc&~kc&(q//n==k//n)&(qb==kb))|(~qc&kc&(kb<qb))|(qc&kc&(kb<=qb)))[None]


def main():
    torch.set_num_threads(1)
    dist.init_process_group('gloo')
    parallel = Parallel(2)
    assert parallel.dp_size == 2
    payload = {'replica': parallel.dp_rank, 'tokens': [1, 2, 3]} if parallel.tp_rank == 0 else None
    assert parallel.broadcast(payload) == {'replica': parallel.dp_rank, 'tokens': [1, 2, 3]}
    assert parallel.failures(None) == []
    assert parallel.failures('simulated stall' if parallel.rank == 2 else None) == ['simulated stall']
    sdar.flash_rms_norm = lambda x,weight,bias,eps: x*torch.rsqrt(x.square().mean(-1,keepdim=True)+eps)*weight
    sdar.liger_kernel_is_available = False
    sdar.fused_flex_attention = lambda query,key,value,attention_mask,scale,**kw: (
        F.scaled_dot_product_attention(query,key,value,attn_mask=attention_mask[:,None],scale=scale,enable_gqa=True),None)
    likelihood.packed_attention = dense_packed
    torch.manual_seed(71)
    cfg = SDARConfig(vocab_size=32,hidden_size=32,intermediate_size=64,num_hidden_layers=2,
        num_attention_heads=4,num_key_value_heads=2,head_dim=8,mask_token_id=31,
        block_size=32,use_cache=False,attention_dropout=0.)
    errors = {}
    for objective in ('gmpo','grpo','on_policy_grpo'):
        reference = SDARPolicy(cfg).double().train()
        model = parallel.shard_model(copy.deepcopy(reference))
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        records = [Response([1]*31,[6,7,2,9,10,11,12],1,True,1.3),
                   Response([2]*33,[7,8,2,10,11],0,True,-.7)]
        if objective == 'on_policy_grpo':
            backward = backward_on_policy_grpo
        else:
            backward = partial(backward_response,objective=objective)
            cache_old_logps(reference,records)
            for r in records:
                r.old_logps += torch.linspace(-.8,.8,len(r.tokens))
        local = copy.deepcopy(records[parallel.dp_rank])
        opt = torch.optim.AdamW(model.parameters(),lr=.001,foreach=False)
        ref_opt = torch.optim.AdamW(reference.parameters(),lr=.001,foreach=False)
        for r in records:
            backward(reference,r,len(records))
        backward(model,local,1,phase_batch_size=4)
        parallel.reduce_gradients(model)
        full_grad = parallel.full_state(SimpleNamespace(state_dict=lambda:{n:p.grad for n,p in model.named_parameters()}),
                                        dtype=torch.float64)
        gradient_error = 0.
        if parallel.tp_rank == 0:
            for n,p in reference.named_parameters():
                gradient_error = max(gradient_error,(full_grad[n]-p.grad).abs().max().item())
                torch.testing.assert_close(full_grad[n],p.grad,atol=2e-6,rtol=2e-5)
        errors[objective] = gradient_error
        ref_norm = torch.nn.utils.clip_grad_norm_(reference.parameters(),1.)
        norm = parallel.clip_grad_norm(model)
        torch.testing.assert_close(norm,ref_norm,atol=2e-6,rtol=2e-5)
        opt.step(); ref_opt.step()
        full = parallel.full_state(model,dtype=torch.float64)
        if parallel.tp_rank == 0:
            for n,p in reference.state_dict().items():
                torch.testing.assert_close(full[n],p,atol=2e-6,rtol=2e-5)
    directory = [tempfile.mkdtemp(prefix='sdar_tp_test_') if parallel.rank == 0 else None]
    dist.broadcast_object_list(directory,src=0)
    path = Path(directory[0])/'checkpoint-1'
    save_full_checkpoint(path,model,opt,config={'batch_prompts':2,'tp_size':2,'dp_size':2},
        completed_steps=1,nonzero_updates=1,elapsed_seconds=1.,parallel=parallel,
        export=lambda p,state: torch.save(state,p/'export.pt'))
    assert read_metadata(path)['tp_size'] == 2
    other = parallel.shard_model(SDARPolicy(cfg).double().train())
    other.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    other_opt = torch.optim.AdamW(other.parameters(),lr=.5,foreach=False)
    restore_full_checkpoint(path,other,other_opt,parallel.rank,parallel=parallel)
    for m,o in [(model,opt),(other,other_opt)]:
        o.zero_grad(set_to_none=True)
        backward(m,local,1,phase_batch_size=4)
        parallel.reduce_gradients(m)
        parallel.clip_grad_norm(m)
        o.step()
    for n,p in model.state_dict().items():
        torch.testing.assert_close(other.state_dict()[n],p,atol=0,rtol=0)
    dist.barrier()
    if parallel.rank == 0:
        shutil.rmtree(directory[0])
        print(json.dumps(dict(tp_size=2,dp_size=2,gradient_max_error=max(errors.values()),
                              gradient_errors=errors,phase_batch_size=4,
                              global_grad_norm=norm.item(),adam_update_matches_reference=True,
                              checkpoint_resume_exact=True,status='passed')),flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
