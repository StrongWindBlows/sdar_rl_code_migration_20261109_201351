"""Head/MLP tensor parallelism and orthogonal data-parallel groups for SDAR.

Each TP group consumes identical responses. Attention/MLP matrices are sharded;
embeddings, the vocabulary head and norms are replicated. Q/K norm gradients
are partial over heads and must be summed across TP before DP averaging.
"""
import os
from collections import OrderedDict
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn


# A full rollout or evaluation can generate over a million tokens at batch size 1.
WORK_TIMEOUT = timedelta(hours=24)


class _CopyToTP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        return x

    @staticmethod
    def backward(ctx, grad):
        grad = grad.contiguous().clone()
        dist.all_reduce(grad, group=ctx.group)
        return grad, None


class _ReduceFromTP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, group):
        out = x.contiguous().clone()
        dist.all_reduce(out, group=group)
        return out

    @staticmethod
    def backward(ctx, grad):
        return grad, None


class TPLinear(nn.Module):
    def __init__(self, linear, dim, parallel):
        super().__init__()
        if linear.weight.shape[dim] % parallel.tp_size:
            raise ValueError('TP size must divide the sharded linear dimension')
        self.dim, self.group = dim, parallel.tp_group
        self.weight = nn.Parameter(linear.weight.detach().chunk(parallel.tp_size, dim=dim)
                                   [parallel.tp_rank].clone().contiguous())
        if linear.bias is None:
            self.register_parameter('bias', None)
        else:
            bias = linear.bias.detach()
            if dim == 0:
                bias = bias.chunk(parallel.tp_size)[parallel.tp_rank]
            self.bias = nn.Parameter(bias.clone())
        self.out_features, self.in_features = self.weight.shape

    def forward(self, x):
        if self.dim == 0:
            return F.linear(_CopyToTP.apply(x, self.group), self.weight, self.bias)
        out = _ReduceFromTP.apply(F.linear(x, self.weight), self.group)
        return out if self.bias is None else out + self.bias


class Parallel:
    def __init__(self, tp_size=1):
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world = dist.get_world_size() if dist.is_initialized() else 1
        if tp_size < 1 or self.world % tp_size:
            raise ValueError('WORLD_SIZE must be divisible by --tp-size')
        self.tp_size, self.dp_size = tp_size, self.world // tp_size
        self.tp_rank, self.dp_rank = self.rank % tp_size, self.rank // tp_size
        self.tp_leader = self.dp_rank * tp_size
        self.tp_group = self.dp_group = self.control_group = None
        self.shard_dims = {}
        if self.world > 1:
            self.control_group = dist.new_group(backend='gloo', timeout=WORK_TIMEOUT)
            for dp in range(self.dp_size):
                ranks = list(range(dp*tp_size, (dp+1)*tp_size))
                group = dist.new_group(ranks, timeout=WORK_TIMEOUT)
                if self.rank in ranks:
                    self.tp_group = group
            for tp in range(tp_size):
                ranks = list(range(tp, self.world, tp_size))
                group = dist.new_group(ranks, timeout=WORK_TIMEOUT)
                if self.rank in ranks:
                    self.dp_group = group

    def shard_model(self, model):
        if self.tp_size == 1:
            return model
        config = model.config
        if config.num_attention_heads % self.tp_size or config.num_key_value_heads % self.tp_size:
            raise ValueError('TP size must divide both attention and KV head counts')
        for i, layer in enumerate(model.model.layers):
            for component, names, dim in (
                ('self_attn', ('q_proj','k_proj','v_proj'), 0),
                ('self_attn', ('o_proj',), 1),
                ('mlp', ('gate_proj','up_proj'), 0),
                ('mlp', ('down_proj',), 1),
            ):
                parent = getattr(layer, component)
                for name in names:
                    linear = TPLinear(getattr(parent,name), dim, self)
                    setattr(parent,name,linear)
                    prefix = f'model.layers.{i}.{component}.{name}'
                    self.shard_dims[prefix+'.weight'] = dim
                    if linear.bias is not None and dim == 0:
                        self.shard_dims[prefix+'.bias'] = 0
        return model

    def broadcast(self, value):
        # Every inference replica must exit before any learner starts NCCL work.
        # A CPU barrier leaves the GPUs available to the remaining subprocesses.
        if self.world > 1:
            dist.barrier(group=self.control_group)
        if self.tp_size == 1:
            return value
        objects = [value if self.tp_rank == 0 else None]
        dist.broadcast_object_list(objects, src=self.tp_leader, group=self.tp_group)
        return objects[0]

    def failures(self, error):
        """Agree on failure across DP replicas before any learner updates."""
        errors = [error]
        if self.world > 1:
            errors = [None]*self.world
            dist.all_gather_object(errors, error, group=self.control_group)
        return list(dict.fromkeys(e for e in errors if e is not None))

    @torch.no_grad()
    def reduce_gradients(self, model):
        for name, p in model.named_parameters():
            if p.grad is None:
                p.grad = torch.zeros_like(p)
            if self.tp_size > 1 and ('.self_attn.q_norm.' in name or '.self_attn.k_norm.' in name):
                dist.all_reduce(p.grad, group=self.tp_group)
            if self.dp_size > 1:
                dist.all_reduce(p.grad, group=self.dp_group)
                p.grad.div_(self.dp_size)

    @torch.no_grad()
    def clip_grad_norm(self, model, maximum=1.):
        first = next(model.parameters())
        total = torch.zeros((), device=first.device, dtype=torch.float64)
        for name, p in model.named_parameters():
            if p.grad is not None and (name in self.shard_dims or self.tp_rank == 0):
                total += torch.linalg.vector_norm(p.grad, ord=2).double().square()
        if self.tp_size > 1:
            dist.all_reduce(total, group=self.tp_group)
        norm = total.sqrt()
        if not torch.isfinite(norm):
            raise FloatingPointError('Nonfinite global gradient norm')
        scale = (maximum/(norm+1e-6)).clamp(max=1).to(first.dtype)
        for p in model.parameters():
            if p.grad is not None:
                p.grad.mul_(scale)
        return norm

    def sum_metrics(self, values, device='cuda'):
        x = torch.tensor(values, dtype=torch.float64, device=device)
        if self.dp_size > 1:
            dist.all_reduce(x, group=self.dp_group)
        return x.tolist()

    @torch.no_grad()
    def full_state(self, model, dtype=torch.bfloat16):
        """All TP ranks participate; only the leader retains the CPU state."""
        state = OrderedDict() if self.tp_rank == 0 else None
        for name, value in model.state_dict().items():
            value = value.detach().to(dtype=dtype).contiguous()
            if name in self.shard_dims:
                shards = [torch.empty_like(value) for _ in range(self.tp_size)]
                dist.all_gather(shards, value, group=self.tp_group)
                if self.tp_rank == 0:
                    state[name] = torch.cat([s.cpu() for s in shards], dim=self.shard_dims[name])
                del shards
            elif self.tp_rank == 0:
                state[name] = value.cpu()
        return state

    def gpu_group(self):
        # Captured before train.py narrows each learner to its own visible GPU.
        visible = os.environ['SDAR_VISIBLE_DEVICES'].split(',')
        local_rank = int(os.environ.get('SDAR_LOCAL_RANK','0'))
        start = local_rank - self.tp_rank
        devices = visible[start:start+self.tp_size]
        if len(devices) != self.tp_size:
            raise ValueError('TP group must fit entirely on the current node')
        return ','.join(devices)
