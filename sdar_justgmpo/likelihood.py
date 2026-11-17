"""Exact SDAR LTR likelihood: native block attention, observed past, masked target.

Phase packing is inherited from the numerically verified SDAR JustGRPO adapter.
"""
from contextlib import nullcontext
from functools import lru_cache
from typing import NamedTuple
import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_block_mask
from vendor.sdar.modeling_sdar import SDARForCausalLM


def phase_inputs(response, block_size: int, mask_id: int, device, phase_batch_size=1,
                 *, reconstruct=False):
    """Yield packed phases, positions, targets, labels and number of noisy branches.

    Each group packs [noisy_0, ..., noisy_k, clean], sharing the clean branch.
    Pad to 128 for Flex Attention tile shapes. Within the last real block these
    are MASK context; subsequent blocks are causally invisible to all targets.
    reconstruct adds all valid phase-0 targets, without projecting its first
    token twice. The other phases and attention layout are unchanged.
    """
    p, t = len(response.prompt), len(response.tokens)
    n = ((p + t + 127) // 128) * 128
    clean = torch.full((n,), mask_id, dtype=torch.long, device=device)
    clean[:p+t] = torch.tensor(response.prompt + response.tokens, device=device)
    index = torch.arange(n, device=device)
    valid = (index >= p) & (index < p+t)
    first = torch.maximum(index // block_size * block_size, index.new_tensor(p))
    phase_index = index - first
    for start in range(0, block_size, phase_batch_size):
        noisy, targets, labels = [], [], []
        for phase in range(start, min(start + phase_batch_size, block_size)):
            target = torch.where(valid if reconstruct and phase == 0 else
                                 valid & (phase_index == phase))[0]
            if not target.numel():
                continue
            targets.append(target + len(noisy) * n)
            labels.append(clean[target])
            noisy.append(clean.masked_fill(valid & (phase_index >= phase), mask_id))
        if not noisy:
            continue
        yield (torch.cat(noisy + [clean])[None], index.repeat(len(noisy) + 1)[None],
               torch.cat(targets), torch.cat(labels), len(noisy))


@lru_cache(maxsize=64)
def packed_attention(n: int, block_size: int, device: str, phase_count=1):
    def visible(b, h, q, k):
        q_branch, k_branch = q // n, k // n
        q_clean, k_clean = q_branch == phase_count, k_branch == phase_count
        qb, kb = (q % n) // block_size, (k % n) // block_size
        noisy_self = (~q_clean) & (q_branch == k_branch) & (qb == kb)
        noisy_past = (~q_clean) & k_clean & (kb < qb)
        clean_past = q_clean & k_clean & (kb <= qb)
        return noisy_self | noisy_past | clean_past
    length = (phase_count + 1) * n
    return create_block_mask(visible, B=None, H=None, Q_LEN=length, KV_LEN=length, device=device)


class SDARPolicy(SDARForCausalLM):
    """Same weights/backbone; explicit likelihood path instead of diffusion loss."""

    def forward(self, input_ids, position_ids, target_positions, labels,
                active_vocab: int, temperature: float = 1.0, phase_count=1,
                *, logit_chunk_size=None):
        n = input_ids.shape[1] // (phase_count + 1)
        mask = packed_attention(n, self.config.block_size, str(input_ids.device), phase_count)
        # autocast does not cast embedding lookups. Keep residuals/RoPE in the
        # compute dtype while gradients flow back to the FP32 master embedding.
        embeds = self.model.embed_tokens(input_ids)
        if torch.is_autocast_enabled(input_ids.device.type):
            embeds = embeds.to(torch.get_autocast_dtype(input_ids.device.type))
        hidden = self.model(inputs_embeds=embeds, position_ids=position_ids,
                            attention_mask=mask, use_cache=False).last_hidden_state
        hidden = hidden[:, target_positions] if labels.ndim == 2 else hidden[0, target_positions]
        if logit_chunk_size is None:
            return self.token_logps(hidden, labels, active_vocab, temperature)
        parts = [self.token_logps(h, y, active_vocab, temperature) for h, y in zip(
            hidden.split(logit_chunk_size, dim=-2), labels.split(logit_chunk_size, dim=-1))]
        return tuple(torch.cat(values, dim=-1) for values in zip(*parts))

    def token_logps(self, hidden, labels, active_vocab, temperature):
        # Full vocabulary matches the rollout sampler.
        logits = F.linear(hidden, self.lm_head.weight[:active_vocab]).float() / temperature
        log_distribution = logits.log_softmax(-1)
        logp = log_distribution.gather(-1, labels[..., None]).squeeze(-1)
        with torch.no_grad():
            entropy = -(log_distribution.exp() * log_distribution).sum(-1)
        return logp, entropy


class ReconstructionPhase(NamedTuple):
    offsets: torch.Tensor
    logps: torch.Tensor
    entropy: torch.Tensor
    full_mask_offsets: torch.Tensor
    full_mask_logps: torch.Tensor
    full_mask_entropy: torch.Tensor


def reconstruction_phases(model, response, temperature=1.0, phase_batch_size=1,
                          logit_chunk_size=128):
    """Yield differentiable sequential and block-full-mask likelihoods together.

    Merge both losses before backward. Vocabulary chunks bound inference memory;
    differentiable callers still retain their graphs until backward.
    """
    device = next(model.parameters()).device
    p, block = len(response.prompt), model.config.block_size
    for group, (ids, positions, targets, labels, count) in enumerate(phase_inputs(
            response, block, model.config.mask_token_id, device, phase_batch_size,
            reconstruct=True)):
        context = torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' else nullcontext()
        with context:
            logps, entropy = model(ids, positions, targets, labels, model.config.vocab_size,
                                   temperature, count, logit_chunk_size=logit_chunk_size)
        absolute = positions[0, targets]
        first = torch.maximum(absolute // block * block, absolute.new_tensor(p))
        full = (targets < ids.shape[1] // (count + 1)) & (group == 0)
        sequential = ~full | (absolute == first)
        offsets = absolute - p
        yield ReconstructionPhase(offsets[sequential], logps[sequential], entropy[sequential],
                                  offsets[full], logps[full], entropy[full])

