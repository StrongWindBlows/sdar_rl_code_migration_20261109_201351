"""Exact SDAR LTR likelihood: native block attention, observed past, masked target.

Phase packing is inherited from the numerically verified SDAR JustGRPO adapter.
"""
from functools import lru_cache
import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_block_mask
from vendor.sdar.modeling_sdar import SDARForCausalLM


def phase_inputs(response, block_size: int, mask_id: int, device):
    """Yield [noisy, clean], repeated positions, target positions and labels.

    Pad to 128 for Flex Attention tile shapes. Within the last real block these
    are MASK context; subsequent blocks are causally invisible to all targets.
    """
    p, t = len(response.prompt), len(response.tokens)
    n = ((p + t + 127) // 128) * 128
    clean = torch.full((n,), mask_id, dtype=torch.long, device=device)
    clean[:p+t] = torch.tensor(response.prompt + response.tokens, device=device)
    index = torch.arange(n, device=device)
    valid = (index >= p) & (index < p+t)
    first = torch.maximum(index // block_size * block_size, index.new_tensor(p))
    phase_index = index - first
    for phase in range(block_size):
        target = valid & (phase_index == phase)
        if not target.any():
            continue
        masked = valid & (phase_index >= phase)
        noisy = clean.masked_fill(masked, mask_id)
        yield (torch.cat([noisy, clean])[None], index.repeat(2)[None],
               torch.where(target)[0], clean[target])


@lru_cache(maxsize=64)
def packed_attention(n: int, block_size: int, device: str):
    def visible(b, h, q, k):
        q_clean, k_clean = q >= n, k >= n
        qb, kb = (q % n) // block_size, (k % n) // block_size
        noisy_self = (~q_clean) & (~k_clean) & (qb == kb)
        noisy_past = (~q_clean) & k_clean & (kb < qb)
        clean_past = q_clean & k_clean & (kb <= qb)
        return noisy_self | noisy_past | clean_past
    return create_block_mask(visible, B=None, H=None, Q_LEN=2*n, KV_LEN=2*n, device=device)


class SDARPolicy(SDARForCausalLM):
    """Same weights/backbone; explicit likelihood path instead of diffusion loss."""

    def forward(self, input_ids, position_ids, target_positions, labels,
                active_vocab: int, temperature: float = 1.0):
        n = input_ids.shape[1] // 2
        mask = packed_attention(n, self.config.block_size, str(input_ids.device))
        # autocast does not cast embedding lookups. Keep residuals/RoPE in the
        # compute dtype while gradients flow back to the FP32 master embedding.
        embeds = self.model.embed_tokens(input_ids)
        if torch.is_autocast_enabled(input_ids.device.type):
            embeds = embeds.to(torch.get_autocast_dtype(input_ids.device.type))
        hidden = self.model(inputs_embeds=embeds, position_ids=position_ids,
                            attention_mask=mask, use_cache=False).last_hidden_state
        hidden = hidden[:, target_positions] if labels.ndim == 2 else hidden[0, target_positions]
        # Full vocabulary matches the rollout sampler.
        logits = F.linear(hidden, self.lm_head.weight[:active_vocab]).float() / temperature
        log_distribution = logits.log_softmax(-1)
        logp = log_distribution.gather(-1, labels[..., None]).squeeze(-1)
        with torch.no_grad():
            entropy = -(log_distribution.exp() * log_distribution).sum(-1)
        return logp, entropy

