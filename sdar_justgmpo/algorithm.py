"""GMPO: token clipping in log space, then a geometric mean per response.

The old log-probabilities are frozen for an entire rollout round. SDAR phases
share one response-level coefficient; they are not separate GMPO objectives.
"""
from contextlib import nullcontext
from dataclasses import dataclass

import torch

from likelihood import phase_inputs


@dataclass
class Response:
    prompt: list[int]
    tokens: list[int]                 # through the first EOS, inclusive
    reward: float
    finished: bool = True
    advantage: float = 0.0
    old_logps: torch.Tensor | None = None


def group_advantages(rewards):
    """GMPO reference: sample standard deviation, epsilon=1e-8."""
    return (rewards - rewards.mean(-1, keepdim=True)) / (
        rewards.std(-1, correction=1, keepdim=True) + 1e-8)


def clipped_log_ratio(new_logps, old_logps, advantages, epsilon=0.4):
    """Algorithm 1: pessimistic token clipping, including negative advantages."""
    sign = torch.where(advantages >= 0, 1.0, -1.0).unsqueeze(-1)
    signed = sign * (new_logps - old_logps)
    return sign * torch.minimum(signed, signed.clamp(-epsilon, epsilon))


def policy_loss(new_logps, old_logps, advantages, mask, epsilon=0.4, objective='gmpo'):
    """One loss per response. GRPO uses exactly the same clipping interval."""
    clipped = clipped_log_ratio(new_logps, old_logps, advantages, epsilon)
    lengths = mask.sum(-1)
    if objective == 'gmpo':
        weight = ((clipped * mask).sum(-1) / lengths).exp()
    elif objective == 'grpo':
        weight = (clipped.exp() * mask).sum(-1) / lengths
    else:
        raise ValueError(f'Unknown objective: {objective}')
    return -advantages * weight


def phase_logps(model, response, temperature=1.0, phase_batch_size=1):
    device = next(model.parameters()).device
    for ids, positions, targets, labels, phase_count in phase_inputs(
            response, model.config.block_size, model.config.mask_token_id, device, phase_batch_size):
        context = torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' else nullcontext()
        with context:
            logps, entropy = model(ids, positions, targets, labels, model.config.vocab_size,
                                   temperature, phase_count)
        yield positions[0, targets] - len(response.prompt), logps, entropy


@torch.no_grad()
def response_logps(model, response, temperature=1.0, phase_batch_size=1):
    logps = torch.empty(len(response.tokens), device=next(model.parameters()).device)
    entropy_sum = 0.0
    for offsets, values, entropy in phase_logps(model, response, temperature, phase_batch_size):
        logps[offsets] = values
        entropy_sum += entropy.sum().item()
    return logps, entropy_sum


@torch.no_grad()
def cache_old_logps(model, responses, temperature=1.0, phase_batch_size=1):
    """Call once, before ANY optimizer update on this rollout batch."""
    for response in responses:
        response.old_logps = response_logps(model, response, temperature, phase_batch_size)[0].cpu()


def backward_on_policy_grpo(model, response, minibatch_size, temperature=1.0, phase_batch_size=1):
    """One forward/backward per phase group, only for one update per rollout batch.

    Before that update, old and current policies coincide: ratio=1, but its
    derivative is grad(logp). Detaching only the denominator preserves both.
    """
    stats = dict(loss=-response.advantage, entropy_sum=0.0, tokens=0, clipped=0,
                 log_ratio_sum=0.0, log_ratio_square_sum=0.0, log_ratio_abs_max=0.0)
    if response.advantage == 0:
        return stats
    for _, logps, entropy in phase_logps(model, response, temperature, phase_batch_size):
        ratio = (logps - logps.detach()).exp()
        loss = -response.advantage * ratio.sum() / len(response.tokens) / minibatch_size
        loss.backward()
        stats['entropy_sum'] += entropy.sum().item()
    stats['tokens'] = len(response.tokens)
    return stats


def backward_rloo(model, response, minibatch_size, temperature=1.0, phase_batch_size=1):
    """On-policy gradient of expected terminal reward, with a leave-one-out baseline."""
    stats = dict(loss=0.0, entropy_sum=0.0, tokens=0, clipped=0,
                 log_ratio_sum=0.0, log_ratio_square_sum=0.0, log_ratio_abs_max=0.0)
    if response.advantage == 0:
        return stats
    for _, logps, entropy in phase_logps(model, response, temperature, phase_batch_size):
        loss = -response.advantage * logps.sum()
        (loss / minibatch_size).backward()
        stats['loss'] += loss.detach().item()
        stats['entropy_sum'] += entropy.sum().item()
    stats['tokens'] = len(response.tokens)
    return stats


def backward_response(model, response, minibatch_size, temperature=1.0,
                      epsilon=0.4, objective='gmpo', phase_batch_size=1):
    """Two-pass, phase-streamed gradient of the complete response objective.

    The first pass computes q=exp(mean(clipped log-ratio)). The second pass
    differentiates -A*stopgrad(q)*sum_phase(clipped log-ratio)/T. Parameters
    remain fixed between passes, and SDAR attention dropout is zero.
    minibatch_size includes zero-advantage responses, matching the paper mean.
    """
    if response.advantage == 0:
        return dict(loss=0.0, entropy_sum=0.0, tokens=0, clipped=0,
                    log_ratio_sum=0.0, log_ratio_square_sum=0.0, log_ratio_abs_max=0.0)
    current, entropy_sum = response_logps(model, response, temperature, phase_batch_size)
    old = response.old_logps.to(current.device)
    advantage = current.new_tensor(response.advantage)
    clipped = clipped_log_ratio(current, old, advantage, epsilon)
    sequence_weight = clipped.mean().exp().detach()
    loss_value = policy_loss(current, old, advantage, torch.ones_like(current, dtype=torch.bool),
                             epsilon, objective).item()
    for offsets, logps, _ in phase_logps(model, response, temperature, phase_batch_size):
        phase_clipped = clipped_log_ratio(logps, old[offsets], advantage, epsilon)
        terms = sequence_weight * phase_clipped if objective == 'gmpo' else phase_clipped.exp()
        loss = -advantage * terms.sum() / len(response.tokens) / minibatch_size
        loss.backward()
    diff = current - old
    return dict(loss=loss_value, entropy_sum=entropy_sum, tokens=len(response.tokens),
                clipped=int((clipped != diff).sum()), log_ratio_sum=diff.sum().item(),
                log_ratio_square_sum=diff.square().sum().item(),
                log_ratio_abs_max=diff.abs().max().item())
