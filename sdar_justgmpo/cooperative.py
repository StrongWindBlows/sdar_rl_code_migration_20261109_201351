"""One-update policy gradient of E_S[R] - beta * KL(S || P).

P samples each current block independently given the clean history. Its log
probability includes the first EOS and marginalizes the unused block tail.
Both distributions share parameters; the direct P gradient is essential.
"""
import torch

from likelihood import reconstruction_phases


def leave_one_out(values):
    size = values.shape[-1]
    return (values - values.mean(-1, keepdim=True)) * size / (size - 1)


@torch.no_grad()
def sequence_logp_gap(model, response, temperature=1.0, phase_batch_size=1,
                      logit_chunk_size=128):
    gap = torch.zeros((), dtype=torch.float64, device=next(model.parameters()).device)
    for phase in reconstruction_phases(model, response, temperature, phase_batch_size, logit_chunk_size):
        gap += phase.logps.double().sum() - phase.full_mask_logps.double().sum()
    return gap.item()


def prepare_coupled(model, responses, group_size, beta, temperature=1.0,
                    phase_batch_size=1, logit_chunk_size=128, progress=None):
    """Freeze sequence gaps and independent leave-one-out utility baselines.

    Responses remain prompt-major here. Centering R - beta*d removes the large
    common gap without changing the expected gradient; no std/length scaling.
    """
    gaps = []
    for response in responses:
        gaps.append(sequence_logp_gap(model, response, temperature, phase_batch_size, logit_chunk_size))
        if progress is not None and (len(gaps) % 32 == 0 or len(gaps) == len(responses)):
            progress(len(gaps), len(responses))
    utility = (torch.tensor([r.reward for r in responses], dtype=torch.float64)
               - beta*torch.tensor(gaps, dtype=torch.float64))
    advantages = leave_one_out(utility.reshape(-1, group_size)).flatten().tolist()
    for response, advantage in zip(responses, advantages):
        response.advantage = advantage
    return gaps


def backward_coupled(model, response, minibatch_size, beta, temperature=1.0,
                     phase_batch_size=1, logit_chunk_size=128):
    """Stream shared graphs, merging sequential and full-mask losses per phase."""
    stats = dict(loss=0.0, entropy_sum=0.0, tokens=len(response.tokens), clipped=0,
                 log_ratio_sum=0.0, log_ratio_square_sum=0.0, log_ratio_abs_max=0.0,
                 full_mask_entropy_sum=0.0)
    for phase in reconstruction_phases(model, response, temperature, phase_batch_size, logit_chunk_size):
        loss = -(response.advantage*phase.logps.sum() + beta*phase.full_mask_logps.sum())
        (loss / minibatch_size).backward()
        stats['loss'] += loss.detach().item()
        stats['entropy_sum'] += phase.entropy.sum().item()
        stats['full_mask_entropy_sum'] += phase.full_mask_entropy.sum().item()
    return stats
