"""Real 4B checks: engine likelihood, EOS, batched rollout, two-pass backward."""
import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from algorithm import Response, response_logps, cache_old_logps, backward_response
from likelihood import SDARPolicy
from vendor.sdar.configuration_sdar import SDARConfig
from runtime import format_prompt
from rollout import Rollout
from lmdeploy.pytorch.strategies.dllm.model_agent import DLLMModelAgentStrategy
from lmdeploy.pytorch import consts


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, default=Path(__file__).resolve().parent.parent/'DiRL/public/SDAR-4B-Chat')
    p.add_argument('--output', type=Path, default=Path(__file__).resolve().parent/'verification/gpu_checks.json')
    p.add_argument('--block-size', type=int, choices=[1,2,4,8,16,32,64,128], default=None)
    p.add_argument('--long-tokens', type=int, default=3000,
                   help='Length of the synthetic trajectory used for backward/memory checks')
    args = p.parse_args()
    if args.long_tokens < 2:
        p.error('--long-tokens must be at least 2')
    torch.set_num_threads(4)
    torch.manual_seed(1234)
    torch.backends.cuda.matmul.allow_tf32 = False
    started = time.time()
    def stage(message):
        print(f'[{time.time()-started:.1f}s] {message}', flush=True)
    config = SDARConfig.from_pretrained(args.model)
    if args.block_size is not None:
        config.block_size = args.block_size
    block_size = config.block_size
    stage(f'Loading model; learner and rollout block_size={block_size}')
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = SDARPolicy.from_pretrained(args.model, config=config,
                                       torch_dtype=torch.float32).cuda().train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    rollout = Rollout(args.model, tokenizer, 128, 4, block_size=block_size)
    rollout.update_weights(model)
    events = []
    original = DLLMModelAgentStrategy.post_sampling

    def capture(self, inputs, logits, next_token_ids, extra_inputs):
        before = extra_inputs.dllm_mask.clone()
        result = original(self, inputs, logits, next_token_ids, extra_inputs)
        positions = torch.where((before==consts.DLLM_MASKED) &
                                (result[1].dllm_mask==consts.DLLM_UNMASKED))[0]
        assert len(inputs.seq_length) == 1
        base = int((inputs.history_lengths+inputs.seq_length)[0])-block_size
        for j in positions.tolist():
            token = int(result[0][j])
            events.append(dict(position=base+j, token=token,
                logp=logits[j].float().log_softmax(-1)[token].item(), vocab=logits.shape[-1]))
        return result

    prompt = format_prompt(tokenizer, 'What is 2 + 3? Reply with only the boxed number.')
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    DLLMModelAgentStrategy.post_sampling = capture
    stage('Checking sampled token positions and engine/learner probabilities')
    try:
        tokens, finished = rollout.sample([prompt],1,71)[0]
    finally:
        DLLMModelAgentStrategy.post_sampling = original
    events = events[:len(tokens)]
    assert [e['position'] for e in events] == list(range(len(prompt_ids),len(prompt_ids)+len(tokens)))
    assert [e['token'] for e in events] == tokens
    assert all(e['vocab']==model.config.vocab_size for e in events)
    response = Response(prompt_ids,tokens,1,finished,1.)
    logps,_ = response_logps(model,response)
    errors = (logps.cpu()-torch.tensor([e['logp'] for e in events])).abs()
    probability_check = dict(block_size=block_size, prompt_ids=prompt_ids, tokens=tokens,
        text=tokenizer.decode(tokens, skip_special_tokens=False), events=events,
        learner_logps=logps.cpu().tolist(), absolute_errors=errors.tolist(),
        max_error=errors.max().item(), mean_error=errors.mean().item())
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.with_name(args.output.stem+'_probabilities.json').write_text(
        json.dumps(probability_check,indent=2)+'\n')
    stage(f'Probability check: max error={errors.max().item():.6f}, mean error={errors.mean().item():.6f}')
    assert errors.max()<.2, 'Engine/learner probability discrepancy exceeds the audit tolerance'
    stage('Checking batched EOS')
    prompts = [format_prompt(tokenizer,q) for q in ['What is 1 + 1?', 'Compute 3 times 4. Answer briefly.']]
    batched = rollout.sample(prompts,2,1234)
    assert len(batched)==4
    for ids,done in batched:
        assert not any(t in rollout.eos_ids for t in ids[:-1])
        assert done == (ids[-1] in rollout.eos_ids)

    stage('Checking backward and an in-memory SGD update with frozen old probabilities')
    cache_old_logps(model,[response])
    frozen = response.old_logps.clone()
    before = backward_response(model,response,1)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True).item()
    assert norm>0
    # One in-memory update exercises frozen-old drift. No base weights are written.
    optimizer = torch.optim.SGD(model.parameters(),lr=1e-2)
    optimizer.step(); optimizer.zero_grad(set_to_none=True)
    after = backward_response(model,response,1)
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
    torch.testing.assert_close(response.old_logps,frozen,atol=0,rtol=0)
    assert before['log_ratio_abs_max'] < 1e-6 and after['log_ratio_abs_max'] > 1e-5
    model.zero_grad(set_to_none=True)

    # A long unfinished trajectory exercises the configured phase/memory path.
    stage(f'Checking {args.long_tokens}-token backward')
    content = [t for t in tokens if t not in rollout.eos_ids]
    assert content, 'Need at least one non-EOS token for the long trajectory'
    long_tokens = (content * (args.long_tokens//len(content)+1))[:args.long_tokens]
    long_response = Response(prompt_ids,long_tokens,0,False,-.5)
    cache_old_logps(model,[long_response])
    long_response.old_logps += torch.linspace(-.8,.8,args.long_tokens)
    long_stats = backward_response(model,long_response,1)
    long_norm = torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True).item()
    assert long_stats['tokens']==args.long_tokens and long_stats['clipped']>0 and long_norm>0
    result = dict(model=str(args.model), block_size=block_size,
        engine_tokens=len(tokens), finished=finished,
        engine_max_logp_error=errors.max().item(),engine_mean_logp_error=errors.mean().item(),
        batched_lengths=[len(t) for t,_ in batched],batched_finished=[d for _,d in batched],
        first_update=before, second_update=after, gradient_norm=norm,
        long_response=long_stats,long_gradient_norm=long_norm,
        old_logps_unchanged=True,elapsed_seconds=time.time()-started,
        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)
    rollout.close()


if __name__=='__main__':
    main()
