# Reused sources

- `sdar/`: copied from `../sdar_justgrpo_audit/vendor/sdar/`, including native configuration defaults and its Apache-2.0 license. The audit project records the original DiRL revision as `b7495efdd88f4f5a585d00b3242de4ecbec23853`. No inherited diffusion training loss is used.
- `gmpo/math_grader.py`: byte-for-byte copy of `../GMPO/understand_r1_zero_main/understand_r1_zero/math_grader.py`, GMPO revision `ae9cbebd8cc6b58a6db6c61cb90e4b13ff131d94`. The source's copyright header and the base project's MIT license are retained.
- `likelihood.py`: SDAR `phase_inputs`, packed attention and likelihood forward from `sdar_justgrpo_audit/algorithm.py`; entropy is returned per token. Attention and target-logit semantics are unchanged.
- `rollout.py`: sequence-major sampling metadata, raw token streaming, full vocabulary, engine setup and BF16 weight synchronization from `sdar_justgrpo_audit/rollout.py`. EOS now stops normally; the returned action sequence includes its first EOS.
- `runtime.py`: distributed reductions, prompt formatting and inference export copied from `sdar_justgrpo_audit/train.py`.
- `checkpointing.py`, `run_state.py`: copied from the audit project. Training keys/fingerprint reflect GMPO modules; offline W&B resume restores local training while creating a new local log segment.
- `tests/test_rollout.py`: reuses the audit project's chunked-prefill and sequence-major metadata regression tests; the termination test covers normal EOS instead of fixed horizon.

`algorithm.py` implements the GMPO paper (arXiv:2507.20673v3, Algorithm 1) and the `gmpo` branch of the local `GMPO/train_zero_math_gmpo.py`, using SDAR's exact LTR likelihood. CPU tests execute that unmodified branch as the objective oracle.

The rollout engine remains the existing sibling `DiRL/lmdeploy`; its source hashes are captured in each full training checkpoint's fingerprint.
