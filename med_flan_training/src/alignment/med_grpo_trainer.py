"""MedGRPOTrainer -- TRL GRPOTrainer subclass with multi-temperature rollout (MDP-GRPO).

Opt-in via GRPOConfig.temperature_schedule (a list of temperatures, one per "bucket" of
the num_generations completions per prompt). When set AND the vLLM colocate generation
path is active, this overrides _generate_single_turn so the G completions for each
prompt are split into len(temperature_schedule) buckets, each generated at its own
temperature, then reassembled into the [prompt][generation] order TRL expects.

Why: for our deterministic medical rewards (exact-match 0/1, or doc-level F1), the easy
classification tasks produce many all-correct groups -- every completion gets reward 1,
the group advantage (r - mean) is 0, and GRPO gets no gradient for that prompt. Sampling
the G completions at varied temperatures (e.g. [0.2, 0.5, 0.8, 1.1]) raises the chance of
a mixed-correctness group, so the prompt actually contributes a learning signal. This is
the transferable part of MDP-GRPO (arXiv:2606.06058); the paper's dual-anchor /
prospect-theoretic parts are tailored to multi-constraint instruction-following and are
not implemented here (they don't map to our 12-task mix).

Limitations (deliberate):
  - vLLM colocate ONLY. The HF .generate() path (Ouro/HRM-Text use_vllm:false) and vLLM
    server mode fall back to the single global self.temperature -- multi-temperature is
    not wired there. For those families, mitigate zero-variance groups via SFT warm-start
    + the Stage-1 easy curriculum instead.
  - num_generations must be divisible by len(temperature_schedule); otherwise falls back
    to single temperature with a warning (never silently mis-trains).
  - The temperature is applied per-bucket by setting self.vllm_generation.temperature
    before each generate call (TRL's VLLMGeneration.generate reads
    self.temperature at line 579). Weights are synced once per optimizer step (mirroring
    the parent's guard), NOT per bucket, so the K buckets all use the current policy.

TRL already (a) zero-advantages zero-std groups (advantage = r - mean = 0) and (b) logs
the fraction as `frac_reward_zero_std`. So the complementary "DAPO dynamic sampling"
(resample zero-advantage groups with fresh generations to recover the compute) is the
only remaining efficiency lever, and it's a bigger _prepare_inputs data-flow change not
included here -- watch `frac_reward_zero_std` in wandb; if it's high after enabling
multi-temperature, resampling can be added as a follow-on.
"""

import logging

from trl import GRPOTrainer
from trl.extras.profiling import profiling_context

logger = logging.getLogger(__name__)


class MedGRPOTrainer(GRPOTrainer):
    """GRPOTrainer with opt-in multi-temperature rollout (see module docstring)."""

    def _generate_single_turn(
        self, prompt_ids, images, multimodal_fields, has_tool_images=False
    ):
        schedule = getattr(self.args, "temperature_schedule", None)
        # No schedule, or not on the vLLM colocate path -> behave exactly like the parent.
        if not schedule or not getattr(self, "use_vllm", False):
            return super()._generate_single_turn(
                prompt_ids, images, multimodal_fields, has_tool_images
            )

        mode = "train" if self.model.training else "eval"
        num_generations = (
            self.num_generations if mode == "train" else self.num_generations_eval
        )
        n_buckets = len(schedule)
        if num_generations % n_buckets != 0:
            logger.warning(
                f"[MedGRPOTrainer] num_generations ({num_generations}) not divisible by "
                f"len(temperature_schedule) ({n_buckets}); falling back to single temperature."
            )
            return super()._generate_single_turn(
                prompt_ids, images, multimodal_fields, has_tool_images
            )

        # Only the vLLM colocate path is supported -- the parent's HF .generate() and
        # server branches apply one temperature to the whole batch.
        if getattr(self.vllm_generation, "mode", None) != "colocate":
            logger.warning(
                "[MedGRPOTrainer] temperature_schedule set but vLLM mode is not 'colocate' "
                f"(got {getattr(self.vllm_generation, 'mode', None)!r}); falling back to single temperature."
            )
            return super()._generate_single_turn(
                prompt_ids, images, multimodal_fields, has_tool_images
            )

        per_bucket = num_generations // n_buckets
        # prompt_ids is ordered [p0_g0..p0_g{G-1}, p1_g0.., ...] -- G contiguous per prompt
        # (the advantage computation does rewards.view(-1, num_generations), which requires
        # this layout). P_local = number of prompts this rank holds.
        if len(prompt_ids) % num_generations != 0:
            logger.warning(
                f"[MedGRPOTrainer] len(prompt_ids) ({len(prompt_ids)}) not divisible by "
                f"num_generations ({num_generations}); falling back to single temperature."
            )
            return super()._generate_single_turn(
                prompt_ids, images, multimodal_fields, has_tool_images
            )
        n_prompts = len(prompt_ids) // num_generations

        # Sync vLLM weights once per optimizer step (mirror the parent's guard at
        # grpo_trainer.py:1827-1830). The K buckets all use the current policy -- we must
        # NOT re-sync per bucket.
        if self.state.global_step != self._last_loaded_step:
            with profiling_context(self, "sync_weights"):
                self.vllm_generation.sync_weights()
            self._last_loaded_step = self.state.global_step

        orig_temperature = self.vllm_generation.temperature
        bucket_completion_ids = []
        bucket_logprobs = []
        try:
            for k, temp in enumerate(schedule):
                # Subset of prompt_ids for bucket k: for each prompt, take generations
                # [k*per_bucket : (k+1)*per_bucket]. Preserves [prompt][generation] order
                # within the bucket.
                subset = []
                for p in range(n_prompts):
                    base = p * num_generations
                    subset.extend(
                        prompt_ids[base + k * per_bucket : base + (k + 1) * per_bucket]
                    )

                # images is aligned 1:1 with prompt_ids (one entry per completion's prompt);
                # subset it the same strided way. None (text-only, our case) -> None.
                if images is None:
                    bucket_images = None
                else:
                    bucket_images = []
                    for p in range(n_prompts):
                        base = p * num_generations
                        bucket_images.extend(
                            images[base + k * per_bucket : base + (k + 1) * per_bucket]
                        )

                self.vllm_generation.temperature = temp
                with profiling_context(self, f"vLLM.generate[T={temp}]"):
                    _, cids, lps, _ = self.vllm_generation.generate(
                        prompts=subset,
                        images=bucket_images,
                        num_generations=per_bucket,
                        profiler=None,
                    )
                # vLLM returns per-token top-k logprobs; keep only the top-1 (sampled token)
                # logprob -- same as the parent at grpo_trainer.py:1841.
                if lps is not None:
                    lps = [[lp[0] for lp in seq] for seq in lps]
                bucket_completion_ids.append(cids)
                bucket_logprobs.append(lps)
        finally:
            self.vllm_generation.temperature = orig_temperature

        # Reassemble into the original [p0_g0..p0_g{G-1}, p1_g0.., ...] order. For prompt p,
        # its G completions are bucket0's per_bucket, then bucket1's per_bucket, ... in
        # generation order. generate() returns completions in input-prompt order, so
        # bucket k's completions for prompt p are cids_k[p*per_bucket:(p+1)*per_bucket].
        completion_ids = []
        for p in range(n_prompts):
            for k in range(n_buckets):
                s = p * per_bucket
                completion_ids.extend(bucket_completion_ids[k][s : s + per_bucket])

        if bucket_logprobs[0] is None:
            logprobs = None
        else:
            logprobs = []
            for p in range(n_prompts):
                for k in range(n_buckets):
                    s = p * per_bucket
                    logprobs.extend(bucket_logprobs[k][s : s + per_bucket])

        return completion_ids, logprobs
