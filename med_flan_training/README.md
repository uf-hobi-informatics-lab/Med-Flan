# med_flan_training

Loop-transformer SFT and GRPO training for the med_flan project, packaged as a
standalone, runnable example. Copied from the parent `alignment-handbook` repo
with host-specific paths and identity scrubbed to `/project/...` placeholders,
and deduplicated to one representative config per (model, training category).

Three model families are included, one size each:
- **HRM-Text-1B** (bp_steps=5)
- **Ouro-2.6B-Thinking**
- **Qwen3-8B**

Training categories per family: std SFT, gated SFT (Ouro), plw005/plw010 (prompt
loss weight), weighted-JSON loss, GRPO (multitask reward), and a 2-stage
curriculum (stage1 warmup -> stage2 continuation). Variants that differed only
in dataset path (v1 / v1_md / v1_md_upsampled / v1a / v1b) were collapsed into
the `v1_md` (markdown) data variant as the canonical representative.

## Layout

```
med_flan_training/
  setup.py  setup.cfg  Makefile  requirements.txt   # pip-installable package
  src/alignment/                              # trainers, configs, HRM subpackage, GRPO rewards
  scripts/                                    # run_sft.py run_grpo.py run_hf_sft.py analyze_token_lengths.py
  experiments/
    wandb.sh  util.sh                         # sourced by launch scripts
    loop_transformer/
      hrm_1b/runai/  hrm_1b/slurm/             # HRM-Text-1B SFT (std, weighted_json)
      ouro/runai/                             # Ouro-2.6B-Thinking (std, gated, plw*, weighted_json, grpo)
      qwen/runai/  qwen/slurm/                 # Qwen3-8B (std, plw*, weighted_json, grpo)
      curriculum/{hrm_1b,ouro,qwen}/runai/    # curriculum stage1 + stage2 per family
      launch_v1_runai.sh  launch_curriculum_stage{1,2}_runai.sh
      analyze_token_lengths.runai.sh
  recipes/
    accelerate_configs/                       # deepspeed_zs2.json (GRPO)
    loop_transformer/
      hrm_1b/                                 # sft_bp5_std, sft_bp5_std_single_gpu, sft_bp5_weighted_json
      ouro/                                   # sft_2_6b_{std,std_gated,plw005,plw010,weighted_json}, grpo_2_6b_multitask
      qwen/                                   # sft_8b_{std,plw005,plw010,weighted_json}, grpo_8b_multitask
      curriculum/{hrm_1b,ouro,qwen}/          # stage1 + stage2 per family
```

## Path convention

All real host paths were replaced with `/project/...` placeholders. Deploy this
directory at `/project` on the training host (or set the `ROOT`/`HOME`/`BASE`
variables at the top of each launch script to wherever it lives). Conventions:

| Placeholder            | Mount this on the host            |
|------------------------|-----------------------------------|
| `/project`             | this repo checkout                |
| `/project/models`      | base model weights (Qwen3, Ouro, HRM-Text) |
| `/project/med_flan`    | med_flan datasets + output checkpoints |
| `/project/cache`       | HF / Triton / dataset caches       |
| `/project/log`         | wandb + slurm stdout logs         |
| `/project/container`   | apptainer image for slurm runs     |

RunAI PVC mounts in the launch scripts use the same paths (`path=/project`,
`path=/project/models`); adjust the `claimname=` values to your cluster.

## Install

```bash
pip install -e .
```

WANDB and user identity are read from the environment (see
`experiments/wandb.sh`); export `WANDB_API_KEY`, `WANDB_BASE_URL`,
`WANDB_USERNAME`, `USER`, `USER_EMAIL` before launching.

## Run

RunAI (single node, 4 GPUs) — SFT, one per family:

```bash
bash experiments/loop_transformer/hrm_1b/runai/sft_bp5_std.runai.sh
bash experiments/loop_transformer/ouro/runai/sft_2_6b_std.runai.sh
bash experiments/loop_transformer/qwen/runai/sft_8b_std.runai.sh
```

GRPO (uses `recipes/accelerate_configs/deepspeed_zs2.json`):

```bash
bash experiments/loop_transformer/ouro/runai/grpo_2_6b_multitask.runai.sh
bash experiments/loop_transformer/qwen/runai/grpo_8b_multitask.runai.sh
```

SLURM — SFT:

```bash
sbatch experiments/loop_transformer/hrm_1b/slurm/sft_bp5_std.slurm
sbatch experiments/loop_transformer/qwen/slurm/sft_8b_std.slurm
```

Curriculum stage1 -> stage2 (orchestrators submit all three families):

```bash
bash experiments/loop_transformer/launch_curriculum_stage1_runai.sh
bash experiments/loop_transformer/launch_curriculum_stage2_runai.sh
```

Or submit all SFT jobs across all families at once:

```bash
bash experiments/loop_transformer/launch_v1_runai.sh
```

Token-length analysis (CPU only):

```bash
bash experiments/loop_transformer/analyze_token_lengths.runai.sh
```

## Entry points

- `scripts/run_sft.py` — trl SFTTrainer path (Ouro, Qwen) — `recipes/loop_transformer/{ouro,qwen}/*.yaml`
- `scripts/run_hf_sft.py` — plain HF Trainer path (HRM-Text) — `recipes/loop_transformer/hrm_1b/*.yaml`
- `scripts/run_grpo.py` — GRPO with `med_multitask_reward` — `recipes/loop_transformer/{ouro,qwen}/grpo_*.yaml`
- `scripts/analyze_token_lengths.py` — tokenizer/chat-template length audit
