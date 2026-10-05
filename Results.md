## Results

Five models were each fine-tuned **once** as a single multi-task model on the union of nine
in-distribution (ID) training sets. One set of weights serves every task, with no per-task
heads, adapters, or prompts. Each model is evaluated on every ID test split and on four
out-of-distribution (OOD) test sets that are never used in training.

| Model | Family | Params | Layers × recurrent steps | Pretraining tokens |
|---|---|---:|---|---:|
| Qwen3-1.7B | Standard transformer | 1.7B | 28 × 1 | 36T |
| Qwen3-8B | Standard transformer | 8.2B | 36 × 1 | 36T |
| Ouro-1.4B-Thinking | Looped transformer | 1.4B | 24 × 4 | 7.7T |
| Ouro-2.6B-Thinking | Looped transformer | 2.6B | 48 × 4 | 7.7T |
| HRM-Text-1B | Hierarchical recurrent | 1B | 16 × 8 (H/L modules) | 40B |

Ouro runs at a fixed 4 recurrent steps (exit gate unused). HRM-Text uses a backpropagation
horizon of bp = 5 and a context window extended from 4,096 to 32,768 tokens for fine-tuning.
"Thinking" refers to Ouro's released reasoning-SFT checkpoint used as the starting point; it
is **not** an inference-time thinking mode. Qwen3 thinking mode is disabled.

All models use full-parameter SFT on 4× NVIDIA B300 GPUs. Qwen3 and Ouro are served with vLLM
and JSON-schema-constrained decoding; HRM-Text is served with PyTorch generation, and its
outputs are validated as JSON.

### Key findings

1. **Looped beats size-matched standard.** Ouro-1.4B-Thinking outperforms the larger
   Qwen3-1.7B on 7 of 12 datasets and ties on the other 5, with **no losses**.
2. **Looped matches 3× larger standard.** Ouro-2.6B-Thinking matches or exceeds Qwen3-8B on
   **11 of 12** datasets with about one third of the parameters. The only loss is CMED, at
   −0.017.
3. **Better OOD generalization.** Recurrent-depth models beat size-matched standard decoders
   on the OOD average: Ouro-1.4B-T +0.071 and HRM-Text-1B +0.037 over Qwen3-1.7B, and
   Ouro-2.6B-T +0.021 over Qwen3-8B.
4. **Architecture matters most for knowledge-heavy QA.** The spread across the five models is
   tiny on UMN (0.003), SHAC-MIMIC (0.006), and Drug-ADE (0.014); moderate on CMED, cohort,
   STS, and family history (0.04–0.09); and large on MedQA (0.227) and MMLU-Pro (0.233).
5. **The type of shift matters more than the model.**
   - New institution (SHAC-UW): all models are within 0.009.
   - New task format (NLI4CT): the models spread by 0.146.
   - New annotation schema (i2b2): every model fails, at 0.10–0.22 concept F1.
6. **The two-stage curriculum does not help overall.** It improves CMED for every model, but
   four non-decomposed tasks decline for every model (see below).

Ties are defined as |Δ| ≤ 0.01, since each configuration was trained once.

### In-distribution test results

| Dataset / metric | Qwen3-1.7B | Ouro-1.4B-T | HRM-Text-1B | Qwen3-8B | Ouro-2.6B-T |
|---|---:|---:|---:|---:|---:|
| Cohort selection (`2018_n2c2_track1`), micro-F1 | 0.8458 | 0.9038 | 0.8921 | **0.9125** | 0.9063 |
| Cohort selection, macro-F1 | 0.6695 | 0.7609 | 0.7597 | **0.8148** | 0.8104 |
| Drug-ADE (`2018_n2c2_track2`), entity F1 † | 0.8689 | 0.8715 | 0.8677 | 0.8696 | **0.8771** |
| Drug-ADE, relation F1 † | 0.8034 | 0.8063 | 0.7991 | 0.8075 | **0.8170** |
| Clinical STS, 5-class (`2019_n2c2_track1_cls`), accuracy | 0.6893 | 0.6966 | **0.7306** | 0.6626 | 0.6748 |
| Clinical STS, macro-F1 | 0.5386 | **0.6193** | 0.6035 | 0.5847 | 0.5954 |
| Family history (`2019_n2c2_track2`), F1 | 0.5618 | 0.6126 | 0.5604 | 0.6442 | **0.6534** |
| CMED (`2022n2c2_track1`), trigger F1 | 0.8196 | 0.8437 | 0.8369 | **0.8480** | 0.8321 |
| CMED, attribute F1 | 0.6269 | **0.6818** | 0.6793 | 0.6817 | 0.6600 |
| CMED, combined F1 | 0.7275 | 0.7644 | 0.7609 | **0.7669** | 0.7501 |
| SHAC-MIMIC (`2022n2c2_track2_mimic`), trigger F1 | 0.9591 | 0.9610 | 0.9631 | **0.9638** | 0.9631 |
| SHAC-MIMIC, argument F1 | 0.7860 | **0.7939** | 0.7835 | 0.7929 | 0.7888 |
| SHAC-MIMIC, combined F1 | 0.8474 | 0.8531 | 0.8472 | **0.8535** | 0.8505 |
| MedNLI (`MedNLI`), accuracy | 0.8875 | 0.8987 | 0.8558 | 0.9051 | **0.9128** |
| MedQA-USMLE US (`MedQA-USMLE-US`), accuracy | 0.4399 | 0.5075 | 0.3857 | 0.5734 | **0.6127** |
| UMN abbreviations (`umn_abbreviation`), accuracy | 0.9845 | 0.9826 | 0.9850 | **0.9855** | 0.9833 |
| **ID average (9 datasets)** | 0.7578 | 0.7842 | 0.7612 | 0.7936 | **0.7990** |

### Out-of-distribution test results

| Dataset / metric | Qwen3-1.7B | Ouro-1.4B-T | HRM-Text-1B | Qwen3-8B | Ouro-2.6B-T |
|---|---:|---:|---:|---:|---:|
| SHAC-UW (`2022n2c2_track2_uw`), trigger F1 | 0.9481 | 0.9473 | 0.9482 | **0.9514** | 0.9510 |
| SHAC-UW, argument F1 | 0.7628 | 0.7718 | 0.7687 | 0.7750 | **0.7752** |
| SHAC-UW, combined F1 | 0.8294 | 0.8349 | 0.8331 | **0.8383** | **0.8383** |
| NLI4CT (`nli4ct`), accuracy | 0.4840 | 0.5940 | **0.6300** | 0.5960 | 0.6200 |
| MMLU-Pro Health (`MMLU-Pro-Health`), accuracy | 0.3524 | 0.4508 | 0.3123 | 0.5067 | **0.5456** |
| 2010 i2b2/VA (`2010_i2b2_va`), concept F1 ‡ | 0.2046 | 0.0980 | 0.1135 | 0.1249 | **0.2233** |
| **OOD average (SHAC-UW, NLI4CT, MMLU-Pro)** | 0.5553 | 0.6266 | 0.5918 | 0.6470 | **0.6680** |
| **Overall average (12 datasets)** | 0.7071 | 0.7448 | 0.7189 | 0.7569 | **0.7662** |

Bold marks the best value per row.

- † **Provisional.** Drug-ADE values are the better of paragraph-level and whole-note
  inference. They will be replaced with whole-note scores.
- ‡ i2b2 is reported as concept F1 only and is excluded from all averages. Relation outputs
  reference concept text instead of concept IDs, so relations cannot be scored.
- NLI4CT is a two-label test set evaluated with the three-label NLI prompt.

**Averages** use one primary metric per dataset:

| Dataset | Primary metric |
|---|---|
| Cohort selection | micro-F1 |
| Drug-ADE | mean of entity and relation F1 |
| STS | accuracy |
| Family history | F1 |
| CMED, SHAC | combined F1 |
| All others | accuracy |

### Thinking vs. base initialization (Ouro)

| Scale | Base | Thinking | Δ (best configuration) | Δ (identical fine-tuning configuration) |
|---|---:|---:|---:|---:|
| 1.4B | 0.7196 | 0.7448 | +0.0252 | +0.0028 |
| 2.6B | 0.7471 | 0.7662 | +0.0191 | +0.0009 |

The table reports the overall average over 12 datasets. Starting from the Thinking checkpoint
gives better fine-tuned models at both scales. Under an identical fine-tuning configuration,
the gap falls below 0.003.

### Two-stage curriculum vs. single-stage training

In stage 1 (3 epochs), the model trains on single-goal extraction subtasks:

| Corpus | Stage-1 subtasks |
|---|---|
| Drug-ADE | entity-only; relation-only (given gold entities) |
| CMED | trigger-only; attribute-only (given gold triggers) |

Stage 2 is standard multi-task training. Values below are changes (curriculum minus
single-stage) under the same fine-tuning configuration. Ouro-2.6B-Thinking was not trained
with the curriculum.

| Dataset | Qwen3-1.7B | Ouro-1.4B-T | HRM-Text-1B | Qwen3-8B |
|---|---:|---:|---:|---:|
| Cohort selection | −0.0068 | −0.0216 | −0.0016 | +0.0058 |
| Drug-ADE | −0.0153 | −0.0101 | +0.0021 | −0.0025 |
| STS | −0.0738 | +0.0020 | +0.0132 | +0.0256 |
| Family history | −0.0375 | −0.0585 | −0.0206 | −0.0413 |
| CMED | +0.0746 | +0.0105 | +0.0384 | +0.0179 |
| SHAC-MIMIC | −0.0286 | −0.0202 | +0.0010 | −0.0112 |
| MedNLI | −0.0137 | −0.0248 | −0.0166 | −0.0093 |
| MedQA | −0.0250 | −0.0520 | −0.0331 | −0.0330 |
| UMN | −0.0382 | −0.0409 | −0.0123 | −0.0124 |
| SHAC-UW | −0.0274 | −0.0218 | −0.0052 | −0.0108 |
| NLI4CT | −0.0182 | −0.0403 | +0.0356 | +0.1189 |
| MMLU-Pro | +0.0059 | −0.0592 | +0.0026 | −0.0054 |
| i2b2 (concept F1) | +0.0084 | +0.0574 | −0.0162 | +0.0152 |
| **ID average** | −0.0183 | −0.0240 | −0.0033 | −0.0067 |
| **OOD average** | −0.0132 | −0.0404 | +0.0110 | +0.0342 |
| **Overall average** | −0.0170 | −0.0281 | +0.0003 | +0.0035 |

Of the two decomposed tasks, CMED improves for every model (+0.011 to +0.075), but Drug-ADE
does not. Family history, MedNLI, MedQA, and UMN decline for every model. i2b2 improves for 3
of 4 models.

### Other observations

- **Paragraph-level inference on Drug-ADE.** Predicting on blank-line paragraph windows and
  recombining the predictions per note scores higher than whole-note inference for most
  models. For HRM-Text-1B, entity F1 rises from 0.8418 to 0.8677 and relation F1 from 0.7562 to
  0.7991.
- **HRM-Text backpropagation horizon.** Relative to bp = 5, the overall average changes by
  −0.0018 at bp = 6, −0.0003 at bp = 7, and −0.0162 at bp = 8. bp = 5 matches the horizon
  HRM-Text was pretrained with.
- **Pipeline ceilings.** Round-tripping gold targets through the converters and scorers
  yields:

  | Dataset | Ceiling |
  |---|---|
  | Drug-ADE | 0.9750 entity F1, 0.9608 relation F1 |
  | CMED | 0.9479 combined F1 |
  | SHAC-MIMIC | 0.9862 combined F1 |
  | SHAC-UW | 0.9872 combined F1 |

  Read model scores against these ceilings, not against 1.0. See
  [Per-task evaluation status](#per-task-evaluation-status).

### Limitations

- The comparison spans model families, so pretraining data, tokenizers, and post-training
  differ alongside recurrence.
- Only accuracy is measured. Inference latency, memory, and compute were not measured.
- Each configuration was trained once (single seed).
- Recurrent depth is fixed, with no adaptive-depth sweep.
- Decoding differs: constrained decoding for Qwen3 and Ouro, but only output validation for
  HRM-Text.
- Note overlap across the MIMIC-derived corpora has not been checked.
