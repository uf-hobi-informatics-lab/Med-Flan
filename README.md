# meddata: multi-task clinical NLP text-to-text project

Single decoder-only LLM, trained on the union of every dataset below (each
example prefixed with a task tag), evaluated per-task on that dataset's own
original held-out split, scored with that task's own official (or closest
available) metric. See the design plan for full background on the task-tag
mechanism, JSON target schemas, and merged-train/per-task-eval protocol.

**14 task tags, 17 datasets.** ~574K merged training examples, ~471M tokens
(cl100k) — see [Data balance](#data-balance) for the per-dataset breakdown.

## Quickstart

Reproduce every number in this README end to end, once every raw dataset
in [Data sources](#data-sources) has been downloaded into its matching
top-level folder:

```bash
python -m multitask.build_unified_dataset   # merged train + per-dataset eval jsonl
python -m multitask.build_hf_datasets       # per-dataset HF DatasetDict, saved separately
python -m multitask.hf_dataset_stats        # token-count stats, regenerates the appendix below
python -m multitask.eval.run_eval           # full self-check: every real scorer, gold-vs-gold
```

## Layout

- `multitask/schemas.py` — task-tag registry: fixed instruction per tag,
  JSON target rendering (`render`/`render_messages`).
- `multitask/builders/` — one module per dataset. `build_x(split)` reads the
  raw released format and yields tagged input/target examples.
- `multitask/build_unified_dataset.py` — runs every builder, writes the
  merged training set and each dataset's own unmerged eval split.
  - `multitask/data/train_unified.jsonl`, `dev_unified.jsonl` — merged
    training/validation (validation is for monitoring joint training only,
    never for the per-task numbers below).
  - `multitask/data/eval/<dataset>.jsonl` — one file per dataset, unmerged,
    gold target embedded in `messages[-1]["content"]`.
- `multitask/reanchor.py` — shared helper that recovers BRAT character
  offsets from a JSON entity's mention text (an LLM can only be expected to
  generate text it saw, not exact byte offsets).
- `multitask/convert_back/` — one module per dataset that needs a real
  format conversion (BRAT `.ann`, n2c2 XML, float-lines) before an official
  scorer can run on it. **This is where the actual JSON→native converters
  live** — not `multitask/eval/`.
- `multitask/eval/` — scorers for the tasks that *don't* need a format
  conversion (`simple_tasks.py`: nli/mcqa/mcqa_explain/qa_yesnomaybe/
  multilabel_cls — gold and prediction are both just a small JSON dict, so
  scoring is a direct field comparison; `discharge_me.py`: ROUGE/BLEU/
  optional-BERTScore over the two plain-text summarization fields), plus
  `run_eval.py`, a combined runner that dispatches real model completions
  for **any** dataset name to the right converter+scorer pipeline (BRAT/XML/
  TSV/float-lines conversion via `convert_back/` where needed, or straight to
  `simple_tasks`/`discharge_me` otherwise) and prints one report.
- `multitask/vendor/` — third-party evaluation code vendored verbatim
  (currently just `brat_scoring`, for SHAC — see below) when `pip install`
  isn't viable in this venv (see Known issues).

## Data sources

Raw datasets are **not** committed to this repo (see `.gitignore`) — only code,
docs, and a `sample_data/<dataset>/` folder with 1-5 real records per dataset
so the on-disk schema is visible without downloading anything. To reproduce a
full run, download each dataset from its source below into the matching
top-level folder, then run the relevant `multitask/builders/` module.

| Folder | Dataset | Access | Source |
|---|---|---|---|
| `2018_n2c2_track1/` | n2c2 2018 Track 1 — Cohort Selection for Clinical Trials | Registration + Data Use Agreement | [n2c2/DBMI portal](https://n2c2.dbmi.hms.harvard.edu/) |
| `2018_n2c2_track2/` | n2c2 2018 Track 2 — Drug/ADE relations | Registration + Data Use Agreement | [n2c2/DBMI portal](https://n2c2.dbmi.hms.harvard.edu/) |
| `2019_n2c2_track1/` | n2c2 2019 Track 1 — Clinical Semantic Textual Similarity | Registration + Data Use Agreement | [n2c2/DBMI portal](https://n2c2.dbmi.hms.harvard.edu/) |
| `2019_n2c2_track2/` | n2c2 2019 Track 2 — Family History extraction | Registration + Data Use Agreement | [n2c2.dbmi.hms.harvard.edu/track2](https://n2c2.dbmi.hms.harvard.edu/track2) |
| `2022n2c2_track1/` | n2c2 2022 Track 1 — CMED (Contextualized Medication Event Extraction) | Registration + Data Use Agreement | [n2c2/DBMI portal](https://n2c2.dbmi.hms.harvard.edu/) |
| `2022n2c2_track2/` | n2c2 2022 Track 2 — SHAC (Social History Annotation Corpus) | Registration + Data Use Agreement | [n2c2.dbmi.hms.harvard.edu/2022-track-2](https://n2c2.dbmi.hms.harvard.edu/2022-track-2) |
| `MedNLI/` | MedNLI — clinical NLI dataset | PhysioNet credentialing + signed DUA | [physionet.org/content/mednli](https://physionet.org/content/mednli/1.0.0/) |
| `discharge-me-.../` | Discharge Me! (BioNLP ACL'24 shared task, MIMIC-IV-derived) | PhysioNet credentialing + signed DUA (needs both MIMIC-IV-Note and MIMIC-IV-ED access) | [physionet.org/content/discharge-me](https://physionet.org/content/discharge-me/1.3/) |
| `2019_COVID_NLP/` | CORD-19 (COVID-19 Open Research Dataset) | Open | [ai2-semanticscholar-cord-19 historical releases](https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/historical_releases.html) |
| `covid/` | BC7-LitCovid (BioCreative VII LitCovid track) | Open | [ftp.ncbi.nlm.nih.gov/pub/lu/LitCovid](https://ftp.ncbi.nlm.nih.gov/pub/lu/LitCovid/) |
| `MedQA-USMLE/`, `medqa/` | MedQA-USMLE (US/Mainland/Taiwan) — `medqa/` is a byte-identical duplicate of `MedQA-USMLE/questions/US` | Open | [github.com/jind11/MedQA](https://github.com/jind11/MedQA) (Google Drive link in repo) |
| `medmcqa/` | MedMCQA — Indian medical entrance exam MCQA | Open (Apache 2.0) | [huggingface.co/datasets/openlifescienceai/medmcqa](https://huggingface.co/datasets/openlifescienceai/medmcqa) |
| `pubmedqa/` | PubMedQA — biomedical research yes/no/maybe QA | Open | [github.com/pubmedqa/pubmedqa](https://github.com/pubmedqa/pubmedqa) (PQA-U/PQA-A via linked Google Drive files) |
| `MMLU-Pro-Health/` | MMLU-Pro, Health discipline subset | Open | [huggingface.co/datasets/TIGER-Lab/MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) |
| `semeval_2015/` | SemEval-2015 Task 14 — clinical disorder recognition/normalization (ShARe corpus, MIMIC-derived notes) | Registration + Data Use Agreement (MIMIC-derived) | [alt.qcri.org/semeval2015/task14](https://alt.qcri.org/semeval2015/task14/) — see the site's "Data and Tools" section, or contact the task organizers |
| `umn_abbreviation/` | UMN Clinical Abbreviation & Acronym Sense Inventory — abbreviation disambiguation in clinical text (Moon, Pakhomov, Melton) | Open (University of Minnesota Digital Conservancy) | [conservancy.umn.edu/handle/11299/137703](https://conservancy.umn.edu/handle/11299/137703) |
| `2010_i2b2_va/` | 2010 i2b2/VA Challenge — concepts, assertions, and relations in clinical text (Uzuner et al.) | Registration + Data Use Agreement | [n2c2/DBMI portal](https://n2c2.dbmi.hms.harvard.edu/) — **not yet obtained; pipeline built ahead of data, see status table below** |

## Citations

Source papers for each dataset/task-design choice above, for a paper draft's
bibliography. **Verify against each paper directly before citing** — these
are recalled from general knowledge of the shared-task literature, not
re-confirmed against the publisher record in this pass, so exact
venue/year/author-order details may need correction:

- **Overall task design** (task-tag + JSON-schema text-to-text formulation
  over a union of clinical NLP tasks): Peng, C. et al. (2024). *A study of
  generative large language model for medical research and healthcare.*
  npj Digital Medicine / JAMIA.
- **2018 n2c2 Track 1** (Cohort Selection): Stubbs, A. et al. (2019).
  *Cohort selection for clinical trials: n2c2 2018 shared task track 1.*
  JAMIA.
- **2018 n2c2 Track 2** (Drug-ADE, `ner_re_medication`): Henry, S. et al.
  (2020). *2018 n2c2 shared task on adverse drug events and medication
  extraction in electronic health records.* JAMIA.
- **2019 n2c2 Track 1** (Clinical STS, `sts_regression`): Wang, Y. et al.
  (2020). *The 2019 n2c2/OHNLP track on clinical semantic textual
  similarity: overview.* JMIR Medical Informatics.
- **2019 n2c2 Track 2** (Family History, `family_history`): n2c2 2019
  Track 2 shared task organizers (Family History extraction) — see task
  page for the official citation.
- **2022 n2c2 Track 1** (CMED, `medication_event`): Mahajan, D. et al.
  (2022). *Overview of the 2022 n2c2 shared task on contextualized
  medication event extraction in clinical notes.* JAMIA.
- **2022 n2c2 Track 2** (SHAC/SDoH, `sdoh_event`): Lybarger, K. et al.
  (2023). *The 2022 n2c2/UW shared task on extracting social determinants
  of health.* JAMIA.
- **MedNLI** (`nli`): Romanov, A. & Shivade, C. (2018). *Lessons from
  natural language inference in the clinical domain.* EMNLP.
- **Discharge Me!** (`summarization`): Xu, R. et al. (2024). *Overview of
  the discharge me shared task on streamlining discharge documentation.*
  BioNLP Workshop @ ACL 2024.
- **SemEval-2015 Task 14** (ShARe, `disorder_normalization`): Elhadad, N.
  et al. (2015). *SemEval-2015 Task 14: Analysis of clinical text.*
  SemEval 2015.
- **UMN Clinical Abbreviation Sense Inventory**
  (`abbreviation_disambiguation`): Moon, S., Pakhomov, S., & Melton, G.
  (2012). *Automated disambiguation of acronyms and abbreviations in
  clinical texts: window and training size considerations.* AMIA Annual
  Symposium.
- **MedQA-USMLE** (`mcqa`): Jin, D. et al. (2021). *What disease does this
  patient have? A large-scale open domain question answering dataset from
  medical exams.* Applied Sciences.
- **MMLU-Pro** (`mcqa`, Health subset): Wang, Y. et al. (2024).
  *MMLU-Pro: a more robust and challenging multi-task language
  understanding benchmark.* NeurIPS.
- **MedMCQA** (`mcqa_explain`): Pal, A., Umapathi, L. K., & Sankarasubbu,
  M. (2022). *MedMCQA: a large-scale multi-subject multi-choice dataset
  for medical domain question answering.* CHIL.
- **PubMedQA** (`qa_yesnomaybe`): Jin, Q. et al. (2019). *PubMedQA: a
  dataset for biomedical research question answering.* EMNLP.
- **BC7-LitCovid** (`multilabel_cls`): Chen, Q. et al. (2021). *Overview of
  the BioCreative VII LitCovid track: multi-label topic classification for
  COVID-19 literature annotation.* BioCreative VII.

## Per-task evaluation status

| Task tag | Dataset | Scorer | Status |
|---|---|---|---|
| `ner_re_medication` | 2018_n2c2_track2 (Drug-ADE) | `Track2-evaluate-ver4.py` (official) | ✅ verified. Tags F1 0.9818, relations F1 0.9366 |
| `medication_event` | 2022n2c2_track1 (CMED) | `eval_script.py` (official) | ✅ verified. Tags F1 0.997, attrs F1 0.906, Combined F1 0.888 |
| `sdoh_event` | 2022n2c2_track2 (SHAC) | vendored `brat_scoring` (organizer's own; no script ships with the release) | ✅ verified. Overall F1 0.9939 |
| `family_history` | 2019_n2c2_track2 | `eval.py` (official) | ✅ verified |
| `cohort_eligibility` | 2018_n2c2_track1 | `track1_eval.py` (official) | ✅ verified. P=R=F1=1.0 on every criterion except KETO-1YR, which is a known div-by-zero artifact of the official script itself (zero gold "met" cases for that criterion in the test set), reproduced identically scoring gold against itself |
| `sts_regression` | 2019_n2c2_track1 (Clinical STS) | `correlation-noconfidence.pl` (official) | ✅ verified. Pearson r = 1.0000. See note below — free-generated float regression is an inherent design compromise, not fully resolved |
| `nli` | MedNLI | `multitask/eval/simple_tasks.py` (exact match) | ✅ verified |
| `mcqa` | MedQA-USMLE (US/Mainland/Taiwan), MMLU-Pro-Health | `multitask/eval/simple_tasks.py` (exact match) | ✅ verified |
| `mcqa_explain` | medmcqa | `multitask/eval/simple_tasks.py` (exact match on `answer` only; `explanation` isn't auto-scored) | ✅ verified. **Uses `dev` as the eval split** — medmcqa's public `test` split ships with `answer: null` for all examples, so it's unscoreable |
| `qa_yesnomaybe` | pubmedqa | `multitask/eval/simple_tasks.py` (exact match) | ✅ verified |
| `multilabel_cls` | covid (LitCovid) | `multitask/eval/simple_tasks.py` (micro P/R/F1) | ✅ verified |
| `summarization` | discharge-me | `multitask/eval/discharge_me.py` — ROUGE-1/2/L + BLEU (+ optional BERTScore), macro-averaged over the two target sections | ✅ verified (partial metric — see note below). Self-check on 25 gold samples: ROUGE-1/2/L = 1.0, BLEU ≈ 100 |
| `abbreviation_disambiguation` | umn_abbreviation | `multitask/eval/simple_tasks.py` (exact match) | ✅ verified |
| `disorder_normalization` | semeval_2015 (SemEval-2015 Task 14 / ShARe) | `multitask/convert_back/disorder_normalization_2015_t14.py` — **self-implemented** strict F (exact span + CUI) and relaxed F (word-overlap + CUI), per Elhadad et al. 2015 §4.1; no persistent public official scorer found | ✅ verified. Strict F1 = Relaxed F1 = 1.0 on gold round-trip. Not an official binary — see note below |
| `concept_assertion_relation` | 2010_i2b2_va | `multitask/convert_back/i2b2_2010.py` — **self-implemented** concept/relation P/R/F1 + assertion accuracy (no official scorer ships with this release) | ⏳ pending data. DUA-gated data not yet obtained — builder/scorer are written against the published `.con`/`.ast`/`.rel` file-format spec (confirmed via `bigbio/n2c2_2010`'s dataloader) but genuinely untested against a real archive |

"Verified" means: gold JSON (built by that dataset's own builder) is
round-tripped back through the converter and the real scorer, expecting
P=R=F1=1.0 (or the metric's equivalent) before ever touching model output —
this isolates converter bugs from model quality. Where the number isn't
exactly 1.0, it's because of an inherent, non-bug limitation, documented
inline in the relevant module and summarized in Known limitations below.

CMED (`2022n2c2_track1`) and SHAC (`2022n2c2_track2`) have no released
`test` split locally, so `dev` stands in as their held-out eval split.

## Data balance

`build_unified_dataset.py` builds `train_unified.jsonl` by concatenating
every dataset's train split and shuffling once — **there is no balancing,
weighting, capping, or oversampling logic anywhere** in this repo. That's a
real problem here: dataset size varies by three orders of magnitude, and
document length varies by another order of magnitude on top of that.

**Per-dataset train counts, average length, and total token volume:**

| Dataset | Task tag | Train | Train tk_mean | Train tokens (≈) |
|---|---|---|---|---|
| discharge-me | summarization | 68,785 | 4303.9 | 296.0M |
| pubmedqa | qa_yesnomaybe | 211,269 | 449.7 | 95.0M |
| medmcqa | mcqa_explain | 182,822 | 207.9 | 38.0M |
| 2018_n2c2_track1 (cohort_eligibility) | cohort_eligibility | 2,626 | 5401.3 | 14.2M |
| covid_LitCovid | multilabel_cls | 24,960 | 343.5 | 8.6M |
| MedQA-USMLE-Mainland | mcqa | 27,400 | 168.6 | 4.6M |
| umn_abbreviation | abbreviation_disambiguation | 19,257 | 166.2 | 3.2M |
| 2018_n2c2_track2 (Drug-ADE) | ner_re_medication | 303 | 10445.2 | 3.2M |
| MedQA-USMLE-US | mcqa | 10,178 | 247.5 | 2.5M |
| MedQA-USMLE-Taiwan | mcqa | 11,298 | 200.7 | 2.3M |
| MedNLI | nli | 11,232 | 78.8 | 0.9M |
| 2022n2c2_track1 (CMED) | medication_event | 350 | 1902.2 | 0.7M |
| 2022n2c2_track2 (SHAC) | sdoh_event | 1,316 | 496.2 | 0.65M |
| semeval_2015 (SemEval-2015 T14) | disorder_normalization | 298 | 1985.1 | 0.6M |
| 2019_n2c2_track1 (Clinical STS) | sts_regression | 1,654 | 132.9 | 0.2M |
| 2019_n2c2_track2 (family history) | family_history | 99 | 869.1 | 0.09M |
| MMLU-Pro-Health | mcqa | 0 (test-only, 823) | n/a | 0 |

Total ≈ 471M train tokens across ~574K examples.

**Two distinct, compounding imbalances:**

1. **Example-count imbalance**: pubmedqa (211,269) vs. family_history (99)
   ≈ **2,134:1**. The seven datasets that are the paper's core clinically-rich
   extraction/regression tasks (`cohort_eligibility`, `ner_re_medication`,
   `sts_regression`, `family_history`, `medication_event`, `sdoh_event`,
   `disorder_normalization`) each have well under 3,000 train examples, while
   three QA/classification/summarization datasets (pubmedqa, medmcqa,
   discharge-me) each have 25,000–211,000, and `abbreviation_disambiguation`
   (umn_abbreviation, 19,257) sits in between.
2. **Token-volume imbalance is worse than example count suggests**, because
   average document length varies ~55x across datasets. `discharge-me`
   alone is **~296M tokens — about 63% of the entire merged training set's
   token volume**. `pubmedqa` + `medmcqa` + `discharge-me` together are
   **~91% of all training tokens**, while the seven small extraction/
   regression datasets above sum to only **~19.6M tokens (~4%)**.
   `family_history` alone is ~0.09M tokens — **0.02% of total training
   volume** — despite being one of the paper's core benchmark tasks.

**Secondary observations:**

- `MMLU-Pro-Health` contributes **zero** training examples (test-only, 823
  examples) — it's a pure generalization probe for the `mcqa` tag, not a
  training source. `mcqa`'s effective training size is 48,876, from the
  three MedQA variants only.
- `cohort_eligibility` already has a **13x per-example amplification** in
  its builder (one training example per criterion per patient, 288 patients
  → 2,626+1,118 examples) — an existing precedent for turning a small
  document-level dataset into more single-signal examples.
- `cohort_eligibility` also already flags **within-task** class imbalance
  (`COHORT_CRITERIA_RARE` in `multitask/schemas.py`: MI-6MOS, KETO-1YR,
  ALCOHOL-ABUSE, DRUG-ABUSE have few positive/negative instances) — a
  different axis (label skew within one dataset) from the cross-dataset
  volume imbalance above, but the same mitigation ideas apply.
- Eval/dev splits are correctly *not* merged or resampled anywhere — this
  imbalance is specific to `train_unified.jsonl`.

**Recommendations (not yet implemented — options to weigh, not a decision):**

1. **Temperature-scaled resampling of `train_unified.jsonl`** (the standard
   T5/mT5-style multi-task mixing fix): sample each dataset with probability
   ∝ `size^alpha` (alpha ≈ 0.3–0.5) instead of raw natural frequency —
   oversamples small datasets, downsamples huge ones, without going fully
   uniform (which would overfit the 99-example `family_history` set almost
   immediately). Would mean a `resample(examples, target_n, rng)` step per
   dataset in `build_unified_dataset.py` before concatenation, with a
   printed before/after count table. Most direct fix; no training-loop
   changes needed.
2. **Weight by token volume, not just example count**, when choosing
   resampling targets — discharge-me's 55x-longer-than-average documents
   mean example-count-based balancing alone still leaves it dominating
   gradient updates. Either fold token count into the temperature-scaling
   formula, or independently cap/chunk very long documents as a second
   lever — this needs care, since eval scores full documents, so any
   truncation strategy must stay consistent between train and eval framing.
3. **For the HF per-dataset pipeline** (`build_hf_datasets.py`, which
   already saves each dataset separately rather than merging):
   `datasets.interleave_datasets(..., probabilities=weights)` at actual
   training time, with temperature-scaled `weights` computed from the same
   counts, is the native HF mechanism for this — no change to the build
   script itself, just documenting the recommended weights.
4. **Oversampling risk**: repeating a 99-example dataset many times per
   epoch risks overfitting on it specifically — monitor per-task eval
   metrics separately during training (already possible via the per-dataset
   eval splits + `multitask/eval/run_eval.py`) rather than assuming
   resampling alone is safe.
5. `MMLU-Pro-Health`'s zero-train-example status is a design fact, not
   something resampling can or should fix — it's meant to stay eval-only.

## Running an eval

### Combined runner (real model output, any dataset)

`multitask/eval/run_eval.py` dispatches a dataset name + `{id: raw completion
string}` dict to the right converter/scorer pipeline and prints that
dataset's real metrics — no need to know which task tag or converter a given
dataset uses:

```python
from multitask.eval.run_eval import run_eval

# id must match the "id" field in multitask/data/eval/<dataset>.jsonl
run_eval("2018_n2c2_track2", {"100130": '{"entities": [...], "relations": [...]}', ...})
run_eval("MedNLI", {"some-id": '{"label": "entailment"}', ...})
run_eval("discharge-me", {"some-id": "DISCHARGE INSTRUCTIONS:\n...\n\nBRIEF HOSPITAL COURSE:\n...", ...})
```

Dataset names match the `Dataset` column above (`2018_n2c2_track2`,
`2018_n2c2_track1`, `2019_n2c2_track1`, `2019_n2c2_track2`, `2022n2c2_track1`,
`2022n2c2_track2`, `discharge-me`, `MedNLI`, `MedQA-USMLE-US/Mainland/Taiwan`,
`MMLU-Pro-Health`, `medmcqa`, `pubmedqa`, `covid (LitCovid)`, `semeval_2015`,
`umn_abbreviation`). Completions may
be a raw string exactly as a model would emit it (parsed leniently — first
JSON object found, or for `sts_regression`/`summarization`, the relevant
regex/section parser) or an already-parsed dict, which is passed through
unchanged (this is how the self-check below feeds gold back).

From the command line, with completions as a JSON file of `{id: completion}`:

```bash
python -m multitask.eval.run_eval 2018_n2c2_track2 completions.json
```

Run with no arguments to self-check **every** dataset end-to-end (gold fed
back as its own prediction through the exact same dispatch path — slow,
runs every real official scorer plus discharge-me's ROUGE/BLEU, intended as
a pre-flight/CI check rather than routine use):

```bash
python -m multitask.eval.run_eval
```

Confirmed: running this end-to-end reproduces every number in the status
table above exactly (e.g. 2018_n2c2_track2 tags F1 0.9818, SHAC F1 0.9939,
STS Pearson r = 1.0000, cohort_eligibility 1.0 except the documented
KETO-1YR artifact, discharge-me perfect ROUGE/BLEU on the full 10,962-row
test split, semeval_2015 strict/relaxed F1 = 1.0, umn_abbreviation accuracy
1.0) — the dispatcher itself introduces no discrepancy versus calling
each module directly.

### Per-dataset self-check (no model needed)

Every dataset with a real converter can also be self-checked standalone
(gold JSON → native format → real scorer) by running its module directly —
useful when iterating on a single converter without waiting on the others:

```bash
python -m multitask.convert_back.ner_re_2018_t2            # 2018 n2c2 track 2 (Drug-ADE)
python -m multitask.convert_back.medication_event_2022_t1  # 2022 n2c2 track 1 (CMED)
python -m multitask.convert_back.sdoh_2022_t2               # 2022 n2c2 track 2 (SHAC)
python -m multitask.convert_back.family_history_2019_t2      # 2019 n2c2 track 2
python -m multitask.convert_back.cohort_eligibility_2018_t1  # 2018 n2c2 track 1
python -m multitask.convert_back.sts_2019_t1                 # 2019 n2c2 track 1 (Clinical STS)
python -m multitask.convert_back.disorder_normalization_2015_t14  # SemEval-2015 Task 14

python -m multitask.eval.simple_tasks   # nli / mcqa / mcqa_explain / qa_yesnomaybe / multilabel_cls / abbreviation_disambiguation, all 9 datasets
python -m multitask.eval.discharge_me   # summarization, 25-sample gold self-check
```

### Calling a single converter/scorer directly

`run_eval()` is a thin dispatcher — nothing it does is unavailable by calling
the underlying building blocks yourself, if you want finer control (e.g.
scoring a subset, or intercepting the intermediate native-format files):

```python
# BRAT-based datasets (ner_re_2018_t2 / medication_event_2022_t1 / sdoh_2022_t2)
from multitask.convert_back.medication_event_2022_t1 import write_anns, _run_eval, GOLD_DIR
from multitask.io_utils import read_text

predictions = [(doc_id, model_output_json, read_text(GOLD_DIR / f"{doc_id}.txt"))
               for doc_id, model_output_json in your_completions.items()]
with tempfile.TemporaryDirectory() as tmp:
    write_anns(predictions, tmp)
    _run_eval(GOLD_DIR, tmp)   # prints the official eval_script.py table

# XML (cohort_eligibility)
from multitask.convert_back.cohort_eligibility_2018_t1 import write_xmls, _run_eval, GOLD_DIR
predictions = [(locator, model_output_json) for locator, model_output_json in your_completions.items()]
with tempfile.TemporaryDirectory() as tmp:
    write_xmls(predictions, tmp)
    _run_eval(GOLD_DIR, tmp)

# Float-lines (sts_regression) -- targets must be in the test file's original line order
from multitask.convert_back.sts_2019_t1 import to_score_lines, _run_eval, GOLD_SCORES
with open("system.txt", "w") as f:
    f.write(to_score_lines(your_completions_in_line_order))
_run_eval(GOLD_SCORES, "system.txt")

# Simple classification-style tasks
from multitask.eval.simple_tasks import score_dataset
metrics = score_dataset("MedNLI", {doc_id: raw_completion_str for doc_id, raw_completion_str in your_completions.items()})

# summarization (discharge-me)
from multitask.eval.discharge_me import score_dataset
metrics = score_dataset({doc_id: raw_completion_str for doc_id, raw_completion_str in your_completions.items()})
```

`your_completions` in each case is whatever your inference step produced,
keyed by the `id` field of the corresponding `multitask/data/eval/<dataset>.jsonl`
row.

## Known limitations

- **Inherent text-ambiguity ceiling on the BRAT datasets** (`ner_re_medication`,
  `medication_event`, `sdoh_event`): a small fraction of gold entities are
  short/common tokens ("2", "in", "iv") repeated many times per document with
  no positional signal to disambiguate from text alone. This affects a real
  generative model identically, so F1 in the 0.98-0.99 range (not 1.0) on a
  perfect gold round-trip is the practical ceiling for text-only offset
  reconstruction on these datasets — not a bug being chased further.
- **`sts_regression`'s float-regression-via-generation is an accepted, not
  fully solved, design compromise.** Free-generating a continuous 0-5 score
  is a poor fit for an LLM: numeric tokenization is inconsistent across
  otherwise-identical training examples, and decoding tends to collapse onto
  a handful of round values, which hurts a Pearson-correlation metric more
  than a classification one. The mitigation in place is modest by design
  (no constrained-decoding infra in this project): training targets are
  rendered as a fixed two-decimal string (`3.50`, not `3.5`), and
  `convert_back/sts_2019_t1.py` parses model output leniently (regex-extract
  the first float, clamp to `[0, 5]`) rather than requiring exact JSON. If
  real model output collapses onto too few distinct values in practice, the
  next step would be discretizing to the 11 anchor values the released test
  gold actually uses (`0, 0.5, 1, ..., 5`) and scoring it as classification
  instead — not implemented speculatively ahead of seeing real output.
- **`cohort_eligibility`'s KETO-1YR "met" precision/recall show as 0.0 even
  on a perfect gold round-trip** — the official `track1_eval.py` computes
  precision/recall as `tp/(tp+fp)` etc. and catches `ZeroDivisionError` by
  returning `0.0`; the test set has zero gold "met" cases for KETO-1YR, so
  this fires identically whether scoring a perfect converter or real gold
  against itself. Not a converter bug.
- **`summarization` (discharge-me)'s metric is a partial stand-in for the
  official one.** The real BioNLP-ACL24 "Discharge Me!" shared task scores a
  4-way composite: ROUGE-1/2/L, BLEU, BERTScore, and MEDCON (a clinical-
  concept-overlap metric computed via QuickUMLS against licensed UMLS data).
  `multitask/eval/discharge_me.py` computes the first three (BERTScore
  optional — off by default in `run_eval.py` since it downloads a ~1.4GB
  model on first use); MEDCON is intentionally not implemented, since faking
  or approximating a metric that needs licensed data would be worse than
  leaving it out. Treat these numbers as directionally useful, not the
  official score.
- **`disorder_normalization`'s strict/relaxed F metric is self-implemented,
  not an official binary.** No persistent public scorer for SemEval-2015
  Task 14 could be found (the task's own validation script was only
  available "during the evaluation period," with no lasting public link).
  `multitask/convert_back/disorder_normalization_2015_t14.py` implements
  strict F (exact span + correct CUI) and relaxed F (word-overlap + correct
  CUI) directly from the formulas in Elhadad et al. 2015 §4.1. Verified the
  same way as every other scorer here — gold round-tripped through the
  converter hits strict F1 = relaxed F1 = 1.0 — but treat the numbers as a
  faithful reimplementation of the paper's metric, not the organizers' own
  code.
- **`umn_abbreviation` has no released train/test split** — the UMN release
  ships `train.txt`/`valid.txt`/`test.txt` pre-split, but the corpus itself
  defines no official benchmark split beyond that; splits are used as-is.
  The source file is Latin-1 encoded (not UTF-8, confirmed by decode
  failure) and its released `start`/`end` columns are offsets into the
  original full document rather than the context snippet shown to the
  model, so they're dropped rather than used for span alignment.

## Known issues

- `pip install git+https://...` (any git-based source build) fails in this
  project's venv with an `AssertionError` in
  `_distutils_hack.override.ensure_local_distutils()` — a systemic
  venv/setuptools issue unrelated to any specific package. Regular
  (non-source-build) `pip install` and `spacy download` work fine. Workaround
  used for `brat_scoring`: clone the repo directly and vendor the package
  files under `multitask/vendor/` instead of installing it.

<!-- BEGIN: multitask dataset stats (auto-generated by multitask/hf_dataset_stats.py -- do not edit by hand) -->
## `multitask/huggingface_format/` datasets

One HuggingFace `datasets` `DatasetDict` per dataset (train/dev/test, whichever splits exist), saved via `DatasetDict.save_to_disk()`. Every dataset shares the same schema:

```
{"messages": [{"role": "system"|"user"|"assistant", "content": str}, ...],
 "task": str, "dataset": str, "id": str}
```

so `tokenizer.apply_chat_template()` can be applied directly. Built by `multitask/build_hf_datasets.py` (`python -m multitask.build_hf_datasets`), from the same per-task builders in `multitask/builders/` that also feed the separate merged flat-text pipeline (`multitask/build_unified_dataset.py`). Reload any dataset with:

```python
from datasets import load_from_disk
ds = load_from_disk("multitask/huggingface_format/<name>")
```

Stats below (example counts, tiktoken `cl100k_base` token-length of the full rendered example) are generated by `multitask/hf_dataset_stats.py` (`python -m multitask.hf_dataset_stats`) -- see `multitask/huggingface_format/dataset_stats.json` for full per-split percentiles and Qwen3-4B token counts.

### Datasets

| dataset | task tag | description | train | dev | test | tk_mean (train) |
|---|---|---|---|---|---|---|
| 2018_n2c2_track1 | cohort_eligibility | Given the patient's longitudinal clinical notes and ONE named clinical trial eligibility criterion (with its definition), decide whether the patient meets that single criterion. Return JSON {"label": "met"} or {"label": "not met"}. | 2626 | - | 1118 | 5401.3 |
| 2018_n2c2_track2 | ner_re_medication | Extract all Drug entities and their attribute entities (Strength, Frequency, Dosage, Route, Duration, Form, Reason, ADE), plus the relation linking each attribute entity to its drug. Return JSON {"entities": [...], "relations": [...]}. | 303 | - | 202 | 10445.2 |
| 2019_n2c2_track1 | sts_regression | Rate the semantic similarity of the two clinical sentences on a 0-5 scale (0 = unrelated, 5 = semantically equivalent), to two decimal places. Return JSON {"similarity": <float, e.g. 3.50>}. | 1654 | - | 412 | 132.9 |
| 2019_n2c2_track2 | family_history | Extract family history from the note. Return JSON with four keys: "family_members" (list of {"member", "side"}), "observations_subtask1" (list of observation mention strings), "living_status" (list of {"member", "side", "score"}), and "observations" (list of {"member", "side", "text", "negation"}). | 99 | - | 115 | 869.1 |
| 2022n2c2_track1 | medication_event | For each medication mention, classify its disposition (Disposition, NoDisposition, or Undetermined); for Disposition mentions also give Action, Certainty, Temporality, Actor, and Negation. Return JSON {"events": [...]}. | 350 | 50 | - | 1902.2 |
| 2022n2c2_track2 | sdoh_event | Extract social-determinants-of-health events (e.g. Tobacco, Alcohol, Drug, Employment, LivingStatus) with their role-filler entities and attribute values. Return JSON {"events": [...]}. | 1316 | 188 | - | 496.2 |
| MMLU-Pro-Health | mcqa | Answer the multiple-choice question by selecting the correct option letter. Return JSON {"answer": <letter>}. | - | - | 823 | 183.6 |
| MedNLI | nli | Determine the relationship between the premise and hypothesis: entailment, contradiction, or neutral. Return JSON {"label": ...}. | 11232 | 1395 | 1422 | 78.8 |
| MedQA-USMLE-Mainland | mcqa | Answer the multiple-choice question by selecting the correct option letter. Return JSON {"answer": <letter>}. | 27400 | 3425 | 3426 | 168.6 |
| MedQA-USMLE-Taiwan | mcqa | Answer the multiple-choice question by selecting the correct option letter. Return JSON {"answer": <letter>}. | 11298 | 1412 | 1413 | 200.7 |
| MedQA-USMLE-US | mcqa | Answer the multiple-choice question by selecting the correct option letter. Return JSON {"answer": <letter>}. | 10178 | 1272 | 1273 | 247.5 |
| covid_LitCovid | multilabel_cls | Assign all applicable topic labels to this article. Return JSON {"labels": [...]}. | 24960 | 6239 | 2500 | 343.5 |
| discharge-me | summarization | Write the discharge instructions and brief hospital course for this patient. Return plain text, not JSON. | 68785 | 14719 | 10962 | 4303.9 |
| medmcqa | mcqa_explain | Answer the multiple-choice question and explain your reasoning. Return JSON {"answer": <letter>, "explanation": ...}. | 182822 | 4183 | - | 207.9 |
| pubmedqa | qa_yesnomaybe | Answer the research question yes, no, or maybe given the context, and give a long-form answer. Return JSON {"decision": ..., "long_answer": ...}. | 211269 | 50 | 500 | 449.7 |
| semeval_2015 | disorder_normalization | Identify all disorder mentions in the clinical text and normalize each to a UMLS/SNOMED-CT CUI, or "CUI-less" if no matching concept exists. Return JSON {"entities": [{"text": ..., "cui": ...}, ...]}. | 298 | - | 133 | 1985.1 |
| umn_abbreviation | abbreviation_disambiguation | Given a clinical abbreviation and its surrounding context, determine what the abbreviation stands for in this specific context. Return JSON {"expansion": ...}. | 19257 | 4127 | 4127 | 166.2 |

### Known gaps / small training sets

Three tasks are built at **document level** (one example per note/patient, the full entity/relation/event set as target) rather than split into many single-label examples the way `cohort_eligibility` (2018_n2c2_track1) was. This makes their train sets far smaller than every other task in the collection:

- `2018_n2c2_track2` (`ner_re_medication`): 303 train examples
- `2019_n2c2_track2` (`family_history`): 99 train examples
- `2022n2c2_track1` (`medication_event`): 350 train examples

By comparison, `cohort_eligibility` (2018_n2c2_track1) was deliberately split per-criterion into 2,626 train examples from the same 202 patients, avoiding this problem -- a template that could be applied to the doc-level tasks above (e.g. one example per entity/event/relation) if their small train sets turn out to limit model quality.

Separately, `2022n2c2_track1`, `2022n2c2_track2` have no test split released locally -- `dev` stands in as the held-out eval split for both in `build_hf_datasets.py`/`build_unified_dataset.py`.
<!-- END: multitask dataset stats -->
