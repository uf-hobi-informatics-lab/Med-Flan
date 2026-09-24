# meddata: multi-task clinical NLP text-to-text project

Single decoder-only LLM, trained on the union of every dataset below (each
example prefixed with a two-level `<<Category: Task>>` tag), evaluated
per-task on that dataset's own original held-out split, scored with that
task's own official (or closest available) metric. See
[`design.md`](./design.md) for full background on the task-tag mechanism,
JSON target schemas, and merged-train/per-task-eval protocol.

**25 task tags** (17 end-to-end + 6 Tier A train-only subtask-enrichment
tags + 2 paragraph/section-window tags, see [Tier A subtask-enrichment datasets](#tier-a-subtask-enrichment-datasets-train-only-additive)
and [Paragraph-window datasets](#paragraph-window-datasets-hf-only-additive-not-scored-by-the-doc-level-pipeline)),
**26 datasets.** ~716K merged training examples, ~473M tokens
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

- `multitask/schemas.py` — two-level task-tag registry (`CATEGORIES` +
  `TASKS`): generic category-level system framing, dataset-specific
  instruction per task, JSON target rendering (`render`/`render_messages`).
  See [Task Taxonomy](#task-taxonomy) and [`design.md`](./design.md).
- `multitask/builders/` — one module per dataset. `build_x(split)` reads the
  raw released format and yields tagged input/target examples.
- `multitask/build_unified_dataset.py` — runs every builder, writes the
  merged training set and each dataset's own unmerged eval split.
  - `multitask/data/train_unified.jsonl`, `dev_unified.jsonl` — merged
    training/validation (validation is for monitoring joint training only,
    never for the per-task numbers below). For datasets with no released dev
    split, `multitask/split_utils.py`'s `carve_dev_from_test()` deterministically
    samples 30% of that dataset's test examples to stand in as dev here —
    this sample overlaps test rather than being removed from it, so the real
    test split (and per-task scoring against it) stays untouched.
  - `multitask/data/eval/<dataset>.jsonl` — one file per dataset, unmerged,
    gold target embedded in `messages[-1]["content"]`.
- `multitask/target_schema.py` — **single source of truth for every task's
  target JSON shape** (canonical key order, closed vocabularies, required
  keys). The target contract used to live in three places that could drift:
  the instruction prose in `schemas.py`, the builder code, and
  `convert_back/`. This declares it once, and drives `canonicalize()` (called
  from `schemas._target_str()`, so every built example is byte-stable for the
  same content), `validate()`, `instruction_gaps()` (cross-checks prose
  against the spec), and `json_schema()` (per-task JSON Schema for
  grammar-constrained decoding at inference).
- `multitask/validate_targets.py` — CI-style gate over the above; exits
  non-zero if any built target fails its spec, is not in canonical key order,
  or uses a vocabulary value the spec doesn't declare. Run with
  `python -m multitask.validate_targets`.
- `multitask/reanchor.py` — shared helper that recovers BRAT character
  offsets from a JSON entity's mention text (an LLM can only be expected to
  generate text it saw, not exact byte offsets). See also
  `brat.align_entity_text()`, which repairs `.ann` mention text that isn't a
  substring of its own document before it ever reaches a target.
- `multitask/brat_scorer.py` — unified tags/relations/events scorer (greedy
  nearest-distance bipartite matching) shared by all three BRAT-standoff
  datasets (`ner_re_medication`, `medication_event`, `sdoh_event`), replacing
  each dataset's separate official/vendored scorer — see its module
  docstring for why (the official scripts' tag/attribute matching is a
  single iteration-order-dependent pass, not a real assignment algorithm).
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
| `2022n2c2_track2/` | n2c2 2022 Track 2 — SHAC (Social History Annotation Corpus); MIMIC and UW subsets kept as separate, non-merged datasets (`2022n2c2_track2_mimic`, `2022n2c2_track2_uw`); the UW test split additionally ships a relation-form re-annotation (`test/uw_r/`) used for an auxiliary, self-implemented derived-relations eval axis | Registration + Data Use Agreement | [n2c2.dbmi.hms.harvard.edu/2022-track-2](https://n2c2.dbmi.hms.harvard.edu/2022-track-2) |
| `MedNLI/` | MedNLI — clinical NLI dataset | PhysioNet credentialing + signed DUA | [physionet.org/content/mednli](https://physionet.org/content/mednli/1.0.0/) |
| `bionli/` | BioNLI — adversarial biomedical NLI pairs (entailment/contradiction) generated from PubMed abstract conclusions via lexico-semantic constraints | Open (CC BY 4.0) | [aclanthology.org/2022.findings-emnlp.374](https://aclanthology.org/2022.findings-emnlp.374) |
| `nli4ct/` | NLI4CT — SemEval-2023 Task 7, entailment/contradiction over clinical trial report sections and statements | Open | [sites.google.com/view/nli4ct/home](https://sites.google.com/view/nli4ct/home) |
| `discharge-me-.../` | Discharge Me! (BioNLP ACL'24 shared task, MIMIC-IV-derived) | PhysioNet credentialing + signed DUA (needs both MIMIC-IV-Note and MIMIC-IV-ED access) | [physionet.org/content/discharge-me](https://physionet.org/content/discharge-me/1.3/) |
| `2019_COVID_NLP/` | CORD-19 (COVID-19 Open Research Dataset) | Open | [ai2-semanticscholar-cord-19 historical releases](https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/historical_releases.html) |
| `covid/` | BC7-LitCovid (BioCreative VII LitCovid track) | Open | [ftp.ncbi.nlm.nih.gov/pub/lu/LitCovid](https://ftp.ncbi.nlm.nih.gov/pub/lu/LitCovid/) |
| `MedQA-USMLE/`, `medqa/` | MedQA-USMLE (US/Mainland/Taiwan) — `medqa/` is a byte-identical duplicate of `MedQA-USMLE/questions/US` | Open | [github.com/jind11/MedQA](https://github.com/jind11/MedQA) (Google Drive link in repo) |
| `medmcqa/` | MedMCQA — Indian medical entrance exam MCQA | Open (Apache 2.0) | [huggingface.co/datasets/openlifescienceai/medmcqa](https://huggingface.co/datasets/openlifescienceai/medmcqa) |
| `pubmedqa/` | PubMedQA — biomedical research yes/no/maybe QA | Open | [github.com/pubmedqa/pubmedqa](https://github.com/pubmedqa/pubmedqa) (PQA-U/PQA-A via linked Google Drive files) |
| `MMLU-Pro-Health/` | MMLU-Pro, Health discipline subset | Open | [huggingface.co/datasets/TIGER-Lab/MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) |
| `semeval_2015/` | SemEval-2015 Task 14 — clinical disorder recognition/normalization (ShARe corpus, MIMIC-derived notes) | Registration + Data Use Agreement (MIMIC-derived) | [alt.qcri.org/semeval2015/task14](https://alt.qcri.org/semeval2015/task14/) — see the site's "Data and Tools" section, or contact the task organizers |
| `umn_abbreviation/` | UMN Clinical Abbreviation & Acronym Sense Inventory — abbreviation disambiguation in clinical text (Moon, Pakhomov, Melton) | Open (University of Minnesota Digital Conservancy) | [conservancy.umn.edu/handle/11299/137703](https://conservancy.umn.edu/handle/11299/137703) |
| `2010i2b2/` | 2010 i2b2/VA Challenge — concepts, assertions, and relations in clinical text (Uzuner et al.) | Registration + Data Use Agreement | [n2c2/DBMI portal](https://n2c2.dbmi.hms.harvard.edu/) |
| `emrqa-msquad/` | emrQA-msquad — SQuAD-v2.0-formatted extractive QA over LLM-summarized clinical notes derived from emrQA (i2b2 medication/relations + heart-disease-risk-factor corpora) | Open (no license field on the Hub dataset card) | [huggingface.co/datasets/Eladio/emrqa-msquad](https://huggingface.co/datasets/Eladio/emrqa-msquad) |
| `emrqa-msquad/` | emrQA-msquad — extractive QA over summarized clinical notes, SQuAD v2.0-framed, derived from emrQA (itself built on i2b2 medication/relations and heart-disease-risk-factor corpora), with LLM-generated (`text-davinci-003`) note summaries per the paper | Open (no license field on the Hub dataset card) | [huggingface.co/datasets/Eladio/emrqa-msquad](https://huggingface.co/datasets/Eladio/emrqa-msquad) |

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
  similarity: overview.* JMIR Medical Informatics. `sts_classification` is
  this project's own derived variant of the same sentence pairs (see
  [Known limitations](#known-limitations)) — not part of the original task
  organizers' release or citation.
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
- **BioNLI** (`nli`): Bastan, M., Surdeanu, M., & Balasubramanian, N.
  (2022). *BioNLI: Generating a biomedical NLI dataset using lexico-semantic
  constraints for adversarial examples.* Findings of the Association for
  Computational Linguistics: EMNLP 2022, pp. 5093–5104.
- **NLI4CT** (`nli`): SemEval-2023 Task 7 organizers. *Multi-evidence
  natural language inference for clinical trial data.* SemEval-2023 — see
  the [task site](https://sites.google.com/view/nli4ct/home) for the
  official citation.
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
- **emrQA-msquad** (`qa_span_extraction`): Jimenez Eladio, J. & Wu, H.
  (2024). *emrQA-msquad: A medical dataset structured with the SQuAD v2.0
  framework, enriched with emrQA medical information.* arXiv:2404.12050.
  Built on the original **emrQA**: Pampari, A., Raghavan, P., Liang, J., &
  Peng, J. (2018). *emrQA: A large corpus for question answering on
  electronic medical records.* EMNLP.
- **emrQA-msquad** (`qa_span_extraction`): Jimenez Eladio, J. & Wu, H.
  (2024). *emrQA-msquad: A medical dataset structured with the SQuAD v2.0
  framework, enriched with emrQA medical information.* arXiv:2404.12050.
- **emrQA** (source corpus for emrQA-msquad, itself built on i2b2
  medication/relations and heart-disease-risk-factor corpora): Pampari, A.
  et al. (2018). *emrQA: A large corpus for question answering on electronic
  medical records.* EMNLP.

## Per-task evaluation status

| Task tag | Dataset | Scorer | Status |
|---|---|---|---|
| `ner_re_medication` | 2018_n2c2_track2 (Drug-ADE) | `multitask/brat_scorer.py` — **self-implemented** greedy-matching scorer, replacing the official `Track2-evaluate-ver4.py` (its tag/relation matching is a single iteration-order-dependent pass, not a real assignment algorithm) | ✅ verified. Overall (micro) tags F1 0.9752 strict / 0.9755 lenient, relations F1 0.9609 strict / 0.9611 lenient — see note below |
| `medication_event` | 2022n2c2_track1 (CMED) | `multitask/brat_scorer.py` — replacing the official `eval_script.py` for the same reason | ✅ verified, against the real `test` split (100 notes). Triggers (medication span + disposition class) F1 0.9792, Attributes (Action/Certainty/Temporality/Actor/Negation) F1 0.9146, Combined F1 0.9479 |
| `sdoh_event` | 2022n2c2_track2_mimic (SHAC, MIMIC subset) | `multitask/brat_scorer.py` — replacing the vendored, event-only `brat_scoring` package (no relation axis) | ✅ verified, against the real `test` split. Triggers F1 0.9951, Roles F1 0.9813, Combined F1 0.9862 |
| `sdoh_event` | 2022n2c2_track2_uw (SHAC, UW subset) | `multitask/brat_scorer.py` | ✅ verified, against the real `test` split. Triggers F1 0.9956, Roles F1 0.9825, Combined F1 0.9872 |
| `sdoh_event` (relations, auxiliary axis) | 2022n2c2_track2_uw `test` only, scored against the `test/uw_r/` relation-form re-annotation | `multitask/convert_back/sdoh_2022_t2.py`'s `score_relations()`, via `multitask.brat_scorer.score_relations` — **self-implemented**, not an official scorer; the trainable target is event-only, relations are mechanically derived from it and scored only as a bonus axis on the 518 UW test docs | ✅ verified. Derived-relations micro P=0.9828 R=0.9825 F1=0.9826 (TP=3192, FP=56, FN=57) |
| `family_history` | 2019_n2c2_track2 | `eval.py` (official) | ✅ verified. **Subtask 2 only** (member/side/living-status/observation, the task's own "end-to-end family history summarization" framing) — Subtask 1's bare entity list was dropped from the trainable target as redundant with Subtask 2's own member/side fields; see `multitask/builders/family_history_2019_t2.py`'s module docstring. P=R=F1=1.0 on gold round-trip |
| `cohort_eligibility` | 2018_n2c2_track1 | `track1_eval.py` (official) | ✅ verified. P=R=F1=1.0 on every criterion except KETO-1YR, which is a known div-by-zero artifact of the official script itself (zero gold "met" cases for that criterion in the test set), reproduced identically scoring gold against itself |
| `sts_regression` | 2019_n2c2_track1 (Clinical STS) | `correlation-noconfidence.pl` (official) | ✅ verified. Pearson r = 1.0000. See note below — free-generated float regression is an inherent design compromise, not fully resolved |
| `sts_classification` | 2019_n2c2_track1_cls | `multitask/eval/simple_tasks.py` (accuracy + macro P/R/F1 + quadratic-weighted Cohen's kappa) — **self-implemented**, additive alternative to `sts_regression`, not from the original task organizers | ✅ verified, gold round-trip on the real `test` split (n=412): accuracy = macro_precision = macro_recall = macro_f1 = quadratic_weighted_kappa = 1.0. See note below |
| `nli` | MedNLI, bionli, nli4ct | `multitask/eval/simple_tasks.py` (exact match) | ✅ verified. accuracy = 1.0 on all three (n=1422, 6308, 500 respectively, gold round-trip) |
| `mcqa` | MedQA-USMLE (US/Mainland/Taiwan), MMLU-Pro-Health | `multitask/eval/simple_tasks.py` (exact match) | ✅ verified |
| `mcqa_explain` | medmcqa | `multitask/eval/simple_tasks.py` (exact match on `answer` only; `explanation` isn't auto-scored) | ✅ verified. **Uses `dev` as the eval split** — medmcqa's public `test` split ships with `answer: null` for all examples, so it's unscoreable |
| `qa_yesnomaybe` | pubmedqa | `multitask/eval/simple_tasks.py` (exact match) | ✅ verified |
| `multilabel_cls` | covid (LitCovid) | `multitask/eval/simple_tasks.py` (micro P/R/F1) | ✅ verified |
| `summarization` | discharge-me | `multitask/eval/discharge_me.py` — ROUGE-1/2/L + BLEU (+ optional BERTScore), macro-averaged over the two target sections | ✅ verified (partial metric — see note below). Self-check on 25 gold samples: ROUGE-1/2/L = 1.0, BLEU ≈ 100 |
| `abbreviation_disambiguation` | umn_abbreviation | `multitask/eval/simple_tasks.py` (exact match) | ✅ verified |
| `disorder_normalization` | semeval_2015 (SemEval-2015 Task 14 / ShARe) | `multitask/convert_back/disorder_normalization_2015_t14.py` — **self-implemented** strict F (exact span + CUI) and relaxed F (word-overlap + CUI), per Elhadad et al. 2015 §4.1; no persistent public official scorer found | ✅ verified. Gold round-trip: Strict F1 = 0.9150, Relaxed F1 = 1.0. Strict is no longer trivially 1.0 — gold now scores against its real offsets instead of being reanchored through the same lossy path as predictions, see note below. Not an official binary |
| `concept_assertion_relation` | 2010_i2b2_va | `multitask/convert_back/i2b2_2010.py` — **self-implemented** concept/relation P/R/F1 + assertion accuracy (no official scorer ships with this release) | ✅ verified. Gold round-trip on the real test split (256 docs): assertion accuracy = relation F1 = 1.0; concept F1 = 0.9996 (not exact — see the module docstring's "Known limitation": ~13/2600 concepts are short, repeated tokens like "T"/"P"/"rate" that reanchor's greedy matching can't disambiguate from text alone, an inherent ceiling any generative model's plain-text output shares) |
| `qa_span_extraction` | emrqa-msquad | `multitask/eval/qa_span.py` — **self-implemented** SQuAD-style normalized exact-match + token-F1, the metric the paper itself reports; no persistent official scorer ships with the Hub release | ✅ verified. Gold round-trip on the real `dev` split (n=32,739): exact_match = f1 = 1.0. **Uses `dev` (HF's `validation`) as the eval split** — the Hub release ships no `test` split, same fallback as `medmcqa` |

"Verified" means: gold JSON (built by that dataset's own builder) is
round-tripped back through the converter and the real scorer, expecting
P=R=F1=1.0 (or the metric's equivalent) before ever touching model output —
this isolates converter bugs from model quality. Where the number isn't
exactly 1.0, it's because of an inherent, non-bug limitation, documented
inline in the relevant module and summarized in Known limitations below.

## Task Taxonomy

Every rendered example carries a two-level tag, `<<Category: Task>>` (e.g.
`<<Information_Extraction: SDoH>>`), split across the chat-messages system
turn: the **category** is a small set of generic, reusable framings — the
part a decoder model learns once across every task that shares an output
contract shape (extraction / classification / QA / regression /
summarization) — and the **task** is the fine-grained tag whose
dataset-specific instruction (label set, JSON schema) sits in the user turn,
immediately before `INPUT:`. See [`design.md`](./design.md) for the full
rationale, the `CATEGORIES`/`TASKS` data structures in `multitask/schemas.py`,
and before/after render examples.

This taxonomy is a **model-facing/analysis grouping only**, tracked as a
`category` field alongside the existing `task`/`dataset`/`id` metadata on
every example. It is fully decoupled from `eval/run_eval.py`'s scorer
dispatch and `convert_back/`'s schema selection, both of which are keyed on
the unchanged low-level `task` string — nothing about eval routing changes.

| Category | Task tag (display) | Task tag (code) | Dataset(s) |
|---|---|---|---|
| `Information_Extraction` | NER_RE | `ner_re_medication` | 2018_n2c2_track2 |
| `Information_Extraction` | Medication_Event | `medication_event` | 2022n2c2_track1 |
| `Information_Extraction` | SDoH | `sdoh_event` | 2022n2c2_track2_mimic, 2022n2c2_track2_uw |
| `Information_Extraction` | Family_History | `family_history` | 2019_n2c2_track2 |
| `Information_Extraction` | Disorder_Normalization | `disorder_normalization` | semeval_2015 |
| `Information_Extraction` | Concept_Assertion_Relation | `concept_assertion_relation` | 2010_i2b2_va |
| `Classification` | NLI | `nli` | MedNLI, bionli, nli4ct |
| `Classification` | Multilabel_Topic | `multilabel_cls` | covid (LitCovid) |
| `Classification` | Cohort_Eligibility | `cohort_eligibility` | 2018_n2c2_track1 |
| `Classification` | Abbreviation_Disambiguation | `abbreviation_disambiguation` | umn_abbreviation |
| `Question_Answering` | MCQA | `mcqa` | MedQA-USMLE-US/Mainland/Taiwan, MMLU-Pro-Health |
| `Question_Answering` | MCQA_Explain | `mcqa_explain` | medmcqa |
| `Question_Answering` | QA_YesNoMaybe | `qa_yesnomaybe` | pubmedqa |
| `Question_Answering` | QA_Span_Extraction | `qa_span_extraction` | emrqa-msquad |
| `Similarity_Regression` | STS_Regression | `sts_regression` | 2019_n2c2_track1 |
| `Classification` | STS_Classification | `sts_classification` | 2019_n2c2_track1_cls |
| `Summarization` | Summarization | `summarization` | discharge-me |

**Two placements are judgment calls, not clean-cut**, documented inline in
`multitask/schemas.py`'s `TASKS` dict and in `design.md`:

- `qa_yesnomaybe` (PubMedQA) is a hybrid: its `decision` field (yes/no/maybe)
  is a closed label (classification-shaped), but the task's overall framing
  (question + context → answer, plus a free-text `long_answer`) is
  QA-shaped like `mcqa`/`mcqa_explain`, and matches PubMedQA's native
  leaderboard framing — categorized as `Question_Answering`.
- `abbreviation_disambiguation` (umn_abbreviation) is word-sense
  disambiguation: the candidate expansions aren't enumerated in-prompt the
  way `nli`'s/`multilabel_cls`'s label sets are, but the task is
  fundamentally "pick the correct sense given context" — categorized as
  `Classification`.

**Relationship to other groupings elsewhere in this README** (none of these
are taxonomy categories — they're orthogonal properties that happen to
overlap with `Information_Extraction`):

- The [Data balance](#data-balance) section's informal "extraction/
  regression" vs. "QA/classification/summarization" split maps onto
  `Information_Extraction` + `Similarity_Regression` vs.
  `Question_Answering` + `Classification` + `Summarization` — note
  `sts_regression` is `Similarity_Regression`, not `Information_Extraction`;
  the older prose grouped it with the extraction tasks by volume profile,
  not task type.
- The "BRAT-based datasets" cluster in [Known limitations](#known-limitations)
  (`ner_re_medication`, `medication_event`, `sdoh_event`) is a
  source-annotation-format fact, entirely a subset of
  `Information_Extraction`.
- The "document-level tasks" cluster (`DOC_LEVEL_TASKS` in
  `multitask/hf_dataset_stats.py`) is an example-granularity/build property,
  also entirely within `Information_Extraction`.

## Data balance

`build_unified_dataset.py` builds `train_unified.jsonl` by concatenating
every dataset's train split and shuffling once — **there is no balancing,
weighting, capping, or oversampling logic anywhere** in this repo. That's a
real problem here: dataset size varies by three orders of magnitude, and
document length varies by another order of magnitude on top of that.

**Per-dataset train counts, tiktoken `cl100k_base` token-length distribution
(min/median/mean/max over the full rendered example), and total token
volume:**

| Dataset | Task tag | Train | tk_min | tk_median | tk_mean | tk_max | Train tokens (≈) |
|---|---|---|---|---|---|---|---|
| discharge-me | summarization | 68,785 | 588 | 4088.0 | 4303.9 | 20469 | 296.0M |
| pubmedqa | qa_yesnomaybe | 211,269 | 94 | 445.0 | 449.7 | 1671 | 95.0M |
| medmcqa | mcqa_explain | 182,822 | 70 | 170.0 | 207.9 | 5419 | 38.0M |
| 2018_n2c2_track1 (cohort_eligibility) | cohort_eligibility | 2,626 | 1748 | 5096.5 | 5401.3 | 15312 | 14.2M |
| covid_LitCovid | multilabel_cls | 24,960 | 47 | 332.0 | 343.5 | 2063 | 8.6M |
| MedQA-USMLE-Mainland | mcqa | 27,400 | 73 | 151.0 | 168.6 | 687 | 4.6M |
| umn_abbreviation | abbreviation_disambiguation | 19,257 | 86 | 162.0 | 166.2 | 1004 | 3.2M |
| MedQA-USMLE-US | mcqa | 10,178 | 86 | 236.5 | 247.5 | 941 | 2.5M |
| MedQA-USMLE-Taiwan | mcqa | 11,298 | 72 | 182.0 | 200.7 | 1349 | 2.3M |
| bionli | nli | 5,544 | 108 | 389.0 | 395.7 | 1454 | 2.2M |
| 2018_n2c2_track2 (Drug-ADE) | ner_re_medication | 303 | 376 | 5135.0 | 5463.5 | 17404 | 1.7M |
| nli4ct | nli | 1,700 | 73 | 439.0 | 521.7 | 2786 | 0.89M |
| MedNLI | nli | 11,232 | 52 | 74.0 | 78.8 | 393 | 0.89M |
| 2022n2c2_track2_uw (SHAC, UW subset) | sdoh_event | 1,751 | 57 | 477.0 | 496.6 | 1589 | 0.87M |
| 2022n2c2_track1 (CMED) | medication_event | 350 | 286 | 1494.0 | 1902.2 | 8352 | 0.67M |
| 2022n2c2_track2_mimic (SHAC, MIMIC subset) | sdoh_event | 1,316 | 56 | 485.5 | 496.3 | 1592 | 0.65M |
| semeval_2015 (SemEval-2015 T14) | disorder_normalization | 298 | 160 | 1304.5 | 1985.1 | 7242 | 0.59M |
| 2019_n2c2_track1_cls (Clinical STS, derived) | sts_classification | 1,654 | 200 | 232.0 | 241.3 | 458 | 0.40M |
| 2019_n2c2_track1 (Clinical STS) | sts_regression | 1,654 | 92 | 123.0 | 132.9 | 349 | 0.22M |
| 2019_n2c2_track2 (family history) | family_history | 99 | 282 | 786.0 | 869.1 | 2418 | 0.09M |
| MMLU-Pro-Health | mcqa | 0 (test-only, 823) | 70 | 165.0 | 183.6 | 1084 | 0 |

Total ≈ 474M train tokens across ~585K examples.

**Two distinct, compounding imbalances:**

1. **Example-count imbalance**: pubmedqa (211,269) vs. family_history (99)
   ≈ **2,134:1**. The eight datasets that are the paper's core clinically-rich
   extraction/regression tasks (`cohort_eligibility`, `ner_re_medication`,
   `sts_regression`, `family_history`, `medication_event`,
   `sdoh_event` (mimic + uw), `disorder_normalization`) each have well under
   3,000 train examples, while three QA/classification/summarization
   datasets (pubmedqa, medmcqa, discharge-me) each have 25,000–211,000, and
   `abbreviation_disambiguation` (umn_abbreviation, 19,257) sits in between —
   the two new NLI datasets (`bionli` 5,544; `nli4ct` 1,700) also land toward
   the smaller end, well below the QA/classification/summarization tier.
2. **Token-volume imbalance is worse than example count suggests**, because
   average document length varies ~69x across datasets (2018_n2c2_track2's
   mean 5463.5 tokens vs. MedNLI's mean 78.8). `discharge-me`
   alone is **~296M tokens — about 63% of the entire merged training set's
   token volume**. `pubmedqa` + `medmcqa` + `discharge-me` together are
   **~91% of all training tokens**, while the eight small extraction/
   regression datasets above sum to only **~18.9M tokens (~4%)**.
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
   resampling targets — discharge-me's much-longer-than-average documents
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
run_eval("2018_n2c2_track2", {"100130": '{"drugs": [...], "unlinked": [...]}', ...})
run_eval("MedNLI", {"some-id": '{"label": "entailment"}', ...})
run_eval("discharge-me", {"some-id": "DISCHARGE INSTRUCTIONS:\n...\n\nBRIEF HOSPITAL COURSE:\n...", ...})
```

Dataset names match the `Dataset` column above (`2018_n2c2_track2`,
`2018_n2c2_track1`, `2019_n2c2_track1`, `2019_n2c2_track1_cls`,
`2019_n2c2_track2`, `2022n2c2_track1`,
`2022n2c2_track2_mimic`, `2022n2c2_track2_uw`, `discharge-me`, `MedNLI`,
`bionli`, `nli4ct`, `MedQA-USMLE-US/Mainland/Taiwan`,
`MMLU-Pro-Health`, `medmcqa`, `pubmedqa`, `covid (LitCovid)`, `semeval_2015`,
`umn_abbreviation`, `emrqa-msquad`). Completions may
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
table above exactly (e.g. 2018_n2c2_track2 tags F1 0.9752, medication_event
Combined F1 0.9479, SHAC mimic Combined F1 0.9862 /
uw Combined F1 0.9872, the uw derived-relations auxiliary axis F1 0.9826, STS
Pearson r = 1.0000, cohort_eligibility 1.0 except the documented KETO-1YR
artifact, discharge-me perfect ROUGE/BLEU on the full 10,962-row test split,
semeval_2015 strict F1 0.9150 / relaxed F1 1.0, umn_abbreviation accuracy 1.0,
MedNLI/bionli/nli4ct accuracy 1.0) — the dispatcher itself introduces no
discrepancy versus calling each module directly.

### Doc-level, offset-free axis (addon, not a replacement)

Every offset-based scorer above (`brat_scorer`, `convert_back/i2b2_2010.py`,
`convert_back/disorder_normalization_2015_t14.py`) has to resolve predicted
text back to a character span (`reanchor()`/`brat.align_entity_text()`)
before it can compare it to gold. `multitask/eval/doc_level.py` adds a
second, additive signal for the 5 tasks whose targets carry span text --
`ner_re_medication`, `medication_event`, `sdoh_event`,
`concept_assertion_relation`, `disorder_normalization` -- that skips span
resolution entirely: per document, it builds plain Python `set`s of tuples
(one axis for bare entities/events, a separate axis for relations/links/
roles/attributes) from the JSON target dicts directly, and scores by exact
set membership (text compared case-insensitively, whitespace-collapsed).
This is exactly how `2019_n2c2_track2/eval.py` already scores
`family_history` (no offsets in that task's target to begin with, so it
needs no module here) -- `doc_level.py` generalizes that same idea to the
other IE tasks, so "did the model name the right concept/relation/event" can
be measured independent of whether its span happened to re-anchor.

```bash
python -m multitask.eval.run_eval 2018_n2c2_track2 completions.json --doc-level
```

runs the normal offset-based scorer AND prints the doc-level axes
afterward. Or call it directly:

```python
from multitask.eval.run_eval import run_doc_level_eval
run_doc_level_eval("2018_n2c2_track2", {"100130": '{"drugs": [...], "unlinked": [...]}', ...})
```

Self-check (gold vs. itself, every axis must be exactly P=R=F1=1.0 -- a
wiring smoke test, not a meaningful accuracy number):

```bash
python -m multitask.eval.doc_level
```

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

python -m multitask.eval.simple_tasks   # nli / mcqa / mcqa_explain / qa_yesnomaybe / multilabel_cls / abbreviation_disambiguation, all 11 datasets
python -m multitask.eval.discharge_me   # summarization, 25-sample gold self-check
python -m multitask.eval.qa_span        # qa_span_extraction (emrqa-msquad), gold self-check
```

### Calling a single converter/scorer directly

`run_eval()` is a thin dispatcher — nothing it does is unavailable by calling
the underlying building blocks yourself, if you want finer control (e.g.
scoring a subset, or intercepting the intermediate native-format files):

```python
# BRAT-based datasets (ner_re_2018_t2 / medication_event_2022_t1 / sdoh_2022_t2)
from multitask.convert_back.medication_event_2022_t1 import write_anns, score_dirs, GOLD_DIR
from multitask.io_utils import read_text

predictions = [(doc_id, model_output_json, read_text(GOLD_DIR / f"{doc_id}.txt"))
               for doc_id, model_output_json in your_completions.items()]
with tempfile.TemporaryDirectory() as tmp:
    write_anns(predictions, tmp)
    score_dirs(GOLD_DIR, tmp)   # prints multitask.brat_scorer's per-axis P/R/F1

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

- **Prompt-clarity pass, 2026-09-24: five instructions closed real gaps
  between what the model is told and what it's expected to produce.**
  Checked every rendered instruction against the actual `huggingface_format`
  data (not just the source in `schemas.py`) for label values the model has
  to guess at. Fixed: (1) `concept_assertion_relation` (+ its `_re_only` and
  `_section` variants) listed the 8 i2b2 relation codes (`TrIP`, `TeRP`,
  `PIP`, ...) with zero definitions -- verified against the released
  `2010i2b2/Relation Annotation Guideline.pdf` and added a one-clause gloss
  per code (e.g. "TrAP (treatment administered for problem, outcome not
  stated)"). (2) `sdoh_event`'s `Type` role for Tobacco/Alcohol/Drug
  triggers was never defined at all (Employment's and LivingStatus's `Type`
  were); confirmed via the real corpus that it means the specific substance
  ("beer"/"wine", "cocaine"/"marijuana", "cigarettes") and added that.
  (3) Same instruction listed `Method` as a uniform role across
  Tobacco/Alcohol/Drug when it's overwhelmingly Drug-only in the data
  (measured: Drug 903 vs. Tobacco 49 vs. Alcohol 2 occurrences) -- noted
  that skew so the model isn't nudged to invent a Method for Alcohol.
  (4) `medication_event_attr_only` named its 5 attribute dimensions
  (Action/Certainty/.../Negation) but never enumerated their legal values,
  unlike the parent `medication_event` task teaching the identical skill --
  now matches. (5) `summarization` (discharge-me) never told the model the
  literal `"DISCHARGE INSTRUCTIONS:\n...\n\nBRIEF HOSPITAL COURSE:\n..."`
  header format that `eval/discharge_me.py`'s `parse_sections()` requires to
  split the two scored sections apart -- a model whose sampled output
  drifted from that exact format (case, spacing) would have silently scored
  0 on both sections with no error. Now stated explicitly as the output
  contract.
- **Residual text-ambiguity ceiling on the BRAT datasets** (`ner_re_medication`,
  `medication_event`, `sdoh_event`): a small fraction of gold entities are
  short/common tokens ("2", "in", "iv") repeated many times per document with
  no positional signal to disambiguate from text alone. This affects a real
  generative model identically, so a perfect gold round-trip doesn't hit
  1.0 on every axis — `sdoh_event` lands at Combined F1 0.9862 (mimic) / 0.9872
  (uw), `ner_re_medication` at tags F1 0.9750 / relations F1 0.9608, and
  `medication_event` at Attributes F1 0.9146 / Combined F1 0.9479.

  **Corrected 2026-09-23:** this section previously reported
  `ner_re_medication` at tags F1 0.9429 / relations F1 0.9039 and attributed
  the whole gap to inherent ambiguity. That was wrong — most of it was a
  bug. 1,460 gold entities in this corpus carry a `.ann` text column that is
  not a substring of the document because the span is split into *adjacent*
  fragments with no gap (`T17 Drug 7829 7838;7838 7839  Prednison e`, where
  the document plainly reads `Prednisone`) or otherwise gains a space the
  document lacks (`0.02  %` vs `0.02 %`). `reanchor._flex_pattern` joins
  mention tokens with `\s+`, which requires at least one whitespace
  character and so can never match these; 316 of 10,575 Drug mentions
  therefore failed to re-anchor, and because this dataset's target nests
  attributes under their drug, each failure silently discarded every
  attribute beneath it (1,543 fillers lost to that cascade). Fixed by
  `brat.align_entity_text()`, which re-slices an entity's text from the
  document when the slice contains no line break — see that function's
  docstring for why the newline-crossing case (3,206 entities here, 1,235 in
  SHAC, 6 in CMED) must be left alone instead. Recall went 0.9297 → 0.9915
  on tags and 0.8707 → 0.9608 on relations.

  **Second correction, same day:** an unalignable prediction (in `to_ann`,
  or in `convert_back/i2b2_2010.py`'s dict-based scorer) used to be dropped
  from the system's output entirely rather than written anywhere -- since
  every precision denominator (`brat_scorer`'s `fp = len(sys_) - tp`,
  i2b2's `len(pred_by_id)`/`len(pred_rel_keys)`) only ever counted what
  survived, a hallucinated entity that couldn't be placed in the document
  cost nothing, making unconstrained hallucination the metric-optimal
  strategy. Every BRAT converter and `convert_back/i2b2_2010.py` now give
  an unresolvable prediction a "poison span" -- past the end of the
  document, monotonically increasing so none collide -- instead of
  dropping it: still a syntactically real prediction (so every scorer,
  including the official cross-check scripts, counts it), but guaranteed
  to never overlap a real gold span, so it can only ever be a false
  positive, never a false true-positive. On a gold round-trip this
  necessarily makes precision equal recall exactly (every case that used to
  cost only a miss now also costs a wrong guess): `ner_re_medication`
  tags/relations moved 0.9752/0.9609 → 0.9750/0.9608 (2018 t2's own
  reanchor fix above already resolved most of its residual failures, so
  the remaining cost here is tiny); `medication_event` moved 0.9848
  triggers/0.9356 attributes/0.9611 combined → 0.9792/0.9146/0.9479 (CMED's
  ambiguity was never the reanchor bug, so this fix is where its real cost
  shows up); `sdoh_event` moved 0.9955/0.9819/0.9867 (mimic) →
  0.9951/0.9813/0.9862 and 0.9959/0.9826/0.9874 (uw) → 0.9956/0.9825/0.9872.
  The `disorder_normalization` and `concept_assertion_relation` rows below
  have the same fix applied; see their own entries for numbers.

  All numbers come from `multitask/brat_scorer.py`, the self-implemented greedy
  bipartite matcher shared by all three BRAT datasets (see its module
  docstring) — replacing each dataset's separate official/vendored scorer,
  which used a single iteration-order-dependent matching pass instead of a
  real assignment algorithm; those original tools (`Track2-evaluate-ver4.py`,
  `eval_script.py`, vendored `brat_scoring`) are left in the repo untouched
  for a standalone leaderboard-comparable cross-check but are no longer the
  default scoring path.
- **`concept_assertion_relation` runs on the public i2b2 2010 subset, so its
  numbers are not comparable to published i2b2 2010 results.** The original
  2010 i2b2/VA challenge used 349 train / 477 test documents; the
  post-challenge public release — what this repo builds from — is 170 train
  / 256 test. Published concept-extraction figures (e.g. GatorTron 0.8996,
  Si et al. 0.9025) are all on the full challenge split and will not line up
  with anything measured here. The one comparable-subset reference point
  found is BLUE/BlueBERT, which reports relation extraction 76.4 on the
  public subset vs 73.7 on the official split. Do not put this repo's
  `concept_assertion_relation` F1 in a table next to challenge-era numbers
  without stating the split.
- **The `sdoh_event`/`2022n2c2_track2_uw` derived-relations axis is
  self-implemented and auxiliary, not part of the trainable target.** The
  model is only ever trained/scored on the event-form JSON target (train/dev
  never have gold relations to learn from); `multitask/convert_back/
  sdoh_2022_t2.py`'s `to_relations_ann()` mechanically re-derives a
  relation-form `.ann` from that same event JSON, and `score_relations()`
  (no official relation-only scorer exists in the vendored `brat_scoring`
  package, which is event-oriented) scores it against the real
  `test/uw_r/` re-annotation, 518 docs, via `multitask.brat_scorer`'s same
  greedy matcher. Verified: derived relations from gold events score
  P=0.9828/R=0.9825/F1=0.9826 against `test/uw_r/` — a
  bonus/generalization-probe axis, not an official n2c2 metric.
- **`sts_regression`'s float-regression-via-generation is an accepted, not
  fully solved, design compromise.** Free-generating a continuous 0-5 score
  is a poor fit for an LLM: numeric tokenization is inconsistent across
  otherwise-identical training examples, and decoding tends to collapse onto
  a handful of round values, which hurts a Pearson-correlation metric more
  than a classification one. The mitigation in place is modest by design
  (no constrained-decoding infra in this project): training targets are
  rendered as a fixed two-decimal string (`3.50`, not `3.5`), and
  `convert_back/sts_2019_t1.py` parses model output leniently (regex-extract
  the first float, clamp to `[0, 5]`) rather than requiring exact JSON. The
  next bullet describes `sts_classification`, the discretized alternative
  added alongside (not replacing) this task, for cases where the ordinal
  categorical framing is preferable to free-generated regression.
- **`sts_classification` is an additive, derived variant of `sts_regression`,
  built by discretizing the same sentence pairs to the nearest of the scale's
  11 half-point anchor values (`0, 0.5, 1, ..., 5`) and then grouping those
  11 anchors into 5 semantically-named ordinal categories (`dissimilar`,
  `weakly_related`, `partially_equivalent`, `mostly_equivalent`,
  `highly_equivalent`), scored as classification (accuracy/macro-P/R/F1)
  instead of Pearson correlation.** This is this project's own design
  choice, not part of the original 2019 n2c2 Track 1 release —
  `sts_regression` is left completely unchanged and remains the
  SOTA-leaderboard-comparable metric. 11 raw anchors were judged too many
  categories for a model to reliably distinguish, so they're grouped
  `{0,0.5}` / `{1,1.5,2}` / `{2.5,3}` / `{3.5,4}` / `{4.5,5}`; this grouping
  was chosen from the *pooled* train+test histogram over the 11 anchors
  (pooled class shares 13.7/23.4/25.7/26.0/11.3%), not equal-width ranges or
  per-split quantiles, because train and test have genuinely different score
  distributions (train mean 2.80/median 3, skewed mid/high; test mean
  1.77/median 1, 58% of test ≤1.0) — no fixed grouping balances both splits
  at once, and test's low-similarity skew is a real property of that split,
  not an artifact of the boundary choice. The anchor-snapping step is
  lossless for the released **test** gold (`clinicalSTS2019.test.gs.sim.txt`,
  n=412): every score already sits exactly on the 11-value grid (0/412
  off-grid). It is **not** lossless for **train**
  (`clinicalSTS2019.train.txt`, n=1654): 561 scores (34%) are fractional
  multi-annotator averages off that grid (e.g. 1.25, 1.65, 2.4, 2.9) and get
  rounded to the nearest anchor before being grouped into a category — a
  real, deliberate simplification of the training signal, not an artifact of
  a bug, and worth disclosing in any paper using this task's training
  labels. Because plain accuracy/macro-F1 treats every miss identically — a
  near-miss (predicted `mostly_equivalent`, gold `highly_equivalent`) scores
  the same as a wild miss (predicted `dissimilar`, gold `highly_equivalent`)
  — the scorer also reports **quadratic-weighted Cohen's kappa**
  (`multitask/eval/simple_tasks.py`,
  pure Python, no numpy/sklearn dependency), which penalizes far misses more
  than near ones, the same convention used for Likert/ordinal-scale scoring
  elsewhere. Verified: gold round-trip on the real test split (n=412) gives
  accuracy = macro_precision = macro_recall = macro_f1 =
  quadratic_weighted_kappa = 1.0.
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
- **`family_history` trains/scores Subtask 2 only, not both released
  subtasks merged into one target.** The 2019 n2c2 Track 2 release defines
  Subtask 1 (bare family-member/observation identification) and Subtask 2
  (the task's own "end-to-end family history summarization" framing:
  member + side + living-status + observation-with-negation, combined). An
  earlier version of this builder asked the model to predict both as one
  JSON object, but Subtask 1's fields are largely redundant with Subtask
  2's own member/side attachments (measured: 930 Subtask-1 observation
  strings vs. 740 Subtask-2 ones on train) -- two different tasks in one
  generation target, not one coherent end-to-end task. Cost of dropping
  Subtask 1: family members mentioned in the note but never assigned a
  living-status score or observation become invisible to this target
  (10.5% of train / 9.9% of test Subtask-1 family members have no Subtask-2
  counterpart) -- accepted, since those members aren't part of the
  "end-to-end family history" Subtask 2 itself is scored on. See
  `multitask/builders/family_history_2019_t2.py`'s module docstring for the
  full rationale.
- **`disorder_normalization`'s strict/relaxed F metric is self-implemented,
  not an official binary.** No persistent public scorer for SemEval-2015
  Task 14 could be found (the task's own validation script was only
  available "during the evaluation period," with no lasting public link).
  `multitask/convert_back/disorder_normalization_2015_t14.py` implements
  strict F (exact span + correct CUI) and relaxed F (word-overlap + correct
  CUI) directly from the formulas in Elhadad et al. 2015 §4.1. Gold spans
  come from the builder's own released offsets, never reanchored; predicted
  spans (including gold-as-prediction in the self-check) go through
  reanchor() like every other BRAT-family module — so gold round-tripped
  through the converter hits relaxed F1 = 1.0 (span-independent, word-
  overlap only) but strict F1 = 0.9150, not 1.0. That's real, not a bug:
  mostly discontinuous mentions (8.0% of test entities), whose space-joined
  reconstruction can never re-match a single contiguous span the way the
  original discontinuous BRAT-style annotation intended. **Corrected
  2026-09-23:** this previously reported strict F1 = relaxed F1 = 1.0,
  because gold was reanchored through the exact same lossy path as
  predictions -- comparing a lossy reconstruction against an equally lossy
  reconstruction rather than against the truth, so it could not fail for
  any span reason. Treat the current numbers as a faithful reimplementation
  of the paper's metric, not the organizers' own code.
- **`umn_abbreviation` has no released train/test split** — the UMN release
  ships `train.txt`/`valid.txt`/`test.txt` pre-split, but the corpus itself
  defines no official benchmark split beyond that; splits are used as-is.
  The source file is Latin-1 encoded (not UTF-8, confirmed by decode
  failure) and its released `start`/`end` columns are offsets into the
  original full document rather than the context snippet shown to the
  model, so they're dropped rather than used for span alignment.
- **`emrqa-msquad`'s train and validation (`dev`) splits share the same
  underlying document pool** — all 253 unique contexts in the release
  appear in both splits, and 1,885 `(question, context)` pairs (~5.8% of
  the 32,739-row validation set) are byte-identical between train and
  validation. This is an inherent property of the dataset as released
  upstream (every question was generated over the same fixed pool of 253
  summarized medical notes), not something introduced by this repo's
  builder — documented here rather than deduplicated, consistent with how
  every other inherent dataset quirk in this section is handled.

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
 "task": str, "category": str, "dataset": str, "id": str}
```

so `tokenizer.apply_chat_template()` can be applied directly. Built by `multitask/build_hf_datasets.py` (`python -m multitask.build_hf_datasets`), from the same per-task builders in `multitask/builders/` that also feed the separate merged flat-text pipeline (`multitask/build_unified_dataset.py`). Reload any dataset with:

```python
from datasets import load_from_disk
ds = load_from_disk("multitask/huggingface_format/<name>")
```

Stats below (example counts, tiktoken `cl100k_base` token-length of the full rendered example -- min/median/mean/max, per split where that split exists) are generated by `multitask/hf_dataset_stats.py` (`python -m multitask.hf_dataset_stats`) -- see `multitask/huggingface_format/dataset_stats.json` for full per-split percentiles (p95/p99) and Qwen3-4B token counts.

### Datasets

| dataset | category | task tag | description | train | dev | test | tk_min (train) | tk_median (train) | tk_mean (train) | tk_max (train) | tk_min (dev) | tk_median (dev) | tk_mean (dev) | tk_max (dev) | tk_min (test) | tk_median (test) | tk_mean (test) | tk_max (test) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2010_i2b2_va | information_extraction | concept_assertion_relation | Extract all problem/test/treatment concepts; for each problem concept also give its assertion status (present, absent, possible, conditional, hypothetical, or associated_with_someone_else); and extract the relations between concepts, one of: TrIP (treatment improves problem), TrWP (treatment worsens problem), TrCP (treatment causes problem), TrAP (treatment administered for problem, outcome not stated), TrNAP (treatment NOT administered because of problem), TeRP (test reveals problem), TeCP (test conducted to investigate problem, outcome not known), PIP (one problem indicates/causes another). Return JSON {"concepts": [{"id", "type", "text", "assertion"}, ...], "relations": [{"type", "arg1", "arg2"}, ...]}. | 170 | 77 | 256 | 273 | 4077.0 | 4740.5 | 19611 | 576 | 4673.0 | 5484.0 | 14100 | 474 | 5115.5 | 5813.8 | 23982 |
| 2010_i2b2_va_ner_only | information_extraction | concept_assertion_relation_ner_only | Extract every problem, test, and treatment concept mention span in the text -- concepts only, no assertion classification, no relation linking. Return JSON {"concepts": [{"type": ..., "text": ...}, ...]}. | 170 | - | - | 155 | 2579.5 | 2884.3 | 12173 | - | - | - | - | - | - | - | - |
| 2010_i2b2_va_re_only | information_extraction | concept_assertion_relation_re_only | This note's problem/test/treatment concepts have already been extracted and are listed after the note text under EXTRACTED_CONCEPTS. For each problem concept, give its assertion status (present, absent, possible, conditional, hypothetical, or associated_with_someone_else); and extract the relations between concepts, one of: TrIP (treatment improves problem), TrWP (treatment worsens problem), TrCP (treatment causes problem), TrAP (treatment administered for problem, outcome not stated), TrNAP (treatment NOT administered because of problem), TeRP (test reveals problem), TeCP (test conducted to investigate problem, outcome not known), PIP (one problem indicates/causes another). Return JSON {"concepts": [{"id", "type", "text", "assertion"}, ...], "relations": [{"type", "arg1", "arg2"}, ...]}. | 170 | - | - | 303 | 4715.0 | 5502.0 | 22928 | - | - | - | - | - | - | - | - |
| 2010_i2b2_va_section | information_extraction | concept_assertion_relation_section | This is one section excerpted from a larger clinical note (e.g. "HISTORY OF PRESENT ILLNESS", "PAST MEDICAL HISTORY"), not the complete note. Extract every problem, test, and treatment concept in THIS EXCERPT ONLY, give each problem concept's assertion status (present, absent, possible, conditional, hypothetical, or associated_with_someone_else), and extract the relations between concepts, one of: TrIP (treatment improves problem), TrWP (treatment worsens problem), TrCP (treatment causes problem), TrAP (treatment administered for problem, outcome not stated), TrNAP (treatment NOT administered because of problem), TeRP (test reveals problem), TeCP (test conducted to investigate problem, outcome not known), PIP (one problem indicates/causes another). Return JSON {"concepts": [{"id", "type", "text", "assertion"}, ...], "relations": [{"type", "arg1", "arg2"}, ...]}. | 1666 | 860 | 2868 | 301 | 450.0 | 713.7 | 9795 | 299 | 468.0 | 787.7 | 10019 | 296 | 476.5 | 754.3 | 10019 |
| 2018_n2c2_track1 | classification | cohort_eligibility | Given the patient's longitudinal clinical notes and ONE named clinical trial eligibility criterion (with its definition), decide whether the patient meets that single criterion. Return JSON {"label": "met"} or {"label": "not met"}. | 2626 | 335 | 1118 | 1785 | 5133.5 | 5438.3 | 15349 | 2108 | 4889.0 | 5331.5 | 11747 | 2099 | 5136.0 | 5422.0 | 11755 |
| 2018_n2c2_track2 | information_extraction | ner_re_medication | Task: Extract every Drug mention in this note and its linked clinical attributes, per the 2018 n2c2 Track 2 ADE/medication-extraction schema (Henry et al. 2020, JAMIA).
Definitions: Drug = a medication name (generic, brand, or drug class). Strength = amount of active ingredient per dose unit (e.g. "40 mg"). Form = the drug's physical form (e.g. tablet, patch, cream). Dosage = amount given per administration (e.g. "2 tablets"). Frequency = how often it's taken (e.g. "twice daily"). Route = how it enters the body (e.g. "PO", "IV", "topically"). Duration = how long it is/was taken. Reason = the medical indication for the drug. ADE = an adverse reaction attributed to the drug. Each attribute links to the Drug mention(s) it describes -- any attribute type may link to more than one Drug, most often Reason/ADE (e.g. one shared side effect attributed to two co-administered drugs).
OutputFormat: Nest each attribute mention under its linked Drug, keyed by attribute type -- omit attribute types with no mentions; list any attribute mention not linked to any drug under "unlinked". Return JSON {"drugs": [{"text": ..., "<AttrType>": [...], ...}, ...], "unlinked": [{"type", "text"}, ...]}. | 303 | 61 | 202 | 618 | 5375.0 | 5702.1 | 17637 | 661 | 4879.0 | 5104.9 | 11248 | 632 | 5385.0 | 5590.5 | 16069 |
| 2018_n2c2_track2_ner_only | information_extraction | ner_re_medication_ner_only | Extract every Drug, Strength, Frequency, Dosage, Route, Duration, Form, Reason, and ADE mention span in the text -- entities only, no relation linking. Return JSON {"entities": [{"type": ..., "text": ...}, ...]}. | 303 | - | - | 389 | 6194.0 | 6416.7 | 19776 | - | - | - | - | - | - | - | - |
| 2018_n2c2_track2_paragraph | information_extraction | ner_re_medication_paragraph | This is one paragraph excerpted from a larger clinical note, not the complete note. Extract every Drug, Strength, Frequency, Dosage, Route, Duration, Form, Reason, and ADE mention in THIS EXCERPT ONLY and link each non-Drug entity to the Drug mention it describes. Return JSON {"drugs": [{"text": ..., "<AttrType>": [...], ...}, ...], "unlinked": [{"type", "text"}, ...]}. | 2954 | 580 | 1935 | 173 | 367.0 | 514.7 | 5079 | 173 | 361.5 | 545.4 | 5189 | 173 | 355.0 | 512.9 | 5189 |
| 2018_n2c2_track2_re_only | information_extraction | ner_re_medication_re_only | This note's Drug/Strength/Frequency/Dosage/Route/Duration/Form/Reason/ADE entities have already been extracted and are listed after the note text under EXTRACTED_ENTITIES. Link each non-Drug entity to the Drug mention it describes (an entity may link to more than one Drug); collect any entity never linked to a Drug under "unlinked". Return JSON {"drugs": [{"text": ..., "<AttrType>": [...], ...}, ...], "unlinked": [{"type", "text"}, ...]}. | 303 | - | - | 462 | 6484.0 | 6717.4 | 20493 | - | - | - | - | - | - | - | - |
| 2019_n2c2_track1 | similarity_regression | sts_regression | Rate the semantic similarity of the two clinical sentences on a 0-5 scale (0 = unrelated, 5 = semantically equivalent), to two decimal places. Return JSON {"similarity": <float, e.g. 3.50>}. | 1654 | 124 | 412 | 130 | 161.0 | 170.9 | 387 | 131 | 147.0 | 148.6 | 187 | 130 | 148.0 | 148.7 | 194 |
| 2019_n2c2_track1_cls | classification | sts_classification | Classify the semantic similarity of the two clinical sentences into one of 5 categories: "dissimilar" (unrelated, or share no clinically meaningful information), "weakly_related" (same general topic but very different clinical content), "partially_equivalent" (share important clinical information but differ in other important details), "mostly_equivalent" (same essential clinical meaning, only minor/unimportant details differ), or "highly_equivalent" (semantically equivalent or nearly so). Return JSON {"category": <one of the 5 values above, as a string>}. | 1654 | 124 | 412 | 197 | 229.0 | 238.3 | 455 | 198 | 214.0 | 215.8 | 254 | 197 | 215.5 | 215.9 | 261 |
| 2019_n2c2_track2 | information_extraction | family_history | Task: Extract family history from the note as an end-to-end summary of member/side/living-status/observation, per the 2019 n2c2 Track 2 Subtask 2 schema (Liu et al. 2018, BioCreative/OHNLP).
Definitions: FamilyMember is one of Father/Mother/Parent/Sister/Brother/Daughter/Son/Child/Grandmother/Grandfather/Grandparent/Cousin/Sibling/Aunt/Uncle, each with a side of family (Paternal/Maternal/NA -- first-degree relatives are always NA). Observation is a disease/condition mentioned for a family member. LivingStatus score = alive_score * healthy_score, each in {Yes=2, NA=1, No=0} (so 4 = alive and healthy, 0 = deceased). Only report a member here if the note gives a living-status score and/or at least one observation for them -- a member mentioned with neither is not part of this summary.
OutputFormat: Return JSON with 2 keys: "living_status" (list of {"member", "side", "score"}), "observations" (list of {"member", "side", "text", "negation": "Negated"|"Non_Negated"}). | 98 | 32 | 107 | 437 | 899.0 | 952.0 | 2307 | 571 | 883.0 | 908.2 | 1619 | 437 | 916.0 | 974.0 | 2195 |
| 2022n2c2_track1 | information_extraction | medication_event | Task: For each medication mention, classify whether a medication change is discussed, per the CMED schema (Mahajan et al. 2021).
Definitions: Disposition = a change is discussed (e.g. "start Plavix"). NoDisposition = no change is discussed (e.g. "continue lisinopril"). Undetermined = unclear from local context (e.g. "Plan: Lasix"). For each Disposition mention, also classify: Action -- Start/Stop/Increase/Decrease/UniqueDose/OtherChange/Unknown; Certainty -- Certain/Hypothetical/Conditional/Unknown; Temporality -- Past/Present/Future/Unknown; Actor -- Physician/Patient/Unknown; Negation -- Negated/NotNegated. A single mention may have 0, 1, or more Disposition events.
OutputFormat: "trigger_text" is the medication mention itself (the entity type in this task is always a medication -- "trigger" names its role as what triggers the event, per this schema's event-extraction terminology, not a different kind of span). Return JSON {"events": [{"id", "type", "trigger_text", "attributes": {"Action", "Certainty", "Temporality", "Actor", "Negation"}}, ...]} -- "attributes" is empty for NoDisposition/Undetermined events. | 350 | 50 | 100 | 563 | 1771.0 | 2179.2 | 8629 | 886 | 2038.0 | 2409.9 | 5092 | 643 | 2040.0 | 2238.1 | 5718 |
| 2022n2c2_track1_attr_only | information_extraction | medication_event_attr_only | This note's medication triggers have already been identified and are listed after the note text under EXTRACTED_TRIGGERS, each with its disposition type. For each Disposition trigger, classify: Action -- Start/Stop/Increase/Decrease/UniqueDose/OtherChange/Unknown; Certainty -- Certain/Hypothetical/Conditional/Unknown; Temporality -- Past/Present/Future/Unknown; Actor -- Physician/Patient/Unknown; Negation -- Negated/NotNegated. Return JSON {"events": [...]}. | 350 | - | - | 418 | 1723.0 | 2169.3 | 8924 | - | - | - | - | - | - | - | - |
| 2022n2c2_track1_trigger_only | information_extraction | medication_event_trigger_only | Extract every medication mention span and classify its disposition type (Disposition, NoDisposition, or Undetermined) -- triggers only, no attribute classification. Return JSON {"triggers": [{"type": ..., "text": ...}, ...]}. | 350 | - | - | 309 | 1297.0 | 1596.8 | 6922 | - | - | - | - | - | - | - | - |
| 2022n2c2_track2_mimic | information_extraction | sdoh_event | Task: Extract social-determinants-of-health (SDOH) events from this note, per the SHAC annotation schema (Lybarger et al., 2022 n2c2 Track 2).
Definitions: an SDOH event is a trigger span (Employment, LivingStatus, Tobacco, Alcohol, or Drug) plus role-labeled argument spans. Employment roles: Status, Type (occupation), Duration, History. LivingStatus roles: Status, Type (alone/with_family/with_others/homeless), Duration, History. Tobacco/Alcohol/Drug roles: Status, Type (the specific substance, e.g. "beer"/"wine" for Alcohol, "cocaine"/"marijuana" for Drug, "cigarettes" for Tobacco), Amount, Frequency, Method (how it's consumed, e.g. "smoke"/"inject" -- mostly Drug-only; almost never annotated for Alcohol and rare for Tobacco, since smoking is the assumed default for both), Duration, History -- Drug covers illicit drugs/marijuana/prescription misuse only (not medical use), Tobacco covers first-hand smoking only. Status is one of none/current/past/future for every trigger type (some values are rare for a given type -- e.g. "future" is rare for Tobacco/Alcohol/Drug, "none" is rare for LivingStatus -- but none is impossible for any of them). Each role filler is {"type", "text"}: type is usually the role's own name, but a Status filler's type is StatusTime (Tobacco/Alcohol/Drug/LivingStatus) or StatusEmploy (Employment), and LivingStatus's Type filler's type is TypeLiving. If a role has more than one filler, number the extras: Type2/Type3, Amount2/Amount3, Duration2, Frequency2, History2, Method2/Method3. Status/Type fillers carry their own attribute value keyed by name -- StatusTimeVal (none/current/past/future), StatusEmployVal (employed/unemployed/retired/on_disability/student/homemaker), or TypeLivingVal (alone/with_family/with_others/homeless).
OutputFormat: Return JSON {"events": [{"id", "type", "trigger_text", "roles": {"<RoleName>": {"type", "text"}, ...}, "attributes": [{"role", "name", "value"}, ...]}, ...]}. | 1316 | 188 | 373 | 560 | 989.5 | 1000.2 | 2096 | 560 | 900.0 | 897.5 | 1570 | 560 | 899.0 | 891.4 | 1394 |
| 2022n2c2_track2_uw | information_extraction | sdoh_event | Task: Extract social-determinants-of-health (SDOH) events from this note, per the SHAC annotation schema (Lybarger et al., 2022 n2c2 Track 2).
Definitions: an SDOH event is a trigger span (Employment, LivingStatus, Tobacco, Alcohol, or Drug) plus role-labeled argument spans. Employment roles: Status, Type (occupation), Duration, History. LivingStatus roles: Status, Type (alone/with_family/with_others/homeless), Duration, History. Tobacco/Alcohol/Drug roles: Status, Type (the specific substance, e.g. "beer"/"wine" for Alcohol, "cocaine"/"marijuana" for Drug, "cigarettes" for Tobacco), Amount, Frequency, Method (how it's consumed, e.g. "smoke"/"inject" -- mostly Drug-only; almost never annotated for Alcohol and rare for Tobacco, since smoking is the assumed default for both), Duration, History -- Drug covers illicit drugs/marijuana/prescription misuse only (not medical use), Tobacco covers first-hand smoking only. Status is one of none/current/past/future for every trigger type (some values are rare for a given type -- e.g. "future" is rare for Tobacco/Alcohol/Drug, "none" is rare for LivingStatus -- but none is impossible for any of them). Each role filler is {"type", "text"}: type is usually the role's own name, but a Status filler's type is StatusTime (Tobacco/Alcohol/Drug/LivingStatus) or StatusEmploy (Employment), and LivingStatus's Type filler's type is TypeLiving. If a role has more than one filler, number the extras: Type2/Type3, Amount2/Amount3, Duration2, Frequency2, History2, Method2/Method3. Status/Type fillers carry their own attribute value keyed by name -- StatusTimeVal (none/current/past/future), StatusEmployVal (employed/unemployed/retired/on_disability/student/homemaker), or TypeLivingVal (alone/with_family/with_others/homeless).
OutputFormat: Return JSON {"events": [{"id", "type", "trigger_text", "roles": {"<RoleName>": {"type", "text"}, ...}, "attributes": [{"role", "name", "value"}, ...]}, ...]}. | 1751 | 259 | 518 | 560 | 981.0 | 1000.1 | 2092 | 563 | 918.0 | 908.0 | 1900 | 560 | 910.0 | 903.1 | 1744 |
| MMLU-Pro-Health | question_answering | mcqa | Answer the multiple-choice question by selecting the correct option letter. Return JSON {"answer": <letter>}. | - | - | 823 | - | - | - | - | - | - | - | - | 111 | 206.0 | 224.6 | 1125 |
| MedNLI | classification | nli | Determine the relationship between the premise and hypothesis: entailment, contradiction, or neutral. Return JSON {"label": ...}. | 11232 | 1395 | 1422 | 91 | 113.0 | 117.8 | 432 | 92 | 113.0 | 119.5 | 247 | 92 | 112.0 | 116.4 | 223 |
| MedQA-USMLE-Mainland | question_answering | mcqa | Answer the multiple-choice question by selecting the correct option letter. Return JSON {"answer": <letter>}. | 27400 | 3425 | 3426 | 114 | 192.0 | 209.6 | 728 | 114 | 193.0 | 210.3 | 706 | 118 | 191.0 | 210.4 | 572 |
| MedQA-USMLE-Taiwan | question_answering | mcqa | Answer the multiple-choice question by selecting the correct option letter. Return JSON {"answer": <letter>}. | 11298 | 1412 | 1413 | 113 | 223.0 | 241.7 | 1390 | 115 | 224.0 | 242.2 | 702 | 115 | 225.0 | 245.8 | 825 |
| MedQA-USMLE-US | question_answering | mcqa | Answer the multiple-choice question by selecting the correct option letter. Return JSON {"answer": <letter>}. | 10178 | 1272 | 1273 | 127 | 277.5 | 288.5 | 982 | 130 | 277.0 | 288.4 | 706 | 140 | 281.0 | 294.7 | 971 |
| bionli | classification | nli | Determine the relationship between the premise and hypothesis: entailment, contradiction, or neutral. Return JSON {"label": ...}. | 5544 | 12806 | 6308 | 147 | 428.0 | 434.7 | 1493 | 162 | 430.0 | 433.5 | 1023 | 191 | 445.0 | 452.9 | 1110 |
| covid_LitCovid | classification | multilabel_cls | Assign all applicable topic labels to this article. Return JSON {"labels": [...]}. | 24960 | 6239 | 2500 | 85 | 370.0 | 381.5 | 2101 | 87 | 373.0 | 385.1 | 2354 | 89 | 403.0 | 409.4 | 1361 |
| discharge-me | summarization | summarization | Write the discharge instructions and brief hospital course for this patient. Return plain text, not JSON, in exactly this format (the scorer splits on these literal headers): "DISCHARGE INSTRUCTIONS:
<text>

BRIEF HOSPITAL COURSE:
<text>". | 68785 | 14719 | 10962 | 654 | 4154.0 | 4369.9 | 20535 | 923 | 4153.0 | 4367.3 | 20623 | 965 | 4142.5 | 4358.1 | 18519 |
| emrqa-msquad | question_answering | qa_span_extraction | Given the clinical note context and a question, extract the exact answer as a verbatim substring of the context. Return JSON {"answer": <exact substring of the context>}. | 130956 | 32739 | - | 280 | 728.0 | 761.6 | 1201 | 284 | 731.0 | 764.8 | 1199 | - | - | - | - |
| medmcqa | question_answering | mcqa_explain | Answer the multiple-choice question and explain your reasoning. Return JSON {"answer": <letter>, "explanation": ...}. | 182822 | 4183 | - | 110 | 210.0 | 247.9 | 5459 | 110 | 168.0 | 203.8 | 1122 | - | - | - | - |
| nli4ct | classification | nli | Determine the relationship between the premise and hypothesis: entailment, contradiction, or neutral. Return JSON {"label": ...}. | 1700 | 200 | 500 | 112 | 478.0 | 560.7 | 2825 | 107 | 462.5 | 516.8 | 2224 | 111 | 460.5 | 565.2 | 2895 |
| pubmedqa | question_answering | qa_yesnomaybe | Answer the research question yes, no, or maybe given the context, and give a long-form answer. Return JSON {"decision": ..., "long_answer": ...}. | 211269 | 50 | 500 | 135 | 486.0 | 490.7 | 1712 | 264 | 476.0 | 473.3 | 675 | 210 | 469.0 | 468.8 | 824 |
| semeval_2015 | information_extraction | disorder_normalization | Identify all disorder mentions in the clinical text and normalize each to a UMLS/SNOMED-CT CUI, or "CUI-less" if no matching concept exists. Return JSON {"entities": [{"text": ..., "cui": ...}, ...]}. | 298 | 40 | 133 | 204 | 1348.5 | 2029.1 | 7286 | 1427 | 3493.0 | 3850.2 | 10139 | 894 | 3264.0 | 3523.3 | 10139 |
| umn_abbreviation | classification | abbreviation_disambiguation | Given a clinical abbreviation and its surrounding context, determine what the abbreviation stands for in this specific context. Return JSON {"expansion": ...}. | 19257 | 4127 | 4127 | 126 | 202.0 | 206.2 | 1044 | 128 | 201.0 | 205.7 | 731 | 129 | 202.0 | 206.7 | 685 |

### Tier A subtask-enrichment datasets (train-only, additive)

These are **not** part of the standard per-task evaluation set. Each one re-frames a subset of an existing IE dataset's own gold annotations into an easier training-only pipeline-stage sub-task (NER-only, RE-only, trigger-only, attribute-only) -- see [`design.md`](./design.md)'s "Fixing doc-level training-data scarcity" section. They carry **no dev/test split** and **no `run_eval.py` dispatch entry**, and are deliberately **excluded from `train_unified.jsonl`** -- include or exclude one for a training run by choosing whether to `concatenate_datasets()` its directory alongside its parent's:

- Subtasks of `concept_assertion_relation`:
  - `2010_i2b2_va_ner_only` (`concept_assertion_relation_ner_only`): 170 train examples
  - `2010_i2b2_va_re_only` (`concept_assertion_relation_re_only`): 170 train examples
- Subtasks of `medication_event`:
  - `2022n2c2_track1_attr_only` (`medication_event_attr_only`): 350 train examples
  - `2022n2c2_track1_trigger_only` (`medication_event_trigger_only`): 350 train examples
- Subtasks of `ner_re_medication`:
  - `2018_n2c2_track2_ner_only` (`ner_re_medication_ner_only`): 303 train examples
  - `2018_n2c2_track2_re_only` (`ner_re_medication_re_only`): 303 train examples

### Paragraph-window datasets (HF-only, additive, not scored by the doc-level pipeline)

A distinct mechanism from the Tier A subtask-enrichment datasets above: instead of relabeling the same whole-note text, each example here is genuinely new -- one paragraph or section excerpted from a note (see [`design.md`](./design.md)'s paragraph-chunking section). Boundaries are blank lines for `ner_re_medication` (measured: splitting this way loses only ~0.24% of gold relations, ~9.7 non-empty paragraphs/doc) or ALL-CAPS section headers for `concept_assertion_relation` (that format has zero blank-line breaks; measured: 0.00% cross-section relation loss, ~7.1 non-empty sections/doc). **Both `train` and `test` splits are built** (per the "prepare all" data-prep instruction, for experimentation flexibility) but this dataset carries **no `run_eval.py` dispatch entry** and is **excluded from `train_unified.jsonl`** -- doc-level prediction/scoring remains the authoritative eval for its parent task; this is purely additive training signal today:

- Paragraph-window views of `concept_assertion_relation`:
  - `2010_i2b2_va_section` (`concept_assertion_relation_section`): 1666 train / 2868 test examples
- Paragraph-window views of `ner_re_medication`:
  - `2018_n2c2_track2_paragraph` (`ner_re_medication_paragraph`): 2954 train / 1935 test examples

### Known gaps / small training sets

Four tasks are built at **document level** (one example per note/patient, the full entity/relation/event set as target) rather than split into many single-label examples the way `cohort_eligibility` (2018_n2c2_track1) was. This makes their train sets far smaller than every other task in the collection:

- `2010_i2b2_va` (`concept_assertion_relation`): 170 train examples
- `2018_n2c2_track2` (`ner_re_medication`): 303 train examples
- `2019_n2c2_track2` (`family_history`): 98 train examples
- `2022n2c2_track1` (`medication_event`): 350 train examples
- `2022n2c2_track2_mimic` (`sdoh_event`): 1316 train examples
- `2022n2c2_track2_uw` (`sdoh_event`): 1751 train examples

By comparison, `cohort_eligibility` (2018_n2c2_track1) was deliberately split per-criterion into 2,626 train examples from the same 202 patients, avoiding this problem -- a template that could be applied to the doc-level tasks above (e.g. one example per entity/event/relation) if their small train sets turn out to limit model quality.

`ner_re_medication` and `medication_event` now have Tier A subtask-enrichment datasets available (see [Tier A subtask-enrichment datasets](#tier-a-subtask-enrichment-datasets-train-only-additive) above) as an opt-in additive lever for this. `ner_re_medication` additionally has a [paragraph-window dataset](#paragraph-window-datasets-hf-only-additive-not-scored-by-the-doc-level-pipeline) that adds genuinely new (not relabeled) training text spans. `concept_assertion_relation` now has the same pair of levers -- Tier A subtasks plus a section-window dataset (boundaries are ALL-CAPS section headers rather than blank lines, since this format has none).

<!-- END: multitask dataset stats -->
