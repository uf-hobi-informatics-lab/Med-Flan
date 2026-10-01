"""
Combined per-dataset eval runner: given a dataset name and real model
completions (raw strings, keyed by the `id` field of that dataset's
multitask/data/eval/<dataset>.jsonl), converts back to native format where
needed and invokes that dataset's real scorer -- the single entry point
described (but not yet built) in the project README.

This module only wires together what already exists in convert_back/ and
eval/*.py; it adds no new scoring logic of its own beyond completion
parsing and dispatch. See README.md's "Running an eval" section and the
per-task status table for what each dataset's scorer actually verifies and
which numbers are an inherent ceiling rather than a converter bug.

Usage:
    from multitask.eval.run_eval import run_eval
    run_eval("2018_n2c2_track2", {"100130": '{"drugs": [...], "unlinked": [...]}', ...})

Or from the command line, with completions as a JSON file of {id: completion}:
    python -m multitask.eval.run_eval 2018_n2c2_track2 completions.json

completions may be a raw string (as real model output would be) for every
task; JSON-shaped tasks are parsed leniently (first {...} object found, via
multitask.eval.simple_tasks.parse_completion) rather than requiring an
exact json.loads -- a real model's output is never guaranteed to be
perfectly-formed JSON on the first try, and the conservative behavior for
anything unparseable is to score it as a miss, not to crash the run.
"""
import json
import sys
import tempfile
from pathlib import Path

from multitask.eval.simple_tasks import DATASETS as SIMPLE_TASK_DATASETS
from multitask.eval.simple_tasks import parse_completion

ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = ROOT / "multitask" / "data" / "eval"

# dataset name -> filename stem under multitask/data/eval/ (differs for the
# two names that contain characters not used in filenames on disk)
_FILENAME_OVERRIDES = {"covid (LitCovid)": "covid_LitCovid"}

# task tag -> dataset name(s) that use it, for the simple-scorer group
_SIMPLE_TASK_OF_DATASET = {name: task for task, names in SIMPLE_TASK_DATASETS.items() for name in names}


def _eval_path(dataset):
    return EVAL_DIR / f"{_FILENAME_OVERRIDES.get(dataset, dataset)}.jsonl"


def _task_of(dataset):
    path = _eval_path(dataset)
    with open(path, encoding="utf-8") as f:
        return json.loads(f.readline())["task"]


def _parsed_target(completion):
    """completion: raw model output string, or already-parsed dict/list
    (self-checks pass gold targets straight through)."""
    if isinstance(completion, (dict, list)):
        return completion
    return parse_completion(completion)


def _run_brat(dataset, completions, module):
    from multitask.io_utils import read_text

    gold_dir = module.GOLD_DIR_BY_DATASET[dataset] if hasattr(module, "GOLD_DIR_BY_DATASET") else module.GOLD_DIR
    predictions = [
        (doc_id, _parsed_target(completion), read_text(gold_dir / f"{doc_id}.txt"))
        for doc_id, completion in completions.items()
    ]
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = module.write_anns(predictions, tmp)
        if hasattr(module, "score_dirs"):
            module.score_dirs(gold_dir, out_dir)
        else:
            module._run_eval(gold_dir, out_dir)


def _run_brat_seq2seq(dataset, completions, module):
    """Parallel to `_run_brat`, for the seq2seq grammar path
    (multitask/seq2seq_grammar.py): completions are raw generated strings,
    never JSON, so this skips `_parsed_target`'s JSON-parse entirely."""
    from multitask.io_utils import read_text

    gold_dir = module.GOLD_DIR_BY_DATASET[dataset]
    predictions = [
        (doc_id, completion, read_text(gold_dir / f"{doc_id}.txt"))
        for doc_id, completion in completions.items()
    ]
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = module.write_anns(predictions, tmp)
        module.score_dirs(gold_dir, out_dir)


def _run_cohort_eligibility(completions):
    from multitask.convert_back import cohort_eligibility_2018_t1 as m

    predictions = [(locator, _parsed_target(c)) for locator, c in completions.items()]
    with tempfile.TemporaryDirectory() as tmp:
        m.write_xmls(predictions, tmp)
        m._run_eval(m.GOLD_DIR, tmp)


def _run_family_history(completions):
    from multitask.convert_back import family_history_2019_t2 as m

    predictions = [(doc_id, _parsed_target(c)) for doc_id, c in completions.items()]
    with tempfile.TemporaryDirectory() as tmp:
        s2_path = m.write_tsv(predictions, tmp)
        m._run_eval(2, m.GOLD_S2, s2_path)


def _run_family_history_seq2seq(completions):
    """Parallel to `_run_family_history`, for the seq2seq grammar path
    (multitask/family_history_seq2seq_grammar.py): completions are raw
    generated strings, never JSON, so this skips `_parsed_target`'s
    JSON-parse entirely -- same reasoning as `_run_brat_seq2seq`."""
    from multitask.convert_back import family_history_2019_t2 as fh
    from multitask.convert_back import family_history_2019_t2_seq2seq as m

    predictions = [(doc_id, m.to_target(c)) for doc_id, c in completions.items()]
    with tempfile.TemporaryDirectory() as tmp:
        s2_path = fh.write_tsv(predictions, tmp)
        fh._run_eval(2, fh.GOLD_S2, s2_path)


def _run_sts(completions):
    from multitask.convert_back import sts_2019_t1 as m

    n = sum(1 for _ in open(m.GOLD_SCORES, encoding="utf-8"))
    # completion ids are "<test filename>#<line index>" -- reorder into the
    # test file's original line order (correlation-noconfidence.pl compares
    # line-by-line against the unmodified gold gs file); a missing index
    # (model didn't answer that pair) falls back to the scale midpoint, the
    # same conservative default parse_similarity() uses for unparseable text.
    by_index = {int(doc_id.rsplit("#", 1)[1]): c for doc_id, c in completions.items()}
    targets = [by_index.get(i) for i in range(n)]

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(m.to_score_lines(targets))
        system_path = f.name
    m._run_eval(m.GOLD_SCORES, system_path)
    Path(system_path).unlink(missing_ok=True)


def _run_concept_assertion_relation(completions):
    from multitask.convert_back import i2b2_2010 as m

    predictions = {doc_id: _parsed_target(c) for doc_id, c in completions.items()}
    print(json.dumps(m.score_dataset(predictions, split="test"), indent=2))


def _run_concept_assertion_relation_seq2seq(completions):
    """Parallel to `_run_concept_assertion_relation`, for the seq2seq grammar
    path (multitask/concept_assertion_relation_seq2seq_grammar.py):
    completions are raw generated strings, never JSON, so this skips
    `_parsed_target`'s JSON-parse entirely and hands the raw text straight to
    `i2b2_2010_seq2seq.score_dataset`, which does its own parsing internally."""
    from multitask.convert_back import i2b2_2010_seq2seq as m

    print(json.dumps(m.score_dataset(completions, split="test"), indent=2))


def run_eval(dataset, completions):
    """dataset: a name from README's status table (e.g. "2018_n2c2_track2",
    "MedNLI", "covid (LitCovid)"). completions: {id: raw completion string}.
    Prints that dataset's real scorer output; returns nothing (the scorers
    themselves print/return metrics in whatever shape their own official
    tool -- or multitask.eval.simple_tasks/discharge_me -- uses)."""
    task = _task_of(dataset)

    if task == "ner_re_medication":
        from multitask.convert_back import ner_re_2018_t2 as m
        return _run_brat(dataset, completions, m)
    if task == "medication_event":
        from multitask.convert_back import medication_event_2022_t1 as m
        return _run_brat(dataset, completions, m)
    if task == "sdoh_event":
        from multitask.convert_back import sdoh_2022_t2 as m
        return _run_brat(dataset, completions, m)
    if task == "sdoh_event_seq2seq":
        from multitask.convert_back import sdoh_2022_t2_seq2seq as m
        return _run_brat_seq2seq(dataset, completions, m)
    if task == "ner_re_medication_seq2seq":
        from multitask.convert_back import ner_re_2018_t2_seq2seq as m
        return _run_brat_seq2seq(dataset, completions, m)
    if task == "medication_event_seq2seq":
        from multitask.convert_back import medication_event_2022_t1_seq2seq as m
        return _run_brat_seq2seq(dataset, completions, m)
    if task == "cohort_eligibility":
        return _run_cohort_eligibility(completions)
    if task == "family_history":
        return _run_family_history(completions)
    if task == "family_history_seq2seq":
        return _run_family_history_seq2seq(completions)
    if task == "sts_regression":
        return _run_sts(completions)
    if task == "concept_assertion_relation":
        return _run_concept_assertion_relation(completions)
    if task == "concept_assertion_relation_seq2seq":
        return _run_concept_assertion_relation_seq2seq(completions)
    if task == "disorder_normalization":
        from multitask.convert_back import disorder_normalization_2015_t14 as m

        predictions = {doc_id: _parsed_target(c) for doc_id, c in completions.items()}
        print(json.dumps(m.score_dataset(predictions, split="test"), indent=2))
        return
    if task == "summarization":
        from multitask.eval.discharge_me import score_dataset
        print(json.dumps(score_dataset(completions, include_bertscore=False), indent=2))
        return
    if task == "qa_span_extraction":
        from multitask.eval.qa_span import score_dataset
        print(json.dumps(score_dataset(completions), indent=2))
        return
    if task in _SIMPLE_TASK_OF_DATASET.values() or dataset in _SIMPLE_TASK_OF_DATASET:
        from multitask.eval.simple_tasks import score_dataset
        print(json.dumps(score_dataset(_FILENAME_OVERRIDES.get(dataset, dataset), completions), indent=2))
        return
    raise ValueError(f"no scorer wired up for task {task!r} (dataset {dataset!r})")


def _doc_level_gold(dataset):
    """{doc_id: raw target dict} for whichever dataset name is passed --
    same raw-target helpers _self_check_all() already uses, just keyed by
    dataset name instead of hardcoded into a loop."""
    if dataset == "2018_n2c2_track2":
        from multitask.convert_back import ner_re_2018_t2 as m
        return {ex["id"]: ex["target"] for ex in m._examples_with_raw_target("test")}
    if dataset == "2022n2c2_track1":
        from multitask.convert_back import medication_event_2022_t1 as m
        return {ex["id"]: ex["target"] for ex in m._examples_with_raw_target("test")}
    if dataset in ("2022n2c2_track2_mimic", "2022n2c2_track2_uw"):
        from multitask.convert_back import sdoh_2022_t2 as m
        source = "mimic" if dataset.endswith("mimic") else "uw"
        return {ex["id"]: ex["target"] for ex in m._examples_with_raw_target(source, "test")}
    if dataset == "2010_i2b2_va":
        from multitask.convert_back import i2b2_2010 as m
        return {ex["id"]: ex["target"] for ex in m._examples_with_raw_target("test")}
    if dataset == "semeval_2015":
        from multitask.convert_back import disorder_normalization_2015_t14 as m
        return {ex["id"]: ex["target"] for ex in m._examples_with_raw_target("test")}
    raise ValueError(f"no doc-level gold loader for dataset {dataset!r} -- see multitask.eval.doc_level")


def run_doc_level_eval(dataset, completions):
    """Addon to run_eval(): scores the same `completions` on the doc-level,
    offset-free axis (multitask.eval.doc_level) instead of the real
    offset-based scorer. Only covers the 5 tasks doc_level.py defines --
    family_history already has its own offset-free official eval.py and
    isn't included here."""
    from multitask.eval import doc_level

    task = _task_of(dataset)
    gold = _doc_level_gold(dataset)
    predictions = {doc_id: _parsed_target(c) for doc_id, c in completions.items()}
    return doc_level.score(task, gold, predictions)


def _self_check_all():
    """Every dataset, gold fed back as its own "prediction" -- exercises
    the full dispatch table end to end. Reuses each convert_back module's
    own _examples_with_raw_target() (the BRAT/XML/TSV/STS datasets) or reads
    gold straight out of multitask/data/eval/<name>.jsonl (everything else)
    -- the same two patterns each module's own standalone self-check already
    uses, so this adds no new gold-loading logic of its own. Slow (runs
    every dataset's real scorer); this is what CI/a pre-flight check would
    call, not routine use."""
    from multitask.convert_back import (
        cohort_eligibility_2018_t1, disorder_normalization_2015_t14,
        family_history_2019_t2, medication_event_2022_t1, ner_re_2018_t2,
        sdoh_2022_t2, sts_2019_t1,
    )

    raw_target_jobs = [
        ("2018_n2c2_track2", ner_re_2018_t2, "test"),
        ("2022n2c2_track1", medication_event_2022_t1, "test"),
        ("2019_n2c2_track2", family_history_2019_t2, "test"),
        ("2019_n2c2_track1", sts_2019_t1, "test"),
        ("semeval_2015", disorder_normalization_2015_t14, "test"),
    ]
    for dataset, module, split in raw_target_jobs:
        print(f"\n=== {dataset} ===")
        completions = {ex["id"]: ex["target"] for ex in module._examples_with_raw_target(split)}
        run_eval(dataset, completions)

    for source, dataset in (("mimic", "2022n2c2_track2_mimic"), ("uw", "2022n2c2_track2_uw")):
        print(f"\n=== {dataset} ===")
        completions = {ex["id"]: ex["target"] for ex in sdoh_2022_t2._examples_with_raw_target(source, "test")}
        run_eval(dataset, completions)

    print("\n=== 2022n2c2_track2_uw relations (derived, auxiliary) ===")
    sdoh_2022_t2._self_check_relations()

    from multitask.convert_back import sdoh_2022_t2_seq2seq

    for source, dataset in (("mimic", "2022n2c2_track2_mimic_seq2seq"), ("uw", "2022n2c2_track2_uw_seq2seq")):
        print(f"\n=== {dataset} ===")
        completions = {ex["id"]: ex["target"] for ex in sdoh_2022_t2_seq2seq._examples_with_raw_target(source, "test")}
        run_eval(dataset, completions)

    from multitask.convert_back import ner_re_2018_t2_seq2seq

    print("\n=== 2018_n2c2_track2_seq2seq ===")
    completions = {ex["id"]: ex["target"] for ex in ner_re_2018_t2_seq2seq._examples_with_raw_target("test")}
    run_eval("2018_n2c2_track2_seq2seq", completions)

    from multitask.convert_back import medication_event_2022_t1_seq2seq

    print("\n=== 2022n2c2_track1_seq2seq ===")
    completions = {ex["id"]: ex["target"] for ex in medication_event_2022_t1_seq2seq._examples_with_raw_target("test")}
    run_eval("2022n2c2_track1_seq2seq", completions)

    from multitask.convert_back import i2b2_2010_seq2seq

    print("\n=== 2010_i2b2_va_seq2seq ===")
    completions = {ex["id"]: ex["target"] for ex in i2b2_2010_seq2seq._examples_with_raw_target("test")}
    run_eval("2010_i2b2_va_seq2seq", completions)

    print("\n=== 2018_n2c2_track1 ===")
    run_eval("2018_n2c2_track1", dict(cohort_eligibility_2018_t1._examples_with_raw_target("test")))

    for dataset in ("MedNLI", "bionli", "nli4ct", "MedQA-USMLE-US", "medmcqa", "pubmedqa", "covid (LitCovid)", "discharge-me", "umn_abbreviation", "2019_n2c2_track1_cls", "emrqa-msquad", "2010_i2b2_va"):
        print(f"\n=== {dataset} ===")
        completions = {}
        with open(_eval_path(dataset), encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                completions[d["id"]] = d["messages"][-1]["content"]
        run_eval(dataset, completions)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--doc-level"]
    doc_level_flag = "--doc-level" in sys.argv[1:]

    if len(args) == 2:
        with open(args[1], encoding="utf-8") as f:
            completions = json.load(f)
        run_eval(args[0], completions)
        if doc_level_flag:
            print(f"\n=== {args[0]} (doc-level, offset-free addon) ===")
            run_doc_level_eval(args[0], completions)
    else:
        _self_check_all()
