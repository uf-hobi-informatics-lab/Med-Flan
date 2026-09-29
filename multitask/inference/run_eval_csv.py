"""
Single-file eval sweep: reads the per-dataset completions JSON files written
by multitask.inference.run_generation (<dir>/<model_tag>__<dataset>.json,
{id: completion}) and scores each dataset with its REAL metrics -- accuracy,
P/R/F1, kappa, correlation, etc. -- into one long-format CSV (no openpyxl/
Excel dependency).

Why this exists instead of just capturing multitask.eval.run_eval's printed
output: most of this repo's scorers already compute a metrics dict
internally (simple_tasks.score_dataset, i2b2_2010.score_dataset,
brat_scorer.score_dirs, sdoh_2022_t2.score_relations, ...) but run_eval.py
only prints it (json.dumps(...)) and returns nothing -- see that module's
docstring on its print-only contract. This script calls each task's own
scorer function directly instead of going through run_eval(), so it gets the
dict back with no text-parsing involved for most tasks. Two datasets
(2018_n2c2_track1/cohort_eligibility, 2019_n2c2_track2/family_history) route
through an EXTERNAL Perl/Python tool that only ever prints a summary table --
for those two, this script captures that printed text (same as run_eval.py
already does) and regexes out the overall summary line, verified against the
exact print format in 2018_n2c2_track1/track1_eval_script/track1_eval.py and
2019_n2c2_track2/eval.py.

Usage:
    python -m multitask.inference.run_eval_csv \\
        --completions-dir /path/to/outputs/completions/ \\
        --out /path/to/outputs/eval_results.csv

Output CSV columns: dataset, metric, value -- one row per (dataset, metric)
pair, long-format so every task family's differently-shaped metrics fit the
same three columns with no blank cells (pivot with
`pandas.read_csv(...).pivot(index="dataset", columns="metric", values="value")`
downstream if a wide table is wanted, e.g. in Excel once openpyxl is
available again). "metric" for a nested scorer result (e.g. brat_scorer's
per-axis P/R/F1) is dot-joined, e.g. "tags.f1", "events.combined.precision".
"""
import argparse
import contextlib
import csv
import io
import json
import re
import tempfile
from pathlib import Path

from multitask import brat_scorer
from multitask.eval.run_eval import _FILENAME_OVERRIDES, _SIMPLE_TASK_OF_DATASET, _parsed_target, _task_of
from multitask.inference.run_generation import DATASET_SPECS
from multitask.io_utils import read_text

# "    Overall (micro)  0.9123  0.8765  ...  0.8949  0.9210" -- see
# 2018_n2c2_track1/track1_eval_script/track1_eval.py's Overall print (9
# %.4f-formatted numbers after the label).
_COHORT_OVERALL_LINE = re.compile(
    r"Overall \((micro|macro)\)\s+"
    r"([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+"
    r"([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)"
)
_COHORT_COLUMNS = (
    "precision_met", "recall_met", "specificity_met", "f1_met",
    "precision_notmet", "recall_notmet", "f1_notmet", "f1_avg", "auc",
)

# "Prevision\tRecall\tF1\n0.9999\t0.9999 \t0.9999" -- see
# 2019_n2c2_track2/eval.py's get_pr_f1 print (verbatim misspelling).
_FAMHX_LINE = re.compile(r"Prevision\tRecall\tF1\s*\n\s*([\d.]+)\s*\t\s*([\d.]+)\s*\t\s*([\d.]+)")


def _flatten(d, prefix=""):
    """Nested dict -> {"a.b.c": value} -- every scorer here returns a dict,
    but the nesting depth differs by task family (flat for simple_tasks,
    one level for brat_scorer's per-axis P/R/F1); flattening once
    generically means the output CSV doesn't need per-task-family
    special-casing."""
    out = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return out


def _score_brat(dataset, completions, module, config_key):
    gold_dir = module.GOLD_DIR_BY_DATASET[dataset] if hasattr(module, "GOLD_DIR_BY_DATASET") else module.GOLD_DIR
    predictions = [
        (doc_id, _parsed_target(c), read_text(gold_dir / f"{doc_id}.txt"))
        for doc_id, c in completions.items()
    ]
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = module.write_anns(predictions, tmp)
        return brat_scorer.score_dirs(gold_dir, out_dir, brat_scorer.CONFIGS[config_key])


def _score_paragraph(completions):
    from multitask.convert_back import ner_re_2018_t2 as m

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = m.write_anns_from_paragraphs(completions, "test", tmp)
        return brat_scorer.score_dirs(m.GOLD_DIR, out_dir, brat_scorer.CONFIGS["2018_n2c2_track2"])


def _score_cohort_eligibility(completions):
    from multitask.convert_back import cohort_eligibility_2018_t1 as m

    predictions = [(locator, _parsed_target(c)) for locator, c in completions.items()]
    buf = io.StringIO()
    with tempfile.TemporaryDirectory() as tmp:
        m.write_xmls(predictions, tmp)
        with contextlib.redirect_stdout(buf):
            m._run_eval(m.GOLD_DIR, tmp)
    text = buf.getvalue()
    result = {}
    for match in _COHORT_OVERALL_LINE.finditer(text):
        kind = match.group(1)
        for col, val in zip(_COHORT_COLUMNS, match.groups()[1:]):
            result[f"{kind}.{col}"] = float(val)
    if not result:
        result["parse_error"] = "could not find 'Overall (micro/macro)' line in scorer output"
    result["raw_output"] = text
    return result


def _score_family_history(completions):
    from multitask.convert_back import family_history_2019_t2 as m

    predictions = [(doc_id, _parsed_target(c)) for doc_id, c in completions.items()]
    buf = io.StringIO()
    with tempfile.TemporaryDirectory() as tmp:
        s2_path = m.write_tsv(predictions, tmp)
        with contextlib.redirect_stdout(buf):
            m._run_eval(2, m.GOLD_S2, s2_path)
    text = buf.getvalue()
    result = {}
    match = _FAMHX_LINE.search(text)
    if match:
        result["precision"], result["recall"], result["f1"] = (float(x) for x in match.groups())
    else:
        result["parse_error"] = "could not find the Prevision/Recall/F1 line in scorer output"
    result["raw_output"] = text
    return result


def score_dataset_structured(dataset, eval_kind, completions):
    """{metric_name: value} for one dataset's completions -- see module
    docstring for why this calls each task's own scorer directly instead of
    going through multitask.eval.run_eval."""
    if eval_kind == "paragraph":
        return _score_paragraph(completions)

    task = _task_of(dataset)
    name = _FILENAME_OVERRIDES.get(dataset, dataset)

    if task in _SIMPLE_TASK_OF_DATASET.values() or dataset in _SIMPLE_TASK_OF_DATASET:
        from multitask.eval.simple_tasks import score_dataset
        return score_dataset(name, completions)
    if task == "concept_assertion_relation":
        from multitask.convert_back import i2b2_2010 as m
        predictions = {doc_id: _parsed_target(c) for doc_id, c in completions.items()}
        return m.score_dataset(predictions, split="test")
    if task == "disorder_normalization":
        from multitask.convert_back import disorder_normalization_2015_t14 as m
        predictions = {doc_id: _parsed_target(c) for doc_id, c in completions.items()}
        return m.score_dataset(predictions, split="test")
    if task == "summarization":
        from multitask.eval.discharge_me import score_dataset
        return score_dataset(completions, include_bertscore=False)
    if task == "qa_span_extraction":
        from multitask.eval.qa_span import score_dataset
        return score_dataset(completions)
    if task == "ner_re_medication":
        from multitask.convert_back import ner_re_2018_t2 as m
        return _score_brat(dataset, completions, m, "2018_n2c2_track2")
    if task == "medication_event":
        from multitask.convert_back import medication_event_2022_t1 as m
        return _score_brat(dataset, completions, m, "2022n2c2_track1")
    if task == "sdoh_event":
        from multitask.convert_back import sdoh_2022_t2 as m
        return _score_brat(dataset, completions, m, "2022n2c2_track2")
    if task == "cohort_eligibility":
        return _score_cohort_eligibility(completions)
    if task == "family_history":
        return _score_family_history(completions)
    raise ValueError(f"no structured scorer wired up for task {task!r} (dataset {dataset!r})")


def run_eval_sweep(completions_dir, dataset_specs=None, out_csv=None, model_tag=None):
    dataset_specs = dataset_specs if dataset_specs is not None else DATASET_SPECS
    completions_dir = Path(completions_dir)

    rows = []  # (dataset, metric, value)
    for name, eval_kind, _eval_name in dataset_specs:
        pattern = f"{model_tag}__{name}.json" if model_tag else f"*__{name}.json"
        matches = sorted(completions_dir.glob(pattern))
        if not matches:
            print(f"[skip] {name}: no file matching {completions_dir / pattern}")
            continue
        if len(matches) > 1:
            print(
                f"[warn] {name}: {len(matches)} completions files match {pattern} "
                f"({[m.name for m in matches]}) -- using the most recently modified one; "
                "pass --model-tag to disambiguate"
            )
        path = max(matches, key=lambda p: p.stat().st_mtime)

        print("=" * 88)
        print(name)
        print("=" * 88)
        completions = json.loads(path.read_text(encoding="utf-8"))
        try:
            flat = _flatten(score_dataset_structured(name, eval_kind, completions))
        except Exception as e:
            print(f"[eval FAILED] {name}: {e!r}")
            flat = {"error": repr(e)}
        for metric, value in flat.items():
            if metric != "raw_output":
                print(f"  {metric}: {value}")
            rows.append((name, metric, value))

    if out_csv:
        out_csv = Path(out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["dataset", "metric", "value"])
            writer.writerows(rows)
        n_datasets = len({r[0] for r in rows})
        print(f"\nwrote {len(rows)} metric rows across {n_datasets} dataset(s) to {out_csv}")

    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--completions-dir", required=True, help="directory from multitask.inference.run_generation")
    parser.add_argument(
        "--model-tag", default=None,
        help="disambiguate when --completions-dir holds runs from more than one model/checkpoint "
             "(matches the '<model_tag>__<dataset>.json' filenames written by run_generation.py's "
             "model_tag()); default: pick the most recently modified match per dataset",
    )
    parser.add_argument(
        "--datasets", nargs="+", default=None,
        help="restrict to these dataset names (default: every entry in DATASET_SPECS present in --completions-dir)",
    )
    parser.add_argument("--out", required=True, help="output .csv path")
    args = parser.parse_args()

    dataset_specs = None
    if args.datasets:
        dataset_specs = [spec for spec in DATASET_SPECS if spec[0] in args.datasets]
        missing = set(args.datasets) - {s[0] for s in dataset_specs}
        if missing:
            raise SystemExit(f"unknown dataset name(s): {sorted(missing)} -- see DATASET_SPECS in run_generation.py")

    run_eval_sweep(args.completions_dir, dataset_specs=dataset_specs, out_csv=args.out, model_tag=args.model_tag)


if __name__ == "__main__":
    main()
