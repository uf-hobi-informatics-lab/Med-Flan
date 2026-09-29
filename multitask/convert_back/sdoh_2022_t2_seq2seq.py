"""
Seq2seq-target -> BRAT .ann converter for `sdoh_event_seq2seq` (Microsoft-
Nuance-style target, see multitask/seq2seq_grammar.py). Additive sibling to
convert_back/sdoh_2022_t2.py's JSON-target `to_ann`/`write_anns`/
`score_dirs`/`_self_check` -- that module, its `sdoh_event` task, and every
existing dataset it scores are untouched by this file.

Input is the raw generated string (the model's completion, or the gold
target string for a self-check round-trip), not a parsed dict -- unlike
sdoh_2022_t2.py's to_ann, this path never calls `reanchor()` at all: every
span is already resolved by `seq2seq_grammar.parse_and_resolve_with_spans`,
which runs its own global constraint solver across the whole generated
sequence at once (see that module's docstring) instead of one independent
nearest-occurrence search per filler.

Scored with the exact same `brat_scorer.CONFIGS["2022n2c2_track2"]` config
sdoh_2022_t2.py's score_dirs uses, so Combined F1 here is directly
comparable to that path's own numbers on the same gold.

Run standalone to self-check both sources' test splits against gold
(gold target string -> .ann -> brat_scorer, no model involved):
    python -m multitask.convert_back.sdoh_2022_t2_seq2seq
"""
import sys
import tempfile
from pathlib import Path

from multitask import brat_scorer
from multitask.seq2seq_grammar import parse_and_resolve_with_spans

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "2022n2c2_track2"


def _gold_dir(source, split):
    return BASE / split / source


# run_eval.py's generic dispatcher looks this up by dataset name, same
# convention as sdoh_2022_t2.py's GOLD_DIR_BY_DATASET.
GOLD_DIR_BY_DATASET = {
    "2022n2c2_track2_mimic_seq2seq": _gold_dir("mimic", "test"),
    "2022n2c2_track2_uw_seq2seq": _gold_dir("uw", "test"),
}


def to_ann(generated_text, doc_text):
    """generated_text: the raw seq2seq completion string. Returns .ann file
    text -- every trigger/filler span already resolved by
    parse_and_resolve_with_spans's constraint solver; poison spans (past
    doc_text, guaranteed span-incompatible with any real gold) come from
    that same call for anything unresolvable, so an unmatched prediction
    still counts as an attempted-and-wrong guess in scoring rather than
    silently vanishing -- same contract as sdoh_2022_t2.py's to_ann."""
    resolved = parse_and_resolve_with_spans(generated_text, doc_text)
    lines = []
    tid = eid = aid = 0

    for ev in resolved["events"]:
        t_start, t_end = ev["trigger_span"]
        tid += 1
        trigger_tid = f"T{tid}"
        lines.append(f"{trigger_tid}\t{ev['type']} {t_start} {t_end}\t{ev['trigger_text']}")

        role_tid_by_name = {}
        role_parts = []
        for role_name, filler in ev["roles"].items():
            f_start, f_end = filler["span"]
            tid += 1
            filler_tid = f"T{tid}"
            lines.append(f"{filler_tid}\t{filler['type']} {f_start} {f_end}\t{filler['text']}")
            role_tid_by_name[role_name] = filler_tid
            role_parts.append(f"{role_name}:{filler_tid}")

        eid += 1
        event_id = f"E{eid}"
        lines.append(f"{event_id}\t" + " ".join([f"{ev['type']}:{trigger_tid}"] + role_parts))

        for attr in ev["attributes"]:
            filler_tid = role_tid_by_name.get(attr["role"])
            if filler_tid is None:
                continue
            aid += 1
            lines.append(f"A{aid}\t{attr['name']} {filler_tid} {attr['value']}")

    return "\n".join(lines) + ("\n" if lines else "")


def write_anns(predictions, out_dir):
    """predictions: iterable of (doc_id, generated_text, doc_text). Same
    shape/contract as sdoh_2022_t2.py's write_anns (paired .ann + .txt --
    the vendored brat_scoring package's get_brat_files() asserts a matching
    .txt exists for every .ann)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for doc_id, generated_text, doc_text in predictions:
        (out_dir / f"{doc_id}.ann").write_text(to_ann(generated_text, doc_text), encoding="utf-8")
        (out_dir / f"{doc_id}.txt").write_text(doc_text, encoding="utf-8")
    return out_dir


def score_dirs(gold_dir, system_dir):
    """Reuses brat_scorer.CONFIGS["2022n2c2_track2"] verbatim -- see module
    docstring for why that makes this directly comparable to
    sdoh_2022_t2.py's own score_dirs numbers."""
    brat_scorer.score_dirs(gold_dir, system_dir, brat_scorer.CONFIGS["2022n2c2_track2"])


def _self_check(source):
    """Round-trip gold test-split target strings (built by
    build_sdoh_event_seq2seq) back to .ann, then score against the real
    gold -- no model involved. See module docstring for why this is
    directly comparable to sdoh_2022_t2.py's own self-check numbers."""
    from multitask.io_utils import read_text

    gd = _gold_dir(source, "test")
    predictions = []
    for ex in _examples_with_raw_target(source, "test"):
        doc_text = read_text(gd / f"{ex['id']}.txt")
        predictions.append((ex["id"], ex["target"], doc_text))

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = write_anns(predictions, tmp)
        print(f"=== self-check ({source}/test, seq2seq): gold target string -> .ann -> brat_scorer ===")
        score_dirs(gd, out_dir)


def _examples_with_raw_target(source, split):
    """Rebuild examples with the raw (serialized) target string attached --
    schemas.example() only keeps rendered text/messages, same reasoning as
    sdoh_2022_t2.py's own _examples_with_raw_target."""
    from multitask.builders.sdoh_2022_t2_seq2seq import build_sdoh_event_seq2seq

    for ex in build_sdoh_event_seq2seq(source, split):
        yield {"id": ex["id"], "target": ex["messages"][-1]["content"]}


if __name__ == "__main__":
    if len(sys.argv) == 3:
        score_dirs(sys.argv[1], sys.argv[2])
    else:
        _self_check("mimic")
        _self_check("uw")
