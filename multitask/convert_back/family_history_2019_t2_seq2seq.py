"""
Seq2seq-grammar -> native TSV converter for 2019 n2c2 Track 2 (family
history), mirroring convert_back/family_history_2019_t2.py's JSON -> TSV
converter but starting from the tag-delimited seq2seq target string
(multitask/family_history_seq2seq_grammar.py) instead of a JSON dict.

No character offsets exist in this task's gold data (see
builders/family_history_2019_t2.py's docstring), so unlike every other
*_seq2seq convert_back module in this family, there is no to_ann/BRAT step
here at all -- `to_target`'s output is handed straight to the *existing*
`write_tsv`/`_run_eval`/`GOLD_S2` from convert_back/family_history_2019_t2.py,
reused verbatim.

Run standalone to self-check the converter against gold (round-trip through
the seq2seq grammar and back should reproduce eval.py P=R=F1=1.0 on the test
split):
    python -m multitask.convert_back.family_history_2019_t2_seq2seq
"""
import tempfile

from multitask.convert_back.family_history_2019_t2 import GOLD_S2, _run_eval, write_tsv
from multitask.family_history_seq2seq_grammar import parse_and_resolve


def to_target(generated_text):
    return parse_and_resolve(generated_text)


def _self_check():
    """Round-trip gold test-split target strings (built by
    build_family_history_seq2seq) back to TSV, then score against the real
    gold -- no model involved."""
    predictions = [(ex["id"], to_target(ex["target"])) for ex in _examples_with_raw_target("test")]
    with tempfile.TemporaryDirectory() as tmp:
        s2_path = write_tsv(predictions, tmp)
        print("=== Subtask 2 (self-check: gold seq2seq string -> TSV -> eval.py) ===")
        _run_eval(2, GOLD_S2, s2_path)


def _examples_with_raw_target(split):
    """Rebuild examples with the raw (serialized) target string attached --
    schemas.example() only keeps rendered text/messages, same reasoning as
    convert_back/family_history_2019_t2.py's own _examples_with_raw_target."""
    from multitask.builders.family_history_2019_t2_seq2seq import build_family_history_seq2seq

    for ex in build_family_history_seq2seq(split):
        yield {"id": ex["id"], "target": ex["messages"][-1]["content"]}


if __name__ == "__main__":
    _self_check()
