"""
Seq2seq target builder for ner_re_medication_seq2seq -- brand-new, additive
alongside the JSON-based ner_re_medication builder in this same package,
never merged with it. Reuses that module's own doc iteration and target
construction (`_iter_docs`, `_build_ner_re_target`) so the two target
shapes can never drift out of sync -- only the serialization at the very
end differs (see multitask/ner_re_medication_seq2seq_grammar.py).
"""
from multitask.builders.ner_re_2018_t2 import _build_ner_re_target, _iter_docs
from multitask.ner_re_medication_seq2seq_grammar import serialize_events
from multitask.schemas import example


def build_ner_re_medication_seq2seq(split):
    for txt_path, doc_text, doc in _iter_docs(split):
        target = _build_ner_re_target(doc)
        target_str = serialize_events(target)
        yield example(
            "ner_re_medication_seq2seq", "2018_n2c2_track2_seq2seq", split,
            txt_path.stem, doc_text, target_str,
        )
