"""
Seq2seq target format for concept_assertion_relation (task tag
concept_assertion_relation_seq2seq) -- brand-new, additive alongside the
JSON-based concept_assertion_relation builder in
multitask/builders/i2b2_2010.py, never merged with it. Reuses that
module's own _iter_records/_build_target/_iter_sections so the target
shapes stay in sync automatically (see
multitask/concept_assertion_relation_seq2seq_grammar.py for the
serialization).
"""
from multitask.builders.i2b2_2010 import _build_target, _iter_records, _iter_sections
from multitask.concept_assertion_relation_seq2seq_grammar import serialize_target
from multitask.schemas import example


def build_concept_assertion_relation_seq2seq(split):
    for doc_id, doc_text, concepts, relations in _iter_records(split):
        target_str = serialize_target(_build_target(concepts, relations))
        yield example(
            "concept_assertion_relation_seq2seq", "2010_i2b2_va_seq2seq", split,
            doc_id, doc_text, target_str,
        )


def build_concept_assertion_relation_seq2seq_section(split):
    """Section-window mirror of build_concept_assertion_relation_section --
    same HF-only, additive convention (train+test built for experimentation
    flexibility, never wired into run_eval.py/convert_back -- doc-level
    scoring stays authoritative)."""
    for doc_id, doc_text, concepts, relations in _iter_records(split):
        for i, (sec_text, sec_concepts, sec_relations) in enumerate(
            _iter_sections(doc_text, concepts, relations)
        ):
            target_str = serialize_target(_build_target(sec_concepts, sec_relations))
            yield example(
                "concept_assertion_relation_seq2seq_section", "2010_i2b2_va_seq2seq_section", split,
                f"{doc_id}__s{i}", sec_text, target_str,
            )
