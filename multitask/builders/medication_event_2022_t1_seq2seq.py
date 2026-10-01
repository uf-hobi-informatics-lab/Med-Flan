"""
Seq2seq target format for medication_event (task tag
medication_event_seq2seq) -- brand-new, additive alongside the JSON-based
medication_event builder above, never merged with it. Reuses that
builder's own _iter_docs/_build_events (and _iter_paragraphs, for the
paragraph-window mirror below) so the target shapes stay in sync
automatically (see multitask/medication_event_seq2seq_grammar.py for the
serialization).
"""
from multitask.builders.medication_event_2022_t1 import _build_events, _iter_docs, _iter_paragraphs
from multitask.medication_event_seq2seq_grammar import serialize_events
from multitask.schemas import example


def build_medication_event_seq2seq(split):
    for txt_path, doc_text, doc in _iter_docs(split):
        events = _build_events(doc)
        target_str = serialize_events({"events": events})
        yield example(
            "medication_event_seq2seq", "2022n2c2_track1_seq2seq", split,
            txt_path.stem, doc_text, target_str,
        )


def build_medication_event_seq2seq_paragraph(split):
    """Paragraph-window mirror of build_medication_event_paragraph -- same
    HF-only, additive convention (train+dev+test all built), just with the
    seq2seq target shape instead of JSON. Never wired into
    run_eval.py/convert_back -- doc-level scoring stays authoritative."""
    for txt_path, doc_text, doc in _iter_docs(split):
        for i, (para_text, events) in enumerate(_iter_paragraphs(doc_text, doc)):
            target_str = serialize_events({"events": events})
            yield example(
                "medication_event_seq2seq_paragraph", "2022n2c2_track1_seq2seq_paragraph", split,
                f"{txt_path.stem}__p{i}", para_text, target_str,
            )
