"""
2022 n2c2 Track 1 (CMED) -- medication disposition event extraction builder.

Each medication mention is a T+E pair with type in
{NoDisposition, Disposition, Undetermined}. Only Disposition events also
carry the 5 attribute dimensions (Action, Certainty, Temporality, Actor,
Negation), attached via A-lines pointing at the E-line id. No relations/
roles are used in the released training data (annotation.conf defines
`reason`/`indication` relations, but they don't occur in practice).
"""
from pathlib import Path

from inference_and_eval.brat import parse_ann
from inference_and_eval.chunk_utils import paragraph_spans, project_doc_onto_span
from inference_and_eval.io_utils import read_text
from inference_and_eval.schemas import example
from inference_and_eval.target_schema import CMED_ATTR_NAMES

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "raw_datasets" / "2022n2c2_track1/trainingdata_v3"
SPLIT_DIRS = {
    "train": BASE / "train",
    "dev": BASE / "dev",
    "test": ROOT / "raw_datasets" / "2022n2c2_track1" / "test",
}

ATTR_NAMES = CMED_ATTR_NAMES


def _iter_docs(split):
    """Yields (txt_path, doc_text, doc) for every doc in split -- shared by
    the full end-to-end builder and the Tier A trigger-only/attr-only
    subtask builders below (see design.md's Tier A section)."""
    split_dir = SPLIT_DIRS.get(split)
    if split_dir is None or not split_dir.exists():
        return
    for txt_path in sorted(split_dir.glob("*.txt")):
        ann_path = txt_path.with_suffix(".ann")
        if not ann_path.exists():
            continue
        yield txt_path, read_text(txt_path), parse_ann(ann_path)


def _build_events(doc):
    """[{"id", "type", "trigger_text", "attributes"}, ...] sorted by trigger
    offset -- the full end-to-end target, also reused as-is by the
    attr-only subtask (same attribute-classification skill, just given
    pre-identified triggers as input instead of having to find them
    first)."""
    # attributes are keyed by their target (an E-line id)
    attrs_by_target = {}
    for a in doc.attributes:
        if a.name in ATTR_NAMES:
            attrs_by_target.setdefault(a.target, {})[a.name] = a.value

    events = []
    for eid, ev in doc.events.items():
        trigger_entity = doc.entities.get(ev.trigger)
        if trigger_entity is None:
            continue
        event_obj = {
            "id": eid,
            "type": ev.type,
            "trigger_text": trigger_entity.text,
            "attributes": attrs_by_target.get(eid, {}),
            "_start": trigger_entity.start,
        }
        events.append(event_obj)

    # Sort by trigger's true document offset, not E-id/file order --
    # same reasoning as ner_re_2018_t2: emitting in reading order is
    # both what a generative model naturally does and what the
    # convert_back reanchor step needs to disambiguate repeated
    # medication mentions (e.g. "Tylenol" named several times).
    events.sort(key=lambda e: e.pop("_start"))
    return events


def build_medication_event(split):
    for txt_path, doc_text, doc in _iter_docs(split):
        target = {"events": _build_events(doc)}
        yield example("medication_event", "2022n2c2_track1", split, txt_path.stem, doc_text, target)


def build_medication_event_trigger_only(split):
    """Tier A subtask: trigger spans + disposition type only, no attribute
    classification -- see design.md."""
    for txt_path, doc_text, doc in _iter_docs(split):
        events = _build_events(doc)
        triggers = [{"type": e["type"], "text": e["trigger_text"]} for e in events]
        yield example(
            "medication_event_trigger_only", "2022n2c2_track1_trigger_only", split, txt_path.stem,
            doc_text, {"triggers": triggers},
        )


def build_medication_event_attr_only(split):
    """Tier A subtask: triggers are pre-identified and given in the input
    (EXTRACTED_TRIGGERS block); the target is the same attribute
    classification as the full end-to-end task -- see design.md."""
    for txt_path, doc_text, doc in _iter_docs(split):
        events = _build_events(doc)
        trigger_list = "\n".join(f"- {e['type']}: {e['trigger_text']}" for e in events)
        input_text = f"{doc_text}\n\nEXTRACTED_TRIGGERS:\n{trigger_list}"
        yield example(
            "medication_event_attr_only", "2022n2c2_track1_attr_only", split, txt_path.stem,
            input_text, {"events": events},
        )


def _iter_paragraphs(doc_text, doc):
    """Yields (para_text, para_events) for every paragraph whose events
    survive the span projection -- mirrors ner_re_2018_t2._iter_paragraphs
    but for track1's event-based target (events/attributes, not entities/
    relations). Paragraphs with no surviving events (headers, blank fields,
    or orphan spans) are skipped -- no training signal. project_doc_onto_span
    keeps events/attributes unchanged and filters by trigger entity, so a
    trigger straddling a boundary is dropped (measured: 2/9012 = 0.022%)."""
    for start, end in paragraph_spans(doc_text):
        para_doc = project_doc_onto_span(doc, start, end)
        events = _build_events(para_doc)
        if not events:
            continue
        yield doc_text[start:end], events


def build_medication_event_paragraph(split):
    """Paragraph-window subtask: same events+attributes target shape as the
    parent task, scoped to one paragraph instead of the whole note -- see
    design.md's paragraph-chunking section. HF-only, additive (train+dev+
    test all built, since track1 ships a real dev split, for experimentation
    flexibility); not wired into the doc-level authoritative eval pipeline
    (run_eval.py/convert_back untouched), mirroring
    ner_re_medication_paragraph. Drops the ~0.02% of events whose trigger
    span straddles two paragraphs."""
    for txt_path, doc_text, doc in _iter_docs(split):
        for i, (para_text, events) in enumerate(_iter_paragraphs(doc_text, doc)):
            yield example(
                "medication_event_paragraph", "2022n2c2_track1_paragraph", split,
                f"{txt_path.stem}__p{i}", para_text, {"events": events},
            )
