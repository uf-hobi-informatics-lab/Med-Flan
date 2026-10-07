"""
2018 n2c2 Track 2 (Drug-ADE) -- joint NER + RE builder.

Entities: Drug, Strength, Frequency, Dosage, Route, Duration, Form, Reason,
ADE. Every relation is binary, always <OtherType>-Drug with Arg2 = the Drug
entity (confirmed both by annotation.conf and by a real gold .ann file,
e.g. gold_standard_test/105885.ann) -- i.e. Drug is always the anchor,
exactly the same star topology as an event trigger. Rather than a flat
entity list + a separate id-based relations list (which forces a
generative model to invent and correctly echo back an id per entity), each
Other-type mention is nested directly under its linked Drug, keyed by
attribute type -- same "attributes attached to entity" convention already
used by medication_event_2022_t1.py/sdoh_2022_t2.py. No relation ids
anywhere; linking is via nesting, not ids.

Known lossiness of that nesting: because an attribute is keyed by its own
entity type, the relation's own label is discarded, so the two disagree
whenever a relation's label doesn't match its Arg1 type. Measured: 15 of
59,810 gold relations (0.025%) -- 14 `ADE-Drug` relations whose Arg1 is
typed `Reason` (e.g. 110521 R120 on Reason "low blood counts") and 1
`Strength-Drug` whose Arg1 is typed `Dosage` (103142 R74). All 15 are in
train, 0 in test, which is why the test-split self-check doesn't surface
them. A flat id-based target would round-trip these exactly; this one
relabels them to their Arg1 type irrecoverably. Accepted as the cost of
dropping entity ids -- the "every relation is <OtherType>-Drug" statement
above holds 99.975% of the time, not 100%.

One real complication (also confirmed from 105885.ann): a single Other
entity can be linked via separate relations to MULTIPLE Drug entities
(e.g. one Reason mention shared by 4 different drugs) -- its text is
simply repeated under each linked drug's attribute list. Other entities
never linked to any Drug are collected under "unlinked" (still needed so
the official scorer's entity-level recall counts them).
"""
from pathlib import Path

from multitask.brat import align_entity_text, parse_ann
from multitask.chunk_utils import paragraph_spans, project_doc_onto_span
from multitask.io_utils import read_text
from multitask.schemas import example

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "raw_datasets" / "2018_n2c2_track2"
SPLIT_DIRS = {"train": BASE / "2018n2c2_track2_training", "test": BASE / "gold_standard_test"}


def _iter_docs(split):
    """Yields (txt_path, doc_text, doc) for every doc in split -- shared by
    the full end-to-end builder and the Tier A NER-only/RE-only subtask
    builders below (see design.md's Tier A section)."""
    split_dir = SPLIT_DIRS.get(split)
    if split_dir is None or not split_dir.exists():
        return
    for txt_path in sorted(split_dir.glob("*.txt")):
        ann_path = txt_path.with_suffix(".ann")
        if not ann_path.exists():
            continue
        doc_text = read_text(txt_path)
        # Repair .ann mention text that isn't a substring of the document
        # (adjacent-fragment spans like "Prednison e" for "Prednisone") --
        # see brat.align_entity_text. Without this a Drug whose text can't
        # be re-anchored silently discards every attribute nested under it.
        yield txt_path, doc_text, align_entity_text(parse_ann(ann_path), doc_text)


def _build_ner_re_target(doc):
    """{"drugs": [...], "unlinked": [...]} for one parsed doc -- the full
    end-to-end target, also reused as-is by the RE-only subtask (same
    second-stage linking skill, just given pre-extracted entities as input
    instead of having to find them first)."""
    drugs = [e for e in doc.entities.values() if e.type == "Drug"]
    drugs.sort(key=lambda e: (e.start, e.end))

    # relations are always <Other>-Drug, Arg1=Other, Arg2=Drug (see
    # module docstring) -- group Other entities by which Drug(s) they're
    # linked to. A given Other entity can appear under more than one
    # Drug (the multi-drug Reason case), so this is not a 1:1 map.
    others_by_drug = {}
    linked_other_ids = set()
    for r in doc.relations.values():
        other = doc.entities.get(r.arg1)
        drug = doc.entities.get(r.arg2)
        if other is None or drug is None or drug.type != "Drug":
            continue
        others_by_drug.setdefault(r.arg2, []).append(other)
        linked_other_ids.add(other.id)

    drugs_out = []
    for d in drugs:
        drug_dict = {"text": d.text}
        attrs = {}
        for other in sorted(others_by_drug.get(d.id, []), key=lambda e: (e.start, e.end)):
            attrs.setdefault(other.type, []).append(other.text)
        drug_dict.update(attrs)
        drugs_out.append(drug_dict)

    unlinked = [
        {"type": e.type, "text": e.text}
        for e in sorted(doc.entities.values(), key=lambda e: (e.start, e.end))
        if e.type != "Drug" and e.id not in linked_other_ids
    ]

    return {"drugs": drugs_out, "unlinked": unlinked}


def build_ner_re_medication(split):
    for txt_path, doc_text, doc in _iter_docs(split):
        target = _build_ner_re_target(doc)
        yield example("ner_re_medication", "2018_n2c2_track2", split, txt_path.stem, doc_text, target)


def build_ner_re_medication_ner_only(split):
    """Tier A subtask: entities only, no relation linking -- see design.md."""
    for txt_path, doc_text, doc in _iter_docs(split):
        entities = sorted(doc.entities.values(), key=lambda e: (e.start, e.end))
        target = {"entities": [{"type": e.type, "text": e.text} for e in entities]}
        yield example(
            "ner_re_medication_ner_only", "2018_n2c2_track2_ner_only", split, txt_path.stem, doc_text, target
        )


def build_ner_re_medication_re_only(split):
    """Tier A subtask: entities are pre-extracted and given in the input
    (EXTRACTED_ENTITIES block); the target is the same drugs/unlinked
    linking as the full end-to-end task -- see design.md."""
    for txt_path, doc_text, doc in _iter_docs(split):
        entities = sorted(doc.entities.values(), key=lambda e: (e.start, e.end))
        entity_list = "\n".join(f"- {e.type}: {e.text}" for e in entities)
        input_text = f"{doc_text}\n\nEXTRACTED_ENTITIES:\n{entity_list}"
        target = _build_ner_re_target(doc)
        yield example(
            "ner_re_medication_re_only", "2018_n2c2_track2_re_only", split, txt_path.stem, input_text, target
        )


def _iter_paragraphs(doc_text, doc):
    """Yields (start, end, para_text, para_doc) for every non-empty
    paragraph in doc_text -- see multitask/chunk_utils.py. Paragraphs with
    no entities (headers, blank fields) are skipped -- no training signal.
    (start, end) are exposed alongside para_text/para_doc so
    convert_back/ner_re_2018_t2.py's paragraph-prediction merge (the
    inference-time reassembly step) can recover each surviving paragraph's
    offset in the whole document without reimplementing this same filter
    -- see that module's _paragraph_items_by_doc."""
    for start, end in paragraph_spans(doc_text):
        para_doc = project_doc_onto_span(doc, start, end)
        if not para_doc.entities:
            continue
        yield start, end, doc_text[start:end], para_doc


def build_ner_re_medication_paragraph(split):
    """Paragraph-window subtask: same full entities+relations target shape
    as the parent task, scoped to one paragraph instead of the whole note
    -- see design.md's paragraph-chunking section. HF-only, additive
    (train+test both built for experimentation flexibility, but not wired
    into the doc-level authoritative eval pipeline); drops the ~0.24% of
    relations that straddle two paragraphs."""
    for txt_path, doc_text, doc in _iter_docs(split):
        for i, (_start, _end, para_text, para_doc) in enumerate(_iter_paragraphs(doc_text, doc)):
            target = _build_ner_re_target(para_doc)
            yield example(
                "ner_re_medication_paragraph", "2018_n2c2_track2_paragraph", split,
                f"{txt_path.stem}__p{i}", para_text, target,
            )
