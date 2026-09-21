"""
2018 n2c2 Track 2 (Drug-ADE) -- joint NER + RE builder.

Entities: Drug, Strength, Frequency, Dosage, Route, Duration, Form, Reason,
ADE. Relations: binary, always <OtherType>-Drug with Arg2 = the Drug entity.
No events/roles needed for this dataset.
"""
from pathlib import Path

from multitask.brat import parse_ann
from multitask.io_utils import read_text
from multitask.schemas import example

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "2018_n2c2_track2"
SPLIT_DIRS = {"train": BASE / "2018n2c2_track2_training", "test": BASE / "gold_standard_test"}


def build_ner_re_medication(split):
    split_dir = SPLIT_DIRS.get(split)
    if split_dir is None or not split_dir.exists():
        return
    for txt_path in sorted(split_dir.glob("*.txt")):
        ann_path = txt_path.with_suffix(".ann")
        if not ann_path.exists():
            continue
        doc_text = read_text(txt_path)
        doc = parse_ann(ann_path)

        # Sort by true document offset, NOT by T-id / file order -- gold
        # .ann files number entities by annotation order, which is not
        # necessarily left-to-right (e.g. a later T-id can sit earlier in
        # the text). Emitting entities in reading order both matches how a
        # generative model naturally scans a document and is required for
        # the convert_back reanchor step's greedy left-to-right matching to
        # correctly disambiguate repeated identical mentions.
        entities = [
            {"id": eid, "type": e.type, "text": e.text}
            for eid, e in sorted(doc.entities.items(), key=lambda kv: (kv[1].start, kv[1].end))
        ]
        relations = [
            {"type": r.type, "arg1": r.arg1, "arg2": r.arg2}
            for r in doc.relations.values()
            if r.arg1 in doc.entities and r.arg2 in doc.entities
        ]
        target = {"entities": entities, "relations": relations}
        yield example("ner_re_medication", "2018_n2c2_track2", split, txt_path.stem, doc_text, target)
