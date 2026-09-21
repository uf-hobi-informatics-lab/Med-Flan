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

from multitask.brat import parse_ann
from multitask.io_utils import read_text
from multitask.schemas import example

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "2022n2c2_track1/trainingdata_v3"
SPLIT_DIRS = {"train": BASE / "train", "dev": BASE / "dev"}
# No official test split present locally for CMED -- flagged in the plan as a gap.

ATTR_NAMES = ("Action", "Certainty", "Temporality", "Actor", "Negation")


def build_medication_event(split):
    split_dir = SPLIT_DIRS.get(split)
    if split_dir is None or not split_dir.exists():
        return
    for txt_path in sorted(split_dir.glob("*.txt")):
        ann_path = txt_path.with_suffix(".ann")
        if not ann_path.exists():
            continue
        doc_text = read_text(txt_path)
        doc = parse_ann(ann_path)

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

        target = {"events": events}
        yield example("medication_event", "2022n2c2_track1", split, txt_path.stem, doc_text, target)
