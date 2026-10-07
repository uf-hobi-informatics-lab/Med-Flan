"""
2022 n2c2 Track 2 (SHAC / SDOH) -- social-determinants event extraction
builder.

Events are named-role frames: a trigger entity (e.g. Tobacco, Alcohol,
Employment, LivingStatus) plus role-labeled argument entities (Status,
Amount, Degree, Duration, Type, ...), bundled by an E-line. Attribute
values (StatusTimeVal, StatusEmployVal, ...) qualify specific role fillers
via A-lines pointing at the filler's T-line id.

The release has two independent sources, MIMIC and UW, each with its own
train/dev/test split under 2022n2c2_track2/<split>/<source>/ -- kept as two
separate datasets (2022n2c2_track2_mimic, 2022n2c2_track2_uw), never
merged, since the two sources come from different institutions/annotators.
The UW test split additionally ships a relation-form re-annotation of the
same events (2022n2c2_track2/test/uw_r/, no matching .txt files) -- that's
handled separately in multitask.convert_back.sdoh_2022_t2 as a derived,
test-only auxiliary evaluation axis, not part of this builder's target.
"""
from pathlib import Path

from inference_and_eval.brat import parse_ann
from inference_and_eval.io_utils import read_text
from inference_and_eval.schemas import example

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "raw_datasets" / "2022n2c2_track2"
SOURCES = ("mimic", "uw")


def _split_dir(source, split):
    return BASE / split / source


def build_sdoh_event(source, split):
    split_dir = _split_dir(source, split)
    if not split_dir.exists():
        return
    for txt_path in sorted(split_dir.glob("*.txt")):
        ann_path = txt_path.with_suffix(".ann")
        if not ann_path.exists():
            continue
        doc_text = read_text(txt_path)
        doc = parse_ann(ann_path)

        attrs_by_target = {}
        for a in doc.attributes:
            attrs_by_target.setdefault(a.target, []).append({"name": a.name, "value": a.value})

        events = []
        for eid, ev in doc.events.items():
            trigger_entity = doc.entities.get(ev.trigger)
            if trigger_entity is None:
                continue
            roles_out = {}
            role_attrs = []
            for role_name, target_id in ev.roles.items():
                filler = doc.entities.get(target_id)
                if filler is None:
                    continue
                roles_out[role_name] = {"type": filler.type, "text": filler.text}
                for a in attrs_by_target.get(target_id, []):
                    role_attrs.append({"role": role_name, "name": a["name"], "value": a["value"]})
            events.append({
                "id": eid,
                "type": ev.type,
                "trigger_text": trigger_entity.text,
                "roles": roles_out,
                "attributes": role_attrs,
                "_start": trigger_entity.start,
            })

        # Sort by trigger's true document offset, not E-id/file order --
        # same reasoning as the other two BRAT-based builders.
        events.sort(key=lambda e: e.pop("_start"))

        target = {"events": events}
        yield example("sdoh_event", f"2022n2c2_track2_{source}", split, txt_path.stem, doc_text, target)
