"""
Generic BRAT standoff (.ann) parser, shared by every NER+RE-family builder
(2018_n2c2_track2, 2022n2c2_track1, 2022n2c2_track2). Confirmed against real
files from all three datasets — see multitask/schemas.py callers for the
per-dataset mapping onto the joint entities/events/relations schema.

Only T (text-bound entity), E (event), R (relation), and A (attribute) lines
are used by these datasets; N/M/#-lines are ignored.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class Entity:
    id: str
    type: str
    start: int
    end: int
    text: str


@dataclass
class Event:
    id: str
    type: str
    trigger: str
    roles: Dict[str, str] = field(default_factory=dict)  # role name -> entity/event id


@dataclass
class Relation:
    id: str
    type: str
    arg1: Optional[str]
    arg2: Optional[str]


@dataclass
class Attribute:
    id: str
    name: str
    target: Optional[str]
    value: Optional[str]


@dataclass
class BratDoc:
    entities: Dict[str, Entity]
    events: Dict[str, Event]
    relations: Dict[str, Relation]
    attributes: List[Attribute]


def parse_ann(path) -> BratDoc:
    entities: Dict[str, Entity] = {}
    events: Dict[str, Event] = {}
    relations: Dict[str, Relation] = {}
    attributes: List[Attribute] = []

    text = Path(path).read_text(encoding="utf-8", errors="replace")
    for raw_line in text.splitlines():
        if not raw_line.strip():
            continue
        if "\t" not in raw_line:
            continue
        ann_id, rest = raw_line.split("\t", 1)

        if ann_id.startswith("T"):
            if "\t" in rest:
                head, mention_text = rest.split("\t", 1)
            else:
                head, mention_text = rest, ""
            head_parts = head.split(" ", 1)
            etype = head_parts[0]
            span_str = head_parts[1] if len(head_parts) > 1 else "0 0"
            first_span = span_str.split(";")[0]
            try:
                start_s, end_s = first_span.split()
                start, end = int(start_s), int(end_s)
            except ValueError:
                start, end = 0, 0
            entities[ann_id] = Entity(ann_id, etype, start, end, mention_text)

        elif ann_id.startswith("E"):
            parts = rest.split()
            if not parts or ":" not in parts[0]:
                continue
            etype, trigger = parts[0].split(":", 1)
            roles: Dict[str, str] = {}
            for p in parts[1:]:
                if ":" not in p:
                    continue
                role, target = p.split(":", 1)
                roles[role] = target
            events[ann_id] = Event(ann_id, etype, trigger, roles)

        elif ann_id.startswith("R"):
            parts = rest.split()
            if not parts:
                continue
            rtype = parts[0]
            args = dict(p.split(":", 1) for p in parts[1:] if ":" in p)
            relations[ann_id] = Relation(ann_id, rtype, args.get("Arg1"), args.get("Arg2"))

        elif ann_id.startswith("A"):
            parts = rest.split()
            if not parts:
                continue
            name = parts[0]
            target = parts[1] if len(parts) > 1 else None
            value = parts[2] if len(parts) > 2 else None
            attributes.append(Attribute(ann_id, name, target, value))
        # N-lines (normalization), M-lines, #-lines (notes): not used here.

    return BratDoc(entities, events, relations, attributes)
