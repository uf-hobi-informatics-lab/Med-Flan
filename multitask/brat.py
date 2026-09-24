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


def align_entity_text(doc, doc_text):
    """Rewrite each entity's mention text to the document slice its offsets
    point at, but only when that slice contains no line break. Mutates and
    returns `doc`.

    A BRAT .ann text column is not always a substring of the document. Two
    distinct causes, which need opposite treatment:

    1. The annotator's span crosses a line break, and the text column
       normalizes the newline to a space ("Q6H (every 6 hours)" for a
       document reading "Q6H (every\\n6 hours)"). Measured as committed:
       3,077 entities in 2018_n2c2_track2, 1,233 in SHAC, 5 in CMED.
       reanchor._flex_pattern already matches these via its `\\s+` join, and
       the normalized form is the better generation target -- a target
       string should not carry a literal newline. LEFT ALONE.

    2. The span is split into adjacent fragments with no gap
       ("7829 7838;7838 7839" -> text column "Prednison e", document
       "Prednisone"), or otherwise gains a space the document lacks
       ("0.02  %" vs "0.02 %", "Release (E.C.)" vs "Release(E.C.)").
       Measured as committed: 1,591 in 2018_n2c2_track2, 1 in SHAC, 1 in
       CMED. `\\s+` requires at least one whitespace character and so can
       NEVER match these -- the entity fails to re-anchor, and in
       ner_re_2018_t2's nested target a failed Drug silently discards every
       attribute under it. Measured cost before this fix: 316/10,575 Drug
       mentions dropped, taking 1,543 fillers with them, i.e. tags F1
       capped at 0.9429 and relations at 0.9039 on a *gold* round-trip
       (0.9752/0.9609 after). FIXED HERE.

    Restricting the rewrite to newline-free slices is what separates the
    two: it repairs case 2 at the source (so the target is a verbatim
    substring the model can actually copy) without dragging newlines into
    case 1's targets. Same reasoning as builders/i2b2_2010.py, which
    already re-slices from doc_text rather than trusting the .con text.

    Empty slices are skipped: a zero-length or inverted span would
    otherwise blank the entity's text entirely, turning a recoverable
    mention into an empty target string. parse_ann now uses min/max over
    fragments so inverted spans shouldn't reach here, but the
    unparseable-span fallback still yields (0, 0), so the guard stays.

    Currently wired into builders/ner_re_2018_t2.py only. Case 2 is
    essentially absent from SHAC (1) and CMED (1), so calling it there
    would be very nearly a no-op -- but not strictly one, which is why
    those corpora are left alone rather than assumed safe.
    """
    for entity in doc.entities.values():
        slice_text = doc_text[entity.start:entity.end]
        if slice_text and slice_text != entity.text and "\n" not in slice_text:
            entity.text = slice_text
    return doc


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
            # Discontinuous spans ("start1 end1;start2 end2;...") collapse to
            # the full (min_start, max_end) range. Use min/max over all
            # fragments rather than first/last: fragments are USUALLY in
            # document order, but not always -- 2022n2c2_track2/test/mimic/
            # 4885.ann has "T6 StatusTime 112 120;81 111", where taking
            # first/last yields the inverted span (112, 111) and therefore an
            # empty slice. Exactly 1 such T-line exists across all 190k in
            # the repo, but an inverted span silently poisons anything that
            # slices doc_text with it (see align_entity_text).
            #
            # This is close to, but not identical to, how the official
            # scorers parse the same line: Track2-evaluate-ver4.py handles
            # only 3/4/5 space-split fields and falls through to a
            # print-and-reuse-stale-variables branch otherwise, so it
            # mishandles the 10 three-fragment T-lines in 2018_n2c2_track2
            # that this code gets right.
            try:
                nums = [int(x) for x in span_str.replace(";", " ").split()]
                start, end = min(nums), max(nums)
            except (ValueError, IndexError):
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
