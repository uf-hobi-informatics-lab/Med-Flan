"""
JSON -> BRAT .ann converter for 2022 n2c2 Track 1 (CMED, medication
disposition events), so evaluation_script/eval_script.py can score model
output unmodified.

Input target schema (see multitask.builders.medication_event_2022_t1):
    {"events": [
        {"id": "E12", "type": "Disposition", "trigger_text": "Prozac",
         "attributes": {"Action": "Start", "Certainty": "Certain",
                         "Temporality": "Present", "Actor": "Physician",
                         "Negation": "NotNegated"}},
        {"id": "E19", "type": "NoDisposition", "trigger_text": "barbituate",
         "attributes": {}},
     ]}

Unlike ner_re_2018_t2, ids are NOT reused verbatim: eval_script.py's parser
asserts a strict 1:1 T<->E mapping (`assert tag_id not in t_e_mapper`), so
IDs must be freshly and consistently assigned as we emit each T/E pair --
reusing the original gold ids risks collisions once entities are dropped
(a trigger reanchor() can't find) and IDs skip around. Also note: the
`Combined` attribute (Action_Negation_Temporality_Certainty_Actor joined)
must NOT be emitted -- eval_script.py synthesizes it itself from the 5 named
A-lines it finds, and emitting our own would double it.

Run standalone to self-check against gold on the real test split. This is
not a perfect round-trip: repeated medication mentions within a doc can
make reanchor() pick a same-text-but-wrong occurrence, so the ceiling here
is inherently below 1.0 rather than an exact P=R=F1=1.0 like the BRAT-based
datasets with less mention repetition.

Scored via multitask.brat_scorer (see that module's docstring) rather than
the official eval_script.py -- that script's tag/attribute matching is a
single iteration-order-dependent pass ("pare down duplicates then count"),
not a real assignment algorithm; brat_scorer's greedy nearest-distance
matcher fixes that. Unlike eval_script.py's single per-trigger "Combined"
figure (1 TP only if the trigger AND all 5 attributes match exactly),
brat_scorer reports triggers and attributes as separate pooled axes plus an
"events.combined" that pools every trigger/role/attribute TP/FP/FN
together -- a coarser aggregate, not attribute-gated per-event accuracy, so
it isn't numerically the same quantity as eval_script.py's Combined even
when both are near 1.0. Current result on the test split (run this file
standalone to reproduce): triggers F1=0.9792, attributes F1=0.9146,
events.combined F1=0.9479. (Precision equals recall exactly on each axis --
an unresolvable prediction now gets a poison span instead of being dropped,
so on a gold round-trip every miss also costs a wrong guess; see to_ann's
docstring.)

Run standalone to self-check:
    python -m multitask.convert_back.medication_event_2022_t1
"""
import tempfile
from pathlib import Path

from multitask import brat_scorer
from multitask.reanchor import reanchor
from multitask.target_schema import CMED_ATTR_NAMES

ROOT = Path(__file__).resolve().parents[2]
GOLD_DIR = ROOT / "2022n2c2_track1" / "test"

ATTR_NAMES = CMED_ATTR_NAMES


def to_ann(target, doc_text):
    """target: the JSON dict (events only). Returns .ann file text.

    An event whose trigger_text can't be found anywhere in doc_text used
    to be dropped entirely (`continue`), which meant a hallucinated
    medication mention -- and every attribute value predicted for it --
    cost nothing in scoring: brat_scorer.score_events's fp accounting
    (`fp=len(valid_sys)-len(trig_matches)` for triggers,
    `fp=len(attr_values(...))` for an unmatched system event's attributes)
    only ever sees events that made it into the .ann at all. Giving the
    trigger a poison span (past doc_text, so it can never overlap a real
    gold trigger) instead of dropping the event keeps it in the .ann as a
    real T/E/A-line block -- score_events then counts it as a false
    positive on every axis automatically, with no changes needed there."""
    used_spans = []
    cursor = 0
    lines = []
    tid = eid = aid = 0
    poison = len(doc_text) + 1
    for ev in target.get("events", []):
        span = reanchor(ev["trigger_text"], doc_text, used_spans, min_start=cursor)
        if span is None:
            start, end = poison, poison + 1
            poison += 1
        else:
            start, end = span
            cursor = max(cursor, end)

        tid += 1
        eid += 1
        t_id, e_id = f"T{tid}", f"E{eid}"
        lines.append(f"{t_id}\t{ev['type']} {start} {end}\t{ev['trigger_text']}")
        lines.append(f"{e_id}\t{ev['type']}:{t_id}")

        for name, value in ev.get("attributes", {}).items():
            if name not in ATTR_NAMES:
                continue  # never emit Combined -- eval_script.py synthesizes it itself
            aid += 1
            lines.append(f"A{aid}\t{name} {e_id} {value}")

    return "\n".join(lines) + ("\n" if lines else "")


def write_anns(predictions, out_dir):
    """predictions: iterable of (doc_id, target, doc_text). Writes one
    <doc_id>.ann per prediction under out_dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for doc_id, target, doc_text in predictions:
        (out_dir / f"{doc_id}.ann").write_text(to_ann(target, doc_text), encoding="utf-8")
    return out_dir


def score_dirs(gold_dir, system_dir):
    """Score system_dir's .ann predictions against gold_dir using the new
    unified brat_scorer (see module docstring) rather than the official
    eval_script.py."""
    brat_scorer.score_dirs(gold_dir, system_dir, brat_scorer.CONFIGS["2022n2c2_track1"])


def _self_check():
    """Round-trip gold test-split JSON (built by the dataset builder) back to
    .ann, then score against the real gold -- should be P=R=F1=1.0."""
    from multitask.io_utils import read_text

    predictions = []
    for ex in _examples_with_raw_target("test"):
        doc_text = read_text(GOLD_DIR / f"{ex['id']}.txt")
        predictions.append((ex["id"], ex["target"], doc_text))

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = write_anns(predictions, tmp)
        print("=== self-check: gold JSON -> .ann -> brat_scorer ===")
        score_dirs(GOLD_DIR, out_dir)


def _examples_with_raw_target(split):
    """Rebuild examples with the raw target dict attached (schemas.example()
    only keeps rendered text/messages)."""
    from multitask.brat import parse_ann
    from multitask.builders.medication_event_2022_t1 import ATTR_NAMES as _ATTR, SPLIT_DIRS

    split_dir = SPLIT_DIRS[split]
    for txt_path in sorted(split_dir.glob("*.txt")):
        ann_path = txt_path.with_suffix(".ann")
        if not ann_path.exists():
            continue
        doc = parse_ann(ann_path)

        attrs_by_target = {}
        for a in doc.attributes:
            if a.name in _ATTR:
                attrs_by_target.setdefault(a.target, {})[a.name] = a.value

        events = []
        for eid, ev in doc.events.items():
            trigger_entity = doc.entities.get(ev.trigger)
            if trigger_entity is None:
                continue
            events.append({
                "id": eid,
                "type": ev.type,
                "trigger_text": trigger_entity.text,
                "attributes": attrs_by_target.get(eid, {}),
                "_start": trigger_entity.start,
            })
        events.sort(key=lambda e: e.pop("_start"))

        yield {"id": txt_path.stem, "target": {"events": events}}


if __name__ == "__main__":
    _self_check()
