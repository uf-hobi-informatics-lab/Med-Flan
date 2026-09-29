"""
2022 n2c2 Track 2 (SHAC / SDOH) -- Microsoft-Nuance-style seq2seq builder.

Brand-new, additive sibling to builders/sdoh_2022_t2.py's `build_sdoh_event`
(task tag `sdoh_event`, JSON target). Nothing here is imported by that
module or by build_hf_datasets.py's existing REGISTRY rows for it; this
file only adds new REGISTRY rows for a new task tag, `sdoh_event_seq2seq`
(see multitask/seq2seq_grammar.py for the target string format).

Same BRAT-parsing logic as build_sdoh_event -- the event dicts built here
are the identical shape (id/type/trigger_text/roles/attributes), just
carrying an extra "span"/"trigger_span" per mention (needed by the
true-negative redaction and curriculum span-chunking below, not by the
core builder) and serialized via `seq2seq_grammar.serialize_events()`
instead of being kept as a JSON target.

Three builders, three new dataset names (train-only augmentation variants
suffixed onto the core name, never merged into it):

- `build_sdoh_event_seq2seq`: one example per doc, core dataset
  (`2022n2c2_track2_{source}_seq2seq`), train+dev+test.
- `build_sdoh_event_seq2seq_true_negative`: train-only
  (`..._seq2seq_trueneg`). For each doc with >=1 event, one extra example
  whose input has every gold entity span (trigger or filler) redacted to
  "X" characters (offsets/length preserved, so the input still reads like
  a real note in shape) and whose target is the empty string -- the
  confirmed true-negative convention for this feature. Explicitly a
  redaction of this corpus's own annotated docs, not independently-curated
  raw notes as in the Ben Abacha et al. paper -- see design.md/the
  approved plan's "Explicit gaps" section; a real, stated approximation,
  not claimed equivalent.
- `build_sdoh_event_seq2seq_curriculum`: train-only
  (`..._seq2seq_curriculum`). Per Ben Abacha et al.'s contiguous-
  annotation-subset augmentation, implemented as span-chunking on
  paragraph windows (multitask/chunk_utils.paragraph_spans) rather than a
  dedicated subset sampler. A window is only used if every event with any
  mention inside it has *every* mention inside it too (a window is
  discarded outright, not partially kept, if some event has one mention
  inside and another outside -- keeping the visible-but-unlabeled mention
  would teach a false negative).
"""
from pathlib import Path

from inference_and_eval.brat import parse_ann
from inference_and_eval.chunk_utils import paragraph_spans
from inference_and_eval.io_utils import read_text
from inference_and_eval.schemas import example
from inference_and_eval.seq2seq_grammar import serialize_events

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "2022n2c2_track2"


def _split_dir(source, split):
    return BASE / split / source


def _events_with_spans(doc):
    """Same event-dict shape as builders/sdoh_2022_t2.py's build_sdoh_event,
    plus a "span"/"trigger_span" (start, end) on every mention -- needed by
    the true-negative redaction and curriculum containment checks below,
    harmless extra keys for serialize_events (which only reads
    type/trigger_text/roles/attributes)."""
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
            roles_out[role_name] = {
                "type": filler.type, "text": filler.text,
                "span": (filler.start, filler.end),
            }
            for a in attrs_by_target.get(target_id, []):
                role_attrs.append({"role": role_name, "name": a["name"], "value": a["value"]})
        events.append({
            "id": eid,
            "type": ev.type,
            "trigger_text": trigger_entity.text,
            "trigger_span": (trigger_entity.start, trigger_entity.end),
            "roles": roles_out,
            "attributes": role_attrs,
            "_start": trigger_entity.start,
        })

    events.sort(key=lambda e: e.pop("_start"))
    return events


def build_sdoh_event_seq2seq(source, split):
    split_dir = _split_dir(source, split)
    if not split_dir.exists():
        return
    for txt_path in sorted(split_dir.glob("*.txt")):
        ann_path = txt_path.with_suffix(".ann")
        if not ann_path.exists():
            continue
        doc_text = read_text(txt_path)
        doc = parse_ann(ann_path)
        events = _events_with_spans(doc)
        target_str = serialize_events(events)
        yield example(
            "sdoh_event_seq2seq", f"2022n2c2_track2_{source}_seq2seq", split,
            txt_path.stem, doc_text, target_str,
        )


def _redact(doc_text, spans):
    chars = list(doc_text)
    for start, end in spans:
        for i in range(max(start, 0), min(end, len(chars))):
            chars[i] = "X"
    return "".join(chars)


def build_sdoh_event_seq2seq_true_negative(source, split):
    if split != "train":
        return
    split_dir = _split_dir(source, split)
    if not split_dir.exists():
        return
    for txt_path in sorted(split_dir.glob("*.txt")):
        ann_path = txt_path.with_suffix(".ann")
        if not ann_path.exists():
            continue
        doc = parse_ann(ann_path)
        if not doc.events:
            continue
        doc_text = read_text(txt_path)
        spans = sorted({(e.start, e.end) for e in doc.entities.values()})
        redacted = _redact(doc_text, spans)
        yield example(
            "sdoh_event_seq2seq", f"2022n2c2_track2_{source}_seq2seq_trueneg", split,
            f"{txt_path.stem}_trueneg", redacted, "",
        )


def build_sdoh_event_seq2seq_curriculum(source, split):
    if split != "train":
        return
    split_dir = _split_dir(source, split)
    if not split_dir.exists():
        return
    for txt_path in sorted(split_dir.glob("*.txt")):
        ann_path = txt_path.with_suffix(".ann")
        if not ann_path.exists():
            continue
        doc_text = read_text(txt_path)
        doc = parse_ann(ann_path)
        events = _events_with_spans(doc)
        if len(events) < 2:
            continue  # nothing to subset -- a single-event doc has no proper contiguous subset to teach

        for wi, (wstart, wend) in enumerate(paragraph_spans(doc_text)):
            window_text = doc_text[wstart:wend]
            if not window_text.strip():
                continue

            kept, discard_window = [], False
            for ev in events:
                spans = [ev["trigger_span"]] + [f["span"] for f in ev["roles"].values()]
                fully_in = all(wstart <= s and e <= wend for s, e in spans)
                any_in = any(wstart <= s < wend for s, e in spans)
                if fully_in:
                    kept.append(ev)
                elif any_in:
                    # Part of this event's mentions sit inside the window
                    # but not all of them -- excluding the event here would
                    # silently teach a false negative on the visible part,
                    # so the whole window is unusable (see module docstring).
                    discard_window = True
                    break
            if discard_window or not kept:
                continue

            target_str = serialize_events(kept)
            yield example(
                "sdoh_event_seq2seq", f"2022n2c2_track2_{source}_seq2seq_curriculum", split,
                f"{txt_path.stem}_p{wi}", window_text, target_str,
            )
