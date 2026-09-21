"""
JSON -> BRAT .ann converter for 2022 n2c2 Track 2 (SHAC / SDOH). The
released dataset ships with no evaluation script at all, but the task's
own organizer (Kevin Lybarger) published one afterwards as the standalone
`brat_scoring` package (https://github.com/Lybarger/brat_scoring, MIT),
used by https://github.com/romanows/SDOH-n2c2 as *the* scorer for this task
-- the closest thing that exists to "official" for SHAC. It's vendored
verbatim under multitask/vendor/brat_scoring/ (pip install of it hit an
unrelated setuptools/_distutils_hack bug in this venv building ANY git+
package; vendoring the ~90KB pure-python package avoids that entirely) and
invoked here unmodified via its own CLI, exactly like the subprocess calls
to the official scripts in the other three convert_back modules.

Input target schema (see multitask.builders.sdoh_2022_t2):
    {"events": [
        {"id": "E2", "type": "Tobacco", "trigger_text": "tobacco",
         "roles": {"Status": {"type": "StatusTime", "text": "No history"}},
         "attributes": [{"role": "Status", "name": "StatusTimeVal", "value": "none"}]},
     ]}

Two BRAT quirks specific to this dataset's annotation convention (not
present in the other two):
  - A role filler can legitimately share the exact same span as its own
    event's trigger (e.g. T1 LivingStatus 17 22 "Lives" / T2 StatusTime
    17 22 "Lives" -- the same token annotated twice, once as the event
    trigger and once as its own status-time role filler). When a role
    filler's text is identical to its trigger's text, that span is reused
    directly instead of re-searching (a second, unrelated occurrence of the
    same word elsewhere in the document would be the wrong match).
  - Role fillers can sit on either side of their trigger in the sentence
    (unlike flat entity lists, which run in one direction through the
    document), so they're anchored with reanchor(..., near=trigger_start)
    -- nearest occurrence in either direction -- rather than a forward-only
    cursor.

Run standalone to self-check against gold on the dev split (no local test
split exists for SHAC) -- should give exact P=R=F1=1.0 on every row:
    python -m multitask.convert_back.sdoh_2022_t2
"""
import subprocess
import sys
import tempfile
from pathlib import Path

from multitask.brat import parse_ann
from multitask.reanchor import reanchor

ROOT = Path(__file__).resolve().parents[2]
GOLD_DIR = ROOT / "2022n2c2_track2" / "Track2_SubtaskA" / "Annotations" / "dev" / "mimic"
SCORER_SCRIPT = ROOT / "multitask" / "vendor" / "brat_scoring" / "brat_scoring" / "run_sdoh_scoring.py"


def to_ann(target, doc_text):
    """target: the JSON dict (events only). Returns .ann file text."""
    used_spans = []
    cursor = 0
    lines = []
    tid = eid = aid = 0

    for ev in target.get("events", []):
        trig_span = reanchor(ev["trigger_text"], doc_text, used_spans, min_start=cursor)
        if trig_span is None:
            continue
        t_start, t_end = trig_span
        cursor = max(cursor, t_end)
        tid += 1
        trigger_tid = f"T{tid}"
        lines.append(f"{trigger_tid}\t{ev['type']} {t_start} {t_end}\t{ev['trigger_text']}")

        role_tid_by_name = {}
        role_parts = []
        for role_name, filler in ev.get("roles", {}).items():
            filler_text = filler["text"]
            if filler_text == ev["trigger_text"]:
                f_span = (t_start, t_end)  # shared-mention convention -- see module docstring
            else:
                # A fresh, per-filler used_spans (not the shared trigger
                # cursor's list): the same underlying entity is often
                # legitimately referenced as a role filler by more than one
                # event (e.g. one "No history" StatusTime shared by both a
                # Tobacco and a Drug event) -- exclusivity would wrongly
                # treat the second reference as unavailable and force it
                # onto a different, wrong occurrence.
                f_span = reanchor(filler_text, doc_text, [], near=t_start)
            if f_span is None:
                continue
            f_start, f_end = f_span
            tid += 1
            filler_tid = f"T{tid}"
            lines.append(f"{filler_tid}\t{filler['type']} {f_start} {f_end}\t{filler_text}")
            role_tid_by_name[role_name] = filler_tid
            role_parts.append(f"{role_name}:{filler_tid}")

        eid += 1
        event_id = f"E{eid}"
        lines.append(f"{event_id}\t" + " ".join([f"{ev['type']}:{trigger_tid}"] + role_parts))

        for attr in ev.get("attributes", []):
            filler_tid = role_tid_by_name.get(attr["role"])
            if filler_tid is None:
                continue
            aid += 1
            lines.append(f"A{aid}\t{attr['name']} {filler_tid} {attr['value']}")

    return "\n".join(lines) + ("\n" if lines else "")


def write_anns(predictions, out_dir):
    """predictions: iterable of (doc_id, target, doc_text). Writes one
    <doc_id>.ann per prediction under out_dir, plus its paired <doc_id>.txt --
    unlike the other two convert_back modules, the vendored brat_scoring
    package's get_brat_files() asserts a matching .txt exists for every .ann
    (it re-tokenizes the text itself for scoring), so the source doc text
    must be written alongside, not just referenced from the gold dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for doc_id, target, doc_text in predictions:
        (out_dir / f"{doc_id}.ann").write_text(to_ann(target, doc_text), encoding="utf-8")
        (out_dir / f"{doc_id}.txt").write_text(doc_text, encoding="utf-8")
    return out_dir


def _event_key_counters(doc):
    """doc: a parsed BRAT Doc (multitask.brat.parse_ann). Returns three
    dict-of-set summaries used only by _self_check()'s pre-flight sanity
    check, not for real scoring (see score_dirs, which shells out to the
    vendored scorer)."""
    triggers, roles = set(), set()
    for ev in doc.events.values():
        trig = doc.entities.get(ev.trigger)
        if trig is None:
            continue
        triggers.add((ev.type, trig.start, trig.end))
        for role_name, target_id in ev.roles.items():
            filler = doc.entities.get(target_id)
            if filler is None:
                continue
            roles.add((ev.type, role_name, filler.start, filler.end))
    return triggers, roles


def _run_eval(gold_dir, system_dir):
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        out_csv = f.name
    result = subprocess.run(
        [sys.executable, str(SCORER_SCRIPT), str(gold_dir), str(system_dir), out_csv],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        return

    import pandas as pd
    df = pd.read_csv(out_csv)
    Path(out_csv).unlink(missing_ok=True)
    print(df.to_string(index=False))
    overall = df[["NT", "NP", "TP"]].sum()
    p = overall["TP"] / overall["NP"] if overall["NP"] else 0.0
    r = overall["TP"] / overall["NT"] if overall["NT"] else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    print(f"\noverall (micro, all rows pooled): P={p:.4f} R={r:.4f} F1={f1:.4f}")


def score_dirs(gold_dir, system_dir):
    """Score system_dir's .ann predictions against gold_dir using the
    vendored brat_scoring package (see module docstring) -- the closest
    thing to an official scorer for SHAC."""
    _run_eval(gold_dir, system_dir)


def _self_check():
    """Round-trip gold dev-split JSON (built by the dataset builder) back to
    .ann, then score against the real gold -- should give exact P=R=F1=1.0
    on every row. Uses dev because SHAC has no local test split."""
    from multitask.io_utils import read_text

    predictions = []
    for ex in _examples_with_raw_target("dev"):
        doc_text = read_text(GOLD_DIR / f"{ex['id']}.txt")
        predictions.append((ex["id"], ex["target"], doc_text))

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = write_anns(predictions, tmp)

        # Fast pre-check (trigger/role span+type sets only, no spacy) so an
        # obvious reanchor regression fails immediately instead of waiting
        # on the slower real scorer below.
        mismatches = 0
        for doc_id, _, _ in predictions:
            gold_doc = parse_ann(GOLD_DIR / f"{doc_id}.ann")
            gen_doc = parse_ann(out_dir / f"{doc_id}.ann")
            if _event_key_counters(gold_doc) != _event_key_counters(gen_doc):
                mismatches += 1
        print(f"pre-check: {mismatches}/{len(predictions)} docs with a trigger/role span+type mismatch")

        print("=== self-check: gold JSON -> .ann -> brat_scoring (vendored) ===")
        score_dirs(GOLD_DIR, out_dir)


def _examples_with_raw_target(split):
    """Rebuild examples with the raw target dict attached (schemas.example()
    only keeps rendered text/messages)."""
    from multitask.builders.sdoh_2022_t2 import SPLIT_DIRS

    split_dir = SPLIT_DIRS[split]
    for txt_path in sorted(split_dir.glob("*.txt")):
        ann_path = txt_path.with_suffix(".ann")
        if not ann_path.exists():
            continue
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
        events.sort(key=lambda e: e.pop("_start"))

        yield {"id": txt_path.stem, "target": {"events": events}}


if __name__ == "__main__":
    if len(sys.argv) == 3:
        score_dirs(sys.argv[1], sys.argv[2])
    else:
        _self_check()
