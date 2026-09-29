"""
JSON -> BRAT .ann converter for 2022 n2c2 Track 2 (SHAC / SDOH). The
released dataset ships with no evaluation script at all; the task's own
organizer (Kevin Lybarger) published one afterwards as the standalone
`brat_scoring` package (https://github.com/Lybarger/brat_scoring, MIT),
used by https://github.com/romanows/SDOH-n2c2 as *the* scorer for this task
-- the closest thing that exists to "official" for SHAC. It's vendored
verbatim under multitask/vendor/brat_scoring/ (pip install of it hit an
unrelated setuptools/_distutils_hack bug in this venv building ANY git+
package; vendoring the ~90KB pure-python package avoids that entirely) but
is event-only (its scoring.py has zero relation-matching code) and no
longer the scoring path used here -- see multitask.brat_scorer, the new
unified scorer this and the other two BRAT-based convert_back modules now
share. Its event/labeled-argument logic was ported from brat_scoring's
get_equivalent_triggers_dist/get_entity_matches (proven to hit P=R=F1=1.0
on this dataset's own round-trip self-check below); brat_scoring itself is
left in the repo untouched for a standalone leaderboard-comparable
cross-check.

Input target schema (see multitask.builders.sdoh_2022_t2):
    {"events": [
        {"id": "E2", "type": "Tobacco", "trigger_text": "tobacco",
         "roles": {"Status": {"type": "StatusTime", "text": "No history"}},
         "attributes": [{"role": "Status", "name": "StatusTimeVal", "value": "none"}]},
     ]}

MIMIC and UW are two independent sources, each with its own gold dir
(2022n2c2_track2/<split>/<source>/); GOLD_DIR is a function of the two, not
a single constant, since they're never merged.

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
    document), so they're anchored with reanchor(..., near=trigger_end)
    -- nearest occurrence in either direction -- rather than a forward-only
    cursor. Anchored to the trigger's *end*, not its start: fillers almost
    always immediately follow the trigger ("Tobacco Use: denies"), so
    anchoring on the start systematically mis-picks among several
    same-text fillers spaced roughly one trigger-width apart (e.g. three
    consecutive "Tobacco/Alcohol/Drug Use: denies" clauses -- start-anchored
    search pulls each "denies" toward the wrong, textually-nearer trigger).

Run standalone to self-check both sources' test splits against gold (event
scoring), plus the derived-relations check against test/uw_r/ (see
to_relations_ann/score_relations below). Like medication_event_2022_t1, this
is not a perfect round-trip -- role fillers with repeated same-text mentions
near their trigger (e.g. two structurally-identical "denies" clauses in a
row) can be genuinely ambiguous even with the near=trigger_end heuristic
above, so the ceiling is inherently below 1.0; the derived-relations axis
(uw/test only, against test/uw_r/) compounds that same per-filler ambiguity
across every role of an event at once (a relation needs both its trigger
AND filler spans to match exactly), so its ceiling is lower still.

Scored via multitask.brat_scorer (see that module's docstring) rather than
the vendored brat_scoring package. One deliberate behavior difference: for
labeled arguments (StatusTime/StatusEmploy/TypeLiving), brat_scoring's
default LABEL mode counts a match on subtype/value equality ALONE, with no
span requirement at all; brat_scorer additionally requires the argument's
span to match (see score_events' role_compat) -- a stricter, more literal
reading of "the same annotated span, correctly classified" that reports a
lower number for the exact same predictions whenever a reanchor()-picked
span lands on the wrong occurrence of a repeated value (e.g. "denies")
despite getting the classification right by luck. Current result (run this
file standalone to reproduce): events.triggers F1=0.9951 (mimic) / 0.9956
(uw), events.roles F1=0.9813 (mimic) / 0.9825 (uw), events.combined
F1=0.9862 (mimic) / 0.9872 (uw); derived relations (uw/test) F1=0.9826
(unaffected by the note below -- it's derived from gold events directly,
never from a hallucination-prone prediction). Precision equals recall
exactly on the events axes -- an unresolvable trigger/filler now gets a
poison span instead of being dropped, so on a gold round-trip every miss
also costs a wrong guess; see to_ann's docstring.
    python -m multitask.convert_back.sdoh_2022_t2
"""
import re
import sys
import tempfile
from pathlib import Path

from inference_and_eval import brat_scorer
from inference_and_eval.brat import BratDoc, parse_ann
from inference_and_eval.reanchor import reanchor

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "2022n2c2_track2"


def _gold_dir(source, split):
    return BASE / split / source


# run_eval.py's generic _run_brat() looks this dict up by dataset name for
# datasets that don't have a single constant GOLD_DIR (see its module
# docstring note about MIMIC/UW never being merged).
GOLD_DIR_BY_DATASET = {
    "2022n2c2_track2_mimic": _gold_dir("mimic", "test"),
    "2022n2c2_track2_uw": _gold_dir("uw", "test"),
}


def to_ann(target, doc_text):
    """target: the JSON dict (events only). Returns .ann file text.

    A trigger or filler whose text can't be found anywhere in doc_text
    gets a poison span (past doc_text, monotonically increasing) instead
    of being dropped. For the trigger, dropping used to skip the whole
    event (`continue`), which meant score_events never saw it and its
    roles/attributes cost nothing in scoring -- poisoning it instead keeps
    the event in the .ann, and score_events's own "unmatched sys event ->
    fp for all its roles/attributes" accounting (see that function's
    docstring) then counts it as a false positive on every axis
    automatically. For a filler within an otherwise-real, matched event,
    the same silent-drop-costs-nothing gap applies at role granularity;
    poisoning it means it still can't match a real gold filler (guaranteed
    span-incompatible) but is no longer invisible to the roles fp count."""
    used_spans = []
    cursor = 0
    lines = []
    tid = eid = aid = 0
    poison = len(doc_text) + 1

    def _poison_span():
        nonlocal poison
        span = (poison, poison + 1)
        poison += 1
        return span

    for ev in target.get("events", []):
        trig_span = reanchor(ev["trigger_text"], doc_text, used_spans, min_start=cursor)
        if trig_span is None:
            trig_span = _poison_span()
        else:
            cursor = max(cursor, trig_span[1])
        t_start, t_end = trig_span
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
                f_span = reanchor(filler_text, doc_text, [], near=t_end)
            if f_span is None:
                f_span = _poison_span()
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


def _relation_role_name(event_type, role_name):
    """uw_r's relation-type string is not always the bare event-line role
    name -- confirmed by exhaustively diffing every (event_type, role_name)
    pair against uw_r's actual relation-type vocabulary:
      - Multi-valued roles (Drug can have Type/Type2/Type3, Tobacco
        Amount/Amount2, etc. -- one E-line role per extra mention of the
        same slot) all collapse to one relation type with the trailing
        digit stripped ("Type2" -> "Type", "Amount3" -> "Amount") -- e.g.
        test/uw/1268.ann's E3 has both Type:T9 and Type2:T16, and
        test/uw_r/1268.ann encodes both as two separate "Drug-Type"
        R-lines, not "Drug-Type"/"Drug-Type2".
      - LivingStatus's "Type" role is the one exception that is NOT the
        bare role name: it's named "TypeLiving" in relations (matching
        that role's own T-line entity type, which -- only for
        LivingStatus -- is literally "TypeLiving" rather than "Type";
        every other event type's Type-role filler has T-line type "Type",
        so this is invisible for them). Confirmed via test/uw_r/1205.ann's
        "LivingStatus-TypeLiving Arg1:T1 Arg2:T4" against test/uw/1205.ann's
        "E1  LivingStatus:T1 Status:T2 History:T3 Type:T4"."""
    if event_type == "LivingStatus" and role_name == "Type":
        return "TypeLiving"
    return re.sub(r"\d+$", "", role_name)


def to_relations_ann(target, doc_text):
    """Derive a relation-form .ann (matching 2022n2c2_track2/test/uw_r/'s
    convention: "R<n>  {EventType}-{RelationRoleName} Arg1:<trigger T-id>
    Arg2:<filler T-id>", see _relation_role_name for how RelationRoleName is
    derived from the event-line role name) from an event-form target JSON.
    Confirmed by direct inspection that this is exactly what uw_r contains
    -- e.g. test/uw_r/1203.ann's "R1  Drug-History Arg1:T5 Arg2:T6"
    corresponds exactly to test/uw/1203.ann's event "E3  Drug:T5
    History:T6 Status:T7 Method:T8" (same trigger+filler spans, decomposed
    into pairwise relations instead of one N-ary event frame) -- so this is
    a mechanical re-encoding of the same event target, not new information,
    and is used only to drive the auxiliary relation-scoring check below
    (UW test only, against 2022n2c2_track2/test/uw_r/)."""
    used_spans = []
    cursor = 0
    lines = []
    tid = rid = 0

    for ev in target.get("events", []):
        trig_span = reanchor(ev["trigger_text"], doc_text, used_spans, min_start=cursor)
        if trig_span is None:
            continue
        t_start, t_end = trig_span
        cursor = max(cursor, t_end)
        tid += 1
        trigger_tid = f"T{tid}"
        lines.append(f"{trigger_tid}\t{ev['type']} {t_start} {t_end}\t{ev['trigger_text']}")

        for role_name, filler in ev.get("roles", {}).items():
            filler_text = filler["text"]
            if filler_text == ev["trigger_text"]:
                f_span = (t_start, t_end)  # shared-mention convention -- see module docstring
            else:
                f_span = reanchor(filler_text, doc_text, [], near=t_end)
            if f_span is None:
                continue
            f_start, f_end = f_span
            tid += 1
            filler_tid = f"T{tid}"
            lines.append(f"{filler_tid}\t{filler['type']} {f_start} {f_end}\t{filler_text}")
            rid += 1
            rel_role = _relation_role_name(ev["type"], role_name)
            lines.append(f"R{rid}\t{ev['type']}-{rel_role} Arg1:{trigger_tid} Arg2:{filler_tid}")

    return "\n".join(lines) + ("\n" if lines else "")


def write_relations_anns(predictions, out_dir):
    """predictions: iterable of (doc_id, target, doc_text). Writes one
    derived-relations <doc_id>.ann per prediction -- no paired .txt needed
    since score_relations() (below) parses spans directly, unlike the
    vendored brat_scoring package used for event scoring."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for doc_id, target, doc_text in predictions:
        (out_dir / f"{doc_id}.ann").write_text(to_relations_ann(target, doc_text), encoding="utf-8")
    return out_dir


def score_relations(gold_dir, system_dir):
    """Micro P/R/F1 over derived relations via multitask.brat_scorer's
    score_relations (arg_type_pairs=None -- score every relation with
    resolvable arguments, regardless of type, since uw_r's type vocabulary
    is a per-event-type/role-name string, not a small fixed set of pairs).
    No official scorer exists for this axis -- the vendored brat_scoring
    package is event-oriented only, with no path for scoring bare relations
    -- so this remains our own metric, documented as such, not an official
    binary (same pattern as disorder_normalization's self-implemented
    strict/relaxed F), just no longer a hand-rolled exact-set matcher."""
    gold_dir, system_dir = Path(gold_dir), Path(system_dir)
    tp = fp = fn = 0
    for ann_path in sorted(gold_dir.glob("*.ann")):
        gold_doc = parse_ann(ann_path)
        sys_path = system_dir / ann_path.name
        sys_doc = parse_ann(sys_path) if sys_path.exists() else BratDoc({}, {}, {}, [])
        m = brat_scorer.score_relations(gold_doc, sys_doc, arg_type_pairs=None, mode="strict")
        tp += m.tp
        fp += m.fp
        fn += m.fn
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    print(f"relations (derived, brat_scorer micro P/R/F1): P={p:.4f} R={r:.4f} F1={f1:.4f} (TP={tp} FP={fp} FN={fn})")
    return {"precision": p, "recall": r, "f1": f1, "tp": tp, "fp": fp, "fn": fn}



def _event_key_counters(doc):
    """doc: a parsed BRAT Doc (multitask.brat.parse_ann). Returns two
    dict-of-set summaries used only by _self_check()'s pre-flight sanity
    check, not for real scoring (see score_dirs, which calls brat_scorer)."""
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


def score_dirs(gold_dir, system_dir):
    """Score system_dir's .ann predictions against gold_dir using the new
    unified brat_scorer (see module docstring) rather than the vendored
    brat_scoring package."""
    brat_scorer.score_dirs(gold_dir, system_dir, brat_scorer.CONFIGS["2022n2c2_track2"])


def _self_check(source):
    """Round-trip gold test-split JSON (built by the dataset builder) back
    to .ann, then score against the real gold -- see module docstring for
    why this isn't an exact P=R=F1=1.0 round-trip."""
    from inference_and_eval.io_utils import read_text

    gd = _gold_dir(source, "test")
    predictions = []
    for ex in _examples_with_raw_target(source, "test"):
        doc_text = read_text(gd / f"{ex['id']}.txt")
        predictions.append((ex["id"], ex["target"], doc_text))

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = write_anns(predictions, tmp)

        # Fast pre-check (trigger/role span+type sets only) so an obvious
        # reanchor regression fails immediately instead of waiting on the
        # slower real scorer below.
        mismatches = 0
        for doc_id, _, _ in predictions:
            gold_doc = parse_ann(gd / f"{doc_id}.ann")
            gen_doc = parse_ann(out_dir / f"{doc_id}.ann")
            if _event_key_counters(gold_doc) != _event_key_counters(gen_doc):
                mismatches += 1
        print(f"pre-check ({source}): {mismatches}/{len(predictions)} docs with a trigger/role span+type mismatch")

        print(f"=== self-check ({source}/test): gold JSON -> .ann -> brat_scorer ===")
        score_dirs(gd, out_dir)


def _self_check_relations():
    """Round-trip gold uw/test event JSON -> derived relations .ann ->
    score against the real test/uw_r/ relation gold. Relations are
    mechanically derivable from the same event annotation (not
    independently annotated information; see to_relations_ann's docstring),
    but the round-trip still isn't exact -- see module docstring (micro F1
    ~0.982, lower than the event axis since a relation needs both its
    trigger and filler spans to reanchor correctly at once)."""
    from inference_and_eval.io_utils import read_text

    uw_r_dir = BASE / "test" / "uw_r"
    gd = _gold_dir("uw", "test")
    predictions = []
    for ex in _examples_with_raw_target("uw", "test"):
        gold_ann = uw_r_dir / f"{ex['id']}.ann"
        if not gold_ann.exists():
            continue
        doc_text = read_text(gd / f"{ex['id']}.txt")
        predictions.append((ex["id"], ex["target"], doc_text))

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = write_relations_anns(predictions, tmp)
        print(f"=== self-check (uw/test relations, derived): gold event JSON -> relations .ann -> uw_r gold ({len(predictions)} docs) ===")
        score_relations(uw_r_dir, out_dir)


def _examples_with_raw_target(source, split):
    """Rebuild examples with the raw target dict attached (schemas.example()
    only keeps rendered text/messages)."""
    from inference_and_eval.builders.sdoh_2022_t2 import _split_dir

    split_dir = _split_dir(source, split)
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
        _self_check("mimic")
        _self_check("uw")
        _self_check_relations()
