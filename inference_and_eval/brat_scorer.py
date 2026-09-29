"""
Unified BRAT scorer for the three brat-standoff datasets in this repo
(2018_n2c2_track2, 2022n2c2_track1, 2022n2c2_track2), replacing three
previously-separate external tools:

  - 2018_n2c2_track2/Track2-evaluate-ver4.py (official)
  - 2022n2c2_track1/evaluation_script/eval_script.py (official)
  - multitask/vendor/brat_scoring/ (Lybarger's package, vendored, event-only)

Both official scripts share the same matching heuristic: iterate system tags,
scan gold tags for an equals() hit, and drop a later system tag if its
matching gold tag was already claimed by an earlier one. That is a single,
iteration-order-dependent pass, not a real assignment algorithm -- with two
system tags overlapping one gold tag (lenient mode), which one "wins" depends
on dict/list iteration order, and a bad order can double-count or mis-count.

The vendored brat_scoring package does better for its one axis (events):
get_equivalent_triggers_dist sorts every compatible (gold, system) pair by
trigger-midpoint distance and greedily claims the globally-closest pair first,
excluding both indices from further matches -- a real (if only greedy, not
globally optimal) assignment. But that package is event-only: scoring.py has
zero relation-matching code, so it cannot score 2018_n2c2_track2 (flat T+R,
no events at all) even in principle.

This module generalizes that one correct idea (_greedy_match) into a single
primitive, then builds all three axes (tags/relations/events) on top of it,
operating entirely on multitask.brat.parse_ann's own dataset-agnostic
Entity/Event/Relation/Attribute objects -- no import from multitask/vendor/,
no dependency on either official script. Those three original tools are left
in the repo untouched, still runnable standalone for a leaderboard-comparable
cross-check, but are no longer the default scoring path once this lands.

Run standalone to fixture-test the scorer itself (small synthetic .ann pairs,
one per axis, with exact expected TP/FP/FN -- see _self_check_fixtures()):
    python -m multitask.brat_scorer
"""
from dataclasses import dataclass
from pathlib import Path

from inference_and_eval.brat import BratDoc, parse_ann


@dataclass
class Measures:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def __add__(self, other):
        return Measures(self.tp + other.tp, self.fp + other.fp, self.fn + other.fn)

    @property
    def precision(self):
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    @property
    def recall(self):
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    @property
    def f1(self):
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def as_dict(self):
        return {"precision": self.precision, "recall": self.recall, "f1": self.f1,
                "tp": self.tp, "fp": self.fp, "fn": self.fn}


def _greedy_match(gold_items, sys_items, is_compatible, distance):
    """Sort every compatible (gold, sys) pair by distance ascending; greedily
    claim the closest pair first, excluding both indices from all further
    matches. A greedy approximation of bipartite min-cost matching (not
    globally optimal -- that would need scipy.optimize.linear_sum_assignment
    or an ILP) but a real assignment algorithm, unlike the single
    iteration-order-dependent pass both official n2c2 scripts use. Same idea
    as brat_scoring.scoring.get_equivalent_triggers_dist, generalized past
    just trigger spans so it can back tags/relations/events alike.

    Returns a list of (gold_index, sys_index) pairs into the input lists."""
    pairs = []
    for i, g in enumerate(gold_items):
        for j, s in enumerate(sys_items):
            if is_compatible(g, s):
                pairs.append((distance(g, s), i, j))
    pairs.sort(key=lambda t: t[0])

    matched_g, matched_s, matches = set(), set(), []
    for _, i, j in pairs:
        if i in matched_g or j in matched_s:
            continue
        matched_g.add(i)
        matched_s.add(j)
        matches.append((i, j))
    return matches


def _midpoint(entity):
    return (entity.start + entity.end) / 2.0


def _span_compatible(a, b, mode):
    if mode == "strict":
        return a.start == b.start and a.end == b.end
    if mode == "lenient":
        return a.start < b.end and b.start < a.end  # any overlap
    raise ValueError(f"unknown mode {mode!r} (expected 'strict' or 'lenient')")


def score_tags(gold_doc, sys_doc, types, mode="strict"):
    """Entity-only axis. Replaces both official scripts' tag scoring."""
    types = set(types)
    gold = [e for e in gold_doc.entities.values() if e.type in types]
    sys_ = [e for e in sys_doc.entities.values() if e.type in types]

    def compat(g, s):
        return g.type == s.type and _span_compatible(g, s, mode)

    def dist(g, s):
        return abs(_midpoint(g) - _midpoint(s))

    matches = _greedy_match(gold, sys_, compat, dist)
    tp = len(matches)
    return Measures(tp=tp, fp=len(sys_) - tp, fn=len(gold) - tp)


def _relation_items(doc, allowed_types):
    items = []
    for r in doc.relations.values():
        if allowed_types is not None and r.type not in allowed_types:
            continue
        e1 = doc.entities.get(r.arg1)
        e2 = doc.entities.get(r.arg2)
        if e1 is None or e2 is None:
            continue
        items.append((r, e1, e2))
    return items


def score_relations(gold_doc, sys_doc, arg_type_pairs=None, mode="strict"):
    """Relation axis: same rtype + both arguments span-compatible per mode.
    No prior correct implementation of this existed anywhere in this repo or
    its vendored/official tooling (brat_scoring has none at all; the two
    official scripts only ever scored flat tags/attributes, not relations
    against each other via a real assignment -- 2018_n2c2_track2's own
    "relations" numbers came from Track2-evaluate-ver4.py's pare-down pass,
    same bug as its tag axis).

    arg_type_pairs: iterable of (arg1_type, arg2_type) pairs whose joined
    "arg1_type-arg2_type" must equal a relation's .type for it to be scored
    (2018_n2c2_track2's convention -- every relation IS "OtherType-Drug").
    None scores every relation with resolvable arguments regardless of type
    (used for sdoh_2022_t2's derived-relations self-check, whose type
    vocabulary is a different per-event-type/role-name string -- see
    sdoh_2022_t2._relation_role_name -- not a fixed small set of pairs)."""
    allowed_types = None if arg_type_pairs is None else {f"{a}-{b}" for a, b in arg_type_pairs}
    gold = _relation_items(gold_doc, allowed_types)
    sys_ = _relation_items(sys_doc, allowed_types)

    def compat(g, s):
        (gr, ge1, ge2), (sr, se1, se2) = g, s
        return (gr.type == sr.type
                and _span_compatible(ge1, se1, mode)
                and _span_compatible(ge2, se2, mode))

    def dist(g, s):
        (_, ge1, ge2), (_, se1, se2) = g, s
        return abs(_midpoint(ge1) - _midpoint(se1)) + abs(_midpoint(ge2) - _midpoint(se2))

    matches = _greedy_match(gold, sys_, compat, dist)
    tp = len(matches)
    return Measures(tp=tp, fp=len(sys_) - tp, fn=len(gold) - tp)


def _attrs_by_target(doc):
    by_target = {}
    for a in doc.attributes:
        by_target.setdefault(a.target, []).append(a)
    return by_target


def _subtype_value(attrs_by_target, entity_id):
    """The comparable subtype/value for a labeled argument -- e.g. SDOH's
    StatusTime entity T7 has a companion A-line "StatusTimeVal T7 current";
    multitask.brat.parse_ann already captures that as
    Attribute(id, name="StatusTimeVal", target="T7", value="current"), so
    this is exactly brat_scoring.labels.tb2entities' subtype = attr.value
    logic, just read off the repo's own generic Attribute list instead of
    the vendored package's object model."""
    vals = attrs_by_target.get(entity_id, [])
    return vals[0].value if vals else None


def score_events(gold_doc, sys_doc, event_config, mode="strict", trigger_mode=None):
    """Two-level greedy match: first triggers (same event type + span
    compatible per trigger_mode, nearest-distance greedy -- same idea as
    brat_scoring.scoring.get_equivalent_triggers_dist), then for each matched
    trigger pair, greedy-match its role arguments the same way (a labeled
    argument -- event_config["labeled_args"], entity *type* names, e.g. SDOH's
    StatusTime/StatusEmploy/TypeLiving -- additionally requires its Val
    attribute value to match, mirroring brat_scoring's LABELED_ARGUMENTS
    split). Event-level categorical attributes (event_config["attributes"],
    e.g. 2022n2c2_track1's Action/Certainty/Temporality/Actor/Negation, which
    attach to the E-id directly rather than to a role-filler entity) are
    compared per matched trigger pair too.

    trigger_mode: span-match strictness for triggers specifically, defaults
    to `mode` if not given. brat_scoring's own run_sdoh_scoring.py defaults
    to OVERLAP (this module's "lenient") for triggers while using EXACT
    ("strict") for span-only arguments -- a trigger is "the same mention"
    once any part of it overlaps, even if reanchor() reconstruction landed
    the span a token or two off; role/argument identity is held to the
    tighter standard. `mode` continues to govern role/argument span
    comparisons.

    Unmatched gold events contribute fn for all their roles/attributes;
    unmatched system events contribute fp for all of theirs -- an event
    whose trigger didn't match at all can't have any of its arguments
    "right by accident".

    Returns {"triggers": Measures, "roles": Measures, "attributes": Measures}.
    """
    trigger_mode = mode if trigger_mode is None else trigger_mode
    types = set(event_config.get("types", ()))
    labeled_arg_types = set(event_config.get("labeled_args", ()))
    attr_names = tuple(event_config.get("attributes", ()))

    gold_attrs = _attrs_by_target(gold_doc)
    sys_attrs = _attrs_by_target(sys_doc)

    def trigger_entity(doc, ev):
        return doc.entities.get(ev.trigger)

    valid_gold = [ev for ev in gold_doc.events.values()
                  if ev.type in types and trigger_entity(gold_doc, ev) is not None]
    valid_sys = [ev for ev in sys_doc.events.values()
                 if ev.type in types and trigger_entity(sys_doc, ev) is not None]

    def trig_compat(g, s):
        return g.type == s.type and _span_compatible(
            trigger_entity(gold_doc, g), trigger_entity(sys_doc, s), trigger_mode)

    def trig_dist(g, s):
        return abs(_midpoint(trigger_entity(gold_doc, g)) - _midpoint(trigger_entity(sys_doc, s)))

    trig_matches = _greedy_match(valid_gold, valid_sys, trig_compat, trig_dist)
    matched_g_idx = {i for i, _ in trig_matches}
    matched_s_idx = {j for _, j in trig_matches}
    triggers = Measures(tp=len(trig_matches),
                         fp=len(valid_sys) - len(trig_matches),
                         fn=len(valid_gold) - len(trig_matches))

    def role_items(doc, ev):
        items = []
        for role_name, target_id in ev.roles.items():
            filler = doc.entities.get(target_id)
            if filler is not None:
                items.append((role_name, filler))
        return items

    def role_compat(g, s):
        (gr, gf), (sr, sf) = g, s
        if gr != sr or gf.type != sf.type or not _span_compatible(gf, sf, mode):
            return False
        if gf.type in labeled_arg_types:
            if _subtype_value(gold_attrs, gf.id) != _subtype_value(sys_attrs, sf.id):
                return False
        return True

    def role_dist(g, s):
        return abs(_midpoint(g[1]) - _midpoint(s[1]))

    roles = Measures()
    for i, j in trig_matches:
        g_roles = role_items(gold_doc, valid_gold[i])
        s_roles = role_items(sys_doc, valid_sys[j])
        matches = _greedy_match(g_roles, s_roles, role_compat, role_dist)
        tp = len(matches)
        roles = roles + Measures(tp=tp, fp=len(s_roles) - tp, fn=len(g_roles) - tp)
    for i, ev in enumerate(valid_gold):
        if i not in matched_g_idx:
            roles = roles + Measures(fn=len(role_items(gold_doc, ev)))
    for j, ev in enumerate(valid_sys):
        if j not in matched_s_idx:
            roles = roles + Measures(fp=len(role_items(sys_doc, ev)))

    def attr_values(attrs_by_target, event_id):
        return {a.name: a.value for a in attrs_by_target.get(event_id, []) if a.name in attr_names}

    attributes = Measures()
    for i, j in trig_matches:
        g_vals = attr_values(gold_attrs, valid_gold[i].id)
        s_vals = attr_values(sys_attrs, valid_sys[j].id)
        for name in attr_names:
            g_v, s_v = g_vals.get(name), s_vals.get(name)
            if g_v is None and s_v is None:
                continue
            if g_v is not None and g_v == s_v:
                attributes = attributes + Measures(tp=1)
            else:
                if g_v is not None:
                    attributes = attributes + Measures(fn=1)
                if s_v is not None:
                    attributes = attributes + Measures(fp=1)
    for i, ev in enumerate(valid_gold):
        if i not in matched_g_idx:
            attributes = attributes + Measures(fn=len(attr_values(gold_attrs, ev.id)))
    for j, ev in enumerate(valid_sys):
        if j not in matched_s_idx:
            attributes = attributes + Measures(fp=len(attr_values(sys_attrs, ev.id)))

    return {"triggers": triggers, "roles": roles, "attributes": attributes}


# Per-dataset config (plain dicts -- no new classes). "tags"/"relations" feed
# score_tags/score_relations; "events" feeds score_events.
CONFIGS = {
    "2018_n2c2_track2": {
        "tags": ("Drug", "Strength", "Duration", "Route", "Form", "ADE", "Dosage", "Reason", "Frequency"),
        "relations": (("Strength", "Drug"), ("Dosage", "Drug"), ("Duration", "Drug"),
                       ("Frequency", "Drug"), ("Form", "Drug"), ("Route", "Drug"),
                       ("Reason", "Drug"), ("ADE", "Drug")),
    },
    "2022n2c2_track1": {
        # No "tags" axis: this converter never emits a bare "Drug" T-line
        # (see medication_event_2022_t1.to_ann) -- every T-line IS an event
        # trigger typed Disposition/NoDisposition/Undetermined, so the
        # events axis alone already covers everything this dataset scores.
        "events": {
            "types": ("Disposition", "NoDisposition", "Undetermined"),
            "labeled_args": (),
            "attributes": ("Action", "Certainty", "Temporality", "Actor", "Negation"),
        },
    },
    "2022n2c2_track2": {
        "events": {
            # 12 SDOH determinants (multitask/vendor/brat_scoring/brat_scoring/constants_sdoh.py)
            "types": ("Alcohol", "Country", "Drug", "Employment", "EnviroExposure", "GenderID",
                       "Insurance", "LivingStatus", "PhysActivity", "Race", "SexualOrient", "Tobacco"),
            "labeled_args": ("StatusTime", "StatusEmploy", "TypeLiving"),  # constants_sdoh.LABELED_ARGUMENTS
            "attributes": (),
            # brat_scoring's run_sdoh_scoring.py defaults score_trig=OVERLAP
            # (this module's "lenient") while scoring arguments EXACT -- see
            # score_events' trigger_mode docstring.
            "trigger_mode": "lenient",
        },
    },
}


def score_dirs(gold_dir, system_dir, config, mode="strict"):
    """Score system_dir's .ann predictions against gold_dir using whichever
    axes are present in config ("tags"/"relations"/"events"). Iterates
    gold_dir's own .ann filenames (a missing system file scores as an empty
    prediction, same convention as every convert_back module's _run_eval).
    Prints per-axis micro P/R/F1 (plus a pooled "events.combined" when an
    events axis is present, matching the shape of module docstrings' single
    "event micro F1" figures) and returns the same numbers as a dict."""
    gold_dir, system_dir = Path(gold_dir), Path(system_dir)
    ann_names = sorted(p.name for p in gold_dir.glob("*.ann"))

    totals = {}
    for name in ann_names:
        gold_doc = parse_ann(gold_dir / name)
        sys_path = system_dir / name
        sys_doc = parse_ann(sys_path) if sys_path.exists() else BratDoc({}, {}, {}, [])

        if "tags" in config:
            totals["tags"] = totals.get("tags", Measures()) + score_tags(gold_doc, sys_doc, config["tags"], mode)
        if "relations" in config:
            totals["relations"] = totals.get("relations", Measures()) + score_relations(
                gold_doc, sys_doc, config["relations"], mode)
        if "events" in config:
            event_config = config["events"]
            trigger_mode = event_config.get("trigger_mode")
            for axis, m in score_events(gold_doc, sys_doc, event_config, mode, trigger_mode).items():
                key = f"events.{axis}"
                totals[key] = totals.get(key, Measures()) + m

    if "events" in config:
        combined = Measures()
        for axis in ("events.triggers", "events.roles", "events.attributes"):
            combined = combined + totals.get(axis, Measures())
        totals["events.combined"] = combined

    results = {}
    for key, m in totals.items():
        results[key] = m.as_dict()
        if m.tp == m.fp == m.fn == 0:
            # An axis this dataset doesn't use (e.g. 2022n2c2_track1 has no
            # role-labeled arguments, 2022n2c2_track2 has no event-level
            # attributes) has a zero denominator on every metric. Printing
            # that as "P=0.0000 R=0.0000 F1=0.0000" reads as a catastrophic
            # failure when the axis is simply measuring nothing.
            print(f"{key}: n/a -- axis not present in this dataset (TP=FP=FN=0)")
        else:
            print(f"{key}: P={m.precision:.4f} R={m.recall:.4f} F1={m.f1:.4f} (TP={m.tp} FP={m.fp} FN={m.fn})")
    return results


def _write_ann(path, text):
    Path(path).write_text(text, encoding="utf-8")


def _self_check_fixtures():
    """Small synthetic gold/sys .ann pairs, one per axis, asserting exact
    expected TP/FP/FN -- modeled on the i2b2_2010.py self-check pattern
    (score() + assert exact P/R/F1 values). Covers: strict exact match,
    lenient overlap with a tie (two system tags overlapping one gold tag --
    must count as 1 TP + 1 FP, not 2 TP, proving the greedy matcher fixes
    the official scripts' order-dependent double-count bug), relation
    matching, and event + labeled-argument matching."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        gold_dir, sys_dir = tmp / "gold", tmp / "sys"
        gold_dir.mkdir()
        sys_dir.mkdir()

        # --- tags: strict exact match + lenient overlap tie ---
        _write_ann(gold_dir / "doc1.ann", "T1\tDrug 0 5\tAspir\nT2\tDrug 10 15\tIbupro\n")
        _write_ann(sys_dir / "doc1.ann",
                   "T1\tDrug 0 5\tAspir\n"       # exact match with gold T1
                   "T2\tDrug 8 13\tXXXXX\n"      # overlaps gold T2 [10,15)
                   "T3\tDrug 12 20\tYYYYY\n")    # also overlaps gold T2 -> tie
        gold_doc = parse_ann(gold_dir / "doc1.ann")
        sys_doc = parse_ann(sys_dir / "doc1.ann")

        m_strict = score_tags(gold_doc, sys_doc, ("Drug",), mode="strict")
        assert (m_strict.tp, m_strict.fp, m_strict.fn) == (1, 2, 1), m_strict

        m_lenient = score_tags(gold_doc, sys_doc, ("Drug",), mode="lenient")
        # T1 matches T1 exactly; of {sys T2, sys T3} only one can claim gold
        # T2 (both overlap it) -- greedy picks the nearer one, the other is a
        # bare FP, so this must be 2 TP / 1 FP / 0 FN, never 3 TP / 0 FP.
        assert (m_lenient.tp, m_lenient.fp, m_lenient.fn) == (2, 1, 0), m_lenient

        # --- relations ---
        _write_ann(gold_dir / "doc2.ann",
                   "T1\tStrength 0 4\t10mg\nT2\tDrug 10 16\tAspirin\n"
                   "T3\tStrength 20 24\t5mg_\nT4\tDrug 30 36\tAdvil_\n"
                   "R1\tStrength-Drug Arg1:T1 Arg2:T2\n"
                   "R2\tStrength-Drug Arg1:T3 Arg2:T4\n")
        _write_ann(sys_dir / "doc2.ann",
                   "T1\tStrength 0 4\t10mg\nT2\tDrug 10 16\tAspirin\n"
                   "T3\tStrength 20 24\t5mg_\n"
                   "R1\tStrength-Drug Arg1:T1 Arg2:T2\n")  # second relation's Drug arg missing
        gold_doc2 = parse_ann(gold_dir / "doc2.ann")
        sys_doc2 = parse_ann(sys_dir / "doc2.ann")
        m_rel = score_relations(gold_doc2, sys_doc2, (("Strength", "Drug"),), mode="strict")
        assert (m_rel.tp, m_rel.fp, m_rel.fn) == (1, 0, 1), m_rel

        # --- events + labeled argument (subtype/value) matching ---
        _write_ann(gold_dir / "doc3.ann",
                   "T1\tTobacco 0 7\tTobacco\nT2\tStatusTime 10 16\tcurren\n"
                   "E1\tTobacco:T1 Status:T2\n"
                   "A1\tStatusTimeVal T2 current\n")
        _write_ann(sys_dir / "doc3.ann",
                   "T1\tTobacco 0 7\tTobacco\nT2\tStatusTime 10 16\tcurren\n"
                   "E1\tTobacco:T1 Status:T2\n"
                   "A1\tStatusTimeVal T2 past\n")  # trigger+span match, value wrong
        gold_doc3 = parse_ann(gold_dir / "doc3.ann")
        sys_doc3 = parse_ann(sys_dir / "doc3.ann")
        event_config = {"types": ("Tobacco",), "labeled_args": ("StatusTime",), "attributes": ()}
        ev = score_events(gold_doc3, sys_doc3, event_config, mode="strict")
        assert (ev["triggers"].tp, ev["triggers"].fp, ev["triggers"].fn) == (1, 0, 0), ev["triggers"]
        # role incompatible (Val mismatch) despite identical span -> 1 FP + 1 FN, not 1 TP
        assert (ev["roles"].tp, ev["roles"].fp, ev["roles"].fn) == (0, 1, 1), ev["roles"]

    print("brat_scorer._self_check_fixtures(): all fixture assertions passed")


if __name__ == "__main__":
    _self_check_fixtures()
