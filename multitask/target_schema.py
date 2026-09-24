"""
Single source of truth for every task's target JSON shape.

Before this module the target contract lived in three places that could --
and did -- drift apart:

1. the instruction prose in `schemas.py` (what the model is *told* to emit),
2. the builder code (what the target *actually* is),
3. `convert_back/*` (what is *accepted* back).

Observed drift at the time this was written: SHAC's builder emits 16 role
names (`Amount2`, `Type3`, `Method3`, ...), 9 filler types and 3 attribute
names with closed value sets, while its instruction enumerates none of
them; `convert_back/ner_re_2018_t2.py` defined a canonical `ATTR_TYPES`
order that the builder ignored, so the same attribute key-set was emitted
in up to 6 different orders across examples.

This module declares each target shape once, and that declaration then
drives:

- `canonicalize()` -- deterministic key order, called from
  `schemas._target_str()` so every built example is byte-stable for the
  same content. Key order is otherwise inherited from gold-file layout,
  which gives the model directly contradictory supervision on
  serialization for identical content.
- `validate()` -- structural + closed-vocabulary checking, so a builder
  emitting an unenumerated key becomes a reported violation rather than
  silent training signal.
- `instruction_gaps()` -- cross-checks a task's closed vocabularies against
  its instruction text, catching drift between (1) and (2).
- `json_schema()` -- exports a JSON Schema per task, the artifact a
  grammar-constrained decoding engine (XGrammar/Outlines/Guidance) consumes
  at inference time.

`canonicalize()` deliberately never drops data: unknown keys are preserved
and moved to the end, so `validate()` can report them instead of them
vanishing silently.

Vocabularies below were harvested from the real corpora rather than read
off the annotation guidelines -- see `validate_targets.py`, which
re-derives them and fails if the data ever disagrees.
"""
from dataclasses import dataclass
import re

# --- 2018 n2c2 track 2 (Drug-ADE). This order is the one
# convert_back/ner_re_2018_t2.py already declared but the builder never
# used; adopting it here is what makes the two agree.
MEDICATION_ATTR_TYPES = (
    "Strength", "Frequency", "Dosage", "Route", "Duration", "Form", "Reason", "ADE",
)

# --- 2022 n2c2 track 1 (CMED). Attribute *names* in official order; the
# value sets are per-name (see CMED_ATTR_VALUES).
CMED_ATTR_NAMES = ("Action", "Certainty", "Temporality", "Actor", "Negation")
CMED_ATTR_VALUES = {
    "Action": ("Start", "Stop", "Increase", "Decrease", "UniqueDose", "OtherChange", "Unknown"),
    "Certainty": ("Certain", "Hypothetical", "Conditional", "Unknown"),
    "Temporality": ("Past", "Present", "Future", "Unknown"),
    "Actor": ("Patient", "Physician", "Unknown"),
    "Negation": ("Negated", "NotNegated"),
}
CMED_EVENT_TYPES = ("Disposition", "NoDisposition", "Undetermined")

# --- 2022 n2c2 track 2 (SHAC). Roles repeat with numeric suffixes when one
# event has several fillers for the same role (Amount/Amount2/Amount3).
# Listed explicitly rather than as base x suffix: only these 16 actually
# occur across both sources and all splits, and an over-permissive
# vocabulary would widen the exported decoding grammar for no reason.
# validate() will flag any new suffix if the corpora ever grow one.
SHAC_EVENT_TYPES = ("Tobacco", "Alcohol", "Drug", "Employment", "LivingStatus")
SHAC_ROLE_NAMES = (
    "Status",
    "Type", "Type2", "Type3",
    "Amount", "Amount2", "Amount3",
    "Duration", "Duration2",
    "Frequency", "Frequency2",
    "History", "History2",
    "Method", "Method2", "Method3",
)
SHAC_FILLER_TYPES = (
    "StatusTime", "StatusEmploy", "Type", "TypeLiving",
    "Amount", "Duration", "Frequency", "History", "Method",
)
SHAC_ATTR_VALUES = {
    "StatusTimeVal": ("none", "current", "past", "future"),
    "StatusEmployVal": ("employed", "unemployed", "retired", "on_disability", "student", "homemaker"),
    "TypeLivingVal": ("alone", "with_family", "with_others", "homeless"),
}

# --- 2010 i2b2/VA
I2B2_CONCEPT_TYPES = ("problem", "test", "treatment")
I2B2_ASSERTIONS = (
    "present", "absent", "possible", "conditional", "hypothetical",
    "associated_with_someone_else",
)
I2B2_RELATION_TYPES = ("TrIP", "TrWP", "TrCP", "TrAP", "TrNAP", "TeRP", "TeCP", "PIP")

# --- 2019 n2c2 track 2 (family history). Both "NA" and "None" occur in the
# released TSVs as the no-side value; kept as-is because the official
# eval.py matches these strings literally, but flagged because two spellings
# of one concept is almost certainly an upstream annotation inconsistency.
FH_SIDES = ("Maternal", "Paternal", "NA", "None")
FH_NEGATION = ("Negated", "Non_Negated")
# "3" never occurs in the released data; the guidelines define 0/1/2/4 only.
FH_LIVING_SCORES = ("0", "1", "2", "4")


# --- spec node types -------------------------------------------------------

@dataclass(frozen=True)
class Str:
    """Free text -- a span copied verbatim from the input, or a CUI."""
    nullable: bool = False


@dataclass(frozen=True)
class Enum:
    """Closed vocabulary. `nullable` allows JSON null (e.g. i2b2 assertion
    is null for non-problem concepts)."""
    values: tuple
    nullable: bool = False


@dataclass(frozen=True)
class Arr:
    item: object


@dataclass(frozen=True)
class Obj:
    """Fixed-key object. `keys` is ordered -- that order IS the canonical
    serialization order."""
    keys: dict
    required: tuple = ()


@dataclass(frozen=True)
class DynObj:
    """Object whose keys come from a closed vocabulary and whose presence is
    data-dependent (a drug's attribute types, a SHAC event's roles). The
    vocabulary order is the canonical key order."""
    key_vocab: tuple
    value: object


@dataclass(frozen=True)
class CoupledObj:
    """Object where one key's vocabulary is selected by another key's value.

    SHAC's attribute triples are the only instance: {"role", "name",
    "value"}, where the legal set for "value" depends entirely on "name"
    (StatusTimeVal -> none/current/past/future, StatusEmployVal -> employed/
    ..., TypeLivingVal -> alone/...). A plain Obj cannot express that, and
    typing "value" as a free string means the exported decoding grammar
    accepts `{"name": "StatusTimeVal", "value": "WRONG"}` -- which
    vllm_structured_infer.py then hands to vLLM as if it were enforced.
    Emitting this as a JSON Schema `anyOf` over one concrete branch per tag
    value closes that, and also makes the 13 attribute values visible to
    vocabularies()/instruction_gaps() instead of invisible.
    """
    keys: dict           # fixed keys shared by every branch (e.g. "role")
    tag: str             # discriminator key name, e.g. "name"
    value_key: str       # key whose vocabulary the tag selects, e.g. "value"
    variants: dict       # tag value -> tuple of legal value_key values
    required: tuple = ()

    @property
    def key_order(self):
        return (*self.keys, self.tag, self.value_key)


# --- per-task specs --------------------------------------------------------

_MED_DRUG = Obj(
    keys={
        "text": Str(),
        **{attr: Arr(Str()) for attr in MEDICATION_ATTR_TYPES},
    },
    required=("text",),
)
_MED_UNLINKED = Obj(
    keys={"type": Enum(MEDICATION_ATTR_TYPES), "text": Str()},
    required=("type", "text"),
)
_MED_NER_RE = Obj(
    keys={"drugs": Arr(_MED_DRUG), "unlinked": Arr(_MED_UNLINKED)},
    required=("drugs", "unlinked"),
)

_CMED_EVENT = Obj(
    keys={
        "id": Str(),
        "type": Enum(CMED_EVENT_TYPES),
        "trigger_text": Str(),
        "attributes": Obj(
            keys={name: Enum(CMED_ATTR_VALUES[name]) for name in CMED_ATTR_NAMES},
        ),
    },
    required=("type", "trigger_text"),
)

_SHAC_EVENT = Obj(
    keys={
        "id": Str(),
        "type": Enum(SHAC_EVENT_TYPES),
        "trigger_text": Str(),
        "roles": DynObj(
            key_vocab=SHAC_ROLE_NAMES,
            value=Obj(
                keys={"type": Enum(SHAC_FILLER_TYPES), "text": Str()},
                required=("type", "text"),
            ),
        ),
        "attributes": Arr(CoupledObj(
            # Only Status and Type fillers ever carry an attribute value --
            # verified across both sources and all splits, and enforced by
            # validate_targets.py step 3, which flagged the other 14 role
            # names as declared-but-never-used. Listing all 16 here would
            # widen the exported decoding grammar for no reason.
            keys={"role": Enum(("Status", "Type"))},
            tag="name",
            value_key="value",
            variants=SHAC_ATTR_VALUES,
            required=("role", "name", "value"),
        )),
    },
    required=("type", "trigger_text", "roles", "attributes"),
)

_I2B2_CONCEPT = Obj(
    keys={
        "id": Str(),
        "type": Enum(I2B2_CONCEPT_TYPES),
        "text": Str(),
        "assertion": Enum(I2B2_ASSERTIONS, nullable=True),
    },
    required=("type", "text"),
)
_I2B2_RELATION = Obj(
    keys={"type": Enum(I2B2_RELATION_TYPES), "arg1": Str(), "arg2": Str()},
    required=("type", "arg1", "arg2"),
)
_I2B2_FULL = Obj(
    keys={"concepts": Arr(_I2B2_CONCEPT), "relations": Arr(_I2B2_RELATION)},
    required=("concepts", "relations"),
)

_TYPED_SPAN_OF = {
    "medication": Obj(keys={"type": Enum(MEDICATION_ATTR_TYPES + ("Drug",)), "text": Str()},
                      required=("type", "text")),
    "i2b2": Obj(keys={"type": Enum(I2B2_CONCEPT_TYPES), "text": Str()},
                required=("type", "text")),
    "cmed": Obj(keys={"type": Enum(CMED_EVENT_TYPES), "text": Str()},
                required=("type", "text")),
}

SPECS = {
    # --- end-to-end IE tasks
    "ner_re_medication": _MED_NER_RE,
    "medication_event": Obj(keys={"events": Arr(_CMED_EVENT)}, required=("events",)),
    "sdoh_event": Obj(keys={"events": Arr(_SHAC_EVENT)}, required=("events",)),
    "concept_assertion_relation": _I2B2_FULL,
    "disorder_normalization": Obj(
        keys={"entities": Arr(Obj(keys={"text": Str(), "cui": Str()}, required=("text", "cui")))},
        required=("entities",),
    ),
    "family_history": Obj(
        keys={
            "living_status": Arr(Obj(
                keys={"member": Str(), "side": Enum(FH_SIDES), "score": Enum(FH_LIVING_SCORES)},
                required=("member", "side", "score"))),
            "observations": Arr(Obj(
                keys={"member": Str(), "side": Enum(FH_SIDES), "text": Str(),
                      "negation": Enum(FH_NEGATION)},
                required=("member", "side", "text", "negation"))),
        },
        required=("living_status", "observations"),
    ),

    # --- Tier A subtask / window variants (same shapes, see design.md)
    "ner_re_medication_ner_only": Obj(
        keys={"entities": Arr(_TYPED_SPAN_OF["medication"])}, required=("entities",)),
    "ner_re_medication_re_only": _MED_NER_RE,
    "ner_re_medication_paragraph": _MED_NER_RE,
    "medication_event_trigger_only": Obj(
        keys={"triggers": Arr(_TYPED_SPAN_OF["cmed"])}, required=("triggers",)),
    "medication_event_attr_only": Obj(keys={"events": Arr(_CMED_EVENT)}, required=("events",)),
    "concept_assertion_relation_ner_only": Obj(
        keys={"concepts": Arr(_TYPED_SPAN_OF["i2b2"])}, required=("concepts",)),
    "concept_assertion_relation_re_only": _I2B2_FULL,
    "concept_assertion_relation_section": _I2B2_FULL,
}


# --- canonicalization ------------------------------------------------------

def canonicalize(task, target):
    """Return `target` with every object's keys in the spec's declared
    order. Unknown keys are preserved, appended after the known ones, so
    nothing is silently dropped -- validate() reports them instead.

    Tasks with no spec (classification/QA/summarization, whose targets are
    single-key objects with no ordering ambiguity) pass through unchanged."""
    spec = SPECS.get(task)
    return _canon(spec, target) if spec is not None else target


def _canon(spec, value):
    if isinstance(spec, Obj) and isinstance(value, dict):
        out = {k: _canon(spec.keys[k], value[k]) for k in spec.keys if k in value}
        out.update({k: v for k, v in value.items() if k not in spec.keys})
        return out
    if isinstance(spec, DynObj) and isinstance(value, dict):
        out = {k: _canon(spec.value, value[k]) for k in spec.key_vocab if k in value}
        out.update({k: v for k, v in value.items() if k not in spec.key_vocab})
        return out
    if isinstance(spec, CoupledObj) and isinstance(value, dict):
        out = {k: value[k] for k in spec.key_order if k in value}
        out.update({k: v for k, v in value.items() if k not in spec.key_order})
        return out
    if isinstance(spec, Arr) and isinstance(value, list):
        return [_canon(spec.item, v) for v in value]
    return value


# --- validation ------------------------------------------------------------

def validate(task, target):
    """Return a list of human-readable violation strings ([] if clean).
    Never raises on malformed input -- a bad shape is a violation, not an
    exception, so callers can aggregate across a whole corpus."""
    spec = SPECS.get(task)
    if spec is None:
        return []
    out = []
    _validate(spec, target, "$", out)
    return out


def _validate(spec, value, path, out):
    if isinstance(spec, Obj):
        if not isinstance(value, dict):
            out.append(f"{path}: expected object, got {type(value).__name__}")
            return
        for k in spec.required:
            if k not in value:
                out.append(f"{path}.{k}: required key missing")
        for k, v in value.items():
            if k in spec.keys:
                _validate(spec.keys[k], v, f"{path}.{k}", out)
            else:
                out.append(f"{path}.{k}: key not in spec")
        return
    if isinstance(spec, CoupledObj):
        if not isinstance(value, dict):
            out.append(f"{path}: expected object, got {type(value).__name__}")
            return
        for k in spec.required:
            if k not in value:
                out.append(f"{path}.{k}: required key missing")
        for k, v in value.items():
            if k in spec.keys:
                _validate(spec.keys[k], v, f"{path}.{k}", out)
            elif k not in (spec.tag, spec.value_key):
                out.append(f"{path}.{k}: key not in spec")
        tag_val = value.get(spec.tag)
        if tag_val not in spec.variants:
            out.append(f"{path}.{spec.tag}: {tag_val!r} not in vocabulary")
        elif value.get(spec.value_key) not in spec.variants[tag_val]:
            out.append(
                f"{path}.{spec.value_key}: {value.get(spec.value_key)!r} "
                f"not valid for {spec.tag}={tag_val!r}"
            )
        return
    if isinstance(spec, DynObj):
        if not isinstance(value, dict):
            out.append(f"{path}: expected object, got {type(value).__name__}")
            return
        for k, v in value.items():
            if k in spec.key_vocab:
                _validate(spec.value, v, f"{path}.{k}", out)
            else:
                out.append(f"{path}.{k}: key not in vocabulary")
        return
    if isinstance(spec, Arr):
        if not isinstance(value, list):
            out.append(f"{path}: expected array, got {type(value).__name__}")
            return
        for i, v in enumerate(value):
            _validate(spec.item, v, f"{path}[{i}]", out)
        return
    if isinstance(spec, Enum):
        if value is None:
            if not spec.nullable:
                out.append(f"{path}: null not allowed")
        elif value not in spec.values:
            out.append(f"{path}: {value!r} not in vocabulary")
        return
    if isinstance(spec, Str):
        if value is None:
            if not spec.nullable:
                out.append(f"{path}: null not allowed")
        elif not isinstance(value, str):
            out.append(f"{path}: expected string, got {type(value).__name__}")


# --- instruction cross-check ----------------------------------------------

def vocabularies(task):
    """Every closed vocabulary reachable in this task's spec, as
    {json_path: values} -- what the instruction text ought to enumerate."""
    spec = SPECS.get(task)
    if spec is None:
        return {}
    found = {}
    _collect(spec, "$", found)
    return found


def _collect(spec, path, found):
    if isinstance(spec, Obj):
        for k, v in spec.keys.items():
            _collect(v, f"{path}.{k}", found)
    elif isinstance(spec, DynObj):
        found[f"{path}[keys]"] = spec.key_vocab
        _collect(spec.value, f"{path}.*", found)
    elif isinstance(spec, CoupledObj):
        for k, v in spec.keys.items():
            _collect(v, f"{path}.{k}", found)
        found[f"{path}.{spec.tag}"] = tuple(spec.variants)
        for tag_val, values in spec.variants.items():
            found[f"{path}.{spec.value_key}[{spec.tag}={tag_val}]"] = values
    elif isinstance(spec, Arr):
        _collect(spec.item, f"{path}[]", found)
    elif isinstance(spec, Enum):
        found[path] = spec.values


def _mentioned(value, instruction):
    """True if `value` appears in `instruction` as a whole token.

    A plain `value in instruction` substring test has real false negatives:
    "Certain" is a substring of "Certainty", so a task whose instruction
    enumerates the attribute *names* ("...Action, Certainty, Temporality...")
    but none of their *values* would be reported as already covering the
    value "Certain". Likewise family_history's score "1" is "mentioned" by
    the "2019" in a citation. Underscores count as word characters so
    "on_disability" is not matched by "disability".
    """
    return re.search(
        rf"(?<![A-Za-z0-9_]){re.escape(value)}(?![A-Za-z0-9_])", instruction
    ) is not None


def observed_vocab(task, target, acc):
    """Accumulate the vocabulary values this target actually uses, keyed by
    the SAME json paths vocabularies() reports, into `acc`
    ({path: set(values)}).

    Walking spec and data together is what makes this comparable to
    vocabularies(). A flatten-then-match approach does not work: flattening
    the data yields concrete paths ("$.events[].roles.Status.type") while
    vocabularies() yields wildcard ones ("$.events[].roles.*.type"), so a
    naive `path in declared` test silently skips every DynObj and CoupledObj
    path -- which is to say all 16 SHAC role keys, its 9 filler types and
    its 13 attribute values.
    """
    spec = SPECS.get(task)
    if spec is not None:
        _observe(spec, target, "$", acc)
    return acc


def _observe(spec, value, path, acc):
    if isinstance(spec, Obj) and isinstance(value, dict):
        for k, v in value.items():
            if k in spec.keys:
                _observe(spec.keys[k], v, f"{path}.{k}", acc)
    elif isinstance(spec, DynObj) and isinstance(value, dict):
        for k, v in value.items():
            if k in spec.key_vocab:
                acc.setdefault(f"{path}[keys]", set()).add(k)
                _observe(spec.value, v, f"{path}.*", acc)
    elif isinstance(spec, CoupledObj) and isinstance(value, dict):
        for k, v in value.items():
            if k in spec.keys:
                _observe(spec.keys[k], v, f"{path}.{k}", acc)
        tag_val = value.get(spec.tag)
        if tag_val in spec.variants:
            acc.setdefault(f"{path}.{spec.tag}", set()).add(tag_val)
            val = value.get(spec.value_key)
            if val in spec.variants[tag_val]:
                key = f"{path}.{spec.value_key}[{spec.tag}={tag_val}]"
                acc.setdefault(key, set()).add(val)
    elif isinstance(spec, Arr) and isinstance(value, list):
        for v in value:
            _observe(spec.item, v, f"{path}[]", acc)
    elif isinstance(spec, Enum) and value is not None:
        acc.setdefault(path, set()).add(value)


def instruction_gaps(task, instruction):
    """Closed-vocabulary members this task's spec allows but its instruction
    never mentions -- i.e. things the model is expected to emit without ever
    being told they exist. Returns {json_path: (missing, ...)}."""
    gaps = {}
    for path, values in vocabularies(task).items():
        missing = tuple(v for v in values if v and not _mentioned(v, instruction))
        if missing:
            gaps[path] = missing
    return gaps


# --- JSON Schema export (for grammar-constrained decoding) -----------------

def json_schema(task):
    """Draft-2020-12 JSON Schema for a task's target, or None if unspecced.
    Consumed by XGrammar/Outlines/Guidance at inference time -- note that
    entity `text` is left an unconstrained string on purpose: constraining
    it to document substrings via grammar is enormous and fights the model,
    so span validity is handled in convert_back instead."""
    spec = SPECS.get(task)
    return _to_json_schema(spec) if spec is not None else None


def _obj_schema(properties, required, order):
    return {
        "type": "object",
        "properties": properties,
        "required": list(required),
        "additionalProperties": False,
        # NOT honored by xgrammar (verified against 0.1.33: it reads the
        # `properties` object's own insertion order, which this function
        # preserves, and ignores this key). Kept because Gemini's structured
        # -output API does read it and it is harmless elsewhere -- but do not
        # rely on it to carry key order. Note also that OpenAI/Azure strict
        # structured-output mode REJECTS unknown keywords and requires
        # `required` to list every property, so this export is not portable
        # there without stripping this key.
        "propertyOrdering": list(order),
    }


def _to_json_schema(spec):
    if isinstance(spec, Obj):
        return _obj_schema(
            {k: _to_json_schema(v) for k, v in spec.keys.items()},
            spec.required, spec.keys,
        )
    if isinstance(spec, DynObj):
        return _obj_schema(
            {k: _to_json_schema(spec.value) for k in spec.key_vocab},
            (), spec.key_vocab,
        )
    if isinstance(spec, CoupledObj):
        # One concrete branch per tag value, so the grammar itself enforces
        # that e.g. name=StatusTimeVal implies value in
        # none/current/past/future -- see CoupledObj's docstring.
        return {"anyOf": [
            _obj_schema(
                {
                    **{k: _to_json_schema(v) for k, v in spec.keys.items()},
                    spec.tag: {"enum": [tag_val]},
                    spec.value_key: {"enum": list(values)},
                },
                spec.required, spec.key_order,
            )
            for tag_val, values in spec.variants.items()
        ]}
    if isinstance(spec, Arr):
        return {"type": "array", "items": _to_json_schema(spec.item)}
    if isinstance(spec, Enum):
        values = list(spec.values) + ([None] if spec.nullable else [])
        return {"enum": values}
    return {"type": ["string", "null"] if spec.nullable else "string"}
