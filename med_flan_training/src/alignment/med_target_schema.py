"""Ported single source of truth for every med multitask task's target JSON shape.

Ported (self-contained, no import from meddata -- the meddata repo is not guaranteed
to be on the training cluster, same reasoning as `med_grpo_rewards.py`) from
`meddata/multitask/target_schema.py`. That module declares each task's target shape
once and derives `canonicalize()` (deterministic key order -- called from meddata's
`schemas._target_str()` so every built example is byte-stable) and `validate()`
(structural + closed-vocabulary checking) from it.

Used here by `weighted_json_loss.py` to know, for a given task and target, which JSON
keys are required vs. optional at each nesting level (needed to correctly weight
object-separator/closing-brace tokens as "decision" vs. "scaffolding" -- see that
module's docstring) and to re-canonicalize key order before locating the JSON
substring inside a rendered chat completion.

Only `canonicalize()`/`validate()` and the node types/`SPECS` they run over are
ported; `instruction_gaps()`, `json_schema()`, `vocabularies()`, `observed_vocab()`
are not needed for loss weighting and are omitted. Re-sync this file if meddata's
target_schema.py changes the target shapes.
"""

from dataclasses import dataclass


# --- 2018 n2c2 track 2 (Drug-ADE).
MEDICATION_ATTR_TYPES = (
    "Strength",
    "Frequency",
    "Dosage",
    "Route",
    "Duration",
    "Form",
    "Reason",
    "ADE",
)

# --- 2022 n2c2 track 1 (CMED). Attribute *names* in official order; the
# value sets are per-name (see CMED_ATTR_VALUES).
CMED_ATTR_NAMES = ("Action", "Certainty", "Temporality", "Actor", "Negation")
CMED_ATTR_VALUES = {
    "Action": (
        "Start",
        "Stop",
        "Increase",
        "Decrease",
        "UniqueDose",
        "OtherChange",
        "Unknown",
    ),
    "Certainty": ("Certain", "Hypothetical", "Conditional", "Unknown"),
    "Temporality": ("Past", "Present", "Future", "Unknown"),
    "Actor": ("Patient", "Physician", "Unknown"),
    "Negation": ("Negated", "NotNegated"),
}
CMED_EVENT_TYPES = ("Disposition", "NoDisposition", "Undetermined")

# --- 2022 n2c2 track 2 (SHAC). Roles repeat with numeric suffixes when one
# event has several fillers for the same role (Amount/Amount2/Amount3).
SHAC_EVENT_TYPES = ("Tobacco", "Alcohol", "Drug", "Employment", "LivingStatus")
SHAC_ROLE_NAMES = (
    "Status",
    "Type",
    "Type2",
    "Type3",
    "Amount",
    "Amount2",
    "Amount3",
    "Duration",
    "Duration2",
    "Frequency",
    "Frequency2",
    "History",
    "History2",
    "Method",
    "Method2",
    "Method3",
)
SHAC_FILLER_TYPES = (
    "StatusTime",
    "StatusEmploy",
    "Type",
    "TypeLiving",
    "Amount",
    "Duration",
    "Frequency",
    "History",
    "Method",
)
SHAC_ATTR_VALUES = {
    "StatusTimeVal": ("none", "current", "past", "future"),
    "StatusEmployVal": (
        "employed",
        "unemployed",
        "retired",
        "on_disability",
        "student",
        "homemaker",
    ),
    "TypeLivingVal": ("alone", "with_family", "with_others", "homeless"),
}

# --- 2010 i2b2/VA
I2B2_CONCEPT_TYPES = ("problem", "test", "treatment")
I2B2_ASSERTIONS = (
    "present",
    "absent",
    "possible",
    "conditional",
    "hypothetical",
    "associated_with_someone_else",
)
I2B2_RELATION_TYPES = ("TrIP", "TrWP", "TrCP", "TrAP", "TrNAP", "TeRP", "TeCP", "PIP")

# --- 2019 n2c2 track 2 (family history).
FH_SIDES = ("Maternal", "Paternal", "NA", "None")
FH_NEGATION = ("Negated", "Non_Negated")
FH_LIVING_SCORES = ("0", "1", "2", "4")


# --- spec node types -------------------------------------------------------


@dataclass(frozen=True)
class Str:
    """Free text -- a span copied verbatim from the input, or a CUI."""

    nullable: bool = False


@dataclass(frozen=True)
class Num:
    """A JSON number (float/int), e.g. sts_regression similarity 0-5."""

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
    ..., TypeLivingVal -> alone/...).
    """

    keys: dict  # fixed keys shared by every branch (e.g. "role")
    tag: str  # discriminator key name, e.g. "name"
    value_key: str  # key whose vocabulary the tag selects, e.g. "value"
    variants: dict  # tag value -> tuple of legal value_key values
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
        "attributes": Arr(
            CoupledObj(
                keys={"role": Enum(("Status", "Type"))},
                tag="name",
                value_key="value",
                variants=SHAC_ATTR_VALUES,
                required=("role", "name", "value"),
            )
        ),
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
    "medication": Obj(
        keys={"type": Enum(MEDICATION_ATTR_TYPES + ("Drug",)), "text": Str()},
        required=("type", "text"),
    ),
    "i2b2": Obj(
        keys={"type": Enum(I2B2_CONCEPT_TYPES), "text": Str()},
        required=("type", "text"),
    ),
    "cmed": Obj(
        keys={"type": Enum(CMED_EVENT_TYPES), "text": Str()}, required=("type", "text")
    ),
}

SPECS = {
    # --- end-to-end IE tasks
    "ner_re_medication": _MED_NER_RE,
    "medication_event": Obj(keys={"events": Arr(_CMED_EVENT)}, required=("events",)),
    "sdoh_event": Obj(keys={"events": Arr(_SHAC_EVENT)}, required=("events",)),
    "concept_assertion_relation": _I2B2_FULL,
    "disorder_normalization": Obj(
        keys={
            "entities": Arr(
                Obj(keys={"text": Str(), "cui": Str()}, required=("text", "cui"))
            )
        },
        required=("entities",),
    ),
    "family_history": Obj(
        keys={
            "living_status": Arr(
                Obj(
                    keys={
                        "member": Str(),
                        "side": Enum(FH_SIDES),
                        "score": Enum(FH_LIVING_SCORES),
                    },
                    required=("member", "side", "score"),
                )
            ),
            "observations": Arr(
                Obj(
                    keys={
                        "member": Str(),
                        "side": Enum(FH_SIDES),
                        "text": Str(),
                        "negation": Enum(FH_NEGATION),
                    },
                    required=("member", "side", "text", "negation"),
                )
            ),
        },
        required=("living_status", "observations"),
    ),
    # --- Tier A subtask / window variants (same shapes as their base task)
    "ner_re_medication_ner_only": Obj(
        keys={"entities": Arr(_TYPED_SPAN_OF["medication"])}, required=("entities",)
    ),
    "ner_re_medication_re_only": _MED_NER_RE,
    "ner_re_medication_paragraph": _MED_NER_RE,
    "medication_event_trigger_only": Obj(
        keys={"triggers": Arr(_TYPED_SPAN_OF["cmed"])}, required=("triggers",)
    ),
    "medication_event_attr_only": Obj(
        keys={"events": Arr(_CMED_EVENT)}, required=("events",)
    ),
    "medication_event_paragraph": Obj(
        keys={"events": Arr(_CMED_EVENT)}, required=("events",)
    ),
    "concept_assertion_relation_ner_only": Obj(
        keys={"concepts": Arr(_TYPED_SPAN_OF["i2b2"])}, required=("concepts",)
    ),
    "concept_assertion_relation_re_only": _I2B2_FULL,
    "concept_assertion_relation_section": _I2B2_FULL,
    # --- classification / question_answering / similarity_regression tasks
    "cohort_eligibility": Obj(
        keys={"label": Enum(("not met", "met"))}, required=("label",)
    ),
    "nli": Obj(
        keys={"label": Enum(("entailment", "contradiction", "neutral"))},
        required=("label",),
    ),
    "sts_classification": Obj(
        keys={
            "category": Enum(
                (
                    "dissimilar",
                    "weakly_related",
                    "partially_equivalent",
                    "mostly_equivalent",
                    "highly_equivalent",
                )
            )
        },
        required=("category",),
    ),
    "similarity_classification": Obj(
        keys={
            "category": Enum(("very_weak", "weak", "moderate", "strong", "very_strong"))
        },
        required=("category",),
    ),
    "qa_yesnomaybe": Obj(
        keys={"decision": Enum(("yes", "no", "maybe")), "long_answer": Str()},
        required=("decision", "long_answer"),
    ),
    "mcqa": Obj(keys={"answer": Enum(tuple("ABCDEFGHIJ"))}, required=("answer",)),
    "mcqa_explain": Obj(
        keys={"answer": Enum(tuple("ABCD")), "explanation": Str()},
        required=("answer", "explanation"),
    ),
    "multilabel_cls": Obj(
        keys={
            "labels": Arr(
                Enum(
                    (
                        "Treatment",
                        "Diagnosis",
                        "Mechanism",
                        "Prevention",
                        "Transmission",
                        "Case Report",
                        "Epidemic Forecasting",
                    )
                )
            )
        },
        required=("labels",),
    ),
    "abbreviation_disambiguation": Obj(
        keys={"expansion": Str()}, required=("expansion",)
    ),
    "qa_span_extraction": Obj(keys={"answer": Str()}, required=("answer",)),
    "sts_regression": Obj(keys={"similarity": Num()}, required=("similarity",)),
}


# --- canonicalization ------------------------------------------------------


def canonicalize(task, target):
    """Return `target` with every object's keys in the spec's declared
    order. Unknown keys are preserved and appended after the known ones, so
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
    if isinstance(spec, Num):
        if value is None:
            if not spec.nullable:
                out.append(f"{path}: null not allowed")
        elif isinstance(value, bool):  # bool is an int subclass -- reject it
            out.append(f"{path}: expected number, got bool")
        elif not isinstance(value, (int, float)):
            out.append(f"{path}: expected number, got {type(value).__name__}")
        return
    if isinstance(spec, Str):
        if value is None:
            if not spec.nullable:
                out.append(f"{path}: null not allowed")
        elif not isinstance(value, str):
            out.append(f"{path}: expected string, got {type(value).__name__}")
