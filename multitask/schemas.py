"""
Task-tag registry for the merged multi-task dataset.

Every training example is rendered as one causal-LM text sequence:

    <<TASK:tag>>
    <fixed instruction for that tag>
    INPUT: <source text>
    OUTPUT: <JSON target, single line — or plain text for PLAIN_TEXT_TASKS>

The tag is what lets a single merged model pick the right output schema;
the instruction sentence is fixed per tag (not per example), so it adds no
information beyond the tag itself.
"""
import json

INSTRUCTIONS = {
    "nli": (
        'Determine the relationship between the premise and hypothesis: '
        'entailment, contradiction, or neutral. Return JSON {"label": ...}.'
    ),
    "mcqa": (
        'Answer the multiple-choice question by selecting the correct option '
        'letter. Return JSON {"answer": <letter>}.'
    ),
    "mcqa_explain": (
        'Answer the multiple-choice question and explain your reasoning. '
        'Return JSON {"answer": <letter>, "explanation": ...}.'
    ),
    "qa_yesnomaybe": (
        'Answer the research question yes, no, or maybe given the context, '
        'and give a long-form answer. Return JSON '
        '{"decision": ..., "long_answer": ...}.'
    ),
    "multilabel_cls": (
        'Assign all applicable topic labels to this article. '
        'Return JSON {"labels": [...]}.'
    ),
    "summarization": (
        'Write the discharge instructions and brief hospital course for '
        'this patient. Return plain text, not JSON.'
    ),
    "sts_regression": (
        'Rate the semantic similarity of the two clinical sentences on a '
        '0-5 scale (0 = unrelated, 5 = semantically equivalent), to two '
        'decimal places. Return JSON {"similarity": <float, e.g. 3.50>}.'
    ),
    "cohort_eligibility": (
        'Given the patient\'s longitudinal clinical notes and ONE named '
        'clinical trial eligibility criterion (with its definition), decide '
        'whether the patient meets that single criterion. '
        'Return JSON {"label": "met"} or {"label": "not met"}.'
    ),
    "ner_re_medication": (
        'Extract all Drug entities and their attribute entities (Strength, '
        'Frequency, Dosage, Route, Duration, Form, Reason, ADE), plus the '
        'relation linking each attribute entity to its drug. '
        'Return JSON {"entities": [...], "relations": [...]}.'
    ),
    "medication_event": (
        'For each medication mention, classify its disposition '
        '(Disposition, NoDisposition, or Undetermined); for Disposition '
        'mentions also give Action, Certainty, Temporality, Actor, and '
        'Negation. Return JSON {"events": [...]}.'
    ),
    "sdoh_event": (
        'Extract social-determinants-of-health events (e.g. Tobacco, '
        'Alcohol, Drug, Employment, LivingStatus) with their role-filler '
        'entities and attribute values. Return JSON {"events": [...]}.'
    ),
    "family_history": (
        'Extract family history from the note. Return JSON with four keys: '
        '"family_members" (list of {"member", "side"}), "observations_subtask1" '
        '(list of observation mention strings), "living_status" (list of '
        '{"member", "side", "score"}), and "observations" (list of {"member", '
        '"side", "text", "negation"}).'
    ),
    "disorder_normalization": (
        'Identify all disorder mentions in the clinical text and normalize '
        'each to a UMLS/SNOMED-CT CUI, or "CUI-less" if no matching concept '
        'exists. Return JSON {"entities": [{"text": ..., "cui": ...}, ...]}.'
    ),
    "abbreviation_disambiguation": (
        'Given a clinical abbreviation and its surrounding context, determine '
        'what the abbreviation stands for in this specific context. '
        'Return JSON {"expansion": ...}.'
    ),
    "concept_assertion_relation": (
        'Extract all problem/test/treatment concepts; for each problem '
        'concept also give its assertion status (present, absent, possible, '
        'conditional, hypothetical, or associated_with_someone_else); and '
        'extract the relations between concepts (TrIP, TrWP, TrCP, TrAP, '
        'TrNAP, TeRP, TeCP, PIP). Return JSON {"concepts": [{"id", "type", '
        '"text", "assertion"}, ...], "relations": [{"type", "arg1", "arg2"}, ...]}.'
    ),
}

# The 13 fixed cohort-eligibility criteria (2018 n2c2 Track 1), in the order
# they appear in the released XML <TAGS> blocks, with their definitions from
# the annotation guidelines. Each criterion is trained/evaluated as its own
# binary ("met" / "not met") example — one patient yields 13 examples, not
# one 13-key example — so the model sees a clean single-label signal per
# criterion instead of having to emit 13 keys at once.
COHORT_CRITERIA = [
    "ABDOMINAL", "ADVANCED-CAD", "ALCOHOL-ABUSE", "ASP-FOR-MI", "CREATININE",
    "DIETSUPP-2MOS", "DRUG-ABUSE", "ENGLISH", "HBA1C", "KETO-1YR",
    "MAJOR-DIABETES", "MAKES-DECISIONS", "MI-6MOS",
]

COHORT_CRITERIA_DEFINITIONS = {
    "ABDOMINAL": "History of intra-abdominal surgery, small/large intestine resection, or small bowel obstruction.",
    "ADVANCED-CAD": "Advanced cardiovascular disease (CAD), defined as at least 2 of: taking 2+ CAD medications, history of myocardial infarction, currently experiencing angina, or ischemia past/present.",
    "ALCOHOL-ABUSE": "Current alcohol use over the weekly recommended limits.",
    "ASP-FOR-MI": "Use of aspirin for prevention of myocardial infarction.",
    "CREATININE": "Serum creatinine level above the upper limit of normal.",
    "DIETSUPP-2MOS": "Taken a dietary supplement (excluding Vitamin D) within the past 2 months.",
    "DRUG-ABUSE": "History of drug abuse, current or past.",
    "ENGLISH": "Patient is able to speak English.",
    "HBA1C": "Any HbA1c value between 6.5% and 9.5%.",
    "KETO-1YR": "Diagnosis of ketoacidosis within the past year.",
    "MAJOR-DIABETES": "Major diabetes-related complication (e.g. amputation, kidney damage, skin conditions, retinopathy, nephropathy, neuropathy) resulting from or strongly correlated with uncontrolled diabetes.",
    "MAKES-DECISIONS": "Patient is able to make their own medical decisions.",
    "MI-6MOS": "Myocardial infarction within the past 6 months.",
}

# These criteria have very few positive/negative instances in the training
# data, which makes them particularly hard to learn — flagged explicitly in
# each example's prompt so the model (and anyone reading a training example)
# knows to expect a skewed, sparse signal for these specific criteria.
COHORT_CRITERIA_RARE = {"MI-6MOS", "KETO-1YR", "ALCOHOL-ABUSE", "DRUG-ABUSE"}


def cohort_criterion_prompt(criterion: str) -> str:
    """Per-example criterion block appended to the patient notes as input."""
    definition = COHORT_CRITERIA_DEFINITIONS[criterion]
    note = (
        " Note: this criterion has very few examples in the training data — "
        "expect a skewed, sparse signal."
        if criterion in COHORT_CRITERIA_RARE else ""
    )
    return f"Criterion: {criterion}\nDefinition: {definition}{note}"

# Tasks whose target is scored as free text (ROUGE/BERTScore-style), so
# JSON-wrapping the target would only hurt the official metric.
PLAIN_TEXT_TASKS = {"summarization"}


def render(task: str, input_text: str, target) -> str:
    """Render one training example as a single causal-LM text sequence."""
    if task not in INSTRUCTIONS:
        raise ValueError(f"unknown task tag: {task!r}")
    if task in PLAIN_TEXT_TASKS:
        target_str = target if isinstance(target, str) else str(target)
    else:
        target_str = json.dumps(target, ensure_ascii=False)
    return (
        f"<<TASK:{task}>>\n"
        f"{INSTRUCTIONS[task]}\n"
        f"INPUT: {input_text}\n"
        f"OUTPUT: {target_str}"
    )


def render_messages(task: str, input_text: str, target) -> list:
    """Render one training example as OpenAI-style chat messages, so
    tokenizer.apply_chat_template() can be applied directly. The task tag +
    fixed instruction go in the system turn, the source text is the user
    turn, and the (JSON, or plain text for PLAIN_TEXT_TASKS) target is the
    assistant turn."""
    if task not in INSTRUCTIONS:
        raise ValueError(f"unknown task tag: {task!r}")
    if task in PLAIN_TEXT_TASKS:
        target_str = target if isinstance(target, str) else str(target)
    else:
        target_str = json.dumps(target, ensure_ascii=False)
    return [
        {"role": "system", "content": f"<<TASK:{task}>>\n{INSTRUCTIONS[task]}"},
        {"role": "user", "content": input_text},
        {"role": "assistant", "content": target_str},
    ]


def example(task, dataset, split, locator, input_text, target):
    """Build one unified-dataset record. Carries both the flat causal-LM
    text sequence (used by build_unified_dataset.py/token_stats.py) and the
    OpenAI-style chat messages (used by build_hf_datasets.py)."""
    return {
        "task": task,
        "dataset": dataset,
        "split": split,
        "id": locator,
        "text": render(task, input_text, target),
        "messages": render_messages(task, input_text, target),
    }
