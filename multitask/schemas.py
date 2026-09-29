"""
Two-level task-tag registry for the merged multi-task dataset.

Every training example is rendered as one causal-LM text sequence:

    <<Category_Display: Task_Display>>
    <generic category-level framing, shared by every task in the category>
    <dataset-specific instruction for this task tag>
    INPUT: <source text>
    OUTPUT: <JSON target, single line — or plain text for PLAIN_TEXT_TASKS>

and, in chat-messages form, split across roles:

    system: <<Category_Display: Task_Display>>\\n<generic category framing>
    user:   <dataset-specific instruction>\\nINPUT: <source text>
    assistant: OUTPUT target

The tag's two levels serve different purposes: the category (`CATEGORIES`)
groups tasks by output-contract shape (extraction / classification / QA /
regression / summarization) and carries the generic, reusable system-level
framing that a decoder model learns once across every task in that family.
The task (`TASKS`) is the fine-grained tag that actually selects the output
schema, and carries the dataset-specific instruction (label sets, JSON key
names) that must sit next to the input, not buried only in a system prompt.

The task tag string (e.g. "sdoh_event") is also the JSON "task" field used
by eval/run_eval.py for scorer dispatch and by convert_back/* for schema
selection — that routing is untouched by this two-level tag; the tag text
is a model-facing/analysis signal only, never parsed out of rendered text
by any code in this repo.
"""
import json

from multitask.target_schema import canonicalize

CATEGORIES = {
    "information_extraction": {
        "display": "Information_Extraction",
        "system": (
            "You are performing information extraction from clinical or "
            "biomedical text: identify the entities, events, relations, or "
            "attributes described in the task-specific instructions that "
            "follow, and return them as the JSON object those instructions "
            "specify."
        ),
    },
    "classification": {
        "display": "Classification",
        "system": (
            "You are performing text classification: read the input and "
            "assign it the label(s) described in the task-specific "
            "instructions that follow, and return them as the JSON object "
            "those instructions specify."
        ),
    },
    "question_answering": {
        "display": "Question_Answering",
        "system": (
            "You are performing question answering: read the question and "
            "any supporting context or options, and answer it as described "
            "in the task-specific instructions that follow, returning the "
            "JSON object those instructions specify."
        ),
    },
    "similarity_regression": {
        "display": "Similarity_Regression",
        "system": (
            "You are performing similarity or regression scoring: read the "
            "input and produce the numeric score described in the "
            "task-specific instructions that follow, returning the JSON "
            "object those instructions specify."
        ),
    },
    "summarization": {
        "display": "Summarization",
        "system": (
            "You are performing summarization: read the input and produce "
            "the summary described in the task-specific instructions that "
            "follow."
        ),
    },
}

# Task-level registry. Each entry's "instruction" text is unchanged from the
# old flat INSTRUCTIONS dict (dataset-specific label sets / JSON schema) —
# only "category" and "display" are new, except for the four IE tasks
# (ner_re_medication, medication_event, sdoh_event, family_history) whose
# instructions were rewritten into Task/Definitions/OutputFormat sections
# with paper/guideline-grounded entity/attribute/event definitions (see
# design.md). "category" placements that are not clean-cut are noted
# inline; see design.md's Task Taxonomy > Ambiguous placements section for
# the full rationale.
TASKS = {
    "nli": {
        "category": "classification",
        "display": "NLI",
        "instruction": (
            "### Task\n"
            "Determine the relationship between the premise and "
            "hypothesis.\n\n"
            "### Labels\n"
            "- **entailment**\n"
            "- **contradiction**\n"
            "- **neutral**\n\n"
            "### Output Format\n"
            'Return JSON `{"label": ...}`.'
        ),
    },
    "mcqa": {
        "category": "question_answering",
        "display": "MCQA",
        "instruction": (
            "### Task\n"
            "Answer the multiple-choice question by selecting the correct "
            "option letter.\n\n"
            "### Output Format\n"
            'Return JSON `{"answer": <letter>}`.'
        ),
    },
    "mcqa_explain": {
        "category": "question_answering",
        "display": "MCQA_Explain",
        "instruction": (
            "### Task\n"
            "Answer the multiple-choice question and explain your "
            "reasoning.\n\n"
            "### Output Format\n"
            'Return JSON `{"answer": <letter>, "explanation": ...}`.'
        ),
    },
    # Hybrid: the "decision" field is a closed yes/no/maybe label
    # (classification-shaped), but the task is framed as question + context
    # -> answer, like mcqa/mcqa_explain, and is PubMedQA's native
    # leaderboard framing -- categorized as question_answering. See
    # design.md's Ambiguous placements section.
    "qa_yesnomaybe": {
        "category": "question_answering",
        "display": "QA_YesNoMaybe",
        "instruction": (
            "### Task\n"
            "Answer the research question yes, no, or maybe given the "
            "context, and give a long-form answer.\n\n"
            "### Output Format\n"
            'Return JSON `{"decision": ..., "long_answer": ...}`.'
        ),
    },
    "multilabel_cls": {
        "category": "classification",
        "display": "Multilabel_Topic",
        "instruction": (
            "### Task\n"
            "Assign all applicable topic labels to this article.\n\n"
            "### Output Format\n"
            'Return JSON `{"labels": [...]}`.'
        ),
    },
    "summarization": {
        "category": "summarization",
        "display": "Summarization",
        "instruction": (
            "### Task\n"
            "Write the discharge instructions and brief hospital course "
            "for this patient.\n\n"
            "### Output Format\n"
            "Return plain text, not JSON, in exactly this format (the "
            "scorer splits on these literal headers): "
            '"DISCHARGE INSTRUCTIONS:\\n<text>\\n\\nBRIEF HOSPITAL COURSE:'
            '\\n<text>".'
        ),
    },
    "sts_regression": {
        "category": "similarity_regression",
        "display": "STS_Regression",
        "instruction": (
            "### Task\n"
            "Rate the semantic similarity of the two clinical sentences "
            "on a 0-5 scale (0 = unrelated, 5 = semantically equivalent), "
            "to two decimal places.\n\n"
            "### Output Format\n"
            'Return JSON `{"similarity": <float, e.g. 3.50>}`.'
        ),
    },
    "sts_classification": {
        "category": "classification",
        "display": "STS_Classification",
        "instruction": (
            "### Task\n"
            "Classify the semantic similarity of the two clinical "
            "sentences into one of 5 categories.\n\n"
            "### Categories\n"
            "- **dissimilar**: unrelated, or share no clinically "
            "meaningful information\n"
            "- **weakly_related**: same general topic but very different "
            "clinical content\n"
            "- **partially_equivalent**: share important clinical "
            "information but differ in other important details\n"
            "- **mostly_equivalent**: same essential clinical meaning, "
            "only minor/unimportant details differ\n"
            "- **highly_equivalent**: semantically equivalent or nearly "
            "so\n\n"
            "### Output Format\n"
            'Return JSON `{"category": <one of the 5 values above, as a '
            'string>}`.'
        ),
    },
    "similarity_classification": {
        "category": "classification",
        "display": "Similarity_Classification",
        "instruction": (
            "### Task\n"
            "Classify the semantic similarity of the two "
            "biomedical-literature sentences into one of 5 "
            "categories.\n\n"
            "### Categories\n"
            "- **very_weak**: unrelated, or share no meaningful "
            "information\n"
            "- **weak**: same general topic but very different content\n"
            "- **moderate**: share some information but differ in other "
            "important details\n"
            "- **strong**: mostly the same meaning, only minor details "
            "differ\n"
            "- **very_strong**: semantically equivalent or nearly so\n\n"
            "### Output Format\n"
            'Return JSON `{"category": <one of the 5 values above, as a '
            'string>}`.'
        ),
    },
    "cohort_eligibility": {
        "category": "classification",
        "display": "Cohort_Eligibility",
        "instruction": (
            "### Task\n"
            "Given the patient's longitudinal clinical notes and ONE "
            "named clinical trial eligibility criterion (with its "
            "definition), decide whether the patient meets that single "
            "criterion.\n\n"
            "### Output Format\n"
            'Return JSON `{"label": "met"}` or `{"label": "not met"}`.'
        ),
    },
    "ner_re_medication": {
        "category": "information_extraction",
        "display": "NER_RE",
        "instruction": (
            "### Task\n"
            "Extract every Drug mention in this note and its linked "
            "clinical attributes, per the 2018 n2c2 Track 2 ADE/"
            "medication-extraction schema (Henry et al. 2020, "
            "JAMIA).\n\n"
            "### Definitions\n"
            "- **Drug**: a medication name (generic, brand, or drug "
            "class)\n"
            '- **Strength**: amount of active ingredient per dose unit '
            '(e.g. "40 mg")\n'
            "- **Form**: the drug's physical form (e.g. tablet, patch, "
            "cream)\n"
            '- **Dosage**: amount given per administration (e.g. "2 '
            'tablets")\n'
            '- **Frequency**: how often it\'s taken (e.g. "twice '
            'daily")\n'
            '- **Route**: how it enters the body (e.g. "PO", "IV", '
            '"topically")\n'
            "- **Duration**: how long it is/was taken\n"
            "- **Reason**: the medical indication for the drug\n"
            "- **ADE**: an adverse reaction attributed to the drug\n\n"
            "Each attribute links to the Drug mention(s) it describes -- "
            "any attribute type may link to more than one Drug, most "
            "often Reason/ADE (e.g. one shared side effect attributed to "
            "two co-administered drugs).\n\n"
            "### Output Format\n"
            "Nest each attribute mention under its linked Drug, keyed by "
            "attribute type -- omit attribute types with no mentions; "
            'list any attribute mention not linked to any drug under '
            '"unlinked". Return JSON `{"drugs": [{"text": ..., '
            '"<AttrType>": [...], ...}, ...], "unlinked": [{"type", '
            '"text"}, ...]}`.'
        ),
    },
    "medication_event": {
        "category": "information_extraction",
        "display": "Medication_Event",
        "instruction": (
            "### Task\n"
            "For each medication mention, classify whether a medication "
            "change is discussed, per the CMED schema (Mahajan et al. "
            "2021).\n\n"
            "### Definitions\n"
            '- **Disposition**: a change is discussed (e.g. "start '
            'Plavix")\n'
            '- **NoDisposition**: no change is discussed (e.g. '
            '"continue lisinopril")\n'
            '- **Undetermined**: unclear from local context (e.g. '
            '"Plan: Lasix")\n\n'
            "For each Disposition mention, also classify:\n"
            "- **Action**: Start/Stop/Increase/Decrease/UniqueDose/"
            "OtherChange/Unknown\n"
            "- **Certainty**: Certain/Hypothetical/Conditional/Unknown\n"
            "- **Temporality**: Past/Present/Future/Unknown\n"
            "- **Actor**: Physician/Patient/Unknown\n"
            "- **Negation**: Negated/NotNegated\n\n"
            "A single mention may have 0, 1, or more Disposition "
            "events.\n\n"
            "### Output Format\n"
            '"trigger_text" is the medication mention itself (the '
            "entity type in this task is always a medication -- "
            '"trigger" names its role as what triggers the event, per '
            "this schema's event-extraction terminology, not a "
            'different kind of span). Return JSON `{"events": [{"id", '
            '"type", "trigger_text", "attributes": {"Action", '
            '"Certainty", "Temporality", "Actor", "Negation"}}, ...]}` '
            '-- "attributes" is empty for NoDisposition/Undetermined '
            "events."
        ),
    },
    "sdoh_event": {
        "category": "information_extraction",
        "display": "SDoH",
        "instruction": (
            "### Task\n"
            "Extract social-determinants-of-health (SDOH) events from "
            "this note, per the SHAC annotation schema (Lybarger et "
            "al., 2022 n2c2 Track 2). An SDOH event is a trigger span "
            "(Employment, LivingStatus, Tobacco, Alcohol, or Drug) plus "
            "role-labeled argument spans.\n\n"
            "### Roles By Trigger Type\n"
            "- **Employment**: Status, Type (occupation), Duration, "
            "History\n"
            "- **LivingStatus**: Status, Type (alone/with_family/"
            "with_others/homeless), Duration, History\n"
            '- **Tobacco/Alcohol/Drug**: Status, Type (the specific '
            'substance, e.g. "beer"/"wine" for Alcohol, "cocaine"/'
            '"marijuana" for Drug, "cigarettes" for Tobacco), Amount, '
            'Frequency, Method (how it\'s consumed, e.g. "smoke"/'
            '"inject" -- mostly Drug-only; almost never annotated for '
            "Alcohol and rare for Tobacco, since smoking is the assumed "
            "default for both), Duration, History -- Drug covers "
            "illicit drugs/marijuana/prescription misuse only (not "
            "medical use), Tobacco covers first-hand smoking only\n\n"
            "### Status Values\n"
            "Status is one of none/current/past/future for every "
            'trigger type (some values are rare for a given type -- '
            'e.g. "future" is rare for Tobacco/Alcohol/Drug, "none" is '
            "rare for LivingStatus -- but none is impossible for any of "
            "them).\n\n"
            "### Role Filler Format\n"
            'Each role filler is `{"type", "text"}`: type is usually '
            "the role's own name, but a Status filler's type is "
            "StatusTime (Tobacco/Alcohol/Drug/LivingStatus) or "
            "StatusEmploy (Employment), and LivingStatus's Type "
            "filler's type is TypeLiving. If a role has more than one "
            "filler, number the extras: Type2/Type3, Amount2/Amount3, "
            "Duration2, Frequency2, History2, Method2/Method3. Status/"
            "Type fillers carry their own attribute value keyed by name "
            "-- StatusTimeVal (none/current/past/future), "
            "StatusEmployVal (employed/unemployed/retired/on_disability/"
            "student/homemaker), or TypeLivingVal (alone/with_family/"
            "with_others/homeless).\n\n"
            "### Output Format\n"
            'Return JSON `{"events": [{"id", "type", "trigger_text", '
            '"roles": {"<RoleName>": {"type", "text"}, ...}, '
            '"attributes": [{"role", "name", "value"}, ...]}, ...]}`.'
        ),
    },
    # Additive sibling to sdoh_event: same underlying SHAC events, same
    # closed vocabularies, but a compact non-JSON structured-sequence
    # target instead of JSON, matching the Microsoft-Nuance T5 approach's
    # #1-ranked 2022 n2c2 Track 2 submission (see
    # multitask/seq2seq_grammar.py). Never touches sdoh_event's own task
    # entry, builder, or convert_back module.
    "sdoh_event_seq2seq": {
        "category": "information_extraction",
        "display": "SDoH_Seq2Seq",
        "instruction": (
            "### Task\n"
            "Extract social-determinants-of-health (SDOH) events from "
            "this note, per the SHAC annotation schema (Lybarger et "
            "al., 2022 n2c2 Track 2). An SDOH event is a trigger span "
            "(Employment, LivingStatus, Tobacco, Alcohol, or Drug) plus "
            "role-labeled argument spans.\n\n"
            "### Roles By Trigger Type\n"
            "- **Employment**: Status, Type (occupation), Duration, "
            "History\n"
            "- **LivingStatus**: Status, Type (alone/with_family/"
            "with_others/homeless), Duration, History\n"
            '- **Tobacco/Alcohol/Drug**: Status, Type (the specific '
            'substance, e.g. "beer"/"wine" for Alcohol, "cocaine"/'
            '"marijuana" for Drug, "cigarettes" for Tobacco), Amount, '
            'Frequency, Method (how it\'s consumed, e.g. "smoke"/'
            '"inject" -- mostly Drug-only; almost never annotated for '
            "Alcohol and rare for Tobacco, since smoking is the assumed "
            "default for both), Duration, History -- Drug covers "
            "illicit drugs/marijuana/prescription misuse only (not "
            "medical use), Tobacco covers first-hand smoking only\n\n"
            "### Status Values\n"
            "Status is one of none/current/past/future for every "
            'trigger type (some values are rare for a given type -- '
            'e.g. "future" is rare for Tobacco/Alcohol/Drug, "none" is '
            "rare for LivingStatus -- but none is impossible for any of "
            "them).\n\n"
            "### Output Format\n"
            "Return plain text, not JSON, in this format (no example "
            "here is exhaustive -- copy trigger/filler text verbatim "
            "from the note; if there are no SDOH events, return an "
            "empty string):\n"
            "`<EventType> trigger_text <RoleName>[!FillerType] "
            "[attr_value] filler_text ...`, one such clause per role, "
            "roles in the fixed order Status, Type, Type2, Type3, "
            "Amount, Amount2, Amount3, Duration, Duration2, Frequency, "
            "Frequency2, History, History2, Method, Method2, Method3 "
            "(numbered suffixes are extra fillers for the same role); "
            'multiple events are joined with " {AND} ". "[attr_value]" '
            "only ever follows a Status or Type role, and is one of "
            "StatusTimeVal's none/current/past/future, StatusEmployVal's "
            "employed/unemployed/retired/on_disability/student/"
            "homemaker, or TypeLivingVal's alone/with_family/"
            "with_others/homeless -- whichever the role's own filler "
            "type implies. Use the literal token \"=TRIGGER=\" in place "
            "of filler_text when a role's filler is the exact same span "
            "as its own event's trigger, instead of repeating the text. "
            "\"!FillerType\" is only needed when a role's filler type "
            "isn't the default for that role (Status defaults to "
            "StatusTime, or StatusEmploy for Employment; Type defaults "
            "to Type, or TypeLiving for LivingStatus; every other role "
            "defaults to its own base name) -- omit it otherwise.\n"
            "Example: `<LivingStatus> lives <Status> [current] "
            "=TRIGGER= <Type> [with_family] with husband {AND} "
            "<Tobacco> smokes <Status> [current] =TRIGGER= <Type> "
            "cigarettes <Type2> cigars`"
        ),
    },
    "family_history": {
        "category": "information_extraction",
        "display": "Family_History",
        "instruction": (
            "### Task\n"
            "Extract family history from the note as an end-to-end "
            "summary of member/side/living-status/observation, per the "
            "2019 n2c2 Track 2 Subtask 2 schema (Liu et al. 2018, "
            "BioCreative/OHNLP).\n\n"
            "### Definitions\n"
            "- **FamilyMember**: one of Father/Mother/Parent/Sister/"
            "Brother/Daughter/Son/Child/Grandmother/Grandfather/"
            "Grandparent/Cousin/Sibling/Aunt/Uncle, each with a side of "
            "family (Paternal/Maternal/NA -- first-degree relatives are "
            "always NA)\n"
            "- **Observation**: a disease/condition mentioned for a "
            "family member\n"
            "- **LivingStatus score** = alive_score * healthy_score, "
            "each in {Yes=2, NA=1, No=0} (so 4 = alive and healthy, 0 = "
            "deceased)\n\n"
            "Only report a member here if the note gives a "
            "living-status score and/or at least one observation for "
            "them -- a member mentioned with neither is not part of "
            "this summary.\n\n"
            "### Output Format\n"
            'Return JSON with 2 keys: "living_status" (list of '
            '{"member", "side", "score"}), "observations" (list of '
            '{"member", "side", "text", "negation": "Negated"|'
            '"Non_Negated"}).'
        ),
    },
    "disorder_normalization": {
        "category": "information_extraction",
        "display": "Disorder_Normalization",
        "instruction": (
            "### Task\n"
            "Identify all disorder mentions in the clinical text and "
            'normalize each to a UMLS/SNOMED-CT CUI, or "CUI-less" if no '
            "matching concept exists.\n\n"
            "### Output Format\n"
            'Return JSON `{"entities": [{"text": ..., "cui": ...}, '
            '...]}`.'
        ),
    },
    # Word-sense disambiguation: the label set (candidate expansions) is not
    # enumerated in-prompt, but the task is fundamentally "pick the correct
    # sense given context" -- categorized as classification. See design.md's
    # Ambiguous placements section.
    "abbreviation_disambiguation": {
        "category": "classification",
        "display": "Abbreviation_Disambiguation",
        "instruction": (
            "### Task\n"
            "Given a clinical abbreviation and its surrounding context, "
            "determine what the abbreviation stands for in this "
            "specific context.\n\n"
            "### Output Format\n"
            'Return JSON `{"expansion": ...}`.'
        ),
    },
    "qa_span_extraction": {
        "category": "question_answering",
        "display": "QA_Span_Extraction",
        "instruction": (
            "### Task\n"
            "Given the clinical note context and a question, extract "
            "the exact answer as a verbatim substring of the "
            "context.\n\n"
            "### Output Format\n"
            'Return JSON `{"answer": <exact substring of the '
            'context>}`.'
        ),
    },
    "concept_assertion_relation": {
        "category": "information_extraction",
        "display": "Concept_Assertion_Relation",
        "instruction": (
            "### Task\n"
            "Extract every problem, test, and treatment concept. For "
            "each problem concept, also give its assertion status. "
            "Link related concepts with one of 8 relation types (same "
            "sentence only).\n\n"
            "### Concept Types\n"
            "- **problem**: an abnormal finding about the patient's "
            "body or mind that could be treated (a disease, sign/"
            "symptom, or injury; excludes normal findings and raw "
            "vital-sign/lab values)\n"
            "- **test**: a procedure/panel/exam performed to find out "
            "about a problem\n"
            "- **treatment**: a procedure, device, or substance given "
            "to resolve a problem\n\n"
            "### Assertion Status (problem concepts only)\n"
            "- **present** (default): stated to exist, now or in the "
            "past\n"
            "- **absent**: stated not to exist, ruled out, or "
            "resolved\n"
            '- **possible**: uncertain, e.g. "likely"/"probably not"/'
            '"suspected"\n'
            "- **conditional**: occurs only under specific "
            "circumstances, e.g. exertional symptoms or a drug "
            "allergy\n"
            "- **hypothetical**: a problem the patient may develop in "
            "the future\n"
            "- **associated_with_someone_else**: concerns someone other "
            "than the patient, e.g. family history\n\n"
            "### Relations (same sentence only)\n"
            "- **TrIP**: treatment improves/cures problem\n"
            "- **TrWP**: treatment given for problem but doesn't help "
            "or worsens it\n"
            "- **TrCP**: treatment causes a different problem than the "
            "one it was given for\n"
            "- **TrAP**: treatment given for problem, outcome not "
            "stated\n"
            "- **TrNAP**: treatment withheld/stopped because of a "
            "problem it didn't cause\n"
            "- **TeRP**: test performed and reveals the problem\n"
            "- **TeCP**: test performed to investigate the problem, "
            "outcome not known\n"
            "- **PIP**: one problem reveals, describes, or causes "
            "another\n\n"
            "### Output Format\n"
            'Return JSON `{"concepts": [{"id", "type", "text", '
            '"assertion"}, ...], "relations": [{"type", "arg1", '
            '"arg2"}, ...]}`.'
        ),
    },
    # --- Tier A subtask-enrichment tasks (see design.md's "Fixing doc-level
    # training-data scarcity" section) -- train-only, additive re-framings of
    # ner_re_medication/medication_event's own gold annotations into an
    # easier pipeline-stage sub-task. "Subtask" is baked into the display
    # string so the rendered tag is unmistakably distinct from the parent
    # end-to-end task's tag. See SUBTASK_OF below for the parent mapping.
    # These never appear in multitask/data/eval/ or train_unified.jsonl --
    # only in their own multitask/huggingface_format/<name>/ directory.
    "ner_re_medication_ner_only": {
        "category": "information_extraction",
        "display": "NER_RE_Subtask_NERonly",
        "instruction": (
            "### Task\n"
            "Extract every Drug, Strength, Frequency, Dosage, Route, "
            "Duration, Form, Reason, and ADE mention span in the text -- "
            "entities only, no relation linking.\n\n"
            "### Output Format\n"
            'Return JSON `{"entities": [{"type": ..., "text": ...}, '
            '...]}`.'
        ),
    },
    "ner_re_medication_re_only": {
        "category": "information_extraction",
        "display": "NER_RE_Subtask_REonly",
        "instruction": (
            "### Task\n"
            "This note's Drug/Strength/Frequency/Dosage/Route/Duration/"
            "Form/Reason/ADE entities have already been extracted and "
            "are listed after the note text under EXTRACTED_ENTITIES. "
            "Link each non-Drug entity to the Drug mention it describes "
            "(an entity may link to more than one Drug); collect any "
            'entity never linked to a Drug under "unlinked".\n\n'
            "### Output Format\n"
            'Return JSON `{"drugs": [{"text": ..., "<AttrType>": [...], '
            '...}, ...], "unlinked": [{"type", "text"}, ...]}`.'
        ),
    },
    "medication_event_trigger_only": {
        "category": "information_extraction",
        "display": "Medication_Event_Subtask_TriggerOnly",
        "instruction": (
            "### Task\n"
            "Extract every medication mention span and classify its "
            "disposition type (Disposition, NoDisposition, or "
            "Undetermined) -- triggers only, no attribute "
            "classification.\n\n"
            "### Output Format\n"
            'Return JSON `{"triggers": [{"type": ..., "text": ...}, '
            '...]}`.'
        ),
    },
    "medication_event_attr_only": {
        "category": "information_extraction",
        "display": "Medication_Event_Subtask_AttrOnly",
        "instruction": (
            "### Task\n"
            "This note's medication triggers have already been "
            "identified and are listed after the note text under "
            "EXTRACTED_TRIGGERS, each with its disposition type. For "
            "each Disposition trigger, classify:\n\n"
            "### Attributes\n"
            "- **Action**: Start/Stop/Increase/Decrease/UniqueDose/"
            "OtherChange/Unknown\n"
            "- **Certainty**: Certain/Hypothetical/Conditional/Unknown\n"
            "- **Temporality**: Past/Present/Future/Unknown\n"
            "- **Actor**: Physician/Patient/Unknown\n"
            "- **Negation**: Negated/NotNegated\n\n"
            "### Output Format\n"
            'Return JSON `{"events": [...]}`.'
        ),
    },
    # --- Paragraph-window task (see design.md's paragraph-chunking
    # section) -- a distinct mechanism from the Tier A subtasks above: this
    # re-frames ner_re_medication's own gold annotations onto a genuinely
    # smaller, distinct text span (one paragraph) rather than relabeling
    # the same whole-note text. HF-only, additive; never appears in
    # multitask/data/eval/ or train_unified.jsonl -- see PARAGRAPH_OF below.
    "ner_re_medication_paragraph": {
        "category": "information_extraction",
        "display": "NER_RE_Paragraph_Window",
        "instruction": (
            "### Task\n"
            "This is one paragraph excerpted from a larger clinical "
            "note, not the complete note. Extract every Drug, Strength, "
            "Frequency, Dosage, Route, Duration, Form, Reason, and ADE "
            "mention in THIS EXCERPT ONLY and link each non-Drug entity "
            "to the Drug mention it describes.\n\n"
            "### Output Format\n"
            'Return JSON `{"drugs": [{"text": ..., "<AttrType>": [...], '
            '...}, ...], "unlinked": [{"type", "text"}, ...]}`.'
        ),
    },
    # --- Paragraph-window task for medication_event (2022n2c2_track1) --
    # same mechanism as ner_re_medication_paragraph above (a genuinely new,
    # smaller text span, not a relabeled whole note), HF-only, additive; see
    # PARAGRAPH_OF below. Keeps the CMED disposition definitions (the model
    # needs them to classify), scoped to "THIS EXCERPT ONLY".
    "medication_event_paragraph": {
        "category": "information_extraction",
        "display": "Medication_Event_Paragraph_Window",
        "instruction": (
            "### Task\n"
            "This is one paragraph excerpted from a larger clinical "
            "note, not the complete note. For each medication mention "
            "in THIS EXCERPT ONLY, classify whether a medication change "
            "is discussed, per the CMED schema (Mahajan et al. "
            "2021).\n\n"
            "### Definitions\n"
            '- **Disposition**: a change is discussed (e.g. "start '
            'Plavix")\n'
            '- **NoDisposition**: no change is discussed (e.g. '
            '"continue lisinopril")\n'
            '- **Undetermined**: unclear from local context (e.g. '
            '"Plan: Lasix")\n\n'
            "For each Disposition mention, also classify:\n"
            "- **Action**: Start/Stop/Increase/Decrease/UniqueDose/"
            "OtherChange/Unknown\n"
            "- **Certainty**: Certain/Hypothetical/Conditional/Unknown\n"
            "- **Temporality**: Past/Present/Future/Unknown\n"
            "- **Actor**: Physician/Patient/Unknown\n"
            "- **Negation**: Negated/NotNegated\n\n"
            "A single mention may have 0, 1, or more Disposition "
            "events.\n\n"
            "### Output Format\n"
            '"trigger_text" is the medication mention itself (the '
            "entity type in this task is always a medication -- "
            '"trigger" names its role as what triggers the event, per '
            "this schema's event-extraction terminology, not a "
            'different kind of span). Return JSON `{"events": [{"id", '
            '"type", "trigger_text", "attributes": {"Action", '
            '"Certainty", "Temporality", "Actor", "Negation"}}, ...]}` '
            '-- "attributes" is empty for NoDisposition/Undetermined '
            "events."
        ),
    },
    # --- Tier A subtask-enrichment tasks for concept_assertion_relation
    # (2010_i2b2_va) -- same train-only/HF-only convention as the
    # ner_re_medication/medication_event subtasks above. See SUBTASK_OF below.
    "concept_assertion_relation_ner_only": {
        "category": "information_extraction",
        "display": "Concept_Assertion_Relation_Subtask_NERonly",
        "instruction": (
            "### Task\n"
            "Extract every problem, test, and treatment concept mention "
            "span in the text -- concepts only, no assertion "
            "classification, no relation linking.\n\n"
            "### Concept Types\n"
            "- **problem**: an abnormal finding about the patient's "
            "body or mind that could be treated (a disease, sign/"
            "symptom, or injury; excludes normal findings and raw "
            "vital-sign/lab values)\n"
            "- **test**: a procedure/panel/exam performed to find out "
            "about a problem\n"
            "- **treatment**: a procedure, device, or substance given "
            "to resolve a problem\n\n"
            "### Output Format\n"
            'Return JSON `{"concepts": [{"type": ..., "text": ...}, '
            '...]}`.'
        ),
    },
    "concept_assertion_relation_re_only": {
        "category": "information_extraction",
        "display": "Concept_Assertion_Relation_Subtask_REonly",
        "instruction": (
            "### Task\n"
            "This note's problem/test/treatment concepts have already "
            "been extracted and are listed after the note text under "
            "EXTRACTED_CONCEPTS. For each problem concept, give its "
            "assertion status. Link related concepts with one of 8 "
            "relation types (same sentence only).\n\n"
            "### Assertion Status\n"
            "- **present** (default): stated to exist, now or in the "
            "past\n"
            "- **absent**: stated not to exist, ruled out, or "
            "resolved\n"
            '- **possible**: uncertain, e.g. "likely"/"probably not"/'
            '"suspected"\n'
            "- **conditional**: occurs only under specific "
            "circumstances, e.g. exertional symptoms or a drug "
            "allergy\n"
            "- **hypothetical**: a problem the patient may develop in "
            "the future\n"
            "- **associated_with_someone_else**: concerns someone other "
            "than the patient, e.g. family history\n\n"
            "### Relations (same sentence only)\n"
            "- **TrIP**: treatment improves/cures problem\n"
            "- **TrWP**: treatment given for problem but doesn't help "
            "or worsens it\n"
            "- **TrCP**: treatment causes a different problem than the "
            "one it was given for\n"
            "- **TrAP**: treatment given for problem, outcome not "
            "stated\n"
            "- **TrNAP**: treatment withheld/stopped because of a "
            "problem it didn't cause\n"
            "- **TeRP**: test performed and reveals the problem\n"
            "- **TeCP**: test performed to investigate the problem, "
            "outcome not known\n"
            "- **PIP**: one problem reveals, describes, or causes "
            "another\n\n"
            "### Output Format\n"
            'Return JSON `{"concepts": [{"id", "type", "text", '
            '"assertion"}, ...], "relations": [{"type", "arg1", '
            '"arg2"}, ...]}`.'
        ),
    },
    # --- Section-window task for concept_assertion_relation (2010_i2b2_va)
    # -- same mechanism as ner_re_medication_paragraph above (new text
    # spans, not relabeled ones), but boundaries are ALL-CAPS section
    # headers instead of blank lines -- this format has none (see
    # chunk_utils.section_spans). HF-only, additive; see PARAGRAPH_OF below.
    "concept_assertion_relation_section": {
        "category": "information_extraction",
        "display": "Concept_Assertion_Relation_Section_Window",
        "instruction": (
            "### Task\n"
            "This is one section excerpted from a larger clinical note "
            '(e.g. "HISTORY OF PRESENT ILLNESS", "PAST MEDICAL '
            'HISTORY"), not the complete note. Extract every problem, '
            "test, and treatment concept in THIS EXCERPT ONLY. For each "
            "problem concept, give its assertion status. Link related "
            "concepts with one of 8 relation types (same sentence "
            "only).\n\n"
            "### Concept Types\n"
            "- **problem**: an abnormal finding about the patient's "
            "body or mind that could be treated (a disease, sign/"
            "symptom, or injury; excludes normal findings and raw "
            "vital-sign/lab values)\n"
            "- **test**: a procedure/panel/exam performed to find out "
            "about a problem\n"
            "- **treatment**: a procedure, device, or substance given "
            "to resolve a problem\n\n"
            "### Assertion Status\n"
            "- **present** (default): stated to exist, now or in the "
            "past\n"
            "- **absent**: stated not to exist, ruled out, or "
            "resolved\n"
            '- **possible**: uncertain, e.g. "likely"/"probably not"/'
            '"suspected"\n'
            "- **conditional**: occurs only under specific "
            "circumstances, e.g. exertional symptoms or a drug "
            "allergy\n"
            "- **hypothetical**: a problem the patient may develop in "
            "the future\n"
            "- **associated_with_someone_else**: concerns someone other "
            "than the patient, e.g. family history\n\n"
            "### Relations (same sentence only)\n"
            "- **TrIP**: treatment improves/cures problem\n"
            "- **TrWP**: treatment given for problem but doesn't help "
            "or worsens it\n"
            "- **TrCP**: treatment causes a different problem than the "
            "one it was given for\n"
            "- **TrAP**: treatment given for problem, outcome not "
            "stated\n"
            "- **TrNAP**: treatment withheld/stopped because of a "
            "problem it didn't cause\n"
            "- **TeRP**: test performed and reveals the problem\n"
            "- **TeCP**: test performed to investigate the problem, "
            "outcome not known\n"
            "- **PIP**: one problem reveals, describes, or causes "
            "another\n\n"
            "### Output Format\n"
            'Return JSON `{"concepts": [{"id", "type", "text", '
            '"assertion"}, ...], "relations": [{"type", "arg1", '
            '"arg2"}, ...]}`.'
        ),
    },
}

# Tier A subtask task-tag -> parent end-to-end task-tag. Single source of
# truth for "is this a Tier A subtask" and "which real task does it
# enrich" -- used by hf_dataset_stats.py's README generation instead of
# pattern-matching on dataset-name suffixes.
SUBTASK_OF = {
    "ner_re_medication_ner_only": "ner_re_medication",
    "ner_re_medication_re_only": "ner_re_medication",
    "medication_event_trigger_only": "medication_event",
    "medication_event_attr_only": "medication_event",
    "concept_assertion_relation_ner_only": "concept_assertion_relation",
    "concept_assertion_relation_re_only": "concept_assertion_relation",
}

# Paragraph-window task-tag -> parent end-to-end task-tag. Kept separate
# from SUBTASK_OF on purpose -- a distinct mechanism (new text spans, not
# new labels on old text) -- so hf_dataset_stats.py can report it under
# its own heading rather than conflating the two.
PARAGRAPH_OF = {
    "ner_re_medication_paragraph": "ner_re_medication",
    "concept_assertion_relation_section": "concept_assertion_relation",
    "medication_event_paragraph": "medication_event",
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
# JSON-wrapping the target would only hurt the official metric. Also home
# to sdoh_event_seq2seq's non-JSON structured-sequence target (see
# multitask/seq2seq_grammar.py) -- same mechanism build_discharge_me
# already relies on for "summarization", just a different plain-text
# shape, not free prose.
PLAIN_TEXT_TASKS = {"summarization", "sdoh_event_seq2seq"}

# sdoh_event_seq2seq task-tag -> parent end-to-end task-tag it re-frames
# the same underlying events into a different target serialization for
# (see multitask/seq2seq_grammar.py). Kept separate from SUBTASK_OF/
# PARAGRAPH_OF -- a distinct mechanism from either (same text AND same
# labels as the parent, just a different output grammar) -- so
# hf_dataset_stats.py can report it under its own heading.
SEQ2SEQ_OF = {"sdoh_event_seq2seq": "sdoh_event"}


def _check_task(task: str) -> None:
    if task not in TASKS:
        raise ValueError(f"unknown task tag: {task!r}")


def _target_str(task: str, target) -> str:
    if task in PLAIN_TEXT_TASKS:
        return target if isinstance(target, str) else str(target)
    # Canonical key order via the single-source-of-truth spec in
    # multitask/target_schema.py. Without it, key order is inherited from
    # gold-file layout, so the same attribute/role key-set serializes
    # several different ways across examples -- contradictory supervision
    # on serialization for identical content.
    return json.dumps(canonicalize(task, target), ensure_ascii=False)


def tag(task: str) -> str:
    """The two-level model-facing tag, e.g. '<<Information Extraction: SDoH>>'.

    Renders the stored underscore-joined `display` identifiers
    (`Information_Extraction`, `Concept_Assertion_Relation`, ...) with
    spaces instead, at render time only -- the underscores stay in `TASKS`/
    `CATEGORIES` since other code (hf_dataset_stats.py's generated README
    table, design.md) still references them as identifiers. Measured impact
    across all 25 task tags: 280 -> 235 cl100k tokens (16% fewer), because
    BPE has real merges for " Extraction"/" Assertion"/" Relation" as
    tokens but not for "_Ex"/"traction"/"_Ass"/"ert"/"ion" -- the
    underscore-joined form forces the tokenizer through several junk
    subword pieces that carry no semantic content the model can bind to.
    """
    _check_task(task)
    meta = TASKS[task]
    category = CATEGORIES[meta["category"]]
    cat_disp = category["display"].replace("_", " ")
    task_disp = meta["display"].replace("_", " ")
    return f"<<{cat_disp}: {task_disp}>>"


def render(task: str, input_text: str, target) -> str:
    """Render one training example as a single causal-LM text sequence."""
    _check_task(task)
    meta = TASKS[task]
    category = CATEGORIES[meta["category"]]
    return (
        f"{tag(task)}\n"
        f"{category['system']}\n"
        f"{meta['instruction']}\n"
        f"INPUT: {input_text}\n"
        f"OUTPUT: {_target_str(task, target)}"
    )


def render_messages(task: str, input_text: str, target) -> list:
    """Render one training example as OpenAI-style chat messages, so
    tokenizer.apply_chat_template() can be applied directly. The two-level
    tag + generic category framing go in the system turn (shared by every
    task in the category); the dataset-specific instruction + source text
    form the user turn; the (JSON, or plain text for PLAIN_TEXT_TASKS)
    target is the assistant turn."""
    _check_task(task)
    meta = TASKS[task]
    category = CATEGORIES[meta["category"]]
    return [
        {"role": "system", "content": f"{tag(task)}\n{category['system']}"},
        {"role": "user", "content": f"{meta['instruction']}\nINPUT: {input_text}"},
        {"role": "assistant", "content": _target_str(task, target)},
    ]


# Some builders append already-extracted context after the raw note text,
# separated by this exact "\n\n<LABEL>:\n" convention (see
# builders/ner_re_2018_t2.py's RE-only and builders/medication_event_2022_t1.py's
# attr-only input_text construction: f"{doc_text}\n\nEXTRACTED_ENTITIES:\n{...}").
# Recognized here so the markdown renderer can give that block its own
# fenced section instead of burying already-structured context (a bullet
# list of pre-extracted entities/triggers, not raw note prose) inside the
# same code fence as the note -- the plain renderer is untouched, since
# flat prose has no such seam to preserve.
_EXTRACTED_CONTEXT_MARKERS = ("EXTRACTED_ENTITIES", "EXTRACTED_TRIGGERS")


def _render_input_sections_md(input_text: str) -> str:
    for marker in _EXTRACTED_CONTEXT_MARKERS:
        needle = f"\n\n{marker}:\n"
        idx = input_text.find(needle)
        if idx == -1:
            continue
        note_text = input_text[:idx]
        extra_text = input_text[idx + len(needle):]
        label = marker.replace("_", " ").title()
        return (
            f"## Input\n```\n{note_text}\n```\n\n"
            f"## {label}\n```\n{extra_text}\n```"
        )
    return f"## Input\n```\n{input_text}\n```"


def render_md(task: str, input_text: str, target) -> str:
    """Markdown-formatted variant of render(): same content, but the
    instruction and input get their own ## sections and the input is
    fenced, so dense instruction prose and raw note text (which often
    contains its own pseudo-headers, e.g. section titles) don't visually
    run together."""
    _check_task(task)
    meta = TASKS[task]
    category = CATEGORIES[meta["category"]]
    return (
        f"{tag(task)}\n"
        f"{category['system']}\n\n"
        f"## Instructions\n"
        f"{meta['instruction']}\n\n"
        f"{_render_input_sections_md(input_text)}\n\n"
        f"## Output\n"
        f"{_target_str(task, target)}"
    )


def render_messages_md(task: str, input_text: str, target) -> list:
    """Markdown-formatted variant of render_messages(): same role split,
    but the user turn separates instruction from input the same way as
    render_md()."""
    _check_task(task)
    meta = TASKS[task]
    category = CATEGORIES[meta["category"]]
    return [
        {"role": "system", "content": f"{tag(task)}\n{category['system']}"},
        {"role": "user", "content": (
            f"## Instructions\n{meta['instruction']}\n\n"
            f"{_render_input_sections_md(input_text)}"
        )},
        {"role": "assistant", "content": _target_str(task, target)},
    ]


def example(task, dataset, split, locator, input_text, target):
    """Build one unified-dataset record. Carries both the flat causal-LM
    text sequence (used by build_unified_dataset.py/token_stats.py) and the
    OpenAI-style chat messages (used by build_hf_datasets.py), plus the
    task's category as per-example metadata."""
    return {
        "task": task,
        "category": TASKS[task]["category"],
        "dataset": dataset,
        "split": split,
        "id": locator,
        "text": render(task, input_text, target),
        "messages": render_messages(task, input_text, target),
        "text_md": render_md(task, input_text, target),
        "messages_md": render_messages_md(task, input_text, target),
    }
