"""
AI-assisted extractors: fields with no deterministic mechanism, requiring
an LLM to read prose and answer with a mandatory supporting citation.

Model: Claude Haiku 4.5 (see field_data_dictionary.py's per-field
ai_extraction_instructions for each task definition). Chosen over GPT-4o
family (Meridian's existing OpenAI pattern) for cost/accuracy on this task
shape: short-context reading comprehension with a verbatim-citation
requirement, where citation faithfulness matters more than raw model size.
Re-evaluate per-field if a pilot run shows Haiku hallucinating excerpts —
Sonnet is the escalation path, not a different provider.

Build order: ceo_came_from_parent (the proof case) first, validated live
against all 3 pilots. Then the 5 fields below it in _BOOL_FIELD_CONFIGS —
all bool/yes-no questions over the same Form 10 + Exhibit 99.1 document,
sharing one generic extractor (extract_form10_bool_field) rather than 5
near-duplicate functions. The two tenure fields (ceo_tenure_at_parent,
cfo_tenure_at_parent) are deliberately NOT here yet — they need a numeric
answer schema plus raw-vs-computed value tracking (an explicit dictionary
requirement, since stated tenure and date-math-derived tenure can
disagree), a different enough shape to warrant their own pass.
"""
import json
import os
from dataclasses import dataclass
from typing import List, Optional

import anthropic

from spinoff_research.cost_tracking import log_ai_call
from spinoff_research.extraction import ExtractedFieldValue, SourceCitation
from spinoff_research.form10_retrieval import (
    Form10NotFoundError,
    find_relevant_sections,
    get_or_ingest_form10,
)
from spinoff_research.status import FieldStatus

_MODEL = "claude-haiku-4-5"

_SYSTEM_PROMPT = (
    "You extract facts from SEC filings for spin-off research. Read the "
    "provided Form 10 excerpts and answer only from what is stated in the "
    "text. Never guess or infer beyond what is written."
)

_BOOL_ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string", "enum": ["true", "false", "not_stated"]},
        "supporting_excerpt": {
            "type": "string",
            "description": "A verbatim quote (copied exactly, word-for-word) from the "
                            "provided text that supports the answer. Empty string if answer is not_stated.",
        },
        "reasoning_summary": {
            "type": "string",
            "description": "One sentence explaining the answer.",
        },
    },
    "required": ["answer", "supporting_excerpt", "reasoning_summary"],
    "additionalProperties": False,
}


class AiExtractionError(Exception):
    pass


_QUOTE_NORMALIZATION = str.maketrans({
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", " ": " ",
})


def _normalize_for_comparison(text: str) -> str:
    """
    Two independent sources of false EXTRACTED_UNCERTAIN, both confirmed
    live:

    1. Claude's JSON output consistently normalizes smart/curly quotes and
       dashes to their ASCII equivalents when echoing a "verbatim" excerpt
       (GE Vernova/Inhibrx source HTML uses curly quotes around "CEO"; the
       echoed excerpt used straight quotes for the same text).
    2. A real, continuous sentence in the source document can straddle a
       _PROSE_CHUNK_CHARS boundary in ingestion.py, landing in two separate
       document_sections rows that this module joins with "\n\n---\n\n".
       Claude reads the (chunked) sections but correctly echoes the
       sentence as continuous prose. ingestion.py cuts exactly at a
       character offset — mid-word splits happen (confirmed live: GE
       Vernova's CFO bio split "appo" | "inted" across the boundary) — so
       the joiner must be removed with NO replacement character, not
       collapsed to a space, or "appo inted" will never match "appointed".

    Comparing after normalizing both sides means the faithfulness check
    verifies substance, not artifacts of typography or chunking — a real
    hallucination still won't match after normalization.
    """
    text = text.replace("\n\n---\n\n", "")
    return " ".join(text.translate(_QUOTE_NORMALIZATION).split())


def _call_claude_bool(spinoff_name: str, sections_text: str, question: str, field_key: str) -> dict:
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    user_prompt = (
        f"Company: {spinoff_name} (the spin-off company being separated).\n\n"
        f"Question: {question}\n\n"
        f"Excerpts from the company's Form 10 filing (selected by keyword "
        f"match — may include irrelevant surrounding text):\n\n{sections_text}\n\n"
        f"If the excerpts do not address this question at all, answer "
        f"not_stated rather than guessing. Respond under 100 words total."
    )
    response = client.messages.create(
        model=_MODEL,
        max_tokens=400,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
        output_config={"format": {"type": "json_schema", "schema": _BOOL_ANSWER_SCHEMA}},
    )
    # Log cost for every real API call, even one whose content we can't
    # parse below — a failed-to-parse response still billed real tokens.
    log_ai_call(
        model=_MODEL, field_key=field_key, caller=spinoff_name,
        input_tokens=response.usage.input_tokens, output_tokens=response.usage.output_tokens,
    )
    text = next((b.text for b in response.content if b.type == "text"), None)
    if text is None:
        raise AiExtractionError(f"No text content in Claude response (stop_reason={response.stop_reason})")
    return json.loads(text)


_ANSWER_TO_RAW_VALUE = {"true": "True", "false": "False", "not_stated": None}


@dataclass
class BoolFieldConfig:
    field_key: str
    question: str
    search_terms: List[str]


_BOOL_FIELD_CONFIGS: List[BoolFieldConfig] = [
    BoolFieldConfig(
        field_key="ceo_came_from_parent",
        question=(
            "Did {spinoff_name}'s Chief Executive Officer, at the time of "
            "separation, previously hold a role at the parent company (any "
            "title, any duration)? Answer false if the CEO was hired "
            "externally at or near separation with no prior parent-company role."
        ),
        search_terms=[
            "chief executive officer", "president and chief executive", "biography",
            "prior to the separation", "prior to the distribution", "served as",
            "management", "executive officers",
        ],
    ),
    BoolFieldConfig(
        field_key="cfo_came_from_parent",
        question=(
            "Did {spinoff_name}'s Chief Financial Officer, at the time of "
            "separation, previously hold a role at the parent company (any "
            "title, any duration)? Answer false if the CFO was hired "
            "externally at or near separation with no prior parent-company role."
        ),
        search_terms=[
            "chief financial officer", "biography", "prior to the separation",
            "prior to the distribution", "served as", "management", "executive officers",
        ],
    ),
    BoolFieldConfig(
        field_key="is_100_percent_spinoff",
        question=(
            "Did the parent company distribute approximately 100% of "
            "{spinoff_name}'s outstanding shares to parent shareholders "
            "(a full spin-off), as opposed to a partial distribution or "
            "carve-out IPO where the parent retained a stake? If the "
            "filing describes a phased or partial distribution, answer "
            "false rather than guessing at the final percentage."
        ),
        search_terms=[
            "distribute all", "100%", "approximately 100%", "retain no shares",
            "retain an ownership interest", "distribution ratio",
        ],
    ),
    BoolFieldConfig(
        field_key="parent_retained_ownership",
        question=(
            "Did the parent company retain any equity stake in "
            "{spinoff_name} after the distribution (any amount, any share "
            "class)? Answer false only if the filing states the parent "
            "retained no ownership interest at all — e.g. it will "
            "distribute all/100% of the outstanding shares."
        ),
        # Broadened after live validation: "retain"/"retained interest"
        # alone missed the actual answer for GE Vernova (a clean 100%
        # spin) — the relevant sentence used "distribute all... on a pro
        # rata basis" and "shares of our common stock it may retain",
        # neither matched by the original narrower term list, while
        # "retain" alone over-matched unrelated boilerplate (retaining
        # employees, retaining earnings) and crowded out the real passage.
        search_terms=[
            "retain", "retained interest", "will continue to own", "will retain",
            "will own up to", "distribute all", "distribution ratio", "pro rata basis",
            "common stock it may retain", "no shares",
        ],
    ),
    BoolFieldConfig(
        field_key="is_regulatory_driven",
        question=(
            "Does the filing frame the spin-off of {spinoff_name} as "
            "required by, or a condition of, a regulatory or antitrust "
            "process (e.g. a merger-clearance divestiture or consent "
            "decree)? Answer false for a spin-off described as a "
            "strategic choice that merely needed routine regulatory "
            "approvals (which every spin-off needs) — only answer true "
            "if the filing frames the separation itself as required by regulators."
        ),
        search_terms=[
            "regulatory approval", "antitrust", "consent decree",
            "required by", "condition of", "regulatory driven",
        ],
    ),
    BoolFieldConfig(
        field_key="initially_paying_dividend",
        question=(
            "Does the filing state that {spinoff_name} intends to pay, or "
            "has adopted a policy of paying, a dividend at or immediately "
            "following separation? Filings often state this negatively "
            "('does not currently intend to pay a dividend') — treat an "
            "explicit statement of no dividend plan as a confident false, "
            "not not_stated."
        ),
        search_terms=[
            "dividend policy", "intends to pay", "initial dividend",
            "does not currently intend to pay", "dividend",
        ],
    ),
]

_BOOL_FIELD_BY_KEY = {c.field_key: c for c in _BOOL_FIELD_CONFIGS}


def extract_form10_bool_field(
    conn,
    transaction_id: int,
    spinoff_cik: str,
    spinoff_name: str,
    field_key: str,
    company_id: Optional[int] = None,
) -> ExtractedFieldValue:
    """
    Generic yes/no/not_stated extractor over a spinco's Form 10 + Exhibit
    99.1: finds (ingesting if needed) the filing, narrows it to
    field-relevant prose sections via field_key's configured search terms,
    and asks Claude Haiku 4.5 the field's configured question with a
    mandatory verbatim excerpt.

    field_key must be a key in _BOOL_FIELD_BY_KEY (see _BOOL_FIELD_CONFIGS
    above for the full list and exact question wording).

    Returns NOT_FOUND if no Form 10 exists at all, EXTRACTED_UNCERTAIN if
    Claude answers but the excerpt can't be verified as an exact substring
    of the source (a real hallucination signal, not just a quality nit —
    treat it as needing human review rather than trusting the citation),
    and NOT_YET_DETERMINABLE if the filing doesn't address the question at all.
    """
    config = _BOOL_FIELD_BY_KEY[field_key]

    try:
        document_ids = get_or_ingest_form10(conn, transaction_id, spinoff_cik, company_id=company_id)
    except Form10NotFoundError as e:
        return ExtractedFieldValue(
            field_key=field_key, extraction_method="ai_assisted",
            status=FieldStatus.NOT_FOUND, model_used=_MODEL,
            sources=[SourceCitation(reasoning_summary=str(e))],
        )
    primary_document_id = document_ids[0]

    sections = find_relevant_sections(conn, document_ids, config.search_terms)
    if not sections:
        return ExtractedFieldValue(
            field_key=field_key, extraction_method="ai_assisted",
            status=FieldStatus.NOT_YET_DETERMINABLE, model_used=_MODEL,
            sources=[SourceCitation(
                document_id=primary_document_id,
                reasoning_summary="No relevant prose sections found via keyword search.",
            )],
        )

    sections_text = "\n\n---\n\n".join(s.content for s in sections)
    question = config.question.format(spinoff_name=spinoff_name)
    try:
        parsed = _call_claude_bool(spinoff_name, sections_text, question, field_key)
    except (anthropic.APIError, json.JSONDecodeError, AiExtractionError) as e:
        return ExtractedFieldValue(
            field_key=field_key, extraction_method="ai_assisted",
            status=FieldStatus.REQUIRES_MANUAL_REVIEW, model_used=_MODEL,
            sources=[SourceCitation(document_id=primary_document_id, reasoning_summary=f"Claude call failed: {e}")],
        )

    answer = parsed.get("answer")
    excerpt = (parsed.get("supporting_excerpt") or "").strip()
    reasoning = parsed.get("reasoning_summary", "")

    if answer == "not_stated" or not excerpt:
        return ExtractedFieldValue(
            field_key=field_key, extraction_method="ai_assisted",
            status=FieldStatus.NOT_YET_DETERMINABLE, model_used=_MODEL,
            sources=[SourceCitation(document_id=primary_document_id, reasoning_summary=reasoning or "Not stated in filing.")],
        )

    normalized_excerpt = _normalize_for_comparison(excerpt)
    excerpt_verified = normalized_excerpt in _normalize_for_comparison(sections_text)
    matching_section = next(
        (s for s in sections if normalized_excerpt in _normalize_for_comparison(s.content)), None
    )

    return ExtractedFieldValue(
        field_key=field_key, extraction_method="ai_assisted",
        status=FieldStatus.EXTRACTED_HIGH_CONFIDENCE if excerpt_verified else FieldStatus.EXTRACTED_UNCERTAIN,
        raw_value=_ANSWER_TO_RAW_VALUE.get(answer),
        normalized_value=_ANSWER_TO_RAW_VALUE.get(answer),
        confidence=0.85 if excerpt_verified else 0.4,
        model_used=_MODEL,
        sources=[SourceCitation(
            document_id=matching_section.document_id if matching_section else primary_document_id,
            section_id=matching_section.section_id if matching_section else None,
            supporting_excerpt=excerpt,
            reasoning_summary=reasoning if excerpt_verified else
                f"{reasoning} [WARNING: excerpt could not be verified verbatim in source text — needs manual review]",
        )],
    )


def extract_ceo_came_from_parent(
    conn,
    transaction_id: int,
    spinoff_cik: str,
    spinoff_name: str,
    company_id: Optional[int] = None,
) -> ExtractedFieldValue:
    """Thin wrapper kept for backward compatibility with existing callers
    (orchestrator.py, tests) — delegates to the generic bool extractor."""
    return extract_form10_bool_field(
        conn, transaction_id, spinoff_cik, spinoff_name, "ceo_came_from_parent", company_id=company_id,
    )
