"""
AI-assisted extractors: fields with no deterministic mechanism, requiring
an LLM to read prose and answer with a mandatory supporting citation.
First (and so far only) field: ceo_came_from_parent — chosen as the proof
case because it's a short yes/no/uncertain classification over a bounded
document (Form 10 Management-section biography), which validates the
pattern before extending to the other 20 AI-assisted fields.

Model: Claude Haiku 4.5 (see field_data_dictionary.py's
ai_extraction_instructions for the task definition). Chosen over GPT-4o
family (Meridian's existing OpenAI pattern) for cost/accuracy on this
specific task shape: short-context reading comprehension with a verbatim-
citation requirement, where citation faithfulness matters more than raw
model size. Re-evaluate per-field if a pilot run shows Haiku hallucinating
excerpts — Sonnet is the escalation path, not a different provider.
"""
import json
import os
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

_CEO_SEARCH_TERMS = [
    "chief executive officer", "president and chief executive", "biography",
    "prior to the separation", "prior to the distribution", "served as",
    "management", "executive officers",
]

_SYSTEM_PROMPT = (
    "You extract facts from SEC filings for spin-off research. Read the "
    "provided Form 10 excerpts and answer only from what is stated in the "
    "text. Never guess or infer beyond what is written."
)

_ANSWER_SCHEMA = {
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
    "–": "-", "—": "-", " ": " ",
})


def _normalize_for_comparison(text: str) -> str:
    """
    Claude's JSON output consistently normalizes smart/curly quotes and
    dashes to their ASCII equivalents when echoing a "verbatim" excerpt
    (confirmed live: GE Vernova and Inhibrx filings use curly quotes around
    "CEO" in source HTML; Claude's excerpt used straight quotes for the
    same text). Comparing after normalizing both sides means the
    faithfulness check verifies substance, not typography — a real
    hallucination still won't match after normalization, but a
    quote-style difference no longer triggers a false EXTRACTED_UNCERTAIN.
    """
    return " ".join(text.translate(_QUOTE_NORMALIZATION).split())


def _build_user_prompt(spinoff_name: str, sections_text: str) -> str:
    return (
        f"Company: {spinoff_name} (the spin-off company being separated).\n\n"
        f"Question: Did {spinoff_name}'s Chief Executive Officer, at the time of "
        f"separation, previously hold a role at the parent company (any title, "
        f"any duration)? Answer false if the CEO was hired externally at or near "
        f"separation with no prior parent-company role.\n\n"
        f"Excerpts from the company's Form 10 filing (Management/biography "
        f"sections, selected by keyword match — may include irrelevant "
        f"surrounding text):\n\n{sections_text}\n\n"
        f"If the excerpts do not mention the CEO's prior employment at all, "
        f"answer not_stated rather than guessing. Respond under 100 words total."
    )


def _call_claude(spinoff_name: str, sections_text: str, field_key: str) -> dict:
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    response = client.messages.create(
        model=_MODEL,
        max_tokens=400,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": _build_user_prompt(spinoff_name, sections_text)}],
        output_config={"format": {"type": "json_schema", "schema": _ANSWER_SCHEMA}},
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


def extract_ceo_came_from_parent(
    conn,
    transaction_id: int,
    spinoff_cik: str,
    spinoff_name: str,
    company_id: Optional[int] = None,
) -> ExtractedFieldValue:
    """
    Finds (ingesting if needed) the spinco's Form 10, narrows it to
    Management/biography-relevant prose sections, and asks Claude Haiku 4.5
    a bounded yes/no/not_stated question with a mandatory verbatim excerpt.

    Returns NOT_FOUND if no Form 10 exists at all, EXTRACTED_UNCERTAIN if
    Claude answers but the excerpt can't be verified as an exact substring
    of the source (a real hallucination signal, not just a quality nit —
    treat it as needing human review rather than trusting the citation),
    and NOT_YET_DETERMINABLE if the filing doesn't discuss the CEO's prior
    employment at all.
    """
    try:
        document_ids = get_or_ingest_form10(conn, transaction_id, spinoff_cik, company_id=company_id)
    except Form10NotFoundError as e:
        return ExtractedFieldValue(
            field_key="ceo_came_from_parent", extraction_method="ai_assisted",
            status=FieldStatus.NOT_FOUND, model_used=_MODEL,
            sources=[SourceCitation(reasoning_summary=str(e))],
        )
    primary_document_id = document_ids[0]

    sections = find_relevant_sections(conn, document_ids, _CEO_SEARCH_TERMS)
    if not sections:
        return ExtractedFieldValue(
            field_key="ceo_came_from_parent", extraction_method="ai_assisted",
            status=FieldStatus.NOT_YET_DETERMINABLE, model_used=_MODEL,
            sources=[SourceCitation(
                document_id=primary_document_id,
                reasoning_summary="No Management/biography-relevant prose sections found via keyword search.",
            )],
        )

    sections_text = "\n\n---\n\n".join(s.content for s in sections)
    try:
        parsed = _call_claude(spinoff_name, sections_text, "ceo_came_from_parent")
    except (anthropic.APIError, json.JSONDecodeError, AiExtractionError) as e:
        return ExtractedFieldValue(
            field_key="ceo_came_from_parent", extraction_method="ai_assisted",
            status=FieldStatus.REQUIRES_MANUAL_REVIEW, model_used=_MODEL,
            sources=[SourceCitation(document_id=primary_document_id, reasoning_summary=f"Claude call failed: {e}")],
        )

    answer = parsed.get("answer")
    excerpt = (parsed.get("supporting_excerpt") or "").strip()
    reasoning = parsed.get("reasoning_summary", "")

    if answer == "not_stated" or not excerpt:
        return ExtractedFieldValue(
            field_key="ceo_came_from_parent", extraction_method="ai_assisted",
            status=FieldStatus.NOT_YET_DETERMINABLE, model_used=_MODEL,
            sources=[SourceCitation(document_id=primary_document_id, reasoning_summary=reasoning or "Not stated in filing.")],
        )

    normalized_excerpt = _normalize_for_comparison(excerpt)
    excerpt_verified = normalized_excerpt in _normalize_for_comparison(sections_text)
    matching_section = next(
        (s for s in sections if normalized_excerpt in _normalize_for_comparison(s.content)), None
    )

    return ExtractedFieldValue(
        field_key="ceo_came_from_parent", extraction_method="ai_assisted",
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
