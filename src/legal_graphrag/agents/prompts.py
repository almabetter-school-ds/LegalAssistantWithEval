"""
LLM-backed logic for the three "reasoning" sub-agents in the pipeline —
Router, Auditor, and Synthesizer. Kept separate from graphrag/extraction.py
(which handles ingestion-time extraction: clauses, conflicts, risk, and
text-to-Cypher) since these three run at query time and reason about
*retrieved evidence* rather than raw document text.

All three now validate their LLM output through a Pydantic model before
returning it, instead of duck-typing a raw dict with .get(). This isn't
LangChain's .with_structured_output() (this project calls the HF SDK
directly, not through LangChain) but it achieves the same goal: every field
downstream code reads is guaranteed to exist with the right type, because
Pydantic fills in defaults for anything the model omitted rather than
letting a missing key surface later as a KeyError three functions away from
where the LLM call actually happened. If validation itself fails outright
(malformed JSON, wrong types Pydantic can't coerce), each function falls
back to an explicit, safe default value rather than raising — a bad LLM
response should degrade the pipeline's confidence, not crash it.
"""

from __future__ import annotations

import json
import re
from typing import Optional

from pydantic import BaseModel, Field, ValidationError

from ..llm_client import call_json, call_text  # call_text kept for any future plain-text needs
from ..tracing import traceable

# ---------------------------------------------------------------------------
# Structured output models
# ---------------------------------------------------------------------------

class RouterDecision(BaseModel):
    routes: list[str] = Field(default_factory=lambda: ["hybrid"])
    query_style: str = "balanced"
    reasoning: str = ""


class EvidenceVerdict(BaseModel):
    sufficient: bool
    reasoning: str
    gaps: list[str] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)


def derive_confidence(evidence_verdict: dict, has_uncertainty: bool) -> str:
    """
    Deterministic "High"/"Medium"/"Low" confidence label for the structured
    answer card (see output_formatting.py), computed from the Auditor's own
    verdict rather than asking the Synthesizer LLM to self-report a
    confidence score — the auditor already reasoned explicitly over
    gaps/contradictions in verify_evidence() above, and re-deriving from
    that (instead of a second, possibly inconsistent LLM judgment) keeps the
    displayed confidence directly traceable to a specific upstream decision.

      - Low:    evidence marked insufficient, OR the auditor flagged
                contradictions (a contradiction is worse than a gap — it
                means the evidence disagrees with itself, not just that
                something's missing).
      - Medium: evidence sufficient, but the auditor flagged gaps, or the
                synthesizer itself flagged has_uncertainty.
      - High:   evidence sufficient, no gaps, no contradictions, no
                synthesizer-flagged uncertainty.
    """
    sufficient = bool(evidence_verdict.get("sufficient", False))
    contradictions = evidence_verdict.get("contradictions") or []
    gaps = evidence_verdict.get("gaps") or []

    if not sufficient or contradictions:
        return "Low"
    if gaps or has_uncertainty:
        return "Medium"
    return "High"


class SynthesizedAnswer(BaseModel):
    answer: str
    evidence: str = ""                        # short (1-3 sentence) grounding excerpt/explanation, distinct
                                                # from `answer` — see output_formatting.py's "Evidence:" line
    document: str = ""                         # contract/document name the answer is drawn from
    source_section: Optional[str] = None       # clause/section heading, e.g. "Section 4.2 (Term)"
    source_page: Optional[str] = None          # page or page range, e.g. "1" or "3-4"
    citations: list[str] = Field(default_factory=list)
    risk_level: Optional[str] = None          # "low" | "medium" | "high" | None
    has_uncertainty: bool = False


# ---------------------------------------------------------------------------
# Query-style classification -> hybrid-search blend weight (alpha)
# ---------------------------------------------------------------------------
# Fixing alpha globally is a compromise: it helps exact clause/citation
# lookups but hurts recall on paraphrase-style questions. Since the Router
# is already classifying the question anyway, it's the natural place to
# also pick the dense/sparse blend weight per query.

ALPHA_BY_QUERY_STYLE = {
    "exact_match": 0.2,   # favor lexical/BM25 — looking for specific wording, a clause number, a citation
    "balanced": 0.4,      # slight lexical lean — legal text still rewards exact term matching most of the time
    "semantic": 0.7,      # favor dense — open-ended "what happens if / explain / summarize" questions
}
DEFAULT_ALPHA = ALPHA_BY_QUERY_STYLE["balanced"]

_EXACT_MATCH_PATTERNS = [
    re.compile(r'"[^"]{3,}"'),
    re.compile(r"\bsection\s+\d", re.IGNORECASE),
    re.compile(r"\bclause\s+\d", re.IGNORECASE),
    re.compile(r"\b(art\.|article)\s+\d", re.IGNORECASE),
    re.compile(r"\bv\.\s+[A-Z]"),
]
_SEMANTIC_KEYWORDS = [
    "why", "explain", "what happens if", "summarize", "summarise", "in plain english",
    "does this mean", "how does", "what if", "implications", "risk of",
]


def classify_query_style(question: str) -> str:
    """Returns "exact_match" | "semantic" | "balanced", via keyword heuristics only (no LLM)."""
    if any(p.search(question) for p in _EXACT_MATCH_PATTERNS):
        return "exact_match"
    q_lower = question.lower()
    if any(kw in q_lower for kw in _SEMANTIC_KEYWORDS):
        return "semantic"
    return "balanced"


# ---------------------------------------------------------------------------
# Router — classifies a question into ONE OR MORE retrieval paths, plus the
# dense/sparse blend weight to use if "hybrid" is among them
# ---------------------------------------------------------------------------
# Real legal questions often need BOTH specialist agents at once — e.g.
# "find all indemnification clauses in contracts involving Vendor X" needs
# GraphRAGAgent (which contracts involve Vendor X) AND HybridSearchAgent
# (semantic search for indemnification language within those results). The
# router can return ["hybrid", "graph"] to fan out to both in parallel;
# "direct" never combines with anything else, since it means "no new
# retrieval at all."

_GRAPH_KEYWORDS = [
    "same clause", "another contract", "other contract", "conflict", "conflicts",
    "judgment", "judgement", "precedent", "interpreted by", "relationship between",
    "across contracts", "multiple contracts", "linked to", "connected to",
    "cites", "citing", "who else", "vendor", "counterparty", "party to",
]

# If a graph-relationship keyword AND a SPECIFIC clause-type keyword both
# appear, the question likely needs both specialists (e.g. "vendor X" +
# "indemnification"). Deliberately excludes generic words like "clause" or
# "provision" — those co-occur with almost any graph-relationship question
# ("conflicting clauses", "clauses interpreted by...") and would over-trigger
# combined mode on questions that only need the graph path.
_CLAUSE_TOPIC_KEYWORDS = [
    "indemnification", "indemnity", "termination", "confidentiality", "liability",
    "non-compete", "non compete", "warranty", "warranties",
    "governing law", "limitation of liability",
]

_ROUTER_SYSTEM_PROMPT = """You classify a legal-document question along two \
independent dimensions.

1. Retrieval path(s) — one or two of:
- "hybrid": finding/quoting specific clause text, a policy, a defined term, \
or searching contracts by attributes (party, date, value, type).
- "graph": traversing RELATIONSHIPS between entities — the same clause in \
multiple contracts, conflicting clauses, clauses interpreted by judgments, \
or multi-hop chains between contracts/parties/precedents.
- "direct": a simple summarization/explanation/follow-up about evidence \
ALREADY gathered in this conversation — no new retrieval at all.
Return TWO paths ("hybrid" AND "graph") only when the question genuinely \
needs both — e.g. "find clause type X in contracts involving party Y" needs \
graph traversal to find party Y's contracts AND semantic search within them \
for clause type X. "direct" is never combined with another path.

2. Query style (only meaningful if "hybrid" is one of the paths) — exactly one of:
- "exact_match": looking for specific wording, a clause number, a defined \
term, or a citation.
- "semantic": open-ended, conceptual, or paraphrase-style question.
- "balanced": neither clearly dominates.

Respond with ONLY a JSON object:
{"routes": ["hybrid"|"graph"|"direct", ...], "query_style": "exact_match"|"semantic"|"balanced", "reasoning": string}"""


def _validated_router_decision(question: str) -> RouterDecision:
    raw = call_json(_ROUTER_SYSTEM_PROMPT, question)
    try:
        decision = RouterDecision.model_validate(raw)
    except ValidationError:
        return RouterDecision(routes=["hybrid"], query_style="balanced",
                               reasoning="router LLM output failed validation; defaulting to hybrid search")

    # "direct" never combines with anything else, and only known routes survive.
    routes = [r for r in decision.routes if r in ("hybrid", "graph", "direct")] or ["hybrid"]
    if "direct" in routes and len(routes) > 1:
        routes = [r for r in routes if r != "direct"]
    decision.routes = routes
    return decision


def classify_routes(question: str) -> tuple[list[str], str, float]:
    """
    Returns (routes, reasoning, alpha). `routes` is a non-empty list, e.g.
    ["hybrid"], ["graph"], ["hybrid", "graph"], or ["direct"]. `alpha` is
    the dense-vs-sparse blend weight for HybridSearchAgent (see
    retrieval.hybrid_search.convex_combination_fusion) — meaningful only
    when "hybrid" is among the routes, but always returned for a consistent
    call signature.
    """
    q_lower = question.lower()
    style = classify_query_style(question)
    alpha = ALPHA_BY_QUERY_STYLE[style]

    graph_match = any(kw in q_lower for kw in _GRAPH_KEYWORDS)
    hybrid_topic_match = any(kw in q_lower for kw in _CLAUSE_TOPIC_KEYWORDS)

    if graph_match and hybrid_topic_match:
        return ["hybrid", "graph"], "matched both a relationship keyword and a clause-topic keyword", alpha
    if graph_match:
        return ["graph"], "matched a relationship/multi-hop keyword pattern", alpha
    if style != "balanced":
        # Confident local classification on both dimensions — no LLM call needed.
        return ["hybrid"], f"keyword-matched query style: {style}", alpha

    # Ambiguous on the fast path: one LLM call for routes + style together.
    decision = _validated_router_decision(question)
    alpha = ALPHA_BY_QUERY_STYLE.get(decision.query_style, DEFAULT_ALPHA)
    return decision.routes, decision.reasoning, alpha


@traceable(name="router.classify_route", run_type="chain")
def classify_route(question: str) -> tuple[str, str, float]:
    """Backward-compatible single-route wrapper around classify_routes(). Prefer classify_routes()."""
    routes, reasoning, alpha = classify_routes(question)
    return routes[0], reasoning, alpha


# ---------------------------------------------------------------------------
# Auditor / EvidenceChecker — verifies retrieved evidence before synthesis
# ---------------------------------------------------------------------------

_AUDITOR_SYSTEM_PROMPT = """You are an evidence auditor for a legal research \
system. You do NOT answer the question. You only assess whether the \
evidence provided is sufficient, relevant, and internally consistent \
enough to answer it responsibly.

Respond with ONLY a JSON object (no prose, no markdown fences):
{
  "sufficient": boolean,
  "reasoning": string,
  "gaps": [string],
  "contradictions": [string]
}"""


@traceable(name="auditor.verify_evidence", run_type="chain")
def verify_evidence(question: str, hybrid_hits: list[dict], graph_hits: list[dict]) -> dict:
    evidence = {"hybrid_search_results": hybrid_hits, "graph_query_results": graph_hits}
    user_prompt = f"Question: {question}\n\nEvidence (JSON):\n{json.dumps(evidence, default=str, indent=2)}"
    raw = call_json(_AUDITOR_SYSTEM_PROMPT, user_prompt)
    try:
        verdict = EvidenceVerdict.model_validate(raw)
    except ValidationError as e:
        verdict = EvidenceVerdict(
            sufficient=False,
            reasoning=f"auditor response failed validation ({e.error_count()} error(s)); treating as insufficient",
        )
    return verdict.model_dump()


# ---------------------------------------------------------------------------
# Synthesizer / AnswerAgent (Adjudicator) — final answer from VERIFIED evidence only
# ---------------------------------------------------------------------------

_SYNTHESIZER_SYSTEM_PROMPT = """You are a legal research assistant writing \
the FINAL answer for a human reviewer. Use ONLY the evidence provided — \
never speculate beyond it.

Respond with ONLY a JSON object (no prose, no markdown fences):
{
  "answer": string,              // the DIRECT answer only — short and specific, e.g.
                                  // "February 18, 2005." or "Yes, Section 4.2 caps liability at $50,000."
                                  // Do NOT restate the evidence or add a long explanation here — that
                                  // belongs in "evidence" below. Plain prose, no markdown.
  "evidence": string,             // 1-3 sentences explaining WHY that's the answer, grounded in and
                                  // quoting/paraphrasing the specific retrieved text that supports it
  "document": string,             // the contract/document name this answer is drawn from — copy from
                                  // the evidence's metadata.document_name field
  "source_section": string|null,  // the clause/section heading from the evidence's metadata.section
                                  // field, if present, e.g. "Section 4.2 (Term)"
  "source_page": string|null,     // the page or page range from the evidence's metadata
                                  // (page_start/page_end), e.g. "1" or "3-4"
  "citations": [string],         // short citation strings pulled out of the answer, e.g.
                                  // "Contract ABC MSA, Clause 4.2", "Smith v. Jones (2019)"
  "risk_level": "low"|"medium"|"high"|null,   // overall risk the evidence reveals, or null if not applicable
  "has_uncertainty": boolean     // true if the evidence auditor flagged gaps/contradictions
                                  // that meaningfully limit confidence in this answer
}

Explicitly reflect any gaps or contradictions the evidence auditor flagged — \
do not paper over them. Do not include markdown formatting inside "answer" \
or "evidence"; plain prose only."""


def _backfill_source_from_top_hit(result: SynthesizedAnswer, hybrid_hits: list[dict],
                                   graph_hits: list[dict]) -> SynthesizedAnswer:
    """
    If the LLM left document/source_section/source_page blank (or the
    validated response fell back to the safe default entirely), backfill
    them programmatically from the top-ranked hit's metadata rather than
    showing a blank Source line — hybrid_hits is already sorted by
    relevance (see hybrid_search.py's reranking step), so its first element
    is the best available grounding even when the LLM didn't echo it back.
    """
    top_hit = (hybrid_hits or graph_hits or [None])[0]
    if not top_hit:
        return result
    meta = top_hit.get("metadata", {}) or {}

    if not result.document:
        result.document = meta.get("document_name", "") or ""
    if not result.source_section:
        result.source_section = meta.get("section")
    if not result.source_page:
        page_start, page_end = meta.get("page_start"), meta.get("page_end")
        if page_start is not None:
            result.source_page = str(page_start) if page_start == page_end else f"{page_start}-{page_end}"
    return result


@traceable(name="synthesizer.synthesize_legal_answer", run_type="chain")
def synthesize_legal_answer(question: str, hybrid_hits: list[dict], graph_hits: list[dict],
                             evidence_verdict: dict) -> dict:
    evidence = {
        "hybrid_search_results": hybrid_hits,
        "graph_query_results": graph_hits,
        "evidence_auditor_verdict": evidence_verdict,
    }
    user_prompt = f"Question: {question}\n\nVerified evidence (JSON):\n{json.dumps(evidence, default=str, indent=2)}"
    raw = call_json(_SYNTHESIZER_SYSTEM_PROMPT, user_prompt, max_tokens=1500)
    try:
        result = SynthesizedAnswer.model_validate(raw)
    except ValidationError:
        # Even the fallback is a validated SynthesizedAnswer — callers never
        # need to guard against a missing "answer" key.
        result = SynthesizedAnswer(
            answer="The system was unable to generate a structured answer from the retrieved evidence. "
                   "Please review the raw evidence directly.",
            evidence="Structured-output validation failed; no grounding excerpt is available.",
            has_uncertainty=True,
        )
    result = _backfill_source_from_top_hit(result, hybrid_hits, graph_hits)
    output = result.model_dump()
    output["confidence"] = derive_confidence(evidence_verdict, output["has_uncertainty"])
    return output


# ---------------------------------------------------------------------------
# Revision — reasons over a human reviewer's feedback on a DRAFT answer and
# produces a revised answer, without going back to retrieval or fabricating
# evidence that wasn't already verified by the Auditor.
# ---------------------------------------------------------------------------
# This is the "make changes" branch of the answer checkpoint's 3-state loop
# (approve / revise-with-comments / reject): a reviewer can say what's wrong
# ("missing Section 4.2", "overstates the risk", "cite the actual clause
# text") and the LLM reasons over that feedback against the SAME evidence
# set, rather than the pipeline just failing the answer outright. The loop
# in agents/legal_pipeline.py keeps returning here until the reviewer
# chooses approve or reject — nothing is written back to Neo4j
# (record_answered_question) on any revision round, only on final approval.

_REVISION_SYSTEM_PROMPT = """You are a legal research assistant revising a \
previously drafted answer based on feedback from a human reviewer. Use \
ONLY the evidence already provided — do not invent new evidence or claims \
that aren't grounded in it. Directly address the reviewer's feedback: \
correct what they flagged, clarify what they found unclear, or add detail \
where they asked for more — while keeping everything else grounded in the \
same verified evidence as the previous answer.

Respond with ONLY a JSON object (no prose, no markdown fences):
{
  "answer": string,               // the DIRECT answer only — short and specific (see synthesizer format)
  "evidence": string,              // 1-3 sentences explaining WHY, grounded in the retrieved evidence
  "document": string,
  "source_section": string|null,
  "source_page": string|null,
  "citations": [string],
  "risk_level": "low"|"medium"|"high"|null,
  "has_uncertainty": boolean
}"""


@traceable(name="synthesizer.revise_legal_answer", run_type="chain")
def revise_legal_answer(question: str, previous_answer: str, hybrid_hits: list[dict],
                         graph_hits: list[dict], evidence_verdict: dict,
                         reviewer_feedback: str) -> dict:
    payload = {
        "previous_answer": previous_answer,
        "reviewer_feedback": reviewer_feedback,
        "hybrid_search_results": hybrid_hits,
        "graph_query_results": graph_hits,
        "evidence_auditor_verdict": evidence_verdict,
    }
    user_prompt = f"Question: {question}\n\nContext for revision (JSON):\n{json.dumps(payload, default=str, indent=2)}"
    raw = call_json(_REVISION_SYSTEM_PROMPT, user_prompt, max_tokens=1500)
    try:
        result = SynthesizedAnswer.model_validate(raw)
    except ValidationError:
        # Fall back to the PRIOR answer unchanged, rather than corrupting
        # state with a malformed revision — the reviewer just sees the same
        # answer again and can retry their feedback or choose approve/reject.
        result = SynthesizedAnswer(answer=previous_answer, evidence="", has_uncertainty=True)
    result = _backfill_source_from_top_hit(result, hybrid_hits, graph_hits)
    output = result.model_dump()
    output["confidence"] = derive_confidence(evidence_verdict, output["has_uncertainty"])
    return output
