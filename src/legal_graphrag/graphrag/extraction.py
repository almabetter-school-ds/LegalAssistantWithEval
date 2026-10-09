"""
LLM-based extraction helpers for the Legal GraphRAG pipeline.

Three extraction jobs, each a separate, narrow LLM call rather than one big
"extract everything" prompt — smaller, single-purpose prompts are easier to
validate, cheaper to retry individually, and let you swap/tune one stage
without touching the others:

    1. extract_clauses()   — chunk text -> individual clauses + any judgment
                              citations mentioned inline (e.g. "as held in
                              XYZ Pvt Ltd v ABC Ltd (2019) 4 SCC 112").
    2. detect_conflicts()  — a document's own clause set -> pairs of clauses
                              that contradict each other (e.g. two different
                              governing-law clauses).
    3. flag_risks()        — a clause set -> risk level + reason per clause.

All three prompt for JSON-only output and are parsed defensively (strip code
fences, tolerate a leading/trailing explanation the model wasn't supposed to
add).
"""

from __future__ import annotations

import json
import re

from ..llm_client import call_json, call_text


# ---------------------------------------------------------------------------
# 4. Text-to-Cypher (GraphRAG query side), with a read-only safety guard
# ---------------------------------------------------------------------------

GRAPH_SCHEMA_DESCRIPTION = """
Nodes:
  (:DocumentJob {job_id, document_name, status})
  (:Contract {contract_id, name, document_name, approved})
  (:Party {name})
  (:Clause {clause_id, text, page_start, page_end, section, clause_type})
  (:Judgment {citation, court, year, summary})
  (:RiskFlag {flag_id, risk_level, reason})

Relationships:
  (:DocumentJob)-[:PRODUCED]->(:Contract)
  (:Contract)-[:HAS_VENDOR]->(:Party)
  (:Contract)-[:CONTAINS_CLAUSE]->(:Clause)
  (:Clause)-[:SAME_CLAUSE_AS {similarity}]->(:Clause)
  (:Clause)-[:CONFLICTS_WITH {reason}]->(:Clause)
  (:Clause)-[:INTERPRETED_BY]->(:Judgment)
  (:Clause)-[:FLAGGED_AS]->(:RiskFlag)
"""

_FORBIDDEN_CYPHER_KEYWORDS = re.compile(
    r"\b(CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP|CALL\s+apoc\.|LOAD\s+CSV)\b", re.IGNORECASE
)


def is_read_only_cypher(cypher: str) -> bool:
    """
    Safety guard for LLM-generated Cypher: reject anything that could
    mutate the graph. This matters specifically because the query is
    generated from untrusted natural-language input — never execute
    generated Cypher without this check.
    """
    return not _FORBIDDEN_CYPHER_KEYWORDS.search(cypher)


_CYPHER_SYSTEM_PROMPT = f"""You translate legal/contract questions into a single \
read-only Cypher query for a Neo4j graph with this schema:
{GRAPH_SCHEMA_DESCRIPTION}

Rules:
- Output ONLY the Cypher query — no prose, no markdown fences, no explanation.
- Use only MATCH / OPTIONAL MATCH / WHERE / WITH / RETURN / ORDER BY / LIMIT / UNWIND.
- Never use CREATE, MERGE, DELETE, SET, REMOVE, or DROP.
- Always RETURN properties needed to cite the source (contract name, clause text or id,
  judgment citation), not whole nodes.
"""


def generate_cypher(question: str) -> str:
    cypher = call_text(_CYPHER_SYSTEM_PROMPT, question)
    cypher = re.sub(r"^```(cypher)?|```$", "", cypher.strip(), flags=re.MULTILINE).strip()
    if not is_read_only_cypher(cypher):
        raise ValueError(f"Generated Cypher failed the read-only safety check:\n{cypher}")
    return cypher


# ---------------------------------------------------------------------------
# 5. Answer synthesis from combined vector-search + graph-query evidence
# ---------------------------------------------------------------------------

_SYNTHESIS_SYSTEM_PROMPT = """You are a legal research assistant. Answer the \
user's question using ONLY the evidence provided below (vector-search hits \
from the document text, and results from a graph query over contracts, \
clauses, and judgments). Cite specific contracts, clause text/ids, and \
judgment citations for every claim. If the evidence does not fully answer \
the question, say so explicitly rather than speculating."""


def synthesize_answer(question: str, vector_hits: list[dict], graph_hits: list[dict]) -> str:
    evidence = {
        "vector_search_results": vector_hits,
        "graph_query_results": graph_hits,
    }
    user_prompt = f"Question: {question}\n\nEvidence (JSON):\n{json.dumps(evidence, default=str, indent=2)}"
    return call_text(_SYNTHESIS_SYSTEM_PROMPT, user_prompt, max_tokens=1200)


# ---------------------------------------------------------------------------
# 1. Clause + inline judgment-citation extraction
# ---------------------------------------------------------------------------

_CLAUSE_SYSTEM_PROMPT = """You are a legal document analyst. Given a chunk of \
contract text, extract each distinct clause it contains.

Respond with ONLY a JSON array (no prose, no markdown fences). Each element:
{
  "clause_type": string,          // e.g. "termination", "governing_law", "indemnification", "confidentiality", "liability_cap", "other"
  "text": string,                 // the clause text, verbatim from the input
  "parties_mentioned": [string],  // any party/company names mentioned in this clause
  "judgment_citations": [         // any court judgments explicitly cited in this clause, else []
    {"citation": string, "court": string|null, "year": integer|null}
  ]
}
If the chunk contains no identifiable clauses, return []."""


def extract_clauses(chunk_text: str) -> list[dict]:
    result = call_json(_CLAUSE_SYSTEM_PROMPT, chunk_text)
    return result if isinstance(result, list) else []


# ---------------------------------------------------------------------------
# 2. Conflict detection across a document's own clauses
# ---------------------------------------------------------------------------

_CONFLICT_SYSTEM_PROMPT = """You are a legal document analyst. Given a JSON \
array of clauses (each with an "id" and "text") from the SAME contract, \
identify pairs that directly contradict each other (e.g. two different \
governing-law jurisdictions, inconsistent notice periods, conflicting \
liability caps).

Respond with ONLY a JSON array (no prose, no markdown fences):
[{"clause_id_a": string, "clause_id_b": string, "reason": string}, ...]
If there are no conflicts, return []."""


def detect_conflicts(clauses: list[dict]) -> list[dict]:
    """
    clauses: [{"id": clause_id, "text": clause_text}, ...] for one contract.
    Returns [{"clause_id_a", "clause_id_b", "reason"}, ...].
    """
    if len(clauses) < 2:
        return []
    payload = json.dumps([{"id": c["id"], "text": c["text"]} for c in clauses])
    result = call_json(_CONFLICT_SYSTEM_PROMPT, payload)
    return result if isinstance(result, list) else []


# ---------------------------------------------------------------------------
# 3. Risk flagging
# ---------------------------------------------------------------------------

_RISK_SYSTEM_PROMPT = """You are a contract risk reviewer. Given a JSON array \
of clauses (each with an "id", "clause_type", and "text"), assign a risk \
level to each.

Respond with ONLY a JSON array (no prose, no markdown fences):
[{"clause_id": string, "risk_level": "low"|"medium"|"high", "reason": string}, ...]
Only include clauses that carry at least low risk worth a reviewer's attention;
omit clauses with no notable risk."""


def flag_risks(clauses: list[dict]) -> list[dict]:
    """
    clauses: [{"id": clause_id, "clause_type": ..., "text": ...}, ...]
    Returns [{"clause_id", "risk_level", "reason"}, ...].
    """
    if not clauses:
        return []
    payload = json.dumps([{"id": c["id"], "clause_type": c.get("clause_type"), "text": c["text"]} for c in clauses])
    result = call_json(_RISK_SYSTEM_PROMPT, payload)
    return result if isinstance(result, list) else []
