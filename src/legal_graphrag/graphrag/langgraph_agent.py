"""
LangGraph GraphRAG agent for legal contracts, built on top of Neo4j + Chroma.

Two separate compiled graphs, matching two separate real-world workflows:

  1. build_ingestion_graph() — after a PDF's text/table chunks have already
     been produced and embedded (see pdf_rag_pipeline.py /
     langgraph_pdf_rag_agent.py), this graph extracts clauses and
     relationships (Clause -[:CONFLICTS_WITH]-> Clause, etc.), persists
     DocumentJob / Contract / Clause / RiskFlag / audit records to Neo4j,
     links clauses to matching clauses in *other* contracts via vector
     similarity, and pauses for a human reviewer to approve or reject the
     whole batch before it's marked final.

  2. build_query_graph() — answers a natural-language question by combining
     semantic search (Chroma, for "what does the text say") with a graph
     query (Neo4j, for "how do these clauses/contracts/judgments relate to
     each other") — genuine GraphRAG, not vector search alone, since
     multi-hop questions like "the same clause in another contract,
     interpreted by multiple judgments" are graph traversals, not
     similarity lookups. The LLM's draft answer is then paused for human
     approval before being treated as final — the model never gets the
     last word on a legal answer.

------------------------------------------------------------------------
Run this in a Colab cell FIRST, in addition to earlier pipeline setup:

    !pip -q install neo4j langgraph anthropic

Environment variables required:
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
    ANTHROPIC_API_KEY
------------------------------------------------------------------------
"""

from __future__ import annotations

import re
import uuid
from typing import Optional, TypedDict

from langgraph.graph import StateGraph, END
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import MemorySaver

from .neo4j_store import Neo4jGraphStore  # noqa: F401  (re-exported for type hints/back-compat)
from .extraction import (
    extract_clauses,
    detect_conflicts,
    flag_risks,
    generate_cypher,
    synthesize_answer,
)
from ..ingestion.pdf_pipeline import query_collection
from ..resources import get_store, get_embedder, get_chroma_collection


# ===========================================================================
# GRAPH 1 — Ingestion: extract clauses/relationships, persist, human-approve
# ===========================================================================

class IngestionState(TypedDict, total=False):
    # inputs
    document_name: str
    contract_name: Optional[str]
    vendor_name: Optional[str]
    text_chunks: list[dict]  # [{"text": str, "page_start": int, "page_end": int, "section": str|None}, ...]

    # intermediate
    job_id: str
    contract_id: str
    clauses: list[dict]           # [{"id","clause_type","text","page_start","page_end","section"}, ...]
    same_clause_links: list[dict]
    conflicts: list[dict]
    risk_flags: list[dict]
    human_decision: dict

    # output
    status: str


def start_job_node(state: IngestionState) -> dict:
    store = get_store()
    job_id = str(uuid.uuid4())
    contract_id = str(uuid.uuid4())

    store.create_document_job(job_id, state["document_name"])
    store.create_contract(contract_id, job_id, state["document_name"], state.get("contract_name"))
    if state.get("vendor_name"):
        store.link_vendor(contract_id, state["vendor_name"])
    store.write_audit_record(job_id, "system", "job_started", f"document={state['document_name']}")

    print(f"[start_job] job_id={job_id} contract_id={contract_id}")
    return {"job_id": job_id, "contract_id": contract_id}


def extract_clauses_node(state: IngestionState) -> dict:
    store = get_store()
    embedder = get_embedder()
    clauses: list[dict] = []

    for chunk in state["text_chunks"]:
        for raw_clause in extract_clauses(chunk["text"]):
            clause_id = str(uuid.uuid4())
            embedding = embedder.encode(raw_clause["text"], normalize_embeddings=True).tolist()

            store.create_clause(
                clause_id=clause_id,
                contract_id=state["contract_id"],
                text=raw_clause["text"],
                embedding=embedding,
                page_start=chunk.get("page_start", 0),
                page_end=chunk.get("page_end", 0),
                section=chunk.get("section"),
                clause_type=raw_clause.get("clause_type"),
            )

            for jc in raw_clause.get("judgment_citations", []):
                store.create_judgment(jc["citation"], jc.get("court"), jc.get("year"), None)
                store.link_interpreted_by(clause_id, jc["citation"])

            clauses.append({
                "id": clause_id,
                "clause_type": raw_clause.get("clause_type"),
                "text": raw_clause["text"],
                "embedding": embedding,
            })

    store.write_audit_record(state["job_id"], "system", "clauses_extracted", f"count={len(clauses)}")
    print(f"[extract_clauses] {len(clauses)} clauses extracted")
    return {"clauses": clauses}


def link_same_clause_node(state: IngestionState) -> dict:
    """
    For every new clause, vector-search Neo4j for similar clauses already
    stored from OTHER contracts, and link matches above the similarity
    threshold with SAME_CLAUSE_AS. This is what makes "the same clause
    appears in another contract" answerable as a graph traversal instead of
    a fuzzy re-search at query time.
    """
    store = get_store()
    links: list[dict] = []

    for clause in state["clauses"]:
        hits = store.find_similar_clauses(clause["id"], clause["embedding"], top_k=3, min_similarity=0.90)
        for hit in hits:
            store.link_same_clause(clause["id"], hit["clause_id"], hit["score"])
            links.append({"clause_id": clause["id"], "matched_clause_id": hit["clause_id"], "score": hit["score"]})

    store.write_audit_record(state["job_id"], "system", "same_clause_links_created", f"count={len(links)}")
    print(f"[link_same_clause] {len(links)} cross-contract clause links created")
    return {"same_clause_links": links}


def detect_conflicts_node(state: IngestionState) -> dict:
    store = get_store()
    conflicts = detect_conflicts([{"id": c["id"], "text": c["text"]} for c in state["clauses"]])
    for pair in conflicts:
        store.create_conflict(pair["clause_id_a"], pair["clause_id_b"], pair["reason"])

    store.write_audit_record(state["job_id"], "system", "conflicts_detected", f"count={len(conflicts)}")
    print(f"[detect_conflicts] {len(conflicts)} conflicting clause pairs found")
    return {"conflicts": conflicts}


def risk_flag_node(state: IngestionState) -> dict:
    store = get_store()
    risks = flag_risks([{"id": c["id"], "clause_type": c["clause_type"], "text": c["text"]} for c in state["clauses"]])
    for r in risks:
        store.create_risk_flag(r["clause_id"], r["risk_level"], r["reason"])

    store.write_audit_record(state["job_id"], "system", "risk_flags_created", f"count={len(risks)}")
    print(f"[risk_flag] {len(risks)} risk flags created")
    return {"risk_flags": risks}


def human_approval_node(state: IngestionState) -> dict:
    """
    Pauses the graph via interrupt() and hands the reviewer a summary to act
    on. Execution resumes only when the caller invokes the graph again with
    Command(resume=<decision dict>) against the same thread_id.
    """
    high_risk = [r for r in state["risk_flags"] if r["risk_level"] == "high"]
    payload = {
        "type": "ingestion_approval_request",
        "job_id": state["job_id"],
        "document_name": state["document_name"],
        "num_clauses": len(state["clauses"]),
        "num_conflicts": len(state["conflicts"]),
        "num_high_risk_flags": len(high_risk),
        "high_risk_flags": high_risk,
        "conflicts": state["conflicts"],
        "message": "Review the extracted clauses/conflicts/risk flags in Neo4j, "
                    "then resume with {'approved': bool, 'reviewer': str, 'comments': str|None}.",
    }
    decision = interrupt(payload)
    return {"human_decision": decision}


def apply_decision_node(state: IngestionState) -> dict:
    store = get_store()
    decision = state["human_decision"]
    approved = bool(decision.get("approved"))
    reviewer = decision.get("reviewer", "unknown")

    store.create_reviewer_decision(state["job_id"], approved, reviewer, decision.get("comments"))
    store.set_contract_approval(state["contract_id"], approved)
    store.update_job_status(state["job_id"], "approved" if approved else "rejected")
    store.write_audit_record(state["job_id"], reviewer, "review_decision", f"approved={approved}")

    print(f"[apply_decision] job {state['job_id']} -> {'approved' if approved else 'rejected'} by {reviewer}")
    return {"status": "approved" if approved else "rejected"}


def build_ingestion_graph():
    graph = StateGraph(IngestionState)
    graph.add_node("start_job", start_job_node)
    graph.add_node("extract_clauses", extract_clauses_node)
    graph.add_node("link_same_clause", link_same_clause_node)
    graph.add_node("detect_conflicts", detect_conflicts_node)
    graph.add_node("risk_flag", risk_flag_node)
    graph.add_node("human_approval", human_approval_node)
    graph.add_node("apply_decision", apply_decision_node)

    graph.set_entry_point("start_job")
    graph.add_edge("start_job", "extract_clauses")
    graph.add_edge("extract_clauses", "link_same_clause")
    graph.add_edge("link_same_clause", "detect_conflicts")
    graph.add_edge("detect_conflicts", "risk_flag")
    graph.add_edge("risk_flag", "human_approval")
    graph.add_edge("human_approval", "apply_decision")
    graph.add_edge("apply_decision", END)

    # A checkpointer is required for interrupt()/resume to work at all — it's
    # what lets the graph "remember" where it paused. MemorySaver is fine for
    # a Colab demo; a real deployment needs a persistent checkpointer (e.g.
    # Postgres) so an approval can survive a process restart while it waits
    # for a human.
    return graph.compile(checkpointer=MemorySaver())


# ===========================================================================
# GRAPH 2 — Query: hybrid vector + graph retrieval, then human-approve the answer
# ===========================================================================

class QueryState(TypedDict, total=False):
    # inputs
    question: str
    collection_name: str

    # intermediate
    query_job_id: str
    vector_hits: list[dict]
    cypher_used: str
    cypher_source: str   # "template" or "generated"
    graph_hits: list[dict]
    draft_answer: str
    human_decision: dict

    # output
    final_answer: Optional[str]
    status: str


def start_query_job_node(state: QueryState) -> dict:
    store = get_store()
    job_id = str(uuid.uuid4())
    store.create_query_job(job_id, state["question"])
    store.write_audit_record(job_id, "system", "query_received", state["question"])
    print(f"[start_query_job] job_id={job_id} question={state['question']!r}")
    return {"query_job_id": job_id}


def vector_search_node(state: QueryState) -> dict:
    """Semantic search over the chunked document text (Chroma) — 'what does the text say'."""
    collection = get_chroma_collection(state["collection_name"])
    embedder = get_embedder()
    hits = query_collection(collection, embedder, state["question"], top_k=5)
    print(f"[vector_search] {len(hits)} vector hits")
    return {"vector_hits": hits}


# --- Template-first Cypher: known high-value question shapes get a vetted,
# --- parameterized query instead of relying on the LLM to write correct
# --- multi-hop Cypher from scratch every time. Faster, cheaper, and safer.

_VENDOR_SAME_CLAUSE_JUDGMENTS_TEMPLATE = """
MATCH (c1:Contract)-[:HAS_VENDOR]->(:Party {name: $vendor_name})
MATCH (c1)-[:CONTAINS_CLAUSE]->(cl:Clause)
MATCH (cl)-[:SAME_CLAUSE_AS]-(cl2:Clause)<-[:CONTAINS_CLAUSE]-(c2:Contract)
WHERE c2 <> c1
MATCH (cl)-[:INTERPRETED_BY]->(j:Judgment)
WITH c1, c2, cl, collect(DISTINCT j.citation) AS judgment_citations
WHERE size(judgment_citations) > 1
RETURN c1.name AS vendor_contract, c2.name AS other_contract,
       cl.clause_id AS clause_id, cl.text AS clause_text, judgment_citations
LIMIT 25
"""

_VENDOR_PATTERN = re.compile(
    r"([A-Z][A-Za-z0-9&.,'\-]*(?:\s+[A-Z][A-Za-z0-9&.,'\-]*)*)\s+(?:is|as)\s+the\s+vendor",
)


def match_template(question: str) -> Optional[tuple[str, dict]]:
    """
    Cheap keyword+regex match for the "vendor X, same clause elsewhere,
    interpreted by multiple judgments" question shape. Returns (cypher,
    params) if matched, else None. This is a heuristic, not an NLU system —
    swap in an LLM-based intent classifier if you need to recognize more
    question shapes reliably; see suggestions in the chat response.
    """
    q_lower = question.lower()
    if "vendor" in q_lower and "same clause" in q_lower and ("judgment" in q_lower or "judgement" in q_lower):
        vendor_match = _VENDOR_PATTERN.search(question)
        if vendor_match:
            return _VENDOR_SAME_CLAUSE_JUDGMENTS_TEMPLATE, {"vendor_name": vendor_match.group(1).strip()}
    return None


def graph_retrieve_node(state: QueryState) -> dict:
    """Structured multi-hop retrieval over Neo4j — 'how do these things relate'."""
    store = get_store()
    template_match = match_template(state["question"])

    if template_match:
        cypher, params = template_match
        source = "template"
    else:
        cypher = generate_cypher(state["question"])  # raises if it fails the read-only guard
        params = {}
        source = "generated"

    hits = store.run_read_query(cypher, params)
    print(f"[graph_retrieve] source={source} hits={len(hits)}")
    return {"cypher_used": cypher, "cypher_source": source, "graph_hits": hits}


def synthesize_answer_node(state: QueryState) -> dict:
    store = get_store()
    answer = synthesize_answer(state["question"], state["vector_hits"], state["graph_hits"])
    store.write_audit_record(state["query_job_id"], "system", "draft_answer_generated", answer[:500])
    print(f"[synthesize_answer] draft ready ({len(answer)} chars)")
    return {"draft_answer": answer}


def human_approval_node_query(state: QueryState) -> dict:
    """
    The LLM's answer is a DRAFT until a human approves it — this is the
    control the user asked for explicitly: the model never gets the final
    word on a legal answer. The reviewer can approve as-is, edit the answer
    text, or reject it outright.
    """
    payload = {
        "type": "answer_approval_request",
        "query_job_id": state["query_job_id"],
        "question": state["question"],
        "draft_answer": state["draft_answer"],
        "cypher_used": state["cypher_used"],
        "cypher_source": state["cypher_source"],
        "graph_hits": state["graph_hits"],
        "vector_hits": state["vector_hits"],
        "message": "Resume with {'approved': bool, 'reviewer': str, "
                    "'edited_answer': str|None, 'comments': str|None}.",
    }
    decision = interrupt(payload)
    return {"human_decision": decision}


def finalize_query_node(state: QueryState) -> dict:
    store = get_store()
    decision = state["human_decision"]
    approved = bool(decision.get("approved"))
    reviewer = decision.get("reviewer", "unknown")

    store.create_reviewer_decision(state["query_job_id"], approved, reviewer, decision.get("comments"))
    store.write_audit_record(state["query_job_id"], reviewer, "review_decision", f"approved={approved}")

    if approved:
        final_answer = decision.get("edited_answer") or state["draft_answer"]
        store.store_query_answer(state["query_job_id"], final_answer)
        store.update_job_status(state["query_job_id"], "answered")
    else:
        final_answer = None
        store.update_job_status(state["query_job_id"], "rejected")

    print(f"[finalize_query] job {state['query_job_id']} -> {'answered' if approved else 'rejected'}")
    return {"final_answer": final_answer, "status": "answered" if approved else "rejected"}


def build_query_graph():
    graph = StateGraph(QueryState)
    graph.add_node("start_query_job", start_query_job_node)
    graph.add_node("vector_search", vector_search_node)
    graph.add_node("graph_retrieve", graph_retrieve_node)
    graph.add_node("synthesize_answer", synthesize_answer_node)
    graph.add_node("human_approval", human_approval_node_query)
    graph.add_node("finalize_query", finalize_query_node)

    graph.set_entry_point("start_query_job")

    # Fan-out: vector search and graph retrieval are independent — run them
    # in parallel rather than sequentially, then fan back in. synthesize_answer
    # only runs once BOTH have completed (LangGraph waits for all incoming
    # edges before running a node).
    graph.add_edge("start_query_job", "vector_search")
    graph.add_edge("start_query_job", "graph_retrieve")
    graph.add_edge("vector_search", "synthesize_answer")
    graph.add_edge("graph_retrieve", "synthesize_answer")

    graph.add_edge("synthesize_answer", "human_approval")
    graph.add_edge("human_approval", "finalize_query")
    graph.add_edge("finalize_query", END)

    return graph.compile(checkpointer=MemorySaver())


# ---------------------------------------------------------------------------
# Demo usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # --- Ingestion demo -----------------------------------------------
    ingestion_app = build_ingestion_graph()
    ingestion_config = {"configurable": {"thread_id": "ingest-demo-1"}}

    result = ingestion_app.invoke(
        {
            "document_name": "abc_ltd_msa.pdf",
            "contract_name": "ABC Ltd Master Service Agreement",
            "vendor_name": "ABC Ltd.",
            "text_chunks": [
                {"text": "This Agreement shall be governed by the laws of India...",
                 "page_start": 3, "page_end": 3, "section": "Governing Law"},
                # ... in practice, pass the text_chunks produced by
                # pdf_rag_pipeline.chunk_pages() / langgraph_pdf_rag_agent.py here.
            ],
        },
        config=ingestion_config,
    )
    print("\n--- PAUSED FOR HUMAN APPROVAL ---")
    print(result["__interrupt__"])

    # Simulates a reviewer approving after inspecting the summary above.
    result = ingestion_app.invoke(
        Command(resume={"approved": True, "reviewer": "jane.doe", "comments": "Looks correct."}),
        config=ingestion_config,
    )
    print("\n--- INGESTION FINAL STATE ---")
    print(result["status"])

    # --- Query demo ------------------------------------------------------
    query_app = build_query_graph()
    query_config = {"configurable": {"thread_id": "query-demo-1"}}

    result = query_app.invoke(
        {
            "question": "Show all contracts where ABC Ltd. is the vendor, the same clause "
                        "appears in another contract, and that clause has been interpreted "
                        "by multiple judgments.",
            "collection_name": "abc_ltd_msa_pdf",
        },
        config=query_config,
    )
    print("\n--- PAUSED FOR HUMAN APPROVAL ---")
    print(result["__interrupt__"])

    # Simulates a reviewer approving the LLM's draft answer as-is.
    result = query_app.invoke(
        Command(resume={"approved": True, "reviewer": "jane.doe", "edited_answer": None}),
        config=query_config,
    )
    print("\n--- FINAL ANSWER ---")
    print(result["final_answer"])
