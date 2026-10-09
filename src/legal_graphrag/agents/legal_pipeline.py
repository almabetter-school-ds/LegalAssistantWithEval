"""
Legal document intelligence pipeline — router + specialist workers +
verifier + synthesizer, as a single LangGraph state machine.

    ┌─────────────┐
    │  start_job  │  creates a QueryJob in Neo4j (audit trail root)
    └──────┬──────┘
           ▼
    ┌─────────────┐
    │   router    │  classifies the question -> "hybrid" | "graph" | "direct"
    └──────┬──────┘
           │ conditional edge on state["route"]
           ├─── "hybrid" ──────► ┌────────────────────┐
           │                     │ hybrid_search_agent │  dense+BM25+RRF+rerank,
           │                     └──────────┬──────────┘  metadata-filtered
           │                                │
           ├─── "graph" ───────► ┌────────────────────┐
           │                     │  graph_rag_agent    │  template-first / text-to-Cypher
           │                     └──────────┬──────────┘  over Neo4j (read-only guarded)
           │                                │
           └─── "direct" ──────────────────►│  (skip retrieval; reuse evidence already
                                             │   present in state, e.g. a follow-up turn)
                                             ▼
                                     ┌───────────────┐
                                     │    auditor    │  EvidenceChecker: is the evidence
                                     └───────┬───────┘  sufficient/consistent? (LLM)
                                             ▼
                              ┌───────────────────────────┐
                              │ human_evidence_checkpoint  │  interrupt() — a human signs
                              └──────────────┬─────────────┘ off on evidence BEFORE generation
                        not approved ────────┤─── approved
                              ▼               ▼
                    ┌──────────────────┐  ┌───────────────┐
                    │ evidence_rejected │  │  synthesizer  │  AnswerAgent/Adjudicator:
                    │   (short-circuit) │  └───────┬───────┘  summary + risk + citations
                    └─────────┬────────┘          ▼
                              │           ┌────────────────────────┐
                              │      ┌───►│ human_answer_checkpoint │  interrupt() — 3-way
                              │      │    └────────────┬───────────┘  reviewer decision
                              │      │                 │
                              │      │     ┌───────────┼────────────────┐
                              │      │  "approve"   "revise"        "reject"
                              │      │     ▼             ▼                ▼
                              │      │ ┌────────┐ ┌───────────────┐ ┌─────────────────┐
                              │      │ │finalize│ │ revise_answer │ │ answer_rejected  │
                              │      │ └───┬────┘ └───────┬───────┘ └────────┬─────────┘
                              │      │     │              │                  │
                              │      │     │              └── loops back ────┘ (until reviewer
                              │      │     │                  to human_answer_checkpoint    picks approve/reject,
                              │      │     │                  ("revise" cannot loop forever — capped        or hits the revision cap)
                              │      │     │                   by max_answer_revisions)
                              │      │     │ conditional: route=="graph" and approved?
                              │      │     ├── yes ──► ┌──────────────┐
                              │      │     │           │ graph_update │  ONLY reached via "approve" —
                              │      │     │           └──────┬───────┘  never on a "revise" round
                              │      │     └── no ──────────┐ │
                              ▼      ▼                      ▼ ▼
                             END ◄──────────────────────────END

Two human-in-the-loop checkpoints:
  1. Evidence checkpoint — a human validates retrieval quality BEFORE the
     LLM is allowed to write anything. Rejecting here stops the pipeline
     before generation ever happens.
  2. Answer checkpoint — a THREE-WAY decision, not a binary approve/reject:
       - "approve": the answer is finalized (and, for graph-path answers,
         written back into Neo4j via graph_update) and returned to the caller.
       - "revise": the reviewer's comments are handed to the LLM
         (revise_legal_answer), which reasons over that feedback against the
         SAME verified evidence — no new retrieval, no fabricated evidence —
         and produces a revised draft. Control loops back to
         human_answer_checkpoint for another round. This repeats until the
         reviewer picks "approve" or "reject", or the pipeline hits
         max_answer_revisions (a safety valve against an infinite loop).
       - "reject": the pipeline stops immediately, final_answer is None.
     Critically, graph_update is only reachable through the "approve" branch
     of finalize — a "revise" round never touches Neo4j, no matter how many
     iterations it takes to land on an approved answer.
"""

from __future__ import annotations

import uuid
from typing import Optional, TypedDict

from langgraph.graph import StateGraph, END
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import MemorySaver

from ..resources import get_store
from ..graphrag.extraction import generate_cypher
from ..graphrag.langgraph_agent import match_template  # the vetted "vendor/same-clause/judgments" template
from ..retrieval.hybrid_search import hybrid_search
from ..tracing import traceable
from .prompts import (
    classify_route,
    verify_evidence,
    synthesize_legal_answer,
    revise_legal_answer,
    DEFAULT_ALPHA,
)


# ===========================================================================
# Shared state
# ===========================================================================

class LegalAgentState(TypedDict, total=False):
    # --- input ---
    question: str
    collection_name: str
    metadata_filter: Optional[dict]

    # --- router ---
    query_job_id: str
    route: str                      # "hybrid" | "graph" | "direct"
    route_reasoning: str
    alpha: float                    # dense/sparse blend weight for HybridSearchAgent, chosen by the router
    force_route: Optional[str]      # eval-only override (see evaluation/ragas_eval.py): when set, skips
                                     # the router's own classification and pins the route directly — used
                                     # for CUAD/RAGAS batch evaluation, where only the vector store is
                                     # populated (no clause graph), so "graph" routing would just return
                                     # empty hits and unfairly tank retrieval metrics on those questions.

    # --- specialist retrieval ---
    hybrid_hits: list[dict]
    graph_hits: list[dict]
    cypher_used: Optional[str]
    cypher_source: Optional[str]    # "template" or "generated"

    # --- verification ---
    evidence_verdict: dict
    evidence_decision: dict         # human decision at the evidence checkpoint

    # --- synthesis ---
    draft_answer: str
    draft_evidence: str
    draft_document: str
    draft_source_section: Optional[str]
    draft_source_page: Optional[str]
    draft_citations: list[str]
    draft_risk_level: Optional[str]
    draft_has_uncertainty: bool
    draft_confidence: str            # "High" | "Medium" | "Low" — see prompts.derive_confidence
    answer_decision: dict           # human decision at the answer checkpoint (this round)
    answer_revision_count: int      # how many "revise" rounds have happened so far
    max_answer_revisions: int       # safety valve: force answer_rejected past this many rounds

    # --- output ---
    final_answer: Optional[str]
    final_structured_answer: Optional[dict]   # full card payload for output_formatting.py:
                                               # {answer, evidence, document, source_section, source_page,
                                               #  citations, risk_level, has_uncertainty, confidence}
    status: str


DEFAULT_MAX_ANSWER_REVISIONS = 3


# ===========================================================================
# 0. Job bootstrap
# ===========================================================================

@traceable(name="pipeline.start_job", run_type="chain")
def start_job_node(state: LegalAgentState) -> dict:
    store = get_store()
    job_id = str(uuid.uuid4())
    store.create_query_job(job_id, state["question"])
    store.write_audit_record(job_id, "system", "query_received", state["question"])
    print(f"[start_job] job_id={job_id}")
    return {
        "query_job_id": job_id,
        # Defaults so downstream nodes never KeyError on a path that was skipped.
        "hybrid_hits": state.get("hybrid_hits", []),
        "graph_hits": state.get("graph_hits", []),
        "cypher_used": None,
        "cypher_source": None,
        "answer_revision_count": 0,
        "max_answer_revisions": state.get("max_answer_revisions", DEFAULT_MAX_ANSWER_REVISIONS),
    }


# ===========================================================================
# 1. Router agent
# ===========================================================================

@traceable(name="router", run_type="chain")
def router_node(state: LegalAgentState) -> dict:
    route, reasoning, alpha = classify_route(state["question"])

    forced = state.get("force_route")
    if forced:
        reasoning = f"[force_route override: {forced}] originally classified as {route}: {reasoning}"
        route = forced

    store = get_store()
    store.write_audit_record(
        state["query_job_id"], "router", "route_selected",
        f"route={route} alpha={alpha}: {reasoning}",
    )
    print(f"[router] route={route} alpha={alpha} ({reasoning})")
    return {"route": route, "route_reasoning": reasoning, "alpha": alpha}


def route_after_router(state: LegalAgentState) -> str:
    return {"hybrid": "hybrid_search_agent", "graph": "graph_rag_agent", "direct": "auditor"}[state["route"]]


# ===========================================================================
# 2a. HybridSearchAgent — dense + lexical retrieval, best for clause text
#     and citation matching (see retrieval/hybrid_search.py for the mechanics)
# ===========================================================================

@traceable(name="pipeline.hybrid_search_agent_node", run_type="chain")
def hybrid_search_agent_node(state: LegalAgentState) -> dict:
    hits = hybrid_search(
        collection_name=state["collection_name"],
        query=state["question"],
        metadata_filter=state.get("metadata_filter"),
        alpha=state.get("alpha", DEFAULT_ALPHA),
    )
    store = get_store()
    store.write_audit_record(
        state["query_job_id"], "hybrid_search_agent", "retrieval_completed",
        f"hits={len(hits)} alpha={state.get('alpha', DEFAULT_ALPHA)}",
    )
    print(f"[hybrid_search_agent] {len(hits)} hits (alpha={state.get('alpha', DEFAULT_ALPHA)})")
    return {"hybrid_hits": hits}


# ===========================================================================
# 2b. GraphRAGAgent — traverses entities/relationships for multi-hop
#     reasoning (precedent chains, cross-contract clause matches, conflicts)
# ===========================================================================

@traceable(name="pipeline.graph_rag_agent_node", run_type="chain")
def graph_rag_agent_node(state: LegalAgentState) -> dict:
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
    store.write_audit_record(
        state["query_job_id"], "graph_rag_agent", "retrieval_completed",
        f"source={source} hits={len(hits)}",
    )
    print(f"[graph_rag_agent] source={source} hits={len(hits)}")
    return {"cypher_used": cypher, "cypher_source": source, "graph_hits": hits}


# ===========================================================================
# 3. Auditor / EvidenceChecker — verifies evidence BEFORE any generation
# ===========================================================================

@traceable(name="pipeline.auditor_node", run_type="chain")
def auditor_node(state: LegalAgentState) -> dict:
    verdict = verify_evidence(state["question"], state["hybrid_hits"], state["graph_hits"])
    store = get_store()
    store.write_audit_record(
        state["query_job_id"], "auditor", "evidence_verified",
        f"sufficient={verdict.get('sufficient')}: {verdict.get('reasoning', '')}",
    )
    print(f"[auditor] sufficient={verdict.get('sufficient')}")
    return {"evidence_verdict": verdict}


@traceable(name="checkpoint.human_evidence", run_type="chain")
def human_evidence_checkpoint_node(state: LegalAgentState) -> dict:
    """
    First human-in-the-loop checkpoint. A reviewer looks at the auditor's
    verdict AND the raw evidence, and decides whether it's adequate to
    generate an answer from — this is what stops a weak/partial retrieval
    from ever reaching the LLM that writes the final answer.
    """
    payload = {
        "type": "evidence_approval_request",
        "query_job_id": state["query_job_id"],
        "question": state["question"],
        "route": state["route"],
        "evidence_verdict": state["evidence_verdict"],
        "hybrid_hits": state["hybrid_hits"],
        "graph_hits": state["graph_hits"],
        "message": "Resume with {'proceed': bool, 'reviewer': str, 'comments': str|None}.",
    }
    decision = interrupt(payload)
    return {"evidence_decision": decision}


def route_after_evidence_checkpoint(state: LegalAgentState) -> str:
    return "synthesizer" if state["evidence_decision"].get("proceed") else "evidence_rejected"


def evidence_rejected_node(state: LegalAgentState) -> dict:
    """Short-circuit: evidence was rejected, so no answer is ever generated."""
    store = get_store()
    decision = state["evidence_decision"]
    store.create_reviewer_decision(
        state["query_job_id"], approved=False, reviewer=decision.get("reviewer", "unknown"),
        comments=decision.get("comments"),
    )
    store.update_job_status(state["query_job_id"], "evidence_rejected")
    store.write_audit_record(
        state["query_job_id"], decision.get("reviewer", "unknown"), "evidence_rejected",
        decision.get("comments") or "no comments",
    )
    print(f"[evidence_rejected] job {state['query_job_id']}")
    return {"final_answer": None, "final_structured_answer": None, "status": "evidence_rejected"}


# ===========================================================================
# 4. Synthesizer / AnswerAgent (Adjudicator) — writes the answer from
#    VERIFIED evidence only: summary, risk, citations, uncertainty
# ===========================================================================

@traceable(name="pipeline.synthesizer_node", run_type="chain")
def synthesizer_node(state: LegalAgentState) -> dict:
    result = synthesize_legal_answer(
        state["question"], state["hybrid_hits"], state["graph_hits"], state["evidence_verdict"]
    )
    store = get_store()
    store.write_audit_record(
        state["query_job_id"], "synthesizer", "draft_answer_generated", result["answer"][:500]
    )
    print(f"[synthesizer] draft ready ({len(result['answer'])} chars, risk={result.get('risk_level')}, "
          f"confidence={result.get('confidence')})")
    return {
        "draft_answer": result["answer"],
        "draft_evidence": result.get("evidence", ""),
        "draft_document": result.get("document", ""),
        "draft_source_section": result.get("source_section"),
        "draft_source_page": result.get("source_page"),
        "draft_citations": result.get("citations", []),
        "draft_risk_level": result.get("risk_level"),
        "draft_has_uncertainty": result.get("has_uncertainty", False),
        "draft_confidence": result.get("confidence", "Low"),
    }


# ===========================================================================
# 5. Second human-in-the-loop checkpoint — THREE-WAY decision on the answer
# ===========================================================================

@traceable(name="checkpoint.human_answer", run_type="chain")
def human_answer_checkpoint_node(state: LegalAgentState) -> dict:
    """
    Pauses for a human reviewer to decide one of three things about the
    current draft answer:
      - "approve": finalize it as-is (or with an optional edited_answer override).
      - "revise": hand `comments` to the LLM to reason over and produce a
        revised draft — loops back to this same checkpoint for another look.
      - "reject": stop the pipeline; no answer is returned.
    """
    payload = {
        "type": "answer_approval_request",
        "query_job_id": state["query_job_id"],
        "question": state["question"],
        "draft_answer": state["draft_answer"],
        "draft_evidence": state.get("draft_evidence", ""),
        "draft_document": state.get("draft_document", ""),
        "draft_source_section": state.get("draft_source_section"),
        "draft_source_page": state.get("draft_source_page"),
        "draft_citations": state.get("draft_citations", []),
        "draft_risk_level": state.get("draft_risk_level"),
        "draft_has_uncertainty": state.get("draft_has_uncertainty", False),
        "draft_confidence": state.get("draft_confidence", "Low"),
        "evidence_verdict": state["evidence_verdict"],
        "revision_round": state.get("answer_revision_count", 0),
        "max_answer_revisions": state.get("max_answer_revisions", DEFAULT_MAX_ANSWER_REVISIONS),
        "message": "Resume with {'action': 'approve'|'revise'|'reject', 'reviewer': str, "
                    "'comments': str|None, 'edited_answer': str|None}. "
                    "'comments' is REQUIRED when action is 'revise' — it's what the LLM "
                    "reasons over to produce the next draft. 'edited_answer', if given with "
                    "'approve', overrides the draft text verbatim instead of using it as-is.",
    }
    decision = interrupt(payload)
    return {"answer_decision": decision}


def route_after_answer_checkpoint(state: LegalAgentState) -> str:
    decision = state["answer_decision"]
    action = decision.get("action")

    if action == "approve":
        return "finalize"

    if action == "revise":
        if state.get("answer_revision_count", 0) >= state.get("max_answer_revisions", DEFAULT_MAX_ANSWER_REVISIONS):
            # Safety valve: too many rounds without a resolution. Stop rather
            # than loop forever — treated as a rejection, not a silent approval.
            return "answer_rejected"
        return "revise_answer"

    # "reject", or any unrecognized/malformed action: never silently finalize
    # an answer nobody actually approved.
    return "answer_rejected"


@traceable(name="pipeline.revise_answer_node", run_type="chain")
def revise_answer_node(state: LegalAgentState) -> dict:
    """
    The "make changes" branch: hands the reviewer's comments to the LLM,
    which reasons over that feedback against the SAME verified evidence
    (no new retrieval, no fabricated evidence) and produces a revised draft.
    Loops back to human_answer_checkpoint for another round. Nothing is
    written to Neo4j here — record_answered_question only ever runs after
    finalize, which is only reachable via "approve".
    """
    decision = state["answer_decision"]
    feedback = decision.get("comments") or ""
    revision_round = state.get("answer_revision_count", 0) + 1

    result = revise_legal_answer(
        question=state["question"],
        previous_answer=state["draft_answer"],
        hybrid_hits=state["hybrid_hits"],
        graph_hits=state["graph_hits"],
        evidence_verdict=state["evidence_verdict"],
        reviewer_feedback=feedback,
    )

    store = get_store()
    store.write_audit_record(
        state["query_job_id"], decision.get("reviewer", "unknown"), "answer_revision_requested",
        f"round={revision_round} feedback={feedback[:300]}",
    )
    print(f"[revise_answer] round {revision_round}: incorporated reviewer feedback")

    return {
        "draft_answer": result["answer"],
        "draft_evidence": result.get("evidence", ""),
        "draft_document": result.get("document", ""),
        "draft_source_section": result.get("source_section"),
        "draft_source_page": result.get("source_page"),
        "draft_citations": result.get("citations", []),
        "draft_risk_level": result.get("risk_level"),
        "draft_has_uncertainty": result.get("has_uncertainty", False),
        "draft_confidence": result.get("confidence", "Low"),
        "answer_revision_count": revision_round,
    }


def answer_rejected_node(state: LegalAgentState) -> dict:
    """Terminal rejection of the answer — no final_answer, no graph write-back."""
    store = get_store()
    decision = state["answer_decision"]
    reviewer = decision.get("reviewer", "unknown")
    reason = decision.get("comments") or (
        "max_answer_revisions exceeded" if decision.get("action") == "revise" else "no comments"
    )

    store.create_reviewer_decision(state["query_job_id"], approved=False, reviewer=reviewer, comments=reason)
    store.update_job_status(state["query_job_id"], "rejected")
    store.write_audit_record(state["query_job_id"], reviewer, "answer_rejected", reason)

    print(f"[answer_rejected] job {state['query_job_id']}: {reason}")
    return {"final_answer": None, "final_structured_answer": None, "status": "rejected"}


# ===========================================================================
# 6. Finalize + optional graph update — ONLY reached via the "approve" action
# ===========================================================================

@traceable(name="pipeline.finalize", run_type="chain")
def finalize_node(state: LegalAgentState) -> dict:
    store = get_store()
    decision = state["answer_decision"]
    reviewer = decision.get("reviewer", "unknown")
    edited = decision.get("edited_answer")
    final_answer = edited or state["draft_answer"]

    store.create_reviewer_decision(state["query_job_id"], approved=True, reviewer=reviewer,
                                    comments=decision.get("comments"))
    store.write_audit_record(state["query_job_id"], reviewer, "review_decision", "approved")
    store.store_query_answer(state["query_job_id"], final_answer)
    store.update_job_status(state["query_job_id"], "answered")

    # If the reviewer overrode the answer text (edited_answer), the rest of
    # the structured card (evidence/document/source/confidence) still
    # reflects the LAST draft the pipeline actually produced — only the
    # headline "answer" field changes on an edit, since evidence/source
    # attribution wasn't something the reviewer rewrote.
    final_structured_answer = {
        "answer": final_answer,
        "evidence": state.get("draft_evidence", ""),
        "document": state.get("draft_document", ""),
        "source_section": state.get("draft_source_section"),
        "source_page": state.get("draft_source_page"),
        "citations": state.get("draft_citations", []),
        "risk_level": state.get("draft_risk_level"),
        "has_uncertainty": state.get("draft_has_uncertainty", False),
        "confidence": state.get("draft_confidence", "Low"),
    }

    print(f"[finalize] job {state['query_job_id']} -> answered "
          f"(after {state.get('answer_revision_count', 0)} revision round(s))")
    return {"final_answer": final_answer, "final_structured_answer": final_structured_answer, "status": "answered"}


def route_after_finalize(state: LegalAgentState) -> str:
    """
    Graph update only makes sense for approved, graph-sourced answers —
    that's the only case where there are Clause nodes worth citing back.
    finalize (and therefore this) is only reachable via "approve", so a
    "revise" round can never trigger a graph update no matter how many
    iterations it takes to get here.
    """
    if state["status"] == "answered" and state["route"] == "graph" and state["graph_hits"]:
        return "graph_update"
    return END


@traceable(name="pipeline.graph_update", run_type="chain")
def graph_update_node(state: LegalAgentState) -> dict:
    """
    Optional post-approval write-back: records this reviewed Q&A as an
    AnsweredQuestion node citing the Clause nodes it drew on (see
    Neo4jGraphStore.record_answered_question). Runs ONLY after human
    approval — the graph is never updated based on an unreviewed, or
    still-being-revised, answer.
    """
    store = get_store()
    cited_clause_ids = sorted({
        row["clause_id"] for row in state["graph_hits"] if isinstance(row, dict) and row.get("clause_id")
    })
    answered_id = store.record_answered_question(
        state["query_job_id"], state["question"], state["final_answer"], cited_clause_ids
    )
    store.write_audit_record(
        state["query_job_id"], "system", "graph_updated",
        f"answered_id={answered_id} cited_clauses={len(cited_clause_ids)}",
    )
    print(f"[graph_update] recorded AnsweredQuestion {answered_id} citing {len(cited_clause_ids)} clause(s)")
    return {}


# ===========================================================================
# Graph assembly
# ===========================================================================

def build_legal_agent_graph():
    graph = StateGraph(LegalAgentState)

    graph.add_node("start_job", start_job_node)
    graph.add_node("router", router_node)
    graph.add_node("hybrid_search_agent", hybrid_search_agent_node)
    graph.add_node("graph_rag_agent", graph_rag_agent_node)
    graph.add_node("auditor", auditor_node)
    graph.add_node("human_evidence_checkpoint", human_evidence_checkpoint_node)
    graph.add_node("evidence_rejected", evidence_rejected_node)
    graph.add_node("synthesizer", synthesizer_node)
    graph.add_node("human_answer_checkpoint", human_answer_checkpoint_node)
    graph.add_node("revise_answer", revise_answer_node)
    graph.add_node("answer_rejected", answer_rejected_node)
    graph.add_node("finalize", finalize_node)
    graph.add_node("graph_update", graph_update_node)

    graph.set_entry_point("start_job")
    graph.add_edge("start_job", "router")

    # Router fans out to exactly one of the three specialist paths.
    graph.add_conditional_edges("router", route_after_router, {
        "hybrid_search_agent": "hybrid_search_agent",
        "graph_rag_agent": "graph_rag_agent",
        "auditor": "auditor",  # "direct" path: skip retrieval entirely
    })
    graph.add_edge("hybrid_search_agent", "auditor")
    graph.add_edge("graph_rag_agent", "auditor")

    # Evidence checkpoint #1: human must approve before generation happens.
    graph.add_edge("auditor", "human_evidence_checkpoint")
    graph.add_conditional_edges("human_evidence_checkpoint", route_after_evidence_checkpoint, {
        "synthesizer": "synthesizer",
        "evidence_rejected": "evidence_rejected",
    })
    graph.add_edge("evidence_rejected", END)

    # Answer checkpoint #2: three-way decision, with "revise" looping back
    # through revise_answer to human_answer_checkpoint until the reviewer
    # picks approve/reject (or the revision cap forces a rejection).
    graph.add_edge("synthesizer", "human_answer_checkpoint")
    graph.add_conditional_edges("human_answer_checkpoint", route_after_answer_checkpoint, {
        "finalize": "finalize",
        "revise_answer": "revise_answer",
        "answer_rejected": "answer_rejected",
    })
    graph.add_edge("revise_answer", "human_answer_checkpoint")
    graph.add_edge("answer_rejected", END)

    # Optional graph write-back, only for approved graph-path answers.
    graph.add_conditional_edges("finalize", route_after_finalize, {
        "graph_update": "graph_update",
        END: END,
    })
    graph.add_edge("graph_update", END)

    # A checkpointer is required for interrupt()/resume to work at all — see
    # graphrag/langgraph_agent.py's note on MemorySaver vs. a persistent
    # checkpointer for production use.
    return graph.compile(checkpointer=MemorySaver())


# ---------------------------------------------------------------------------
# Demo usage — shows all three answer-checkpoint outcomes: one "revise"
# round, then an "approve".
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = build_legal_agent_graph()
    config = {"configurable": {"thread_id": "legal-agent-demo-1"}}

    result = app.invoke(
        {
            "question": "Show all contracts where ABC Ltd. is the vendor, the same clause "
                        "appears in another contract, and that clause has been interpreted "
                        "by multiple judgments.",
            "collection_name": "abc_ltd_msa_pdf",
        },
        config=config,
    )

    print("\n--- PAUSED: EVIDENCE CHECKPOINT ---")
    print(result["__interrupt__"])

    result = app.invoke(
        Command(resume={"proceed": True, "reviewer": "jane.doe", "comments": "Evidence looks solid."}),
        config=config,
    )

    print("\n--- PAUSED: ANSWER CHECKPOINT (round 1) ---")
    print(result["__interrupt__"])

    # Reviewer asks for a change instead of approving outright.
    result = app.invoke(
        Command(resume={
            "action": "revise",
            "reviewer": "jane.doe",
            "comments": "Missing Section 4.2 termination clauses — please check whether "
                        "the termination notice period is also part of the conflict.",
        }),
        config=config,
    )

    print("\n--- PAUSED: ANSWER CHECKPOINT (round 2, after revision) ---")
    print(result["__interrupt__"])

    # Reviewer approves the revised answer.
    result = app.invoke(
        Command(resume={"action": "approve", "reviewer": "jane.doe"}),
        config=config,
    )

    print("\n--- FINAL ANSWER ---")
    print(result["final_answer"])
