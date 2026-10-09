"""
Wraps the UNMODIFIED agents/legal_pipeline.build_legal_agent_graph() as
HTTP endpoints. No changes to that module — this only calls it, exactly
like scripts/run_demo.py's interactive CLI does.

Each thread_id is one question working through the pipeline's two human
checkpoints (evidence, then answer). POST /start begins a thread and runs
it to the first checkpoint. The two decision endpoints resume via
Command(resume=...) — same mechanism, same payload shapes as
scripts/run_demo.py.
"""

from __future__ import annotations

import uuid
from functools import lru_cache

from fastapi import APIRouter, HTTPException
from langgraph.types import Command

from ..agents.legal_pipeline import build_legal_agent_graph
from .jobs import get_query_thread, list_query_threads, register_query_thread
from .schemas import (
    AnswerDecisionRequest,
    EvidenceDecisionRequest,
    QueryListItem,
    QueryStartRequest,
    QueryStateResponse,
    RetrievedChunk,
    TechnicalDetails,
)

router = APIRouter(prefix="/api/query", tags=["query"])


@lru_cache(maxsize=1)
def _app():
    """One compiled graph (and its MemorySaver checkpointer) per process, shared across requests."""
    return build_legal_agent_graph()


def _config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def _extract_interrupt(state: dict) -> tuple[str | None, dict | None]:
    """state (as returned by app.invoke()) carries pending interrupts under "__interrupt__"."""
    interrupts = state.get("__interrupt__")
    if not interrupts:
        return None, None
    payload = interrupts[0].value
    return payload.get("type"), payload


def _chunks(hits: list[dict]) -> list[RetrievedChunk]:
    out = []
    for h in hits or []:
        meta = h.get("metadata", {}) or {}
        score = h.get("rerank_score", h.get("dense_distance", h.get("bm25_score")))
        out.append(RetrievedChunk(
            text=h.get("text", ""),
            document_name=meta.get("document_name"),
            page_start=meta.get("page_start"),
            page_end=meta.get("page_end"),
            section=meta.get("section"),
            score=score,
        ))
    return out


def _technical(state: dict) -> TechnicalDetails:
    return TechnicalDetails(
        route=state.get("route"),
        alpha=state.get("alpha"),
        route_reasoning=state.get("route_reasoning"),
        hybrid_hits=_chunks(state.get("hybrid_hits", [])),
        graph_hits=_chunks(state.get("graph_hits", [])),
        cypher_used=state.get("cypher_used"),
        cypher_source=state.get("cypher_source"),
        evidence_verdict=state.get("evidence_verdict", {}) or {},
        answer_revision_count=state.get("answer_revision_count", 0),
        citations=state.get("draft_citations", []) or [],
        risk_level=state.get("draft_risk_level"),
        has_uncertainty=state.get("draft_has_uncertainty", False),
    )


def _response(thread_id: str, question: str, state: dict) -> QueryStateResponse:
    interrupt_type, interrupt_payload = _extract_interrupt(state)
    if interrupt_type == "evidence_approval_request":
        status = "awaiting_evidence_approval"
    elif interrupt_type == "answer_approval_request":
        status = "awaiting_answer_approval"
    else:
        status = state.get("status", "unknown")

    return QueryStateResponse(
        thread_id=thread_id,
        question=question,
        status=status,
        interrupt_type=interrupt_type,
        interrupt_payload=interrupt_payload,
        draft_answer=state.get("draft_answer"),
        final_answer=state.get("final_answer"),
        technical=_technical(state) if state.get("route") else None,
    )


@router.post("/start", response_model=QueryStateResponse)
def start_query(req: QueryStartRequest) -> QueryStateResponse:
    thread_id = str(uuid.uuid4())
    register_query_thread(thread_id, req.question, req.collection_name)
    state = _app().invoke(
        {"question": req.question, "collection_name": req.collection_name, "metadata_filter": req.metadata_filter},
        config=_config(thread_id),
    )
    return _response(thread_id, req.question, state)


@router.get("", response_model=list[QueryListItem])
def list_threads() -> list[QueryListItem]:
    threads = list_query_threads()
    out = []
    for t in threads:
        snapshot = _app().get_state(_config(t["thread_id"]))
        status = (snapshot.values or {}).get("status") or ("awaiting_review" if snapshot.next else "unknown")
        out.append(QueryListItem(thread_id=t["thread_id"], question=t["question"],
                                  collection_name=t["collection_name"], status=status,
                                  created_at=t["created_at"]))
    return out


@router.get("/{thread_id}", response_model=QueryStateResponse)
def get_query_status(thread_id: str) -> QueryStateResponse:
    meta = get_query_thread(thread_id)
    if meta is None:
        raise HTTPException(status_code=404, detail="Unknown thread_id")

    snapshot = _app().get_state(_config(thread_id))
    state = dict(snapshot.values or {})

    if "__interrupt__" not in state and snapshot.tasks:
        for task in snapshot.tasks:
            if getattr(task, "interrupts", None):
                state["__interrupt__"] = list(task.interrupts)
                break

    return _response(thread_id, meta["question"], state)


@router.post("/{thread_id}/evidence-decision", response_model=QueryStateResponse)
def submit_evidence_decision(thread_id: str, decision: EvidenceDecisionRequest) -> QueryStateResponse:
    meta = get_query_thread(thread_id)
    if meta is None:
        raise HTTPException(status_code=404, detail="Unknown thread_id")
    state = _app().invoke(Command(resume=decision.model_dump()), config=_config(thread_id))
    return _response(thread_id, meta["question"], state)


@router.post("/{thread_id}/answer-decision", response_model=QueryStateResponse)
def submit_answer_decision(thread_id: str, decision: AnswerDecisionRequest) -> QueryStateResponse:
    meta = get_query_thread(thread_id)
    if meta is None:
        raise HTTPException(status_code=404, detail="Unknown thread_id")
    if decision.action not in ("approve", "revise", "reject"):
        raise HTTPException(status_code=422, detail="action must be one of: approve, revise, reject")
    state = _app().invoke(Command(resume=decision.model_dump()), config=_config(thread_id))
    return _response(thread_id, meta["question"], state)
