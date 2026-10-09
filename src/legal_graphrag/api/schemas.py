"""Request/response schemas — field names match the UNMODIFIED LegalAgentState in agents/legal_pipeline.py exactly."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Query pipeline (agents/legal_pipeline.py — unmodified)
# ---------------------------------------------------------------------------

class QueryStartRequest(BaseModel):
    question: str
    collection_name: str
    metadata_filter: Optional[dict] = None


class EvidenceDecisionRequest(BaseModel):
    proceed: bool
    reviewer: str = "web-ui"
    comments: Optional[str] = None


class AnswerDecisionRequest(BaseModel):
    action: str  # "approve" | "revise" | "reject"
    reviewer: str = "web-ui"
    comments: Optional[str] = None
    edited_answer: Optional[str] = None


class RetrievedChunk(BaseModel):
    text: str
    document_name: Optional[str] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    section: Optional[str] = None
    score: Optional[float] = None


class TechnicalDetails(BaseModel):
    """Everything the frontend renders inside collapsible sections — never the headline output."""
    route: Optional[str] = None
    alpha: Optional[float] = None
    route_reasoning: Optional[str] = None
    hybrid_hits: list[RetrievedChunk] = Field(default_factory=list)
    graph_hits: list[RetrievedChunk] = Field(default_factory=list)
    cypher_used: Optional[str] = None
    cypher_source: Optional[str] = None
    evidence_verdict: dict = Field(default_factory=dict)
    answer_revision_count: int = 0
    citations: list[str] = Field(default_factory=list)
    risk_level: Optional[str] = None
    has_uncertainty: bool = False


class QueryStateResponse(BaseModel):
    thread_id: str
    question: str
    status: str  # "awaiting_evidence_approval" | "awaiting_answer_approval" | "answered" | "rejected" | "evidence_rejected"
    interrupt_type: Optional[str] = None      # "evidence_approval_request" | "answer_approval_request" | None
    interrupt_payload: Optional[dict] = None  # raw payload — rendered inside a collapsible in the frontend
    draft_answer: Optional[str] = None        # plain text, per SynthesizedAnswer.answer
    final_answer: Optional[str] = None        # plain text, per LegalAgentState.final_answer
    technical: Optional[TechnicalDetails] = None


class QueryListItem(BaseModel):
    thread_id: str
    question: str
    collection_name: str
    status: str
    created_at: str


# ---------------------------------------------------------------------------
# Ingestion (ingestion/pdf_pipeline.py — unmodified)
# ---------------------------------------------------------------------------

class IngestJobResponse(BaseModel):
    job_id: str
    status: str  # "pending" | "running" | "done" | "error"
    filename: str
    collection_name: Optional[str] = None
    result: Optional[dict] = None
    error: Optional[str] = None


class CollectionsResponse(BaseModel):
    collections: list[str]
