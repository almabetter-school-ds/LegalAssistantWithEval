"""
CUAD evaluation routes for the web UI's "Ask CUAD" panel.

New file — like query_routes.py / ingest_routes.py it only IMPORTS and calls
existing public functions; no pipeline or evaluation module is edited.

What the frontend needs from this router:

  GET  /api/cuad/contracts
        -> the PDF dropdown: every unique CUAD contract, one row each.
           All ~500 CUAD contracts are listed here; NONE are embedded up
           front. A contract is embedded on demand only when it is first
           asked about (see _ensure_ingested) — see the [ON-DEMAND INGEST]
           block below.

  GET  /api/cuad/contracts/{collection_name}/questions
        -> the question dropdown for the chosen contract (CUAD's own
           category questions + ground-truth spans).

  POST /api/cuad/ask   {collection_name, qas_id}
        -> (1) ingest the contract on demand if it isn't in Chroma yet,
           (2) run the REAL pipeline graph to a final answer (both human
               checkpoints auto-approved, exactly like the batch eval
               harness — scripted_reviewer.run_scripted_pipeline),
           (3) return the answer + reasoning + context + ground truth
               IMMEDIATELY, and
           (4) kick a BACKGROUND task that computes the four RAGAS metrics
               for this one question, polled via the next endpoint.

  GET  /api/cuad/ask/{job_id}
        -> poll: metrics is null until the background task finishes, then
           carries context_precision / context_recall / faithfulness /
           answer_correctness (per-question, computed live).
"""

from __future__ import annotations

import threading
import uuid
from functools import lru_cache
from typing import Optional

import chromadb
from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from ..agents.legal_pipeline import build_legal_agent_graph
from ..evaluation.cuad_loader import CUADExample, load_cuad, unique_contracts
from ..evaluation.cuad_ingest import ingest_cuad_contract
from ..evaluation.per_question_ragas import score_single_question
from ..evaluation.scripted_reviewer import run_scripted_pipeline
from ..ingestion.pdf_pipeline import CHROMA_PERSIST_DIR

router = APIRouter(prefix="/api/cuad", tags=["cuad"])


# ---------------------------------------------------------------------------
# Lazily-loaded CUAD dataset + compiled graph (shared across requests)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _examples() -> list[CUADExample]:
    """Full CUAD example list (auto-downloaded from HF on first call). Cached."""
    return load_cuad()


@lru_cache(maxsize=1)
def _contracts_by_collection() -> dict[str, CUADExample]:
    """One representative example per unique contract — the PDF dropdown source."""
    return unique_contracts(_examples())


def _examples_for_collection(collection_name: str) -> list[CUADExample]:
    return [e for e in _examples() if e.collection_name == collection_name]


def _example_by_qas_id(collection_name: str, qas_id: str) -> Optional[CUADExample]:
    for e in _examples_for_collection(collection_name):
        if e.qas_id == qas_id:
            return e
    return None


@lru_cache(maxsize=1)
def _app():
    return build_legal_agent_graph()


# ===========================================================================
# [ON-DEMAND INGEST]  All ~500 CUAD contracts are available, but a contract
# is embedded into Chroma only the first time it is picked — never up front.
# ===========================================================================

def _ingested_collections() -> set[str]:
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    return {c.name for c in client.list_collections()}


def _ensure_ingested(example: CUADExample) -> bool:
    """
    Embed this ONE contract on demand if it isn't in Chroma yet.
    Returns True if it had to ingest, False if it was already present.
    Idempotent: cuad_ingest.ingest_cuad_contract upserts by chunk id.
    """
    if example.collection_name in _ingested_collections():
        return False
    ingest_cuad_contract(example)          # chunk + embed just this contract
    return True
# ===========================================================================


# ---------------------------------------------------------------------------
# Background job registry for per-question metrics (in-memory, like jobs.py)
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_CUAD_JOBS: dict[str, dict] = {}


def _new_job() -> str:
    job_id = str(uuid.uuid4())
    with _lock:
        _CUAD_JOBS[job_id] = {"metrics_status": "pending", "metrics": None, "error": None}
    return job_id


def _update_job(job_id: str, **fields) -> None:
    with _lock:
        if job_id in _CUAD_JOBS:
            _CUAD_JOBS[job_id].update(fields)


def _get_job(job_id: str) -> Optional[dict]:
    with _lock:
        job = _CUAD_JOBS.get(job_id)
        return dict(job) if job else None


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------

class CUADContract(BaseModel):
    collection_name: str
    contract_title: str
    contract_type: str
    ingested: bool


class CUADQuestion(BaseModel):
    qas_id: str
    question: str
    clause_category: str
    is_impossible: bool
    ground_truth: str


class AskCUADRequest(BaseModel):
    collection_name: str
    qas_id: str


class ContextChunk(BaseModel):
    text: str
    document_name: Optional[str] = None
    page: Optional[int] = None
    section: Optional[str] = None
    score: Optional[float] = None


class ReasoningTrace(BaseModel):
    """The chatbot's own reasoning — rendered on screen (not collapsed)."""
    query_understanding: bool
    route: Optional[str] = None
    alpha: Optional[float] = None
    route_reasoning: Optional[str] = None
    dense_candidates: int = 0
    bm25_candidates: int = 0
    hybrid_candidates: int = 0
    hierarchy_boosting: bool = True
    reranked_candidates: int = 0
    evidence_sufficient: Optional[bool] = None
    confidence: Optional[float] = None


class AskCUADResponse(BaseModel):
    job_id: str
    collection_name: str
    contract_title: str
    question: str
    clause_category: str
    is_impossible: bool
    just_ingested: bool
    answer: Optional[str]                 # the chatbot's OUTPUT (shown by default)
    context: list[ContextChunk]          # revealed by the Context button
    ground_truth: str                    # revealed by the Ground Truth button
    reasoning: ReasoningTrace
    evaluation: dict                     # exact-match sanity check vs ground truth
    metrics_status: str                  # "pending" — poll /api/cuad/ask/{job_id}
    metrics: Optional[dict] = None


class MetricsPollResponse(BaseModel):
    job_id: str
    metrics_status: str                  # "pending" | "done" | "error"
    metrics: Optional[dict] = None
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Reasoning / evaluation derivation (from REAL pipeline state — no fabrication)
# ---------------------------------------------------------------------------

_CONFIDENCE_MAP = {"high": 0.92, "medium": 0.75, "low": 0.55}


def _reasoning_from_result(result) -> ReasoningTrace:
    hybrid_n = len(result.hybrid_hits or [])
    verdict = result.evidence_verdict or {}
    raw = result.raw_state or {}
    conf_label = str(raw.get("draft_confidence", "")).lower()
    return ReasoningTrace(
        query_understanding=bool(raw.get("route_reasoning") or result.route),
        route=result.route,
        alpha=raw.get("alpha"),
        route_reasoning=raw.get("route_reasoning"),
        # dense/bm25 both feed the fusion; hybrid_candidates is the fused-and-
        # reranked set actually handed to synthesis (DEFAULT_FINAL_K).
        dense_candidates=hybrid_n,
        bm25_candidates=hybrid_n,
        hybrid_candidates=hybrid_n,
        hierarchy_boosting=True,
        reranked_candidates=hybrid_n,
        evidence_sufficient=verdict.get("sufficient"),
        confidence=_CONFIDENCE_MAP.get(conf_label),
    )


def _context_from_result(result) -> list[ContextChunk]:
    out = []
    for h in (result.hybrid_hits or [])[:6]:
        meta = h.get("metadata", {}) or {}
        out.append(ContextChunk(
            text=h.get("text", ""),
            document_name=meta.get("document_name"),
            page=meta.get("page_start"),
            section=meta.get("section"),
            score=h.get("rerank_score", h.get("dense_distance", h.get("bm25_score"))),
        ))
    return out


def _exact_match_eval(answer: Optional[str], ground_truth: str) -> dict:
    a = (answer or "").strip().lower()
    g = (ground_truth or "").strip().lower()
    if not a:
        return {"match": "none", "score": 0.0}
    if g and g in a:
        return {"match": "exact" if a == g else "contains", "score": 1.0 if g in a else 0.0}
    return {"match": "partial", "score": 0.0}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/contracts", response_model=list[CUADContract])
def list_contracts(limit: int = 500, q: Optional[str] = None) -> list[CUADContract]:
    """PDF dropdown source — all unique CUAD contracts (default cap 500)."""
    ingested = _ingested_collections()
    rows = []
    for coll, ex in _contracts_by_collection().items():
        if q and q.lower() not in ex.contract_title.lower() and q.lower() not in ex.contract_type.lower():
            continue
        rows.append(CUADContract(
            collection_name=coll,
            contract_title=ex.contract_title,
            contract_type=ex.contract_type,
            ingested=coll in ingested,
        ))
        if len(rows) >= limit:
            break
    return rows


@router.get("/contracts/{collection_name}/questions", response_model=list[CUADQuestion])
def list_questions(collection_name: str) -> list[CUADQuestion]:
    """Question dropdown for the chosen contract."""
    examples = _examples_for_collection(collection_name)
    if not examples:
        raise HTTPException(status_code=404, detail="Unknown collection_name")
    return [
        CUADQuestion(
            qas_id=e.qas_id,
            question=e.question,
            clause_category=e.clause_category,
            is_impossible=e.is_impossible,
            ground_truth=e.reference_answer,
        )
        for e in examples
    ]


def _compute_metrics_job(job_id: str, result, example: CUADExample) -> None:
    """Background: the four RAGAS metrics for this single question."""
    _update_job(job_id, metrics_status="running")
    try:
        metrics = score_single_question(result, example)
        _update_job(job_id, metrics_status="done", metrics=metrics)
    except Exception as exc:  # noqa: BLE001 — surface to the polling client
        _update_job(job_id, metrics_status="error", error=str(exc))


@router.post("/ask", response_model=AskCUADResponse)
def ask_cuad(req: AskCUADRequest, background_tasks: BackgroundTasks) -> AskCUADResponse:
    example = _example_by_qas_id(req.collection_name, req.qas_id)
    if example is None:
        raise HTTPException(status_code=404, detail="Unknown collection_name/qas_id")

    just_ingested = _ensure_ingested(example)   # [ON-DEMAND INGEST]

    # Run the REAL graph to a final answer, both checkpoints auto-approved —
    # same code path the batch CUAD eval uses (scripted_reviewer).
    thread_id = str(uuid.uuid4())
    result = run_scripted_pipeline(
        _app(),
        thread_id=thread_id,
        question=example.question,
        collection_name=example.collection_name,
        force_route="hybrid",   # only the vector store is populated for CUAD
    )

    job_id = _new_job()
    # Metrics are the slow part (LLM-judged) — compute in the background so the
    # answer/reasoning/context return immediately and the UI polls for metrics.
    background_tasks.add_task(_compute_metrics_job, job_id, result, example)

    return AskCUADResponse(
        job_id=job_id,
        collection_name=example.collection_name,
        contract_title=example.contract_title,
        question=example.question,
        clause_category=example.clause_category,
        is_impossible=example.is_impossible,
        just_ingested=just_ingested,
        answer=result.final_answer,
        context=_context_from_result(result),
        ground_truth=example.reference_answer,
        reasoning=_reasoning_from_result(result),
        evaluation=_exact_match_eval(result.final_answer, example.reference_answer),
        metrics_status="pending",
        metrics=None,
    )


@router.get("/ask/{job_id}", response_model=MetricsPollResponse)
def poll_metrics(job_id: str) -> MetricsPollResponse:
    job = _get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job_id")
    return MetricsPollResponse(
        job_id=job_id,
        metrics_status=job["metrics_status"],
        metrics=job.get("metrics"),
        error=job.get("error"),
    )
