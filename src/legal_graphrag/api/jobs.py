"""
In-memory registries used only by the API layer — no pipeline files import
this. Reset on process restart, same caveat as the pipeline's own
MemorySaver checkpointer (see agents/legal_pipeline.py).

  - QUERY_THREADS: thread_id -> {question, collection_name, created_at},
    for listing/reopening threads. The actual pipeline state for each
    thread lives in LangGraph's checkpointer, not here.
  - INGEST_JOBS: background ingestion job status/result, since
    ingestion/pdf_pipeline.run_pipeline() is synchronous and can take a
    while (OCR fallback, embedding) — the API runs it via BackgroundTasks
    and polls this dict.
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Optional

_lock = threading.Lock()
QUERY_THREADS: dict[str, dict] = {}
INGEST_JOBS: dict[str, dict] = {}


def register_query_thread(thread_id: str, question: str, collection_name: str) -> None:
    with _lock:
        QUERY_THREADS[thread_id] = {
            "question": question,
            "collection_name": collection_name,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }


def list_query_threads() -> list[dict]:
    with _lock:
        return [{"thread_id": tid, **meta} for tid, meta in QUERY_THREADS.items()]


def get_query_thread(thread_id: str) -> Optional[dict]:
    with _lock:
        return QUERY_THREADS.get(thread_id)


def register_ingest_job(job_id: str, filename: str) -> None:
    with _lock:
        INGEST_JOBS[job_id] = {"status": "pending", "filename": filename, "collection_name": None,
                                "result": None, "error": None}


def update_ingest_job(job_id: str, **fields) -> None:
    with _lock:
        if job_id in INGEST_JOBS:
            INGEST_JOBS[job_id].update(fields)


def get_ingest_job(job_id: str) -> Optional[dict]:
    with _lock:
        job = INGEST_JOBS.get(job_id)
        return dict(job) if job else None
