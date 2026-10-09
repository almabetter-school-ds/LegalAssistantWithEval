"""
Review feedback / error-log routes.

Upload & Ask runs the human-in-the-loop pipeline, but the pipeline's own
checkpoints only affect the CURRENT thread — a reviewer's "revise" or "reject"
isn't recorded anywhere for later. This router adds a durable log so those
reviews are collected for future reference (retraining data, error analysis,
regression tracking).

Each record captures the shape the team asked for:

    question, retrieved_evidence, expected_answer, actual_answer, failure_type

plus a timestamp, the collection, the reviewer, and free-text comments.

Records are appended as JSON Lines to data/metadata/review_feedback.jsonl
(override with LEGAL_GRAPHRAG_FEEDBACK_PATH). Append-only + one JSON object per
line means it's safe to write incrementally and trivial to load later with
pandas.read_json(path, lines=True).

Endpoints:
    POST /api/feedback         append one record        -> {ok, id, path, count}
    GET  /api/feedback         list recent records      -> {records, count, path}
    GET  /api/feedback/export  download the raw JSONL    -> file
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter
from fastapi.responses import FileResponse
from pydantic import BaseModel

router = APIRouter(prefix="/api/feedback", tags=["feedback"])

_LOCK = threading.Lock()


def _feedback_path() -> Path:
    p = os.environ.get("LEGAL_GRAPHRAG_FEEDBACK_PATH", "data/metadata/review_feedback.jsonl")
    path = Path(p)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


class EvidenceItem(BaseModel):
    text: str = ""
    document_name: Optional[str] = None
    section: Optional[str] = None
    page: Optional[int] = None
    score: Optional[float] = None


class FeedbackRecord(BaseModel):
    # the five fields the team asked to collect:
    question: str
    retrieved_evidence: list[EvidenceItem] = []
    expected_answer: Optional[str] = None      # reviewer's correction / edited answer
    actual_answer: Optional[str] = None        # what the model drafted
    failure_type: str                          # "revise" | "reject" | "incorrect" | "other"
    # context:
    collection_name: Optional[str] = None
    thread_id: Optional[str] = None
    reviewer: Optional[str] = None
    comments: Optional[str] = None


class FeedbackStored(BaseModel):
    ok: bool
    id: str
    path: str
    count: int


class FeedbackList(BaseModel):
    records: list[dict]
    count: int
    path: str


@router.post("", response_model=FeedbackStored)
def add_feedback(rec: FeedbackRecord) -> FeedbackStored:
    entry = rec.model_dump()
    entry["id"] = str(uuid.uuid4())
    entry["logged_at"] = datetime.now(timezone.utc).isoformat()

    path = _feedback_path()
    with _LOCK:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        count = sum(1 for _ in open(path, "r", encoding="utf-8"))

    return FeedbackStored(ok=True, id=entry["id"], path=str(path), count=count)


@router.get("", response_model=FeedbackList)
def list_feedback(limit: int = 200) -> FeedbackList:
    path = _feedback_path()
    records: list[dict] = []
    if path.exists():
        with _LOCK, open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    records.reverse()  # newest first
    return FeedbackList(records=records[:limit], count=len(records), path=str(path))


@router.get("/export")
def export_feedback():
    path = _feedback_path()
    if not path.exists():
        path.write_text("", encoding="utf-8")
    return FileResponse(str(path), media_type="application/x-ndjson", filename="review_feedback.jsonl")


# Defensive: ensure nested forward refs are resolved regardless of import order.
EvidenceItem.model_rebuild()
FeedbackRecord.model_rebuild()