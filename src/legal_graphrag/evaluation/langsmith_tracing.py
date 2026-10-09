"""
LangSmith project/tag configuration for CUAD/RAGAS evaluation runs.

See ../tracing.py for the base @traceable wiring and the list of named
checkpoints that show up in every trace. This module only adds evaluation-
specific context on top of that: a dedicated LangSmith project so batch
eval runs don't mix into interactive/manual traces, and metadata tags so
individual questions are filterable by CUAD contract type and
answerable/is_impossible status directly in the LangSmith UI.
"""

from __future__ import annotations

import os

from ..tracing import configure_langsmith, tracing_enabled
from .cuad_loader import CUADExample

DEFAULT_EVAL_PROJECT = "legal-graphrag-cuad-eval"


def setup_eval_tracing(project_name: str = DEFAULT_EVAL_PROJECT) -> bool:
    """
    Call once at the start of a CUAD eval run (see scripts/run_cuad_eval.py).
    Returns True if tracing is actually active (langsmith installed +
    LANGSMITH_API_KEY set), False otherwise — the eval harness runs fine
    either way, this is just informational for the run's console output.
    """
    configure_langsmith(project_name)
    active = tracing_enabled() and bool(os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY"))
    return active


def run_metadata_for(example: CUADExample) -> dict:
    """Per-question LangSmith tags/metadata. See call_kwargs_for() for how to pass these in safely."""
    return {
        "tags": [
            "cuad-eval",
            f"contract_type:{example.contract_type}",
            "is_impossible" if example.is_impossible else "answerable",
        ],
        "metadata": {
            "qas_id": example.qas_id,
            "contract_title": example.contract_title,
            "contract_type": example.contract_type,
            "clause_category": example.clause_category,
            "is_impossible": example.is_impossible,
        },
    }


def call_kwargs_for(example: CUADExample) -> dict:
    """
    Extra kwargs to splat into scripted_reviewer.run_scripted_pipeline(...):

        run_scripted_pipeline(app, thread_id, question, collection, **call_kwargs_for(example))

    Only includes `langsmith_extra` when tracing is actually active — the
    real @langsmith.traceable decorator understands and strips that kwarg,
    but the no-op fallback in tracing.py (used when langsmith isn't
    installed) just returns the function unchanged, so passing
    `langsmith_extra` unconditionally would raise a TypeError on an
    untraced run. This keeps the harness runnable with or without
    langsmith installed.
    """
    if not tracing_enabled():
        return {}
    return {"langsmith_extra": run_metadata_for(example)}
