"""
LangSmith tracing helpers.

LangSmith tracing is OPTIONAL.

The pipeline runs identically with or without LangSmith tracing.

LangGraph automatically traces LangChain/LangGraph nodes when:

    LANGCHAIN_TRACING_V2=true

The @traceable decorator is used for lower-level functions that LangGraph
cannot automatically see into, such as direct Hugging Face InferenceClient
calls.
"""

from __future__ import annotations

import functools
import os
from typing import Any, Callable, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


# ---------------------------------------------------------------------------
# Optional LangSmith import
# ---------------------------------------------------------------------------

try:
    from langsmith import traceable as _ls_traceable

    _LANGSMITH_INSTALLED = True

except ImportError:  # pragma: no cover
    _LANGSMITH_INSTALLED = False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def tracing_enabled() -> bool:
    """
    True only when LangSmith is installed and tracing is enabled.
    """

    if not _LANGSMITH_INSTALLED:
        return False

    return (
        os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true"
        or os.getenv("LANGSMITH_TRACING", "").lower() == "true"
    )


def configure_langsmith(
    project_name: str,
    api_key: str | None = None,
) -> None:
    """
    Configure LangSmith tracing.

    Safe to call even when langsmith is not installed.
    """

    os.environ["LANGCHAIN_PROJECT"] = project_name

    if api_key:
        os.environ["LANGCHAIN_API_KEY"] = api_key

    elif (
        os.getenv("LANGSMITH_API_KEY")
        and not os.getenv("LANGCHAIN_API_KEY")
    ):
        os.environ["LANGCHAIN_API_KEY"] = os.environ[
            "LANGSMITH_API_KEY"
        ]

    if os.getenv("LANGCHAIN_API_KEY"):
        os.environ["LANGCHAIN_TRACING_V2"] = "true"


# ---------------------------------------------------------------------------
# Generic traceable decorator
# ---------------------------------------------------------------------------

def traceable(
    name: str,
    run_type: str = "chain",
    **traceable_kwargs,
) -> Callable[[F], F]:
    """
    Thin wrapper around langsmith.traceable.

    If LangSmith is unavailable, returns a no-op decorator.
    """

    if not _LANGSMITH_INSTALLED:

        def _noop_decorator(fn: F) -> F:
            return fn

        return _noop_decorator

    return _ls_traceable(
        name=name,
        run_type=run_type,
        **traceable_kwargs,
    )


# ---------------------------------------------------------------------------
# Hugging Face tracing
# ---------------------------------------------------------------------------

def trace_huggingface_call(
    name: str = "llm.huggingface",
):
    """
    Decorator for functions that make Hugging Face InferenceClient calls.

    Example:

        @trace_huggingface_call("llm.ragas_judge")
        def call_hf(...):
            ...
    """

    def decorator(fn: F) -> F:

        if not _LANGSMITH_INSTALLED or not tracing_enabled():
            return fn

        return _ls_traceable(
            name=name,
            run_type="llm",
        )(fn)

    return decorator

# """
# LangSmith tracing helpers.

# LangSmith tracing is OPTIONAL — the pipeline runs identically with or
# without it. When `langsmith` is installed and tracing is enabled (see
# configure_langsmith()), every decorated checkpoint below shows up as its
# own named run in the LangSmith UI, nested under the parent LangGraph
# invocation trace that LangGraph produces automatically (LangGraph is built
# on langchain-core Runnables, so node execution is traced for free once
# LANGCHAIN_TRACING_V2=true is set — @traceable below is only needed for the
# raw `anthropic` SDK calls in llm_client.py, which LangGraph can't see into
# on its own).

# Traced checkpoints (see agents/legal_pipeline.py and agents/prompts.py):
#   - "router.classify_route"            — route + alpha decision
#   - "retrieval.hybrid_search_agent"     — dense+BM25+RRF+rerank call
#   - "retrieval.graph_rag_agent"         — template/text-to-Cypher + Neo4j read
#   - "auditor.verify_evidence"           — EvidenceChecker LLM call
#   - "checkpoint.human_evidence"         — interrupt() payload + resume decision
#   - "synthesizer.synthesize_legal_answer" — AnswerAgent LLM call
#   - "checkpoint.human_answer"           — interrupt() payload + resume decision
#   - "synthesizer.revise_legal_answer"   — revision LLM call (only on "revise")
#   - "pipeline.finalize"                 — approved answer persisted
#   - "pipeline.graph_update"             — Neo4j write-back (graph route only)

# Each is tagged so CUAD/RAGAS evaluation runs are filterable in the
# LangSmith UI separately from interactive/manual usage — see
# evaluation/langsmith_tracing.py, which sets these tags for a batch run.
# """

# from __future__ import annotations

# import functools
# import os
# from typing import Any, Callable, TypeVar

# F = TypeVar("F", bound=Callable[..., Any])

# try:
#     from langsmith import traceable as _ls_traceable
#     from langsmith.wrappers import wrap_anthropic as _ls_wrap_anthropic

#     _LANGSMITH_INSTALLED = True
# except ImportError:  # pragma: no cover - exercised whenever langsmith isn't installed
#     _LANGSMITH_INSTALLED = False


# def tracing_enabled() -> bool:
#     """True only if langsmith is installed AND the user opted in via env vars."""
#     if not _LANGSMITH_INSTALLED:
#         return False
#     return os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true" or bool(
#         os.getenv("LANGSMITH_TRACING", "").lower() == "true"
#     )


# def configure_langsmith(project_name: str, api_key: str | None = None) -> None:
#     """
#     Call once at process/script startup (see evaluation/langsmith_tracing.py)
#     to point tracing at a specific LangSmith project. Safe to call even if
#     langsmith isn't installed — it just sets env vars that go unused.

#     Only flips LANGCHAIN_TRACING_V2 on if an API key is actually available
#     (passed in here, or already present via LANGSMITH_API_KEY/
#     LANGCHAIN_API_KEY) — never force tracing "on" project-wide with no key
#     to send traces with, since tracing_enabled() (checked by every
#     @traceable call site and wrap_anthropic_client) trusts this flag.
#     """
#     os.environ["LANGCHAIN_PROJECT"] = project_name
#     if api_key:
#         os.environ["LANGCHAIN_API_KEY"] = api_key
#     elif os.getenv("LANGSMITH_API_KEY") and not os.getenv("LANGCHAIN_API_KEY"):
#         os.environ["LANGCHAIN_API_KEY"] = os.environ["LANGSMITH_API_KEY"]

#     if os.getenv("LANGCHAIN_API_KEY"):
#         os.environ["LANGCHAIN_TRACING_V2"] = "true"


# def traceable(name: str, run_type: str = "chain", **traceable_kwargs) -> Callable[[F], F]:
#     """
#     Thin wrapper around langsmith.traceable that degrades to a no-op
#     decorator when langsmith isn't installed or tracing isn't enabled, so
#     every module in this package can decorate its checkpoint functions
#     unconditionally without an import-time dependency on langsmith.
#     """
#     if not _LANGSMITH_INSTALLED:
#         def _noop_decorator(fn: F) -> F:
#             return fn
#         return _noop_decorator

#     return _ls_traceable(name=name, run_type=run_type, **traceable_kwargs)


# def wrap_anthropic_client(client):
#     """
#     Wraps the shared anthropic.Anthropic client (see llm_client.py) so every
#     messages.create() call is logged as an LLM run with token counts, model
#     name, and full prompt/completion — not just the @traceable-wrapped
#     function that called it. No-op passthrough if langsmith isn't installed
#     or tracing isn't enabled.
#     """
#     if not tracing_enabled():
#         return client
#     return _ls_wrap_anthropic(client)
