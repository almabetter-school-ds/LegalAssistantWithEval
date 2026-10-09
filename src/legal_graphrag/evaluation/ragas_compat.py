"""
RAGAS compatibility shim — makes evaluate() robust to the empty-trace crash.

The bug
-------
RAGAS computes every metric successfully, then, while constructing the
EvaluationResult, calls ragas.callbacks.parse_run_traces() to attach a
debug-only `.traces` attribute. That function does:

    root_traces = [t for t in traces.values() if t.parent_run_id == parent_run_id]
    root_trace  = root_traces[0]          # <-- IndexError when the list is empty

`root_traces` comes out empty whenever the judge LLM / embeddings wrapper does
NOT emit LangChain callback run-events that root at the evaluation run_id. This
project's judge is a *custom* BaseChatModel (HuggingFaceChatModel in
ragas_eval.py) whose `_generate` doesn't propagate the callback manager, so no
root run is ever recorded -> empty traces -> IndexError, AFTER the scores are
already in hand.

The fix
-------
`.traces` is purely for post-hoc introspection; `.scores` / `.to_pandas()` are
already populated before it runs. So we wrap parse_run_traces to return an empty
list instead of throwing. This turns a hard crash into a graceful "no debug
traces available", and evaluate() returns normally with valid metrics.

We patch the name in BOTH namespaces because dataset_schema.py did
`from ragas.callbacks import parse_run_traces` (binding its own reference),
while other call sites may use `callbacks.parse_run_traces`.

Call install_ragas_trace_guard() once before evaluate(). Idempotent.
"""

from __future__ import annotations


def install_ragas_trace_guard() -> bool:
    """Patch ragas.parse_run_traces to be crash-proof. Returns True if installed."""
    try:
        from ragas import callbacks as _cb
    except Exception:
        return False  # ragas not installed / import failed — nothing to guard

    if getattr(_cb, "_lg_trace_guard_installed", False):
        return True

    _orig = _cb.parse_run_traces

    def _safe_parse_run_traces(*args, **kwargs):
        try:
            return _orig(*args, **kwargs)
        except (IndexError, KeyError, ValueError):
            # Empty/malformed trace tree — scores are unaffected; traces are
            # debug-only, so an empty list is the correct, harmless fallback.
            return []

    _cb.parse_run_traces = _safe_parse_run_traces

    # dataset_schema imported the symbol directly — rebind it there too.
    try:
        from ragas import dataset_schema as _ds
        _ds.parse_run_traces = _safe_parse_run_traces
    except Exception:
        pass

    _cb._lg_trace_guard_installed = True
    return True