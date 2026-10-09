"""
Per-question CUAD scoring — the "compute metrics for ONE question, right now"
path the web UI needs, built entirely on top of the UNMODIFIED batch harness
in evaluation/ragas_eval.py (no changes to that module).

Where ragas_eval.run_ragas_evaluation() is designed for a whole CUAD subset
(many rows, one aggregate table), the frontend's "Ask CUAD" flow needs the
four RAGAS metrics — Context Precision, Context Recall, Faithfulness, and
Answer Correctness — for the single (contract, question) pair the user just
picked, so it can show them live next to that one answer.

This module does exactly that and nothing more:

  * ANSWERABLE question  -> wrap the single ScriptedRunResult in one RagasRow
    (via ragas_eval.build_ragas_rows) and run the same four metrics over a
    one-row dataset (ragas_eval.run_ragas_evaluation), then read the four
    numbers back out of the returned pandas frame.

  * UNANSWERABLE question (CUAD is_impossible) -> there is no reference span
    to score Precision/Recall/Correctness against, so we fall back to
    ragas_eval.score_absence_detection (same semantics as the batch path)
    and report "correctly identified absence" instead of the four metrics.

Everything RAGAS-specific (judge LLM, embeddings, metric definitions) is
inherited unchanged from ragas_eval — this file only reshapes the input to
a single row and reshapes the output to a small dict the API can serialise.
"""

from __future__ import annotations

import math
from typing import Optional

from .cuad_loader import CUADExample
from .ragas_eval import (
    build_ragas_rows,
    run_ragas_evaluation,
    score_absence_detection,
)
from .scripted_reviewer import ScriptedRunResult

# RAGAS column name -> the key the frontend expects. Kept explicit so a RAGAS
# version bump that renames a column fails loudly here instead of silently
# returning None for a metric.
_METRIC_COLUMNS = {
    "context_precision": ("llm_context_precision_with_reference", "context_precision"),
    "context_recall": ("context_recall", "llm_context_recall"),
    "faithfulness": ("faithfulness",),
    "answer_correctness": ("answer_correctness",),
}


def _first_present(row: dict, candidates: tuple[str, ...]) -> Optional[float]:
    for name in candidates:
        if name not in row:
            continue
        value = row[name]
        if value is None:
            return None
        try:
            value = float(value)
        except (TypeError, ValueError):
            # non-numeric (e.g. a string label) — treat as "not available"
            return None
        if math.isnan(value):
            # RAGAS returns NaN for a metric it couldn't compute — surface as
            # None, not a misleading 0.0.
            return None
        return value
    return None


def score_answerable_question(
    result: ScriptedRunResult, example: CUADExample
) -> dict:
    """
    Runs the four RAGAS metrics for ONE answerable CUAD question and returns:

        {
          "answerable": True,
          "context_precision": float|None,
          "context_recall":    float|None,
          "faithfulness":      float|None,
          "answer_correctness":float|None,
        }

    Prefer EvaluationResult._repr_dict — the SAME representation that
    scripts/run_cuad_eval.py reads successfully. Fall back to to_pandas()
    only for RAGAS versions that don't expose _repr_dict. (Reading only
    to_pandas() was returning N/A for every metric.)
    """
    rows = build_ragas_rows([result], [example])   # exactly one RagasRow
    if not rows:
        raise RuntimeError("build_ragas_rows() returned no rows for the question.")
    evaluation = run_ragas_evaluation(rows)         # same four metrics as the batch path

    # Preferred path: identical to run_cuad_eval.py's summary extraction.
    if hasattr(evaluation, "_repr_dict"):
        scores = dict(evaluation._repr_dict)
    else:
        frame = evaluation.to_pandas()
        print("[single-ragas] dataframe columns:", list(frame.columns))
        scores = frame.iloc[0].to_dict() if len(frame) else {}
    print("[single-ragas] raw scores:", scores)

    return {
        "answerable": True,
        "context_precision": _first_present(scores, _METRIC_COLUMNS["context_precision"]),
        "context_recall": _first_present(scores, _METRIC_COLUMNS["context_recall"]),
        "faithfulness": _first_present(scores, _METRIC_COLUMNS["faithfulness"]),
        "answer_correctness": _first_present(scores, _METRIC_COLUMNS["answer_correctness"]),
    }


def score_unanswerable_question(
    result: ScriptedRunResult, example: CUADExample
) -> dict:
    """
    For a CUAD is_impossible question there is no reference span, so the four
    RAGAS metrics don't apply. Report absence-detection instead — identical
    semantics to ragas_eval.score_absence_detection in the batch harness.
    """
    scored = score_absence_detection([result], [example])[0]
    return {
        "answerable": False,
        "correctly_identified_absence": scored.correctly_identified_absence,
        "evidence_marked_insufficient": scored.evidence_marked_insufficient,
        "evidence_rejected": scored.evidence_rejected,
        "answer_stated_absence": scored.answer_stated_absence,
        # keep the four keys present (None) so the frontend can render one shape
        "context_precision": None,
        "context_recall": None,
        "faithfulness": None,
        "answer_correctness": None,
    }


def score_single_question(result: ScriptedRunResult, example: CUADExample) -> dict:
    """Dispatch on CUAD's is_impossible flag — the one entry point the API calls."""
    if example.is_impossible:
        return score_unanswerable_question(result, example)
    return score_answerable_question(result, example)

# """
# Per-question CUAD scoring — the "compute metrics for ONE question, right now"
# path the web UI needs, built entirely on top of the UNMODIFIED batch harness
# in evaluation/ragas_eval.py (no changes to that module).

# Where ragas_eval.run_ragas_evaluation() is designed for a whole CUAD subset
# (many rows, one aggregate table), the frontend's "Ask CUAD" flow needs the
# four RAGAS metrics — Context Precision, Context Recall, Faithfulness, and
# Answer Correctness — for the single (contract, question) pair the user just
# picked, so it can show them live next to that one answer.

# This module does exactly that and nothing more:

#   * ANSWERABLE question  -> wrap the single ScriptedRunResult in one RagasRow
#     (via ragas_eval.build_ragas_rows) and run the same four metrics over a
#     one-row dataset (ragas_eval.run_ragas_evaluation), then read the four
#     numbers back out of the returned pandas frame.

#   * UNANSWERABLE question (CUAD is_impossible) -> there is no reference span
#     to score Precision/Recall/Correctness against, so we fall back to
#     ragas_eval.score_absence_detection (same semantics as the batch path)
#     and report "correctly identified absence" instead of the four metrics.

# Everything RAGAS-specific (judge LLM, embeddings, metric definitions) is
# inherited unchanged from ragas_eval — this file only reshapes the input to
# a single row and reshapes the output to a small dict the API can serialise.
# """

# from __future__ import annotations

# import math
# from typing import Optional

# from .cuad_loader import CUADExample
# from .ragas_eval import (
#     build_ragas_rows,
#     run_ragas_evaluation,
#     score_absence_detection,
# )
# from .scripted_reviewer import ScriptedRunResult

# # RAGAS column name -> the key the frontend expects. Kept explicit so a RAGAS
# # version bump that renames a column fails loudly here instead of silently
# # returning None for a metric.
# _METRIC_COLUMNS = {
#     "context_precision": ("llm_context_precision_with_reference", "context_precision"),
#     "context_recall": ("context_recall", "llm_context_recall"),
#     "faithfulness": ("faithfulness",),
#     "answer_correctness": ("answer_correctness",),
# }


# def _first_present(row: dict, candidates: tuple[str, ...]) -> Optional[float]:
#     for name in candidates:
#         if name in row:
#             value = row[name]
#             # RAGAS returns NaN for a metric it couldn't compute (e.g. empty
#             # retrieved_contexts) — surface that as None, not a misleading 0.0.
#             if value is None or (isinstance(value, float) and math.isnan(value)):
#                 return None
#             return float(value)
#     return None


# def score_answerable_question(
#     result: ScriptedRunResult, example: CUADExample
# ) -> dict:
#     """
#     Runs the four RAGAS metrics for ONE answerable CUAD question and returns:

#         {
#           "answerable": True,
#           "context_precision": float|None,
#           "context_recall":    float|None,
#           "faithfulness":      float|None,
#           "answer_correctness":float|None,
#         }

#     Any metric RAGAS could not compute comes back as None (never a fake 0.0).
#     """
#     rows = build_ragas_rows([result], [example])   # exactly one RagasRow
#     evaluation = run_ragas_evaluation(rows)         # same four metrics as the batch path
#     frame = evaluation.to_pandas()
#     row = frame.iloc[0].to_dict() if len(frame) else {}

#     return {
#         "answerable": True,
#         "context_precision": _first_present(row, _METRIC_COLUMNS["context_precision"]),
#         "context_recall": _first_present(row, _METRIC_COLUMNS["context_recall"]),
#         "faithfulness": _first_present(row, _METRIC_COLUMNS["faithfulness"]),
#         "answer_correctness": _first_present(row, _METRIC_COLUMNS["answer_correctness"]),
#     }


# def score_unanswerable_question(
#     result: ScriptedRunResult, example: CUADExample
# ) -> dict:
#     """
#     For a CUAD is_impossible question there is no reference span, so the four
#     RAGAS metrics don't apply. Report absence-detection instead — identical
#     semantics to ragas_eval.score_absence_detection in the batch harness.
#     """
#     scored = score_absence_detection([result], [example])[0]
#     return {
#         "answerable": False,
#         "correctly_identified_absence": scored.correctly_identified_absence,
#         "evidence_marked_insufficient": scored.evidence_marked_insufficient,
#         "evidence_rejected": scored.evidence_rejected,
#         "answer_stated_absence": scored.answer_stated_absence,
#         # keep the four keys present (None) so the frontend can render one shape
#         "context_precision": None,
#         "context_recall": None,
#         "faithfulness": None,
#         "answer_correctness": None,
#     }


# def score_single_question(result: ScriptedRunResult, example: CUADExample) -> dict:
#     """Dispatch on CUAD's is_impossible flag — the one entry point the API calls."""
#     if example.is_impossible:
#         return score_unanswerable_question(result, example)
#     return score_answerable_question(result, example)
