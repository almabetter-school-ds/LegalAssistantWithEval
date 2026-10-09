"""
CUAD + RAGAS evaluation CLI for the legal_graphrag query pipeline.

    python -m scripts.run_cuad_eval --n-contracts 20 --seed 42 \
        --out data/metadata/cuad_eval_results.json

Auto-downloads CUAD from Hugging Face (theatticusproject/cuad) on first run
and caches it locally — pass --cuad-json to use your own local copy instead:

    python -m scripts.run_cuad_eval --cuad-json data/uploads/CUAD_v1.json \
        --n-contracts 20 --seed 42 --out data/metadata/cuad_eval_results.json

What this does, end to end:
  1. Loads CUAD_v1.json (evaluation/cuad_loader.py) and takes a deterministic
     sample of contracts (bounds ingestion + LLM spend — see --n-contracts).
  2. Ingests each sampled contract into its own Chroma collection ONLY
     (evaluation/cuad_ingest.py) — no Neo4j clause extraction.
  3. For every question about a sampled contract, drives the REAL pipeline
     graph (agents/legal_pipeline.py) end-to-end via a scripted auto-approve
     reviewer (evaluation/scripted_reviewer.py) that resumes both
     interrupt()s with Command(resume=...), forced onto the `hybrid` route.
  4. Splits results into answerable / is_impossible (cuad_loader.split_answerable):
       - answerable -> scored with RAGAS (faithfulness, context_precision,
         context_recall, answer_correctness) against CUAD's ground truth spans.
       - is_impossible -> scored with a separate "correctly identified
         absence" accuracy (evaluation/ragas_eval.score_absence_detection).
  5. Writes a JSON results file and prints a summary. If LangSmith tracing
     is enabled (LANGSMITH_API_KEY set), every question's full pipeline
     trace — router, retrieval, auditor, both checkpoints, synthesizer,
     finalize — is visible in the `legal-graphrag-cuad-eval` LangSmith
     project (see evaluation/langsmith_tracing.py).

Requires the optional evaluation dependencies — see requirements.txt's
"Evaluation" section (ragas, langchain-anthropic, langchain-huggingface,
langsmith) — in addition to the base project requirements. Also requires a
running Neo4j instance (used for the pipeline's job/audit-trail records,
even though CUAD contracts themselves are never written to it — see
evaluation/cuad_ingest.py's module docstring) and ANTHROPIC_API_KEY.
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from legal_graphrag.agents.legal_pipeline import build_legal_agent_graph
from legal_graphrag.evaluation.cuad_loader import (
    CUAD_REPO_ID,
    load_cuad,
    sample_contracts,
    split_answerable,
    unique_contracts,
)
from legal_graphrag.evaluation.cuad_ingest import ingest_cuad_contracts
from legal_graphrag.evaluation.langsmith_tracing import call_kwargs_for, setup_eval_tracing
from legal_graphrag.evaluation.ragas_eval import (
    absence_detection_accuracy,
    build_ragas_rows,
    run_ragas_evaluation,
    score_absence_detection,
)
from legal_graphrag.evaluation.scripted_reviewer import run_scripted_pipeline


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cuad-json", default=None,
                    help="Path to a local CUAD_v1.json (SQuAD-2.0 format). "
                         "Omit to auto-download from Hugging Face (theatticusproject/cuad).")
    p.add_argument("--n-contracts", type=int, default=20, help="Number of unique contracts to sample.")
    p.add_argument("--n-questions", type=int,default=None,help="Maximum number of questions to evaluate after contract sampling.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="data/metadata/cuad_eval_results.json")
    p.add_argument("--skip-ingest", action="store_true",
                    help="Skip re-ingestion (use if these contracts' Chroma collections already exist).")
    p.add_argument("--langsmith-project", default=None, help="Override the default LangSmith project name.")
    return p.parse_args()


def _truncate(text: str | None, limit: int = 600) -> str:
    """One-line-safe preview of a (possibly multi-line / long) string."""
    if not text:
        return "(empty)"
    s = " ".join(str(text).split())        # collapse whitespace/newlines
    return s if len(s) <= limit else s[:limit] + " …"


def print_answers(examples, results) -> None:
    """
    Print every question's PIPELINE ANSWER next to its CUAD GROUND TRUTH.

    Covers BOTH answerable and is_impossible questions (unlike RAGAS rows,
    which exclude is_impossible). `result.final_answer` is the pipeline's
    output; `example.reference_answer` is CUAD's human annotation (or an
    explicit "no such clause" string for is_impossible cases).
    """
    print("\n" + "=" * 78)
    print("[answers] pipeline output vs CUAD ground truth")
    print("=" * 78)
    for i, (example, result) in enumerate(zip(examples, results), start=1):
        tag = "is_impossible" if example.is_impossible else "answerable"
        print(f"\n[{i}/{len(examples)}] {example.contract_title}")
        print(f"    category    : {example.clause_category}  ({tag})")
        print(f"    question    : {_truncate(example.question, 200)}")
        print(f"    >> ANSWER   : {_truncate(result.final_answer)}")
        print(f"    ground truth: {_truncate(example.reference_answer)}")
        print(f"    status      : {result.status}  |  evidence_sufficient="
              f"{result.evidence_verdict.get('sufficient')}")
    print("=" * 78 + "\n")


def main() -> None:
    args = parse_args()

    tracing_active = setup_eval_tracing(args.langsmith_project) if args.langsmith_project else setup_eval_tracing()
    print(f"[tracing] LangSmith tracing {'ENABLED' if tracing_active else 'disabled'} "
          f"(set LANGSMITH_API_KEY to enable)")

    print(f"[load] parsing {args.cuad_json or f'CUAD from Hugging Face ({CUAD_REPO_ID})'}")
    all_examples = load_cuad(args.cuad_json)
    examples = sample_contracts(all_examples, n_contracts=args.n_contracts, seed=args.seed)
    # Limit the number of questions AFTER selecting contracts.
    if args.n_questions is not None:
        examples = examples[:args.n_questions]
    answerable, unanswerable = split_answerable(examples)
    print(f"[load] sampled {len(unique_contracts(examples))} contracts -> "
          f"{len(examples)} questions ({len(answerable)} answerable, {len(unanswerable)} is_impossible)")

    if not args.skip_ingest:
        print("[ingest] embedding sampled contracts into Chroma (vector store only)")
        ingest_cuad_contracts(unique_contracts(examples))
    else:
        print("[ingest] skipped (--skip-ingest)")

    print("[pipeline] building graph + running scripted evaluation")
    app = build_legal_agent_graph()

    results = []
    for i, example in enumerate(examples, start=1):
        print(f"  [{i}/{len(examples)}] {example.contract_title!r} :: {example.clause_category}")
        result = run_scripted_pipeline(
            app,
            thread_id=f"cuad-eval-{uuid.uuid4()}",
            question=example.question,
            collection_name=example.collection_name,
            **call_kwargs_for(example),
        )
        results.append(result)

    # ------------------------------------------------------------------
    # Print each pipeline ANSWER next to its CUAD ground truth, so the
    # terminal shows the actual answers (not just status + RAGAS scores).
    # ------------------------------------------------------------------
    print_answers(examples, results)

    # Split results in lockstep with the examples split above.
    answerable_results = [r for e, r in zip(examples, results) if not e.is_impossible]
    unanswerable_results = [r for e, r in zip(examples, results) if e.is_impossible]

    summary: dict = {"n_contracts": len(unique_contracts(examples)), "n_questions": len(examples)}

    # Persist the per-question answers into the summary too, so the JSON
    # results file is a complete record (answer + ground truth per question).
    summary["answers"] = [
        {
            "contract_title": e.contract_title,
            "clause_category": e.clause_category,
            "is_impossible": e.is_impossible,
            "question": e.question,
            "answer": r.final_answer,
            "ground_truth": e.reference_answer,
            "status": r.status,
            "evidence_sufficient": r.evidence_verdict.get("sufficient"),
        }
        for e, r in zip(examples, results)
    ]

    if unanswerable:
        print("[score] absence-detection accuracy (is_impossible questions)")
        absence_scored = score_absence_detection(unanswerable_results, unanswerable)
        summary["absence_detection_accuracy"] = absence_detection_accuracy(absence_scored)
        summary["absence_detection_n"] = len(absence_scored)
        print(f"         {summary['absence_detection_accuracy']:.1%} over {len(absence_scored)} questions")

    if answerable:
        print("[score] RAGAS metrics (answerable questions)")
        rows = build_ragas_rows(answerable_results, answerable)
        ragas_result = run_ragas_evaluation(rows)
        ragas_scores = {k: float(v) for k, v in ragas_result._repr_dict.items()} \
            if hasattr(ragas_result, "_repr_dict") else dict(ragas_result)
        summary["ragas"] = ragas_scores
        summary["ragas_n"] = len(rows)
        for metric, score in ragas_scores.items():
            print(f"         {metric}: {score:.3f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[done] summary written to {out_path}")


if __name__ == "__main__":
    main()
# """
# CUAD + RAGAS evaluation CLI for the legal_graphrag query pipeline.

#     python -m scripts.run_cuad_eval --n-contracts 20 --seed 42 \
#         --out data/metadata/cuad_eval_results.json

# Auto-downloads CUAD from Hugging Face (theatticusproject/cuad) on first run
# and caches it locally — pass --cuad-json to use your own local copy instead:

#     python -m scripts.run_cuad_eval --cuad-json data/uploads/CUAD_v1.json \
#         --n-contracts 20 --seed 42 --out data/metadata/cuad_eval_results.json

# What this does, end to end:
#   1. Loads CUAD_v1.json (evaluation/cuad_loader.py) and takes a deterministic
#      sample of contracts (bounds ingestion + LLM spend — see --n-contracts).
#   2. Ingests each sampled contract into its own Chroma collection ONLY
#      (evaluation/cuad_ingest.py) — no Neo4j clause extraction.
#   3. For every question about a sampled contract, drives the REAL pipeline
#      graph (agents/legal_pipeline.py) end-to-end via a scripted auto-approve
#      reviewer (evaluation/scripted_reviewer.py) that resumes both
#      interrupt()s with Command(resume=...), forced onto the `hybrid` route.
#   4. Splits results into answerable / is_impossible (cuad_loader.split_answerable):
#        - answerable -> scored with RAGAS (faithfulness, context_precision,
#          context_recall, answer_correctness) against CUAD's ground truth spans.
#        - is_impossible -> scored with a separate "correctly identified
#          absence" accuracy (evaluation/ragas_eval.score_absence_detection).
#   5. Writes a JSON results file and prints a summary. If LangSmith tracing
#      is enabled (LANGSMITH_API_KEY set), every question's full pipeline
#      trace — router, retrieval, auditor, both checkpoints, synthesizer,
#      finalize — is visible in the `legal-graphrag-cuad-eval` LangSmith
#      project (see evaluation/langsmith_tracing.py).

# Requires the optional evaluation dependencies — see requirements.txt's
# "Evaluation" section (ragas, langchain-anthropic, langchain-huggingface,
# langsmith) — in addition to the base project requirements. Also requires a
# running Neo4j instance (used for the pipeline's job/audit-trail records,
# even though CUAD contracts themselves are never written to it — see
# evaluation/cuad_ingest.py's module docstring) and ANTHROPIC_API_KEY.
# """

# from __future__ import annotations

# import argparse
# import json
# import sys
# import uuid
# from pathlib import Path

# sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# from legal_graphrag.agents.legal_pipeline import build_legal_agent_graph
# from legal_graphrag.evaluation.cuad_loader import (
#     CUAD_REPO_ID,
#     load_cuad,
#     sample_contracts,
#     split_answerable,
#     unique_contracts,
# )
# from legal_graphrag.evaluation.cuad_ingest import ingest_cuad_contracts
# from legal_graphrag.evaluation.langsmith_tracing import call_kwargs_for, setup_eval_tracing
# from legal_graphrag.evaluation.ragas_eval import (
#     absence_detection_accuracy,
#     build_ragas_rows,
#     run_ragas_evaluation,
#     score_absence_detection,
# )
# from legal_graphrag.evaluation.scripted_reviewer import run_scripted_pipeline


# def parse_args() -> argparse.Namespace:
#     p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
#     p.add_argument("--cuad-json", default=None,
#                     help="Path to a local CUAD_v1.json (SQuAD-2.0 format). "
#                          "Omit to auto-download from Hugging Face (theatticusproject/cuad).")
#     p.add_argument("--n-contracts", type=int, default=20, help="Number of unique contracts to sample.")
#     p.add_argument("--n-questions", type=int,default=None,help="Maximum number of questions to evaluate after contract sampling.")
#     p.add_argument("--seed", type=int, default=42)
#     p.add_argument("--out", default="data/metadata/cuad_eval_results.json")
#     p.add_argument("--skip-ingest", action="store_true",
#                     help="Skip re-ingestion (use if these contracts' Chroma collections already exist).")
#     p.add_argument("--langsmith-project", default=None, help="Override the default LangSmith project name.")
#     return p.parse_args()


# def main() -> None:
#     args = parse_args()

#     tracing_active = setup_eval_tracing(args.langsmith_project) if args.langsmith_project else setup_eval_tracing()
#     print(f"[tracing] LangSmith tracing {'ENABLED' if tracing_active else 'disabled'} "
#           f"(set LANGSMITH_API_KEY to enable)")

#     print(f"[load] parsing {args.cuad_json or f'CUAD from Hugging Face ({CUAD_REPO_ID})'}")
#     all_examples = load_cuad(args.cuad_json)
#     examples = sample_contracts(all_examples, n_contracts=args.n_contracts, seed=args.seed)
#     # Limit the number of questions AFTER selecting contracts.
#     if args.n_questions is not None:
#         examples = examples[:args.n_questions]
#     answerable, unanswerable = split_answerable(examples)
#     print(f"[load] sampled {len(unique_contracts(examples))} contracts -> "
#           f"{len(examples)} questions ({len(answerable)} answerable, {len(unanswerable)} is_impossible)")

#     if not args.skip_ingest:
#         print("[ingest] embedding sampled contracts into Chroma (vector store only)")
#         ingest_cuad_contracts(unique_contracts(examples))
#     else:
#         print("[ingest] skipped (--skip-ingest)")

#     print("[pipeline] building graph + running scripted evaluation")
#     app = build_legal_agent_graph()

#     results = []
#     for i, example in enumerate(examples, start=1):
#         print(f"  [{i}/{len(examples)}] {example.contract_title!r} :: {example.clause_category}")
#         result = run_scripted_pipeline(
#             app,
#             thread_id=f"cuad-eval-{uuid.uuid4()}",
#             question=example.question,
#             collection_name=example.collection_name,
#             **call_kwargs_for(example),
#         )
#         results.append(result)

#     # Split results in lockstep with the examples split above.
#     answerable_results = [r for e, r in zip(examples, results) if not e.is_impossible]
#     unanswerable_results = [r for e, r in zip(examples, results) if e.is_impossible]

#     summary: dict = {"n_contracts": len(unique_contracts(examples)), "n_questions": len(examples)}

#     if unanswerable:
#         print("[score] absence-detection accuracy (is_impossible questions)")
#         absence_scored = score_absence_detection(unanswerable_results, unanswerable)
#         summary["absence_detection_accuracy"] = absence_detection_accuracy(absence_scored)
#         summary["absence_detection_n"] = len(absence_scored)
#         print(f"         {summary['absence_detection_accuracy']:.1%} over {len(absence_scored)} questions")

#     if answerable:
#         print("[score] RAGAS metrics (answerable questions)")
#         rows = build_ragas_rows(answerable_results, answerable)
#         ragas_result = run_ragas_evaluation(rows)
#         ragas_scores = {k: float(v) for k, v in ragas_result._repr_dict.items()} \
#             if hasattr(ragas_result, "_repr_dict") else dict(ragas_result)
#         summary["ragas"] = ragas_scores
#         summary["ragas_n"] = len(rows)
#         for metric, score in ragas_scores.items():
#             print(f"         {metric}: {score:.3f}")

#     out_path = Path(args.out)
#     out_path.parent.mkdir(parents=True, exist_ok=True)
#     with open(out_path, "w", encoding="utf-8") as f:
#         json.dump(summary, f, indent=2, default=str)
#     print(f"[done] summary written to {out_path}")


# if __name__ == "__main__":
#     main()
