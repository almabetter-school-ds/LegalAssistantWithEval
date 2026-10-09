"""
Scripted (auto-approve) reviewer — drives the REAL pipeline graph through
both human-in-the-loop interrupt()s for CUAD-scale batch evaluation.

Why this instead of calling hybrid_search()/synthesize_legal_answer()
directly: bypassing the graph would evaluate a different code path than the
one actually deployed (no router, no auditor, no audit trail, no
checkpoint payload shape) — the point of using the real graph, resumed with
Command(resume=...), is that the trace and metrics reflect the same
pipeline a human reviewer uses, just with the two interrupt()s answered by
a script instead of a person. This is evaluation-only: nowhere in the main
application is a human checkpoint auto-approved.

Policy — deliberately simple, and deliberately NOT the same as "the
pipeline decided the evidence/answer was good":
  - Evidence checkpoint: always resumes with proceed=True. The point of
    RAGAS's context_precision/context_recall/faithfulness metrics is to
    MEASURE retrieval and answer quality — if the scripted reviewer only
    let "good" evidence through, we'd never see the retrieval failures
    those metrics exist to catch. The auditor's own `sufficient` verdict is
    still captured in the result (see `evidence_sufficient` below) so you
    can slice RAGAS scores by "auditor thought this was sufficient" vs not,
    as a sanity check on the auditor itself.
  - Answer checkpoint: always resumes with action="approve" on the first
    draft — no scripted "revise" loop. Simulating a revise round would mean
    scripting fake reviewer feedback, which isn't a real reviewer judgment
    and would just be evaluating the revision LLM call against comments a
    human never actually gave.

If you want a stricter policy (e.g. auto-reject when the auditor's
`sufficient` is False) for a different evaluation, swap in a different
`EvidencePolicy`/`AnswerPolicy` callable below — the harness in
ragas_eval.py only depends on the two function signatures, not this
specific always-approve implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from langgraph.types import Command

from ..tracing import traceable

EvidencePolicy = Callable[[dict], dict]   # interrupt payload -> resume decision dict
AnswerPolicy = Callable[[dict], dict]     # interrupt payload -> resume decision dict

REVIEWER_NAME = "scripted-eval-reviewer"


def always_approve_evidence(payload: dict) -> dict:
    return {
        "proceed": True,
        "reviewer": REVIEWER_NAME,
        "comments": "auto-approved for batch evaluation",
    }


def always_approve_answer(payload: dict) -> dict:
    return {
        "action": "approve",
        "reviewer": REVIEWER_NAME,
        "comments": "auto-approved for batch evaluation",
    }


def reject_evidence_if_insufficient(payload: dict) -> dict:
    """Alternate, stricter evidence policy: only proceed if the auditor said `sufficient`."""
    sufficient = bool(payload.get("evidence_verdict", {}).get("sufficient"))
    return {
        "proceed": sufficient,
        "reviewer": REVIEWER_NAME,
        "comments": "auto-approved for batch evaluation" if sufficient
        else "auto-rejected: auditor flagged evidence as insufficient",
    }


@dataclass
class ScriptedRunResult:
    query_job_id: Optional[str]
    route: Optional[str]
    hybrid_hits: list[dict]
    graph_hits: list[dict]
    evidence_verdict: dict
    evidence_decision: dict
    final_answer: Optional[str]
    status: Optional[str]
    answer_decision: dict
    raw_state: dict


@traceable(name="evaluation.run_scripted_pipeline", run_type="chain")
def run_scripted_pipeline(
    app,
    thread_id: str,
    question: str,
    collection_name: str,
    metadata_filter: Optional[dict] = None,
    force_route: Optional[str] = "hybrid",
    evidence_policy: EvidencePolicy = always_approve_evidence,
    answer_policy: AnswerPolicy = always_approve_answer,
    max_steps: int = 6,
) -> ScriptedRunResult:
    """
    Invokes the compiled legal_pipeline graph (`app`, from
    build_legal_agent_graph()) for a single question, and resumes past both
    interrupt()s automatically using the given policies, instead of waiting
    for a human to call app.invoke(Command(resume=...)) interactively.

    force_route="hybrid" by default (see cuad_ingest.py's module docstring
    for why: only the vector store is populated for CUAD eval, not Neo4j's
    clause graph) — pass None to let the router classify normally.
    """
    config = {"configurable": {"thread_id": thread_id}}
    state: dict[str, Any] = app.invoke(
        {
            "question": question,
            "collection_name": collection_name,
            "metadata_filter": metadata_filter,
            "force_route": force_route,
        },
        config=config,
    )

    steps = 0
    while "__interrupt__" in state and steps < max_steps:
        steps += 1
        interrupt_payload = state["__interrupt__"][0].value

        if interrupt_payload.get("type") == "evidence_approval_request":
            decision = evidence_policy(interrupt_payload)
        elif interrupt_payload.get("type") == "answer_approval_request":
            decision = answer_policy(interrupt_payload)
        else:  # pragma: no cover - defensive: an interrupt type this harness doesn't know about
            raise RuntimeError(f"scripted_reviewer: unrecognized interrupt payload type: {interrupt_payload}")

        state = app.invoke(Command(resume=decision), config=config)

    if "__interrupt__" in state:
        raise RuntimeError(
            f"scripted_reviewer: pipeline still paused after {max_steps} resumes "
            f"(thread_id={thread_id}) — increase max_steps or check for a policy bug."
        )
    print("[answer]", state.get("final_answer"))
    return ScriptedRunResult(
        query_job_id=state.get("query_job_id"),
        route=state.get("route"),
        hybrid_hits=state.get("hybrid_hits", []),
        graph_hits=state.get("graph_hits", []),
        evidence_verdict=state.get("evidence_verdict", {}),
        evidence_decision=state.get("evidence_decision", {}),
        final_answer=state.get("final_answer"),
        status=state.get("status"),
        answer_decision=state.get("answer_decision", {}),
        raw_state=state,
    )
