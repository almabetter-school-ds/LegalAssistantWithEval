"""
Renders a SynthesizedAnswer (agents/prompts.py) — surfaced through
LegalAgentState's `final_structured_answer`/`draft_...` fields (see
agents/legal_pipeline.py) — into user-facing output.

Three output shapes, all built from the SAME structured dict so they never
drift out of sync with each other:

  - format_answer_text()  — the plain "Document / Question / Answer /
    Evidence / Confidence / Source" text block.
  - format_answer_card()  — the boxed terminal/UI card (Answer, Evidence,
    Source sections, box-drawing characters).
  - format_technical_details() — retrieval + reasoning internals (route,
    retrieved chunks, evidence auditor verdict, citations, revision count)
    behind a collapsible "▶ Retrieval & reasoning details" section, so the
    headline answer stays short and the audit trail stays one click away
    rather than gone. Markdown output uses a real collapsible
    <details><summary> block (renders collapsed by default in GitHub,
    most markdown viewers, and Streamlit's st.markdown); plain-text/CLI
    output uses a "▶ ..." teaser line with a --verbose flag to expand
    (see scripts/run_demo.py).

None of this changes what the pipeline computes — it's a pure rendering
layer over `LegalAgentState`.
"""

from __future__ import annotations

from typing import Optional


def _source_line(structured: dict) -> str:
    document = structured.get("document") or "Unknown document"
    section = structured.get("source_section")
    page = structured.get("source_page")

    location = section or (f"Page {page}" if page else None)
    if section and page:
        location = f"{section}, Page {page}"

    return f"{document} · {location}" if location else document


def format_answer_text(structured: dict, question: str) -> str:
    """
    The plain template:

        Document: DOMINIADVISORTRUST...
        Question: What is the agreement date?
        Answer:
        The agreement date is February 18, 2005.
        Evidence:
        The agreement identifies February 18, 2005 as the date of the Sponsorship Agreement.
        Confidence: High
        Source: Section/Clause/Page X
    """
    document = structured.get("document") or "Unknown document"
    answer = structured.get("answer", "").strip()
    evidence = structured.get("evidence", "").strip()
    confidence = structured.get("confidence", "Low")

    lines = [
        f"Document: {document}",
        f"Question: {question}",
        "Answer:",
        answer,
        "Evidence:",
        evidence or "(no grounding excerpt available)",
        f"Confidence: {confidence}",
        f"Source: {_source_line(structured)}",
    ]
    return "\n".join(lines)


def _wrap_box(lines: list[str], width: int = 47) -> str:
    """Box-draws `lines` (already word-wrapped by the caller, to width - 4 chars) into a fixed-width ASCII card."""
    content_width = width - 4  # 1 space padding + border char on each side
    top = "┌" + "─" * (width - 2) + "┐"
    bottom = "└" + "─" * (width - 2) + "┘"
    sep = "├" + "─" * (width - 2) + "┤"

    out = [top]
    for line in lines:
        if line == "__SEP__":
            out.append(sep)
            continue
        out.append(f"│ {line.ljust(content_width)} │")
    out.append(bottom)
    return "\n".join(out)


def _wrap_text(text: str, width: int) -> list[str]:
    import textwrap
    return textwrap.wrap(text, width=width, break_long_words=True, break_on_hyphens=False) or [""]


def format_answer_card(structured: dict, width: int = 49) -> str:
    """
    The boxed UI card:

        ┌─────────────────────────────────────────────┐
        │ Answer                                       │
        ├─────────────────────────────────────────────┤
        │ February 18, 2005                            │
        │                                               │
        │ Confidence: High                              │
        ├─────────────────────────────────────────────┤
        │ Evidence                                     │
        │                                               │
        │ "..."                                        │
        ├─────────────────────────────────────────────┤
        │ Source                                       │
        │ Sponsorship Agreement · Page 1                │
        └─────────────────────────────────────────────┘
    """
    text_width = width - 4
    answer = structured.get("answer", "").strip()
    evidence = structured.get("evidence", "").strip() or "(no grounding excerpt available)"
    confidence = structured.get("confidence", "Low")

    lines = ["Answer", "__SEP__"]
    lines += _wrap_text(answer, text_width)
    lines.append("")
    lines.append(f"Confidence: {confidence}")
    lines.append("__SEP__")
    lines.append("Evidence")
    lines.append("")
    lines += _wrap_text(f'"{evidence}"', text_width)
    lines.append("__SEP__")
    lines.append("Source")
    lines += _wrap_text(_source_line(structured), text_width)

    return _wrap_box(lines, width=width)


def format_technical_details(state: dict, markdown: bool = True) -> str:
    """
    Retrieval + reasoning internals that back the headline answer:
    route/alpha, retrieved chunk ids + scores, the evidence auditor's full
    verdict (gaps/contradictions), citations, and how many revision rounds
    it took to get here. Renders as a collapsible <details> block by
    default (markdown=True); pass markdown=False for a plain indented block
    (used by the CLI's --verbose flag).
    """
    route = state.get("route", "?")
    alpha = state.get("alpha")
    hybrid_hits = state.get("hybrid_hits", []) or []
    graph_hits = state.get("graph_hits", []) or []
    verdict = state.get("evidence_verdict", {}) or {}
    citations = state.get("draft_citations") or (state.get("final_structured_answer") or {}).get("citations", [])
    revision_count = state.get("answer_revision_count", 0)

    body_lines = [
        f"Route: {route}" + (f" (alpha={alpha})" if alpha is not None else ""),
        f"Revision rounds: {revision_count}",
        "",
        f"Evidence auditor: sufficient={verdict.get('sufficient')}",
    ]
    if verdict.get("reasoning"):
        body_lines.append(f"  reasoning: {verdict['reasoning']}")
    for gap in verdict.get("gaps", []) or []:
        body_lines.append(f"  gap: {gap}")
    for c in verdict.get("contradictions", []) or []:
        body_lines.append(f"  contradiction: {c}")

    body_lines.append("")
    body_lines.append(f"Retrieved chunks ({len(hybrid_hits)} hybrid, {len(graph_hits)} graph):")
    for hit in hybrid_hits[:5]:
        meta = hit.get("metadata", {}) or {}
        score = hit.get("rerank_score", hit.get("dense_distance", hit.get("bm25_score")))
        snippet = (hit.get("text", "") or "")[:160].replace("\n", " ")
        body_lines.append(
            f"  - [{meta.get('document_name', '?')} p.{meta.get('page_start', '?')}] "
            f"(score={score}) {snippet}..."
        )
    if len(hybrid_hits) > 5:
        body_lines.append(f"  ... and {len(hybrid_hits) - 5} more")

    if citations:
        body_lines.append("")
        body_lines.append("Citations: " + "; ".join(citations))

    body = "\n".join(body_lines)

    if not markdown:
        indented = "\n".join(f"    {line}" for line in body.split("\n"))
        return f"\u25b6 Retrieval & reasoning details (--verbose to expand)\n{indented}"

    return (
        "<details>\n"
        "<summary>▶ Retrieval &amp; reasoning details</summary>\n\n"
        f"```\n{body}\n```\n"
        "</details>"
    )


def render_full_answer(state: dict, verbose: bool = False, markdown: bool = False) -> str:
    """
    Convenience entrypoint: card (or plain text block) + technical details,
    from a full LegalAgentState (or any dict with `final_structured_answer`/
    `question` and the retrieval/evidence fields). Prefer this over calling
    the pieces separately unless you need just one.

    verbose: for markdown=False output, include the technical details block
      inline instead of the collapsed teaser line (there's no real collapse
      mechanism in a plain terminal — see scripts/run_demo.py's --verbose flag).
    markdown: use format_answer_text() + a real collapsible <details> block,
      suitable for a markdown-rendering UI, instead of the ASCII card.
    """
    structured = state.get("final_structured_answer") or {}
    question = state.get("question", "")

    if not structured or not structured.get("answer"):
        return "No approved answer is available for this question."

    if markdown:
        head = format_answer_text(structured, question)
        return f"{head}\n\n{format_technical_details(state, markdown=True)}"

    card = format_answer_card(structured)
    if verbose:
        return f"{card}\n\n{format_technical_details(state, markdown=False)}"
    return f"{card}\n\n\u25b6 Retrieval & reasoning details (rerun with --verbose to expand)"
