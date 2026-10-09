#!/usr/bin/env python
"""
CLI demo for the full local pipeline: run from the project root, e.g.

    python -m scripts.run_demo ingest --file data/uploads/sample.pdf --vendor "ABC Ltd."
    python -m scripts.run_demo ask --question "..." --collection sample_pdf

Both subcommands pause mid-run for a real human-in-the-loop approval,
prompted right here in the terminal via input() — this is the same
interrupt()/Command(resume=...) mechanism the graphs use in production;
a real deployment would show the interrupt payload in a review UI instead
of a terminal prompt, but the pause/resume plumbing is identical.
"""

import argparse
import json
import os
import re
import sys

from langgraph.types import Command

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from legal_graphrag.ingestion.pdf_pipeline import (  # noqa: E402
    extract_page_content,
    detect_low_text_pages,
    ocr_pages,
    merge_ocr_results,
    compute_page_sections,
    chunk_pages,
    build_table_records,
    build_table_chunks,
    persist_metadata,
    embed_and_store,
    DEFAULT_METADATA_DIR,
)
from legal_graphrag.graphrag.langgraph_agent import (  # noqa: E402
    build_ingestion_graph,
)
from legal_graphrag.agents.legal_pipeline import build_legal_agent_graph  # noqa: E402
from legal_graphrag.retrieval.hybrid_search import invalidate_bm25_cache  # noqa: E402
from legal_graphrag.output_formatting import render_full_answer  # noqa: E402


def _prompt_for_decision(interrupt_payload: dict, edit_capable: bool = False) -> dict:
    print("\n=== HUMAN APPROVAL REQUIRED ===")
    print(json.dumps(interrupt_payload, indent=2, default=str))

    approved = input("\nApprove? [y/N]: ").strip().lower() == "y"
    reviewer = input("Reviewer name: ").strip() or "unknown"
    comments = input("Comments (optional): ").strip() or None

    decision = {"approved": approved, "reviewer": reviewer, "comments": comments}
    if edit_capable and approved:
        edited = input("Edit the answer before approving? Leave blank to accept as-is: ").strip()
        decision["edited_answer"] = edited or None
    return decision


def run_ingest(args: argparse.Namespace) -> None:
    document_name = os.path.basename(args.file)
    collection_name = args.collection or re.sub(r"\W+", "_", document_name)

    print(f"[1/4] Extracting text and tables from {document_name}")
    pages, raw_tables = extract_page_content(args.file)

    low_text_pages = detect_low_text_pages(pages)
    if low_text_pages:
        print(f"[2/4] Running OCR on {len(low_text_pages)} page(s): {low_text_pages}")
        ocr_results = ocr_pages(args.file, low_text_pages)
        pages = merge_ocr_results(pages, ocr_results)
    else:
        print("[2/4] No OCR needed")

    print("[3/4] Cleaning/chunking text and tables, embedding into the vector store")
    page_sections = compute_page_sections(pages)
    text_chunks = chunk_pages(pages, document_name, page_sections)
    table_records = build_table_records(raw_tables, document_name, page_sections)
    table_chunks = build_table_chunks(table_records, start_chunk_index=len(text_chunks))
    all_chunks = text_chunks + table_chunks

    persist_metadata(all_chunks, os.path.join(DEFAULT_METADATA_DIR, f"{document_name}.metadata.json"))
    embed_and_store(all_chunks, collection_name)
    invalidate_bm25_cache(collection_name)
    print(f"       vector collection '{collection_name}' populated with {len(all_chunks)} chunks")

    print("[4/4] Extracting clauses/relationships into Neo4j (this may take a while)")
    graphrag_chunks = [
        {"text": c.text, "page_start": c.page_start, "page_end": c.page_end, "section": c.section}
        for c in text_chunks
    ]

    app = build_ingestion_graph()
    config = {"configurable": {"thread_id": f"ingest-{document_name}"}}

    result = app.invoke(
        {
            "document_name": document_name,
            "contract_name": args.contract_name,
            "vendor_name": args.vendor,
            "text_chunks": graphrag_chunks,
        },
        config=config,
    )

    interrupt_payload = result["__interrupt__"][0].value
    decision = _prompt_for_decision(interrupt_payload)

    result = app.invoke(Command(resume=decision), config=config)
    print(f"\nFinal status: {result['status']}")


def _prompt_evidence_decision(interrupt_payload: dict) -> dict:
    print("\n=== HUMAN APPROVAL REQUIRED: EVIDENCE ===")
    print(json.dumps(interrupt_payload, indent=2, default=str))
    proceed = input("\nProceed to synthesis with this evidence? [y/N]: ").strip().lower() == "y"
    reviewer = input("Reviewer name: ").strip() or "unknown"
    comments = input("Comments (optional): ").strip() or None
    return {"proceed": proceed, "reviewer": reviewer, "comments": comments}


def _prompt_answer_decision(interrupt_payload: dict) -> dict:
    print("\n=== HUMAN APPROVAL REQUIRED: ANSWER ===")
    print(json.dumps(interrupt_payload, indent=2, default=str))
    action = input("\nAction [approve/revise/reject]: ").strip().lower() or "reject"
    reviewer = input("Reviewer name: ").strip() or "unknown"
    decision = {"action": action, "reviewer": reviewer}
    if action == "revise":
        decision["comments"] = input("Feedback for revision (required): ").strip()
    else:
        decision["comments"] = input("Comments (optional): ").strip() or None
    if action == "approve":
        edited = input("Edit the answer before approving? Leave blank to accept as-is: ").strip()
        decision["edited_answer"] = edited or None
    return decision


def run_ask(args: argparse.Namespace) -> None:
    app = build_legal_agent_graph()
    config = {"configurable": {"thread_id": f"query-{abs(hash(args.question))}"}}

    state = app.invoke(
        {"question": args.question, "collection_name": args.collection, "metadata_filter": None},
        config=config,
    )

    # Two sequential human checkpoints: evidence, then answer (with a
    # possible "revise" loop back to the answer checkpoint) — see
    # agents/legal_pipeline.py's module docstring for the full state machine.
    while "__interrupt__" in state:
        interrupt_payload = state["__interrupt__"][0].value
        if interrupt_payload["type"] == "evidence_approval_request":
            decision = _prompt_evidence_decision(interrupt_payload)
        else:
            decision = _prompt_answer_decision(interrupt_payload)
        state = app.invoke(Command(resume=decision), config=config)

    if state.get("final_answer"):
        print()
        print(render_full_answer(state, verbose=args.verbose))
    else:
        print(f"\nNo approved answer (status: {state.get('status')}); nothing to show.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Legal GraphRAG demo CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest", help="Ingest a PDF: vector store + GraphRAG extraction + approval")
    ingest.add_argument("--file", required=True, help="Path to the PDF")
    ingest.add_argument("--vendor", default=None, help="Vendor/counterparty name for this contract")
    ingest.add_argument("--contract-name", default=None, help="Human-readable contract name")
    ingest.add_argument("--collection", default=None, help="Chroma collection name (default: derived from filename)")
    ingest.set_defaults(func=run_ingest)

    ask = subparsers.add_parser("ask", help="Ask a hybrid vector+graph question, with human approval on the answer")
    ask.add_argument("--question", required=True)
    ask.add_argument("--collection", required=True, help="Chroma collection name to search")
    ask.add_argument("--verbose", action="store_true",
                      help="Expand the 'Retrieval & reasoning details' section inline instead of collapsing it")
    ask.set_defaults(func=run_ask)

    return parser


def main() -> None:
    from legal_graphrag.resources import preload_models
    print("Preloading models (embedder, reranker)...")
    preload_models(include_neo4j=False)  # Neo4j connects lazily on first actual use

    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
