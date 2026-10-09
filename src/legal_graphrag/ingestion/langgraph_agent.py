"""
LangGraph agentic version of the PDF -> RAG ingestion pipeline (text + tables).

This wraps the same stages implemented in pdf_rag_pipeline.py as nodes in a
LangGraph StateGraph, instead of one linear function. The point of doing this
as a graph rather than a straight-line script is the conditional routing: the
OCR node is only visited if the detection node actually found low-text
pages — on a fully-native-text PDF, the graph skips straight past OCR to
chunking. That branch is decided at runtime from the shared state, which is
the "agentic" part LangGraph adds on top of the pipeline.

------------------------------------------------------------------------
Run this in a Colab cell FIRST, in addition to pdf_rag_pipeline.py's setup:

    !apt-get -qq update && apt-get -qq install -y poppler-utils tesseract-ocr

    # --no-deps avoids pip downgrading Colab's preinstalled torch/torchvision
    # to satisfy sentence-transformers' pin — see pdf_rag_pipeline.py for why.
    !pip -q install pdfplumber pdf2image pytesseract chromadb langgraph
    !pip -q install sentence-transformers --no-deps
    !pip -q install --upgrade-strategy only-if-needed \
        transformers tokenizers huggingface-hub safetensors scikit-learn scipy Pillow tqdm

    # Then: Runtime -> Restart session before running any code below.

Then, with pdf_rag_pipeline.py in the same directory/session:

    from langgraph_pdf_rag_agent import build_graph

    graph = build_graph()
    final_state = graph.invoke({
        "pdf_path": "data/uploads/my_document.pdf",   # or an absolute /content/... path in Colab
        "query": "What is the termination clause?",
    })
    print(final_state["query_results"])
------------------------------------------------------------------------
"""

from __future__ import annotations

import os
import re
from typing import Optional, TypedDict

from langgraph.graph import StateGraph, END

from .pdf_pipeline import (
    PageRecord,
    TableRecord,
    ChunkRecord,
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
    query_collection,
    DEFAULT_METADATA_DIR,
)


# ---------------------------------------------------------------------------
# Shared graph state
# ---------------------------------------------------------------------------

class PipelineState(TypedDict, total=False):
    # inputs
    pdf_path: str
    document_name: str
    collection_name: str
    metadata_output_path: str
    query: Optional[str]

    # intermediate state, populated as nodes run
    pages: list[PageRecord]
    raw_tables: dict[int, list]
    low_text_pages: list[int]
    ocr_results: dict[int, str]
    page_sections: dict[int, Optional[str]]
    text_chunks: list[ChunkRecord]
    table_records: list[TableRecord]
    table_chunks: list[ChunkRecord]
    chunks: list[ChunkRecord]
    metadata_path: str

    # outputs
    collection: object          # chromadb Collection (not serializable, kept in-memory only)
    embedder: object            # SentenceTransformer instance
    query_results: list[dict]


# ---------------------------------------------------------------------------
# Nodes — each one reads what it needs from state and returns only the keys
# it updates, which LangGraph merges into the running state.
# ---------------------------------------------------------------------------

def extract_node(state: PipelineState) -> dict:
    document_name = os.path.basename(state["pdf_path"])
    print(f"[extract] {document_name}")
    pages, raw_tables = extract_page_content(state["pdf_path"])
    num_raw_tables = sum(len(t) for t in raw_tables.values())
    print(f"[extract] {num_raw_tables} table(s) found across {len(raw_tables)} page(s)")
    return {"document_name": document_name, "pages": pages, "raw_tables": raw_tables}


def detect_low_text_node(state: PipelineState) -> dict:
    low_text_pages = detect_low_text_pages(state["pages"])
    print(f"[detect] {len(low_text_pages)} of {len(state['pages'])} pages need OCR: {low_text_pages}")
    return {"low_text_pages": low_text_pages}


def ocr_node(state: PipelineState) -> dict:
    print(f"[ocr] running OCR on pages {state['low_text_pages']}")
    ocr_results = ocr_pages(state["pdf_path"], state["low_text_pages"])
    pages = merge_ocr_results(state["pages"], ocr_results)
    return {"ocr_results": ocr_results, "pages": pages}


def chunk_node(state: PipelineState) -> dict:
    page_sections = compute_page_sections(state["pages"])
    text_chunks = chunk_pages(state["pages"], state["document_name"], page_sections)
    print(f"[chunk] {len(text_chunks)} text chunks produced")
    return {"page_sections": page_sections, "text_chunks": text_chunks}


def table_node(state: PipelineState) -> dict:
    table_records = build_table_records(state["raw_tables"], state["document_name"], state["page_sections"])
    table_chunks = build_table_chunks(table_records, start_chunk_index=len(state["text_chunks"]))
    print(f"[table] {len(table_chunks)} table chunks produced")
    chunks = state["text_chunks"] + table_chunks
    return {"table_records": table_records, "table_chunks": table_chunks, "chunks": chunks}


def persist_node(state: PipelineState) -> dict:
    metadata_output_path = state.get(
        "metadata_output_path",
        os.path.join(DEFAULT_METADATA_DIR, f"{state['document_name']}.metadata.json"),
    )
    metadata_path = persist_metadata(state["chunks"], metadata_output_path)
    print(f"[persist] metadata written to {metadata_path}")
    return {"metadata_path": metadata_path}


def embed_store_node(state: PipelineState) -> dict:
    collection_name = state.get(
        "collection_name", re.sub(r"\W+", "_", state["document_name"])
    )
    collection, embedder = embed_and_store(state["chunks"], collection_name)
    print(f"[embed_store] stored in collection '{collection_name}'")
    return {"collection_name": collection_name, "collection": collection, "embedder": embedder}


def query_node(state: PipelineState) -> dict:
    hits = query_collection(state["collection"], state["embedder"], state["query"])
    print(f"[query] '{state['query']}' -> {len(hits)} results")
    for h in hits:
        m = h["metadata"]
        tag = "[TABLE]" if m["content_type"] == "table" else "[TEXT] "
        print(f"    {tag} p.{m['page_start']}-{m['page_end']} [{m['section'] or 'no section'}] "
              f"(dist={h['distance']:.4f}): {h['text'][:120]}...")
    return {"query_results": hits}


# ---------------------------------------------------------------------------
# Conditional routing
# ---------------------------------------------------------------------------

def route_after_detect(state: PipelineState) -> str:
    """Skip the OCR node entirely if no pages need it."""
    return "ocr" if state["low_text_pages"] else "chunk"


def route_after_embed(state: PipelineState) -> str:
    """Only run the query node if the caller actually supplied a query."""
    return "query" if state.get("query") else END


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_graph():
    graph = StateGraph(PipelineState)

    graph.add_node("extract", extract_node)
    graph.add_node("detect", detect_low_text_node)
    graph.add_node("ocr", ocr_node)
    graph.add_node("chunk", chunk_node)
    graph.add_node("table", table_node)
    graph.add_node("persist", persist_node)
    graph.add_node("embed_store", embed_store_node)
    graph.add_node("query", query_node)

    graph.set_entry_point("extract")
    graph.add_edge("extract", "detect")

    # Branch: only visit OCR if detect_node actually found low-text pages.
    graph.add_conditional_edges("detect", route_after_detect, {"ocr": "ocr", "chunk": "chunk"})
    graph.add_edge("ocr", "chunk")

    graph.add_edge("chunk", "table")
    graph.add_edge("table", "persist")
    graph.add_edge("persist", "embed_store")

    # Branch: only run a retrieval query if one was provided as input.
    graph.add_conditional_edges("embed_store", route_after_embed, {"query": "query", END: END})
    graph.add_edge("query", END)

    return graph.compile()


if __name__ == "__main__":
    # Example (adjust the path to a PDF uploaded in your Colab session):
    app = build_graph()
    final_state = app.invoke({
        "pdf_path": "data/uploads/sample.pdf",
        "query": "What is the termination clause?",
    })
    print("\nDone. Text chunks:", len(final_state["text_chunks"]),
          "| Table chunks:", len(final_state["table_chunks"]),
          "| OCR pages:", len(final_state["low_text_pages"]))
