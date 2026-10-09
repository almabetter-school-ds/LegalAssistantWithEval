"""
Vector-store-only ingestion of CUAD contracts, for evaluation.

Deliberately skips graphrag/extraction.py's clause/conflict/risk extraction
and neo4j_store.py's document/clause writes: CUAD-scale ingestion (hundreds
of contracts) through that path would mean many extra LLM calls per
contract before evaluation even starts, and RAGAS's metrics (faithfulness,
context precision/recall, answer correctness) only assess retrieval +
answer quality — which the `hybrid` route (Chroma + BM25) covers on its
own. Neo4j is still used at query time (see agents/legal_pipeline.py's
start_job/audit-trail writes), just never populated with CUAD clause data —
which is also why evaluation/ragas_eval.py forces `force_route="hybrid"`
rather than letting the router pick "graph" and hit an empty graph.

Reuses ingestion/pdf_pipeline.py's text-cleaning/chunking/embedding
machinery (clean_text, chunk_pages, embed_and_store) directly against
CUAD's plain-text `context` field, via a single synthetic PageRecord —
CUAD contracts arrive as already-extracted text, so the PDF/OCR-specific
stages (pdfplumber extraction, OCR fallback, table extraction) don't apply
and are skipped entirely.
"""

from __future__ import annotations

from ..ingestion.pdf_pipeline import PageRecord, chunk_pages, embed_and_store
from ..retrieval.contract_metadata import build_chunk_metadata
from ..retrieval.hybrid_search import invalidate_bm25_cache
from ..tracing import traceable
from .cuad_loader import CUADExample


def _contract_extra_metadata(example: CUADExample) -> dict:
    """
    Builds the same flattened metadata shape contract_metadata.py produces
    for a real PDF ingest — contract_type comes straight from CUAD's title
    (see cuad_loader._parse_contract_type), so no LLM call is needed to
    extract it here. Party names, dates, and monetary value aren't in
    CUAD's title, so those default to empty/zero, same as
    build_chunk_metadata() does for any field an extraction pass couldn't
    determine — this only affects metadata_filter-based tests, not the
    plain question-answering ones CUAD's `qas` are built around.
    """
    return build_chunk_metadata({"contract_type": example.contract_type, "parties": []})


@traceable(name="evaluation.ingest_cuad_contract", run_type="chain")
def ingest_cuad_contract(example: CUADExample) -> str:
    """
    Chunks and embeds ONE CUAD contract's full text into its own Chroma
    collection (named `example.collection_name`). Idempotent: embed_and_store
    upserts by chunk_id, so re-running this for a contract already ingested
    just re-embeds the same chunks under the same ids.

    Returns the collection_name it wrote to.
    """
    page = PageRecord(page_number=1, text=example.contract_text, source="cuad_text", char_count=len(example.contract_text))
    chunks = chunk_pages([page], document_name=example.contract_title, page_sections={1: None})

    embed_and_store(
        chunks,
        collection_name=example.collection_name,
        extra_metadata=_contract_extra_metadata(example),
    )
    invalidate_bm25_cache(example.collection_name)  # so hybrid_search's next call rebuilds the BM25 index
    return example.collection_name


def ingest_cuad_contracts(contracts_by_collection: dict[str, CUADExample]) -> list[str]:
    """
    Ingests one contract per (unique) collection — see
    cuad_loader.unique_contracts() for how to build this dict from a full
    example list. Returns the list of collection_names ingested.
    """
    collection_names = []
    for i, (collection_name, example) in enumerate(contracts_by_collection.items(), start=1):
        print(f"[{i}/{len(contracts_by_collection)}] ingesting {example.contract_title!r} "
              f"({example.contract_type}) -> collection '{collection_name}'")
        ingest_cuad_contract(example)
        collection_names.append(collection_name)
    return collection_names
