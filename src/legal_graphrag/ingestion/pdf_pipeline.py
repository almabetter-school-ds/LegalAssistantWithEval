"""
pdf_pipeline.py
===============

Hierarchy-aware PDF ingestion pipeline.

Document hierarchy:

    Document
        |
        +--> Page
                |
                +--> Section
                        |
                        +--> Clause
                                |
                                +--> Subclause
                                        |
                                        +--> Chunk / Table

The ingestion pipeline preserves the hierarchy in Chroma metadata.

Supported metadata:

    document_name
    page_start
    page_end

    section

    clause_number
    clause_title

    parent_clause

    subclause_number
    subclause_title

    content_type
    sources

Document-level metadata:

    contract_type
    parties
    governing_law_country

    effective_date_epoch
    end_date_epoch

    monetary_value

IMPORTANT TABLE RULE
--------------------

Tables are kept intact as logical chunks. A table is NOT split merely
because it is large. Instead: content_type = "table", and hierarchy
metadata (whatever clause/subclause was "open" at that point in the page)
is propagated to it.

HOW CLAUSE/SUBCLAUSE METADATA IS ACTUALLY POPULATED
-----------------------------------------------------
detect_clause_headings() scans each page's normalized text for
heading-shaped patterns ("12.3 Indemnification", "ARTICLE 12 —
TERMINATION", "Section 13 Governing Law") via regex. A heading with no
dot in its number ("12") is a CLAUSE; a heading with a dot ("12.3",
"12.3.1") is a SUBCLAUSE of the clause named by its first component.
_split_page_into_hierarchy_blocks() then splits that page's text at each
detected heading's offset and stamps every resulting block with the full
hierarchy state "open" at that point — a running _HierarchyCursor is
threaded across the whole page loop in run_pipeline() so a clause that
spans multiple pages (very common) carries forward correctly instead of
resetting at each page boundary.

This is a regex heuristic, not authoritative structure: it can miss
unconventional heading formats (roman numerals, lettered subclauses like
"12(a)"), and text between two detected headings is attributed to
whichever clause/subclause opened first — acceptable for retrieval
boosting and citation, not for anything requiring guaranteed-exact
clause attribution.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

import pdfplumber
import pytesseract
from pdf2image import convert_from_path

from ..resources import get_chroma_collection, get_embedder
from ..tracing import traceable

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MIN_CHARS_FOR_TEXT_PAGE = 20   # pages with fewer extracted characters than this are treated as "no text"
OCR_DPI = 300                  # resolution used when rasterizing a page for OCR — higher = better OCR, slower
CHUNK_SIZE_CHARS = 1000        # target characters per chunk
CHUNK_OVERLAP_CHARS = 150      # overlap between consecutive chunks, preserves context across chunk boundaries
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

# Resolved relative to the project root (two levels up from src/legal_graphrag/ingestion/)
# so it works the same whether run from VS Code, a terminal, or Colab (with /content mounted).
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
CHROMA_PERSIST_DIR = os.path.join(_PROJECT_ROOT, "data", "chroma_db")
DEFAULT_METADATA_DIR = os.path.join(_PROJECT_ROOT, "data", "metadata")

TEXT_CONTENT_TYPE = "text"
TABLE_CONTENT_TYPE = "table"

DEFAULT_CHUNK_SIZE = CHUNK_SIZE_CHARS
DEFAULT_CHUNK_OVERLAP = CHUNK_OVERLAP_CHARS
DEFAULT_COLLECTION_NAME = "legal_knowledge_base"

_HEADING_RE = re.compile(r"^[A-Z0-9][A-Za-z0-9 ,.:\-()]{2,80}$")


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class PageRecord:
    """One PDF page after native extraction or OCR."""
    page_number: int
    text: str
    source: str
    char_count: int
    section: Optional[str] = None


@dataclass
class TableRecord:
    """One intact table extracted from a PDF page."""
    table_id: str
    document_name: str
    page_number: int
    table_index: int
    section: Optional[str]
    markdown: str
    num_rows: int
    num_cols: int


@dataclass
class ChunkRecord:
    """Backward-compatible chunk representation used by older callers (see legacy chunk_pages() below)."""
    chunk_id: str
    document_name: str
    chunk_index: int
    text: str
    page_start: int
    page_end: int
    section: Optional[str]
    clause_number: Optional[str] = None
    clause_title: Optional[str] = None
    parent_clause: Optional[str] = None
    subclause_number: Optional[str] = None
    subclause_title: Optional[str] = None
    content_type: str = TEXT_CONTENT_TYPE
    sources: list[str] = field(default_factory=list)


@dataclass
class HierarchyContext:
    """Current location in the document hierarchy, as process_blocks() walks a block list."""
    document_name: str = ""
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    section: Optional[str] = None
    clause_number: Optional[str] = None
    clause_title: Optional[str] = None
    parent_clause: Optional[str] = None
    subclause_number: Optional[str] = None
    subclause_title: Optional[str] = None
    sources: list[str] = field(default_factory=list)


@dataclass
class DocumentMetadata:
    """Document-level metadata propagated to every chunk."""
    contract_type: Optional[str] = None
    parties: Optional[str] = None
    governing_law_country: Optional[str] = None
    effective_date_epoch: Optional[int] = None
    end_date_epoch: Optional[int] = None
    monetary_value: Optional[float] = None


@dataclass
class PipelineChunk:
    """Final logical chunk before vector storage."""
    text: str
    metadata: dict[str, Any]
    chunk_index: int = 0


@dataclass
class _HierarchyCursor:
    """
    Tracks the clause/subclause "currently open" as run_pipeline() walks
    the document page by page — a plain mutable holder threaded through
    _split_page_into_hierarchy_blocks() calls so a clause spanning
    multiple pages carries forward instead of resetting at each page
    boundary. Not the same as HierarchyContext (which process_blocks()
    owns, driven by fully-formed block dicts) — this is the earlier stage
    that PRODUCES those dicts from raw page text.
    """
    clause_number: Optional[str] = None
    clause_title: Optional[str] = None
    subclause_number: Optional[str] = None
    subclause_title: Optional[str] = None


# ============================================================================
# TEXT NORMALIZATION
# ============================================================================

def normalize_text(text: str) -> str:
    """
    Normalize extracted PDF text while preserving legal meaning: repairs
    hyphenated line-break words, normalizes line endings, collapses
    horizontal whitespace and excessive blank lines. Deliberately does
    NOT collapse single newlines to spaces (unlike this file's predecessor)
    — real line breaks are kept intact, which is what lets
    detect_clause_headings() below anchor on them.
    """
    if not text:
        return ""

    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # dehyphenate across line breaks, e.g. "contrac-\ntual"
    text = text.replace("\x00", " ")
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)            # collapse repeated spaces/tabs
    text = re.sub(r"\n{3,}", "\n\n", text)          # collapse 3+ blank lines to 1
    text = "".join(ch for ch in text if ch.isprintable() or ch in "\n\t")
    return text.strip()


# ============================================================================
# CLAUSE NUMBER NORMALIZATION
# ============================================================================

def normalize_clause_number(value: Any) -> Optional[str]:
    """Normalize values such as 12 / "12.3" / "12.3." into a clean string, or None."""
    if value is None:
        return None
    value = str(value).strip()
    if not value:
        return None
    return value.rstrip(".")


# ============================================================================
# CLAUSE / SUBCLAUSE HEADING DETECTION
# ============================================================================
#
# A heading with NO dot in its number ("12", "ARTICLE 12") is a CLAUSE.
# A heading WITH a dot ("12.3", "12.3.1") is a SUBCLAUSE, whose parent
# clause is its first dot-separated component.

_HEADING_STOP_WORDS = (
    "The|This|That|These|Those|Each|Any|All|It|Such|Either|Both|No|Notwithstanding|"
    "Subject|Except|Unless|If|In|For|Upon|Where|When|As"
)

_CLAUSE_HEADING_RE = re.compile(
    r"(?:^|\n|\.\s)"                                               # start of text, any line break, or after a sentence
    r"(?:(?i:Section|Article|Clause)\s+)?"                         # optional leading label, case-insensitive
                                                                    # (contracts commonly write "ARTICLE 12" in caps)
    r"(\d{1,3}(?:\.\d{1,3}){0,3})\.?\s+"                           # clause/subclause number, e.g. "12", "12.3"
    r"([A-Z][A-Za-z0-9(),/&'\u2019-]*"                             # first title word
    rf"(?:\s+(?!(?:{_HEADING_STOP_WORDS})\b)[A-Z][A-Za-z0-9(),/&'\u2019-]*){{0,4}})"  # up to 4 more, stoplist-guarded
    r"(?=[\s.]|$)"
)


def detect_clause_headings(text: str) -> list[tuple[int, str, str, str]]:
    """
    Returns [(start_offset, level, number, title), ...] sorted by offset,
    where level is "clause" (number has no dot) or "subclause" (number
    has one or more dots). See module docstring for the heuristic's scope.
    """
    headings = []
    for m in _CLAUSE_HEADING_RE.finditer(text):
        number = m.group(1)
        title = m.group(2).strip()
        level = "clause" if "." not in number else "subclause"
        headings.append((m.start(), level, number, title))
    return headings


def _parent_clause_number(subclause_number: Optional[str]) -> Optional[str]:
    if not subclause_number or "." not in subclause_number:
        return None
    return subclause_number.split(".", 1)[0]


def _split_page_into_hierarchy_blocks(
    page: "PageRecord",
    section: Optional[str],
    cursor: _HierarchyCursor,
    sources: list[str],
) -> list[dict[str, Any]]:
    """
    Splits one page's normalized text at every detected clause/subclause
    heading, mutating `cursor` in place as headings are encountered, and
    returns one block dict per segment with the FULL hierarchy state
    explicitly stamped on every field (never omitted) — this is what
    closes the "hierarchy metadata never populated" gap: run_pipeline()
    no longer emits one flat per-page block with no clause info at all.

    Every returned block carries clause_number/clause_title/parent_clause/
    subclause_number/subclause_title explicitly (even as None for preamble
    text before the first heading), so process_blocks() never has to infer
    anything — it just applies exactly what's here.
    """
    text = normalize_text(page.text)
    if not text:
        return []

    headings = detect_clause_headings(text)
    boundaries = sorted(set([0, len(text)] + [h[0] for h in headings]))

    blocks: list[dict[str, Any]] = []
    heading_iter = iter(headings)
    next_heading = next(heading_iter, None)

    for i in range(len(boundaries) - 1):
        seg_start, seg_end = boundaries[i], boundaries[i + 1]

        # Apply every heading that starts exactly at this boundary BEFORE
        # emitting the segment that follows it, so the segment's own text
        # is tagged with the heading it belongs to, not the previous one.
        while next_heading is not None and next_heading[0] == seg_start:
            _, level, number, title = next_heading
            if level == "clause":
                cursor.clause_number = number
                cursor.clause_title = title
                cursor.subclause_number = None
                cursor.subclause_title = None
            else:  # subclause
                parent = _parent_clause_number(number)
                if parent and parent != cursor.clause_number:
                    # Defensive: a subclause number implies a parent clause
                    # number that doesn't match what's currently open (e.g.
                    # a missed/garbled parent heading) — trust the
                    # subclause's own number over the stale cursor state,
                    # but we genuinely don't know this parent's title.
                    cursor.clause_number = parent
                    cursor.clause_title = None
                cursor.subclause_number = number
                cursor.subclause_title = title
            next_heading = next(heading_iter, None)

        segment_text = text[seg_start:seg_end].strip()
        if not segment_text:
            continue

        blocks.append({
            "type": TEXT_CONTENT_TYPE,
            "text": segment_text,
            "page": page.page_number,
            "section": section,
            "clause_number": cursor.clause_number,
            "clause_title": cursor.clause_title,
            # clause_number here is always the top-level component (no
            # dots) by construction, so it has no further parent of its
            # own — parent_clause is reserved for a deeper hierarchy than
            # this 2-level (clause/subclause) scheme supports. Explicitly
            # None (not omitted) so process_blocks() clears any stale
            # value rather than inheriting one from a prior block.
            "parent_clause": None,
            "subclause_number": cursor.subclause_number,
            "subclause_title": cursor.subclause_title,
            "sources": sources,
        })

    return blocks


# ============================================================================
# SECTION DETECTION
# ============================================================================

def detect_section_heading(text: str) -> Optional[str]:
    """
    Best-effort heading detection for page-level SECTIONS (coarser than
    clause detection above) — if the first non-empty line of a page looks
    like a heading, treat it as this page's section label.
    """
    for line in text.splitlines():
        candidate = line.strip()
        if not candidate:
            continue
        if len(candidate) <= 80 and _HEADING_RE.match(candidate):
            return candidate
        break
    return None


def compute_page_sections(pages: list[PageRecord]) -> dict[int, Optional[str]]:
    """Carries the most recently detected section heading forward across pages."""
    result: dict[int, Optional[str]] = {}
    current_section: Optional[str] = None
    for page in pages:
        heading = detect_section_heading(page.text)
        if heading:
            current_section = heading
        result[page.page_number] = current_section
    return result


# ============================================================================
# HIERARCHY METADATA
# ============================================================================

def build_hierarchy_metadata(
    context: HierarchyContext,
    document_metadata: Optional[DocumentMetadata] = None,
    content_type: str = TEXT_CONTENT_TYPE,
    chunk_index: Optional[int] = None,
) -> dict[str, Any]:
    """Builds the final Chroma metadata payload. Hierarchy metadata is propagated to EVERY chunk, tables included."""
    document_metadata = document_metadata or DocumentMetadata()

    metadata: dict[str, Any] = {
        "document_name": context.document_name,
        "page_start": context.page_start,
        "page_end": context.page_end,
        "section": context.section,
        "clause_number": normalize_clause_number(context.clause_number),
        "clause_title": context.clause_title,
        "parent_clause": normalize_clause_number(context.parent_clause),
        "subclause_number": normalize_clause_number(context.subclause_number),
        "subclause_title": context.subclause_title,
        "content_type": content_type,
        "sources": " | ".join(context.sources),
    }

    if document_metadata.contract_type is not None:
        metadata["contract_type"] = document_metadata.contract_type
    if document_metadata.parties is not None:
        metadata["parties"] = document_metadata.parties
    if document_metadata.governing_law_country is not None:
        metadata["governing_law_country"] = document_metadata.governing_law_country.upper()
    if document_metadata.effective_date_epoch is not None:
        metadata["effective_date_epoch"] = document_metadata.effective_date_epoch
    if document_metadata.end_date_epoch is not None:
        metadata["end_date_epoch"] = document_metadata.end_date_epoch
    if document_metadata.monetary_value is not None:
        metadata["monetary_value"] = document_metadata.monetary_value
    if chunk_index is not None:
        metadata["chunk_index"] = chunk_index

    return metadata


# ============================================================================
# CHUNK ID
# ============================================================================

def create_chunk_id(document_name: str, chunk_index: int, text: str, metadata: dict[str, Any]) -> str:
    """Deterministic Chroma id — re-ingesting the same document upserts instead of duplicating."""
    hierarchy = "|".join([
        str(metadata.get("section", "")),
        str(metadata.get("clause_number", "")),
        str(metadata.get("subclause_number", "")),
        str(metadata.get("content_type", "")),
    ])
    raw = f"{document_name}|{chunk_index}|{hierarchy}|{text}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return f"chunk_{digest}"


# ============================================================================
# TEXT CHUNKING
# ============================================================================

def split_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP) -> list[str]:
    """
    Splits text into overlapping chunks, respecting paragraph boundaries
    where possible. An oversized paragraph (longer than chunk_size) is
    further split with a sliding character window so no returned chunk
    ever exceeds chunk_size.
    """
    text = normalize_text(text)
    if not text:
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0:
        raise ValueError("overlap cannot be negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current = ""

    def split_oversized(paragraph: str) -> list[str]:
        result: list[str] = []
        start = 0
        length = len(paragraph)
        while start < length:
            end = min(start + chunk_size, length)
            piece = paragraph[start:end].strip()
            if piece:
                result.append(piece)
            if end >= length:
                break
            start = end - overlap
        return result

    for paragraph in paragraphs:
        if len(paragraph) > chunk_size:
            if current:
                chunks.append(current.strip())
                current = ""
            chunks.extend(split_oversized(paragraph))
            continue

        if not current:
            current = paragraph
            continue

        candidate = current + "\n\n" + paragraph
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            chunks.append(current.strip())
            if overlap > 0:
                tail = current[max(0, len(current) - overlap):].strip()
                current = tail + "\n\n" + paragraph
                if len(current) > chunk_size:  # overlap itself made it too large -> start fresh
                    current = paragraph
            else:
                current = paragraph

    if current.strip():
        chunks.append(current.strip())

    return chunks


# ============================================================================
# TABLE HANDLING
# ============================================================================

def table_to_markdown(table: list[list[Optional[str]]]) -> str:
    """Converts a pdfplumber raw table into a GitHub-flavored markdown table. The table remains one logical unit."""
    if not table:
        return ""

    def clean_cell(cell: Optional[str]) -> str:
        return (cell or "").replace("\n", " ").replace("|", "/").strip()

    normalized_rows = [[clean_cell(c) for c in row] for row in table if row]
    if not normalized_rows:
        return ""

    width = max(len(row) for row in normalized_rows)
    normalized_rows = [row + [""] * (width - len(row)) for row in normalized_rows]
    header, *body = normalized_rows

    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    lines += ["| " + " | ".join(row) + " |" for row in body]
    return "\n".join(lines)


def table_to_text(table: Any) -> str:
    """Converts a pandas DataFrame, an object exposing .rows, or a raw list-of-rows into markdown. Never splits it."""
    if table is None:
        return ""
    if hasattr(table, "to_markdown"):
        try:
            return table.to_markdown(index=False)
        except Exception:
            pass
    if isinstance(table, list):
        return table_to_markdown(table)
    rows = getattr(table, "rows", None)
    if rows is not None:
        return table_to_markdown(list(rows))
    return str(table)


def create_table_chunk(
    table: Any, context: HierarchyContext, document_metadata: DocumentMetadata, chunk_index: int,
) -> Optional[PipelineChunk]:
    """Creates ONE intact table chunk, with hierarchy propagated from the current document location."""
    table_text = normalize_text(table_to_text(table))
    if not table_text:
        return None
    metadata = build_hierarchy_metadata(context, document_metadata, TABLE_CONTENT_TYPE, chunk_index)
    return PipelineChunk(text=table_text, metadata=metadata, chunk_index=chunk_index)


# ============================================================================
# NORMAL TEXT CHUNK
# ============================================================================

def create_text_chunks(
    text: str, context: HierarchyContext, document_metadata: DocumentMetadata,
    starting_index: int, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[PipelineChunk]:
    """Creates normal text chunks while propagating hierarchy metadata to every one of them."""
    chunks = split_text(text=text, chunk_size=chunk_size, overlap=overlap)
    assert all(len(c) <= chunk_size for c in chunks), "split_text() produced a chunk larger than chunk_size"

    results: list[PipelineChunk] = []
    for offset, chunk in enumerate(chunks):
        chunk_index = starting_index + offset
        metadata = build_hierarchy_metadata(context, document_metadata, TEXT_CONTENT_TYPE, chunk_index)
        results.append(PipelineChunk(text=chunk, metadata=metadata, chunk_index=chunk_index))
    return results


# ============================================================================
# HIERARCHY PROPAGATION (for external extractors that hand tables in separately)
# ============================================================================

def propagate_hierarchy_to_table(table_metadata: dict[str, Any], surrounding_metadata: dict[str, Any]) -> dict[str, Any]:
    """Fills in any hierarchy field the table itself doesn't already have, from its surrounding context."""
    hierarchy_fields = [
        "document_name", "page_start", "page_end", "section",
        "clause_number", "clause_title", "parent_clause",
        "subclause_number", "subclause_title", "sources",
    ]
    result = dict(surrounding_metadata)
    result.update(table_metadata)
    for field_name in hierarchy_fields:
        if not result.get(field_name) and surrounding_metadata.get(field_name):
            result[field_name] = surrounding_metadata[field_name]
    result["content_type"] = TABLE_CONTENT_TYPE
    return result


# ============================================================================
# GENERIC PAGE / BLOCK INGESTION
# ============================================================================

_HIERARCHY_FIELDS = ("section", "clause_number", "clause_title", "parent_clause", "subclause_number", "subclause_title")


def process_blocks(
    blocks: list[dict[str, Any]],
    document_name: str,
    document_metadata: DocumentMetadata,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[PipelineChunk]:
    """
    Processes already-extracted PDF blocks, e.g.:

        {"type": "text", "text": "...", "page": 12, "section": "Termination",
         "clause_number": "12", "clause_title": "Termination", "parent_clause": None,
         "subclause_number": "12.3", "subclause_title": "Termination Obligations",
         "sources": [...]}

    or {"type": "table", "table": ..., "page": 34, ...}.

    HIERARCHY FIELD SEMANTICS (this is the fix for the "stale leakage"
    bug): a hierarchy field's PRESENCE in a block — not its value — is
    what controls whether it's applied. `"clause_number": None` explicitly
    CLEARS the running context back to None; omitting the key entirely
    means "no opinion, carry forward whatever's already open". The
    previous behavior only ever overwrote with non-None values and could
    never clear a field, so a later block reusing an earlier clause's
    subclause_number (etc.) went undetected.

    Cascading resets: if a block changes `section` without also specifying
    `clause_number`, the clause/subclause are cleared too (a new section
    implies whatever clause was open no longer applies) — same for a block
    that changes `clause_number` without specifying `subclause_number`.
    Blocks that explicitly supply the FULL hierarchy on every call (as
    _split_page_into_hierarchy_blocks() above does) never rely on this
    cascade — it exists for callers that pass partial blocks.
    """
    results: list[PipelineChunk] = []
    current_context = HierarchyContext(document_name=document_name)
    next_chunk_index = 0

    for block in blocks:
        block_type = (block.get("type", TEXT_CONTENT_TYPE) or TEXT_CONTENT_TYPE).lower()

        if block.get("page") is not None:
            current_context.page_start = block["page"]
            current_context.page_end = block["page"]

        if "section" in block and block["section"] != current_context.section and "clause_number" not in block:
            current_context.clause_number = None
            current_context.clause_title = None
            current_context.parent_clause = None
            current_context.subclause_number = None
            current_context.subclause_title = None

        if ("clause_number" in block and block["clause_number"] != current_context.clause_number
                and "subclause_number" not in block):
            current_context.subclause_number = None
            current_context.subclause_title = None

        for field_name in _HIERARCHY_FIELDS:
            if field_name in block:
                setattr(current_context, field_name, block[field_name])  # value may be None -> explicit clear

        if "sources" in block and block["sources"] is not None:
            current_context.sources = block["sources"]

        if block_type == TABLE_CONTENT_TYPE:
            table_chunk = create_table_chunk(
                table=block.get("table", block.get("text")),
                context=current_context,
                document_metadata=document_metadata,
                chunk_index=next_chunk_index,
            )
            if table_chunk:
                results.append(table_chunk)
                next_chunk_index += 1
            continue

        text_chunks = create_text_chunks(
            text=block.get("text", ""),
            context=current_context,
            document_metadata=document_metadata,
            starting_index=next_chunk_index,
            chunk_size=chunk_size,
            overlap=overlap,
        )
        results.extend(text_chunks)
        next_chunk_index += len(text_chunks)

    return results


# ============================================================================
# PDF EXTRACTION
# ============================================================================

def extract_page_content(pdf_path: str) -> tuple[list[PageRecord], dict[int, list[list[list[Optional[str]]]]]]:
    """Extracts native PDF text and tables page-by-page via pdfplumber. Tables are kept separate — never sliced."""
    pages: list[PageRecord] = []
    raw_tables: dict[int, list[list[list[Optional[str]]]]] = {}

    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            text = normalize_text(page.extract_text() or "")
            pages.append(PageRecord(page_number=page_number, text=text, source="pdfplumber", char_count=len(text)))

            try:
                tables = page.extract_tables()
            except Exception:
                tables = []
            if tables:
                raw_tables[page_number] = tables

    return pages, raw_tables


def detect_low_text_pages(pages: list[PageRecord], threshold: int = MIN_CHARS_FOR_TEXT_PAGE) -> list[int]:
    """Returns page numbers whose native text layer is below `threshold` chars — candidates for OCR."""
    return [p.page_number for p in pages if p.char_count < threshold]


def ocr_pages(pdf_path: str, page_numbers: list[int], dpi: int = OCR_DPI) -> dict[int, str]:
    """OCRs only the given pages (not the whole PDF) — keeps the pipeline fast on mostly-native documents."""
    if not page_numbers:
        return {}

    results: dict[int, str] = {}
    for page_number in page_numbers:
        try:
            images = convert_from_path(pdf_path, dpi=dpi, first_page=page_number, last_page=page_number)
            if not images:
                results[page_number] = ""
                continue
            results[page_number] = normalize_text(pytesseract.image_to_string(images[0]))
        except Exception as exc:
            print(f"[OCR] Failed on page {page_number}: {exc}")
            results[page_number] = ""

    return results


def merge_ocr_results(pages: list[PageRecord], ocr_results: dict[int, str]) -> list[PageRecord]:
    """Replaces a low-text page's native extraction with its OCR output, unless OCR produced nothing usable."""
    merged: list[PageRecord] = []
    for page in pages:
        if page.page_number not in ocr_results:
            merged.append(page)
            continue
        ocr_text = ocr_results[page.page_number]
        if ocr_text:
            merged.append(PageRecord(page_number=page.page_number, text=ocr_text, source="ocr",
                                      char_count=len(ocr_text), section=page.section))
        else:
            merged.append(page)  # don't destroy usable native text merely because OCR returned nothing
    return merged


# ============================================================================
# LEGACY (backward-compatible, character-window) CHUNKING API
# ============================================================================

def chunk_pages(
    pages: list[PageRecord], document_name: str, page_sections: dict[int, Optional[str]],
    chunk_size: int = CHUNK_SIZE_CHARS, overlap: int = CHUNK_OVERLAP_CHARS, start_chunk_index: int = 0,
) -> list[ChunkRecord]:
    """
    Backward-compatible page-level chunking (no clause/subclause
    detection — see process_blocks()/_split_page_into_hierarchy_blocks()
    for the canonical, hierarchy-aware ingestion path this file uses now).
    Kept for callers that only need PageRecord -> ChunkRecord without the
    full block/hierarchy machinery.
    """
    chunks: list[ChunkRecord] = []
    chunk_index = start_chunk_index

    for page in pages:
        text = normalize_text(page.text)
        if not text:
            continue
        for chunk_text in split_text(text, chunk_size=chunk_size, overlap=overlap):
            chunks.append(ChunkRecord(
                chunk_id=str(uuid.uuid4()), document_name=document_name, chunk_index=chunk_index,
                text=chunk_text, page_start=page.page_number, page_end=page.page_number,
                section=page_sections.get(page.page_number), content_type=TEXT_CONTENT_TYPE, sources=[page.source],
            ))
            chunk_index += 1

    return chunks


def build_table_records(
    raw_tables: dict[int, list[list[list[Optional[str]]]]], document_name: str, page_sections: dict[int, Optional[str]],
) -> list[TableRecord]:
    records: list[TableRecord] = []
    for page_number, tables in sorted(raw_tables.items()):
        for table_index, table in enumerate(tables):
            markdown = table_to_markdown(table)
            if not markdown:
                continue
            records.append(TableRecord(
                table_id=str(uuid.uuid4()), document_name=document_name, page_number=page_number,
                table_index=table_index, section=page_sections.get(page_number), markdown=markdown,
                num_rows=len(table), num_cols=max((len(r) for r in table), default=0),
            ))
    return records


def build_table_chunks(table_records: list[TableRecord], start_chunk_index: int = 0) -> list[ChunkRecord]:
    return [
        ChunkRecord(
            chunk_id=t.table_id, document_name=t.document_name, chunk_index=start_chunk_index + i,
            text=t.markdown, page_start=t.page_number, page_end=t.page_number, section=t.section,
            content_type=TABLE_CONTENT_TYPE, sources=["pdfplumber"],
        )
        for i, t in enumerate(table_records)
    ]


def embed_and_store(
    chunks: list[ChunkRecord], collection_name: str,
    embedding_model_name: str = EMBEDDING_MODEL_NAME, persist_dir: str = CHROMA_PERSIST_DIR,
    extra_metadata: Optional[dict[str, Any]] = None,
):
    """Backward-compatible adapter: wraps legacy ChunkRecord objects and routes them through store_chunks()."""
    if not chunks:
        return get_chroma_collection(collection_name), get_embedder()

    extra_metadata = extra_metadata or {}
    pipeline_chunks = []
    for chunk in chunks:
        metadata = {
            "document_name": chunk.document_name,
            "chunk_index": chunk.chunk_index,
            "page_start": chunk.page_start,
            "page_end": chunk.page_end,
            "section": chunk.section or "",
            "clause_number": chunk.clause_number or "",
            "clause_title": chunk.clause_title or "",
            "parent_clause": chunk.parent_clause or "",
            "subclause_number": chunk.subclause_number or "",
            "subclause_title": chunk.subclause_title or "",
            "content_type": chunk.content_type,
            "sources": " | ".join(chunk.sources),
            **extra_metadata,
        }
        metadata = {k: v for k, v in metadata.items() if v is not None}
        pipeline_chunks.append(PipelineChunk(text=chunk.text, metadata=metadata, chunk_index=chunk.chunk_index))

    store_chunks(collection_name=collection_name, chunks=pipeline_chunks)
    return get_chroma_collection(collection_name), get_embedder()


# ============================================================================
# CHROMA STORAGE
# ============================================================================

def store_chunks(collection_name: str, chunks: list[PipelineChunk]) -> int:
    """Embeds and stores chunks in Chroma. Tables are embedded exactly like text chunks but stay one logical unit."""
    if not chunks:
        return 0

    collection = get_chroma_collection(collection_name)
    embedder = get_embedder()
    texts = [c.text for c in chunks]
    embeddings = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=True).tolist()

    ids: list[str] = []
    metadatas: list[dict] = []
    for chunk in chunks:
        chunk_id = create_chunk_id(
            document_name=chunk.metadata.get("document_name", ""),
            chunk_index=chunk.chunk_index, text=chunk.text, metadata=chunk.metadata,
        )
        ids.append(chunk_id)
        # Chroma metadata can't hold None/nested structures — keep the payload flat and drop Nones.
        metadatas.append({k: v for k, v in chunk.metadata.items() if v is not None})

    collection.upsert(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)
    return len(chunks)

# ============================================================================
# CHROMA QUERY / RETRIEVAL
# ============================================================================

def query_collection(
    query: str,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    n_results: int = 5,
    where: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Dense vector retrieval from Chroma.

    Parameters
    ----------
    query:
        Natural-language user question.

    collection_name:
        Chroma collection to search.

    n_results:
        Number of chunks to retrieve.

    where:
        Optional Chroma metadata filter.

        Example:
            {
                "document_name": "contract.pdf"
            }

    Returns
    -------
    dict
        Raw Chroma query response containing:
            ids
            documents
            metadatas
            distances
    """

    if not query or not query.strip():
        return {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }

    if n_results <= 0:
        raise ValueError("n_results must be greater than zero")

    collection = get_chroma_collection(collection_name)
    embedder = get_embedder()

    # ------------------------------------------------------------------
    # Embed query using the SAME embedding model used during ingestion
    # ------------------------------------------------------------------

    query_embedding = embedder.encode(
        [query.strip()],
        normalize_embeddings=True,
        show_progress_bar=False,
    ).tolist()

    # ------------------------------------------------------------------
    # Query Chroma
    # ------------------------------------------------------------------

    query_kwargs = {
        "query_embeddings": query_embedding,
        "n_results": n_results,
        "include": [
            "documents",
            "metadatas",
            "distances",
        ],
    }

    if where:
        query_kwargs["where"] = where

    results = collection.query(**query_kwargs)

    return results



def query_collection_records(
    query: str,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    n_results: int = 5,
    where: Optional[dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    """
    Returns Chroma retrieval results as a flat list of records.

    This format is easier for LangGraph, FastAPI and the frontend.
    """

    results = query_collection(
        query=query,
        collection_name=collection_name,
        n_results=n_results,
        where=where,
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]
    ids = results.get("ids", [[]])[0]

    records = []

    for i, document in enumerate(documents):

        metadata = (
            metadatas[i]
            if i < len(metadatas)
            else {}
        )

        distance = (
            distances[i]
            if i < len(distances)
            else None
        )

        chunk_id = (
            ids[i]
            if i < len(ids)
            else None
        )

        records.append({
            "id": chunk_id,
            "text": document,
            "metadata": metadata,
            "distance": distance,

            # Convenient top-level fields for the agent/UI
            "document_name": metadata.get("document_name"),
            "page_start": metadata.get("page_start"),
            "page_end": metadata.get("page_end"),
            "section": metadata.get("section"),
            "clause_number": metadata.get("clause_number"),
            "clause_title": metadata.get("clause_title"),
            "parent_clause": metadata.get("parent_clause"),
            "subclause_number": metadata.get("subclause_number"),
            "subclause_title": metadata.get("subclause_title"),
            "content_type": metadata.get("content_type"),
            "sources": metadata.get("sources"),
        })

    return records
# ============================================================================
# BM25 CACHE INVALIDATION
# ============================================================================

def invalidate_retrieval_cache(collection_name: str) -> None:
    """Invalidates the BM25 cache after ingestion. Import is local to avoid a module-level circular dependency."""
    try:
        from .hybrid_search import invalidate_bm25_cache
        invalidate_bm25_cache(collection_name)
    except ImportError:
        pass  # retrieval module may not be installed/available during standalone ingestion


# ============================================================================
# PUBLIC INGESTION API
# ============================================================================

@traceable(name="pdf_pipeline.ingest", run_type="chain")
def ingest_blocks(
    blocks: list[dict[str, Any]],
    document_name: str,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    contract_type: Optional[str] = None,
    parties: Optional[str] = None,
    governing_law_country: Optional[str] = None,
    effective_date_epoch: Optional[int] = None,
    end_date_epoch: Optional[int] = None,
    monetary_value: Optional[float] = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> dict[str, Any]:
    """Main ingestion entrypoint for pre-extracted blocks — see process_blocks()'s docstring for the block shape."""
    document_metadata = DocumentMetadata(
        contract_type=contract_type, parties=parties, governing_law_country=governing_law_country,
        effective_date_epoch=effective_date_epoch, end_date_epoch=end_date_epoch, monetary_value=monetary_value,
    )
    chunks = process_blocks(blocks, document_name, document_metadata, chunk_size, overlap)
    stored = store_chunks(collection_name, chunks)
    invalidate_retrieval_cache(collection_name)

    return {
        "document_name": document_name,
        "chunks_created": len(chunks),
        "stored": stored,
        "tables": sum(1 for c in chunks if c.metadata.get("content_type") == TABLE_CONTENT_TYPE),
        "text_chunks": sum(1 for c in chunks if c.metadata.get("content_type") == TEXT_CONTENT_TYPE),
    }


def run_pipeline(
    pdf_path: str, collection_name: Optional[str] = None,
    metadata_output_path: Optional[str] = None, sample_query: Optional[str] = None,
) -> dict[str, Any]:
    """
    Full PDF ingestion pipeline:

        PDF -> pdfplumber extraction -> low-text detection -> OCR fallback
            -> hierarchy-aware blocks (clause/subclause detection per page)
            -> process_blocks() -> Chroma -> BM25 cache invalidation

    The "hierarchy-aware blocks" step is what was missing before: each
    page's text is now split at detected clause/subclause headings via
    _split_page_into_hierarchy_blocks(), with a single _HierarchyCursor
    threaded across the whole page loop so a clause spanning multiple
    pages carries forward correctly.
    """
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    document_name = os.path.basename(pdf_path)
    collection_name = collection_name or DEFAULT_COLLECTION_NAME
    os.makedirs(DEFAULT_METADATA_DIR, exist_ok=True)
    metadata_output_path = metadata_output_path or os.path.join(
        DEFAULT_METADATA_DIR, f"{document_name}.metadata.json"
    )

    print(f"[1/7] Extracting PDF: {document_name}")
    pages, raw_tables = extract_page_content(pdf_path)
    print(f"       pages={len(pages)} tables={sum(len(v) for v in raw_tables.values())}")

    print("[2/7] Detecting low-text pages")
    low_text_pages = detect_low_text_pages(pages)
    print(f"       OCR candidates={len(low_text_pages)}")

    print("[3/7] Running OCR fallback")
    ocr_results = ocr_pages(pdf_path, low_text_pages)
    pages = merge_ocr_results(pages, ocr_results)

    print("[4/7] Building hierarchy-aware blocks (clause/subclause detection)")
    page_sections = compute_page_sections(pages)
    cursor = _HierarchyCursor()
    blocks: list[dict[str, Any]] = []

    for page in pages:
        section = page_sections.get(page.page_number)

        blocks.extend(_split_page_into_hierarchy_blocks(page, section, cursor, sources=[page.source]))

        # Tables get whatever clause/subclause is open as of the END of
        # this page's text (pdfplumber doesn't give tables a char offset
        # tying them to a specific position within the page's text, so
        # this is a page-granularity approximation, not exact).
        for table in raw_tables.get(page.page_number, []):
            blocks.append({
                "type": TABLE_CONTENT_TYPE,
                "table": table,
                "page": page.page_number,
                "section": section,
                "clause_number": cursor.clause_number,
                "clause_title": cursor.clause_title,
                "parent_clause": None,
                "subclause_number": cursor.subclause_number,
                "subclause_title": cursor.subclause_title,
                "sources": ["pdfplumber"],
            })

    print(f"       blocks={len(blocks)}")

    print("[5/7] Processing hierarchy-aware chunks")
    document_metadata = DocumentMetadata()
    chunks = process_blocks(
        blocks=blocks, document_name=document_name, document_metadata=document_metadata,
        chunk_size=DEFAULT_CHUNK_SIZE, overlap=DEFAULT_CHUNK_OVERLAP,
    )
    print(f"       chunks={len(chunks)}")

    print("[6/7] Persisting metadata")
    metadata_payload = [
        {
            "chunk_id": create_chunk_id(
                document_name=chunk.metadata.get("document_name", document_name),
                chunk_index=chunk.chunk_index, text=chunk.text, metadata=chunk.metadata,
            ),
            "text": chunk.text,
            "metadata": chunk.metadata,
        }
        for chunk in chunks
    ]
    with open(metadata_output_path, "w", encoding="utf-8") as f:
        json.dump(metadata_payload, f, ensure_ascii=False, indent=2)
    print(f"       metadata={metadata_output_path}")

    print("[7/7] Embedding and storing")
    stored = store_chunks(collection_name=collection_name, chunks=chunks)
    invalidate_retrieval_cache(collection_name)

    return {
        "document_name": document_name,
        "collection_name": collection_name,
        "metadata_path": metadata_output_path,
        "num_pages": len(pages),
        "num_ocr_pages": len(ocr_results),
        "num_text_chunks": sum(1 for c in chunks if c.metadata.get("content_type") == TEXT_CONTENT_TYPE),
        "num_tables": sum(1 for c in chunks if c.metadata.get("content_type") == TABLE_CONTENT_TYPE),
        "num_chunks": len(chunks),
        "stored": stored,
    }


# """

# 1. Imports
# 2. Constants
# 3. Dataclasses
#    ├── DocumentMetadata
#    ├── HierarchyContext
#    ├── PipelineChunk
#    ├── PageRecord
#    ├── TableRecord
#    └── ChunkRecord only if legacy compatibility is required

# 4. Text utilities
#    ├── normalize_text()
#    └── normalize_clause_number()

# 5. Chunk utilities
#    ├── split_text()
#    └── create_chunk_id()

# 6. Hierarchy utilities
#    └── build_hierarchy_metadata()

# 7. Table utilities
#    ├── table_to_markdown()
#    └── table_to_text()

# 8. Chunk creation
#    ├── create_table_chunk()
#    └── create_text_chunks()

# 9. Hierarchy processing
#    ├── propagate_hierarchy_to_table()
#    └── process_blocks()

# 10. Storage
#     └── store_chunks()

# 11. BM25
#     └── invalidate_retrieval_cache()

# 12. Public API
#     ├── ingest_blocks()
#     └── run_pipeline()
# pdf_pipeline.py
# ===============

# Hierarchy-aware PDF ingestion pipeline.

# Document hierarchy:

#     Document
#         |
#         +--> Page
#                 |
#                 +--> Section
#                         |
#                         +--> Clause
#                                 |
#                                 +--> Subclause
#                                         |
#                                         +--> Chunk / Table

# The ingestion pipeline preserves the hierarchy in Chroma metadata.

# Supported metadata:

#     document_name
#     page_start
#     page_end

#     section

#     clause_number
#     clause_title

#     parent_clause

#     subclause_number
#     subclause_title

#     content_type
#     sources

# Document-level metadata:

#     contract_type
#     parties
#     governing_law_country

#     effective_date_epoch
#     end_date_epoch

#     monetary_value

# IMPORTANT TABLE RULE
# --------------------

# Tables are kept intact as logical chunks.

# A table is NOT split merely because it is large.

# Instead:

#     content_type = "table"

# and hierarchy metadata is propagated to it.

# Example:

#     Document
#         |
#         +--> Page 34
#                 |
#                 +--> Section: Financial Statements
#                         |
#                         +--> Clause 8
#                                 |
#                                 +--> 8.2 Annual Financial Statements
#                                         |
#                                         +--> TABLE
#                                              content_type = table
#                                              clause_number = 8
#                                              subclause_number = 8.2

# This allows hybrid retrieval to locate the table through both:

#     1. lexical/table content
#     2. hierarchy metadata

# """

# from __future__ import annotations


# import json
# import os
# import re
# import uuid


# import hashlib
# import re
# from dataclasses import dataclass, field
# from dataclasses import dataclass, asdict, field
# from typing import Any, Optional

# import pdfplumber
# from pdf2image import convert_from_path

# import pytesseract

# from sentence_transformers import SentenceTransformer
# import chromadb
# import pytesseract
# ###
# from ..config import (
#     CHROMA_PERSIST_DIR,
#     EMBEDDING_MODEL_NAME,
# )
# ###

# from ..resources import get_embedder,get_chroma_collection
# from .. import config

# # from ..resources import (
# #     get_chroma_collection,
# #     get_embedder,
# # )

# from ..tracing import traceable


# # ---------------------------------------------------------------------------
# # Config
# # ---------------------------------------------------------------------------
 


# MIN_CHARS_FOR_TEXT_PAGE = 20   # pages with fewer extracted characters than this are treated as "no text"
# OCR_DPI = 300                  # resolution used when rasterizing a page for OCR — higher = better OCR, slower
# CHUNK_SIZE_CHARS = 1000        # target characters per chunk
# CHUNK_OVERLAP_CHARS = 150      # overlap between consecutive chunks, preserves context across chunk boundaries
# EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
 
# # Resolved relative to the project root (two levels up from src/legal_graphrag/ingestion/)
# # so it works the same whether run from VS Code, a terminal, or Colab (with /content mounted).
# _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# CHROMA_PERSIST_DIR = os.path.join(_PROJECT_ROOT, "data", "chroma_db")
# DEFAULT_METADATA_DIR = os.path.join(_PROJECT_ROOT, "data", "metadata")



# # ============================================================================
# # Content types
# # ============================================================================
# TEXT_CONTENT_TYPE = "text"
# TABLE_CONTENT_TYPE = "table"

# #============================================================================
# # Defaults
# #============================================================================
# DEFAULT_CHUNK_SIZE = CHUNK_SIZE_CHARS
# DEFAULT_CHUNK_OVERLAP = CHUNK_OVERLAP_CHARS
# DEFAULT_COLLECTION_NAME = "legal_knowledge_base"


# _HEADING_RE = re.compile(
#     r"^[A-Z0-9][A-Za-z0-9 ,.:\-()]{2,80}$"
# )
# # ============================================================================
# # DATA STRUCTURES
# # ============================================================================
# def extract_page_content(
#     pdf_path: str,
# ) -> tuple[
#     list[PageRecord],
#     dict[int, list[list[list[Optional[str]]]]],
# ]:
#     """
#     Extract native PDF text and tables page-by-page.

#     Text extraction is handled by pdfplumber.

#     Tables are returned separately because tables must remain
#     intact logical retrieval units.
#     """

#     pages: list[PageRecord] = []

#     raw_tables: dict[
#         int,
#         list[list[list[Optional[str]]]]
#     ] = {}

#     with pdfplumber.open(pdf_path) as pdf:

#         for page_number, page in enumerate(
#             pdf.pages,
#             start=1,
#         ):

#             text = page.extract_text() or ""

#             text = normalize_text(text)

#             pages.append(
#                 PageRecord(
#                     page_number=page_number,
#                     text=text,
#                     source="pdfplumber",
#                     char_count=len(text.strip()),
#                 )
#             )

#             try:
#                 tables = page.extract_tables()
#             except Exception:
#                 tables = []

#             if tables:
#                 raw_tables[page_number] = tables

#     return pages, raw_tables

# @dataclass
# class HierarchyContext:
#     """
#     Current location in the document hierarchy.
#     """

#     document_name: str = ""

#     page_start: Optional[int] = None
#     page_end: Optional[int] = None

#     section: Optional[str] = None

#     clause_number: Optional[str] = None
#     clause_title: Optional[str] = None

#     parent_clause: Optional[str] = None

#     subclause_number: Optional[str] = None
#     subclause_title: Optional[str] = None

#     sources: list[str] = field(
#         default_factory=list
#     )


# @dataclass
# class DocumentMetadata:
#     """
#     Document-level metadata propagated to every chunk.
#     """

#     contract_type: Optional[str] = None

#     parties: Optional[str] = None

#     governing_law_country: Optional[str] = None

#     effective_date_epoch: Optional[int] = None

#     end_date_epoch: Optional[int] = None

#     monetary_value: Optional[float] = None


# @dataclass
# class PipelineChunk:
#     """
#     Final logical chunk before vector storage.
#     """

#     text: str

#     metadata: dict[str, Any]

#     chunk_index: int = 0


# #=================================================================
# # Restore PDF extraction
# #=================================================================
# def extract_page_content(
#     pdf_path: str,
# ) -> tuple[
#     list[PageRecord],
#     dict[int, list[list[list[Optional[str]]]]],
# ]:
#     """
#     Extract native PDF text and tables page-by-page.

#     Text extraction is handled by pdfplumber.

#     Tables are returned separately because tables must remain
#     intact logical retrieval units.
#     """

#     pages: list[PageRecord] = []

#     raw_tables: dict[
#         int,
#         list[list[list[Optional[str]]]]
#     ] = {}

#     with pdfplumber.open(pdf_path) as pdf:

#         for page_number, page in enumerate(
#             pdf.pages,
#             start=1,
#         ):

#             text = page.extract_text() or ""

#             text = normalize_text(text)

#             pages.append(
#                 PageRecord(
#                     page_number=page_number,
#                     text=text,
#                     source="pdfplumber",
#                     char_count=len(text.strip()),
#                 )
#             )

#             try:
#                 tables = page.extract_tables()
#             except Exception:
#                 tables = []

#             if tables:
#                 raw_tables[page_number] = tables

#     return pages, raw_tables


# def detect_low_text_pages(
#     pages: list[PageRecord],
#     threshold: int = MIN_CHARS_FOR_TEXT_PAGE,
# ) -> list[int]:
#     """
#     Return pages whose native PDF text layer contains very little text.

#     These pages are candidates for OCR.
#     """

#     return [
#         page.page_number
#         for page in pages
#         if page.char_count < threshold
#     ]

# def ocr_pages(
#     pdf_path: str,
#     page_numbers: list[int],
#     dpi: int = OCR_DPI,
# ) -> dict[int, str]:
#     """
#     OCR only pages that failed native text extraction.

#     This avoids rasterizing the entire PDF unnecessarily.
#     """

#     if not page_numbers:
#         return {}

#     results: dict[int, str] = {}

#     for page_number in page_numbers:

#         try:

#             images = convert_from_path(
#                 pdf_path,
#                 dpi=dpi,
#                 first_page=page_number,
#                 last_page=page_number,
#             )

#             if not images:
#                 results[page_number] = ""
#                 continue

#             text = pytesseract.image_to_string(
#                 images[0]
#             )

#             results[page_number] = normalize_text(
#                 text
#             )

#         except Exception as exc:

#             print(
#                 f"[OCR] Failed on page "
#                 f"{page_number}: {exc}"
#             )

#             results[page_number] = ""

#     return results




# def merge_ocr_results(
#     pages: list[PageRecord],
#     ocr_results: dict[int, str],
# ) -> list[PageRecord]:
#     """
#     Replace low-text native extraction with OCR output.

#     Native extraction is retained when OCR produces no useful text.
#     """

#     merged: list[PageRecord] = []

#     for page in pages:

#         if page.page_number not in ocr_results:
#             merged.append(page)
#             continue

#         ocr_text = normalize_text(
#             ocr_results[page.page_number]
#         )

#         if ocr_text:

#             merged.append(
#                 PageRecord(
#                     page_number=page.page_number,
#                     text=ocr_text,
#                     source="ocr",
#                     char_count=len(
#                         ocr_text
#                     ),
#                     section=page.section,
#                 )
#             )

#         else:

#             # Don't destroy usable native text
#             # merely because OCR returned nothing.
#             merged.append(page)

#     return merged





# def detect_section_heading(
#     text: str,
# ) -> Optional[str]:
#     """
#     Best-effort heading detection.

#     This is intentionally conservative. The hierarchy-aware
#     block processor remains responsible for richer hierarchy.
#     """

#     for line in text.splitlines():

#         candidate = line.strip()

#         if not candidate:
#             continue

#         if (
#             len(candidate) <= 80
#             and _HEADING_RE.match(candidate)
#         ):
#             return candidate

#         break

#     return None


# def compute_page_sections(
#     pages: list[PageRecord],
# ) -> dict[int, Optional[str]]:
#     """
#     Carry the most recent detected section forward across pages.
#     """

#     result: dict[int, Optional[str]] = {}

#     current_section: Optional[str] = None

#     for page in pages:

#         heading = detect_section_heading(
#             page.text
#         )

#         if heading:
#             current_section = heading

#         result[page.page_number] = current_section

#     return result


# def chunk_pages(
#     pages: list[PageRecord],
#     document_name: str,
#     page_sections: dict[int, Optional[str]],
#     chunk_size: int = CHUNK_SIZE_CHARS,
#     overlap: int = CHUNK_OVERLAP_CHARS,
#     start_chunk_index: int = 0,
# ) -> list[ChunkRecord]:
#     """
#     Backward-compatible page chunking API.

#     Converts PageRecord objects into ChunkRecord objects.

#     The newer hierarchy-aware process_blocks() remains the
#     canonical ingestion path.
#     """

#     chunks: list[ChunkRecord] = []

#     chunk_index = start_chunk_index

#     for page in pages:

#         text = normalize_text(
#             page.text
#         )

#         if not text:
#             continue

#         page_chunks = split_text(
#             text,
#             chunk_size=chunk_size,
#             overlap=overlap,
#         )

#         for chunk_text in page_chunks:

#             chunks.append(
#                 ChunkRecord(
#                     chunk_id=str(
#                         uuid.uuid4()
#                     ),
#                     document_name=document_name,
#                     chunk_index=chunk_index,
#                     text=chunk_text,
#                     page_start=page.page_number,
#                     page_end=page.page_number,
#                     section=page_sections.get(
#                         page.page_number
#                     ),
#                     content_type=TEXT_CONTENT_TYPE,
#                     sources=[
#                         page.source
#                     ],
#                 )
#             )

#             chunk_index += 1

#     return chunks
# # ============================================================================
# # TEXT NORMALIZATION
# # ============================================================================

# def normalize_text(
#     text: str,
# ) -> str:
#     """
#     Normalize extracted PDF text while preserving legal meaning.
#     """

#     if not text:
#         return ""

#     text = text.replace(
#         "\x00",
#         " ",
#     )

#     text = text.replace(
#         "\r\n",
#         "\n",
#     )

#     text = text.replace(
#         "\r",
#         "\n",
#     )

#     # Collapse excessive horizontal whitespace.
#     text = re.sub(
#         r"[ \t]+",
#         " ",
#         text,
#     )

#     # Avoid destroying paragraph boundaries.
#     text = re.sub(
#         r"\n{3,}",
#         "\n\n",
#         text,
#     )

#     return text.strip()


# # ============================================================================
# # CLAUSE NUMBER NORMALIZATION
# # ============================================================================

# def normalize_clause_number(
#     value: Any,
# ) -> Optional[str]:
#     """
#     Normalize values such as:

#         12
#         12.3
#         12.3.1

#     into strings.
#     """

#     if value is None:
#         return None

#     value = str(value).strip()

#     if not value:
#         return None

#     value = value.rstrip(".")

#     return value


# # ============================================================================
# # HIERARCHY METADATA
# # ============================================================================

# def build_hierarchy_metadata(
#     context: HierarchyContext,
#     document_metadata: Optional[DocumentMetadata] = None,
#     content_type: str = TEXT_CONTENT_TYPE,
#     chunk_index: Optional[int] = None,
# ) -> dict[str, Any]:
#     """
#     Build the final Chroma metadata payload.

#     Hierarchy metadata is intentionally propagated to EVERY chunk.

#     This includes table chunks.
#     """

#     document_metadata = (
#         document_metadata
#         or DocumentMetadata()
#     )

#     metadata: dict[str, Any] = {
#         # ------------------------------------------------------------
#         # Document
#         # ------------------------------------------------------------
#         "document_name":
#             context.document_name,

#         # ------------------------------------------------------------
#         # Page
#         # ------------------------------------------------------------
#         "page_start":
#             context.page_start,

#         "page_end":
#             context.page_end,

#         # ------------------------------------------------------------
#         # Section
#         # ------------------------------------------------------------
#         "section":
#             context.section,

#         # ------------------------------------------------------------
#         # Clause
#         # ------------------------------------------------------------
#         "clause_number":
#             normalize_clause_number(
#                 context.clause_number
#             ),

#         "clause_title":
#             context.clause_title,

#         # ------------------------------------------------------------
#         # Parent clause
#         # ------------------------------------------------------------
#         "parent_clause":
#             normalize_clause_number(
#                 context.parent_clause
#             ),

#         # ------------------------------------------------------------
#         # Subclause
#         # ------------------------------------------------------------
#         "subclause_number":
#             normalize_clause_number(
#                 context.subclause_number
#             ),

#         "subclause_title":
#             context.subclause_title,

#         # ------------------------------------------------------------
#         # Content
#         # ------------------------------------------------------------
#         "content_type":
#             content_type,

#         # ------------------------------------------------------------
#         # Source
#         # ------------------------------------------------------------
#         "sources":
#             " | ".join(
#                 context.sources
#             ),
#     }

#     # ------------------------------------------------------------
#     # Document-level metadata
#     # ------------------------------------------------------------

#     if document_metadata.contract_type is not None:
#         metadata[
#             "contract_type"
#         ] = document_metadata.contract_type

#     if document_metadata.parties is not None:
#         metadata[
#             "parties"
#         ] = document_metadata.parties

#     if document_metadata.governing_law_country is not None:
#         metadata[
#             "governing_law_country"
#         ] = (
#             document_metadata
#             .governing_law_country
#             .upper()
#         )

#     if (
#         document_metadata
#         .effective_date_epoch
#         is not None
#     ):
#         metadata[
#             "effective_date_epoch"
#         ] = document_metadata.effective_date_epoch

#     if (
#         document_metadata
#         .end_date_epoch
#         is not None
#     ):
#         metadata[
#             "end_date_epoch"
#         ] = document_metadata.end_date_epoch

#     if (
#         document_metadata.monetary_value
#         is not None
#     ):
#         metadata[
#             "monetary_value"
#         ] = document_metadata.monetary_value

#     if chunk_index is not None:
#         metadata[
#             "chunk_index"
#         ] = chunk_index

#     return metadata


# # ============================================================================
# # CHUNK ID
# # ============================================================================

# def create_chunk_id(
#     document_name: str,
#     chunk_index: int,
#     text: str,
#     metadata: dict[str, Any],
# ) -> str:
#     """
#     Generate deterministic Chroma IDs.

#     This helps prevent duplicate ingestion of the same logical chunk.
#     """

#     hierarchy = "|".join(
#         [
#             str(
#                 metadata.get(
#                     "section",
#                     "",
#                 )
#             ),
#             str(
#                 metadata.get(
#                     "clause_number",
#                     "",
#                 )
#             ),
#             str(
#                 metadata.get(
#                     "subclause_number",
#                     "",
#                 )
#             ),
#             str(
#                 metadata.get(
#                     "content_type",
#                     "",
#                 )
#             ),
#         ]
#     )

#     raw = (
#         f"{document_name}|"
#         f"{chunk_index}|"
#         f"{hierarchy}|"
#         f"{text}"
#     )

#     digest = hashlib.sha256(
#         raw.encode("utf-8")
#     ).hexdigest()

#     return f"chunk_{digest}"


# # ============================================================================
# # TEXT CHUNKING
# # ============================================================================

# def split_text(
#     text: str,
#     chunk_size: int = DEFAULT_CHUNK_SIZE,
#     overlap: int = DEFAULT_CHUNK_OVERLAP,
# ) -> list[str]:
#     """
#     Split text while respecting paragraph boundaries where possible.

#     Important:
#     An oversized paragraph is further split using a sliding
#     character window so no returned chunk exceeds chunk_size.
#     """

#     text = normalize_text(text)

#     if not text:
#         return []

#     if chunk_size <= 0:
#         raise ValueError(
#             "chunk_size must be greater than zero"
#         )

#     if overlap < 0:
#         raise ValueError(
#             "overlap cannot be negative"
#         )

#     if overlap >= chunk_size:
#         raise ValueError(
#             "overlap must be smaller than chunk_size"
#         )

#     paragraphs = [
#         paragraph.strip()
#         for paragraph in re.split(
#             r"\n\s*\n",
#             text,
#         )
#         if paragraph.strip()
#     ]

#     chunks: list[str] = []
#     current = ""

#     def split_oversized(
#         paragraph: str,
#     ) -> list[str]:

#         result: list[str] = []

#         start = 0
#         length = len(paragraph)

#         while start < length:

#             end = min(
#                 start + chunk_size,
#                 length,
#             )

#             piece = paragraph[
#                 start:end
#             ].strip()

#             if piece:
#                 result.append(piece)

#             if end >= length:
#                 break

#             start = end - overlap

#         return result

#     for paragraph in paragraphs:

#         # --------------------------------------------------------
#         # Oversized paragraph
#         # --------------------------------------------------------

#         if len(paragraph) > chunk_size:

#             if current:
#                 chunks.append(
#                     current.strip()
#                 )
#                 current = ""

#             chunks.extend(
#                 split_oversized(
#                     paragraph
#                 )
#             )

#             continue

#         # --------------------------------------------------------
#         # Normal paragraph
#         # --------------------------------------------------------

#         if not current:

#             current = paragraph
#             continue

#         candidate = (
#             current
#             + "\n\n"
#             + paragraph
#         )

#         if len(candidate) <= chunk_size:

#             current = candidate

#         else:

#             chunks.append(
#                 current.strip()
#             )

#             if overlap > 0:

#                 tail = current[
#                     max(
#                         0,
#                         len(current)
#                         - overlap,
#                     ):
#                 ].strip()

#                 current = (
#                     tail
#                     + "\n\n"
#                     + paragraph
#                 )

#                 # If the overlap itself makes the
#                 # chunk too large, start fresh.
#                 if len(current) > chunk_size:
#                     current = paragraph

#             else:

#                 current = paragraph

#     if current.strip():
#         chunks.append(
#             current.strip()
#         )

#     return chunks


# def table_to_markdown(
#     table: list[list[Optional[str]]],
# ) -> str:
#     """
#     Convert a pdfplumber table to Markdown.

#     The table remains one logical unit.
#     """

#     if not table:
#         return ""

#     def clean_cell(
#         cell: Optional[str],
#     ) -> str:
#         return (
#             (cell or "")
#             .replace("\n", " ")
#             .replace("|", "/")
#             .strip()
#         )

#     normalized_rows = [
#         [
#             clean_cell(cell)
#             for cell in row
#         ]
#         for row in table
#     ]

#     if not normalized_rows:
#         return ""

#     width = max(
#         len(row)
#         for row in normalized_rows
#     )

#     normalized_rows = [
#         row + [""] * (width - len(row))
#         for row in normalized_rows
#     ]

#     header = normalized_rows[0]
#     body = normalized_rows[1:]

#     lines = [
#         "| "
#         + " | ".join(header)
#         + " |",
#         "| "
#         + " | ".join(
#             "---"
#             for _ in header
#         )
#         + " |",
#     ]

#     for row in body:

#         lines.append(
#             "| "
#             + " | ".join(row)
#             + " |"
#         )

#     return "\n".join(lines)
# # ============================================================================
# # TABLE HANDLING
# # ============================================================================

# def table_to_text(
#     table: Any,
# ) -> str:
#     """
#     Convert a table object into a retrieval-friendly textual representation.

#     IMPORTANT:

#     This function does NOT split the table.

#     The entire table becomes one logical retrieval chunk.

#     The representation is deliberately simple so BM25 can match:

#         revenue
#         assets
#         liabilities
#         2025
#         2024
#         clause-specific terminology

#     while dense retrieval can still understand the semantic content.
#     """

#     if table is None:
#         return ""

#     # ------------------------------------------------------------
#     # Pandas DataFrame
#     # ------------------------------------------------------------

#     if hasattr(
#         table,
#         "to_markdown",
#     ):

#         try:
#             return table.to_markdown(
#                 index=False
#             )
#         except Exception:
#             pass

#     # ------------------------------------------------------------
#     # Object with rows
#     # ------------------------------------------------------------

#     rows = getattr(
#         table,
#         "rows",
#         None,
#     )

#     if rows is not None:

#         output: list[str] = []

#         for row in rows:

#             if isinstance(
#                 row,
#                 (list, tuple),
#             ):
#                 output.append(
#                     " | ".join(
#                         str(cell)
#                         for cell in row
#                     )
#                 )
#             else:
#                 output.append(
#                     str(row)
#                 )

#         return "\n".join(
#             output
#         )

#     # ------------------------------------------------------------
#     # List of rows
#     # ------------------------------------------------------------

#     if isinstance(
#         table,
#         list,
#     ):

#         output = []

#         for row in table:

#             if isinstance(
#                 row,
#                 (list, tuple),
#             ):
#                 output.append(
#                     " | ".join(
#                         str(cell)
#                         for cell in row
#                     )
#                 )
#             else:
#                 output.append(
#                     str(row)
#                 )

#         return "\n".join(
#             output
#         )

#     return str(table)


# def create_table_chunk(
#     table: Any,
#     context: HierarchyContext,
#     document_metadata: DocumentMetadata,
#     chunk_index: int,
# ) -> Optional[PipelineChunk]:
#     """
#     Create ONE intact table chunk.

#     The hierarchy is propagated from the current document location.

#     Example:

#         content_type = table
#         section = Financial Statements
#         clause_number = 8
#         clause_title = Financial Reporting
#         parent_clause = 8
#         subclause_number = 8.2
#         subclause_title = Annual Statements
#     """

#     table_text = normalize_text(
#         table_to_text(table)
#     )

#     if not table_text:
#         return None

#     metadata = build_hierarchy_metadata(
#         context=context,
#         document_metadata=document_metadata,
#         content_type=TABLE_CONTENT_TYPE,
#         chunk_index=chunk_index,
#     )

#     return PipelineChunk(
#         text=table_text,
#         metadata=metadata,
#         chunk_index=chunk_index,
#     )


# # ============================================================================
# # NORMAL TEXT CHUNK
# # ============================================================================

# def create_text_chunks(
#     text: str,
#     context: HierarchyContext,
#     document_metadata: DocumentMetadata,
#     starting_index: int,
#     chunk_size: int = DEFAULT_CHUNK_SIZE,
#     overlap: int = DEFAULT_CHUNK_OVERLAP,
# ) -> list[PipelineChunk]:
#     """
#     Create normal text chunks while propagating hierarchy metadata.
#     """

#     chunks = split_text(
#         text=text,
#         chunk_size=chunk_size,
#         overlap=overlap,
#     )
#     assert all(
#     len(chunk) <= chunk_size
#     for chunk in chunks
#     ), "split_text() produced a chunk larger than chunk_size"

#     results: list[PipelineChunk] = []

#     for offset, chunk in enumerate(
#         chunks
#     ):

#         chunk_index = (
#             starting_index
#             + offset
#         )

#         metadata = build_hierarchy_metadata(
#             context=context,
#             document_metadata=document_metadata,
#             content_type=TEXT_CONTENT_TYPE,
#             chunk_index=chunk_index,
#         )

#         results.append(
#             PipelineChunk(
#                 text=chunk,
#                 metadata=metadata,
#                 chunk_index=chunk_index,
#             )
#         )

#     return results


# # ============================================================================
# # HIERARCHY PROPAGATION
# # ============================================================================

# def propagate_hierarchy_to_table(
#     table_metadata: dict[str, Any],
#     surrounding_metadata: dict[str, Any],
# ) -> dict[str, Any]:
#     """
#     Explicitly propagate hierarchy metadata to a table.

#     Existing table-specific values are preserved.

#     Surrounding hierarchy is used only when the table itself does not
#     already contain the value.

#     This function is useful when the PDF extractor returns tables as
#     separate objects from the surrounding text.
#     """

#     hierarchy_fields = [
#         "document_name",
#         "page_start",
#         "page_end",
#         "section",
#         "clause_number",
#         "clause_title",
#         "parent_clause",
#         "subclause_number",
#         "subclause_title",
#         "sources",
#     ]

#     result = dict(
#         surrounding_metadata
#     )

#     result.update(
#         table_metadata
#     )

#     for field_name in hierarchy_fields:

#         if (
#             not result.get(
#                 field_name
#             )
#             and surrounding_metadata.get(
#                 field_name
#             )
#         ):
#             result[field_name] = (
#                 surrounding_metadata[
#                     field_name
#                 ]
#             )

#     result[
#         "content_type"
#     ] = TABLE_CONTENT_TYPE

#     return result


# # ============================================================================
# # GENERIC PAGE / BLOCK INGESTION
# # ============================================================================
# @dataclass
# class PageRecord:
#     """One PDF page after native extraction or OCR."""

#     page_number: int
#     text: str
#     source: str
#     char_count: int
#     section: Optional[str] = None


# @dataclass
# class TableRecord:
#     """One intact table extracted from a PDF page."""

#     table_id: str
#     document_name: str
#     page_number: int
#     table_index: int
#     section: Optional[str]
#     markdown: str
#     num_rows: int
#     num_cols: int


# @dataclass
# class ChunkRecord:
#     """
#     Backward-compatible chunk representation used by older callers.

#     The active hierarchy-aware pipeline internally uses PipelineChunk.
#     """

#     chunk_id: str
#     document_name: str
#     chunk_index: int
#     text: str
#     page_start: int
#     page_end: int
#     section: Optional[str]
#     content_type: str = TEXT_CONTENT_TYPE
#     sources: list[str] = field(default_factory=list)
# def process_blocks(
#     blocks: list[dict[str, Any]],
#     document_name: str,
#     document_metadata: DocumentMetadata,
#     chunk_size: int = DEFAULT_CHUNK_SIZE,
#     overlap: int = DEFAULT_CHUNK_OVERLAP,
# ) -> list[PipelineChunk]:
#     """
#     Process already-extracted PDF blocks.

#     This function expects the PDF extraction layer to provide blocks such as:

#         {
#             "type": "text",
#             "text": "...",
#             "page": 12,
#             "section": "Termination",
#             "clause_number": "12",
#             "clause_title": "Termination",
#             "parent_clause": "12",
#             "subclause_number": "12.3",
#             "subclause_title": "Termination Obligations",
#             "sources": [...]
#         }

#     OR:

#         {
#             "type": "table",
#             "table": ...,
#             "page": 34,
#             ...
#         }

#     The extraction layer remains independent from Chroma.
#     """

#     results: list[PipelineChunk] = []

#     current_context = HierarchyContext(
#         document_name=document_name
#     )

#     next_chunk_index = 0

#     for block in blocks:

#         block_type = (
#             block.get(
#                 "type",
#                 TEXT_CONTENT_TYPE,
#             )
#             or TEXT_CONTENT_TYPE
#         ).lower()

#         # ------------------------------------------------------------
#         # Update hierarchy context.
#         # ------------------------------------------------------------

#         if block.get("page") is not None:

#             current_context.page_start = (
#                 block["page"]
#             )

#             current_context.page_end = (
#                 block["page"]
#             )

#         hierarchy_fields = [
#             "section",
#             "clause_number",
#             "clause_title",
#             "parent_clause",
#             "subclause_number",
#             "subclause_title",
#             "sources",
#         ]

#         for field_name in hierarchy_fields:

#             if field_name in block:

#                 value = block[
#                     field_name
#                 ]

#                 if value is not None:

#                     setattr(
#                         current_context,
#                         field_name,
#                         value,
#                     )

#         # ------------------------------------------------------------
#         # TABLE
#         # ------------------------------------------------------------

#         if block_type == TABLE_CONTENT_TYPE:

#             table_chunk = create_table_chunk(
#                 table=block.get(
#                     "table",
#                     block.get(
#                         "text"
#                     ),
#                 ),
#                 context=current_context,
#                 document_metadata=document_metadata,
#                 chunk_index=next_chunk_index,
#             )

#             if table_chunk:

#                 results.append(
#                     table_chunk
#                 )

#                 next_chunk_index += 1

#             continue

#         # ------------------------------------------------------------
#         # TEXT
#         # ------------------------------------------------------------

#         text = block.get(
#             "text",
#             "",
#         )

#         text_chunks = create_text_chunks(
#             text=text,
#             context=current_context,
#             document_metadata=document_metadata,
#             starting_index=next_chunk_index,
#             chunk_size=chunk_size,
#             overlap=overlap,
#         )

#         results.extend(
#             text_chunks
#         )

#         next_chunk_index += len(
#             text_chunks
#         )

#     return results


# def build_table_records(
#     raw_tables: dict[
#         int,
#         list[list[list[Optional[str]]]]
#     ],
#     document_name: str,
#     page_sections: dict[int, Optional[str]],
# ) -> list[TableRecord]:

#     records: list[TableRecord] = []

#     for page_number, tables in sorted(
#         raw_tables.items()
#     ):

#         for table_index, table in enumerate(
#             tables
#         ):

#             markdown = table_to_markdown(
#                 table
#             )

#             if not markdown:
#                 continue

#             records.append(
#                 TableRecord(
#                     table_id=str(
#                         uuid.uuid4()
#                     ),
#                     document_name=document_name,
#                     page_number=page_number,
#                     table_index=table_index,
#                     section=page_sections.get(
#                         page_number
#                     ),
#                     markdown=markdown,
#                     num_rows=len(table),
#                     num_cols=max(
#                         (
#                             len(row)
#                             for row in table
#                         ),
#                         default=0,
#                     ),
#                 )
#             )

#     return records

# def build_table_chunks(
#     table_records: list[TableRecord],
#     start_chunk_index: int = 0,
# ) -> list[ChunkRecord]:

#     chunks: list[ChunkRecord] = []

#     for offset, table in enumerate(
#         table_records
#     ):

#         chunks.append(
#             ChunkRecord(
#                 chunk_id=table.table_id,
#                 document_name=table.document_name,
#                 chunk_index=(
#                     start_chunk_index
#                     + offset
#                 ),
#                 text=table.markdown,
#                 page_start=table.page_number,
#                 page_end=table.page_number,
#                 section=table.section,
#                 content_type=TABLE_CONTENT_TYPE,
#                 sources=["pdfplumber"],
#             )
#         )

#     return chunks


# def table_to_markdown(
#     table: list[list[Optional[str]]],
# ) -> str:
#     """
#     Convert a pdfplumber table to Markdown.

#     The table remains one logical retrieval unit.
#     """

#     if not table:
#         return ""

#     def clean_cell(cell: Optional[str]) -> str:
#         return (
#             (cell or "")
#             .replace("\n", " ")
#             .replace("|", "/")
#             .strip()
#         )

#     normalized_rows = [
#         [clean_cell(cell) for cell in row]
#         for row in table
#         if row
#     ]

#     if not normalized_rows:
#         return ""

#     width = max(
#         len(row)
#         for row in normalized_rows
#     )

#     normalized_rows = [
#         row + [""] * (width - len(row))
#         for row in normalized_rows
#     ]

#     header = normalized_rows[0]
#     body = normalized_rows[1:]

#     lines = [
#         "| " + " | ".join(header) + " |",
#         "| " + " | ".join("---" for _ in header) + " |",
#     ]

#     for row in body:
#         lines.append(
#             "| " + " | ".join(row) + " |"
#         )

#     return "\n".join(lines)


# def table_to_text(table: Any) -> str:
#     """
#     Convert supported table objects into one retrieval-ready text block.

#     Tables are NEVER split here.
#     """

#     if table is None:
#         return ""

#     # Pandas DataFrame
#     if hasattr(table, "to_markdown"):
#         try:
#             return table.to_markdown(index=False)
#         except Exception:
#             pass

#     # pdfplumber/list-of-rows
#     if isinstance(table, list):
#         return table_to_markdown(table)

#     # Object exposing rows
#     rows = getattr(table, "rows", None)

#     if rows is not None:
#         return table_to_markdown(list(rows))

#     return str(table)

# def embed_and_store(
#     chunks: list[ChunkRecord],
#     collection_name: str,
#     embedding_model_name: str = EMBEDDING_MODEL_NAME,
#     persist_dir: str = CHROMA_PERSIST_DIR,
#     extra_metadata: Optional[dict[str, Any]] = None,
# ):
#     """
#     Backward-compatible adapter for the legacy ChunkRecord API.

#     New code should use store_chunks().
#     """

#     if not chunks:
#         return (
#             get_chroma_collection(collection_name),
#             get_embedder(),
#         )

#     extra_metadata = extra_metadata or {}

#     pipeline_chunks = []

#     for chunk in chunks:

#         metadata = {
#             "document_name": chunk.document_name,
#             "chunk_index": chunk.chunk_index,
#             "page_start": chunk.page_start,
#             "page_end": chunk.page_end,
#             "section": chunk.section or "",
#             "content_type": chunk.content_type,
#             "sources": " | ".join(chunk.sources),
#             **extra_metadata,
#         }

#         metadata = {
#             key: value
#             for key, value in metadata.items()
#             if value is not None
#         }

#         pipeline_chunks.append(
#             PipelineChunk(
#                 text=chunk.text,
#                 metadata=metadata,
#                 chunk_index=chunk.chunk_index,
#             )
#         )

#     store_chunks(
#         collection_name=collection_name,
#         chunks=pipeline_chunks,
#     )

#     return (
#         get_chroma_collection(collection_name),
#         get_embedder(),
#     )
# # ============================================================================
# # CHROMA STORAGE
# # ============================================================================

# def store_chunks(
#     collection_name: str,
#     chunks: list[PipelineChunk],
# ) -> int:
#     """
#     Embed and store chunks in Chroma.

#     Tables are embedded exactly like other chunks but remain one logical
#     document in the vector store.
#     """
#     from ..resources import get_chroma_collection

#     if not chunks:
#         return 0

#     collection = get_chroma_collection(
#         collection_name
#     )

#     embedder = get_embedder()

#     texts = [
#         chunk.text
#         for chunk in chunks
#     ]

#     embeddings = embedder.encode(
#         texts,
#         normalize_embeddings=True,
#         show_progress_bar=True,
#     ).tolist()

#     ids: list[str] = []

#     metadatas: list[dict] = []

#     for chunk in chunks:

#         chunk_id = create_chunk_id(
#             document_name=chunk.metadata.get(
#                 "document_name",
#                 "",
#             ),
#             chunk_index=chunk.chunk_index,
#             text=chunk.text,
#             metadata=chunk.metadata,
#         )

#         ids.append(
#             chunk_id
#         )

#         # Chroma metadata cannot safely contain arbitrary nested
#         # structures. Keep the payload flat.
#         metadata = {
#             key: value
#             for key, value in chunk.metadata.items()
#             if value is not None
#         }

#         metadatas.append(
#             metadata
#         )

#     collection.upsert(
#         ids=ids,
#         documents=texts,
#         embeddings=embeddings,
#         metadatas=metadatas,
#     )

#     return len(chunks)


# # ============================================================================
# # BM25 CACHE INVALIDATION
# # ============================================================================

# def invalidate_retrieval_cache(
#     collection_name: str,
# ) -> None:
#     """
#     Invalidate the BM25 cache after ingestion.

#     Import is deliberately local to avoid creating a module-level
#     circular dependency.
#     """

#     try:

#         from .hybrid_search import (
#             invalidate_bm25_cache,
#         )

#         invalidate_bm25_cache(
#             collection_name
#         )

#     except ImportError as exc:
#         raise RuntimeError(
#             "Unable to import hybrid_search.invalidate_bm25_cache"
#         ) from exc
#         # Retrieval module may not be installed/available during
#         # standalone ingestion.
#         # pass


# # ============================================================================
# # PUBLIC INGESTION API
# # ============================================================================

# @traceable(
#     name="pdf_pipeline.ingest",
#     run_type="chain",
# )
# def ingest_blocks(
#     blocks: list[dict[str, Any]],
#     document_name: str,
#     collection_name: str = DEFAULT_COLLECTION_NAME,
#     contract_type: Optional[str] = None,
#     parties: Optional[str] = None,
#     governing_law_country: Optional[str] = None,
#     effective_date_epoch: Optional[int] = None,
#     end_date_epoch: Optional[int] = None,
#     monetary_value: Optional[float] = None,
#     chunk_size: int = DEFAULT_CHUNK_SIZE,
#     overlap: int = DEFAULT_CHUNK_OVERLAP,
# ) -> dict[str, Any]:
#     """
#     Main ingestion entrypoint.

#     Example:

#         result = ingest_blocks(
#             blocks=extracted_blocks,
#             document_name="Contract.pdf",
#             collection_name="legal_knowledge_base",
#             contract_type="Lease Agreement",
#             parties="ABC Ltd; XYZ Ltd",
#             governing_law_country="US",
#             effective_date_epoch=...,
#             end_date_epoch=...,
#             monetary_value=500000,
#         )

#     Returns:

#         {
#             "document_name": "...",
#             "chunks_created": 100,
#             "stored": 100,
#         }
#     """

#     document_metadata = DocumentMetadata(
#         contract_type=contract_type,
#         parties=parties,
#         governing_law_country=governing_law_country,
#         effective_date_epoch=effective_date_epoch,
#         end_date_epoch=end_date_epoch,
#         monetary_value=monetary_value,
#     )

#     chunks = process_blocks(
#         blocks=blocks,
#         document_name=document_name,
#         document_metadata=document_metadata,
#         chunk_size=chunk_size,
#         overlap=overlap,
#     )

#     stored = store_chunks(
#         collection_name=collection_name,
#         chunks=chunks,
#     )

#     # New chunks mean the lexical index must be rebuilt.
#     invalidate_retrieval_cache(
#         collection_name
#     )

#     return {
#         "document_name":
#             document_name,

#         "chunks_created":
#             len(chunks),

#         "stored":
#             stored,

#         "tables":
#             sum(
#                 1
#                 for chunk in chunks
#                 if chunk.metadata.get(
#                     "content_type"
#                 )
#                 == TABLE_CONTENT_TYPE
#             ),

#         "text_chunks":
#             sum(
#                 1
#                 for chunk in chunks
#                 if chunk.metadata.get(
#                     "content_type"
#                 )
#                 == TEXT_CONTENT_TYPE
#             ),
#     }
# def run_pipeline(
#     pdf_path: str,
#     collection_name: Optional[str] = None,
#     metadata_output_path: Optional[str] = None,
#     sample_query: Optional[str] = None,
# ) -> dict[str, Any]:
#     """
#     Full PDF ingestion pipeline.

#     PDF
#       ↓
#     pdfplumber extraction
#       ↓
#     low-text detection
#       ↓
#     OCR fallback
#       ↓
#     hierarchy-aware blocks
#       ↓
#     process_blocks()
#       ↓
#     Chroma
#       ↓
#     BM25 cache invalidation
#     """

#     if not os.path.isfile(pdf_path):
#         raise FileNotFoundError(
#             f"PDF not found: {pdf_path}"
#         )

#     document_name = os.path.basename(
#         pdf_path
#     )

#     collection_name = (
#         collection_name
#         or DEFAULT_COLLECTION_NAME
#     )

#     os.makedirs(
#         DEFAULT_METADATA_DIR,
#         exist_ok=True,
#     )

#     metadata_output_path = (
#         metadata_output_path
#         or os.path.join(
#             DEFAULT_METADATA_DIR,
#             f"{document_name}.metadata.json",
#         )
#     )

#     print(
#         f"[1/7] Extracting PDF: "
#         f"{document_name}"
#     )

#     pages, raw_tables = (
#         extract_page_content(
#             pdf_path
#         )
#     )

#     print(
#         f"       pages={len(pages)}"
#     )

#     print(
#         f"       tables={sum(len(v) for v in raw_tables.values())}"
#     )

#     print(
#         "[2/7] Detecting low-text pages"
#     )

#     low_text_pages = (
#         detect_low_text_pages(
#             pages
#         )
#     )

#     print(
#         f"       OCR candidates="
#         f"{len(low_text_pages)}"
#     )

#     print(
#         "[3/7] Running OCR fallback"
#     )

#     ocr_results = ocr_pages(
#         pdf_path,
#         low_text_pages,
#     )

#     pages = merge_ocr_results(
#         pages,
#         ocr_results,
#     )

#     print(
#         "[4/7] Building hierarchy blocks"
#     )

#     page_sections = (
#         compute_page_sections(
#             pages
#         )
#     )

#     blocks: list[dict[str, Any]] = []

#     for page in pages:

#         section = page_sections.get(
#             page.page_number
#         )

#         if page.text.strip():

#             blocks.append(
#                 {
#                     "type": TEXT_CONTENT_TYPE,
#                     "text": page.text,
#                     "page": page.page_number,
#                     "section": section,
#                     "sources": [
#                         page.source
#                     ],
#                 }
#             )

#         for table in raw_tables.get(
#             page.page_number,
#             [],
#         ):

#             blocks.append(
#                 {
#                     "type": TABLE_CONTENT_TYPE,
#                     "table": table,
#                     "page": page.page_number,
#                     "section": section,
#                     "sources": [
#                         "pdfplumber"
#                     ],
#                 }
#             )

#     print(
#         f"       blocks={len(blocks)}"
#     )

#     print(
#         "[5/7] Processing hierarchy-aware chunks"
#     )

#     document_metadata = (
#         DocumentMetadata()
#     )

#     chunks = process_blocks(
#         blocks=blocks,
#         document_name=document_name,
#         document_metadata=document_metadata,
#         chunk_size=DEFAULT_CHUNK_SIZE,
#         overlap=DEFAULT_CHUNK_OVERLAP,
#     )

#     print(
#         f"       chunks={len(chunks)}"
#     )

#     print(
#         "[6/7] Persisting metadata"
#     )

#     metadata_payload = [
#         {
#             "chunk_id": create_chunk_id(
#                 document_name=
#                     chunk.metadata.get(
#                         "document_name",
#                         document_name,
#                     ),
#                 chunk_index=
#                     chunk.chunk_index,
#                 text=chunk.text,
#                 metadata=chunk.metadata,
#             ),
#             "text": chunk.text,
#             "metadata": chunk.metadata,
#         }
#         for chunk in chunks
#     ]

#     with open(
#         metadata_output_path,
#         "w",
#         encoding="utf-8",
#     ) as file:

#         json.dump(
#             metadata_payload,
#             file,
#             ensure_ascii=False,
#             indent=2,
#         )

#     print(
#         f"       metadata="
#         f"{metadata_output_path}"
#     )

#     print(
#         "[7/7] Embedding and storing"
#     )

#     stored = store_chunks(
#         collection_name=collection_name,
#         chunks=chunks,
#     )

#     invalidate_retrieval_cache(
#         collection_name
#     )

#     result = {
#         "document_name":
#             document_name,

#         "collection_name":
#             collection_name,

#         "metadata_path":
#             metadata_output_path,

#         "num_pages":
#             len(pages),

#         "num_ocr_pages":
#             len(ocr_results),

#         "num_text_chunks":
#             sum(
#                 1
#                 for chunk in chunks
#                 if chunk.metadata.get(
#                     "content_type"
#                 ) == TEXT_CONTENT_TYPE
#             ),

#         "num_tables":
#             sum(
#                 1
#                 for chunk in chunks
#                 if chunk.metadata.get(
#                     "content_type"
#                 ) == TABLE_CONTENT_TYPE
#             ),

#         "num_chunks":
#             len(chunks),

#         "stored":
#             stored,
#     }

#     return result

# # """
# # Hierarchy-aware Hybrid Retrieval
# # ---------------------------------

# # Retrieval hierarchy:

# #     document
# #         -> page
# #             -> section
# #                 -> clause
# #                     -> subclause
# #                         -> chunk

# # Pipeline:

# #     Query
# #       |
# #       +--> Dense retrieval
# #       |
# #       +--> BM25 retrieval
# #       |
# #       +--> Hybrid fusion
# #       |
# #       +--> Hierarchy-aware boosting
# #       |
# #       +--> Deduplication
# #       |
# #       +--> Parent / sibling context expansion
# #       |
# #       +--> Cross-encoder reranking
# #       |
# #       +--> Final results

# # Designed to work with the ingestion metadata:

# #     document_name
# #     page_start
# #     page_end
# #     section
# #     clause_number
# #     clause_title
# #     parent_clause
# #     subclause_number
# #     subclause_title
# #     content_type
# #     sources

# # Additional document-level metadata:

# #     contract_type
# #     parties
# #     governing_law_country
# #     effective_date_epoch
# #     end_date_epoch
# #     monetary_value
# # """

# # from __future__ import annotations

# # import re
# # from collections import defaultdict
# # from typing import Optional

# # from rank_bm25 import BM25Okapi

# # from ..resources import (
# #     get_chroma_collection,
# #     get_embedder,
# #     get_reranker,
# # )

# # from ..tracing import traceable


# # # ============================================================================
# # # CONFIGURATION
# # # ============================================================================

# # DEFAULT_DENSE_K = 30
# # DEFAULT_SPARSE_K = 30

# # DEFAULT_FUSION_K = 30
# # DEFAULT_FINAL_K = 6

# # DEFAULT_ALPHA = 0.45

# # # Number of chunks to retrieve around a selected chunk
# # DEFAULT_NEIGHBOR_WINDOW = 1

# # # Number of hierarchy-expanded candidates allowed before reranking
# # DEFAULT_MAX_EXPANDED_CANDIDATES = 40


# # # Hierarchy boosts
# # SECTION_BOOST = 0.08

# # CLAUSE_NUMBER_BOOST = 0.20
# # CLAUSE_TITLE_BOOST = 0.15

# # PARENT_CLAUSE_BOOST = 0.10

# # SUBCLAUSE_NUMBER_BOOST = 0.25
# # SUBCLAUSE_TITLE_BOOST = 0.15

# # DOCUMENT_NAME_BOOST = 0.05

# # TABLE_BOOST = 0.05


# # # ============================================================================
# # # BM25 CACHE
# # # ============================================================================

# # # collection_name ->
# # #
# # # (
# # #   bm25,
# # #   ids,
# # #   texts,
# # #   metadatas
# # # )
# # #
# # _bm25_cache: dict[
# #     str,
# #     tuple[BM25Okapi, list[str], list[str], list[dict]]
# # ] = {}


# # # ============================================================================
# # # TOKENIZATION
# # # ============================================================================

# # def _tokenize(text: str) -> list[str]:
# #     """
# #     Tokenizer designed for legal / financial documents.

# #     Keeps:
# #         12
# #         12.3
# #         12.3.1
# #         indemnification
# #         termination
# #         force
# #         majeure

# #     Also creates normalized alphanumeric tokens.
# #     """

# #     if not text:
# #         return []

# #     text = text.lower()

# #     # Preserve clause numbers such as:
# #     #
# #     # 12
# #     # 12.3
# #     # 12.3.1
# #     #
# #     clause_numbers = re.findall(
# #         r"\b\d+(?:\.\d+){0,4}\b",
# #         text
# #     )

# #     # Normal lexical tokens
# #     tokens = re.findall(
# #         r"[a-z0-9]+(?:[-'][a-z0-9]+)*",
# #         text
# #     )

# #     # Add clause numbers explicitly.
# #     tokens.extend(clause_numbers)

# #     return tokens


# # # ============================================================================
# # # BM25 INDEX
# # # ============================================================================

# # def _build_bm25_index(
# #     collection_name: str,
# # ):
# #     """
# #     Build the BM25 index from Chroma.
# #     """

# #     collection = get_chroma_collection(collection_name)

# #     raw = collection.get(
# #         include=[
# #             "documents",
# #             "metadatas",
# #         ]
# #     )

# #     ids = raw["ids"]
# #     texts = raw["documents"]
# #     metadatas = raw["metadatas"]

# #     tokenized_documents = [
# #         _tokenize(text)
# #         for text in texts
# #     ]

# #     bm25 = BM25Okapi(tokenized_documents)

# #     _bm25_cache[collection_name] = (
# #         bm25,
# #         ids,
# #         texts,
# #         metadatas,
# #     )

# #     return _bm25_cache[collection_name]


# # def _get_bm25_index(
# #     collection_name: str,
# #     refresh: bool = False,
# # ):
# #     """
# #     Get cached BM25 index.

# #     Set refresh=True after ingestion.
# #     """

# #     if (
# #         not refresh
# #         and collection_name in _bm25_cache
# #     ):
# #         return _bm25_cache[collection_name]

# #     return _build_bm25_index(collection_name)


# # def invalidate_bm25_cache(
# #     collection_name: str,
# # ) -> None:
# #     """
# #     Call immediately after ingesting a document.
# #     """

# #     _bm25_cache.pop(
# #         collection_name,
# #         None,
# #     )


# # # ============================================================================
# # # METADATA FILTERING
# # # ============================================================================

# # def build_where_clause(
# #     filters: dict,
# # ) -> Optional[dict]:

# #     clauses = []

# #     if filters.get("contract_type"):
# #         clauses.append(
# #             {
# #                 "contract_type":
# #                 filters["contract_type"]
# #             }
# #         )

# #     if filters.get("governing_law_country"):
# #         clauses.append(
# #             {
# #                 "governing_law_country":
# #                 filters["governing_law_country"].upper()
# #             }
# #         )

# #     if filters.get("min_effective_date_epoch") is not None:
# #         clauses.append(
# #             {
# #                 "effective_date_epoch": {
# #                     "$gte":
# #                     filters["min_effective_date_epoch"]
# #                 }
# #             }
# # """
# # PDF -> RAG ingestion pipeline for Google Colab.

# # Pipeline stages:
# #     1. Extract text AND tables per page with pdfplumber.
# #     2. Detect pages with little or no extractable text (scanned/image pages).
# #     3. Run OCR (pytesseract) on just those pages.
# #     4. Clean the merged text and chunk it with sliding-window overlap,
# #        tracking which page(s) and section each chunk came from.
# #     5. Turn any extracted tables into their own markdown-formatted chunks
# #        (kept intact, never sliced by the sliding window).
# #     6. Persist page/section/source/table metadata to a JSON file for
# #        traceability.
# #     7. Embed every chunk (text and table) with sentence-transformers and
# #        store it in a local Chroma vector collection for RAG.

# # ------------------------------------------------------------------------
# # Run this in a Colab cell FIRST (system packages + python packages):

# #     !apt-get -qq update && apt-get -qq install -y poppler-utils tesseract-ocr

# #     # --no-deps is important: Colab ships a specific torch/torchvision pair
# #     # already. Installing sentence-transformers normally lets pip silently
# #     # downgrade torch to satisfy its pin, which breaks torchvision (still
# #     # expecting the newer torch) and can crash embedding with
# #     # "RuntimeError: Numpy is not available". --no-deps keeps Colab's
# #     # existing torch/torchvision untouched; we then add sentence-transformers'
# #     # other dependencies explicitly, without letting them touch torch either.
# #     !pip -q install pdfplumber pdf2image pytesseract chromadb
# #     !pip -q install sentence-transformers --no-deps
# #     !pip -q install --upgrade-strategy only-if-needed \
# #         transformers tokenizers huggingface-hub safetensors scikit-learn scipy Pillow tqdm

# #     # Then: Runtime -> Restart session (required — a pip install that touches
# #     # any of torch's neighbors needs a fresh process, not just a fresh cell).

# # Then upload a PDF (or mount Drive) and run:

# #     from pdf_rag_pipeline import run_pipeline
# #     run_pipeline("data/uploads/my_document.pdf")   # or an absolute /content/... path in Colab
# # ------------------------------------------------------------------------
# # """

# # from __future__ import annotations

# # import json
# # import os
# # import re
# # import uuid
# # from dataclasses import dataclass, asdict, field
# # from typing import Optional

# # import pdfplumber
# # from pdf2image import convert_from_path
# # import pytesseract

# # from sentence_transformers import SentenceTransformer
# # import chromadb


# # # ---------------------------------------------------------------------------
# # # Config
# # # ---------------------------------------------------------------------------

# # MIN_CHARS_FOR_TEXT_PAGE = 20   # pages with fewer extracted characters than this are treated as "no text"
# # OCR_DPI = 300                  # resolution used when rasterizing a page for OCR — higher = better OCR, slower
# # CHUNK_SIZE_CHARS = 1000        # target characters per chunk
# # CHUNK_OVERLAP_CHARS = 150      # overlap between consecutive chunks, preserves context across chunk boundaries
# # EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

# # # Resolved relative to the project root (two levels up from src/legal_graphrag/ingestion/)
# # # so it works the same whether run from VS Code, a terminal, or Colab (with /content mounted).
# # _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# # CHROMA_PERSIST_DIR = os.path.join(_PROJECT_ROOT, "data", "chroma_db")
# # DEFAULT_METADATA_DIR = os.path.join(_PROJECT_ROOT, "data", "metadata")


# # # ---------------------------------------------------------------------------
# # # Data structures
# # # ---------------------------------------------------------------------------

# # @dataclass
# # class PageRecord:
# #     """One page of the source PDF, after extraction (and OCR if needed)."""
# #     page_number: int              # 1-indexed
# #     text: str
# #     source: str                   # "pdfplumber" or "ocr"
# #     char_count: int
# #     section: Optional[str] = None  # heading detected on this page, if any


# # @dataclass
# # class TableRecord:
# #     """One table extracted from a page, kept as its own unit (never chunked)."""
# #     table_id: str
# #     document_name: str
# #     page_number: int
# #     table_index: int              # index of this table within its page (0-based)
# #     section: Optional[str]
# #     markdown: str
# #     num_rows: int
# #     num_cols: int


# # @dataclass
# # class ChunkRecord:
# #     """One chunk of the final content — either prose text or a whole table."""
# #     chunk_id: str
# #     document_name: str
# #     chunk_index: int
# #     text: str
# #     page_start: int
# #     page_end: int
# #     section: Optional[str]
# #     content_type: str = "text"                    # "text" or "table"
# #     sources: list = field(default_factory=list)    # e.g. ["pdfplumber"], ["ocr"], or both if it spans pages


# # # ---------------------------------------------------------------------------
# # # Step 1: Extract text AND tables with pdfplumber
# # # ---------------------------------------------------------------------------

# # def extract_page_content(pdf_path: str) -> tuple[list[PageRecord], dict[int, list[list[list[Optional[str]]]]]]:
# #     """
# #     Extract text and tables page-by-page using pdfplumber, in a single pass.

# #     pdfplumber only recovers text/tables that are actually encoded in the
# #     PDF (native text layer + vector-drawn table lines). Scanned pages, or
# #     pages that are just an embedded image, will come back with little/no
# #     text AND no detected tables here — that's expected, and is exactly what
# #     step 2 checks for on the text side. Tables on scanned pages are a known
# #     gap: pdfplumber can't detect table structure from an image with no
# #     vector content, so a scanned table will not be recovered by this
# #     pipeline (only its raw OCR'd text will be, as part of the page text).

# #     Returns:
# #         pages: one PageRecord per page (text only; tables handled separately).
# #         raw_tables: {page_number: [table, ...]} where each table is pdfplumber's
# #                     raw list-of-rows-of-cells representation.
# #     """
# #     pages: list[PageRecord] = []
# #     raw_tables: dict[int, list[list[list[Optional[str]]]]] = {}

# #     with pdfplumber.open(pdf_path) as pdf:
# #         for i, page in enumerate(pdf.pages, start=1):
# #             raw_text = page.extract_text() or ""
# #             pages.append(
# #                 PageRecord(
# #                     page_number=i,
# #                     text=raw_text,
# #                     source="pdfplumber",
# #                     char_count=len(raw_text.strip()),
# #                 )
# #             )

# #             # extract_tables() uses pdfplumber's default line-detection settings,
# #             # which works well for ruled/bordered tables. Borderless or
# #             # whitespace-aligned tables may need custom table_settings — see
# #             # pdfplumber's docs if extraction misses tables you expect to find.
# #             tables_on_page = page.extract_tables()
# #             if tables_on_page:
# #                 raw_tables[i] = tables_on_page

# #     return pages, raw_tables


# # # ---------------------------------------------------------------------------
# # # Step 2: Detect pages with little or no text
# # # ---------------------------------------------------------------------------

# # def detect_low_text_pages(pages: list[PageRecord], threshold: int = MIN_CHARS_FOR_TEXT_PAGE) -> list[int]:
# #     """
# #     Return the page numbers whose pdfplumber extraction fell below `threshold`
# #     characters — these are the candidates that need OCR.
# #     """
# #     return [p.page_number for p in pages if p.char_count < threshold]


# # # ---------------------------------------------------------------------------
# # # Step 3: OCR the low-text pages
# # # ---------------------------------------------------------------------------

# # def ocr_pages(pdf_path: str, page_numbers: list[int], dpi: int = OCR_DPI) -> dict[int, str]:
# #     """
# #     Rasterize only the given pages and run Tesseract OCR on each.

# #     We deliberately only rasterize/OCR the pages that need it (not the whole
# #     PDF) — OCR is slow and unnecessary for pages that already have a clean
# #     text layer, so this keeps the pipeline fast on mostly-native documents.
# #     """
# #     if not page_numbers:
# #         return {}

# #     ocr_results: dict[int, str] = {}

# #     # pdf2image's first_page/last_page let us rasterize one page at a time
# #     # instead of the whole document, which keeps memory bounded for large PDFs.
# #     for page_number in page_numbers:
# #         images = convert_from_path(
# #             pdf_path, dpi=dpi, first_page=page_number, last_page=page_number
# #         )
# #         if not images:
# #             ocr_results[page_number] = ""
# #             continue

# #         ocr_text = pytesseract.image_to_string(images[0])
# #         ocr_results[page_number] = ocr_text

# #     return ocr_results


# # def merge_ocr_results(pages: list[PageRecord], ocr_results: dict[int, str]) -> list[PageRecord]:
# #     """Replace a page's text/source with its OCR output, where OCR was run."""
# #     merged = []
# #     for page in pages:
# #         if page.page_number in ocr_results:
# #             ocr_text = ocr_results[page.page_number]
# #             merged.append(
# #                 PageRecord(
# #                     page_number=page.page_number,
# #                     text=ocr_text,
# #                     source="ocr",
# #                     char_count=len(ocr_text.strip()),
# #                 )
# #             )
# #         else:
# #             merged.append(page)
# #     return merged


# # # ---------------------------------------------------------------------------
# # # Section detection (shared by text chunking and table records)
# # # ---------------------------------------------------------------------------

# # _HEADING_RE = re.compile(r"^[A-Z0-9][A-Za-z0-9 ,.:\-()]{2,80}$")


# # def detect_section_heading(text: str) -> Optional[str]:
# #     """
# #     Cheap heuristic: if the first non-empty line of a page looks like a
# #     heading (short, starts with a capital/number, no trailing period-heavy
# #     prose), treat it as this page's section label. This is intentionally
# #     simple — swap in a layout-aware tool (e.g. Docling) if you need
# #     accurate structural headings rather than a best-effort guess.
# #     """
# #     for line in text.splitlines():
# #         candidate = line.strip()
# #         if not candidate:
# #             continue
# #         if len(candidate) <= 80 and _HEADING_RE.match(candidate):
# #             return candidate
# #         break  # only ever look at the first non-empty line
# #     return None


# # def compute_page_sections(pages: list[PageRecord]) -> dict[int, Optional[str]]:
# #     """
# #     Walk the pages in order, carrying forward the most recently detected
# #     heading as the "current section" for every page until a new heading is
# #     found. Shared by text chunking and table-record building so a table on
# #     page 5 and a text chunk on page 5 agree on which section they belong to.
# #     """
# #     page_sections: dict[int, Optional[str]] = {}
# #     cursor_section: Optional[str] = None
# #     for page in pages:
# #         heading = detect_section_heading(page.text)
# #         if heading:
# #             cursor_section = heading
# #         page_sections[page.page_number] = cursor_section
# #     return page_sections


# # # ---------------------------------------------------------------------------
# # # Step 4: Clean and chunk text
# # # ---------------------------------------------------------------------------

# # def clean_text(text: str) -> str:
# #     """
# #     Normalize whitespace and repair common PDF-extraction artifacts:
# #       - hyphenated line-break words ("contrac-\\ntual" -> "contractual")
# #       - single newlines inside a paragraph collapsed to spaces
# #       - runs of blank lines collapsed to one
# #       - stray non-printable characters stripped
# #     """
# #     text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)          # dehyphenate across line breaks
# #     text = re.sub(r"[ \t]+", " ", text)                     # collapse repeated spaces/tabs
# #     text = re.sub(r"\n{3,}", "\n\n", text)                  # collapse 3+ blank lines to 1
# #     text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)             # single newlines -> space (keep paragraph breaks)
# #     text = "".join(ch for ch in text if ch.isprintable() or ch in "\n\t")
# #     return text.strip()


# # def chunk_pages(pages: list[PageRecord], document_name: str,
# #                  page_sections: dict[int, Optional[str]],
# #                  chunk_size: int = CHUNK_SIZE_CHARS,
# #                  overlap: int = CHUNK_OVERLAP_CHARS,
# #                  start_chunk_index: int = 0) -> list[ChunkRecord]:
# #     """
# #     Clean each page's text, then chunk the *concatenated* document text with
# #     a sliding window — chunks are allowed to span page boundaries (continuous
# #     prose in legal documents routinely does), but each chunk still records
# #     the exact page range and section it came from, which is what makes
# #     citations possible later. Table content is handled separately by
# #     build_table_chunks() so tables are never sliced mid-row.
# #     """
# #     buffer_parts = []
# #     offset_to_page: list[tuple[int, int, str]] = []  # (start_offset, page_number, source)
# #     running_offset = 0

# #     for page in pages:
# #         cleaned = clean_text(page.text)
# #         offset_to_page.append((running_offset, page.page_number, page.source))
# #         buffer_parts.append(cleaned)
# #         running_offset += len(cleaned) + 1  # +1 for the joining space added below
# #         buffer_parts.append(" ")

# #     full_text = "".join(buffer_parts)

# #     def lookup(offset: int) -> tuple[int, str]:
# #         """Find the page/source active at a given character offset."""
# #         page_number, source = pages[0].page_number, pages[0].source
# #         for start, pnum, src in offset_to_page:
# #             if start <= offset:
# #                 page_number, source = pnum, src
# #             else:
# #                 break
# #         return page_number, source

# #     chunks: list[ChunkRecord] = []
# #     start = 0
# #     chunk_index = start_chunk_index
# #     text_len = len(full_text)

# #     while start < text_len:
# #         end = min(start + chunk_size, text_len)
# #         chunk_text = full_text[start:end].strip()

# #         if chunk_text:
# #             page_start, source_start = lookup(start)
# #             page_end, source_end = lookup(max(end - 1, start))
# #             sources = sorted({source_start, source_end})

# #             chunks.append(
# #                 ChunkRecord(
# #                     chunk_id=str(uuid.uuid4()),
# #                     document_name=document_name,
# #                     chunk_index=chunk_index,
# #                     text=chunk_text,
# #                     page_start=page_start,
# #                     page_end=page_end,
# #                     section=page_sections.get(page_start),
# #                     content_type="text",
# #                     sources=sources,
# #                 )
# #             )
# #             chunk_index += 1

# #         if end == text_len:
# #             break
# #         start = end - overlap  # step forward, re-including `overlap` characters of context

# #     return chunks


# # # ---------------------------------------------------------------------------
# # # Step 5: Turn extracted tables into their own chunks
# # # ---------------------------------------------------------------------------

# # def table_to_markdown(table: list[list[Optional[str]]]) -> str:
# #     """
# #     Render a pdfplumber raw table (list of rows, each a list of cell strings
# #     or None) as a GitHub-flavored markdown table. Markdown is used because
# #     it's compact, human-readable in retrieved results, and embeds reasonably
# #     well — cell structure (rows/columns) survives instead of being flattened
# #     into ambiguous whitespace-separated text.
# #     """
# #     if not table:
# #         return ""

# #     def clean_cell(cell: Optional[str]) -> str:
# #         return (cell or "").replace("\n", " ").replace("|", "/").strip()

# #     header, *body_rows = table
# #     header_line = "| " + " | ".join(clean_cell(c) for c in header) + " |"
# #     separator_line = "| " + " | ".join("---" for _ in header) + " |"
# #     body_lines = [
# #         "| " + " | ".join(clean_cell(c) for c in row) + " |"
# #         for row in body_rows
# #     ]
# #     return "\n".join([header_line, separator_line, *body_lines])


# # def build_table_records(raw_tables: dict[int, list[list[list[Optional[str]]]]],
# #                           document_name: str,
# #                           page_sections: dict[int, Optional[str]]) -> list[TableRecord]:
# #     """Convert every raw extracted table into a TableRecord with its markdown rendering."""
# #     records: list[TableRecord] = []
# #     for page_number, tables_on_page in sorted(raw_tables.items()):
# #         for table_index, raw_table in enumerate(tables_on_page):
# #             markdown = table_to_markdown(raw_table)
# #             if not markdown:
# #                 continue
# #             records.append(
# #                 TableRecord(
# #                     table_id=str(uuid.uuid4()),
# #                     document_name=document_name,
# #                     page_number=page_number,
# #                     table_index=table_index,
# #                     section=page_sections.get(page_number),
# #                     markdown=markdown,
# #                     num_rows=len(raw_table),
# #                     num_cols=max((len(r) for r in raw_table), default=0),
# #                 )
# #             )
# #     return records


# # def build_table_chunks(table_records: list[TableRecord], start_chunk_index: int = 0) -> list[ChunkRecord]:
# #     """
# #     Wrap each TableRecord as its own ChunkRecord (content_type="table") so it
# #     flows through persistence and embedding the same way text chunks do,
# #     while never being split by the sliding-window chunker.
# #     """
# #     chunks: list[ChunkRecord] = []
# #     for i, t in enumerate(table_records):
# #         chunks.append(
# #             ChunkRecord(
# #                 chunk_id=t.table_id,
# #                 document_name=t.document_name,
# #                 chunk_index=start_chunk_index + i,
# #                 text=t.markdown,
# #                 page_start=t.page_number,
# #                 page_end=t.page_number,
# #                 section=t.section,
# #                 content_type="table",
# #                 sources=["pdfplumber"],
# #             )
# #         )
# #     return chunks


# # # ---------------------------------------------------------------------------
# # # Step 6: Persist metadata for traceability
# # # ---------------------------------------------------------------------------

# # def persist_metadata(chunks: list[ChunkRecord], output_path: str) -> str:
# #     """
# #     Write every chunk's metadata (and text) to a JSON file, so the exact
# #     page/section/source/content-type that backs any embedded chunk can
# #     always be audited later, independent of whatever's in the vector store.
# #     """
# #     payload = [asdict(c) for c in chunks]
# #     with open(output_path, "w", encoding="utf-8") as f:
# #         json.dump(payload, f, ensure_ascii=False, indent=2)
# #     return output_path


# # # ---------------------------------------------------------------------------
# # # Step 7: Embed and store in a vector database
# # # ---------------------------------------------------------------------------

# # def embed_and_store(chunks: list[ChunkRecord], collection_name: str,
# #                      embedding_model_name: str = EMBEDDING_MODEL_NAME,
# #                      persist_dir: str = CHROMA_PERSIST_DIR,
# #                      extra_metadata: Optional[dict] = None):
# #     """
# #     Embed every chunk (text or table) and upsert it into a persistent local
# #     Chroma collection.

# #     extra_metadata: contract-level fields (contract_type, parties,
# #     effective_date_epoch, end_date_epoch, monetary_value,
# #     governing_law_country — see retrieval/contract_metadata.py) merged into
# #     EVERY chunk's metadata, since Chroma filters per-chunk, not per-document.
# #     This is what HybridSearchAgent's metadata_filter filters on.

# #     Chroma is used here (rather than a hosted vector DB) because it needs no
# #     external service or credentials — it runs entirely inside the Colab
# #     session and persists to disk, which is the simplest thing that works for
# #     a notebook environment.
# #     """
# #     extra_metadata = extra_metadata or {}
# #     embedder = SentenceTransformer(embedding_model_name)
# #     texts = [c.text for c in chunks]
# #     # convert_to_tensor=True + Tensor.tolist() avoids routing through torch's
# #     # Tensor.numpy() bridge. In some Colab sessions that bridge breaks with
# #     # "RuntimeError: Numpy is not available" (usually a torch/numpy version
# #     # mismatch after pip installing new packages without restarting the
# #     # runtime) — .tolist() converts directly without touching numpy at all,
# #     # so embedding still works even when that bridge is broken.
# #     embeddings = embedder.encode(
# #         texts, normalize_embeddings=True, show_progress_bar=True, convert_to_tensor=True
# #     ).tolist()

# #     client = chromadb.PersistentClient(path=persist_dir)
# #     collection = client.get_or_create_collection(name=collection_name)

# #     collection.upsert(
# #         ids=[c.chunk_id for c in chunks],
# #         embeddings=embeddings,
# #         documents=texts,
# #         metadatas=[
# #             {
# #                 "document_name": c.document_name,
# #                 "chunk_index": c.chunk_index,
# #                 "page_start": c.page_start,
# #                 "page_end": c.page_end,
# #                 "section": c.section or "",
# #                 "content_type": c.content_type,
# #                 "sources": ",".join(c.sources),
# #                 **extra_metadata,
# #             }
# #             for c in chunks
# #         ],
# #     )

# #     return collection, embedder


# # def query_collection(collection, embedder: SentenceTransformer, query: str, top_k: int = 5):
# #     """Embed a query and retrieve the top_k most similar chunks, with citations."""
# #     query_embedding = embedder.encode([query], normalize_embeddings=True).tolist()
# #     results = collection.query(query_embeddings=query_embedding, n_results=top_k)

# #     hits = []
# #     for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
# #         hits.append({"text": doc, "metadata": meta, "distance": dist})
# #     return hits


# # # ---------------------------------------------------------------------------
# # # Orchestration
# # # ---------------------------------------------------------------------------

# # def run_pipeline(pdf_path: str, collection_name: Optional[str] = None,
# #                   metadata_output_path: Optional[str] = None,
# #                   sample_query: Optional[str] = None) -> dict:
# #     """
# #     Run the full extract -> OCR-fallback -> clean/chunk -> table-extract ->
# #     persist -> embed pipeline for a single PDF, and optionally run one
# #     sample retrieval query.
# #     """
# #     document_name = os.path.basename(pdf_path)
# #     collection_name = collection_name or re.sub(r"\W+", "_", document_name)
# #     metadata_output_path = metadata_output_path or os.path.join(DEFAULT_METADATA_DIR, f"{document_name}.metadata.json")

# #     print(f"[1/7] Extracting text and tables with pdfplumber: {document_name}")
# #     pages, raw_tables = extract_page_content(pdf_path)
# #     num_raw_tables = sum(len(t) for t in raw_tables.values())
# #     print(f"       {num_raw_tables} table(s) found across {len(raw_tables)} page(s)")

# #     print("[2/7] Detecting low-text pages")
# #     low_text_pages = detect_low_text_pages(pages)
# #     print(f"       {len(low_text_pages)} of {len(pages)} pages need OCR: {low_text_pages}")

# #     print("[3/7] Running OCR on low-text pages")
# #     ocr_results = ocr_pages(pdf_path, low_text_pages)
# #     pages = merge_ocr_results(pages, ocr_results)

# #     print("[4/7] Cleaning and chunking text")
# #     page_sections = compute_page_sections(pages)
# #     text_chunks = chunk_pages(pages, document_name, page_sections)
# #     print(f"       {len(text_chunks)} text chunks produced")

# #     print("[5/7] Building table chunks")
# #     table_records = build_table_records(raw_tables, document_name, page_sections)
# #     table_chunks = build_table_chunks(table_records, start_chunk_index=len(text_chunks))
# #     print(f"       {len(table_chunks)} table chunks produced")

# #     chunks = text_chunks + table_chunks

# #     print("[6/7] Persisting page/section/source/table metadata")
# #     metadata_path = persist_metadata(chunks, metadata_output_path)
# #     print(f"       written to {metadata_path}")

# #     print("[7/7] Embedding and storing in Chroma")
# #     collection, embedder = embed_and_store(chunks, collection_name)
# #     print(f"       stored in collection '{collection_name}'")

# #     result = {
# #         "document_name": document_name,
# #         "collection_name": collection_name,
# #         "metadata_path": metadata_path,
# #         "num_pages": len(pages),
# #         "num_ocr_pages": len(low_text_pages),
# #         "num_text_chunks": len(text_chunks),
# #         "num_tables": len(table_records),
# #         "num_chunks": len(chunks),
# #     }

# #     if sample_query:
# #         hits = query_collection(collection, embedder, sample_query)
# #         result["sample_query"] = sample_query
# #         result["sample_results"] = hits
# #         print(f"\nSample query: {sample_query!r}")
# #         for h in hits:
# #             m = h["metadata"]
# #             tag = "[TABLE]" if m["content_type"] == "table" else "[TEXT] "
# #             print(f"  {tag} p.{m['page_start']}-{m['page_end']} [{m['section'] or 'no section'}] "
# #                   f"(dist={h['distance']:.4f}): {h['text'][:120]}...")

# #     return result


# # if __name__ == "__main__":
# #     # Example (adjust the path to a PDF uploaded in your Colab session):
# #     run_pipeline("data/uploads/sample.pdf", sample_query="What is the termination clause?")
