"""
hybrid_search.py
================

Hierarchy-aware hybrid retrieval for legal / financial documents.

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
                                        +--> Chunk

Retrieval pipeline:

    Query
      |
      +--> Dense retrieval
      |
      +--> BM25 retrieval
      |
      +--> Hybrid fusion
      |
      +--> Hierarchy-aware boosting
      |
      +--> Deduplication
      |
      +--> Parent / sibling expansion
      |
      +--> Cross-encoder reranking
      |
      +--> Final results


Metadata expected from pdf_pipeline.py:

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

Document metadata:

    contract_type
    parties
    governing_law_country

    effective_date_epoch
    end_date_epoch
    monetary_value


TABLE SUPPORT
-------------

Tables remain intact as retrieval chunks.

A table is therefore retrieved through:

    1. its own semantic content
    2. its own lexical content
    3. its hierarchy metadata

Example:

    Query:

        "What are the liabilities shown in clause 8.2?"

can retrieve:

        clause_number = 8
        subclause_number = 8.2
        content_type = table

even if the table itself does not repeat the complete clause title.


BM25
----

BM25 is maintained in memory.

The cache is invalidated by pdf_pipeline.py after ingestion.

For very large production collections, move sparse retrieval to a
native sparse vector backend such as Qdrant.
"""

from __future__ import annotations

import math
import re

from collections import Counter
from dataclasses import dataclass
from typing import Any, Optional

from ..resources import get_chroma_collection
from ..resources import get_embedder
from ..resources import get_reranker

from ..tracing import traceable


# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_DENSE_K = 30
DEFAULT_SPARSE_K = 30

DEFAULT_FUSION_K = 30
DEFAULT_FINAL_K = 6

DEFAULT_ALPHA = 0.45

DEFAULT_NEIGHBOR_WINDOW = 1

DEFAULT_MAX_EXPANDED_CANDIDATES = 40


# ============================================================================
# HIERARCHY BOOSTS
# ============================================================================

SECTION_BOOST = 0.08

CLAUSE_NUMBER_BOOST = 0.20
CLAUSE_TITLE_BOOST = 0.15

PARENT_CLAUSE_BOOST = 0.10

SUBCLAUSE_NUMBER_BOOST = 0.25
SUBCLAUSE_TITLE_BOOST = 0.15

DOCUMENT_NAME_BOOST = 0.05

TABLE_BOOST = 0.05


# ============================================================================
# BM25 CACHE
# ============================================================================

_bm25_cache: dict[
    str,
    tuple[
        "LegalBM25",
        list[str],
        list[str],
        list[dict],
    ],
] = {}


# ============================================================================
# TOKENIZATION
# ============================================================================

_TOKEN_RE = re.compile(
    r"[a-z0-9]+"
)


def _tokenize_raw(
    text: str,
) -> list[str]:
    """
    Raw tokenizer.

    Keeps:

        termination
        indemnification
        revenue
        12
        12.3
        12.3.1
    """

    if not text:
        return []

    return _TOKEN_RE.findall(
        text.lower()
    )


LEGAL_STOPWORDS = {
    # Articles
    "a",
    "an",
    "the",

    # Auxiliaries
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "has",
    "have",
    "had",
    "do",
    "does",
    "did",
    "can",
    "could",
    "would",
    "should",
    "will",

    # Prepositions
    "of",
    "to",
    "in",
    "on",
    "for",
    "from",
    "by",
    "with",
    "at",
    "into",
    "about",
    "over",
    "under",

    # Conjunctions
    "and",
    "or",
    "but",
    "if",
    "then",
    "than",

    # Pronouns
    "it",
    "its",
    "this",
    "that",
    "these",
    "those",

    # Question words
    "what",
    "which",
    "when",
    "where",
    "who",
    "how",
}


def _tokenize(
    text: str,
) -> list[str]:
    """
    Legal-aware tokenizer.

    Generic English stopwords are removed, but legal-domain words
    such as:

        agreement
        party
        parties
        shall
        termination
        effective
        date
        law
        liability
        indemnity

    are retained.
    """

    tokens = _tokenize_raw(
        text
    )

    return [
        token
        for token in tokens
        if token not in LEGAL_STOPWORDS
    ]


# ============================================================================
# LEGAL PHRASES
# ============================================================================

LEGAL_PHRASES = (
    "governing law",
    "effective date",
    "expiration date",
    "termination date",
    "termination agreement",
    "confidential information",
    "intellectual property",
    "limitation of liability",
    "liability limitation",
    "indemnification",
    "indemnity",
    "force majeure",
    "change of control",
    "notice period",
    "notice provision",
    "non compete",
    "non solicitation",
    "assignment",
    "representations and warranties",
    "representation and warranty",
    "warranty",
    "warranties",
)


def _extract_query_phrases(
    query: str,
) -> list[str]:
    """
    Find known legal phrases occurring in the query.
    """

    normalized = re.sub(
        r"\s+",
        " ",
        query.lower(),
    ).strip()

    return [
        phrase
        for phrase in LEGAL_PHRASES
        if phrase in normalized
    ]


# ============================================================================
# CLAUSE NUMBER EXTRACTION
# ============================================================================

def _extract_hierarchy_numbers(
    query: str,
) -> dict[str, list[str]]:
    """
    Extract likely clause / subclause numbers from a query.

    Examples:

        "clause 12.3"
        "section 8.2"
        "12.3.1"

    """

    numbers = re.findall(
        r"\b\d+(?:\.\d+){0,4}\b",
        query,
    )

    result = {
        "clause_numbers": [],
        "subclause_numbers": [],
    }

    for number in numbers:

        components = number.split(".")

        if len(components) == 1:
            result[
                "clause_numbers"
            ].append(number)

        else:
            result[
                "subclause_numbers"
            ].append(number)

            # Parent clause of 12.3.1 is 12.
            result[
                "clause_numbers"
            ].append(
                components[0]
            )

    return result


# ============================================================================
# LEGAL BM25
# ============================================================================

@dataclass
class LegalBM25:

    tokenized_corpus: list[list[str]]

    k1: float = 1.5

    b: float = 0.75

    phrase_bonus: float = 2.0

    def __post_init__(
        self,
    ) -> None:

        self.N = len(
            self.tokenized_corpus
        )

        self.doc_len = [
            len(tokens)
            for tokens
            in self.tokenized_corpus
        ]

        self.avgdl = (
            sum(self.doc_len)
            / self.N
            if self.N
            else 0.0
        )

        self.doc_freqs: list[
            Counter[str]
        ] = []

        self.df: Counter[str] = (
            Counter()
        )

        for tokens in (
            self.tokenized_corpus
        ):

            frequencies = Counter(
                tokens
            )

            self.doc_freqs.append(
                frequencies
            )

            for term in frequencies:

                self.df[term] += 1

        self.idf: dict[
            str,
            float,
        ] = {}

        for term, df in (
            self.df.items()
        ):

            self.idf[term] = math.log(
                1.0
                + (
                    self.N
                    - df
                    + 0.5
                )
                / (
                    df
                    + 0.5
                )
            )

    def get_scores(
        self,
        query_tokens: list[str],
    ) -> list[float]:

        if not self.tokenized_corpus:
            return []

        scores = [
            0.0
            for _ in range(
                self.N
            )
        ]

        for term in query_tokens:

            if term not in self.idf:
                continue

            idf = self.idf[
                term
            ]

            for index, frequencies in (
                enumerate(
                    self.doc_freqs
                )
            ):

                tf = frequencies.get(
                    term,
                    0,
                )

                if tf == 0:
                    continue

                doc_length = (
                    self.doc_len[
                        index
                    ]
                )

                if self.avgdl == 0:

                    length_normalization = (
                        1.0
                    )

                else:

                    length_normalization = (
                        1.0
                        - self.b
                        + self.b
                        * doc_length
                        / self.avgdl
                    )

                denominator = (
                    tf
                    + self.k1
                    * length_normalization
                )

                scores[index] += (
                    idf
                    * tf
                    * (
                        self.k1
                        + 1.0
                    )
                    / denominator
                )

        return scores


# ============================================================================
# BM25 INDEX
# ============================================================================

def _build_bm25_index(
    collection_name: str,
):
    """
    Load all Chroma documents and build the lexical index.
    """

    collection = (
        get_chroma_collection(
            collection_name
        )
    )

    raw = collection.get(
        include=[
            "documents",
            "metadatas",
        ]
    )

    ids = raw[
        "ids"
    ]

    texts = raw[
        "documents"
    ]

    metadatas = raw[
        "metadatas"
    ]

    tokenized_documents = [
        _tokenize(text)
        for text in texts
    ]

    bm25 = LegalBM25(
        tokenized_corpus=(
            tokenized_documents
        ),
        k1=1.5,
        b=0.75,
        phrase_bonus=2.0,
    )

    _bm25_cache[
        collection_name
    ] = (
        bm25,
        ids,
        texts,
        metadatas,
    )

    return _bm25_cache[
        collection_name
    ]


def _get_bm25_index(
    collection_name: str,
    refresh: bool = False,
):
    """
    Get cached BM25 index.

    refresh=True forces reconstruction.
    """

    if (
        not refresh
        and collection_name
        in _bm25_cache
    ):

        return _bm25_cache[
            collection_name
        ]

    return _build_bm25_index(
        collection_name
    )


def invalidate_bm25_cache(
    collection_name: str,
) -> None:
    """
    Called by pdf_pipeline.py after ingestion.
    """

    _bm25_cache.pop(
        collection_name,
        None,
    )


# ============================================================================
# METADATA FILTERING
# ============================================================================

def build_where_clause(
    filters: dict,
) -> Optional[dict]:
    """
    Convert application metadata filters into Chroma syntax.
    """

    clauses: list[dict] = []

    if filters.get(
        "contract_type"
    ):

        clauses.append(
            {
                "contract_type":
                    filters[
                        "contract_type"
                    ]
            }
        )

    if filters.get(
        "governing_law_country"
    ):

        clauses.append(
            {
                "governing_law_country":
                    filters[
                        "governing_law_country"
                    ].upper()
            }
        )

    if filters.get(
        "min_effective_date_epoch"
    ) is not None:

        clauses.append(
            {
                "effective_date_epoch": {
                    "$gte":
                        filters[
                            "min_effective_date_epoch"
                        ]
                }
            }
        )

    if filters.get(
        "max_effective_date_epoch"
    ) is not None:

        clauses.append(
            {
                "effective_date_epoch": {
                    "$lte":
                        filters[
                            "max_effective_date_epoch"
                        ]
                }
            }
        )

    if filters.get(
        "min_end_date_epoch"
    ) is not None:

        clauses.append(
            {
                "end_date_epoch": {
                    "$gte":
                        filters[
                            "min_end_date_epoch"
                        ]
                }
            }
        )

    if filters.get(
        "max_end_date_epoch"
    ) is not None:

        clauses.append(
            {
                "end_date_epoch": {
                    "$lte":
                        filters[
                            "max_end_date_epoch"
                        ]
                }
            }
        )

    if filters.get(
        "min_monetary_value"
    ) is not None:

        clauses.append(
            {
                "monetary_value": {
                    "$gte":
                        filters[
                            "min_monetary_value"
                        ]
                }
            }
        )

    if filters.get(
        "max_monetary_value"
    ) is not None:

        clauses.append(
            {
                "monetary_value": {
                    "$lte":
                        filters[
                            "max_monetary_value"
                        ]
                }
            }
        )

    if not clauses:
        return None

    if len(clauses) == 1:
        return clauses[0]

    return {
        "$and": clauses
    }


def _matches_where(
    metadata: dict,
    where: Optional[dict],
) -> bool:
    """
    Apply the same filters to the in-memory BM25 index.
    """

    if not where:
        return True

    if "$and" in where:

        return all(
            _matches_where(
                metadata,
                clause,
            )
            for clause in where[
                "$and"
            ]
        )

    (
        field,
        condition,
    ), = where.items()

    value = metadata.get(
        field
    )

    if isinstance(
        condition,
        dict,
    ):

        if "$gte" in condition:

            if (
                value is None
                or value
                < condition[
                    "$gte"
                ]
            ):
                return False

        if "$lte" in condition:

            if (
                value is None
                or value
                > condition[
                    "$lte"
                ]
            ):
                return False

        return True

    return value == condition


# ============================================================================
# DENSE RETRIEVAL
# ============================================================================

def _dense_search(
    collection_name: str,
    query: str,
    top_k: int,
    where: Optional[dict],
) -> list[dict]:
    """
    Semantic retrieval through Chroma.
    """

    collection = (
        get_chroma_collection(
            collection_name
        )
    )

    embedder = get_embedder()

    query_embedding = (
        embedder.encode(
            [query],
            normalize_embeddings=True,
        ).tolist()
    )

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        where=where,
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    hits: list[dict] = []

    if not results.get(
        "ids"
    ):
        return hits

    for (
        doc_id,
        text,
        metadata,
        distance,
    ) in zip(
        results["ids"][0],
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):

        hits.append(
            {
                "id":
                    doc_id,

                "text":
                    text,

                "metadata":
                    metadata,

                "dense_distance":
                    float(distance),
            }
        )

    return hits


# ============================================================================
# PHRASE BOOSTING
# ============================================================================

def _apply_phrase_boost(
    scores: list[float],
    texts: list[str],
    phrases: list[str],
    phrase_bonus: float = 2.0,
) -> list[float]:

    if not phrases:
        return scores

    boosted = scores.copy()

    for index, text in enumerate(
        texts
    ):

        normalized = re.sub(
            r"\s+",
            " ",
            text.lower(),
        )

        for phrase in phrases:

            if phrase in normalized:

                boosted[
                    index
                ] += phrase_bonus

    return boosted


# ============================================================================
# SPARSE RETRIEVAL
# ============================================================================

def _sparse_search(
    collection_name: str,
    query: str,
    top_k: int,
    where: Optional[dict],
) -> list[dict]:
    """
    Legal-aware BM25 retrieval.
    """

    (
        bm25,
        ids,
        texts,
        metadatas,
    ) = _get_bm25_index(
        collection_name
    )

    query_tokens = _tokenize(
        query
    )

    scores = bm25.get_scores(
        query_tokens
    )

    phrases = (
        _extract_query_phrases(
            query
        )
    )

    scores = _apply_phrase_boost(
        scores=scores,
        texts=texts,
        phrases=phrases,
        phrase_bonus=(
            bm25.phrase_bonus
        ),
    )

    eligible = [
        index
        for index in range(
            len(ids)
        )
        if _matches_where(
            metadatas[index],
            where,
        )
    ]

    ranked = sorted(
        eligible,
        key=lambda index:
            scores[index],
        reverse=True,
    )

    ranked = ranked[
        :top_k
    ]

    return [
        {
            "id":
                ids[index],

            "text":
                texts[index],

            "metadata":
                metadatas[index],

            "bm25_score":
                float(
                    scores[index]
                ),
        }
        for index in ranked
    ]


# ============================================================================
# SCORE NORMALIZATION
# ============================================================================

def min_max_normalize(
    scores: dict[str, float],
    reverse: bool = False,
) -> dict[str, float]:

    if not scores:
        return {}

    values = list(
        scores.values()
    )

    minimum = min(
        values
    )

    maximum = max(
        values
    )

    if minimum == maximum:

        return {
            doc_id: 1.0
            for doc_id in scores
        }

    normalized = {}

    for doc_id, score in (
        scores.items()
    ):

        if reverse:

            value = (
                maximum - score
            ) / (
                maximum - minimum
            )

        else:

            value = (
                score - minimum
            ) / (
                maximum - minimum
            )

        normalized[
            doc_id
        ] = float(value)

    return normalized


# ============================================================================
# CONVEX FUSION
# ============================================================================

def convex_combination_fusion(
    dense_hits: list[dict],
    sparse_hits: list[dict],
    alpha: float = DEFAULT_ALPHA,
) -> dict[str, float]:
    """
    Convex hybrid fusion.

        score =
            alpha * dense
            +
            (1-alpha) * sparse

    Chroma distance is reversed because lower distance is better.
    """

    raw_dense = {
        hit["id"]:
            hit["dense_distance"]
        for hit in dense_hits
    }

    raw_sparse = {
        hit["id"]:
            hit["bm25_score"]
        for hit in sparse_hits
    }

    dense_normalized = (
        min_max_normalize(
            raw_dense,
            reverse=True,
        )
    )

    sparse_normalized = (
        min_max_normalize(
            raw_sparse,
            reverse=False,
        )
    )

    all_ids = (
        set(
            dense_normalized
        )
        |
        set(
            sparse_normalized
        )
    )

    scores = {}

    for doc_id in all_ids:

        dense_score = (
            dense_normalized.get(
                doc_id,
                0.0,
            )
        )

        sparse_score = (
            sparse_normalized.get(
                doc_id,
                0.0,
            )
        )

        scores[
            doc_id
        ] = (
            alpha
            * dense_score
            +
            (1.0 - alpha)
            * sparse_score
        )

    return dict(
        sorted(
            scores.items(),
            key=lambda item:
                item[1],
            reverse=True,
        )
    )


# ============================================================================
# RRF
# ============================================================================

def reciprocal_rank_fusion(
    id_lists: list[list[str]],
    k: int = 60,
) -> dict[str, float]:
    """
    Reciprocal Rank Fusion.
    """

    scores: dict[
        str,
        float,
    ] = {}

    for id_list in id_lists:

        for rank, doc_id in enumerate(
            id_list
        ):

            scores[
                doc_id
            ] = (
                scores.get(
                    doc_id,
                    0.0,
                )
                +
                1.0
                / (
                    k
                    + rank
                    + 1
                )
            )

    return dict(
        sorted(
            scores.items(),
            key=lambda item:
                item[1],
            reverse=True,
        )
    )


# ============================================================================
# HIERARCHY MATCHING
# ============================================================================

def _normalized(
    value: Any,
) -> str:

    if value is None:
        return ""

    return str(
        value
    ).strip().lower()


def _contains_term(
    value: Any,
    query: str,
) -> bool:

    value_normalized = _normalized(
        value
    )

    query_normalized = _normalized(
        query
    )

    if not value_normalized:
        return False

    return (
        query_normalized
        in value_normalized
    )


# ============================================================================
# HIERARCHY BOOST
# ============================================================================

def hierarchy_boost(
    query: str,
    metadata: dict,
) -> float:
    """
    Calculate hierarchy-aware retrieval boost.

    The query is inspected for:

        section names
        clause numbers
        subclause numbers
        clause titles
        document names

    Exact hierarchy references receive stronger boosts than generic
    lexical matches.
    """

    boost = 0.0

    hierarchy_numbers = (
        _extract_hierarchy_numbers(
            query
        )
    )

    clause_numbers = (
        hierarchy_numbers[
            "clause_numbers"
        ]
    )

    subclause_numbers = (
        hierarchy_numbers[
            "subclause_numbers"
        ]
    )

    # ------------------------------------------------------------
    # Clause number
    # ------------------------------------------------------------

    candidate_clause = (
        _normalized(
            metadata.get(
                "clause_number"
            )
        )
    )

    for clause_number in (
        clause_numbers
    ):

        if (
            candidate_clause
            == _normalized(
                clause_number
            )
        ):

            boost += (
                CLAUSE_NUMBER_BOOST
            )

    # ------------------------------------------------------------
    # Subclause number
    # ------------------------------------------------------------

    candidate_subclause = (
        _normalized(
            metadata.get(
                "subclause_number"
            )
        )
    )

    for number in (
        subclause_numbers
    ):

        if (
            candidate_subclause
            == _normalized(
                number
            )
        ):

            boost += (
                SUBCLAUSE_NUMBER_BOOST
            )

    # ------------------------------------------------------------
    # Clause title
    # ------------------------------------------------------------

    clause_title = (
        _normalized(
            metadata.get(
                "clause_title"
            )
        )
    )

    if (
        clause_title
        and _query_overlaps_field(
            query,
            clause_title,
        )
    ):

        boost += (
            CLAUSE_TITLE_BOOST
        )

    # ------------------------------------------------------------
    # Subclause title
    # ------------------------------------------------------------

    subclause_title = (
        _normalized(
            metadata.get(
                "subclause_title"
            )
        )
    )

    if (
        subclause_title
        and _query_overlaps_field(
            query,
            subclause_title,
        )
    ):

        boost += (
            SUBCLAUSE_TITLE_BOOST
        )

    # ------------------------------------------------------------
    # Parent clause
    # ------------------------------------------------------------

    parent_clause = (
        _normalized(
            metadata.get(
                "parent_clause"
            )
        )
    )

    for clause_number in (
        clause_numbers
    ):

        if (
            parent_clause
            == _normalized(
                clause_number
            )
        ):

            boost += (
                PARENT_CLAUSE_BOOST
            )

    # ------------------------------------------------------------
    # Section
    # ------------------------------------------------------------

    section = (
        _normalized(
            metadata.get(
                "section"
            )
        )
    )

    if (
        section
        and _query_overlaps_field(
            query,
            section,
        )
    ):

        boost += (
            SECTION_BOOST
        )

    # ------------------------------------------------------------
    # Document name
    # ------------------------------------------------------------

    document_name = (
        _normalized(
            metadata.get(
                "document_name"
            )
        )
    )

    if (
        document_name
        and _query_overlaps_field(
            query,
            document_name,
        )
    ):

        boost += (
            DOCUMENT_NAME_BOOST
        )

    # ------------------------------------------------------------
    # Tables
    # ------------------------------------------------------------

    if (
        metadata.get(
            "content_type"
        )
        == "table"
    ):

        boost += (
            TABLE_BOOST
        )

    return boost


def _query_overlaps_field(
    query: str,
    field: str,
) -> bool:
    """
    Determine whether meaningful query tokens overlap with a hierarchy
    field.

    Generic stopwords are ignored.
    """

    query_tokens = set(
        _tokenize(query)
    )

    field_tokens = set(
        _tokenize(field)
    )

    if not query_tokens:
        return False

    return bool(
        query_tokens
        &
        field_tokens
    )


# ============================================================================
# APPLY HIERARCHY BOOST
# ============================================================================

def apply_hierarchy_boost(
    query: str,
    candidates: list[dict],
) -> list[dict]:
    """
    Add hierarchy_boost_score to every candidate.
    """

    for candidate in candidates:

        boost = hierarchy_boost(
            query=query,
            metadata=(
                candidate.get(
                    "metadata",
                    {},
                )
            ),
        )

        candidate[
            "hierarchy_boost"
        ] = float(
            boost
        )

        candidate[
            "hybrid_score"
        ] = (
            candidate.get(
                "fusion_score",
                0.0,
            )
            + boost
        )

    return sorted(
        candidates,
        key=lambda candidate:
            candidate[
                "hybrid_score"
            ],
        reverse=True,
    )


# ============================================================================
# DEDUPLICATION
# ============================================================================

def _deduplicate_hits(
    hits: list[dict],
) -> list[dict]:
    """
    Deduplicate the same logical chunk.

    We use:

        document_name
        chunk_index
        text

    rather than only Chroma UUID.
    """

    seen: set[
        tuple[
            str,
            Any,
            str,
        ]
    ] = set()

    unique: list[
        dict
    ] = []

    for hit in hits:

        metadata = hit.get(
            "metadata",
            {},
        )

        key = (
            str(
                metadata.get(
                    "document_name",
                    "",
                )
            ),
            metadata.get(
                "chunk_index"
            ),
            (
                hit.get(
                    "text",
                    "",
                )
                or ""
            ).strip(),
        )

        if key in seen:
            continue

        seen.add(
            key
        )

        unique.append(
            hit
        )

    return unique


# ============================================================================
# PARENT / SIBLING EXPANSION
# ============================================================================

def _hierarchy_key(
    metadata: dict,
) -> tuple:
    """
    Create a hierarchy key.

    Used to identify chunks belonging to the same logical clause.
    """

    return (
        metadata.get(
            "document_name"
        ),

        metadata.get(
            "section"
        ),

        metadata.get(
            "clause_number"
        ),

        metadata.get(
            "parent_clause"
        ),

        metadata.get(
            "subclause_number"
        ),
    )


def _same_clause(
    left: dict,
    right: dict,
) -> bool:

    left_meta = left.get(
        "metadata",
        {},
    )

    right_meta = right.get(
        "metadata",
        {},
    )

    return (
        left_meta.get(
            "document_name"
        )
        ==
        right_meta.get(
            "document_name"
        )
        and
        left_meta.get(
            "clause_number"
        )
        ==
        right_meta.get(
            "clause_number"
        )
    )


def _same_subclause(
    left: dict,
    right: dict,
) -> bool:

    left_meta = left.get(
        "metadata",
        {},
    )

    right_meta = right.get(
        "metadata",
        {},
    )

    return (
        _same_clause(
            left,
            right,
        )
        and
        left_meta.get(
            "subclause_number"
        )
        ==
        right_meta.get(
            "subclause_number"
        )
    )


def expand_hierarchy_context(
    collection_name: str,
    candidates: list[dict],
    max_candidates: int = DEFAULT_MAX_EXPANDED_CANDIDATES,
) -> list[dict]:
    """
    Recover parent/sibling context.

    For every high-ranking candidate we retrieve other chunks from the
    same clause/subclause.

    This is especially important when:

        chunk 42 = beginning of clause
        chunk 43 = middle of clause
        chunk 44 = table
        chunk 45 = continuation

    If chunk 44 is retrieved, the surrounding clause can still be
    recovered.

    IMPORTANT:

    The table remains intact. Expansion does NOT split or modify it.
    """

    if not candidates:
        return []

    collection = (
        get_chroma_collection(
            collection_name
        )
    )

    # ------------------------------------------------------------
    # Seed candidates
    # ------------------------------------------------------------

    seed_candidates = candidates[
        :min(
            len(candidates),
            DEFAULT_FUSION_K,
        )
    ]

    # ------------------------------------------------------------
    # Determine documents / clauses to expand
    # ------------------------------------------------------------

    target_documents = set()
    target_clauses = set()
    target_subclauses = set()

    for candidate in (
        seed_candidates
    ):

        metadata = candidate.get(
            "metadata",
            {},
        )

        document = metadata.get(
            "document_name"
        )

        clause = metadata.get(
            "clause_number"
        )

        subclause = metadata.get(
            "subclause_number"
        )

        if document:
            target_documents.add(
                document
            )

        if (
            document
            and clause
        ):
            target_clauses.add(
                (
                    document,
                    clause,
                )
            )

        if (
            document
            and subclause
        ):
            target_subclauses.add(
                (
                    document,
                    subclause,
                )
            )

    # ------------------------------------------------------------
    # Chroma does not conveniently express all of these hierarchy
    # conditions as one generic OR in every configuration.
    #
    # Retrieve a bounded set and filter locally.
    # ------------------------------------------------------------

    try:

        raw = collection.get(
            include=[
                "documents",
                "metadatas",
            ]
        )

    except Exception:

        return candidates

    expanded = list(
        candidates
    )

    existing_ids = {
        candidate[
            "id"
        ]
        for candidate in candidates
    }

    for (
        doc_id,
        text,
        metadata,
    ) in zip(
        raw["ids"],
        raw["documents"],
        raw["metadatas"],
    ):

        if doc_id in existing_ids:
            continue

        document = metadata.get(
            "document_name"
        )

        clause = metadata.get(
            "clause_number"
        )

        subclause = metadata.get(
            "subclause_number"
        )

        is_same_subclause = (
            (
                document,
                subclause,
            )
            in target_subclauses
        )

        is_same_clause = (
            (
                document,
                clause,
            )
            in target_clauses
        )

        if not (
            is_same_subclause
            or is_same_clause
        ):
            continue

        expanded.append(
            {
                "id":
                    doc_id,

                "text":
                    text,

                "metadata":
                    metadata,

                "expanded":
                    True,

                "fusion_score":
                    0.0,

                "hierarchy_boost":
                    0.0,

                "hybrid_score":
                    0.0,
            }
        )

        existing_ids.add(
            doc_id
        )

        if len(expanded) >= (
            max_candidates
        ):
            break

    return _deduplicate_hits(
        expanded
    )


# ============================================================================
# RERANKING
# ============================================================================

def rerank(
    query: str,
    candidates: list[dict],
    top_k: int,
) -> list[dict]:
    """
    Cross-encoder reranking.

    The hierarchy-expanded candidates are all evaluated against the
    original user query.
    """

    if not candidates:
        return []

    reranker = get_reranker()

    pairs = [
        (
            query,
            candidate[
                "text"
            ],
        )
        for candidate in candidates
    ]

    scores = reranker.predict(
        pairs
    )

    for candidate, score in zip(
        candidates,
        scores,
    ):

        candidate[
            "rerank_score"
        ] = float(
            score
        )

    return sorted(
        candidates,
        key=lambda candidate:
            candidate[
                "rerank_score"
            ],
        reverse=True,
    )[:top_k]


# ============================================================================
# PARTIES FILTER
# ============================================================================

def _apply_parties_filter(
    candidates: list[dict],
    parties_contains: Optional[str],
) -> list[dict]:

    if not parties_contains:
        return candidates

    needle = (
        parties_contains
        .lower()
        .strip()
    )

    return [
        candidate
        for candidate in candidates
        if needle
        in (
            candidate.get(
                "metadata",
                {},
            ).get(
                "parties",
                "",
            )
            or ""
        ).lower()
    ]


# ============================================================================
# PUBLIC HYBRID SEARCH
# ============================================================================

@traceable(
    name="hybrid_search_agent.search",
    run_type="retriever",
)
def hybrid_search(
    collection_name: str,
    query: str,
    metadata_filter: Optional[
        dict
    ] = None,
    dense_k: int = DEFAULT_DENSE_K,
    sparse_k: int = DEFAULT_SPARSE_K,
    fusion_k: int = DEFAULT_FUSION_K,
    final_k: int = DEFAULT_FINAL_K,
    fusion_method: str = "convex",
    alpha: float = DEFAULT_ALPHA,
    expand_hierarchy: bool = True,
    rerank_results: bool = True,
) -> list[dict]:
    """
    Complete hierarchy-aware hybrid retrieval.

    Parameters
    ----------
    collection_name:
        Chroma collection.

    query:
        User question.

    metadata_filter:
        Optional document-level filters.

    dense_k:
        Number of semantic candidates.

    sparse_k:
        Number of BM25 candidates.

    fusion_k:
        Candidates retained after fusion.

    final_k:
        Number of final results.

    fusion_method:
        "convex" or "rrf".

    alpha:
        Dense weight for convex fusion.

        0.25 -> lexical heavy
        0.45 -> balanced legal retrieval
        0.70 -> semantic heavy

    expand_hierarchy:
        Whether to retrieve surrounding clause/subclause chunks.

    rerank_results:
        Whether to apply the cross-encoder.

    Returns
    -------
    list[dict]
        Final ranked retrieval results.
    """

    # ------------------------------------------------------------------
    # Effective alpha for convex fusion.  score = alpha*dense + (1-alpha)*sparse
    #   - `alpha` arrives from the caller (the router sets state["alpha"], which
    #     is what the [router]/[hybrid_search_agent] lines in legal_pipeline.py
    #     print — those are the ROUTER's value, NOT necessarily what fusion uses).
    #   - HYBRID_ALPHA=<0..1> (env) overrides it here so you can force the blend
    #     without touching the router. 0.5 = balanced, 0.7 = semantic-heavy.
    # We ALWAYS print the effective alpha below so you can see the real weights.
    # ------------------------------------------------------------------
    import os as _os
    _router_alpha = alpha
    _alpha_env = _os.getenv("HYBRID_ALPHA")
    if _alpha_env is not None and _alpha_env.strip() != "":
        try:
            alpha = min(1.0, max(0.0, float(_alpha_env)))
            print(f"[hybrid] HYBRID_ALPHA override: router alpha={_router_alpha} -> using alpha={alpha}")
        except ValueError:
            print(f"[hybrid] HYBRID_ALPHA={_alpha_env!r} is not a number — ignoring, using alpha={alpha}")
    _hybrid_debug = _os.getenv("HYBRID_DEBUG", "").strip().lower() in ("1", "true", "yes")

    metadata_filter = (
        metadata_filter
        or {}
    )

    # ========================================================================
    # 1. METADATA FILTER
    # ========================================================================

    where = build_where_clause(
        metadata_filter
    )

    # ========================================================================
    # 2. DENSE RETRIEVAL
    # ========================================================================

    dense_hits = _dense_search(
        collection_name=(
            collection_name
        ),
        query=query,
        top_k=dense_k,
        where=where,
    )

    # ========================================================================
    # 3. BM25 RETRIEVAL
    # ========================================================================

    sparse_hits = _sparse_search(
        collection_name=(
            collection_name
        ),
        query=query,
        top_k=sparse_k,
        where=where,
    )

    # Always report the EFFECTIVE fusion weights + per-source hit counts, so
    # you can confirm what alpha actually drove retrieval (dense vs sparse).
    print(
        f"[hybrid] fusion={fusion_method} "
        f"alpha={alpha:.2f} (dense weight) | 1-alpha={1.0 - alpha:.2f} (sparse weight) | "
        f"dense_hits={len(dense_hits)} sparse_hits={len(sparse_hits)}"
    )
    if _hybrid_debug:
        _top_dense = [round(h.get("dense_distance", 0.0), 4) for h in dense_hits[:5]]
        _top_sparse = [round(h.get("bm25_score", 0.0), 4) for h in sparse_hits[:5]]
        print(f"[hybrid] top_dense_distances={_top_dense} top_sparse_bm25={_top_sparse}")
    if not dense_hits:
        print("[hybrid] WARNING: dense returned 0 hits — check embeddings / collection.")

    # ========================================================================
    # 4. MERGE CANDIDATES
    # ========================================================================

    by_id: dict[
        str,
        dict,
    ] = {}

    for hit in (
        dense_hits
        + sparse_hits
    ):

        doc_id = hit[
            "id"
        ]

        if doc_id not in by_id:

            by_id[
                doc_id
            ] = {}

        by_id[
            doc_id
        ].update(
            hit
        )

    # ========================================================================
    # 5. HYBRID FUSION
    # ========================================================================

    if (
        fusion_method.lower()
        == "rrf"
    ):

        fused_scores = (
            reciprocal_rank_fusion(
                [
                    [
                        hit["id"]
                        for hit
                        in dense_hits
                    ],
                    [
                        hit["id"]
                        for hit
                        in sparse_hits
                    ],
                ]
            )
        )

    else:

        fused_scores = (
            convex_combination_fusion(
                dense_hits=dense_hits,
                sparse_hits=sparse_hits,
                alpha=alpha,
            )
        )

    # ========================================================================
    # 6. CREATE FUSED CANDIDATES
    # ========================================================================

    fused_ids = list(
        fused_scores.keys()
    )[:fusion_k]

    candidates: list[
        dict
    ] = []

    for doc_id in fused_ids:

        if doc_id not in by_id:
            continue

        candidate = dict(
            by_id[doc_id]
        )

        candidate[
            "fusion_score"
        ] = float(
            fused_scores[
                doc_id
            ]
        )

        candidates.append(
            candidate
        )

    # ========================================================================
    # 7. HIERARCHY-AWARE BOOSTING
    # ========================================================================

    candidates = (
        apply_hierarchy_boost(
            query=query,
            candidates=candidates,
        )
    )

    # ========================================================================
    # 8. DEDUPLICATION
    # ========================================================================

    candidates = (
        _deduplicate_hits(
            candidates
        )
    )

    # ========================================================================
    # 9. DOCUMENT-LEVEL PARTIES FILTER
    # ========================================================================

    candidates = (
        _apply_parties_filter(
            candidates=candidates,
            parties_contains=(
                metadata_filter.get(
                    "parties_contains"
                )
            ),
        )
    )

    # ========================================================================
    # 10. PARENT / SIBLING EXPANSION
    # ========================================================================

    if expand_hierarchy:

        candidates = (
            expand_hierarchy_context(
                collection_name=(
                    collection_name
                ),
                candidates=candidates,
                max_candidates=(
                    DEFAULT_MAX_EXPANDED_CANDIDATES
                ),
            )
        )

        # Apply hierarchy scoring again because expanded candidates need
        # hierarchy scores as well.

        candidates = (
            apply_hierarchy_boost(
                query=query,
                candidates=candidates,
            )
        )

    # ========================================================================
    # 11. CROSS-ENCODER RERANKING
    # ========================================================================

    if rerank_results:

        candidates = rerank(
            query=query,
            candidates=candidates,
            top_k=final_k,
        )

    else:

        candidates = sorted(
            candidates,
            key=lambda candidate:
                candidate.get(
                    "hybrid_score",
                    0.0,
                ),
            reverse=True,
        )[:final_k]

    # ========================================================================
    # 12. RETURN FINAL RESULTS
    # ========================================================================

    return candidates


# """
# Hybrid (dense + lexical) retrieval with metadata filtering and cross-encoder
# reranking.

# This is what HybridSearchAgent calls.

# Architecture
# -------------
# Query
#   |
#   +------------------------+
#   |                        |
#   v                        v
# Dense retrieval       Legal BM25 retrieval
# Chroma ANN             in-memory lexical index
#   |                        |
#   +-----------+------------+
#               |
#               v
#        Score normalization
#               |
#               v
#        Convex / RRF fusion
#               |
#               v
#           Deduplication
#               |
#               v
#       Cross-encoder reranker
#               |
#               v
#           Top-k results


# Why hybrid instead of dense-only
# --------------------------------
# - Dense embeddings are strong at semantic/paraphrase matching.
# - BM25 is strong at exact lexical matching.
# - Legal contracts contain many repeated generic words, so the lexical
#   tokenizer intentionally removes a conservative set of generic English
#   stopwords.
# - Legal phrases such as "governing law" and "effective date" receive a
#   small phrase-level boost.
# - A cross-encoder reranker then evaluates the fused candidate set jointly.

# BM25 implementation
# -------------------
# This module implements BM25 directly rather than depending on rank_bm25.

# The scoring structure follows the standard Okapi BM25 formulation:

#     IDF(t) =
#         log(1 + (N - df(t) + 0.5) / (df(t) + 0.5))

#     score(D,Q) =
#         sum_t IDF(t)
#         * tf(t,D) * (k1 + 1)
#         / (tf(t,D) + k1 * (1 - b + b * |D| / avgdl))

# Important:
# - The implementation uses positive Robertson-style IDF.
# - Generic English stopwords are removed before indexing/querying.
# - Legal terms such as "agreement", "party", "shall", "termination",
#   etc. are intentionally retained.
# - Phrase boosting is applied after BM25 scoring.

# Known limitation
# ----------------
# The BM25 index is still in-memory and cached per Chroma collection.

# That is acceptable for a demo/small evaluation corpus but is not ideal
# for a very large production corpus because:
# - all Chroma documents are loaded into application memory;
# - rebuilding the index has startup/cold-start cost;
# - cache invalidation must happen after ingestion.

# For production scale, a native sparse+dense backend such as Qdrant,
# Milvus, or Weaviate would be preferable.

# The public hybrid_search() API intentionally remains unchanged so the
# backend can be swapped later without changing the LangGraph agent.
# """

# from __future__ import annotations

# import math
# import re
# from collections import Counter
# from dataclasses import dataclass
# from typing import Optional


# from ..resources import (
#     get_chroma_collection,
#     get_embedder,
#     get_reranker,
# )
# from ..tracing import traceable


# # ============================================================================
# # Tokenization
# # ============================================================================

# _TOKEN_RE = re.compile(r"[a-z0-9]+")


# def _tokenize_raw(text: str) -> list[str]:
#     """
#     Raw tokenizer.

#     This preserves the original project's tokenization behavior:

#         re.findall(r"[a-z0-9]+", text.lower())

#     It is useful for debugging / compatibility experiments.
#     """
#     return _TOKEN_RE.findall(text.lower())


# # ---------------------------------------------------------------------------
# # Conservative legal stopword list
# #
# # Important:
# # We intentionally DO NOT remove legal-domain words such as:
# #
# #   agreement
# #   party
# #   parties
# #   shall
# #   may
# #   termination
# #   effective
# #   date
# #   law
# #   governed
# #   liability
# #   indemnity
# #
# # Those words can be highly informative in legal retrieval.
# # ---------------------------------------------------------------------------

# LEGAL_STOPWORDS = {
#     # articles
#     "a",
#     "an",
#     "the",

#     # common verbs / auxiliaries
#     "is",
#     "are",
#     "was",
#     "were",
#     "be",
#     "been",
#     "being",
#     "has",
#     "have",
#     "had",
#     "do",
#     "does",
#     "did",
#     "can",
#     "could",
#     "would",
#     "should",
#     "will",

#     # common prepositions
#     "of",
#     "to",
#     "in",
#     "on",
#     "for",
#     "from",
#     "by",
#     "with",
#     "at",
#     "into",
#     "about",
#     "over",
#     "under",

#     # common conjunctions
#     "and",
#     "or",
#     "but",
#     "if",
#     "then",
#     "than",

#     # common demonstratives / pronouns
#     "it",
#     "its",
#     "this",
#     "that",
#     "these",
#     "those",

#     # common question words
#     "what",
#     "which",
#     "when",
#     "where",
#     "who",
#     "how",
# }


# def _tokenize(text: str) -> list[str]:
#     """
#     Production legal-aware tokenizer.

#     Generic English stopwords are removed because they occur in almost every
#     legal contract chunk and therefore provide little retrieval signal.

#     Example:

#         What is the governing law of this agreement?

#     becomes approximately:

#         ["governing", "law", "agreement"]

#     rather than:

#         ["what", "is", "the", "governing", "law", "of", "this", "agreement"]
#     """
#     tokens = _tokenize_raw(text)

#     return [
#         token
#         for token in tokens
#         if token not in LEGAL_STOPWORDS
#     ]


# # ============================================================================
# # Legal phrase handling
# # ============================================================================

# LEGAL_PHRASES = (
#     "governing law",
#     "effective date",
#     "expiration date",
#     "termination date",
#     "termination agreement",
#     "confidential information",
#     "intellectual property",
#     "limitation of liability",
#     "liability limitation",
#     "indemnification",
#     "indemnity",
#     "force majeure",
#     "change of control",
#     "notice period",
#     "notice provision",
#     "non compete",
#     "non solicitation",
#     "assignment",
#     "representations and warranties",
#     "representation and warranty",
#     "warranty",
#     "warranties",
# )


# def _extract_query_phrases(query: str) -> list[str]:
#     """
#     Extract known legal phrases appearing in the query.

#     Example:

#         What is the governing law of this agreement?

#     ->

#         ["governing law"]
#     """
#     normalized = re.sub(
#         r"\s+",
#         " ",
#         query.lower(),
#     ).strip()

#     return [
#         phrase
#         for phrase in LEGAL_PHRASES
#         if phrase in normalized
#     ]


# def _apply_phrase_boost(
#     scores: list[float],
#     texts: list[str],
#     phrases: list[str],
#     phrase_bonus: float = 2.0,
# ) -> list[float]:
#     """
#     Add a small lexical bonus when an exact legal phrase occurs in a chunk.

#     This is intentionally a modest bonus rather than a replacement for BM25.
#     BM25 remains the primary lexical scoring mechanism.
#     """
#     if not phrases:
#         return scores

#     boosted_scores = scores.copy()

#     for index, text in enumerate(texts):
#         normalized_text = re.sub(
#             r"\s+",
#             " ",
#             text.lower(),
#         )

#         for phrase in phrases:
#             if phrase in normalized_text:
#                 boosted_scores[index] += phrase_bonus

#     return boosted_scores


# # ============================================================================
# # BM25 implementation
# # ============================================================================

# @dataclass
# class LegalBM25:
#     """
#     Lightweight Okapi BM25 implementation.

#     Parameters
#     ----------
#     tokenized_corpus:
#         List of tokenized documents.

#     k1:
#         Term-frequency saturation parameter.

#     b:
#         Document-length normalization parameter.

#     phrase_bonus:
#         Stored here for configuration visibility. Actual phrase boosting is
#         applied separately by _apply_phrase_boost().
#     """

#     tokenized_corpus: list[list[str]]

#     k1: float = 1.5
#     b: float = 0.75
#     phrase_bonus: float = 2.0

#     def __post_init__(self) -> None:
#         self.N = len(self.tokenized_corpus)

#         self.doc_len: list[int] = [
#             len(tokens)
#             for tokens in self.tokenized_corpus
#         ]

#         self.avgdl = (
#             sum(self.doc_len) / self.N
#             if self.N > 0
#             else 0.0
#         )

#         # Term frequencies for each document.
#         self.doc_freqs: list[Counter[str]] = []

#         # Document frequency:
#         # number of documents containing the term.
#         self.df: Counter[str] = Counter()

#         for tokens in self.tokenized_corpus:

#             frequencies = Counter(tokens)

#             self.doc_freqs.append(frequencies)

#             # Count each term ONCE per document.
#             for term in frequencies:
#                 self.df[term] += 1

#         # Positive Robertson-style IDF.
#         #
#         # This avoids the negative-IDF behavior you observed with rank_bm25
#         # for very common words.
#         self.idf: dict[str, float] = {}

#         for term, df in self.df.items():

#             self.idf[term] = math.log(
#                 1.0
#                 + (
#                     self.N
#                     - df
#                     + 0.5
#                 )
#                 / (
#                     df
#                     + 0.5
#                 )
#             )

#     def get_scores(
#         self,
#         query_tokens: list[str],
#     ) -> list[float]:
#         """
#         Return one BM25 score per corpus document.
#         """

#         if not self.tokenized_corpus:
#             return []

#         scores = [0.0] * self.N

#         for term in query_tokens:

#             # Ignore query terms not present in corpus.
#             if term not in self.idf:
#                 continue

#             idf = self.idf[term]

#             for doc_index, frequencies in enumerate(
#                 self.doc_freqs
#             ):

#                 tf = frequencies.get(term, 0)

#                 if tf == 0:
#                     continue

#                 doc_length = self.doc_len[doc_index]

#                 if self.avgdl == 0:
#                     length_normalization = 1.0
#                 else:
#                     length_normalization = (
#                         1.0
#                         - self.b
#                         + self.b
#                         * doc_length
#                         / self.avgdl
#                     )

#                 denominator = (
#                     tf
#                     + self.k1
#                     * length_normalization
#                 )

#                 contribution = (
#                     idf
#                     * tf
#                     * (self.k1 + 1.0)
#                     / denominator
#                 )

#                 scores[doc_index] += contribution

#         return scores


# # ============================================================================
# # BM25 index cache
# # ============================================================================

# _bm25_cache: dict[
#     str,
#     tuple[
#         LegalBM25,
#         list[str],
#         list[str],
#         list[dict],
#     ],
# ] = {}


# def _get_bm25_index(
#     collection_name: str,
#     refresh: bool = False,
# ):
#     """
#     Build or retrieve the cached BM25 index for a Chroma collection.
#     """

#     if (
#         not refresh
#         and collection_name in _bm25_cache
#     ):
#         return _bm25_cache[collection_name]

#     collection = get_chroma_collection(
#         collection_name
#     )

#     raw = collection.get(
#         include=[
#             "documents",
#             "metadatas",
#         ]
#     )

#     ids = raw["ids"]
#     texts = raw["documents"]
#     metadatas = raw["metadatas"]

#     tokenized_documents = [
#         _tokenize(text)
#         for text in texts
#     ]

#     bm25 = LegalBM25(
#         tokenized_corpus=tokenized_documents,
#         k1=1.5,
#         b=0.75,
#         phrase_bonus=2.0,
#     )

#     _bm25_cache[collection_name] = (
#         bm25,
#         ids,
#         texts,
#         metadatas,
#     )

#     return _bm25_cache[collection_name]


# def invalidate_bm25_cache(
#     collection_name: str,
# ) -> None:
#     """
#     Invalidate the BM25 cache after new documents are ingested.

#     Call this immediately after embed_and_store().

#     Example:

#         embed_and_store(...)
#         invalidate_bm25_cache(collection_name)
#     """

#     _bm25_cache.pop(
#         collection_name,
#         None,
#     )


# # ============================================================================
# # Metadata filtering
# # ============================================================================

# def build_where_clause(
#     filters: dict,
# ) -> Optional[dict]:
#     """
#     Convert application-level metadata filters into Chroma's `where` clause.
#     """

#     clauses: list[dict] = []

#     if filters.get("contract_type"):
#         clauses.append(
#             {
#                 "contract_type":
#                     filters["contract_type"]
#             }
#         )

#     if filters.get(
#         "governing_law_country"
#     ):
#         clauses.append(
#             {
#                 "governing_law_country":
#                     filters[
#                         "governing_law_country"
#                     ].upper()
#             }
#         )

#     if (
#         filters.get(
#             "min_effective_date_epoch"
#         )
#         is not None
#     ):
#         clauses.append(
#             {
#                 "effective_date_epoch": {
#                     "$gte":
#                         filters[
#                             "min_effective_date_epoch"
#                         ]
#                 }
#             }
#         )

#     if (
#         filters.get(
#             "max_effective_date_epoch"
#         )
#         is not None
#     ):
#         clauses.append(
#             {
#                 "effective_date_epoch": {
#                     "$lte":
#                         filters[
#                             "max_effective_date_epoch"
#                         ]
#                 }
#             }
#         )

#     if (
#         filters.get(
#             "min_monetary_value"
#         )
#         is not None
#     ):
#         clauses.append(
#             {
#                 "monetary_value": {
#                     "$gte":
#                         filters[
#                             "min_monetary_value"
#                         ]
#                 }
#             }
#         )

#     if (
#         filters.get(
#             "max_monetary_value"
#         )
#         is not None
#     ):
#         clauses.append(
#             {
#                 "monetary_value": {
#                     "$lte":
#                         filters[
#                             "max_monetary_value"
#                         ]
#                 }
#             }
#         )

#     if not clauses:
#         return None

#     if len(clauses) == 1:
#         return clauses[0]

#     return {
#         "$and": clauses
#     }


# def _matches_where(
#     meta: dict,
#     where: Optional[dict],
# ) -> bool:
#     """
#     Python-side equivalent of build_where_clause().

#     Chroma applies the metadata filter natively for dense retrieval.
#     BM25 is an in-memory index, so the same filtering must be applied
#     here in Python.
#     """

#     if not where:
#         return True

#     if "$and" in where:
#         return all(
#             _matches_where(
#                 meta,
#                 clause,
#             )
#             for clause in where["$and"]
#         )

#     (field, condition), = where.items()

#     value = meta.get(field)

#     if isinstance(condition, dict):

#         if (
#             "$gte" in condition
#             and not (
#                 value is not None
#                 and value >= condition["$gte"]
#             )
#         ):
#             return False

#         if (
#             "$lte" in condition
#             and not (
#                 value is not None
#                 and value <= condition["$lte"]
#             )
#         ):
#             return False

#         return True

#     return value == condition


# # ============================================================================
# # Dense retrieval
# # ============================================================================

# def _dense_search(
#     collection_name: str,
#     query: str,
#     top_k: int,
#     where: Optional[dict],
# ) -> list[dict]:
#     """
#     Dense semantic retrieval using Chroma.

#     Chroma applies metadata filtering during ANN retrieval.
#     """

#     collection = get_chroma_collection(
#         collection_name
#     )

#     embedder = get_embedder()

#     query_embedding = embedder.encode(
#         [query],
#         normalize_embeddings=True,
#     ).tolist()

#     results = collection.query(
#         query_embeddings=query_embedding,
#         n_results=top_k,
#         where=where,
#     )

#     hits: list[dict] = []

#     if results["ids"]:

#         for (
#             doc_id,
#             doc,
#             meta,
#             dist,
#         ) in zip(
#             results["ids"][0],
#             results["documents"][0],
#             results["metadatas"][0],
#             results["distances"][0],
#         ):

#             hits.append(
#                 {
#                     "id": doc_id,
#                     "text": doc,
#                     "metadata": meta,
#                     "dense_distance": dist,
#                 }
#             )

#     return hits


# # ============================================================================
# # Sparse / BM25 retrieval
# # ============================================================================

# def _sparse_search(
#     collection_name: str,
#     query: str,
#     top_k: int,
#     where: Optional[dict],
# ) -> list[dict]:
#     """
#     Legal-aware BM25 retrieval.

#     Pipeline:

#         query
#           |
#           v
#         legal tokenization
#           |
#           v
#         BM25 scoring
#           |
#           v
#         legal phrase boosting
#           |
#           v
#         metadata filtering
#           |
#           v
#         top-k
#     """

#     (
#         bm25,
#         ids,
#         texts,
#         metadatas,
#     ) = _get_bm25_index(
#         collection_name
#     )

#     # ------------------------------------------------------------------
#     # Query tokenization
#     # ------------------------------------------------------------------

#     query_tokens = _tokenize(query)

#     # ------------------------------------------------------------------
#     # BM25 scoring
#     # ------------------------------------------------------------------

#     scores = bm25.get_scores(
#         query_tokens
#     )

#     # ------------------------------------------------------------------
#     # Legal phrase boosting
#     # ------------------------------------------------------------------

#     query_phrases = _extract_query_phrases(
#         query
#     )

#     scores = _apply_phrase_boost(
#         scores=scores,
#         texts=texts,
#         phrases=query_phrases,
#         phrase_bonus=bm25.phrase_bonus,
#     )

#     # ------------------------------------------------------------------
#     # Metadata filtering BEFORE top-k selection
#     # ------------------------------------------------------------------

#     eligible = [
#         index
#         for index in range(len(ids))
#         if _matches_where(
#             metadatas[index],
#             where,
#         )
#     ]

#     eligible_ranked = sorted(
#         eligible,
#         key=lambda index: scores[index],
#         reverse=True,
#     )[:top_k]

#     return [
#         {
#             "id": ids[index],
#             "text": texts[index],
#             "metadata": metadatas[index],
#             "bm25_score": float(
#                 scores[index]
#             ),
#         }
#         for index in eligible_ranked
#     ]


# # ============================================================================
# # Fusion strategies
# # ============================================================================

# def reciprocal_rank_fusion(
#     id_lists: list[list[str]],
#     k: int = 60,
# ) -> dict[str, float]:
#     """
#     Reciprocal Rank Fusion.

#     score(d) =
#         sum(
#             1 / (k + rank)
#         )

#     RRF uses only ranking and does not require dense and sparse raw scores
#     to be numerically comparable.
#     """

#     scores: dict[str, float] = {}

#     for id_list in id_lists:

#         for rank, doc_id in enumerate(
#             id_list
#         ):

#             scores[doc_id] = (
#                 scores.get(
#                     doc_id,
#                     0.0,
#                 )
#                 + 1.0
#                 / (
#                     k
#                     + rank
#                     + 1
#                 )
#             )

#     return scores


# def min_max_normalize(
#     scores: dict[str, float],
#     reverse: bool = False,
# ) -> dict[str, float]:
#     """
#     Normalize scores independently into [0, 1].

#     reverse=True means lower raw values are better, which is appropriate
#     for Chroma distance scores.
#     """

#     if not scores:
#         return {}

#     values = list(
#         scores.values()
#     )

#     min_val = min(values)
#     max_val = max(values)

#     if min_val == max_val:
#         return {
#             doc_id: 1.0
#             for doc_id in scores
#         }

#     normalized: dict[str, float] = {}

#     for doc_id, score in scores.items():

#         if reverse:
#             norm_score = (
#                 max_val - score
#             ) / (
#                 max_val - min_val
#             )
#         else:
#             norm_score = (
#                 score - min_val
#             ) / (
#                 max_val - min_val
#             )

#         normalized[doc_id] = float(
#             norm_score
#         )

#     return normalized


# def convex_combination_fusion(
#     dense_hits: list[dict],
#     sparse_hits: list[dict],
#     alpha: float = 0.4,
#     dense_score_key: str = "dense_distance",
#     sparse_score_key: str = "bm25_score",
#     dense_is_distance: bool = True,
# ) -> dict[str, float]:
#     """
#     Combine dense and sparse retrieval scores.

#     Final Score =
#         alpha * Dense_Norm
#         +
#         (1 - alpha) * Sparse_Norm

#     alpha = 0.2 means:

#         20% dense
#         80% BM25

#     alpha = 0.4 means:

#         40% dense
#         60% BM25

#     alpha = 0.7 means:

#         70% dense
#         30% BM25
#     """

#     if not dense_hits and not sparse_hits:
#         return {}

#     # ------------------------------------------------------------------
#     # Raw score mappings
#     # ------------------------------------------------------------------

#     raw_dense = {
#         hit["id"]:
#             hit[dense_score_key]
#         for hit in dense_hits
#     }

#     raw_sparse = {
#         hit["id"]:
#             hit[sparse_score_key]
#         for hit in sparse_hits
#     }

#     # ------------------------------------------------------------------
#     # Independent normalization
#     # ------------------------------------------------------------------

#     norm_dense = min_max_normalize(
#         raw_dense,
#         reverse=dense_is_distance,
#     )

#     norm_sparse = min_max_normalize(
#         raw_sparse,
#         reverse=False,
#     )

#     # ------------------------------------------------------------------
#     # Union of candidate IDs
#     # ------------------------------------------------------------------

#     all_doc_ids = (
#         set(norm_dense.keys())
#         | set(norm_sparse.keys())
#     )

#     fused_scores: dict[str, float] = {}

#     for doc_id in all_doc_ids:

#         dense_score = norm_dense.get(
#             doc_id,
#             0.0,
#         )

#         sparse_score = norm_sparse.get(
#             doc_id,
#             0.0,
#         )

#         fused_scores[doc_id] = (
#             alpha * dense_score
#             + (1.0 - alpha)
#             * sparse_score
#         )

#     # ------------------------------------------------------------------
#     # Sort descending
#     # ------------------------------------------------------------------

#     return dict(
#         sorted(
#             fused_scores.items(),
#             key=lambda item: item[1],
#             reverse=True,
#         )
#     )


# # ============================================================================
# # Cross-encoder reranking
# # ============================================================================

# def rerank(
#     query: str,
#     candidates: list[dict],
#     top_k: int,
# ) -> list[dict]:
#     """
#     Cross-encoder reranking.

#     The reranker is loaded through resources.get_reranker() so that the
#     application continues to use its existing singleton model.
#     """

#     if not candidates:
#         return []

#     reranker = get_reranker()

#     pairs = [
#         (
#             query,
#             candidate["text"],
#         )
#         for candidate in candidates
#     ]

#     scores = reranker.predict(
#         pairs
#     )

#     for candidate, score in zip(
#         candidates,
#         scores,
#     ):
#         candidate[
#             "rerank_score"
#         ] = float(score)

#     return sorted(
#         candidates,
#         key=lambda candidate:
#             candidate["rerank_score"],
#         reverse=True,
#     )[:top_k]


# # ============================================================================
# # Deduplication
# # ============================================================================

# def _deduplicate_hits(
#     hits: list[dict],
# ) -> list[dict]:
#     """
#     Remove duplicate underlying chunks.

#     We intentionally deduplicate using:

#         (
#             document_name,
#             chunk_index,
#             text
#         )

#     rather than Chroma's UUID.

#     This catches the case where the same chunk was ingested more than once
#     under different Chroma IDs.

#     The first occurrence wins because candidates arrive already ordered by
#     fused relevance.
#     """

#     unique_hits: list[dict] = []

#     seen: set[
#         tuple[
#             object,
#             object,
#             str,
#         ]
#     ] = set()

#     for hit in hits:

#         metadata = hit.get(
#             "metadata",
#             {},
#         )

#         key = (
#             metadata.get(
#                 "document_name"
#             ),
#             metadata.get(
#                 "chunk_index"
#             ),
#             (
#                 hit.get(
#                     "text",
#                     ""
#                 )
#                 or ""
#             ).strip(),
#         )

#         if key in seen:
#             continue

#         seen.add(key)
#         unique_hits.append(hit)

#     return unique_hits


# # ============================================================================
# # Public entrypoint
# # ============================================================================

# @traceable(
#     name="hybrid_search_agent.search",
#     run_type="retriever",
# )
# def hybrid_search(
#     collection_name: str,
#     query: str,
#     metadata_filter: Optional[dict] = None,
#     dense_k: int = 20,
#     sparse_k: int = 20,
#     fusion_k: int = 15,
#     final_k: int = 6,
#     fusion_method: str = "convex",
#     alpha: float = 0.4,
# ) -> list[dict]:
#     """
#     Full retrieval pipeline.

#         Dense search
#               +
#         Legal BM25 search
#               |
#               v
#         Dense/BM25 fusion
#               |
#               v
#         Candidate selection
#               |
#               v
#         Deduplication
#               |
#               v
#         parties_contains filtering
#               |
#               v
#         Cross-encoder reranking
#               |
#               v
#         final_k results

#     Parameters
#     ----------
#     collection_name:
#         Chroma collection to search.

#     query:
#         User question.

#     metadata_filter:
#         Optional contract-level metadata filters.

#     dense_k:
#         Number of dense candidates.

#     sparse_k:
#         Number of BM25 candidates.

#     fusion_k:
#         Number of fused candidates passed forward.

#     final_k:
#         Number of reranked results returned.

#     fusion_method:
#         "convex" or "rrf".

#     alpha:
#         Dense weight for convex fusion.

#         0.2 = lexical-heavy
#         0.4 = balanced/legal lexical-heavy
#         0.7 = semantic-heavy

#     Returns
#     -------
#     list[dict]

#     Each result contains fields such as:

#         id
#         text
#         metadata
#         dense_distance
#         bm25_score
#         rerank_score
#     """

#     metadata_filter = (
#         metadata_filter
#         or {}
#     )

#     # ------------------------------------------------------------------
#     # Convert application filters into Chroma/Python predicates
#     # ------------------------------------------------------------------

#     where = build_where_clause(
#         metadata_filter
#     )

#     # ------------------------------------------------------------------
#     # Dense retrieval
#     # ------------------------------------------------------------------

#     dense_hits = _dense_search(
#         collection_name=collection_name,
#         query=query,
#         top_k=dense_k,
#         where=where,
#     )

#     # ------------------------------------------------------------------
#     # Sparse / BM25 retrieval
#     # ------------------------------------------------------------------

#     sparse_hits = _sparse_search(
#         collection_name=collection_name,
#         query=query,
#         top_k=sparse_k,
#         where=where,
#     )

#     # ------------------------------------------------------------------
#     # Merge dense + sparse hits by Chroma ID
#     # ------------------------------------------------------------------

#     by_id: dict[
#         str,
#         dict,
#     ] = {}

#     for hit in (
#         dense_hits
#         + sparse_hits
#     ):

#         if hit["id"] not in by_id:
#             by_id[
#                 hit["id"]
#             ] = {}

#         by_id[
#             hit["id"]
#         ].update(hit)

#     # ------------------------------------------------------------------
#     # Fusion
#     # ------------------------------------------------------------------

#     if fusion_method.lower() == "rrf":

#         fused_scores = (
#             reciprocal_rank_fusion(
#                 [
#                     [
#                         hit["id"]
#                         for hit in dense_hits
#                     ],
#                     [
#                         hit["id"]
#                         for hit in sparse_hits
#                     ],
#                 ]
#             )
#         )

#     else:

#         fused_scores = (
#             convex_combination_fusion(
#                 dense_hits=dense_hits,
#                 sparse_hits=sparse_hits,
#                 alpha=alpha,
#                 dense_is_distance=True,
#             )
#         )

#     # ------------------------------------------------------------------
#     # Keep only top fusion_k candidates
#     # ------------------------------------------------------------------

#     fused_ranked_ids = list(
#         fused_scores.keys()
#     )[:fusion_k]

#     candidates = [
#         by_id[doc_id]
#         for doc_id in fused_ranked_ids
#         if doc_id in by_id
#     ]

#     # ------------------------------------------------------------------
#     # Deduplicate underlying chunks
#     # ------------------------------------------------------------------

#     candidates = _deduplicate_hits(
#         candidates
#     )

#     # ------------------------------------------------------------------
#     # parties_contains is intentionally handled here because it is not
#     # represented by build_where_clause().
#     # ------------------------------------------------------------------

#     parties_needle = (
#         metadata_filter.get(
#             "parties_contains"
#         )
#     )

#     if parties_needle:

#         needle = (
#             parties_needle
#             .lower()
#             .strip()
#         )

#         candidates = [
#             candidate
#             for candidate in candidates
#             if needle
#             in (
#                 candidate[
#                     "metadata"
#                 ].get(
#                     "parties",
#                     "",
#                 )
#                 or ""
#             ).lower()
#         ]

#     # ------------------------------------------------------------------
#     # Cross-encoder reranking
#     # ------------------------------------------------------------------

#     return rerank(
#         query=query,
#         candidates=candidates,
#         top_k=final_k,
#     )
# # #==================
# # """
# # Hybrid (dense + lexical) retrieval, with metadata filtering and cross-encoder
# # reranking. This is what HybridSearchAgent calls.

# # Why hybrid instead of dense-only:
# #   - Dense embeddings are strong at semantic/paraphrase matching ("termination
# #     for convenience" ~ "either party may end this agreement without cause")
# #     but weak at exact string matching — a specific clause number, a defined
# #     term used verbatim, a citation.
# #   - BM25 (lexical, term-frequency based) is the opposite: strong at exact
# #     term matching, weak at paraphrase.
# #   - The two branches' raw scores are fused with a weighted convex
# #     combination after independently min-max normalizing each to [0, 1] —
# #     see convex_combination_fusion() below for why, over rank-only fusion
# #     (RRF, kept here as an alternate strategy).
# #   - A cross-encoder reranker then scores each (query, candidate) pair
# #     *jointly* — much more accurate than comparing two independently-encoded
# #     vectors — but it's too slow to run against an entire collection, so it
# #     only reranks the ~15-20 candidates that already survived the cheap
# #     dense+sparse fusion stage.

# # KNOWN LIMITATION — in-memory BM25 (see _get_bm25_index below):
# #   This module builds and caches a BM25 index by pulling every document out
# #   of the Chroma collection into memory. That's fine for a demo/small corpus,
# #   but it means high memory use and cold-start latency on large collections,
# #   and the index only refreshes when explicitly invalidated (see
# #   invalidate_bm25_cache), not automatically on ingestion. The correct
# #   production fix is to move to a vector database with NATIVE sparse+dense
# #   hybrid indexing and native filtered lexical search — e.g. Qdrant, Milvus,
# #   or Weaviate — which eliminates client-side index building entirely and
# #   lets the database engine apply metadata filters during scoring rather than
# #   after it. That's a backend swap, not a change to this module's public
# #   hybrid_search() signature: _dense_search/_sparse_search/_get_bm25_index
# #   are the only functions that would need to be replaced with native
# #   filtered-hybrid-query calls against the new store.
# # """

# # from __future__ import annotations

# # import re
# # from typing import Optional

# # from rank_bm25 import BM25Okapi

# # from ..resources import get_chroma_collection, get_embedder, get_reranker
# # from ..tracing import traceable

# # # ---------------------------------------------------------------------------
# # # BM25 index (in-memory, cached per collection)
# # # ---------------------------------------------------------------------------

# # _bm25_cache: dict[str, tuple[BM25Okapi, list[str], list[str], list[dict]]] = {}


# # def _tokenize(text: str) -> list[str]:
# #     return re.findall(r"[a-z0-9]+", text.lower())


# # def _get_bm25_index(collection_name: str, refresh: bool = False):
# #     if not refresh and collection_name in _bm25_cache:
# #         return _bm25_cache[collection_name]

# #     collection = get_chroma_collection(collection_name)
# #     raw = collection.get(include=["documents", "metadatas"])
# #     ids, texts, metadatas = raw["ids"], raw["documents"], raw["metadatas"]
# #     bm25 = BM25Okapi([_tokenize(t) for t in texts])

# #     _bm25_cache[collection_name] = (bm25, ids, texts, metadatas)
# #     return _bm25_cache[collection_name]


# # def invalidate_bm25_cache(collection_name: str) -> None:
# #     """
# #     Call this right after ingesting new documents into `collection_name`
# #     (e.g. from scripts/run_demo.py, right after embed_and_store()) so the
# #     next hybrid_search() call rebuilds the BM25 index instead of silently
# #     searching a stale one that doesn't include the documents you just added.
# #     Not wired in automatically from pdf_pipeline.embed_and_store() itself,
# #     to avoid a circular import between ingestion and retrieval — call it
# #     explicitly from whichever orchestration layer just finished ingesting.
# #     """
# #     _bm25_cache.pop(collection_name, None)


# # # ---------------------------------------------------------------------------
# # # Metadata filter -> Chroma `where` clause, and a matching Python-side
# # # predicate for the BM25 branch (which Chroma's `where` can't reach).
# # # ---------------------------------------------------------------------------

# # def build_where_clause(filters: dict) -> Optional[dict]:
# #     clauses: list[dict] = []
# #     if filters.get("contract_type"):
# #         clauses.append({"contract_type": filters["contract_type"]})
# #     if filters.get("governing_law_country"):
# #         clauses.append({"governing_law_country": filters["governing_law_country"].upper()})
# #     if filters.get("min_effective_date_epoch") is not None:
# #         clauses.append({"effective_date_epoch": {"$gte": filters["min_effective_date_epoch"]}})
# #     if filters.get("max_effective_date_epoch") is not None:
# #         clauses.append({"effective_date_epoch": {"$lte": filters["max_effective_date_epoch"]}})
# #     if filters.get("min_monetary_value") is not None:
# #         clauses.append({"monetary_value": {"$gte": filters["min_monetary_value"]}})
# #     if filters.get("max_monetary_value") is not None:
# #         clauses.append({"monetary_value": {"$lte": filters["max_monetary_value"]}})

# #     if not clauses:
# #         return None
# #     return clauses[0] if len(clauses) == 1 else {"$and": clauses}


# # def _matches_where(meta: dict, where: Optional[dict]) -> bool:
# #     """Python-side equivalent of build_where_clause(), for filtering BM25 candidates."""
# #     if not where:
# #         return True
# #     if "$and" in where:
# #         return all(_matches_where(meta, clause) for clause in where["$and"])
# #     (field, condition), = where.items()
# #     value = meta.get(field)
# #     if isinstance(condition, dict):
# #         if "$gte" in condition and not (value is not None and value >= condition["$gte"]):
# #             return False
# #         if "$lte" in condition and not (value is not None and value <= condition["$lte"]):
# #             return False
# #         return True
# #     return value == condition


# # # ---------------------------------------------------------------------------
# # # Dense and sparse branches
# # # ---------------------------------------------------------------------------

# # def _dense_search(collection_name: str, query: str, top_k: int, where: Optional[dict]) -> list[dict]:
# #     """Chroma applies `where` natively DURING the ANN search — filtering here was never the issue."""
# #     collection = get_chroma_collection(collection_name)
# #     embedder = get_embedder()
# #     query_embedding = embedder.encode([query], normalize_embeddings=True).tolist()
# #     results = collection.query(query_embeddings=query_embedding, n_results=top_k, where=where)

# #     hits = []
# #     if results["ids"]:
# #         for doc_id, doc, meta, dist in zip(
# #             results["ids"][0], results["documents"][0], results["metadatas"][0], results["distances"][0]
# #         ):
# #             hits.append({"id": doc_id, "text": doc, "metadata": meta, "dense_distance": dist})
# #     return hits


# # def _sparse_search(collection_name: str, query: str, top_k: int, where: Optional[dict]) -> list[dict]:
# #     """
# #     Filter FIRST (restrict to the set of documents that satisfy `where`),
# #     THEN rank that eligible set by BM25 score — not the other way around.
# #     Ranking the whole corpus first and filtering afterward risks returning
# #     fewer than top_k results (or zero) under a restrictive filter even when
# #     plenty of matching-and-relevant documents exist further down the ranked
# #     list; filtering the candidate pool before selecting top_k guarantees we
# #     never miss an eligible match because it happened to rank outside some
# #     arbitrary pre-filter cutoff.
# #     """
# #     bm25, ids, texts, metadatas = _get_bm25_index(collection_name)
# #     scores = bm25.get_scores(_tokenize(query))

# #     eligible = [i for i in range(len(ids)) if _matches_where(metadatas[i], where)]
# #     eligible_ranked = sorted(eligible, key=lambda i: scores[i], reverse=True)[:top_k]

# #     return [
# #         {"id": ids[i], "text": texts[i], "metadata": metadatas[i], "bm25_score": float(scores[i])}
# #         for i in eligible_ranked
# #     ]


# # # ---------------------------------------------------------------------------
# # # Fusion strategies
# # # ---------------------------------------------------------------------------

# # def reciprocal_rank_fusion(id_lists: list[list[str]], k: int = 60) -> dict[str, float]:
# #     """
# #     Rank-only fusion: score(d) = sum, over lists containing d, of
# #     1 / (k + rank_in_that_list). Doesn't need the two branches' raw scores
# #     to be comparable at all — only their relative ordering — which makes it
# #     a safe default when you don't want to tune a blend weight. Kept as an
# #     alternate to convex_combination_fusion (see hybrid_search()'s
# #     fusion_method parameter).
# #     """
# #     scores: dict[str, float] = {}
# #     for id_list in id_lists:
# #         for rank, doc_id in enumerate(id_list):
# #             scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
# #     return scores


# # def min_max_normalize(scores: dict[str, float], reverse: bool = False) -> dict[str, float]:
# #     """Normalizes scores to the [0, 1] range using Min-Max scaling.

# #     Args:
# #         scores: Dictionary mapping document IDs to raw scores.
# #         reverse: Set to True if lower raw score is better (e.g. L2 or Cosine
# #           Distance).

# #     Returns:
# #         Dictionary mapping document IDs to normalized scores in [0, 1].
# #     """
# #     if not scores:
# #         return {}
# #     vals = list(scores.values())
# #     min_val, max_val = min(vals), max(vals)
# #     # Edge case: all candidates have the identical raw score.
# #     if min_val == max_val:
# #         return {doc_id: 1.0 for doc_id in scores}
# #     normalized = {}
# #     for doc_id, score in scores.items():
# #         if reverse:
# #             # For distance metrics where lower is better.
# #             norm_score = (max_val - score) / (max_val - min_val)
# #         else:
# #             # For similarity/BM25 metrics where higher is better.
# #             norm_score = (score - min_val) / (max_val - min_val)
# #         normalized[doc_id] = float(norm_score)
# #     return normalized


# # def convex_combination_fusion(
# #     dense_hits: list[dict],
# #     sparse_hits: list[dict],
# #     alpha: float = 0.4,
# #     dense_score_key: str = "dense_distance",
# #     sparse_score_key: str = "bm25_score",
# #     dense_is_distance: bool = True,
# # ) -> dict[str, float]:
# #     """Combines dense and sparse results using a weighted convex combination.

# #     Final Score = alpha * Dense_Norm + (1 - alpha) * Sparse_Norm

# #     Default alpha=0.4 weights lexical (BM25) matching more heavily than
# #     dense similarity (40% dense / 60% sparse) — legal text leans on exact
# #     clause wording, defined terms, and citations, where lexical match is
# #     often the stronger signal; raise alpha toward 1.0 if your queries skew
# #     more paraphrase/semantic than exact-term.

# #     Args:
# #         dense_hits: List of dicts, e.g. [{"id": "doc1", "dense_distance": 0.12}]
# #         sparse_hits: List of dicts, e.g. [{"id": "doc1", "bm25_score": 14.2}]
# #         alpha: Weight for dense retrieval (0.0 to 1.0). High alpha = favor
# #           dense.
# #         dense_score_key: Dictionary key containing the dense metric.
# #         sparse_score_key: Dictionary key containing the sparse metric.
# #         dense_is_distance: Set True if dense score is distance (lower =
# #           better).

# #     Returns:
# #         Dictionary mapping doc IDs to their combined score, sorted descending.
# #     """
# #     # 1. Extract raw scores into mappings.
# #     raw_dense = {h["id"]: h[dense_score_key] for h in dense_hits}
# #     raw_sparse = {h["id"]: h[sparse_score_key] for h in sparse_hits}

# #     # 2. Normalize both score distributions independently to [0, 1].
# #     norm_dense = min_max_normalize(raw_dense, reverse=dense_is_distance)
# #     norm_sparse = min_max_normalize(raw_sparse, reverse=False)

# #     # 3. Combine scores across the union of all candidate document IDs.
# #     all_doc_ids = set(norm_dense.keys()).union(set(norm_sparse.keys()))
# #     fused_scores = {}
# #     for doc_id in all_doc_ids:
# #         # If a document is missing in one branch, treat its normalized score as 0.0.
# #         d_score = norm_dense.get(doc_id, 0.0)
# #         s_score = norm_sparse.get(doc_id, 0.0)
# #         fused_scores[doc_id] = (alpha * d_score) + ((1.0 - alpha) * s_score)

# #     # Sort candidates by fused score descending.
# #     return dict(sorted(fused_scores.items(), key=lambda item: item[1], reverse=True))


# # def rerank(query: str, candidates: list[dict], top_k: int) -> list[dict]:
# #     """
# #     Cross-encoder rerank of the fused candidate set. The reranker itself is
# #     a module-level singleton loaded via resources.get_reranker() — NOT
# #     instantiated here — see resources.preload_models() to load it eagerly
# #     at application startup instead of lazily on the first call.
# #     """
# #     if not candidates:
# #         return []
# #     reranker = get_reranker()
# #     pairs = [(query, c["text"]) for c in candidates]
# #     scores = reranker.predict(pairs)
# #     for c, s in zip(candidates, scores):
# #         c["rerank_score"] = float(s)
# #     return sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)[:top_k]


# # # ---------------------------------------------------------------------------
# # # Public entrypoint
# # # ---------------------------------------------------------------------------

# # def _deduplicate_hits(hits: list[dict]) -> list[dict]:
# #     """
# #     Collapses duplicate chunks that made it into the candidate set as
# #     separate entries. This happens when the same underlying text exists in
# #     Chroma under more than one id (e.g. a document re-ingested without
# #     first removing the prior copy) — the by-id merge in hybrid_search()
# #     can't catch this, since it dedupes on Chroma id, and these are
# #     genuinely different ids that happen to hold identical content.
# #     Deduplicating by (document_name, chunk_index, text) instead catches it.
# #     Order is preserved, so the first (highest fused-score) occurrence of
# #     each duplicate is what's kept.
# #     """
# #     unique_hits = []
# #     seen = set()
# #     for hit in hits:
# #         key = (
# #             hit["metadata"].get("document_name"),
# #             hit["metadata"].get("chunk_index"),
# #             hit["text"].strip(),
# #         )
# #         if key not in seen:
# #             seen.add(key)
# #             unique_hits.append(hit)
# #     return unique_hits


# # @traceable(name="hybrid_search_agent.search", run_type="retriever")
# # def hybrid_search(
# #     collection_name: str,
# #     query: str,
# #     metadata_filter: Optional[dict] = None,
# #     dense_k: int = 20,
# #     sparse_k: int = 20,
# #     fusion_k: int = 15,
# #     final_k: int = 6,
# #     fusion_method: str = "convex",
# #     alpha: float = 0.4,
# # ) -> list[dict]:
# #     """
# #     Full pipeline: dense search + BM25 search (each metadata-filtered) ->
# #     fusion ("convex" weighted combination by default, or "rrf" for
# #     rank-only fusion) -> parties_contains post-filter -> cross-encoder
# #     rerank -> top final_k results, each:
# #     {"id", "text", "metadata", "rerank_score", ...}.
# #     """
# #     metadata_filter = metadata_filter or {}
# #     where = build_where_clause(metadata_filter)

# #     dense_hits = _dense_search(collection_name, query, dense_k, where)
# #     sparse_hits = _sparse_search(collection_name, query, sparse_k, where)

# #     by_id: dict[str, dict] = {}
# #     for h in dense_hits + sparse_hits:
# #         by_id.setdefault(h["id"], {}).update(h)

# #     if fusion_method == "rrf":
# #         fused_scores = reciprocal_rank_fusion(
# #             [[h["id"] for h in dense_hits], [h["id"] for h in sparse_hits]]
# #         )
# #     else:
# #         fused_scores = convex_combination_fusion(
# #             dense_hits=dense_hits,
# #             sparse_hits=sparse_hits,
# #             alpha=alpha,
# #             dense_is_distance=True,  # Chroma returns distances where lower = better
# #         )

# #     fused_ranked_ids = list(fused_scores.keys())[:fusion_k]
# #     candidates = [by_id[i] for i in fused_ranked_ids if i in by_id]
# #     candidates = _deduplicate_hits(candidates)

# #     parties_needle = metadata_filter.get("parties_contains")
# #     if parties_needle:
# #         needle = parties_needle.lower()
# #         candidates = [c for c in candidates if needle in (c["metadata"].get("parties", "") or "").lower()]

# #     return rerank(query, candidates, top_k=final_k)

# # #==============
# # # """
# # # Hybrid (dense + lexical) retrieval, with metadata filtering and cross-encoder
# # # reranking. This is what HybridSearchAgent calls.

# # # Why hybrid instead of dense-only:
# # #   - Dense embeddings are strong at semantic/paraphrase matching ("termination
# # #     for convenience" ~ "either party may end this agreement without cause")
# # #     but weak at exact string matching — a specific clause number, a defined
# # #     term used verbatim, a citation.
# # #   - BM25 (lexical, term-frequency based) is the opposite: strong at exact
# # #     term matching, weak at paraphrase.
# # #   - The two branches' raw scores are fused with a weighted convex
# # #     combination after independently min-max normalizing each to [0, 1] —
# # #     see convex_combination_fusion() below for why, over rank-only fusion
# # #     (RRF, kept here as an alternate strategy).
# # #   - A cross-encoder reranker then scores each (query, candidate) pair
# # #     *jointly* — much more accurate than comparing two independently-encoded
# # #     vectors — but it's too slow to run against an entire collection, so it
# # #     only reranks the ~15-20 candidates that already survived the cheap
# # #     dense+sparse fusion stage.

# # # KNOWN LIMITATION — in-memory BM25 (see _get_bm25_index below):
# # #   This module builds and caches a BM25 index by pulling every document out
# # #   of the Chroma collection into memory. That's fine for a demo/small corpus,
# # #   but it means high memory use and cold-start latency on large collections,
# # #   and the index only refreshes when explicitly invalidated (see
# # #   invalidate_bm25_cache), not automatically on ingestion. The correct
# # #   production fix is to move to a vector database with NATIVE sparse+dense
# # #   hybrid indexing and native filtered lexical search — e.g. Qdrant, Milvus,
# # #   or Weaviate — which eliminates client-side index building entirely and
# # #   lets the database engine apply metadata filters during scoring rather than
# # #   after it. That's a backend swap, not a change to this module's public
# # #   hybrid_search() signature: _dense_search/_sparse_search/_get_bm25_index
# # #   are the only functions that would need to be replaced with native
# # #   filtered-hybrid-query calls against the new store.
# # # """

# # # from __future__ import annotations

# # # import re
# # # from typing import Optional

# # # from rank_bm25 import BM25Okapi

# # # from ..resources import get_chroma_collection, get_embedder, get_reranker
# # # from ..tracing import traceable

# # # # ---------------------------------------------------------------------------
# # # # BM25 index (in-memory, cached per collection)
# # # # ---------------------------------------------------------------------------

# # # _bm25_cache: dict[str, tuple[BM25Okapi, list[str], list[str], list[dict]]] = {}


# # # def _tokenize(text: str) -> list[str]:
# # #     return re.findall(r"[a-z0-9]+", text.lower())


# # # def _get_bm25_index(collection_name: str, refresh: bool = False):
# # #     if not refresh and collection_name in _bm25_cache:
# # #         return _bm25_cache[collection_name]

# # #     collection = get_chroma_collection(collection_name)
# # #     raw = collection.get(include=["documents", "metadatas"])
# # #     ids, texts, metadatas = raw["ids"], raw["documents"], raw["metadatas"]
# # #     bm25 = BM25Okapi([_tokenize(t) for t in texts])

# # #     _bm25_cache[collection_name] = (bm25, ids, texts, metadatas)
# # #     return _bm25_cache[collection_name]


# # # def invalidate_bm25_cache(collection_name: str) -> None:
# # #     """
# # #     Call this right after ingesting new documents into `collection_name`
# # #     (e.g. from scripts/run_demo.py, right after embed_and_store()) so the
# # #     next hybrid_search() call rebuilds the BM25 index instead of silently
# # #     searching a stale one that doesn't include the documents you just added.
# # #     Not wired in automatically from pdf_pipeline.embed_and_store() itself,
# # #     to avoid a circular import between ingestion and retrieval — call it
# # #     explicitly from whichever orchestration layer just finished ingesting.
# # #     """
# # #     _bm25_cache.pop(collection_name, None)


# # # # ---------------------------------------------------------------------------
# # # # Metadata filter -> Chroma `where` clause, and a matching Python-side
# # # # predicate for the BM25 branch (which Chroma's `where` can't reach).
# # # # ---------------------------------------------------------------------------

# # # def build_where_clause(filters: dict) -> Optional[dict]:
# # #     clauses: list[dict] = []
# # #     if filters.get("contract_type"):
# # #         clauses.append({"contract_type": filters["contract_type"]})
# # #     if filters.get("governing_law_country"):
# # #         clauses.append({"governing_law_country": filters["governing_law_country"].upper()})
# # #     if filters.get("min_effective_date_epoch") is not None:
# # #         clauses.append({"effective_date_epoch": {"$gte": filters["min_effective_date_epoch"]}})
# # #     if filters.get("max_effective_date_epoch") is not None:
# # #         clauses.append({"effective_date_epoch": {"$lte": filters["max_effective_date_epoch"]}})
# # #     if filters.get("min_monetary_value") is not None:
# # #         clauses.append({"monetary_value": {"$gte": filters["min_monetary_value"]}})
# # #     if filters.get("max_monetary_value") is not None:
# # #         clauses.append({"monetary_value": {"$lte": filters["max_monetary_value"]}})

# # #     if not clauses:
# # #         return None
# # #     return clauses[0] if len(clauses) == 1 else {"$and": clauses}


# # # def _matches_where(meta: dict, where: Optional[dict]) -> bool:
# # #     """Python-side equivalent of build_where_clause(), for filtering BM25 candidates."""
# # #     if not where:
# # #         return True
# # #     if "$and" in where:
# # #         return all(_matches_where(meta, clause) for clause in where["$and"])
# # #     (field, condition), = where.items()
# # #     value = meta.get(field)
# # #     if isinstance(condition, dict):
# # #         if "$gte" in condition and not (value is not None and value >= condition["$gte"]):
# # #             return False
# # #         if "$lte" in condition and not (value is not None and value <= condition["$lte"]):
# # #             return False
# # #         return True
# # #     return value == condition


# # # # ---------------------------------------------------------------------------
# # # # Dense and sparse branches
# # # # ---------------------------------------------------------------------------

# # # def _dense_search(collection_name: str, query: str, top_k: int, where: Optional[dict]) -> list[dict]:
# # #     """Chroma applies `where` natively DURING the ANN search — filtering here was never the issue."""
# # #     collection = get_chroma_collection(collection_name)
# # #     embedder = get_embedder()
# # #     query_embedding = embedder.encode([query], normalize_embeddings=True).tolist()
# # #     results = collection.query(query_embeddings=query_embedding, n_results=top_k, where=where)

# # #     hits = []
# # #     if results["ids"]:
# # #         for doc_id, doc, meta, dist in zip(
# # #             results["ids"][0], results["documents"][0], results["metadatas"][0], results["distances"][0]
# # #         ):
# # #             hits.append({"id": doc_id, "text": doc, "metadata": meta, "dense_distance": dist})
# # #     return hits


# # # def _sparse_search(collection_name: str, query: str, top_k: int, where: Optional[dict]) -> list[dict]:
# # #     """
# # #     Filter FIRST (restrict to the set of documents that satisfy `where`),
# # #     THEN rank that eligible set by BM25 score — not the other way around.
# # #     Ranking the whole corpus first and filtering afterward risks returning
# # #     fewer than top_k results (or zero) under a restrictive filter even when
# # #     plenty of matching-and-relevant documents exist further down the ranked
# # #     list; filtering the candidate pool before selecting top_k guarantees we
# # #     never miss an eligible match because it happened to rank outside some
# # #     arbitrary pre-filter cutoff.
# # #     """
# # #     bm25, ids, texts, metadatas = _get_bm25_index(collection_name)
# # #     scores = bm25.get_scores(_tokenize(query))

# # #     eligible = [i for i in range(len(ids)) if _matches_where(metadatas[i], where)]
# # #     eligible_ranked = sorted(eligible, key=lambda i: scores[i], reverse=True)[:top_k]

# # #     return [
# # #         {"id": ids[i], "text": texts[i], "metadata": metadatas[i], "bm25_score": float(scores[i])}
# # #         for i in eligible_ranked
# # #     ]


# # # # ---------------------------------------------------------------------------
# # # # Fusion strategies
# # # # ---------------------------------------------------------------------------

# # # def reciprocal_rank_fusion(id_lists: list[list[str]], k: int = 60) -> dict[str, float]:
# # #     """
# # #     Rank-only fusion: score(d) = sum, over lists containing d, of
# # #     1 / (k + rank_in_that_list). Doesn't need the two branches' raw scores
# # #     to be comparable at all — only their relative ordering — which makes it
# # #     a safe default when you don't want to tune a blend weight. Kept as an
# # #     alternate to convex_combination_fusion (see hybrid_search()'s
# # #     fusion_method parameter).
# # #     """
# # #     scores: dict[str, float] = {}
# # #     for id_list in id_lists:
# # #         for rank, doc_id in enumerate(id_list):
# # #             scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
# # #     return scores


# # # def min_max_normalize(scores: dict[str, float], reverse: bool = False) -> dict[str, float]:
# # #     """Normalizes scores to the [0, 1] range using Min-Max scaling.

# # #     Args:
# # #         scores: Dictionary mapping document IDs to raw scores.
# # #         reverse: Set to True if lower raw score is better (e.g. L2 or Cosine
# # #           Distance).

# # #     Returns:
# # #         Dictionary mapping document IDs to normalized scores in [0, 1].
# # #     """
# # #     if not scores:
# # #         return {}
# # #     vals = list(scores.values())
# # #     min_val, max_val = min(vals), max(vals)
# # #     # Edge case: all candidates have the identical raw score.
# # #     if min_val == max_val:
# # #         return {doc_id: 1.0 for doc_id in scores}
# # #     normalized = {}
# # #     for doc_id, score in scores.items():
# # #         if reverse:
# # #             # For distance metrics where lower is better.
# # #             norm_score = (max_val - score) / (max_val - min_val)
# # #         else:
# # #             # For similarity/BM25 metrics where higher is better.
# # #             norm_score = (score - min_val) / (max_val - min_val)
# # #         normalized[doc_id] = float(norm_score)
# # #     return normalized


# # # def convex_combination_fusion(
# # #     dense_hits: list[dict],
# # #     sparse_hits: list[dict],
# # #     alpha: float = 0.4,
# # #     dense_score_key: str = "dense_distance",
# # #     sparse_score_key: str = "bm25_score",
# # #     dense_is_distance: bool = True,
# # # ) -> dict[str, float]:
# # #     """Combines dense and sparse results using a weighted convex combination.

# # #     Final Score = alpha * Dense_Norm + (1 - alpha) * Sparse_Norm

# # #     Default alpha=0.4 weights lexical (BM25) matching more heavily than
# # #     dense similarity (40% dense / 60% sparse) — legal text leans on exact
# # #     clause wording, defined terms, and citations, where lexical match is
# # #     often the stronger signal; raise alpha toward 1.0 if your queries skew
# # #     more paraphrase/semantic than exact-term.

# # #     Args:
# # #         dense_hits: List of dicts, e.g. [{"id": "doc1", "dense_distance": 0.12}]
# # #         sparse_hits: List of dicts, e.g. [{"id": "doc1", "bm25_score": 14.2}]
# # #         alpha: Weight for dense retrieval (0.0 to 1.0). High alpha = favor
# # #           dense.
# # #         dense_score_key: Dictionary key containing the dense metric.
# # #         sparse_score_key: Dictionary key containing the sparse metric.
# # #         dense_is_distance: Set True if dense score is distance (lower =
# # #           better).

# # #     Returns:
# # #         Dictionary mapping doc IDs to their combined score, sorted descending.
# # #     """
# # #     # 1. Extract raw scores into mappings.
# # #     raw_dense = {h["id"]: h[dense_score_key] for h in dense_hits}
# # #     raw_sparse = {h["id"]: h[sparse_score_key] for h in sparse_hits}

# # #     # 2. Normalize both score distributions independently to [0, 1].
# # #     norm_dense = min_max_normalize(raw_dense, reverse=dense_is_distance)
# # #     norm_sparse = min_max_normalize(raw_sparse, reverse=False)

# # #     # 3. Combine scores across the union of all candidate document IDs.
# # #     all_doc_ids = set(norm_dense.keys()).union(set(norm_sparse.keys()))
# # #     fused_scores = {}
# # #     for doc_id in all_doc_ids:
# # #         # If a document is missing in one branch, treat its normalized score as 0.0.
# # #         d_score = norm_dense.get(doc_id, 0.0)
# # #         s_score = norm_sparse.get(doc_id, 0.0)
# # #         fused_scores[doc_id] = (alpha * d_score) + ((1.0 - alpha) * s_score)

# # #     # Sort candidates by fused score descending.
# # #     return dict(sorted(fused_scores.items(), key=lambda item: item[1], reverse=True))


# # # def rerank(query: str, candidates: list[dict], top_k: int) -> list[dict]:
# # #     """
# # #     Cross-encoder rerank of the fused candidate set. The reranker itself is
# # #     a module-level singleton loaded via resources.get_reranker() — NOT
# # #     instantiated here — see resources.preload_models() to load it eagerly
# # #     at application startup instead of lazily on the first call.
# # #     """
# # #     if not candidates:
# # #         return []
# # #     reranker = get_reranker()
# # #     pairs = [(query, c["text"]) for c in candidates]
# # #     scores = reranker.predict(pairs)
# # #     for c, s in zip(candidates, scores):
# # #         c["rerank_score"] = float(s)
# # #     return sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)[:top_k]


# # # # ---------------------------------------------------------------------------
# # # # Public entrypoint
# # # # ---------------------------------------------------------------------------

# # # @traceable(name="retrieval.hybrid_search_agent", run_type="retriever")
# # # def hybrid_search(
# # #     collection_name: str,
# # #     query: str,
# # #     metadata_filter: Optional[dict] = None,
# # #     dense_k: int = 20,
# # #     sparse_k: int = 20,
# # #     fusion_k: int = 15,
# # #     final_k: int = 6,
# # #     fusion_method: str = "convex",
# # #     alpha: float = 0.4,
# # # ) -> list[dict]:
# # #     """
# # #     Full pipeline: dense search + BM25 search (each metadata-filtered) ->
# # #     fusion ("convex" weighted combination by default, or "rrf" for
# # #     rank-only fusion) -> parties_contains post-filter -> cross-encoder
# # #     rerank -> top final_k results, each:
# # #     {"id", "text", "metadata", "rerank_score", ...}.
# # #     """
# # #     metadata_filter = metadata_filter or {}
# # #     where = build_where_clause(metadata_filter)

# # #     dense_hits = _dense_search(collection_name, query, dense_k, where)
# # #     sparse_hits = _sparse_search(collection_name, query, sparse_k, where)

# # #     by_id: dict[str, dict] = {}
# # #     for h in dense_hits + sparse_hits:
# # #         by_id.setdefault(h["id"], {}).update(h)

# # #     if fusion_method == "rrf":
# # #         fused_scores = reciprocal_rank_fusion(
# # #             [[h["id"] for h in dense_hits], [h["id"] for h in sparse_hits]]
# # #         )
# # #     else:
# # #         fused_scores = convex_combination_fusion(
# # #             dense_hits=dense_hits,
# # #             sparse_hits=sparse_hits,
# # #             alpha=alpha,
# # #             dense_is_distance=True,  # Chroma returns distances where lower = better
# # #         )

# # #     fused_ranked_ids = list(fused_scores.keys())[:fusion_k]
# # #     candidates = [by_id[i] for i in fused_ranked_ids if i in by_id]

# # #     parties_needle = metadata_filter.get("parties_contains")
# # #     if parties_needle:
# # #         needle = parties_needle.lower()
# # #         candidates = [c for c in candidates if needle in (c["metadata"].get("parties", "") or "").lower()]

# # #     return rerank(query, candidates, top_k=final_k)
# # #

# """
# hybrid_search.py
# ================

# Hierarchy-aware hybrid retrieval for legal / financial documents.

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
#                                         +--> Chunk

# Retrieval pipeline:

#     Query
#       |
#       +--> Dense retrieval
#       |
#       +--> BM25 retrieval
#       |
#       +--> Hybrid fusion
#       |
#       +--> Hierarchy-aware boosting
#       |
#       +--> Deduplication
#       |
#       +--> Parent / sibling expansion
#       |
#       +--> Cross-encoder reranking
#       |
#       +--> Final results


# Metadata expected from pdf_pipeline.py:

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

# Document metadata:

#     contract_type
#     parties
#     governing_law_country

#     effective_date_epoch
#     end_date_epoch
#     monetary_value


# TABLE SUPPORT
# -------------

# Tables remain intact as retrieval chunks.

# A table is therefore retrieved through:

#     1. its own semantic content
#     2. its own lexical content
#     3. its hierarchy metadata

# Example:

#     Query:

#         "What are the liabilities shown in clause 8.2?"

# can retrieve:

#         clause_number = 8
#         subclause_number = 8.2
#         content_type = table

# even if the table itself does not repeat the complete clause title.


# BM25
# ----

# BM25 is maintained in memory.

# The cache is invalidated by pdf_pipeline.py after ingestion.

# For very large production collections, move sparse retrieval to a
# native sparse vector backend such as Qdrant.
# """

# from __future__ import annotations

# import math
# import re

# from collections import Counter
# from dataclasses import dataclass
# from typing import Any, Optional

# from ..resources import get_chroma_collection
# from ..resources import get_embedder
# from ..resources import get_reranker

# from ..tracing import traceable


# # ============================================================================
# # CONFIGURATION
# # ============================================================================

# DEFAULT_DENSE_K = 30
# DEFAULT_SPARSE_K = 30

# DEFAULT_FUSION_K = 30
# DEFAULT_FINAL_K = 6

# DEFAULT_ALPHA = 0.45

# DEFAULT_NEIGHBOR_WINDOW = 1

# DEFAULT_MAX_EXPANDED_CANDIDATES = 40


# # ============================================================================
# # HIERARCHY BOOSTS
# # ============================================================================

# SECTION_BOOST = 0.08

# CLAUSE_NUMBER_BOOST = 0.20
# CLAUSE_TITLE_BOOST = 0.15

# PARENT_CLAUSE_BOOST = 0.10

# SUBCLAUSE_NUMBER_BOOST = 0.25
# SUBCLAUSE_TITLE_BOOST = 0.15

# DOCUMENT_NAME_BOOST = 0.05

# TABLE_BOOST = 0.05


# # ============================================================================
# # BM25 CACHE
# # ============================================================================

# _bm25_cache: dict[
#     str,
#     tuple[
#         "LegalBM25",
#         list[str],
#         list[str],
#         list[dict],
#     ],
# ] = {}


# # ============================================================================
# # TOKENIZATION
# # ============================================================================

# _TOKEN_RE = re.compile(
#     r"[a-z0-9]+"
# )


# def _tokenize_raw(
#     text: str,
# ) -> list[str]:
#     """
#     Raw tokenizer.

#     Keeps:

#         termination
#         indemnification
#         revenue
#         12
#         12.3
#         12.3.1
#     """

#     if not text:
#         return []

#     return _TOKEN_RE.findall(
#         text.lower()
#     )


# LEGAL_STOPWORDS = {
#     # Articles
#     "a",
#     "an",
#     "the",

#     # Auxiliaries
#     "is",
#     "are",
#     "was",
#     "were",
#     "be",
#     "been",
#     "being",
#     "has",
#     "have",
#     "had",
#     "do",
#     "does",
#     "did",
#     "can",
#     "could",
#     "would",
#     "should",
#     "will",

#     # Prepositions
#     "of",
#     "to",
#     "in",
#     "on",
#     "for",
#     "from",
#     "by",
#     "with",
#     "at",
#     "into",
#     "about",
#     "over",
#     "under",

#     # Conjunctions
#     "and",
#     "or",
#     "but",
#     "if",
#     "then",
#     "than",

#     # Pronouns
#     "it",
#     "its",
#     "this",
#     "that",
#     "these",
#     "those",

#     # Question words
#     "what",
#     "which",
#     "when",
#     "where",
#     "who",
#     "how",
# }


# def _tokenize(
#     text: str,
# ) -> list[str]:
#     """
#     Legal-aware tokenizer.

#     Generic English stopwords are removed, but legal-domain words
#     such as:

#         agreement
#         party
#         parties
#         shall
#         termination
#         effective
#         date
#         law
#         liability
#         indemnity

#     are retained.
#     """

#     tokens = _tokenize_raw(
#         text
#     )

#     return [
#         token
#         for token in tokens
#         if token not in LEGAL_STOPWORDS
#     ]


# # ============================================================================
# # LEGAL PHRASES
# # ============================================================================

# LEGAL_PHRASES = (
#     "governing law",
#     "effective date",
#     "expiration date",
#     "termination date",
#     "termination agreement",
#     "confidential information",
#     "intellectual property",
#     "limitation of liability",
#     "liability limitation",
#     "indemnification",
#     "indemnity",
#     "force majeure",
#     "change of control",
#     "notice period",
#     "notice provision",
#     "non compete",
#     "non solicitation",
#     "assignment",
#     "representations and warranties",
#     "representation and warranty",
#     "warranty",
#     "warranties",
# )


# def _extract_query_phrases(
#     query: str,
# ) -> list[str]:
#     """
#     Find known legal phrases occurring in the query.
#     """

#     normalized = re.sub(
#         r"\s+",
#         " ",
#         query.lower(),
#     ).strip()

#     return [
#         phrase
#         for phrase in LEGAL_PHRASES
#         if phrase in normalized
#     ]


# # ============================================================================
# # CLAUSE NUMBER EXTRACTION
# # ============================================================================

# def _extract_hierarchy_numbers(
#     query: str,
# ) -> dict[str, list[str]]:
#     """
#     Extract likely clause / subclause numbers from a query.

#     Examples:

#         "clause 12.3"
#         "section 8.2"
#         "12.3.1"

#     """

#     numbers = re.findall(
#         r"\b\d+(?:\.\d+){0,4}\b",
#         query,
#     )

#     result = {
#         "clause_numbers": [],
#         "subclause_numbers": [],
#     }

#     for number in numbers:

#         components = number.split(".")

#         if len(components) == 1:
#             result[
#                 "clause_numbers"
#             ].append(number)

#         else:
#             result[
#                 "subclause_numbers"
#             ].append(number)

#             # Parent clause of 12.3.1 is 12.
#             result[
#                 "clause_numbers"
#             ].append(
#                 components[0]
#             )

#     return result


# # ============================================================================
# # LEGAL BM25
# # ============================================================================

# @dataclass
# class LegalBM25:

#     tokenized_corpus: list[list[str]]

#     k1: float = 1.5

#     b: float = 0.75

#     phrase_bonus: float = 2.0

#     def __post_init__(
#         self,
#     ) -> None:

#         self.N = len(
#             self.tokenized_corpus
#         )

#         self.doc_len = [
#             len(tokens)
#             for tokens
#             in self.tokenized_corpus
#         ]

#         self.avgdl = (
#             sum(self.doc_len)
#             / self.N
#             if self.N
#             else 0.0
#         )

#         self.doc_freqs: list[
#             Counter[str]
#         ] = []

#         self.df: Counter[str] = (
#             Counter()
#         )

#         for tokens in (
#             self.tokenized_corpus
#         ):

#             frequencies = Counter(
#                 tokens
#             )

#             self.doc_freqs.append(
#                 frequencies
#             )

#             for term in frequencies:

#                 self.df[term] += 1

#         self.idf: dict[
#             str,
#             float,
#         ] = {}

#         for term, df in (
#             self.df.items()
#         ):

#             self.idf[term] = math.log(
#                 1.0
#                 + (
#                     self.N
#                     - df
#                     + 0.5
#                 )
#                 / (
#                     df
#                     + 0.5
#                 )
#             )

#     def get_scores(
#         self,
#         query_tokens: list[str],
#     ) -> list[float]:

#         if not self.tokenized_corpus:
#             return []

#         scores = [
#             0.0
#             for _ in range(
#                 self.N
#             )
#         ]

#         for term in query_tokens:

#             if term not in self.idf:
#                 continue

#             idf = self.idf[
#                 term
#             ]

#             for index, frequencies in (
#                 enumerate(
#                     self.doc_freqs
#                 )
#             ):

#                 tf = frequencies.get(
#                     term,
#                     0,
#                 )

#                 if tf == 0:
#                     continue

#                 doc_length = (
#                     self.doc_len[
#                         index
#                     ]
#                 )

#                 if self.avgdl == 0:

#                     length_normalization = (
#                         1.0
#                     )

#                 else:

#                     length_normalization = (
#                         1.0
#                         - self.b
#                         + self.b
#                         * doc_length
#                         / self.avgdl
#                     )

#                 denominator = (
#                     tf
#                     + self.k1
#                     * length_normalization
#                 )

#                 scores[index] += (
#                     idf
#                     * tf
#                     * (
#                         self.k1
#                         + 1.0
#                     )
#                     / denominator
#                 )

#         return scores


# # ============================================================================
# # BM25 INDEX
# # ============================================================================

# def _build_bm25_index(
#     collection_name: str,
# ):
#     """
#     Load all Chroma documents and build the lexical index.
#     """

#     collection = (
#         get_chroma_collection(
#             collection_name
#         )
#     )

#     raw = collection.get(
#         include=[
#             "documents",
#             "metadatas",
#         ]
#     )

#     ids = raw[
#         "ids"
#     ]

#     texts = raw[
#         "documents"
#     ]

#     metadatas = raw[
#         "metadatas"
#     ]

#     tokenized_documents = [
#         _tokenize(text)
#         for text in texts
#     ]

#     bm25 = LegalBM25(
#         tokenized_corpus=(
#             tokenized_documents
#         ),
#         k1=1.5,
#         b=0.75,
#         phrase_bonus=2.0,
#     )

#     _bm25_cache[
#         collection_name
#     ] = (
#         bm25,
#         ids,
#         texts,
#         metadatas,
#     )

#     return _bm25_cache[
#         collection_name
#     ]


# def _get_bm25_index(
#     collection_name: str,
#     refresh: bool = False,
# ):
#     """
#     Get cached BM25 index.

#     refresh=True forces reconstruction.
#     """

#     if (
#         not refresh
#         and collection_name
#         in _bm25_cache
#     ):

#         return _bm25_cache[
#             collection_name
#         ]

#     return _build_bm25_index(
#         collection_name
#     )


# def invalidate_bm25_cache(
#     collection_name: str,
# ) -> None:
#     """
#     Called by pdf_pipeline.py after ingestion.
#     """

#     _bm25_cache.pop(
#         collection_name,
#         None,
#     )


# # ============================================================================
# # METADATA FILTERING
# # ============================================================================

# def build_where_clause(
#     filters: dict,
# ) -> Optional[dict]:
#     """
#     Convert application metadata filters into Chroma syntax.
#     """

#     clauses: list[dict] = []

#     if filters.get(
#         "contract_type"
#     ):

#         clauses.append(
#             {
#                 "contract_type":
#                     filters[
#                         "contract_type"
#                     ]
#             }
#         )

#     if filters.get(
#         "governing_law_country"
#     ):

#         clauses.append(
#             {
#                 "governing_law_country":
#                     filters[
#                         "governing_law_country"
#                     ].upper()
#             }
#         )

#     if filters.get(
#         "min_effective_date_epoch"
#     ) is not None:

#         clauses.append(
#             {
#                 "effective_date_epoch": {
#                     "$gte":
#                         filters[
#                             "min_effective_date_epoch"
#                         ]
#                 }
#             }
#         )

#     if filters.get(
#         "max_effective_date_epoch"
#     ) is not None:

#         clauses.append(
#             {
#                 "effective_date_epoch": {
#                     "$lte":
#                         filters[
#                             "max_effective_date_epoch"
#                         ]
#                 }
#             }
#         )

#     if filters.get(
#         "min_end_date_epoch"
#     ) is not None:

#         clauses.append(
#             {
#                 "end_date_epoch": {
#                     "$gte":
#                         filters[
#                             "min_end_date_epoch"
#                         ]
#                 }
#             }
#         )

#     if filters.get(
#         "max_end_date_epoch"
#     ) is not None:

#         clauses.append(
#             {
#                 "end_date_epoch": {
#                     "$lte":
#                         filters[
#                             "max_end_date_epoch"
#                         ]
#                 }
#             }
#         )

#     if filters.get(
#         "min_monetary_value"
#     ) is not None:

#         clauses.append(
#             {
#                 "monetary_value": {
#                     "$gte":
#                         filters[
#                             "min_monetary_value"
#                         ]
#                 }
#             }
#         )

#     if filters.get(
#         "max_monetary_value"
#     ) is not None:

#         clauses.append(
#             {
#                 "monetary_value": {
#                     "$lte":
#                         filters[
#                             "max_monetary_value"
#                         ]
#                 }
#             }
#         )

#     if not clauses:
#         return None

#     if len(clauses) == 1:
#         return clauses[0]

#     return {
#         "$and": clauses
#     }


# def _matches_where(
#     metadata: dict,
#     where: Optional[dict],
# ) -> bool:
#     """
#     Apply the same filters to the in-memory BM25 index.
#     """

#     if not where:
#         return True

#     if "$and" in where:

#         return all(
#             _matches_where(
#                 metadata,
#                 clause,
#             )
#             for clause in where[
#                 "$and"
#             ]
#         )

#     (
#         field,
#         condition,
#     ), = where.items()

#     value = metadata.get(
#         field
#     )

#     if isinstance(
#         condition,
#         dict,
#     ):

#         if "$gte" in condition:

#             if (
#                 value is None
#                 or value
#                 < condition[
#                     "$gte"
#                 ]
#             ):
#                 return False

#         if "$lte" in condition:

#             if (
#                 value is None
#                 or value
#                 > condition[
#                     "$lte"
#                 ]
#             ):
#                 return False

#         return True

#     return value == condition


# # ============================================================================
# # DENSE RETRIEVAL
# # ============================================================================

# def _dense_search(
#     collection_name: str,
#     query: str,
#     top_k: int,
#     where: Optional[dict],
# ) -> list[dict]:
#     """
#     Semantic retrieval through Chroma.
#     """

#     collection = (
#         get_chroma_collection(
#             collection_name
#         )
#     )

#     embedder = get_embedder()

#     query_embedding = (
#         embedder.encode(
#             [query],
#             normalize_embeddings=True,
#         ).tolist()
#     )

#     results = collection.query(
#         query_embeddings=query_embedding,
#         n_results=top_k,
#         where=where,
#         include=[
#             "documents",
#             "metadatas",
#             "distances",
#         ],
#     )

#     hits: list[dict] = []

#     if not results.get(
#         "ids"
#     ):
#         return hits

#     for (
#         doc_id,
#         text,
#         metadata,
#         distance,
#     ) in zip(
#         results["ids"][0],
#         results["documents"][0],
#         results["metadatas"][0],
#         results["distances"][0],
#     ):

#         hits.append(
#             {
#                 "id":
#                     doc_id,

#                 "text":
#                     text,

#                 "metadata":
#                     metadata,

#                 "dense_distance":
#                     float(distance),
#             }
#         )

#     return hits


# # ============================================================================
# # PHRASE BOOSTING
# # ============================================================================

# def _apply_phrase_boost(
#     scores: list[float],
#     texts: list[str],
#     phrases: list[str],
#     phrase_bonus: float = 2.0,
# ) -> list[float]:

#     if not phrases:
#         return scores

#     boosted = scores.copy()

#     for index, text in enumerate(
#         texts
#     ):

#         normalized = re.sub(
#             r"\s+",
#             " ",
#             text.lower(),
#         )

#         for phrase in phrases:

#             if phrase in normalized:

#                 boosted[
#                     index
#                 ] += phrase_bonus

#     return boosted


# # ============================================================================
# # SPARSE RETRIEVAL
# # ============================================================================

# def _sparse_search(
#     collection_name: str,
#     query: str,
#     top_k: int,
#     where: Optional[dict],
# ) -> list[dict]:
#     """
#     Legal-aware BM25 retrieval.
#     """

#     (
#         bm25,
#         ids,
#         texts,
#         metadatas,
#     ) = _get_bm25_index(
#         collection_name
#     )

#     query_tokens = _tokenize(
#         query
#     )

#     scores = bm25.get_scores(
#         query_tokens
#     )

#     phrases = (
#         _extract_query_phrases(
#             query
#         )
#     )

#     scores = _apply_phrase_boost(
#         scores=scores,
#         texts=texts,
#         phrases=phrases,
#         phrase_bonus=(
#             bm25.phrase_bonus
#         ),
#     )

#     eligible = [
#         index
#         for index in range(
#             len(ids)
#         )
#         if _matches_where(
#             metadatas[index],
#             where,
#         )
#     ]

#     ranked = sorted(
#         eligible,
#         key=lambda index:
#             scores[index],
#         reverse=True,
#     )

#     ranked = ranked[
#         :top_k
#     ]

#     return [
#         {
#             "id":
#                 ids[index],

#             "text":
#                 texts[index],

#             "metadata":
#                 metadatas[index],

#             "bm25_score":
#                 float(
#                     scores[index]
#                 ),
#         }
#         for index in ranked
#     ]


# # ============================================================================
# # SCORE NORMALIZATION
# # ============================================================================

# def min_max_normalize(
#     scores: dict[str, float],
#     reverse: bool = False,
# ) -> dict[str, float]:

#     if not scores:
#         return {}

#     values = list(
#         scores.values()
#     )

#     minimum = min(
#         values
#     )

#     maximum = max(
#         values
#     )

#     if minimum == maximum:

#         return {
#             doc_id: 1.0
#             for doc_id in scores
#         }

#     normalized = {}

#     for doc_id, score in (
#         scores.items()
#     ):

#         if reverse:

#             value = (
#                 maximum - score
#             ) / (
#                 maximum - minimum
#             )

#         else:

#             value = (
#                 score - minimum
#             ) / (
#                 maximum - minimum
#             )

#         normalized[
#             doc_id
#         ] = float(value)

#     return normalized


# # ============================================================================
# # CONVEX FUSION
# # ============================================================================

# def convex_combination_fusion(
#     dense_hits: list[dict],
#     sparse_hits: list[dict],
#     alpha: float = DEFAULT_ALPHA,
# ) -> dict[str, float]:
#     """
#     Convex hybrid fusion.

#         score =
#             alpha * dense
#             +
#             (1-alpha) * sparse

#     Chroma distance is reversed because lower distance is better.
#     """

#     raw_dense = {
#         hit["id"]:
#             hit["dense_distance"]
#         for hit in dense_hits
#     }

#     raw_sparse = {
#         hit["id"]:
#             hit["bm25_score"]
#         for hit in sparse_hits
#     }

#     dense_normalized = (
#         min_max_normalize(
#             raw_dense,
#             reverse=True,
#         )
#     )

#     sparse_normalized = (
#         min_max_normalize(
#             raw_sparse,
#             reverse=False,
#         )
#     )

#     all_ids = (
#         set(
#             dense_normalized
#         )
#         |
#         set(
#             sparse_normalized
#         )
#     )

#     scores = {}

#     for doc_id in all_ids:

#         dense_score = (
#             dense_normalized.get(
#                 doc_id,
#                 0.0,
#             )
#         )

#         sparse_score = (
#             sparse_normalized.get(
#                 doc_id,
#                 0.0,
#             )
#         )

#         scores[
#             doc_id
#         ] = (
#             alpha
#             * dense_score
#             +
#             (1.0 - alpha)
#             * sparse_score
#         )

#     return dict(
#         sorted(
#             scores.items(),
#             key=lambda item:
#                 item[1],
#             reverse=True,
#         )
#     )


# # ============================================================================
# # RRF
# # ============================================================================

# def reciprocal_rank_fusion(
#     id_lists: list[list[str]],
#     k: int = 60,
# ) -> dict[str, float]:
#     """
#     Reciprocal Rank Fusion.
#     """

#     scores: dict[
#         str,
#         float,
#     ] = {}

#     for id_list in id_lists:

#         for rank, doc_id in enumerate(
#             id_list
#         ):

#             scores[
#                 doc_id
#             ] = (
#                 scores.get(
#                     doc_id,
#                     0.0,
#                 )
#                 +
#                 1.0
#                 / (
#                     k
#                     + rank
#                     + 1
#                 )
#             )

#     return dict(
#         sorted(
#             scores.items(),
#             key=lambda item:
#                 item[1],
#             reverse=True,
#         )
#     )


# # ============================================================================
# # HIERARCHY MATCHING
# # ============================================================================

# def _normalized(
#     value: Any,
# ) -> str:

#     if value is None:
#         return ""

#     return str(
#         value
#     ).strip().lower()


# def _contains_term(
#     value: Any,
#     query: str,
# ) -> bool:

#     value_normalized = _normalized(
#         value
#     )

#     query_normalized = _normalized(
#         query
#     )

#     if not value_normalized:
#         return False

#     return (
#         query_normalized
#         in value_normalized
#     )


# # ============================================================================
# # HIERARCHY BOOST
# # ============================================================================

# def hierarchy_boost(
#     query: str,
#     metadata: dict,
# ) -> float:
#     """
#     Calculate hierarchy-aware retrieval boost.

#     The query is inspected for:

#         section names
#         clause numbers
#         subclause numbers
#         clause titles
#         document names

#     Exact hierarchy references receive stronger boosts than generic
#     lexical matches.
#     """

#     boost = 0.0

#     hierarchy_numbers = (
#         _extract_hierarchy_numbers(
#             query
#         )
#     )

#     clause_numbers = (
#         hierarchy_numbers[
#             "clause_numbers"
#         ]
#     )

#     subclause_numbers = (
#         hierarchy_numbers[
#             "subclause_numbers"
#         ]
#     )

#     # ------------------------------------------------------------
#     # Clause number
#     # ------------------------------------------------------------

#     candidate_clause = (
#         _normalized(
#             metadata.get(
#                 "clause_number"
#             )
#         )
#     )

#     for clause_number in (
#         clause_numbers
#     ):

#         if (
#             candidate_clause
#             == _normalized(
#                 clause_number
#             )
#         ):

#             boost += (
#                 CLAUSE_NUMBER_BOOST
#             )

#     # ------------------------------------------------------------
#     # Subclause number
#     # ------------------------------------------------------------

#     candidate_subclause = (
#         _normalized(
#             metadata.get(
#                 "subclause_number"
#             )
#         )
#     )

#     for number in (
#         subclause_numbers
#     ):

#         if (
#             candidate_subclause
#             == _normalized(
#                 number
#             )
#         ):

#             boost += (
#                 SUBCLAUSE_NUMBER_BOOST
#             )

#     # ------------------------------------------------------------
#     # Clause title
#     # ------------------------------------------------------------

#     clause_title = (
#         _normalized(
#             metadata.get(
#                 "clause_title"
#             )
#         )
#     )

#     if (
#         clause_title
#         and _query_overlaps_field(
#             query,
#             clause_title,
#         )
#     ):

#         boost += (
#             CLAUSE_TITLE_BOOST
#         )

#     # ------------------------------------------------------------
#     # Subclause title
#     # ------------------------------------------------------------

#     subclause_title = (
#         _normalized(
#             metadata.get(
#                 "subclause_title"
#             )
#         )
#     )

#     if (
#         subclause_title
#         and _query_overlaps_field(
#             query,
#             subclause_title,
#         )
#     ):

#         boost += (
#             SUBCLAUSE_TITLE_BOOST
#         )

#     # ------------------------------------------------------------
#     # Parent clause
#     # ------------------------------------------------------------

#     parent_clause = (
#         _normalized(
#             metadata.get(
#                 "parent_clause"
#             )
#         )
#     )

#     for clause_number in (
#         clause_numbers
#     ):

#         if (
#             parent_clause
#             == _normalized(
#                 clause_number
#             )
#         ):

#             boost += (
#                 PARENT_CLAUSE_BOOST
#             )

#     # ------------------------------------------------------------
#     # Section
#     # ------------------------------------------------------------

#     section = (
#         _normalized(
#             metadata.get(
#                 "section"
#             )
#         )
#     )

#     if (
#         section
#         and _query_overlaps_field(
#             query,
#             section,
#         )
#     ):

#         boost += (
#             SECTION_BOOST
#         )

#     # ------------------------------------------------------------
#     # Document name
#     # ------------------------------------------------------------

#     document_name = (
#         _normalized(
#             metadata.get(
#                 "document_name"
#             )
#         )
#     )

#     if (
#         document_name
#         and _query_overlaps_field(
#             query,
#             document_name,
#         )
#     ):

#         boost += (
#             DOCUMENT_NAME_BOOST
#         )

#     # ------------------------------------------------------------
#     # Tables
#     # ------------------------------------------------------------

#     if (
#         metadata.get(
#             "content_type"
#         )
#         == "table"
#     ):

#         boost += (
#             TABLE_BOOST
#         )

#     return boost


# def _query_overlaps_field(
#     query: str,
#     field: str,
# ) -> bool:
#     """
#     Determine whether meaningful query tokens overlap with a hierarchy
#     field.

#     Generic stopwords are ignored.
#     """

#     query_tokens = set(
#         _tokenize(query)
#     )

#     field_tokens = set(
#         _tokenize(field)
#     )

#     if not query_tokens:
#         return False

#     return bool(
#         query_tokens
#         &
#         field_tokens
#     )


# # ============================================================================
# # APPLY HIERARCHY BOOST
# # ============================================================================

# def apply_hierarchy_boost(
#     query: str,
#     candidates: list[dict],
# ) -> list[dict]:
#     """
#     Add hierarchy_boost_score to every candidate.
#     """

#     for candidate in candidates:

#         boost = hierarchy_boost(
#             query=query,
#             metadata=(
#                 candidate.get(
#                     "metadata",
#                     {},
#                 )
#             ),
#         )

#         candidate[
#             "hierarchy_boost"
#         ] = float(
#             boost
#         )

#         candidate[
#             "hybrid_score"
#         ] = (
#             candidate.get(
#                 "fusion_score",
#                 0.0,
#             )
#             + boost
#         )

#     return sorted(
#         candidates,
#         key=lambda candidate:
#             candidate[
#                 "hybrid_score"
#             ],
#         reverse=True,
#     )


# # ============================================================================
# # DEDUPLICATION
# # ============================================================================

# def _deduplicate_hits(
#     hits: list[dict],
# ) -> list[dict]:
#     """
#     Deduplicate the same logical chunk.

#     We use:

#         document_name
#         chunk_index
#         text

#     rather than only Chroma UUID.
#     """

#     seen: set[
#         tuple[
#             str,
#             Any,
#             str,
#         ]
#     ] = set()

#     unique: list[
#         dict
#     ] = []

#     for hit in hits:

#         metadata = hit.get(
#             "metadata",
#             {},
#         )

#         key = (
#             str(
#                 metadata.get(
#                     "document_name",
#                     "",
#                 )
#             ),
#             metadata.get(
#                 "chunk_index"
#             ),
#             (
#                 hit.get(
#                     "text",
#                     "",
#                 )
#                 or ""
#             ).strip(),
#         )

#         if key in seen:
#             continue

#         seen.add(
#             key
#         )

#         unique.append(
#             hit
#         )

#     return unique


# # ============================================================================
# # PARENT / SIBLING EXPANSION
# # ============================================================================

# def _hierarchy_key(
#     metadata: dict,
# ) -> tuple:
#     """
#     Create a hierarchy key.

#     Used to identify chunks belonging to the same logical clause.
#     """

#     return (
#         metadata.get(
#             "document_name"
#         ),

#         metadata.get(
#             "section"
#         ),

#         metadata.get(
#             "clause_number"
#         ),

#         metadata.get(
#             "parent_clause"
#         ),

#         metadata.get(
#             "subclause_number"
#         ),
#     )


# def _same_clause(
#     left: dict,
#     right: dict,
# ) -> bool:

#     left_meta = left.get(
#         "metadata",
#         {},
#     )

#     right_meta = right.get(
#         "metadata",
#         {},
#     )

#     return (
#         left_meta.get(
#             "document_name"
#         )
#         ==
#         right_meta.get(
#             "document_name"
#         )
#         and
#         left_meta.get(
#             "clause_number"
#         )
#         ==
#         right_meta.get(
#             "clause_number"
#         )
#     )


# def _same_subclause(
#     left: dict,
#     right: dict,
# ) -> bool:

#     left_meta = left.get(
#         "metadata",
#         {},
#     )

#     right_meta = right.get(
#         "metadata",
#         {},
#     )

#     return (
#         _same_clause(
#             left,
#             right,
#         )
#         and
#         left_meta.get(
#             "subclause_number"
#         )
#         ==
#         right_meta.get(
#             "subclause_number"
#         )
#     )


# def expand_hierarchy_context(
#     collection_name: str,
#     candidates: list[dict],
#     max_candidates: int = DEFAULT_MAX_EXPANDED_CANDIDATES,
# ) -> list[dict]:
#     """
#     Recover parent/sibling context.

#     For every high-ranking candidate we retrieve other chunks from the
#     same clause/subclause.

#     This is especially important when:

#         chunk 42 = beginning of clause
#         chunk 43 = middle of clause
#         chunk 44 = table
#         chunk 45 = continuation

#     If chunk 44 is retrieved, the surrounding clause can still be
#     recovered.

#     IMPORTANT:

#     The table remains intact. Expansion does NOT split or modify it.
#     """

#     if not candidates:
#         return []

#     collection = (
#         get_chroma_collection(
#             collection_name
#         )
#     )

#     # ------------------------------------------------------------
#     # Seed candidates
#     # ------------------------------------------------------------

#     seed_candidates = candidates[
#         :min(
#             len(candidates),
#             DEFAULT_FUSION_K,
#         )
#     ]

#     # ------------------------------------------------------------
#     # Determine documents / clauses to expand
#     # ------------------------------------------------------------

#     target_documents = set()
#     target_clauses = set()
#     target_subclauses = set()

#     for candidate in (
#         seed_candidates
#     ):

#         metadata = candidate.get(
#             "metadata",
#             {},
#         )

#         document = metadata.get(
#             "document_name"
#         )

#         clause = metadata.get(
#             "clause_number"
#         )

#         subclause = metadata.get(
#             "subclause_number"
#         )

#         if document:
#             target_documents.add(
#                 document
#             )

#         if (
#             document
#             and clause
#         ):
#             target_clauses.add(
#                 (
#                     document,
#                     clause,
#                 )
#             )

#         if (
#             document
#             and subclause
#         ):
#             target_subclauses.add(
#                 (
#                     document,
#                     subclause,
#                 )
#             )

#     # ------------------------------------------------------------
#     # Chroma does not conveniently express all of these hierarchy
#     # conditions as one generic OR in every configuration.
#     #
#     # Retrieve a bounded set and filter locally.
#     # ------------------------------------------------------------

#     try:

#         raw = collection.get(
#             include=[
#                 "documents",
#                 "metadatas",
#             ]
#         )

#     except Exception:

#         return candidates

#     expanded = list(
#         candidates
#     )

#     existing_ids = {
#         candidate[
#             "id"
#         ]
#         for candidate in candidates
#     }

#     for (
#         doc_id,
#         text,
#         metadata,
#     ) in zip(
#         raw["ids"],
#         raw["documents"],
#         raw["metadatas"],
#     ):

#         if doc_id in existing_ids:
#             continue

#         document = metadata.get(
#             "document_name"
#         )

#         clause = metadata.get(
#             "clause_number"
#         )

#         subclause = metadata.get(
#             "subclause_number"
#         )

#         is_same_subclause = (
#             (
#                 document,
#                 subclause,
#             )
#             in target_subclauses
#         )

#         is_same_clause = (
#             (
#                 document,
#                 clause,
#             )
#             in target_clauses
#         )

#         if not (
#             is_same_subclause
#             or is_same_clause
#         ):
#             continue

#         expanded.append(
#             {
#                 "id":
#                     doc_id,

#                 "text":
#                     text,

#                 "metadata":
#                     metadata,

#                 "expanded":
#                     True,

#                 "fusion_score":
#                     0.0,

#                 "hierarchy_boost":
#                     0.0,

#                 "hybrid_score":
#                     0.0,
#             }
#         )

#         existing_ids.add(
#             doc_id
#         )

#         if len(expanded) >= (
#             max_candidates
#         ):
#             break

#     return _deduplicate_hits(
#         expanded
#     )


# # ============================================================================
# # RERANKING
# # ============================================================================

# def rerank(
#     query: str,
#     candidates: list[dict],
#     top_k: int,
# ) -> list[dict]:
#     """
#     Cross-encoder reranking.

#     The hierarchy-expanded candidates are all evaluated against the
#     original user query.
#     """

#     if not candidates:
#         return []

#     reranker = get_reranker()

#     pairs = [
#         (
#             query,
#             candidate[
#                 "text"
#             ],
#         )
#         for candidate in candidates
#     ]

#     scores = reranker.predict(
#         pairs
#     )

#     for candidate, score in zip(
#         candidates,
#         scores,
#     ):

#         candidate[
#             "rerank_score"
#         ] = float(
#             score
#         )

#     return sorted(
#         candidates,
#         key=lambda candidate:
#             candidate[
#                 "rerank_score"
#             ],
#         reverse=True,
#     )[:top_k]


# # ============================================================================
# # PARTIES FILTER
# # ============================================================================

# def _apply_parties_filter(
#     candidates: list[dict],
#     parties_contains: Optional[str],
# ) -> list[dict]:

#     if not parties_contains:
#         return candidates

#     needle = (
#         parties_contains
#         .lower()
#         .strip()
#     )

#     return [
#         candidate
#         for candidate in candidates
#         if needle
#         in (
#             candidate.get(
#                 "metadata",
#                 {},
#             ).get(
#                 "parties",
#                 "",
#             )
#             or ""
#         ).lower()
#     ]


# # ============================================================================
# # PUBLIC HYBRID SEARCH
# # ============================================================================

# @traceable(
#     name="hybrid_search_agent.search",
#     run_type="retriever",
# )
# def hybrid_search(
#     collection_name: str,
#     query: str,
#     metadata_filter: Optional[
#         dict
#     ] = None,
#     dense_k: int = DEFAULT_DENSE_K,
#     sparse_k: int = DEFAULT_SPARSE_K,
#     fusion_k: int = DEFAULT_FUSION_K,
#     final_k: int = DEFAULT_FINAL_K,
#     fusion_method: str = "convex",
#     alpha: float = DEFAULT_ALPHA,
#     expand_hierarchy: bool = True,
#     rerank_results: bool = True,
# ) -> list[dict]:
#     """
#     Complete hierarchy-aware hybrid retrieval.

#     Parameters
#     ----------
#     collection_name:
#         Chroma collection.

#     query:
#         User question.

#     metadata_filter:
#         Optional document-level filters.

#     dense_k:
#         Number of semantic candidates.

#     sparse_k:
#         Number of BM25 candidates.

#     fusion_k:
#         Candidates retained after fusion.

#     final_k:
#         Number of final results.

#     fusion_method:
#         "convex" or "rrf".

#     alpha:
#         Dense weight for convex fusion.

#         0.25 -> lexical heavy
#         0.45 -> balanced legal retrieval
#         0.70 -> semantic heavy

#     expand_hierarchy:
#         Whether to retrieve surrounding clause/subclause chunks.

#     rerank_results:
#         Whether to apply the cross-encoder.

#     Returns
#     -------
#     list[dict]
#         Final ranked retrieval results.
#     """

#     metadata_filter = (
#         metadata_filter
#         or {}
#     )

#     # ========================================================================
#     # 1. METADATA FILTER
#     # ========================================================================

#     where = build_where_clause(
#         metadata_filter
#     )

#     # ========================================================================
#     # 2. DENSE RETRIEVAL
#     # ========================================================================

#     dense_hits = _dense_search(
#         collection_name=(
#             collection_name
#         ),
#         query=query,
#         top_k=dense_k,
#         where=where,
#     )

#     # ========================================================================
#     # 3. BM25 RETRIEVAL
#     # ========================================================================

#     sparse_hits = _sparse_search(
#         collection_name=(
#             collection_name
#         ),
#         query=query,
#         top_k=sparse_k,
#         where=where,
#     )

#     # ========================================================================
#     # 4. MERGE CANDIDATES
#     # ========================================================================

#     by_id: dict[
#         str,
#         dict,
#     ] = {}

#     for hit in (
#         dense_hits
#         + sparse_hits
#     ):

#         doc_id = hit[
#             "id"
#         ]

#         if doc_id not in by_id:

#             by_id[
#                 doc_id
#             ] = {}

#         by_id[
#             doc_id
#         ].update(
#             hit
#         )

#     # ========================================================================
#     # 5. HYBRID FUSION
#     # ========================================================================

#     if (
#         fusion_method.lower()
#         == "rrf"
#     ):

#         fused_scores = (
#             reciprocal_rank_fusion(
#                 [
#                     [
#                         hit["id"]
#                         for hit
#                         in dense_hits
#                     ],
#                     [
#                         hit["id"]
#                         for hit
#                         in sparse_hits
#                     ],
#                 ]
#             )
#         )

#     else:

#         fused_scores = (
#             convex_combination_fusion(
#                 dense_hits=dense_hits,
#                 sparse_hits=sparse_hits,
#                 alpha=alpha,
#             )
#         )

#     # ========================================================================
#     # 6. CREATE FUSED CANDIDATES
#     # ========================================================================

#     fused_ids = list(
#         fused_scores.keys()
#     )[:fusion_k]

#     candidates: list[
#         dict
#     ] = []

#     for doc_id in fused_ids:

#         if doc_id not in by_id:
#             continue

#         candidate = dict(
#             by_id[doc_id]
#         )

#         candidate[
#             "fusion_score"
#         ] = float(
#             fused_scores[
#                 doc_id
#             ]
#         )

#         candidates.append(
#             candidate
#         )

#     # ========================================================================
#     # 7. HIERARCHY-AWARE BOOSTING
#     # ========================================================================

#     candidates = (
#         apply_hierarchy_boost(
#             query=query,
#             candidates=candidates,
#         )
#     )

#     # ========================================================================
#     # 8. DEDUPLICATION
#     # ========================================================================

#     candidates = (
#         _deduplicate_hits(
#             candidates
#         )
#     )

#     # ========================================================================
#     # 9. DOCUMENT-LEVEL PARTIES FILTER
#     # ========================================================================

#     candidates = (
#         _apply_parties_filter(
#             candidates=candidates,
#             parties_contains=(
#                 metadata_filter.get(
#                     "parties_contains"
#                 )
#             ),
#         )
#     )

#     # ========================================================================
#     # 10. PARENT / SIBLING EXPANSION
#     # ========================================================================

#     if expand_hierarchy:

#         candidates = (
#             expand_hierarchy_context(
#                 collection_name=(
#                     collection_name
#                 ),
#                 candidates=candidates,
#                 max_candidates=(
#                     DEFAULT_MAX_EXPANDED_CANDIDATES
#                 ),
#             )
#         )

#         # Apply hierarchy scoring again because expanded candidates need
#         # hierarchy scores as well.

#         candidates = (
#             apply_hierarchy_boost(
#                 query=query,
#                 candidates=candidates,
#             )
#         )

#     # ========================================================================
#     # 11. CROSS-ENCODER RERANKING
#     # ========================================================================

#     if rerank_results:

#         candidates = rerank(
#             query=query,
#             candidates=candidates,
#             top_k=final_k,
#         )

#     else:

#         candidates = sorted(
#             candidates,
#             key=lambda candidate:
#                 candidate.get(
#                     "hybrid_score",
#                     0.0,
#                 ),
#             reverse=True,
#         )[:final_k]

#     # ========================================================================
#     # 12. RETURN FINAL RESULTS
#     # ========================================================================

#     return candidates


# # """
# # Hybrid (dense + lexical) retrieval with metadata filtering and cross-encoder
# # reranking.

# # This is what HybridSearchAgent calls.

# # Architecture
# # -------------
# # Query
# #   |
# #   +------------------------+
# #   |                        |
# #   v                        v
# # Dense retrieval       Legal BM25 retrieval
# # Chroma ANN             in-memory lexical index
# #   |                        |
# #   +-----------+------------+
# #               |
# #               v
# #        Score normalization
# #               |
# #               v
# #        Convex / RRF fusion
# #               |
# #               v
# #           Deduplication
# #               |
# #               v
# #       Cross-encoder reranker
# #               |
# #               v
# #           Top-k results


# # Why hybrid instead of dense-only
# # --------------------------------
# # - Dense embeddings are strong at semantic/paraphrase matching.
# # - BM25 is strong at exact lexical matching.
# # - Legal contracts contain many repeated generic words, so the lexical
# #   tokenizer intentionally removes a conservative set of generic English
# #   stopwords.
# # - Legal phrases such as "governing law" and "effective date" receive a
# #   small phrase-level boost.
# # - A cross-encoder reranker then evaluates the fused candidate set jointly.

# # BM25 implementation
# # -------------------
# # This module implements BM25 directly rather than depending on rank_bm25.

# # The scoring structure follows the standard Okapi BM25 formulation:

# #     IDF(t) =
# #         log(1 + (N - df(t) + 0.5) / (df(t) + 0.5))

# #     score(D,Q) =
# #         sum_t IDF(t)
# #         * tf(t,D) * (k1 + 1)
# #         / (tf(t,D) + k1 * (1 - b + b * |D| / avgdl))

# # Important:
# # - The implementation uses positive Robertson-style IDF.
# # - Generic English stopwords are removed before indexing/querying.
# # - Legal terms such as "agreement", "party", "shall", "termination",
# #   etc. are intentionally retained.
# # - Phrase boosting is applied after BM25 scoring.

# # Known limitation
# # ----------------
# # The BM25 index is still in-memory and cached per Chroma collection.

# # That is acceptable for a demo/small evaluation corpus but is not ideal
# # for a very large production corpus because:
# # - all Chroma documents are loaded into application memory;
# # - rebuilding the index has startup/cold-start cost;
# # - cache invalidation must happen after ingestion.

# # For production scale, a native sparse+dense backend such as Qdrant,
# # Milvus, or Weaviate would be preferable.

# # The public hybrid_search() API intentionally remains unchanged so the
# # backend can be swapped later without changing the LangGraph agent.
# # """

# # from __future__ import annotations

# # import math
# # import re
# # from collections import Counter
# # from dataclasses import dataclass
# # from typing import Optional


# # from ..resources import (
# #     get_chroma_collection,
# #     get_embedder,
# #     get_reranker,
# # )
# # from ..tracing import traceable


# # # ============================================================================
# # # Tokenization
# # # ============================================================================

# # _TOKEN_RE = re.compile(r"[a-z0-9]+")


# # def _tokenize_raw(text: str) -> list[str]:
# #     """
# #     Raw tokenizer.

# #     This preserves the original project's tokenization behavior:

# #         re.findall(r"[a-z0-9]+", text.lower())

# #     It is useful for debugging / compatibility experiments.
# #     """
# #     return _TOKEN_RE.findall(text.lower())


# # # ---------------------------------------------------------------------------
# # # Conservative legal stopword list
# # #
# # # Important:
# # # We intentionally DO NOT remove legal-domain words such as:
# # #
# # #   agreement
# # #   party
# # #   parties
# # #   shall
# # #   may
# # #   termination
# # #   effective
# # #   date
# # #   law
# # #   governed
# # #   liability
# # #   indemnity
# # #
# # # Those words can be highly informative in legal retrieval.
# # # ---------------------------------------------------------------------------

# # LEGAL_STOPWORDS = {
# #     # articles
# #     "a",
# #     "an",
# #     "the",

# #     # common verbs / auxiliaries
# #     "is",
# #     "are",
# #     "was",
# #     "were",
# #     "be",
# #     "been",
# #     "being",
# #     "has",
# #     "have",
# #     "had",
# #     "do",
# #     "does",
# #     "did",
# #     "can",
# #     "could",
# #     "would",
# #     "should",
# #     "will",

# #     # common prepositions
# #     "of",
# #     "to",
# #     "in",
# #     "on",
# #     "for",
# #     "from",
# #     "by",
# #     "with",
# #     "at",
# #     "into",
# #     "about",
# #     "over",
# #     "under",

# #     # common conjunctions
# #     "and",
# #     "or",
# #     "but",
# #     "if",
# #     "then",
# #     "than",

# #     # common demonstratives / pronouns
# #     "it",
# #     "its",
# #     "this",
# #     "that",
# #     "these",
# #     "those",

# #     # common question words
# #     "what",
# #     "which",
# #     "when",
# #     "where",
# #     "who",
# #     "how",
# # }


# # def _tokenize(text: str) -> list[str]:
# #     """
# #     Production legal-aware tokenizer.

# #     Generic English stopwords are removed because they occur in almost every
# #     legal contract chunk and therefore provide little retrieval signal.

# #     Example:

# #         What is the governing law of this agreement?

# #     becomes approximately:

# #         ["governing", "law", "agreement"]

# #     rather than:

# #         ["what", "is", "the", "governing", "law", "of", "this", "agreement"]
# #     """
# #     tokens = _tokenize_raw(text)

# #     return [
# #         token
# #         for token in tokens
# #         if token not in LEGAL_STOPWORDS
# #     ]


# # # ============================================================================
# # # Legal phrase handling
# # # ============================================================================

# # LEGAL_PHRASES = (
# #     "governing law",
# #     "effective date",
# #     "expiration date",
# #     "termination date",
# #     "termination agreement",
# #     "confidential information",
# #     "intellectual property",
# #     "limitation of liability",
# #     "liability limitation",
# #     "indemnification",
# #     "indemnity",
# #     "force majeure",
# #     "change of control",
# #     "notice period",
# #     "notice provision",
# #     "non compete",
# #     "non solicitation",
# #     "assignment",
# #     "representations and warranties",
# #     "representation and warranty",
# #     "warranty",
# #     "warranties",
# # )


# # def _extract_query_phrases(query: str) -> list[str]:
# #     """
# #     Extract known legal phrases appearing in the query.

# #     Example:

# #         What is the governing law of this agreement?

# #     ->

# #         ["governing law"]
# #     """
# #     normalized = re.sub(
# #         r"\s+",
# #         " ",
# #         query.lower(),
# #     ).strip()

# #     return [
# #         phrase
# #         for phrase in LEGAL_PHRASES
# #         if phrase in normalized
# #     ]


# # def _apply_phrase_boost(
# #     scores: list[float],
# #     texts: list[str],
# #     phrases: list[str],
# #     phrase_bonus: float = 2.0,
# # ) -> list[float]:
# #     """
# #     Add a small lexical bonus when an exact legal phrase occurs in a chunk.

# #     This is intentionally a modest bonus rather than a replacement for BM25.
# #     BM25 remains the primary lexical scoring mechanism.
# #     """
# #     if not phrases:
# #         return scores

# #     boosted_scores = scores.copy()

# #     for index, text in enumerate(texts):
# #         normalized_text = re.sub(
# #             r"\s+",
# #             " ",
# #             text.lower(),
# #         )

# #         for phrase in phrases:
# #             if phrase in normalized_text:
# #                 boosted_scores[index] += phrase_bonus

# #     return boosted_scores


# # # ============================================================================
# # # BM25 implementation
# # # ============================================================================

# # @dataclass
# # class LegalBM25:
# #     """
# #     Lightweight Okapi BM25 implementation.

# #     Parameters
# #     ----------
# #     tokenized_corpus:
# #         List of tokenized documents.

# #     k1:
# #         Term-frequency saturation parameter.

# #     b:
# #         Document-length normalization parameter.

# #     phrase_bonus:
# #         Stored here for configuration visibility. Actual phrase boosting is
# #         applied separately by _apply_phrase_boost().
# #     """

# #     tokenized_corpus: list[list[str]]

# #     k1: float = 1.5
# #     b: float = 0.75
# #     phrase_bonus: float = 2.0

# #     def __post_init__(self) -> None:
# #         self.N = len(self.tokenized_corpus)

# #         self.doc_len: list[int] = [
# #             len(tokens)
# #             for tokens in self.tokenized_corpus
# #         ]

# #         self.avgdl = (
# #             sum(self.doc_len) / self.N
# #             if self.N > 0
# #             else 0.0
# #         )

# #         # Term frequencies for each document.
# #         self.doc_freqs: list[Counter[str]] = []

# #         # Document frequency:
# #         # number of documents containing the term.
# #         self.df: Counter[str] = Counter()

# #         for tokens in self.tokenized_corpus:

# #             frequencies = Counter(tokens)

# #             self.doc_freqs.append(frequencies)

# #             # Count each term ONCE per document.
# #             for term in frequencies:
# #                 self.df[term] += 1

# #         # Positive Robertson-style IDF.
# #         #
# #         # This avoids the negative-IDF behavior you observed with rank_bm25
# #         # for very common words.
# #         self.idf: dict[str, float] = {}

# #         for term, df in self.df.items():

# #             self.idf[term] = math.log(
# #                 1.0
# #                 + (
# #                     self.N
# #                     - df
# #                     + 0.5
# #                 )
# #                 / (
# #                     df
# #                     + 0.5
# #                 )
# #             )

# #     def get_scores(
# #         self,
# #         query_tokens: list[str],
# #     ) -> list[float]:
# #         """
# #         Return one BM25 score per corpus document.
# #         """

# #         if not self.tokenized_corpus:
# #             return []

# #         scores = [0.0] * self.N

# #         for term in query_tokens:

# #             # Ignore query terms not present in corpus.
# #             if term not in self.idf:
# #                 continue

# #             idf = self.idf[term]

# #             for doc_index, frequencies in enumerate(
# #                 self.doc_freqs
# #             ):

# #                 tf = frequencies.get(term, 0)

# #                 if tf == 0:
# #                     continue

# #                 doc_length = self.doc_len[doc_index]

# #                 if self.avgdl == 0:
# #                     length_normalization = 1.0
# #                 else:
# #                     length_normalization = (
# #                         1.0
# #                         - self.b
# #                         + self.b
# #                         * doc_length
# #                         / self.avgdl
# #                     )

# #                 denominator = (
# #                     tf
# #                     + self.k1
# #                     * length_normalization
# #                 )

# #                 contribution = (
# #                     idf
# #                     * tf
# #                     * (self.k1 + 1.0)
# #                     / denominator
# #                 )

# #                 scores[doc_index] += contribution

# #         return scores


# # # ============================================================================
# # # BM25 index cache
# # # ============================================================================

# # _bm25_cache: dict[
# #     str,
# #     tuple[
# #         LegalBM25,
# #         list[str],
# #         list[str],
# #         list[dict],
# #     ],
# # ] = {}


# # def _get_bm25_index(
# #     collection_name: str,
# #     refresh: bool = False,
# # ):
# #     """
# #     Build or retrieve the cached BM25 index for a Chroma collection.
# #     """

# #     if (
# #         not refresh
# #         and collection_name in _bm25_cache
# #     ):
# #         return _bm25_cache[collection_name]

# #     collection = get_chroma_collection(
# #         collection_name
# #     )

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

# #     bm25 = LegalBM25(
# #         tokenized_corpus=tokenized_documents,
# #         k1=1.5,
# #         b=0.75,
# #         phrase_bonus=2.0,
# #     )

# #     _bm25_cache[collection_name] = (
# #         bm25,
# #         ids,
# #         texts,
# #         metadatas,
# #     )

# #     return _bm25_cache[collection_name]


# # def invalidate_bm25_cache(
# #     collection_name: str,
# # ) -> None:
# #     """
# #     Invalidate the BM25 cache after new documents are ingested.

# #     Call this immediately after embed_and_store().

# #     Example:

# #         embed_and_store(...)
# #         invalidate_bm25_cache(collection_name)
# #     """

# #     _bm25_cache.pop(
# #         collection_name,
# #         None,
# #     )


# # # ============================================================================
# # # Metadata filtering
# # # ============================================================================

# # def build_where_clause(
# #     filters: dict,
# # ) -> Optional[dict]:
# #     """
# #     Convert application-level metadata filters into Chroma's `where` clause.
# #     """

# #     clauses: list[dict] = []

# #     if filters.get("contract_type"):
# #         clauses.append(
# #             {
# #                 "contract_type":
# #                     filters["contract_type"]
# #             }
# #         )

# #     if filters.get(
# #         "governing_law_country"
# #     ):
# #         clauses.append(
# #             {
# #                 "governing_law_country":
# #                     filters[
# #                         "governing_law_country"
# #                     ].upper()
# #             }
# #         )

# #     if (
# #         filters.get(
# #             "min_effective_date_epoch"
# #         )
# #         is not None
# #     ):
# #         clauses.append(
# #             {
# #                 "effective_date_epoch": {
# #                     "$gte":
# #                         filters[
# #                             "min_effective_date_epoch"
# #                         ]
# #                 }
# #             }
# #         )

# #     if (
# #         filters.get(
# #             "max_effective_date_epoch"
# #         )
# #         is not None
# #     ):
# #         clauses.append(
# #             {
# #                 "effective_date_epoch": {
# #                     "$lte":
# #                         filters[
# #                             "max_effective_date_epoch"
# #                         ]
# #                 }
# #             }
# #         )

# #     if (
# #         filters.get(
# #             "min_monetary_value"
# #         )
# #         is not None
# #     ):
# #         clauses.append(
# #             {
# #                 "monetary_value": {
# #                     "$gte":
# #                         filters[
# #                             "min_monetary_value"
# #                         ]
# #                 }
# #             }
# #         )

# #     if (
# #         filters.get(
# #             "max_monetary_value"
# #         )
# #         is not None
# #     ):
# #         clauses.append(
# #             {
# #                 "monetary_value": {
# #                     "$lte":
# #                         filters[
# #                             "max_monetary_value"
# #                         ]
# #                 }
# #             }
# #         )

# #     if not clauses:
# #         return None

# #     if len(clauses) == 1:
# #         return clauses[0]

# #     return {
# #         "$and": clauses
# #     }


# # def _matches_where(
# #     meta: dict,
# #     where: Optional[dict],
# # ) -> bool:
# #     """
# #     Python-side equivalent of build_where_clause().

# #     Chroma applies the metadata filter natively for dense retrieval.
# #     BM25 is an in-memory index, so the same filtering must be applied
# #     here in Python.
# #     """

# #     if not where:
# #         return True

# #     if "$and" in where:
# #         return all(
# #             _matches_where(
# #                 meta,
# #                 clause,
# #             )
# #             for clause in where["$and"]
# #         )

# #     (field, condition), = where.items()

# #     value = meta.get(field)

# #     if isinstance(condition, dict):

# #         if (
# #             "$gte" in condition
# #             and not (
# #                 value is not None
# #                 and value >= condition["$gte"]
# #             )
# #         ):
# #             return False

# #         if (
# #             "$lte" in condition
# #             and not (
# #                 value is not None
# #                 and value <= condition["$lte"]
# #             )
# #         ):
# #             return False

# #         return True

# #     return value == condition


# # # ============================================================================
# # # Dense retrieval
# # # ============================================================================

# # def _dense_search(
# #     collection_name: str,
# #     query: str,
# #     top_k: int,
# #     where: Optional[dict],
# # ) -> list[dict]:
# #     """
# #     Dense semantic retrieval using Chroma.

# #     Chroma applies metadata filtering during ANN retrieval.
# #     """

# #     collection = get_chroma_collection(
# #         collection_name
# #     )

# #     embedder = get_embedder()

# #     query_embedding = embedder.encode(
# #         [query],
# #         normalize_embeddings=True,
# #     ).tolist()

# #     results = collection.query(
# #         query_embeddings=query_embedding,
# #         n_results=top_k,
# #         where=where,
# #     )

# #     hits: list[dict] = []

# #     if results["ids"]:

# #         for (
# #             doc_id,
# #             doc,
# #             meta,
# #             dist,
# #         ) in zip(
# #             results["ids"][0],
# #             results["documents"][0],
# #             results["metadatas"][0],
# #             results["distances"][0],
# #         ):

# #             hits.append(
# #                 {
# #                     "id": doc_id,
# #                     "text": doc,
# #                     "metadata": meta,
# #                     "dense_distance": dist,
# #                 }
# #             )

# #     return hits


# # # ============================================================================
# # # Sparse / BM25 retrieval
# # # ============================================================================

# # def _sparse_search(
# #     collection_name: str,
# #     query: str,
# #     top_k: int,
# #     where: Optional[dict],
# # ) -> list[dict]:
# #     """
# #     Legal-aware BM25 retrieval.

# #     Pipeline:

# #         query
# #           |
# #           v
# #         legal tokenization
# #           |
# #           v
# #         BM25 scoring
# #           |
# #           v
# #         legal phrase boosting
# #           |
# #           v
# #         metadata filtering
# #           |
# #           v
# #         top-k
# #     """

# #     (
# #         bm25,
# #         ids,
# #         texts,
# #         metadatas,
# #     ) = _get_bm25_index(
# #         collection_name
# #     )

# #     # ------------------------------------------------------------------
# #     # Query tokenization
# #     # ------------------------------------------------------------------

# #     query_tokens = _tokenize(query)

# #     # ------------------------------------------------------------------
# #     # BM25 scoring
# #     # ------------------------------------------------------------------

# #     scores = bm25.get_scores(
# #         query_tokens
# #     )

# #     # ------------------------------------------------------------------
# #     # Legal phrase boosting
# #     # ------------------------------------------------------------------

# #     query_phrases = _extract_query_phrases(
# #         query
# #     )

# #     scores = _apply_phrase_boost(
# #         scores=scores,
# #         texts=texts,
# #         phrases=query_phrases,
# #         phrase_bonus=bm25.phrase_bonus,
# #     )

# #     # ------------------------------------------------------------------
# #     # Metadata filtering BEFORE top-k selection
# #     # ------------------------------------------------------------------

# #     eligible = [
# #         index
# #         for index in range(len(ids))
# #         if _matches_where(
# #             metadatas[index],
# #             where,
# #         )
# #     ]

# #     eligible_ranked = sorted(
# #         eligible,
# #         key=lambda index: scores[index],
# #         reverse=True,
# #     )[:top_k]

# #     return [
# #         {
# #             "id": ids[index],
# #             "text": texts[index],
# #             "metadata": metadatas[index],
# #             "bm25_score": float(
# #                 scores[index]
# #             ),
# #         }
# #         for index in eligible_ranked
# #     ]


# # # ============================================================================
# # # Fusion strategies
# # # ============================================================================

# # def reciprocal_rank_fusion(
# #     id_lists: list[list[str]],
# #     k: int = 60,
# # ) -> dict[str, float]:
# #     """
# #     Reciprocal Rank Fusion.

# #     score(d) =
# #         sum(
# #             1 / (k + rank)
# #         )

# #     RRF uses only ranking and does not require dense and sparse raw scores
# #     to be numerically comparable.
# #     """

# #     scores: dict[str, float] = {}

# #     for id_list in id_lists:

# #         for rank, doc_id in enumerate(
# #             id_list
# #         ):

# #             scores[doc_id] = (
# #                 scores.get(
# #                     doc_id,
# #                     0.0,
# #                 )
# #                 + 1.0
# #                 / (
# #                     k
# #                     + rank
# #                     + 1
# #                 )
# #             )

# #     return scores


# # def min_max_normalize(
# #     scores: dict[str, float],
# #     reverse: bool = False,
# # ) -> dict[str, float]:
# #     """
# #     Normalize scores independently into [0, 1].

# #     reverse=True means lower raw values are better, which is appropriate
# #     for Chroma distance scores.
# #     """

# #     if not scores:
# #         return {}

# #     values = list(
# #         scores.values()
# #     )

# #     min_val = min(values)
# #     max_val = max(values)

# #     if min_val == max_val:
# #         return {
# #             doc_id: 1.0
# #             for doc_id in scores
# #         }

# #     normalized: dict[str, float] = {}

# #     for doc_id, score in scores.items():

# #         if reverse:
# #             norm_score = (
# #                 max_val - score
# #             ) / (
# #                 max_val - min_val
# #             )
# #         else:
# #             norm_score = (
# #                 score - min_val
# #             ) / (
# #                 max_val - min_val
# #             )

# #         normalized[doc_id] = float(
# #             norm_score
# #         )

# #     return normalized


# # def convex_combination_fusion(
# #     dense_hits: list[dict],
# #     sparse_hits: list[dict],
# #     alpha: float = 0.4,
# #     dense_score_key: str = "dense_distance",
# #     sparse_score_key: str = "bm25_score",
# #     dense_is_distance: bool = True,
# # ) -> dict[str, float]:
# #     """
# #     Combine dense and sparse retrieval scores.

# #     Final Score =
# #         alpha * Dense_Norm
# #         +
# #         (1 - alpha) * Sparse_Norm

# #     alpha = 0.2 means:

# #         20% dense
# #         80% BM25

# #     alpha = 0.4 means:

# #         40% dense
# #         60% BM25

# #     alpha = 0.7 means:

# #         70% dense
# #         30% BM25
# #     """

# #     if not dense_hits and not sparse_hits:
# #         return {}

# #     # ------------------------------------------------------------------
# #     # Raw score mappings
# #     # ------------------------------------------------------------------

# #     raw_dense = {
# #         hit["id"]:
# #             hit[dense_score_key]
# #         for hit in dense_hits
# #     }

# #     raw_sparse = {
# #         hit["id"]:
# #             hit[sparse_score_key]
# #         for hit in sparse_hits
# #     }

# #     # ------------------------------------------------------------------
# #     # Independent normalization
# #     # ------------------------------------------------------------------

# #     norm_dense = min_max_normalize(
# #         raw_dense,
# #         reverse=dense_is_distance,
# #     )

# #     norm_sparse = min_max_normalize(
# #         raw_sparse,
# #         reverse=False,
# #     )

# #     # ------------------------------------------------------------------
# #     # Union of candidate IDs
# #     # ------------------------------------------------------------------

# #     all_doc_ids = (
# #         set(norm_dense.keys())
# #         | set(norm_sparse.keys())
# #     )

# #     fused_scores: dict[str, float] = {}

# #     for doc_id in all_doc_ids:

# #         dense_score = norm_dense.get(
# #             doc_id,
# #             0.0,
# #         )

# #         sparse_score = norm_sparse.get(
# #             doc_id,
# #             0.0,
# #         )

# #         fused_scores[doc_id] = (
# #             alpha * dense_score
# #             + (1.0 - alpha)
# #             * sparse_score
# #         )

# #     # ------------------------------------------------------------------
# #     # Sort descending
# #     # ------------------------------------------------------------------

# #     return dict(
# #         sorted(
# #             fused_scores.items(),
# #             key=lambda item: item[1],
# #             reverse=True,
# #         )
# #     )


# # # ============================================================================
# # # Cross-encoder reranking
# # # ============================================================================

# # def rerank(
# #     query: str,
# #     candidates: list[dict],
# #     top_k: int,
# # ) -> list[dict]:
# #     """
# #     Cross-encoder reranking.

# #     The reranker is loaded through resources.get_reranker() so that the
# #     application continues to use its existing singleton model.
# #     """

# #     if not candidates:
# #         return []

# #     reranker = get_reranker()

# #     pairs = [
# #         (
# #             query,
# #             candidate["text"],
# #         )
# #         for candidate in candidates
# #     ]

# #     scores = reranker.predict(
# #         pairs
# #     )

# #     for candidate, score in zip(
# #         candidates,
# #         scores,
# #     ):
# #         candidate[
# #             "rerank_score"
# #         ] = float(score)

# #     return sorted(
# #         candidates,
# #         key=lambda candidate:
# #             candidate["rerank_score"],
# #         reverse=True,
# #     )[:top_k]


# # # ============================================================================
# # # Deduplication
# # # ============================================================================

# # def _deduplicate_hits(
# #     hits: list[dict],
# # ) -> list[dict]:
# #     """
# #     Remove duplicate underlying chunks.

# #     We intentionally deduplicate using:

# #         (
# #             document_name,
# #             chunk_index,
# #             text
# #         )

# #     rather than Chroma's UUID.

# #     This catches the case where the same chunk was ingested more than once
# #     under different Chroma IDs.

# #     The first occurrence wins because candidates arrive already ordered by
# #     fused relevance.
# #     """

# #     unique_hits: list[dict] = []

# #     seen: set[
# #         tuple[
# #             object,
# #             object,
# #             str,
# #         ]
# #     ] = set()

# #     for hit in hits:

# #         metadata = hit.get(
# #             "metadata",
# #             {},
# #         )

# #         key = (
# #             metadata.get(
# #                 "document_name"
# #             ),
# #             metadata.get(
# #                 "chunk_index"
# #             ),
# #             (
# #                 hit.get(
# #                     "text",
# #                     ""
# #                 )
# #                 or ""
# #             ).strip(),
# #         )

# #         if key in seen:
# #             continue

# #         seen.add(key)
# #         unique_hits.append(hit)

# #     return unique_hits


# # # ============================================================================
# # # Public entrypoint
# # # ============================================================================

# # @traceable(
# #     name="hybrid_search_agent.search",
# #     run_type="retriever",
# # )
# # def hybrid_search(
# #     collection_name: str,
# #     query: str,
# #     metadata_filter: Optional[dict] = None,
# #     dense_k: int = 20,
# #     sparse_k: int = 20,
# #     fusion_k: int = 15,
# #     final_k: int = 6,
# #     fusion_method: str = "convex",
# #     alpha: float = 0.4,
# # ) -> list[dict]:
# #     """
# #     Full retrieval pipeline.

# #         Dense search
# #               +
# #         Legal BM25 search
# #               |
# #               v
# #         Dense/BM25 fusion
# #               |
# #               v
# #         Candidate selection
# #               |
# #               v
# #         Deduplication
# #               |
# #               v
# #         parties_contains filtering
# #               |
# #               v
# #         Cross-encoder reranking
# #               |
# #               v
# #         final_k results

# #     Parameters
# #     ----------
# #     collection_name:
# #         Chroma collection to search.

# #     query:
# #         User question.

# #     metadata_filter:
# #         Optional contract-level metadata filters.

# #     dense_k:
# #         Number of dense candidates.

# #     sparse_k:
# #         Number of BM25 candidates.

# #     fusion_k:
# #         Number of fused candidates passed forward.

# #     final_k:
# #         Number of reranked results returned.

# #     fusion_method:
# #         "convex" or "rrf".

# #     alpha:
# #         Dense weight for convex fusion.

# #         0.2 = lexical-heavy
# #         0.4 = balanced/legal lexical-heavy
# #         0.7 = semantic-heavy

# #     Returns
# #     -------
# #     list[dict]

# #     Each result contains fields such as:

# #         id
# #         text
# #         metadata
# #         dense_distance
# #         bm25_score
# #         rerank_score
# #     """

# #     metadata_filter = (
# #         metadata_filter
# #         or {}
# #     )

# #     # ------------------------------------------------------------------
# #     # Convert application filters into Chroma/Python predicates
# #     # ------------------------------------------------------------------

# #     where = build_where_clause(
# #         metadata_filter
# #     )

# #     # ------------------------------------------------------------------
# #     # Dense retrieval
# #     # ------------------------------------------------------------------

# #     dense_hits = _dense_search(
# #         collection_name=collection_name,
# #         query=query,
# #         top_k=dense_k,
# #         where=where,
# #     )

# #     # ------------------------------------------------------------------
# #     # Sparse / BM25 retrieval
# #     # ------------------------------------------------------------------

# #     sparse_hits = _sparse_search(
# #         collection_name=collection_name,
# #         query=query,
# #         top_k=sparse_k,
# #         where=where,
# #     )

# #     # ------------------------------------------------------------------
# #     # Merge dense + sparse hits by Chroma ID
# #     # ------------------------------------------------------------------

# #     by_id: dict[
# #         str,
# #         dict,
# #     ] = {}

# #     for hit in (
# #         dense_hits
# #         + sparse_hits
# #     ):

# #         if hit["id"] not in by_id:
# #             by_id[
# #                 hit["id"]
# #             ] = {}

# #         by_id[
# #             hit["id"]
# #         ].update(hit)

# #     # ------------------------------------------------------------------
# #     # Fusion
# #     # ------------------------------------------------------------------

# #     if fusion_method.lower() == "rrf":

# #         fused_scores = (
# #             reciprocal_rank_fusion(
# #                 [
# #                     [
# #                         hit["id"]
# #                         for hit in dense_hits
# #                     ],
# #                     [
# #                         hit["id"]
# #                         for hit in sparse_hits
# #                     ],
# #                 ]
# #             )
# #         )

# #     else:

# #         fused_scores = (
# #             convex_combination_fusion(
# #                 dense_hits=dense_hits,
# #                 sparse_hits=sparse_hits,
# #                 alpha=alpha,
# #                 dense_is_distance=True,
# #             )
# #         )

# #     # ------------------------------------------------------------------
# #     # Keep only top fusion_k candidates
# #     # ------------------------------------------------------------------

# #     fused_ranked_ids = list(
# #         fused_scores.keys()
# #     )[:fusion_k]

# #     candidates = [
# #         by_id[doc_id]
# #         for doc_id in fused_ranked_ids
# #         if doc_id in by_id
# #     ]

# #     # ------------------------------------------------------------------
# #     # Deduplicate underlying chunks
# #     # ------------------------------------------------------------------

# #     candidates = _deduplicate_hits(
# #         candidates
# #     )

# #     # ------------------------------------------------------------------
# #     # parties_contains is intentionally handled here because it is not
# #     # represented by build_where_clause().
# #     # ------------------------------------------------------------------

# #     parties_needle = (
# #         metadata_filter.get(
# #             "parties_contains"
# #         )
# #     )

# #     if parties_needle:

# #         needle = (
# #             parties_needle
# #             .lower()
# #             .strip()
# #         )

# #         candidates = [
# #             candidate
# #             for candidate in candidates
# #             if needle
# #             in (
# #                 candidate[
# #                     "metadata"
# #                 ].get(
# #                     "parties",
# #                     "",
# #                 )
# #                 or ""
# #             ).lower()
# #         ]

# #     # ------------------------------------------------------------------
# #     # Cross-encoder reranking
# #     # ------------------------------------------------------------------

# #     return rerank(
# #         query=query,
# #         candidates=candidates,
# #         top_k=final_k,
# #     )
# # # #==================
# # # """
# # # Hybrid (dense + lexical) retrieval, with metadata filtering and cross-encoder
# # # reranking. This is what HybridSearchAgent calls.

# # # Why hybrid instead of dense-only:
# # #   - Dense embeddings are strong at semantic/paraphrase matching ("termination
# # #     for convenience" ~ "either party may end this agreement without cause")
# # #     but weak at exact string matching — a specific clause number, a defined
# # #     term used verbatim, a citation.
# # #   - BM25 (lexical, term-frequency based) is the opposite: strong at exact
# # #     term matching, weak at paraphrase.
# # #   - The two branches' raw scores are fused with a weighted convex
# # #     combination after independently min-max normalizing each to [0, 1] —
# # #     see convex_combination_fusion() below for why, over rank-only fusion
# # #     (RRF, kept here as an alternate strategy).
# # #   - A cross-encoder reranker then scores each (query, candidate) pair
# # #     *jointly* — much more accurate than comparing two independently-encoded
# # #     vectors — but it's too slow to run against an entire collection, so it
# # #     only reranks the ~15-20 candidates that already survived the cheap
# # #     dense+sparse fusion stage.

# # # KNOWN LIMITATION — in-memory BM25 (see _get_bm25_index below):
# # #   This module builds and caches a BM25 index by pulling every document out
# # #   of the Chroma collection into memory. That's fine for a demo/small corpus,
# # #   but it means high memory use and cold-start latency on large collections,
# # #   and the index only refreshes when explicitly invalidated (see
# # #   invalidate_bm25_cache), not automatically on ingestion. The correct
# # #   production fix is to move to a vector database with NATIVE sparse+dense
# # #   hybrid indexing and native filtered lexical search — e.g. Qdrant, Milvus,
# # #   or Weaviate — which eliminates client-side index building entirely and
# # #   lets the database engine apply metadata filters during scoring rather than
# # #   after it. That's a backend swap, not a change to this module's public
# # #   hybrid_search() signature: _dense_search/_sparse_search/_get_bm25_index
# # #   are the only functions that would need to be replaced with native
# # #   filtered-hybrid-query calls against the new store.
# # # """

# # # from __future__ import annotations

# # # import re
# # # from typing import Optional

# # # from rank_bm25 import BM25Okapi

# # # from ..resources import get_chroma_collection, get_embedder, get_reranker
# # # from ..tracing import traceable

# # # # ---------------------------------------------------------------------------
# # # # BM25 index (in-memory, cached per collection)
# # # # ---------------------------------------------------------------------------

# # # _bm25_cache: dict[str, tuple[BM25Okapi, list[str], list[str], list[dict]]] = {}


# # # def _tokenize(text: str) -> list[str]:
# # #     return re.findall(r"[a-z0-9]+", text.lower())


# # # def _get_bm25_index(collection_name: str, refresh: bool = False):
# # #     if not refresh and collection_name in _bm25_cache:
# # #         return _bm25_cache[collection_name]

# # #     collection = get_chroma_collection(collection_name)
# # #     raw = collection.get(include=["documents", "metadatas"])
# # #     ids, texts, metadatas = raw["ids"], raw["documents"], raw["metadatas"]
# # #     bm25 = BM25Okapi([_tokenize(t) for t in texts])

# # #     _bm25_cache[collection_name] = (bm25, ids, texts, metadatas)
# # #     return _bm25_cache[collection_name]


# # # def invalidate_bm25_cache(collection_name: str) -> None:
# # #     """
# # #     Call this right after ingesting new documents into `collection_name`
# # #     (e.g. from scripts/run_demo.py, right after embed_and_store()) so the
# # #     next hybrid_search() call rebuilds the BM25 index instead of silently
# # #     searching a stale one that doesn't include the documents you just added.
# # #     Not wired in automatically from pdf_pipeline.embed_and_store() itself,
# # #     to avoid a circular import between ingestion and retrieval — call it
# # #     explicitly from whichever orchestration layer just finished ingesting.
# # #     """
# # #     _bm25_cache.pop(collection_name, None)


# # # # ---------------------------------------------------------------------------
# # # # Metadata filter -> Chroma `where` clause, and a matching Python-side
# # # # predicate for the BM25 branch (which Chroma's `where` can't reach).
# # # # ---------------------------------------------------------------------------

# # # def build_where_clause(filters: dict) -> Optional[dict]:
# # #     clauses: list[dict] = []
# # #     if filters.get("contract_type"):
# # #         clauses.append({"contract_type": filters["contract_type"]})
# # #     if filters.get("governing_law_country"):
# # #         clauses.append({"governing_law_country": filters["governing_law_country"].upper()})
# # #     if filters.get("min_effective_date_epoch") is not None:
# # #         clauses.append({"effective_date_epoch": {"$gte": filters["min_effective_date_epoch"]}})
# # #     if filters.get("max_effective_date_epoch") is not None:
# # #         clauses.append({"effective_date_epoch": {"$lte": filters["max_effective_date_epoch"]}})
# # #     if filters.get("min_monetary_value") is not None:
# # #         clauses.append({"monetary_value": {"$gte": filters["min_monetary_value"]}})
# # #     if filters.get("max_monetary_value") is not None:
# # #         clauses.append({"monetary_value": {"$lte": filters["max_monetary_value"]}})

# # #     if not clauses:
# # #         return None
# # #     return clauses[0] if len(clauses) == 1 else {"$and": clauses}


# # # def _matches_where(meta: dict, where: Optional[dict]) -> bool:
# # #     """Python-side equivalent of build_where_clause(), for filtering BM25 candidates."""
# # #     if not where:
# # #         return True
# # #     if "$and" in where:
# # #         return all(_matches_where(meta, clause) for clause in where["$and"])
# # #     (field, condition), = where.items()
# # #     value = meta.get(field)
# # #     if isinstance(condition, dict):
# # #         if "$gte" in condition and not (value is not None and value >= condition["$gte"]):
# # #             return False
# # #         if "$lte" in condition and not (value is not None and value <= condition["$lte"]):
# # #             return False
# # #         return True
# # #     return value == condition


# # # # ---------------------------------------------------------------------------
# # # # Dense and sparse branches
# # # # ---------------------------------------------------------------------------

# # # def _dense_search(collection_name: str, query: str, top_k: int, where: Optional[dict]) -> list[dict]:
# # #     """Chroma applies `where` natively DURING the ANN search — filtering here was never the issue."""
# # #     collection = get_chroma_collection(collection_name)
# # #     embedder = get_embedder()
# # #     query_embedding = embedder.encode([query], normalize_embeddings=True).tolist()
# # #     results = collection.query(query_embeddings=query_embedding, n_results=top_k, where=where)

# # #     hits = []
# # #     if results["ids"]:
# # #         for doc_id, doc, meta, dist in zip(
# # #             results["ids"][0], results["documents"][0], results["metadatas"][0], results["distances"][0]
# # #         ):
# # #             hits.append({"id": doc_id, "text": doc, "metadata": meta, "dense_distance": dist})
# # #     return hits


# # # def _sparse_search(collection_name: str, query: str, top_k: int, where: Optional[dict]) -> list[dict]:
# # #     """
# # #     Filter FIRST (restrict to the set of documents that satisfy `where`),
# # #     THEN rank that eligible set by BM25 score — not the other way around.
# # #     Ranking the whole corpus first and filtering afterward risks returning
# # #     fewer than top_k results (or zero) under a restrictive filter even when
# # #     plenty of matching-and-relevant documents exist further down the ranked
# # #     list; filtering the candidate pool before selecting top_k guarantees we
# # #     never miss an eligible match because it happened to rank outside some
# # #     arbitrary pre-filter cutoff.
# # #     """
# # #     bm25, ids, texts, metadatas = _get_bm25_index(collection_name)
# # #     scores = bm25.get_scores(_tokenize(query))

# # #     eligible = [i for i in range(len(ids)) if _matches_where(metadatas[i], where)]
# # #     eligible_ranked = sorted(eligible, key=lambda i: scores[i], reverse=True)[:top_k]

# # #     return [
# # #         {"id": ids[i], "text": texts[i], "metadata": metadatas[i], "bm25_score": float(scores[i])}
# # #         for i in eligible_ranked
# # #     ]


# # # # ---------------------------------------------------------------------------
# # # # Fusion strategies
# # # # ---------------------------------------------------------------------------

# # # def reciprocal_rank_fusion(id_lists: list[list[str]], k: int = 60) -> dict[str, float]:
# # #     """
# # #     Rank-only fusion: score(d) = sum, over lists containing d, of
# # #     1 / (k + rank_in_that_list). Doesn't need the two branches' raw scores
# # #     to be comparable at all — only their relative ordering — which makes it
# # #     a safe default when you don't want to tune a blend weight. Kept as an
# # #     alternate to convex_combination_fusion (see hybrid_search()'s
# # #     fusion_method parameter).
# # #     """
# # #     scores: dict[str, float] = {}
# # #     for id_list in id_lists:
# # #         for rank, doc_id in enumerate(id_list):
# # #             scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
# # #     return scores


# # # def min_max_normalize(scores: dict[str, float], reverse: bool = False) -> dict[str, float]:
# # #     """Normalizes scores to the [0, 1] range using Min-Max scaling.

# # #     Args:
# # #         scores: Dictionary mapping document IDs to raw scores.
# # #         reverse: Set to True if lower raw score is better (e.g. L2 or Cosine
# # #           Distance).

# # #     Returns:
# # #         Dictionary mapping document IDs to normalized scores in [0, 1].
# # #     """
# # #     if not scores:
# # #         return {}
# # #     vals = list(scores.values())
# # #     min_val, max_val = min(vals), max(vals)
# # #     # Edge case: all candidates have the identical raw score.
# # #     if min_val == max_val:
# # #         return {doc_id: 1.0 for doc_id in scores}
# # #     normalized = {}
# # #     for doc_id, score in scores.items():
# # #         if reverse:
# # #             # For distance metrics where lower is better.
# # #             norm_score = (max_val - score) / (max_val - min_val)
# # #         else:
# # #             # For similarity/BM25 metrics where higher is better.
# # #             norm_score = (score - min_val) / (max_val - min_val)
# # #         normalized[doc_id] = float(norm_score)
# # #     return normalized


# # # def convex_combination_fusion(
# # #     dense_hits: list[dict],
# # #     sparse_hits: list[dict],
# # #     alpha: float = 0.4,
# # #     dense_score_key: str = "dense_distance",
# # #     sparse_score_key: str = "bm25_score",
# # #     dense_is_distance: bool = True,
# # # ) -> dict[str, float]:
# # #     """Combines dense and sparse results using a weighted convex combination.

# # #     Final Score = alpha * Dense_Norm + (1 - alpha) * Sparse_Norm

# # #     Default alpha=0.4 weights lexical (BM25) matching more heavily than
# # #     dense similarity (40% dense / 60% sparse) — legal text leans on exact
# # #     clause wording, defined terms, and citations, where lexical match is
# # #     often the stronger signal; raise alpha toward 1.0 if your queries skew
# # #     more paraphrase/semantic than exact-term.

# # #     Args:
# # #         dense_hits: List of dicts, e.g. [{"id": "doc1", "dense_distance": 0.12}]
# # #         sparse_hits: List of dicts, e.g. [{"id": "doc1", "bm25_score": 14.2}]
# # #         alpha: Weight for dense retrieval (0.0 to 1.0). High alpha = favor
# # #           dense.
# # #         dense_score_key: Dictionary key containing the dense metric.
# # #         sparse_score_key: Dictionary key containing the sparse metric.
# # #         dense_is_distance: Set True if dense score is distance (lower =
# # #           better).

# # #     Returns:
# # #         Dictionary mapping doc IDs to their combined score, sorted descending.
# # #     """
# # #     # 1. Extract raw scores into mappings.
# # #     raw_dense = {h["id"]: h[dense_score_key] for h in dense_hits}
# # #     raw_sparse = {h["id"]: h[sparse_score_key] for h in sparse_hits}

# # #     # 2. Normalize both score distributions independently to [0, 1].
# # #     norm_dense = min_max_normalize(raw_dense, reverse=dense_is_distance)
# # #     norm_sparse = min_max_normalize(raw_sparse, reverse=False)

# # #     # 3. Combine scores across the union of all candidate document IDs.
# # #     all_doc_ids = set(norm_dense.keys()).union(set(norm_sparse.keys()))
# # #     fused_scores = {}
# # #     for doc_id in all_doc_ids:
# # #         # If a document is missing in one branch, treat its normalized score as 0.0.
# # #         d_score = norm_dense.get(doc_id, 0.0)
# # #         s_score = norm_sparse.get(doc_id, 0.0)
# # #         fused_scores[doc_id] = (alpha * d_score) + ((1.0 - alpha) * s_score)

# # #     # Sort candidates by fused score descending.
# # #     return dict(sorted(fused_scores.items(), key=lambda item: item[1], reverse=True))


# # # def rerank(query: str, candidates: list[dict], top_k: int) -> list[dict]:
# # #     """
# # #     Cross-encoder rerank of the fused candidate set. The reranker itself is
# # #     a module-level singleton loaded via resources.get_reranker() — NOT
# # #     instantiated here — see resources.preload_models() to load it eagerly
# # #     at application startup instead of lazily on the first call.
# # #     """
# # #     if not candidates:
# # #         return []
# # #     reranker = get_reranker()
# # #     pairs = [(query, c["text"]) for c in candidates]
# # #     scores = reranker.predict(pairs)
# # #     for c, s in zip(candidates, scores):
# # #         c["rerank_score"] = float(s)
# # #     return sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)[:top_k]


# # # # ---------------------------------------------------------------------------
# # # # Public entrypoint
# # # # ---------------------------------------------------------------------------

# # # def _deduplicate_hits(hits: list[dict]) -> list[dict]:
# # #     """
# # #     Collapses duplicate chunks that made it into the candidate set as
# # #     separate entries. This happens when the same underlying text exists in
# # #     Chroma under more than one id (e.g. a document re-ingested without
# # #     first removing the prior copy) — the by-id merge in hybrid_search()
# # #     can't catch this, since it dedupes on Chroma id, and these are
# # #     genuinely different ids that happen to hold identical content.
# # #     Deduplicating by (document_name, chunk_index, text) instead catches it.
# # #     Order is preserved, so the first (highest fused-score) occurrence of
# # #     each duplicate is what's kept.
# # #     """
# # #     unique_hits = []
# # #     seen = set()
# # #     for hit in hits:
# # #         key = (
# # #             hit["metadata"].get("document_name"),
# # #             hit["metadata"].get("chunk_index"),
# # #             hit["text"].strip(),
# # #         )
# # #         if key not in seen:
# # #             seen.add(key)
# # #             unique_hits.append(hit)
# # #     return unique_hits


# # # @traceable(name="hybrid_search_agent.search", run_type="retriever")
# # # def hybrid_search(
# # #     collection_name: str,
# # #     query: str,
# # #     metadata_filter: Optional[dict] = None,
# # #     dense_k: int = 20,
# # #     sparse_k: int = 20,
# # #     fusion_k: int = 15,
# # #     final_k: int = 6,
# # #     fusion_method: str = "convex",
# # #     alpha: float = 0.4,
# # # ) -> list[dict]:
# # #     """
# # #     Full pipeline: dense search + BM25 search (each metadata-filtered) ->
# # #     fusion ("convex" weighted combination by default, or "rrf" for
# # #     rank-only fusion) -> parties_contains post-filter -> cross-encoder
# # #     rerank -> top final_k results, each:
# # #     {"id", "text", "metadata", "rerank_score", ...}.
# # #     """
# # #     metadata_filter = metadata_filter or {}
# # #     where = build_where_clause(metadata_filter)

# # #     dense_hits = _dense_search(collection_name, query, dense_k, where)
# # #     sparse_hits = _sparse_search(collection_name, query, sparse_k, where)

# # #     by_id: dict[str, dict] = {}
# # #     for h in dense_hits + sparse_hits:
# # #         by_id.setdefault(h["id"], {}).update(h)

# # #     if fusion_method == "rrf":
# # #         fused_scores = reciprocal_rank_fusion(
# # #             [[h["id"] for h in dense_hits], [h["id"] for h in sparse_hits]]
# # #         )
# # #     else:
# # #         fused_scores = convex_combination_fusion(
# # #             dense_hits=dense_hits,
# # #             sparse_hits=sparse_hits,
# # #             alpha=alpha,
# # #             dense_is_distance=True,  # Chroma returns distances where lower = better
# # #         )

# # #     fused_ranked_ids = list(fused_scores.keys())[:fusion_k]
# # #     candidates = [by_id[i] for i in fused_ranked_ids if i in by_id]
# # #     candidates = _deduplicate_hits(candidates)

# # #     parties_needle = metadata_filter.get("parties_contains")
# # #     if parties_needle:
# # #         needle = parties_needle.lower()
# # #         candidates = [c for c in candidates if needle in (c["metadata"].get("parties", "") or "").lower()]

# # #     return rerank(query, candidates, top_k=final_k)

# # # #==============
# # # # """
# # # # Hybrid (dense + lexical) retrieval, with metadata filtering and cross-encoder
# # # # reranking. This is what HybridSearchAgent calls.

# # # # Why hybrid instead of dense-only:
# # # #   - Dense embeddings are strong at semantic/paraphrase matching ("termination
# # # #     for convenience" ~ "either party may end this agreement without cause")
# # # #     but weak at exact string matching — a specific clause number, a defined
# # # #     term used verbatim, a citation.
# # # #   - BM25 (lexical, term-frequency based) is the opposite: strong at exact
# # # #     term matching, weak at paraphrase.
# # # #   - The two branches' raw scores are fused with a weighted convex
# # # #     combination after independently min-max normalizing each to [0, 1] —
# # # #     see convex_combination_fusion() below for why, over rank-only fusion
# # # #     (RRF, kept here as an alternate strategy).
# # # #   - A cross-encoder reranker then scores each (query, candidate) pair
# # # #     *jointly* — much more accurate than comparing two independently-encoded
# # # #     vectors — but it's too slow to run against an entire collection, so it
# # # #     only reranks the ~15-20 candidates that already survived the cheap
# # # #     dense+sparse fusion stage.

# # # # KNOWN LIMITATION — in-memory BM25 (see _get_bm25_index below):
# # # #   This module builds and caches a BM25 index by pulling every document out
# # # #   of the Chroma collection into memory. That's fine for a demo/small corpus,
# # # #   but it means high memory use and cold-start latency on large collections,
# # # #   and the index only refreshes when explicitly invalidated (see
# # # #   invalidate_bm25_cache), not automatically on ingestion. The correct
# # # #   production fix is to move to a vector database with NATIVE sparse+dense
# # # #   hybrid indexing and native filtered lexical search — e.g. Qdrant, Milvus,
# # # #   or Weaviate — which eliminates client-side index building entirely and
# # # #   lets the database engine apply metadata filters during scoring rather than
# # # #   after it. That's a backend swap, not a change to this module's public
# # # #   hybrid_search() signature: _dense_search/_sparse_search/_get_bm25_index
# # # #   are the only functions that would need to be replaced with native
# # # #   filtered-hybrid-query calls against the new store.
# # # # """

# # # # from __future__ import annotations

# # # # import re
# # # # from typing import Optional

# # # # from rank_bm25 import BM25Okapi

# # # # from ..resources import get_chroma_collection, get_embedder, get_reranker
# # # # from ..tracing import traceable

# # # # # ---------------------------------------------------------------------------
# # # # # BM25 index (in-memory, cached per collection)
# # # # # ---------------------------------------------------------------------------

# # # # _bm25_cache: dict[str, tuple[BM25Okapi, list[str], list[str], list[dict]]] = {}


# # # # def _tokenize(text: str) -> list[str]:
# # # #     return re.findall(r"[a-z0-9]+", text.lower())


# # # # def _get_bm25_index(collection_name: str, refresh: bool = False):
# # # #     if not refresh and collection_name in _bm25_cache:
# # # #         return _bm25_cache[collection_name]

# # # #     collection = get_chroma_collection(collection_name)
# # # #     raw = collection.get(include=["documents", "metadatas"])
# # # #     ids, texts, metadatas = raw["ids"], raw["documents"], raw["metadatas"]
# # # #     bm25 = BM25Okapi([_tokenize(t) for t in texts])

# # # #     _bm25_cache[collection_name] = (bm25, ids, texts, metadatas)
# # # #     return _bm25_cache[collection_name]


# # # # def invalidate_bm25_cache(collection_name: str) -> None:
# # # #     """
# # # #     Call this right after ingesting new documents into `collection_name`
# # # #     (e.g. from scripts/run_demo.py, right after embed_and_store()) so the
# # # #     next hybrid_search() call rebuilds the BM25 index instead of silently
# # # #     searching a stale one that doesn't include the documents you just added.
# # # #     Not wired in automatically from pdf_pipeline.embed_and_store() itself,
# # # #     to avoid a circular import between ingestion and retrieval — call it
# # # #     explicitly from whichever orchestration layer just finished ingesting.
# # # #     """
# # # #     _bm25_cache.pop(collection_name, None)


# # # # # ---------------------------------------------------------------------------
# # # # # Metadata filter -> Chroma `where` clause, and a matching Python-side
# # # # # predicate for the BM25 branch (which Chroma's `where` can't reach).
# # # # # ---------------------------------------------------------------------------

# # # # def build_where_clause(filters: dict) -> Optional[dict]:
# # # #     clauses: list[dict] = []
# # # #     if filters.get("contract_type"):
# # # #         clauses.append({"contract_type": filters["contract_type"]})
# # # #     if filters.get("governing_law_country"):
# # # #         clauses.append({"governing_law_country": filters["governing_law_country"].upper()})
# # # #     if filters.get("min_effective_date_epoch") is not None:
# # # #         clauses.append({"effective_date_epoch": {"$gte": filters["min_effective_date_epoch"]}})
# # # #     if filters.get("max_effective_date_epoch") is not None:
# # # #         clauses.append({"effective_date_epoch": {"$lte": filters["max_effective_date_epoch"]}})
# # # #     if filters.get("min_monetary_value") is not None:
# # # #         clauses.append({"monetary_value": {"$gte": filters["min_monetary_value"]}})
# # # #     if filters.get("max_monetary_value") is not None:
# # # #         clauses.append({"monetary_value": {"$lte": filters["max_monetary_value"]}})

# # # #     if not clauses:
# # # #         return None
# # # #     return clauses[0] if len(clauses) == 1 else {"$and": clauses}


# # # # def _matches_where(meta: dict, where: Optional[dict]) -> bool:
# # # #     """Python-side equivalent of build_where_clause(), for filtering BM25 candidates."""
# # # #     if not where:
# # # #         return True
# # # #     if "$and" in where:
# # # #         return all(_matches_where(meta, clause) for clause in where["$and"])
# # # #     (field, condition), = where.items()
# # # #     value = meta.get(field)
# # # #     if isinstance(condition, dict):
# # # #         if "$gte" in condition and not (value is not None and value >= condition["$gte"]):
# # # #             return False
# # # #         if "$lte" in condition and not (value is not None and value <= condition["$lte"]):
# # # #             return False
# # # #         return True
# # # #     return value == condition


# # # # # ---------------------------------------------------------------------------
# # # # # Dense and sparse branches
# # # # # ---------------------------------------------------------------------------

# # # # def _dense_search(collection_name: str, query: str, top_k: int, where: Optional[dict]) -> list[dict]:
# # # #     """Chroma applies `where` natively DURING the ANN search — filtering here was never the issue."""
# # # #     collection = get_chroma_collection(collection_name)
# # # #     embedder = get_embedder()
# # # #     query_embedding = embedder.encode([query], normalize_embeddings=True).tolist()
# # # #     results = collection.query(query_embeddings=query_embedding, n_results=top_k, where=where)

# # # #     hits = []
# # # #     if results["ids"]:
# # # #         for doc_id, doc, meta, dist in zip(
# # # #             results["ids"][0], results["documents"][0], results["metadatas"][0], results["distances"][0]
# # # #         ):
# # # #             hits.append({"id": doc_id, "text": doc, "metadata": meta, "dense_distance": dist})
# # # #     return hits


# # # # def _sparse_search(collection_name: str, query: str, top_k: int, where: Optional[dict]) -> list[dict]:
# # # #     """
# # # #     Filter FIRST (restrict to the set of documents that satisfy `where`),
# # # #     THEN rank that eligible set by BM25 score — not the other way around.
# # # #     Ranking the whole corpus first and filtering afterward risks returning
# # # #     fewer than top_k results (or zero) under a restrictive filter even when
# # # #     plenty of matching-and-relevant documents exist further down the ranked
# # # #     list; filtering the candidate pool before selecting top_k guarantees we
# # # #     never miss an eligible match because it happened to rank outside some
# # # #     arbitrary pre-filter cutoff.
# # # #     """
# # # #     bm25, ids, texts, metadatas = _get_bm25_index(collection_name)
# # # #     scores = bm25.get_scores(_tokenize(query))

# # # #     eligible = [i for i in range(len(ids)) if _matches_where(metadatas[i], where)]
# # # #     eligible_ranked = sorted(eligible, key=lambda i: scores[i], reverse=True)[:top_k]

# # # #     return [
# # # #         {"id": ids[i], "text": texts[i], "metadata": metadatas[i], "bm25_score": float(scores[i])}
# # # #         for i in eligible_ranked
# # # #     ]


# # # # # ---------------------------------------------------------------------------
# # # # # Fusion strategies
# # # # # ---------------------------------------------------------------------------

# # # # def reciprocal_rank_fusion(id_lists: list[list[str]], k: int = 60) -> dict[str, float]:
# # # #     """
# # # #     Rank-only fusion: score(d) = sum, over lists containing d, of
# # # #     1 / (k + rank_in_that_list). Doesn't need the two branches' raw scores
# # # #     to be comparable at all — only their relative ordering — which makes it
# # # #     a safe default when you don't want to tune a blend weight. Kept as an
# # # #     alternate to convex_combination_fusion (see hybrid_search()'s
# # # #     fusion_method parameter).
# # # #     """
# # # #     scores: dict[str, float] = {}
# # # #     for id_list in id_lists:
# # # #         for rank, doc_id in enumerate(id_list):
# # # #             scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
# # # #     return scores


# # # # def min_max_normalize(scores: dict[str, float], reverse: bool = False) -> dict[str, float]:
# # # #     """Normalizes scores to the [0, 1] range using Min-Max scaling.

# # # #     Args:
# # # #         scores: Dictionary mapping document IDs to raw scores.
# # # #         reverse: Set to True if lower raw score is better (e.g. L2 or Cosine
# # # #           Distance).

# # # #     Returns:
# # # #         Dictionary mapping document IDs to normalized scores in [0, 1].
# # # #     """
# # # #     if not scores:
# # # #         return {}
# # # #     vals = list(scores.values())
# # # #     min_val, max_val = min(vals), max(vals)
# # # #     # Edge case: all candidates have the identical raw score.
# # # #     if min_val == max_val:
# # # #         return {doc_id: 1.0 for doc_id in scores}
# # # #     normalized = {}
# # # #     for doc_id, score in scores.items():
# # # #         if reverse:
# # # #             # For distance metrics where lower is better.
# # # #             norm_score = (max_val - score) / (max_val - min_val)
# # # #         else:
# # # #             # For similarity/BM25 metrics where higher is better.
# # # #             norm_score = (score - min_val) / (max_val - min_val)
# # # #         normalized[doc_id] = float(norm_score)
# # # #     return normalized


# # # # def convex_combination_fusion(
# # # #     dense_hits: list[dict],
# # # #     sparse_hits: list[dict],
# # # #     alpha: float = 0.4,
# # # #     dense_score_key: str = "dense_distance",
# # # #     sparse_score_key: str = "bm25_score",
# # # #     dense_is_distance: bool = True,
# # # # ) -> dict[str, float]:
# # # #     """Combines dense and sparse results using a weighted convex combination.

# # # #     Final Score = alpha * Dense_Norm + (1 - alpha) * Sparse_Norm

# # # #     Default alpha=0.4 weights lexical (BM25) matching more heavily than
# # # #     dense similarity (40% dense / 60% sparse) — legal text leans on exact
# # # #     clause wording, defined terms, and citations, where lexical match is
# # # #     often the stronger signal; raise alpha toward 1.0 if your queries skew
# # # #     more paraphrase/semantic than exact-term.

# # # #     Args:
# # # #         dense_hits: List of dicts, e.g. [{"id": "doc1", "dense_distance": 0.12}]
# # # #         sparse_hits: List of dicts, e.g. [{"id": "doc1", "bm25_score": 14.2}]
# # # #         alpha: Weight for dense retrieval (0.0 to 1.0). High alpha = favor
# # # #           dense.
# # # #         dense_score_key: Dictionary key containing the dense metric.
# # # #         sparse_score_key: Dictionary key containing the sparse metric.
# # # #         dense_is_distance: Set True if dense score is distance (lower =
# # # #           better).

# # # #     Returns:
# # # #         Dictionary mapping doc IDs to their combined score, sorted descending.
# # # #     """
# # # #     # 1. Extract raw scores into mappings.
# # # #     raw_dense = {h["id"]: h[dense_score_key] for h in dense_hits}
# # # #     raw_sparse = {h["id"]: h[sparse_score_key] for h in sparse_hits}

# # # #     # 2. Normalize both score distributions independently to [0, 1].
# # # #     norm_dense = min_max_normalize(raw_dense, reverse=dense_is_distance)
# # # #     norm_sparse = min_max_normalize(raw_sparse, reverse=False)

# # # #     # 3. Combine scores across the union of all candidate document IDs.
# # # #     all_doc_ids = set(norm_dense.keys()).union(set(norm_sparse.keys()))
# # # #     fused_scores = {}
# # # #     for doc_id in all_doc_ids:
# # # #         # If a document is missing in one branch, treat its normalized score as 0.0.
# # # #         d_score = norm_dense.get(doc_id, 0.0)
# # # #         s_score = norm_sparse.get(doc_id, 0.0)
# # # #         fused_scores[doc_id] = (alpha * d_score) + ((1.0 - alpha) * s_score)

# # # #     # Sort candidates by fused score descending.
# # # #     return dict(sorted(fused_scores.items(), key=lambda item: item[1], reverse=True))


# # # # def rerank(query: str, candidates: list[dict], top_k: int) -> list[dict]:
# # # #     """
# # # #     Cross-encoder rerank of the fused candidate set. The reranker itself is
# # # #     a module-level singleton loaded via resources.get_reranker() — NOT
# # # #     instantiated here — see resources.preload_models() to load it eagerly
# # # #     at application startup instead of lazily on the first call.
# # # #     """
# # # #     if not candidates:
# # # #         return []
# # # #     reranker = get_reranker()
# # # #     pairs = [(query, c["text"]) for c in candidates]
# # # #     scores = reranker.predict(pairs)
# # # #     for c, s in zip(candidates, scores):
# # # #         c["rerank_score"] = float(s)
# # # #     return sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)[:top_k]


# # # # # ---------------------------------------------------------------------------
# # # # # Public entrypoint
# # # # # ---------------------------------------------------------------------------

# # # # @traceable(name="retrieval.hybrid_search_agent", run_type="retriever")
# # # # def hybrid_search(
# # # #     collection_name: str,
# # # #     query: str,
# # # #     metadata_filter: Optional[dict] = None,
# # # #     dense_k: int = 20,
# # # #     sparse_k: int = 20,
# # # #     fusion_k: int = 15,
# # # #     final_k: int = 6,
# # # #     fusion_method: str = "convex",
# # # #     alpha: float = 0.4,
# # # # ) -> list[dict]:
# # # #     """
# # # #     Full pipeline: dense search + BM25 search (each metadata-filtered) ->
# # # #     fusion ("convex" weighted combination by default, or "rrf" for
# # # #     rank-only fusion) -> parties_contains post-filter -> cross-encoder
# # # #     rerank -> top final_k results, each:
# # # #     {"id", "text", "metadata", "rerank_score", ...}.
# # # #     """
# # # #     metadata_filter = metadata_filter or {}
# # # #     where = build_where_clause(metadata_filter)

# # # #     dense_hits = _dense_search(collection_name, query, dense_k, where)
# # # #     sparse_hits = _sparse_search(collection_name, query, sparse_k, where)

# # # #     by_id: dict[str, dict] = {}
# # # #     for h in dense_hits + sparse_hits:
# # # #         by_id.setdefault(h["id"], {}).update(h)

# # # #     if fusion_method == "rrf":
# # # #         fused_scores = reciprocal_rank_fusion(
# # # #             [[h["id"] for h in dense_hits], [h["id"] for h in sparse_hits]]
# # # #         )
# # # #     else:
# # # #         fused_scores = convex_combination_fusion(
# # # #             dense_hits=dense_hits,
# # # #             sparse_hits=sparse_hits,
# # # #             alpha=alpha,
# # # #             dense_is_distance=True,  # Chroma returns distances where lower = better
# # # #         )

# # # #     fused_ranked_ids = list(fused_scores.keys())[:fusion_k]
# # # #     candidates = [by_id[i] for i in fused_ranked_ids if i in by_id]

# # # #     parties_needle = metadata_filter.get("parties_contains")
# # # #     if parties_needle:
# # # #         needle = parties_needle.lower()
# # # #         candidates = [c for c in candidates if needle in (c["metadata"].get("parties", "") or "").lower()]

# # # #     return rerank(query, candidates, top_k=final_k)
# # # # 