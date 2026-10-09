from __future__ import annotations

from typing import Optional

from sentence_transformers import SentenceTransformer, CrossEncoder
from .config import (
    EMBEDDING_MODEL_NAME,
    CHROMA_PERSIST_DIR,
)
# from .ingestion.pdf_pipeline import EMBEDDING_MODEL_NAME 


import chromadb

from . import config
from .graphrag.neo4j_store import Neo4jGraphStore


RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_store: Optional[Neo4jGraphStore] = None
_embedder: Optional[SentenceTransformer] = None
_reranker: Optional[CrossEncoder] = None
_chroma_client = None


def get_store() -> Neo4jGraphStore:
    global _store

    if _store is None:
        _store = Neo4jGraphStore(
            uri=config.require_env("NEO4J_URI"),
            user=config.require_env("NEO4J_USER"),
            password=config.require_env("NEO4J_PASSWORD"),
        )
        _store.ensure_schema()

    return _store


def get_embedder() -> SentenceTransformer:
    global _embedder

    if _embedder is None:
        _embedder = SentenceTransformer(
            EMBEDDING_MODEL_NAME
        )

    return _embedder


def get_reranker() -> CrossEncoder:
    global _reranker

    if _reranker is None:
        _reranker = CrossEncoder(RERANKER_MODEL_NAME)

    return _reranker


def get_chroma_collection(collection_name: str):
    global _chroma_client

    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(
            path=config.CHROMA_PERSIST_DIR
        )

    return _chroma_client.get_or_create_collection(
        name=collection_name
    )


def preload_models(include_neo4j: bool = True) -> None:
    get_embedder()
    get_reranker()

    if include_neo4j:
        get_store()

# """
# Shared external-resource singletons.

# Deliberately kept OUT of any LangGraph state and created here as module-level
# singletons instead. LangGraph state must be checkpoint-serializable to
# support pausing for human approval and resuming later (possibly in a
# different process) — a Neo4j driver, a Chroma collection, and a loaded
# embedding model are none of those things, so graph nodes fetch them from
# here rather than carrying them through state.

# Centralizing them in one module (rather than each of graphrag/, retrieval/,
# and agents/ defining its own copies) also means the embedding model and the
# Neo4j driver are each loaded exactly once per process, no matter how many of
# those subpackages end up using them.
# """

# from __future__ import annotations

# import os

# from typing import Optional

# from sentence_transformers import SentenceTransformer, CrossEncoder
# import chromadb

# from . import config
# from .graphrag.neo4j_store import Neo4jGraphStore
# from .ingestion.pdf_pipeline import EMBEDDING_MODEL_NAME as VECTOR_EMBEDDING_MODEL, CHROMA_PERSIST_DIR

# RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# _store: Optional[Neo4jGraphStore] = None
# _embedder: Optional[SentenceTransformer] = None
# _reranker: Optional[CrossEncoder] = None
# _chroma_client = None


# def get_store() -> Neo4jGraphStore:
#     global _store
#     if _store is None:
#         _store = Neo4jGraphStore(
#             uri=config.require_env("NEO4J_URI"),
#             user=config.require_env("NEO4J_USER"),
#             password=config.require_env("NEO4J_PASSWORD"),
#         )
#         _store.ensure_schema()
#     return _store


# def get_embedder() -> SentenceTransformer:
#     global _embedder
#     if _embedder is None:
#         _embedder = SentenceTransformer(VECTOR_EMBEDDING_MODEL)
#     return _embedder


# def get_reranker() -> CrossEncoder:
#     """
#     Cross-encoder used by retrieval.hybrid_search.rerank(). Lazily
#     instantiated on first use like the other singletons here, but see
#     preload_models() below if you want it (and the embedder) loaded eagerly
#     at process startup instead of on the first incoming request.
#     """
#     global _reranker
#     if _reranker is None:
#         _reranker = CrossEncoder(RERANKER_MODEL_NAME)
#     return _reranker


# def get_chroma_collection(collection_name: str):
#     global _chroma_client
#     if _chroma_client is None:
#         _chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
#     return _chroma_client.get_or_create_collection(name=collection_name)


# def preload_models(include_neo4j: bool = True) -> None:
#     """
#     Eagerly loads the embedding model and cross-encoder reranker (and,
#     optionally, connects to Neo4j) at APPLICATION STARTUP rather than lazily
#     on the first request. Call this once when your process boots (e.g. at
#     the top of scripts/run_demo.py's main(), or in a FastAPI/Streamlit
#     startup hook) so model-loading latency is paid once at boot, not on
#     whichever unlucky request happens to arrive first.
#     """
#     get_embedder()
#     get_reranker()
#     if include_neo4j:
#         get_store()
