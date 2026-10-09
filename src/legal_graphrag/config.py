"""
Centralized environment configuration.

Import this module (or call load_env()) once, early, so every other module
can read os.environ directly without each one calling load_dotenv() itself.
"""

from __future__ import annotations

import os
from dotenv import load_dotenv


from pathlib import Path

_loaded = False

# EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")  # or your exact model
# # CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")

#==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHROMA_PERSIST_DIR = str(
    PROJECT_ROOT / "data" / "chroma_db"
)

EMBEDDING_MODEL_NAME = (
    "sentence-transformers/all-mpnet-base-v2"
)

DEFAULT_COLLECTION_NAME = (
    "legal_knowledge_base"
)

DEFAULT_CHUNK_SIZE = 1200
DEFAULT_CHUNK_OVERLAP = 150

TABLE_CONTENT_TYPE = "table"
TEXT_CONTENT_TYPE = "text"


#==========================================================


def load_env(dotenv_path: str | None = None) -> None:
    global _loaded
    if not _loaded:
        load_dotenv(dotenv_path)
        _loaded = True


def require_env(name: str) -> str:
    """Fetch a required environment variable, failing fast with a clear message if missing."""
    value = os.getenv(name)
    if not value:
        raise EnvironmentError(
            f"Required environment variable '{name}' is not set. "
            f"Add it to your .env file (see .env.example) or export it directly."
        )
    return value


load_env()
