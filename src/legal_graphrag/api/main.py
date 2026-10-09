"""
FastAPI app for the UNMODIFIED legal_graphrag pipeline: wraps
agents/legal_pipeline.py (query) and ingestion/pdf_pipeline.py (ingest)
over HTTP, and serves the static frontend/ UI. No pipeline file is edited
by this app — it only imports and calls existing public functions, same as
scripts/run_demo.py.

Run:
    

Then open http://127.0.0.1:8000/
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from ..resources import preload_models
from .cuad_routes import router as cuad_router  # NEW: CUAD evaluation panel
from .feedback_routes import router as feedback_router  # NEW: review error-log
from .ingest_routes import router as ingest_router
from .query_routes import router as query_router

FRONTEND_DIR = Path(__file__).resolve().parents[3] / "frontend"

app = FastAPI(
    title="legal_graphrag API",
    description="Query + ingestion pipeline for legal_graphrag (unmodified), with human-in-the-loop review over HTTP.",
    version="0.1.0",
)

# Permissive CORS for local development — tighten before deploying anywhere reachable.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

app.include_router(query_router)
app.include_router(ingest_router)
app.include_router(cuad_router)  # NEW
app.include_router(feedback_router)  # NEW


@app.on_event("startup")
def _startup() -> None:
    # Non-fatal: if Neo4j/env vars aren't configured yet, the API still
    # starts (so you can serve the frontend and iterate on config) — but
    # query/ingest calls will fail until it's fixed.
    try:
        preload_models(include_neo4j=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[startup] preload_models failed (will retry lazily per-request): {exc}")


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


# """
# FastAPI app for the UNMODIFIED legal_graphrag pipeline: wraps
# agents/legal_pipeline.py (query) and ingestion/pdf_pipeline.py (ingest)
# over HTTP, and serves the static frontend/ UI. No pipeline file is edited
# by this app — it only imports and calls existing public functions, same as
# scripts/run_demo.py.

# Run:
#     uvicorn legal_graphrag.api.main:app --reload --app-dir src

# Then open http://127.0.0.1:8000/
# """

# from __future__ import annotations

# from pathlib import Path

# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.staticfiles import StaticFiles

# from ..resources import preload_models
# from .cuad_routes import router as cuad_router  # NEW: CUAD evaluation panel
# from .ingest_routes import router as ingest_router
# from .query_routes import router as query_router

# FRONTEND_DIR = Path(__file__).resolve().parents[3] / "frontend"

# app = FastAPI(
#     title="legal_graphrag API",
#     description="Query + ingestion pipeline for legal_graphrag (unmodified), with human-in-the-loop review over HTTP.",
#     version="0.1.0",
# )

# # Permissive CORS for local development — tighten before deploying anywhere reachable.
# app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# app.include_router(query_router)
# app.include_router(ingest_router)
# app.include_router(cuad_router)  # NEW


# @app.on_event("startup")
# def _startup() -> None:
#     # Non-fatal: if Neo4j/env vars aren't configured yet, the API still
#     # starts (so you can serve the frontend and iterate on config) — but
#     # query/ingest calls will fail until it's fixed.
#     try:
#         preload_models(include_neo4j=True)
#     except Exception as exc:  # noqa: BLE001
#         print(f"[startup] preload_models failed (will retry lazily per-request): {exc}")


# @app.get("/api/health")
# def health() -> dict:
#     return {"status": "ok"}


# if FRONTEND_DIR.exists():
#     app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
