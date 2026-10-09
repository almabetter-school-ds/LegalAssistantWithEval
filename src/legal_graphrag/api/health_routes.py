"""
Readiness endpoint (liveness already exists at /api/health).

  GET /api/health    LIVENESS  — defined in main.py: is the process up? No
                     external work, always 200. This is what the container
                     HEALTHCHECK calls, so a momentarily-unavailable Neo4j
                     never marks the app unhealthy and triggers a restart.

  GET /health/ready  READINESS — added here: can the app actually serve queries
                     right now? Best-effort checks of Chroma (persist dir
                     writable) and Neo4j (bolt reachable). 200 when all pass,
                     503 with per-dependency detail otherwise. Use this for a
                     load-balancer readiness gate or a startup wait — NOT for
                     the container liveness probe.

All dependency imports are lazy and wrapped, so importing this module can never
fail and a broken dependency degrades to a reported "down" rather than a crash.
"""

from __future__ import annotations

import os

from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter(tags=["health"])


def _check_chroma() -> dict:
    """Chroma is embedded — verify its persist dir exists and is writable."""
    path = os.getenv("CHROMA_PERSIST_DIR", "data/chroma_db")
    try:
        os.makedirs(path, exist_ok=True)
        probe = os.path.join(path, ".health_probe")
        with open(probe, "w") as f:
            f.write("ok")
        os.remove(probe)
        return {"ok": True, "path": path}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "path": path, "error": str(exc)}


def _check_neo4j() -> dict:
    """Best-effort Neo4j reachability with a short timeout."""
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "")
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(
            uri, auth=(user, password),
            connection_timeout=3, connection_acquisition_timeout=3,
        )
        try:
            driver.verify_connectivity()
            with driver.session() as s:
                s.run("RETURN 1").consume()
            return {"ok": True, "uri": uri}
        finally:
            driver.close()
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "uri": uri, "error": str(exc)}


@router.get("/health/ready")
def readiness() -> JSONResponse:
    """Readiness: dependencies reachable. 200 if all ok, else 503 with detail."""
    checks = {"chroma": _check_chroma(), "neo4j": _check_neo4j()}
    all_ok = all(c.get("ok") for c in checks.values())
    body = {"status": "ready" if all_ok else "not_ready", "checks": checks}
    return JSONResponse(status_code=200 if all_ok else 503, content=body)
