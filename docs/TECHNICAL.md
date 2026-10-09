# legal_graphrag — API + Frontend Technical Documentation

Scope: this documents ONLY the FastAPI backend (`src/legal_graphrag/api/`)
and TypeScript frontend (`frontend/`) added in this change. The pipeline
itself (`agents/legal_pipeline.py`, `ingestion/pdf_pipeline.py`,
`graphrag/`, `retrieval/`) is **unmodified** — every file listed under
"Files added" below is new; nothing existing was edited.

## Architecture

```
┌─────────────────────────────┐
│ frontend/index.html          │  static shell, loads dist/main.js
│ frontend/dist/main.js         │  compiled from src/*.ts (tsc)
└──────────────┬────────────────┘
               │ fetch() — same-origin (served by the same FastAPI process)
┌──────────────▼────────────────┐
│ api/main.py (FastAPI)          │
│ api/query_routes.py            │──▶ agents/legal_pipeline.build_legal_agent_graph()  [UNMODIFIED]
│ api/ingest_routes.py           │──▶ ingestion/pdf_pipeline.run_pipeline()             [UNMODIFIED]
└─────────────────────────────────┘
```

`api/query_routes.py` and `api/ingest_routes.py` only **call** the
pipeline's existing public functions — `build_legal_agent_graph()` and
`run_pipeline()` — exactly as `scripts/run_demo.py` already does. No
pipeline file imports anything from `api/`.

## Files added

```
src/legal_graphrag/api/__init__.py       # package docstring only
src/legal_graphrag/api/schemas.py        # Pydantic request/response models — field names match
                                          # LegalAgentState/SynthesizedAnswer exactly (plain string
                                          # answer; no confidence/evidence/document/source fields —
                                          # those don't exist in this version of the pipeline)
src/legal_graphrag/api/jobs.py           # in-memory thread/job registries (process-local, non-durable)
src/legal_graphrag/api/query_routes.py   # wraps build_legal_agent_graph()
src/legal_graphrag/api/ingest_routes.py  # wraps run_pipeline(); lists Chroma collections via its
                                          # own client (resources.py is untouched)
src/legal_graphrag/api/main.py           # FastAPI app: routers + CORS + static frontend mount

frontend/index.html                      # HTML shell, loads dist/main.js as an ES module
frontend/src/types.ts                    # TS interfaces mirroring api/schemas.py
frontend/src/api.ts                      # typed fetch wrapper, one function per endpoint
frontend/src/render.ts                   # rendering rules — see "Output vs. collapsible" below
frontend/src/main.ts                     # DOM wiring: tabs, Ask/Ingest/Threads flows
frontend/tsconfig.json                   # strict TS compiler config
frontend/package.json                    # devDependency: typescript; "build"/"watch" scripts

docs/TECHNICAL.md                        # this file
```

## Imports / installs required

**Python (backend):**
```bash
pip install fastapi "uvicorn[standard]" python-multipart
```
`python-multipart` is required by FastAPI for the file-upload endpoint
(`POST /api/ingest`) — omit it and that route fails at request time with a
clear error asking for it.

**Node (frontend build only — not needed at runtime):**
```bash
cd frontend
npm install      # installs devDependency: typescript ^5.5
npm run build    # tsc — compiles src/*.ts -> dist/*.js
```
Requires Node.js (tested with Node 22 / npm 10). The compiled `dist/*.js`
files are what actually get served — `npm`/`tsc` are a build-time-only
dependency, not needed on the machine running `uvicorn`.

## Running it

```bash
# 1. Backend deps
pip install fastapi "uvicorn[standard]" python-multipart

# 2. Frontend build (compiles TS -> frontend/dist/*.js)
cd frontend && npm install && npm run build && cd ..

# 3. Env vars (same as scripts/run_demo.py needs)
export ANTHROPIC_API_KEY=sk-ant-...
export NEO4J_URI=bolt://localhost:7687
export NEO4J_USER=neo4j
export NEO4J_PASSWORD=...

# 4. Run
uvicorn legal_graphrag.api.main:app --reload --app-dir src
```

Open **http://127.0.0.1:8000/** (Swagger docs at `/docs`).

During frontend development, run `npm run watch` in `frontend/` in a
second terminal to recompile on save — `uvicorn --reload` picks up the
regenerated `dist/*.js` automatically since it's served as a static file
(no backend restart needed for frontend-only changes).

## API reference

### Query pipeline

| Method | Path | Body | Notes |
|---|---|---|---|
| POST | `/api/query/start` | `{question, collection_name, metadata_filter?}` | Starts a thread, runs to the first checkpoint (or straight to `answered`, since both checkpoints in this pipeline are unconditional). |
| GET | `/api/query` | — | Lists threads started this process. |
| GET | `/api/query/{thread_id}` | — | Peeks current status without resuming. |
| POST | `/api/query/{thread_id}/evidence-decision` | `{proceed, reviewer?, comments?}` | Resumes the evidence checkpoint. |
| POST | `/api/query/{thread_id}/answer-decision` | `{action, reviewer?, comments?, edited_answer?}` | `action`: `approve` \| `revise` \| `reject`. |

`status` is one of: `awaiting_evidence_approval`, `awaiting_answer_approval`,
`answered`, `rejected`, `evidence_rejected`.

```bash
curl -X POST localhost:8000/api/query/start -H 'Content-Type: application/json' \
  -d '{"question": "What is the agreement date?", "collection_name": "my_contract"}'

curl -X POST localhost:8000/api/query/<thread_id>/evidence-decision -H 'Content-Type: application/json' \
  -d '{"proceed": true, "reviewer": "jane.doe"}'

curl -X POST localhost:8000/api/query/<thread_id>/answer-decision -H 'Content-Type: application/json' \
  -d '{"action": "approve", "reviewer": "jane.doe"}'
# -> {"status": "answered", "final_answer": "The agreement date is February 18, 2005.", ...}
```

### Ingestion

| Method | Path | Body | Notes |
|---|---|---|---|
| POST | `/api/ingest` | multipart `file` (+ optional `collection_name` query param) | Returns `job_id` immediately; runs in the background. |
| GET | `/api/ingest/{job_id}` | — | Poll status: `pending`/`running`/`done`/`error`. |
| GET | `/api/ingest/meta/collections` | — | Chroma collection names available to query. |

## Frontend design: Output vs. collapsible

Per spec, the rendering layer (`frontend/src/render.ts`) enforces one rule:
the **Answer** text (`renderAnswerOutput`) is the only content block that
is never collapsed. Everything else renders inside `<details>`:

- `renderCitationsDetails` — citations, risk level, uncertainty flag
- `renderTechnicalDetails` — route, alpha, retrieved chunks (hybrid + graph),
  Cypher used, evidence auditor verdict (gaps/contradictions), revision count
- The evidence checkpoint's raw retrieval is shown the same way (approve/
  reject buttons stay visible; the supporting evidence sits in the same
  collapsible technical-details block)
- The ingestion job's full result JSON is also collapsible

This mapping is enforced in code (not just CSS) — `renderQueryState()` in
`render.ts` is the single place that decides what's headline vs.
collapsible, so there's one function to check if you want to change that
rule later.

## Notes / limitations

- **State is in-memory.** The pipeline's own `MemorySaver` checkpointer
  (inside `build_legal_agent_graph()`, unmodified) and `api/jobs.py`'s
  thread/job registries both reset on process restart. Not durable —
  fine for manual inspection, not for production.
- **CORS is wide open** (`allow_origins=["*"]`) for local development.
  Tighten before deploying anywhere reachable.
- **TypeScript is a build-time dependency only.** The frontend ships as
  plain compiled JS (`frontend/dist/*.js`) — no TS toolchain is required
  wherever `uvicorn` actually runs, only wherever you last ran `npm run build`.
- **`preload_models()` failures at startup are non-fatal** (see
  `api/main.py`) — the API still boots so you can serve the frontend while
  fixing Neo4j/env config, but query/ingest calls will fail until resolved.
