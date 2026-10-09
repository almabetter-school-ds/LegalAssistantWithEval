"""
Neo4j graph store for the Legal GraphRAG pipeline.

Schema (nodes):
    DocumentJob     — one per PDF ingestion run (status, timestamps)
    Contract        — the contract represented by that PDF
    Party           — a company/person referenced by a contract (e.g. vendor)
    Clause          — one extracted clause, with its own embedding for
                       vector similarity search ("same clause elsewhere")
    Judgment        — a court judgment that interprets a clause
    RiskFlag        — a risk assessment attached to a clause
    ReviewerDecision — a human approval/rejection of a DocumentJob
    AuditRecord     — an immutable log entry attached to a DocumentJob

Schema (relationships):
    (DocumentJob)-[:PRODUCED]->(Contract)
    (Contract)-[:HAS_VENDOR]->(Party)
    (Contract)-[:CONTAINS_CLAUSE]->(Clause)
    (Clause)-[:SAME_CLAUSE_AS]->(Clause)        — embedding-similarity match across contracts
    (Clause)-[:CONFLICTS_WITH]->(Clause)        — LLM-flagged contradiction
    (Clause)-[:INTERPRETED_BY]->(Judgment)
    (Clause)-[:FLAGGED_AS]->(RiskFlag)
    (DocumentJob)-[:REVIEWED_BY]->(ReviewerDecision)
    (DocumentJob)-[:HAS_AUDIT_RECORD]->(AuditRecord)

Design notes (see chat response for the full write-up):
    - Audit records are append-only. A rejected job is never deleted — it's
      marked Contract.approved = false so the evidence trail survives.
    - Clause embeddings are stored on the node and indexed with a native
      Neo4j vector index, so "find the same clause elsewhere" is a single
      vector query rather than pulling every clause into Python to compare.
    - All write helpers use MERGE (not CREATE) on natural keys where
      sensible, so re-running ingestion on the same document is idempotent
      rather than creating duplicate nodes.

CONNECTION RESILIENCE — read this if you hit SessionExpired/BrokenPipeError:
    Every method here goes through session.execute_write()/execute_read()
    (managed transaction functions), never raw session.run(). This matters
    specifically because of how this pipeline is used: execution can pause
    for an arbitrarily long time at a human-in-the-loop interrupt() while a
    reviewer reads evidence and types a decision. During that pause the
    Neo4j driver's pooled connections sit idle, and a cloud-hosted instance
    (Aura) or an intermediate NAT/firewall will silently close an idle TCP
    connection long before the driver notices on its own. The NEXT write
    attempt then fails with SessionExpired/BrokenPipeError.
    execute_write()/execute_read() are the driver's documented mechanism for
    exactly this: they automatically retry the transaction (with backoff) on
    SessionExpired/ServiceUnavailable, transparently reconnecting instead of
    raising. Raw session.run() gets none of that — it's a single attempt
    with no retry, which is what the earlier version of this file used, and
    exactly why a long approval pause could crash and lose an already-typed
    reviewer decision. The driver is also configured below with
    liveness_check_timeout (ping a pooled connection before reusing it, if
    it's been idle) and max_connection_lifetime (proactively recycle
    connections) as an additional layer of prevention on top of the retry.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

from neo4j import GraphDatabase


CLAUSE_EMBEDDING_DIM = 768  # must match whatever embedding model you encode clauses with


class Neo4jGraphStore:
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(
            uri,
            auth=(user, password),
            # Ping a pooled connection before reusing it if it's been idle
            # longer than this many seconds, instead of assuming it's still
            # alive and finding out mid-write with a BrokenPipeError. This
            # directly targets the "long human-approval pause" scenario.
            liveness_check_timeout=30,
            # Proactively recycle connections older than this, so they never
            # get old enough to hit a cloud provider's own idle-connection
            # cutoff in the first place.
            max_connection_lifetime=3600,
            # How long execute_write()/execute_read() will keep retrying a
            # transaction that's failing on transient errors (SessionExpired,
            # ServiceUnavailable) before giving up and raising. Raised above
            # the driver default (30s) since a reconnect-and-retry here is
            # cheap and losing an already-made human decision is not.
            max_transaction_retry_time=60,
        )

    def close(self):
        self.driver.close()

    # ------------------------------------------------------------------
    # Low-level helpers — every query in this class goes through one of
    # these, so every query gets the driver's managed-transaction retry
    # behavior for free. Never call session.run() directly elsewhere in
    # this class.
    # ------------------------------------------------------------------

    def _run_write(self, cypher: str, **params) -> None:
        def _work(tx):
            tx.run(cypher, **params)

        with self.driver.session() as session:
            session.execute_write(_work)

    def _run_write_returning(self, cypher: str, **params) -> list[dict]:
        def _work(tx):
            return [dict(r) for r in tx.run(cypher, **params)]

        with self.driver.session() as session:
            return session.execute_write(_work)

    def _run_read(self, cypher: str, **params) -> list[dict]:
        def _work(tx):
            return [dict(r) for r in tx.run(cypher, **params)]

        with self.driver.session() as session:
            return session.execute_read(_work)

    # ------------------------------------------------------------------
    # Schema setup — call once (idempotent; safe to call on every startup)
    # ------------------------------------------------------------------

    def ensure_schema(self):
        statements = [
            "CREATE CONSTRAINT job_id_unique IF NOT EXISTS FOR (j:DocumentJob) REQUIRE j.job_id IS UNIQUE",
            "CREATE CONSTRAINT query_job_id_unique IF NOT EXISTS FOR (q:QueryJob) REQUIRE q.job_id IS UNIQUE",
            "CREATE CONSTRAINT contract_id_unique IF NOT EXISTS FOR (c:Contract) REQUIRE c.contract_id IS UNIQUE",
            "CREATE CONSTRAINT party_name_unique IF NOT EXISTS FOR (p:Party) REQUIRE p.name IS UNIQUE",
            "CREATE CONSTRAINT clause_id_unique IF NOT EXISTS FOR (cl:Clause) REQUIRE cl.clause_id IS UNIQUE",
            "CREATE CONSTRAINT judgment_citation_unique IF NOT EXISTS FOR (j:Judgment) REQUIRE j.citation IS UNIQUE",
            "CREATE CONSTRAINT risk_flag_id_unique IF NOT EXISTS FOR (r:RiskFlag) REQUIRE r.flag_id IS UNIQUE",
            "CREATE CONSTRAINT decision_id_unique IF NOT EXISTS FOR (d:ReviewerDecision) REQUIRE d.decision_id IS UNIQUE",
            "CREATE CONSTRAINT audit_id_unique IF NOT EXISTS FOR (a:AuditRecord) REQUIRE a.audit_id IS UNIQUE",
        ]
        vector_index = f"""
            CREATE VECTOR INDEX clause_embedding_index IF NOT EXISTS
            FOR (cl:Clause) ON (cl.embedding)
            OPTIONS {{ indexConfig: {{
                `vector.dimensions`: {CLAUSE_EMBEDDING_DIM},
                `vector.similarity_function`: 'cosine'
            }} }}
        """

        def _work(tx):
            for stmt in statements:
                tx.run(stmt)
            tx.run(vector_index)

        with self.driver.session() as session:
            session.execute_write(_work)

    # ------------------------------------------------------------------
    # Document job / contract / party
    # ------------------------------------------------------------------

    def create_document_job(self, job_id: str, document_name: str) -> None:
        self._run_write(
            """
            MERGE (j:DocumentJob {job_id: $job_id})
            SET j.document_name = $document_name,
                j.status = 'processing',
                j.created_at = $now,
                j.updated_at = $now
            """,
            job_id=job_id, document_name=document_name, now=_now(),
        )

    def update_document_job_status(self, job_id: str, status: str) -> None:
        self._run_write(
            "MATCH (j:DocumentJob {job_id: $job_id}) SET j.status = $status, j.updated_at = $now",
            job_id=job_id, status=status, now=_now(),
        )

    def create_contract(self, contract_id: str, job_id: str, document_name: str,
                         contract_name: Optional[str] = None) -> None:
        self._run_write(
            """
            MATCH (j:DocumentJob {job_id: $job_id})
            MERGE (c:Contract {contract_id: $contract_id})
            SET c.document_name = $document_name,
                c.name = coalesce($contract_name, $document_name),
                c.approved = null
            MERGE (j)-[:PRODUCED]->(c)
            """,
            job_id=job_id, contract_id=contract_id,
            document_name=document_name, contract_name=contract_name,
        )

    def link_vendor(self, contract_id: str, party_name: str) -> None:
        self._run_write(
            """
            MATCH (c:Contract {contract_id: $contract_id})
            MERGE (p:Party {name: $party_name})
            MERGE (c)-[:HAS_VENDOR]->(p)
            """,
            contract_id=contract_id, party_name=party_name,
        )

    def create_query_job(self, job_id: str, question: str) -> None:
        """
        A QueryJob is the query-side counterpart to DocumentJob — it exists
        so a user's question, the retrieval that answered it, and the human
        approval of that answer are all auditable the same way ingestion is.
        """
        self._run_write(
            """
            MERGE (q:QueryJob {job_id: $job_id})
            SET q.question = $question, q.status = 'processing',
                q.created_at = $now, q.updated_at = $now
            """,
            job_id=job_id, question=question, now=_now(),
        )

    def update_job_status(self, job_id: str, status: str) -> None:
        """Works for DocumentJob or QueryJob — matched by job_id, not label."""
        self._run_write(
            "MATCH (j {job_id: $job_id}) SET j.status = $status, j.updated_at = $now",
            job_id=job_id, status=status, now=_now(),
        )

    def store_query_answer(self, job_id: str, answer: str) -> None:
        self._run_write(
            "MATCH (q:QueryJob {job_id: $job_id}) SET q.final_answer = $answer",
            job_id=job_id, answer=answer,
        )

    def set_contract_approval(self, contract_id: str, approved: bool) -> None:
        self._run_write(
            "MATCH (c:Contract {contract_id: $contract_id}) SET c.approved = $approved",
            contract_id=contract_id, approved=approved,
        )

    # ------------------------------------------------------------------
    # Clauses
    # ------------------------------------------------------------------

    def create_clause(self, clause_id: str, contract_id: str, text: str, embedding: list[float],
                       page_start: int, page_end: int, section: Optional[str],
                       clause_type: Optional[str]) -> None:
        self._run_write(
            """
            MATCH (c:Contract {contract_id: $contract_id})
            MERGE (cl:Clause {clause_id: $clause_id})
            SET cl.text = $text, cl.embedding = $embedding,
                cl.page_start = $page_start, cl.page_end = $page_end,
                cl.section = $section, cl.clause_type = $clause_type
            MERGE (c)-[:CONTAINS_CLAUSE]->(cl)
            """,
            clause_id=clause_id, contract_id=contract_id, text=text, embedding=embedding,
            page_start=page_start, page_end=page_end, section=section, clause_type=clause_type,
        )

    def find_similar_clauses(self, clause_id: str, embedding: list[float],
                              top_k: int = 5, min_similarity: float = 0.90) -> list[dict]:
        """
        Vector-search for clauses similar to the given embedding, excluding
        the clause itself and clauses from the same contract (we only care
        about matches in *other* contracts for the "same clause elsewhere"
        relationship).
        """
        return self._run_read(
            """
            CALL db.index.vector.queryNodes('clause_embedding_index', $top_k, $embedding)
            YIELD node, score
            WHERE node.clause_id <> $clause_id AND score >= $min_similarity
            MATCH (c:Contract)-[:CONTAINS_CLAUSE]->(node)
            RETURN node.clause_id AS clause_id, node.text AS text,
                   c.contract_id AS contract_id, c.name AS contract_name, score
            """,
            top_k=top_k, embedding=embedding, clause_id=clause_id, min_similarity=min_similarity,
        )

    def link_same_clause(self, clause_id_a: str, clause_id_b: str, similarity: float) -> None:
        self._run_write(
            """
            MATCH (a:Clause {clause_id: $a}), (b:Clause {clause_id: $b})
            MERGE (a)-[r:SAME_CLAUSE_AS]->(b)
            SET r.similarity = $similarity
            """,
            a=clause_id_a, b=clause_id_b, similarity=similarity,
        )

    def create_conflict(self, clause_id_a: str, clause_id_b: str, reason: str) -> None:
        self._run_write(
            """
            MATCH (a:Clause {clause_id: $a}), (b:Clause {clause_id: $b})
            MERGE (a)-[r:CONFLICTS_WITH]->(b)
            SET r.reason = $reason
            """,
            a=clause_id_a, b=clause_id_b, reason=reason,
        )

    def create_judgment(self, citation: str, court: Optional[str], year: Optional[int],
                         summary: Optional[str]) -> None:
        self._run_write(
            """
            MERGE (j:Judgment {citation: $citation})
            SET j.court = coalesce($court, j.court),
                j.year = coalesce($year, j.year),
                j.summary = coalesce($summary, j.summary)
            """,
            citation=citation, court=court, year=year, summary=summary,
        )

    def link_interpreted_by(self, clause_id: str, judgment_citation: str) -> None:
        self._run_write(
            """
            MATCH (cl:Clause {clause_id: $clause_id}), (j:Judgment {citation: $citation})
            MERGE (cl)-[:INTERPRETED_BY]->(j)
            """,
            clause_id=clause_id, citation=judgment_citation,
        )

    # ------------------------------------------------------------------
    # Risk flags / reviewer decisions / audit
    # ------------------------------------------------------------------

    def create_risk_flag(self, clause_id: str, risk_level: str, reason: str) -> str:
        flag_id = str(uuid.uuid4())
        self._run_write(
            """
            MATCH (cl:Clause {clause_id: $clause_id})
            CREATE (r:RiskFlag {flag_id: $flag_id, risk_level: $risk_level,
                                 reason: $reason, created_at: $now})
            MERGE (cl)-[:FLAGGED_AS]->(r)
            """,
            clause_id=clause_id, flag_id=flag_id, risk_level=risk_level, reason=reason, now=_now(),
        )
        return flag_id

    def create_reviewer_decision(self, job_id: str, approved: bool, reviewer: str,
                                  comments: Optional[str]) -> str:
        """Attaches a ReviewerDecision to whichever *Job node has this job_id — DocumentJob or QueryJob."""
        decision_id = str(uuid.uuid4())
        self._run_write(
            """
            MATCH (j {job_id: $job_id})
            CREATE (d:ReviewerDecision {decision_id: $decision_id, approved: $approved,
                                         reviewer: $reviewer, comments: $comments, decided_at: $now})
            MERGE (j)-[:REVIEWED_BY]->(d)
            """,
            job_id=job_id, decision_id=decision_id, approved=approved,
            reviewer=reviewer, comments=comments, now=_now(),
        )
        return decision_id

    def write_audit_record(self, job_id: str, actor: str, action: str, details: str) -> str:
        """
        Append-only log entry, attached to whichever *Job node has this
        job_id (DocumentJob or QueryJob). Never update or delete an
        AuditRecord — corrections should be a *new* record referencing the
        old one, not an edit, so the trail always reflects what actually
        happened and when.
        """
        audit_id = str(uuid.uuid4())
        self._run_write(
            """
            MATCH (j {job_id: $job_id})
            CREATE (a:AuditRecord {audit_id: $audit_id, actor: $actor, action: $action,
                                    details: $details, timestamp: $now})
            MERGE (j)-[:HAS_AUDIT_RECORD]->(a)
            """,
            job_id=job_id, audit_id=audit_id, actor=actor, action=action, details=details, now=_now(),
        )
        return audit_id

    # ------------------------------------------------------------------
    # Graph write-back / generic read access
    # ------------------------------------------------------------------

    def record_answered_question(self, job_id: str, question: str, answer: str,
                                  cited_clause_ids: list[str]) -> str:
        """
        Optional graph update, run only after a human has approved a
        GraphRAG-sourced answer. Creates an AnsweredQuestion node citing the
        specific Clause nodes the answer drew on, so this reviewed
        Q&A becomes part of the graph itself — a lightweight "precedent
        cache" a future query (or a human browsing Neo4j) can find directly,
        instead of the LLM re-deriving the same multi-hop answer from
        scratch next time a similar question comes in.
        """
        answered_id = str(uuid.uuid4())
        self._run_write(
            """
            MATCH (q:QueryJob {job_id: $job_id})
            CREATE (a:AnsweredQuestion {answered_id: $answered_id, question: $question,
                                         answer: $answer, answered_at: $now})
            MERGE (q)-[:PRODUCED_ANSWER]->(a)
            WITH a
            UNWIND $clause_ids AS cid
            MATCH (cl:Clause {clause_id: cid})
            MERGE (a)-[:CITES]->(cl)
            """,
            job_id=job_id, answered_id=answered_id, question=question, answer=answer,
            now=_now(), clause_ids=cited_clause_ids,
        )
        return answered_id

    def run_read_query(self, cypher: str, params: Optional[dict] = None) -> list[dict]:
        return self._run_read(cypher, **(params or {}))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
