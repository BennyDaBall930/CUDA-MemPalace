"""
knowledge_graph.py — Temporal Entity-Relationship Graph for MemPalace
=====================================================================

Real knowledge graph with:
  - Entity nodes (people, projects, tools, concepts)
  - Typed relationship edges (daughter_of, does, loves, works_on, etc.)
  - Temporal validity (valid_from → valid_to — knows WHEN facts are true)
  - Closet references (links back to the verbatim memory)

Storage: SQLite (local, no dependencies, no subscriptions)
Query: entity-first traversal with time filtering

This is what competes with Zep's temporal knowledge graph.
Zep uses Neo4j in the cloud ($25/mo+). We use SQLite locally (free).

Usage:
    from mempalace.knowledge_graph import KnowledgeGraph

    kg = KnowledgeGraph()
    kg.add_triple("Max", "child_of", "Alice", valid_from="2015-04-01")
    kg.add_triple("Max", "does", "swimming", valid_from="2025-01-01")
    kg.add_triple("Max", "loves", "chess", valid_from="2025-10-01")

    # Query: everything about Max
    kg.query_entity("Max")

    # Query: what was true about Max in January 2026?
    kg.query_entity("Max", as_of="2026-01-15")

    # Query: who is connected to Alice?
    kg.query_entity("Alice", direction="both")

    # Invalidate: Max's sports injury resolved
    kg.invalidate("Max", "has_issue", "sports_injury", ended="2026-02-15")
"""

import hashlib
import json
import os
import sqlite3
import threading
from datetime import date, datetime
from pathlib import Path


DEFAULT_KG_PATH = os.path.expanduser("~/.mempalace/knowledge_graph.sqlite3")


class KnowledgeGraph:
    def __init__(self, db_path: str = None):
        self.db_path = db_path or DEFAULT_KG_PATH
        db_parent = Path(self.db_path).parent
        db_parent.mkdir(parents=True, exist_ok=True)
        try:
            db_parent.chmod(0o700)
        except (OSError, NotImplementedError):
            pass
        self._connection = None
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self):
        conn = self._conn()
        conn.executescript("""
            PRAGMA journal_mode=WAL;

            CREATE TABLE IF NOT EXISTS entities (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                type TEXT DEFAULT 'unknown',
                properties TEXT DEFAULT '{}',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS triples (
                id TEXT PRIMARY KEY,
                subject TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object TEXT NOT NULL,
                fact_type TEXT DEFAULT 'relation',
                scope TEXT DEFAULT 'global',
                valid_from TEXT,
                valid_to TEXT,
                confidence REAL DEFAULT 1.0,
                source_closet TEXT,
                source_drawer_id TEXT,
                source_file TEXT,
                support_text TEXT,
                status TEXT DEFAULT 'active',
                supersedes TEXT,
                superseded_by TEXT,
                extracted_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (subject) REFERENCES entities(id),
                FOREIGN KEY (object) REFERENCES entities(id)
            );

            CREATE INDEX IF NOT EXISTS idx_triples_subject ON triples(subject);
            CREATE INDEX IF NOT EXISTS idx_triples_object ON triples(object);
            CREATE INDEX IF NOT EXISTS idx_triples_predicate ON triples(predicate);
            CREATE INDEX IF NOT EXISTS idx_triples_valid ON triples(valid_from, valid_to);
            CREATE INDEX IF NOT EXISTS idx_triples_scope ON triples(scope);
            CREATE INDEX IF NOT EXISTS idx_triples_status ON triples(status);

            CREATE TABLE IF NOT EXISTS kg_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                operation TEXT NOT NULL,
                payload TEXT NOT NULL
            );
        """)
        self._migrate_triples(conn)
        conn.commit()

    def _migrate_triples(self, conn):
        """Add Fresh Start structured-memory columns to older KG databases."""
        existing_columns = {
            row["name"] for row in conn.execute("PRAGMA table_info(triples)").fetchall()
        }
        column_defs = {
            "fact_type": "TEXT DEFAULT 'relation'",
            "scope": "TEXT DEFAULT 'global'",
            "source_drawer_id": "TEXT",
            "support_text": "TEXT",
            "status": "TEXT DEFAULT 'active'",
            "supersedes": "TEXT",
            "superseded_by": "TEXT",
            "updated_at": "TEXT DEFAULT CURRENT_TIMESTAMP",
        }
        for column, definition in column_defs.items():
            if column not in existing_columns:
                conn.execute(f"ALTER TABLE triples ADD COLUMN {column} {definition}")

    def _conn(self):
        if self._connection is None:
            self._connection = sqlite3.connect(self.db_path, timeout=10, check_same_thread=False)
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.row_factory = sqlite3.Row
        return self._connection

    def close(self):
        """Close the database connection."""
        with self._lock:
            if self._connection is not None:
                self._connection.close()
                self._connection = None

    def _entity_id(self, name: str) -> str:
        return name.lower().replace(" ", "_").replace("'", "")

    def _normalize_predicate(self, predicate: str) -> str:
        return predicate.lower().replace(" ", "_")

    def _normalize_scope(self, scope: str = None) -> str:
        return (scope or "global").strip() or "global"

    def _log_event(self, operation: str, payload: dict):
        with self._lock:
            conn = self._conn()
            with conn:
                conn.execute(
                    "INSERT INTO kg_events (operation, payload) VALUES (?, ?)",
                    (operation, json.dumps(payload, sort_keys=True)),
                )

    def _row_to_fact(
        self,
        row,
        *,
        direction: str = "outgoing",
        subject_name: str = None,
        object_name: str = None,
    ) -> dict:
        support = []
        if row["source_closet"]:
            support.append({"kind": "closet", "id": row["source_closet"]})
        if row["source_drawer_id"]:
            support.append({"kind": "drawer", "id": row["source_drawer_id"]})
        if row["source_file"]:
            support.append({"kind": "file", "id": row["source_file"]})
        if row["support_text"]:
            support.append({"kind": "text", "text": row["support_text"]})

        return {
            "id": row["id"],
            "direction": direction,
            "subject": subject_name,
            "predicate": row["predicate"],
            "object": object_name,
            "fact_type": row["fact_type"] or "relation",
            "scope": row["scope"] or "global",
            "valid_from": row["valid_from"],
            "valid_to": row["valid_to"],
            "confidence": row["confidence"],
            "source_closet": row["source_closet"],
            "source_drawer_id": row["source_drawer_id"],
            "source_file": row["source_file"],
            "support_text": row["support_text"],
            "support": support,
            "status": row["status"] or "active",
            "supersedes": row["supersedes"],
            "superseded_by": row["superseded_by"],
            "current": row["valid_to"] is None and (row["status"] or "active") == "active",
        }

    # ── Write operations ──────────────────────────────────────────────────

    def add_entity(self, name: str, entity_type: str = "unknown", properties: dict = None):
        """Add or update an entity node."""
        eid = self._entity_id(name)
        props = json.dumps(properties or {})
        with self._lock:
            conn = self._conn()
            with conn:
                conn.execute(
                    "INSERT OR REPLACE INTO entities (id, name, type, properties) VALUES (?, ?, ?, ?)",
                    (eid, name, entity_type, props),
                )
        return eid

    def add_triple(
        self,
        subject: str,
        predicate: str,
        obj: str,
        valid_from: str = None,
        valid_to: str = None,
        confidence: float = 1.0,
        source_closet: str = None,
        source_drawer_id: str = None,
        source_file: str = None,
        fact_type: str = "relation",
        scope: str = "global",
        support_text: str = None,
        supersedes: str = None,
        status: str = "active",
        log_event: bool = True,
    ):
        """
        Add a relationship triple: subject → predicate → object.

        Examples:
            add_triple("Max", "child_of", "Alice", valid_from="2015-04-01")
            add_triple("Max", "does", "swimming", valid_from="2025-01-01")
            add_triple("Alice", "worried_about", "Max injury", valid_from="2026-01", valid_to="2026-02")
        """
        sub_id = self._entity_id(subject)
        obj_id = self._entity_id(obj)
        pred = self._normalize_predicate(predicate)
        scope = self._normalize_scope(scope)
        fact_type = (fact_type or "relation").strip() or "relation"
        status = (status or "active").strip() or "active"

        # Auto-create entities if they don't exist
        with self._lock:
            conn = self._conn()
            with conn:
                conn.execute(
                    "INSERT OR IGNORE INTO entities (id, name) VALUES (?, ?)", (sub_id, subject)
                )
                conn.execute(
                    "INSERT OR IGNORE INTO entities (id, name) VALUES (?, ?)", (obj_id, obj)
                )

                # Check for existing identical triple
                existing = conn.execute(
                    """SELECT id FROM triples
                       WHERE subject=? AND predicate=? AND object=? AND scope=?
                         AND fact_type=? AND valid_to IS NULL AND status='active'""",
                    (sub_id, pred, obj_id, scope, fact_type),
                ).fetchone()

                if existing:
                    return existing["id"]  # Already exists and still valid

                triple_id = f"t_{sub_id}_{pred}_{obj_id}_{hashlib.sha256(f'{valid_from}{datetime.now().isoformat()}'.encode()).hexdigest()[:12]}"

                conn.execute(
                    """INSERT INTO triples (
                           id, subject, predicate, object, fact_type, scope,
                           valid_from, valid_to, confidence, source_closet,
                           source_drawer_id, source_file, support_text, status, supersedes
                       )
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        triple_id,
                        sub_id,
                        pred,
                        obj_id,
                        fact_type,
                        scope,
                        valid_from,
                        valid_to,
                        confidence,
                        source_closet,
                        source_drawer_id,
                        source_file,
                        support_text,
                        status,
                        supersedes,
                    ),
                )
        if log_event:
            self._log_event(
                "add_triple",
                {
                    "subject": subject,
                    "predicate": predicate,
                    "object": obj,
                    "valid_from": valid_from,
                    "valid_to": valid_to,
                    "confidence": confidence,
                    "source_closet": source_closet,
                    "source_drawer_id": source_drawer_id,
                    "source_file": source_file,
                    "fact_type": fact_type,
                    "scope": scope,
                    "support_text": support_text,
                    "supersedes": supersedes,
                    "status": status,
                },
            )
        return triple_id

    def add_fact(self, subject: str, predicate: str, obj: str, **kwargs):
        """Add a typed/scoped fact while preserving the legacy triple API."""
        return self.add_triple(subject, predicate, obj, **kwargs)

    def invalidate(
        self,
        subject: str,
        predicate: str,
        obj: str,
        ended: str = None,
        scope: str = None,
        reason: str = None,
        log_event: bool = True,
    ):
        """Mark a relationship as no longer valid (set valid_to date)."""
        sub_id = self._entity_id(subject)
        obj_id = self._entity_id(obj)
        pred = self._normalize_predicate(predicate)
        ended = ended or date.today().isoformat()

        with self._lock:
            conn = self._conn()
            with conn:
                query = (
                    "UPDATE triples SET valid_to=?, status='expired', updated_at=CURRENT_TIMESTAMP "
                    "WHERE subject=? AND predicate=? AND object=? AND valid_to IS NULL"
                )
                params = [ended, sub_id, pred, obj_id]
                if scope is not None:
                    query += " AND scope=?"
                    params.append(self._normalize_scope(scope))
                cursor = conn.execute(query, params)
                changed = cursor.rowcount
        if log_event:
            self._log_event(
                "invalidate",
                {
                    "subject": subject,
                    "predicate": predicate,
                    "object": obj,
                    "ended": ended,
                    "scope": scope,
                    "reason": reason,
                },
            )
        return changed

    def supersede_fact(
        self,
        subject: str,
        predicate: str,
        old_object: str,
        new_object: str,
        *,
        valid_from: str = None,
        ended: str = None,
        scope: str = "global",
        reason: str = None,
        log_event: bool = True,
        **kwargs,
    ):
        """Close an old fact and add the replacement with an explicit supersession link."""
        sub_id = self._entity_id(subject)
        pred = self._normalize_predicate(predicate)
        old_obj_id = self._entity_id(old_object)
        scope = self._normalize_scope(scope)
        ended = ended or valid_from or date.today().isoformat()

        with self._lock:
            conn = self._conn()
            old_rows = conn.execute(
                """SELECT id FROM triples
                   WHERE subject=? AND predicate=? AND object=? AND scope=?
                     AND valid_to IS NULL AND status='active'""",
                (sub_id, pred, old_obj_id, scope),
            ).fetchall()
        superseded_ids = [row["id"] for row in old_rows]
        self.invalidate(
            subject,
            predicate,
            old_object,
            ended=ended,
            scope=scope,
            reason=reason or "superseded",
            log_event=False,
        )
        new_id = self.add_triple(
            subject,
            predicate,
            new_object,
            valid_from=valid_from,
            scope=scope,
            supersedes=",".join(superseded_ids) if superseded_ids else None,
            log_event=False,
            **kwargs,
        )
        if superseded_ids:
            with self._lock:
                conn = self._conn()
                with conn:
                    conn.executemany(
                        "UPDATE triples SET superseded_by=?, updated_at=CURRENT_TIMESTAMP WHERE id=?",
                        [(new_id, old_id) for old_id in superseded_ids],
                    )
        if log_event:
            self._log_event(
                "supersede_fact",
                {
                    "subject": subject,
                    "predicate": predicate,
                    "old_object": old_object,
                    "new_object": new_object,
                    "valid_from": valid_from,
                    "ended": ended,
                    "scope": scope,
                    "reason": reason,
                    "kwargs": kwargs,
                },
            )
        return {"new_fact_id": new_id, "superseded_fact_ids": superseded_ids}

    # ── Query operations ──────────────────────────────────────────────────

    def query_entity(
        self,
        name: str,
        as_of: str = None,
        direction: str = "outgoing",
        scope: str = None,
        current_only: bool = False,
    ):
        """
        Get all relationships for an entity.

        direction: "outgoing" (entity → ?), "incoming" (? → entity), "both"
        as_of: date string — only return facts valid at that time
        """
        eid = self._entity_id(name)
        scope = self._normalize_scope(scope) if scope is not None else None

        results = []
        with self._lock:
            conn = self._conn()

            if direction in ("outgoing", "both"):
                query = "SELECT t.*, e.name as obj_name FROM triples t JOIN entities e ON t.object = e.id WHERE t.subject = ?"
                params = [eid]
                if scope is not None:
                    query += " AND t.scope = ?"
                    params.append(scope)
                if as_of:
                    query += " AND (t.valid_from IS NULL OR t.valid_from <= ?) AND (t.valid_to IS NULL OR t.valid_to >= ?)"
                    params.extend([as_of, as_of])
                if current_only:
                    query += " AND t.valid_to IS NULL AND t.status = 'active'"
                for row in conn.execute(query, params).fetchall():
                    results.append(
                        self._row_to_fact(
                            row,
                            direction="outgoing",
                            subject_name=name,
                            object_name=row["obj_name"],
                        )
                    )

            if direction in ("incoming", "both"):
                query = "SELECT t.*, e.name as sub_name FROM triples t JOIN entities e ON t.subject = e.id WHERE t.object = ?"
                params = [eid]
                if scope is not None:
                    query += " AND t.scope = ?"
                    params.append(scope)
                if as_of:
                    query += " AND (t.valid_from IS NULL OR t.valid_from <= ?) AND (t.valid_to IS NULL OR t.valid_to >= ?)"
                    params.extend([as_of, as_of])
                if current_only:
                    query += " AND t.valid_to IS NULL AND t.status = 'active'"
                for row in conn.execute(query, params).fetchall():
                    results.append(
                        self._row_to_fact(
                            row,
                            direction="incoming",
                            subject_name=row["sub_name"],
                            object_name=name,
                        )
                    )

        return results

    def query_relationship(self, predicate: str, as_of: str = None, scope: str = None):
        """Get all triples with a given relationship type."""
        pred = self._normalize_predicate(predicate)
        query = """
            SELECT t.*, s.name as sub_name, o.name as obj_name
            FROM triples t
            JOIN entities s ON t.subject = s.id
            JOIN entities o ON t.object = o.id
            WHERE t.predicate = ?
        """
        params = [pred]
        if scope is not None:
            query += " AND t.scope = ?"
            params.append(self._normalize_scope(scope))
        if as_of:
            query += " AND (t.valid_from IS NULL OR t.valid_from <= ?) AND (t.valid_to IS NULL OR t.valid_to >= ?)"
            params.extend([as_of, as_of])

        results = []
        with self._lock:
            conn = self._conn()
            for row in conn.execute(query, params).fetchall():
                results.append(
                    self._row_to_fact(
                        row,
                        subject_name=row["sub_name"],
                        object_name=row["obj_name"],
                    )
                )
        return results

    def timeline(self, entity_name: str = None):
        """Get all facts in chronological order, optionally filtered by entity."""
        with self._lock:
            conn = self._conn()
            if entity_name:
                eid = self._entity_id(entity_name)
                rows = conn.execute(
                    """
                    SELECT t.*, s.name as sub_name, o.name as obj_name
                    FROM triples t
                    JOIN entities s ON t.subject = s.id
                    JOIN entities o ON t.object = o.id
                    WHERE (t.subject = ? OR t.object = ?)
                    ORDER BY t.valid_from ASC NULLS LAST
                    LIMIT 100
                """,
                    (eid, eid),
                ).fetchall()
            else:
                rows = conn.execute("""
                    SELECT t.*, s.name as sub_name, o.name as obj_name
                    FROM triples t
                    JOIN entities s ON t.subject = s.id
                    JOIN entities o ON t.object = o.id
                    ORDER BY t.valid_from ASC NULLS LAST
                    LIMIT 100
                """).fetchall()

        return [
            {
                "id": r["id"],
                "subject": r["sub_name"],
                "predicate": r["predicate"],
                "object": r["obj_name"],
                "fact_type": r["fact_type"] or "relation",
                "scope": r["scope"] or "global",
                "valid_from": r["valid_from"],
                "valid_to": r["valid_to"],
                "status": r["status"] or "active",
                "current": r["valid_to"] is None and (r["status"] or "active") == "active",
            }
            for r in rows
        ]

    # ── Stats ─────────────────────────────────────────────────────────────

    def stats(self):
        with self._lock:
            conn = self._conn()
            entities = conn.execute("SELECT COUNT(*) as cnt FROM entities").fetchone()["cnt"]
            triples = conn.execute("SELECT COUNT(*) as cnt FROM triples").fetchone()["cnt"]
            current = conn.execute(
                "SELECT COUNT(*) as cnt FROM triples WHERE valid_to IS NULL AND status='active'"
            ).fetchone()["cnt"]
            expired = triples - current
            predicates = [
                r["predicate"]
                for r in conn.execute(
                    "SELECT DISTINCT predicate FROM triples ORDER BY predicate"
                ).fetchall()
            ]
            scopes = [
                r["scope"]
                for r in conn.execute(
                    "SELECT DISTINCT scope FROM triples ORDER BY scope"
                ).fetchall()
            ]
            fact_types = [
                r["fact_type"]
                for r in conn.execute(
                    "SELECT DISTINCT fact_type FROM triples ORDER BY fact_type"
                ).fetchall()
            ]
            superseded = conn.execute(
                "SELECT COUNT(*) as cnt FROM triples WHERE superseded_by IS NOT NULL"
            ).fetchone()["cnt"]
        return {
            "entities": entities,
            "triples": triples,
            "current_facts": current,
            "expired_facts": expired,
            "superseded_facts": superseded,
            "relationship_types": predicates,
            "scopes": scopes,
            "fact_types": fact_types,
        }

    def structured_recall(
        self,
        entity: str,
        *,
        as_of: str = None,
        direction: str = "both",
        scope: str = None,
    ) -> dict:
        """Return current typed facts separately from their support evidence."""
        facts = self.query_entity(
            entity,
            as_of=as_of,
            direction=direction,
            scope=scope,
            current_only=as_of is None,
        )
        answer_facts = [
            {
                "id": fact["id"],
                "subject": fact["subject"],
                "predicate": fact["predicate"],
                "object": fact["object"],
                "fact_type": fact["fact_type"],
                "scope": fact["scope"],
                "valid_from": fact["valid_from"],
                "valid_to": fact["valid_to"],
                "confidence": fact["confidence"],
                "current": fact["current"],
            }
            for fact in facts
        ]
        support = [
            {
                "fact_id": fact["id"],
                "source_closet": fact["source_closet"],
                "source_drawer_id": fact["source_drawer_id"],
                "source_file": fact["source_file"],
                "support_text": fact["support_text"],
                "support": fact["support"],
            }
            for fact in facts
            if fact["support"]
            or fact["source_closet"]
            or fact["source_drawer_id"]
            or fact["source_file"]
            or fact["support_text"]
        ]
        return {
            "entity": entity,
            "scope": scope or "all",
            "as_of": as_of,
            "answer_facts": answer_facts,
            "support": support,
            "count": len(answer_facts),
            "policy": (
                "Structured facts are explicit memory surfaces. Support entries are evidence "
                "for review, not hidden answer-deciding truth."
            ),
        }

    def maintenance_report(self) -> dict:
        """Return cleanup/consolidation/recovery diagnostics without mutating data."""
        with self._lock:
            conn = self._conn()
            duplicate_groups = conn.execute(
                """
                SELECT subject, predicate, object, scope, fact_type, COUNT(*) as cnt
                FROM triples
                WHERE valid_to IS NULL AND status='active'
                GROUP BY subject, predicate, object, scope, fact_type
                HAVING cnt > 1
                """
            ).fetchall()
            orphan_entities = conn.execute(
                """
                SELECT COUNT(*) as cnt
                FROM entities e
                WHERE NOT EXISTS (SELECT 1 FROM triples t WHERE t.subject=e.id OR t.object=e.id)
                """
            ).fetchone()["cnt"]
            event_count = conn.execute("SELECT COUNT(*) as cnt FROM kg_events").fetchone()["cnt"]
        return {
            "duplicate_active_fact_groups": len(duplicate_groups),
            "orphan_entities": orphan_entities,
            "replay_events": event_count,
            "recommendation": (
                "Run consolidate for duplicate active fact groups; run cleanup for orphan entities."
            ),
        }

    def consolidate(self, *, dry_run: bool = True) -> dict:
        """Mark duplicate active facts as duplicates while preserving the first fact."""
        with self._lock:
            conn = self._conn()
            groups = conn.execute(
                """
                SELECT subject, predicate, object, scope, fact_type
                FROM triples
                WHERE valid_to IS NULL AND status='active'
                GROUP BY subject, predicate, object, scope, fact_type
                HAVING COUNT(*) > 1
                """
            ).fetchall()
            duplicate_ids = []
            for group in groups:
                rows = conn.execute(
                    """
                    SELECT id FROM triples
                    WHERE subject=? AND predicate=? AND object=? AND scope=? AND fact_type=?
                      AND valid_to IS NULL AND status='active'
                    ORDER BY CASE WHEN id LIKE 't_%' THEN 0 ELSE 1 END, extracted_at ASC, id ASC
                    """,
                    (
                        group["subject"],
                        group["predicate"],
                        group["object"],
                        group["scope"],
                        group["fact_type"],
                    ),
                ).fetchall()
                duplicate_ids.extend(row["id"] for row in rows[1:])
            if not dry_run and duplicate_ids:
                with conn:
                    conn.executemany(
                        "UPDATE triples SET status='duplicate', updated_at=CURRENT_TIMESTAMP WHERE id=?",
                        [(fact_id,) for fact_id in duplicate_ids],
                    )
        if not dry_run and duplicate_ids:
            self._log_event("consolidate", {"duplicate_ids": duplicate_ids})
        return {
            "dry_run": dry_run,
            "duplicate_groups": len(groups),
            "duplicate_facts": duplicate_ids,
            "changed": 0 if dry_run else len(duplicate_ids),
        }

    def cleanup(self, *, dry_run: bool = True) -> dict:
        """Remove orphan entities only; facts are preserved or explicitly expired."""
        with self._lock:
            conn = self._conn()
            rows = conn.execute(
                """
                SELECT id FROM entities e
                WHERE NOT EXISTS (SELECT 1 FROM triples t WHERE t.subject=e.id OR t.object=e.id)
                ORDER BY id
                """
            ).fetchall()
            entity_ids = [row["id"] for row in rows]
            if not dry_run and entity_ids:
                with conn:
                    conn.executemany("DELETE FROM entities WHERE id=?", [(eid,) for eid in entity_ids])
        if not dry_run and entity_ids:
            self._log_event("cleanup", {"orphan_entity_ids": entity_ids})
        return {
            "dry_run": dry_run,
            "orphan_entities": entity_ids,
            "changed": 0 if dry_run else len(entity_ids),
        }

    def export_replay_events(self, limit: int = 1000) -> list[dict]:
        """Export structured KG write events for recovery/replay."""
        limit = max(1, int(limit))
        with self._lock:
            conn = self._conn()
            rows = conn.execute(
                "SELECT id, timestamp, operation, payload FROM kg_events ORDER BY id ASC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            {
                "id": row["id"],
                "timestamp": row["timestamp"],
                "operation": row["operation"],
                "payload": json.loads(row["payload"]),
            }
            for row in rows
        ]

    def replay_events(self, events: list[dict], *, clear_first: bool = False) -> dict:
        """Replay exported KG events into this database for recovery tests/tools."""
        if clear_first:
            with self._lock:
                conn = self._conn()
                with conn:
                    conn.execute("DELETE FROM triples")
                    conn.execute("DELETE FROM entities")
                    conn.execute("DELETE FROM kg_events")

        applied = 0
        skipped = 0
        for event in events or []:
            operation = event.get("operation")
            payload = dict(event.get("payload") or {})
            if operation == "add_triple":
                obj = payload.pop("object")
                self.add_triple(obj=obj, log_event=False, **payload)
                applied += 1
            elif operation == "invalidate":
                obj = payload.pop("object")
                self.invalidate(obj=obj, log_event=False, **payload)
                applied += 1
            elif operation == "supersede_fact":
                kwargs = payload.pop("kwargs", {}) or {}
                self.supersede_fact(log_event=False, **payload, **kwargs)
                applied += 1
            else:
                skipped += 1
        return {"applied": applied, "skipped": skipped, "cleared": clear_first}

    # ── Seed from known facts ─────────────────────────────────────────────

    def seed_from_entity_facts(self, entity_facts: dict):
        """
        Seed the knowledge graph from fact_checker.py ENTITY_FACTS.
        This bootstraps the graph with known ground truth.
        """
        for key, facts in entity_facts.items():
            name = facts.get("full_name", key.capitalize())
            etype = facts.get("type", "person")
            self.add_entity(
                name,
                etype,
                {
                    "gender": facts.get("gender", ""),
                    "birthday": facts.get("birthday", ""),
                },
            )

            # Relationships
            parent = facts.get("parent")
            if parent:
                self.add_triple(
                    name, "child_of", parent.capitalize(), valid_from=facts.get("birthday")
                )

            partner = facts.get("partner")
            if partner:
                self.add_triple(name, "married_to", partner.capitalize())

            relationship = facts.get("relationship", "")
            if relationship == "daughter":
                self.add_triple(
                    name,
                    "is_child_of",
                    facts.get("parent", "").capitalize() or name,
                    valid_from=facts.get("birthday"),
                )
            elif relationship == "husband":
                self.add_triple(name, "is_partner_of", facts.get("partner", name).capitalize())
            elif relationship == "brother":
                self.add_triple(name, "is_sibling_of", facts.get("sibling", name).capitalize())
            elif relationship == "dog":
                self.add_triple(name, "is_pet_of", facts.get("owner", name).capitalize())
                self.add_entity(name, "animal")

            # Interests
            for interest in facts.get("interests", []):
                self.add_triple(name, "loves", interest.capitalize(), valid_from="2025-01-01")
