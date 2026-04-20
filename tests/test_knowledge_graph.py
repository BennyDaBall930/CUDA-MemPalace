"""
test_knowledge_graph.py — Tests for the temporal knowledge graph.

Covers: entity CRUD, triple CRUD, temporal queries, invalidation,
timeline, stats, and edge cases (duplicate triples, ID collisions).
"""

from mempalace.knowledge_graph import KnowledgeGraph


class TestEntityOperations:
    def test_add_entity(self, kg):
        eid = kg.add_entity("Alice", entity_type="person")
        assert eid == "alice"

    def test_add_entity_normalizes_id(self, kg):
        eid = kg.add_entity("Dr. Chen", entity_type="person")
        assert eid == "dr._chen"

    def test_add_entity_upsert(self, kg):
        kg.add_entity("Alice", entity_type="person")
        kg.add_entity("Alice", entity_type="engineer")
        # Should not raise — INSERT OR REPLACE
        stats = kg.stats()
        assert stats["entities"] == 1


class TestTripleOperations:
    def test_add_triple_creates_entities(self, kg):
        tid = kg.add_triple("Alice", "knows", "Bob")
        assert tid.startswith("t_alice_knows_bob_")
        stats = kg.stats()
        assert stats["entities"] == 2  # auto-created

    def test_add_triple_with_dates(self, kg):
        tid = kg.add_triple("Max", "does", "swimming", valid_from="2025-01-01")
        assert tid.startswith("t_max_does_swimming_")

    def test_duplicate_triple_returns_existing_id(self, kg):
        tid1 = kg.add_triple("Alice", "knows", "Bob")
        tid2 = kg.add_triple("Alice", "knows", "Bob")
        assert tid1 == tid2

    def test_invalidated_triple_allows_re_add(self, kg):
        tid1 = kg.add_triple("Alice", "works_at", "Acme")
        kg.invalidate("Alice", "works_at", "Acme", ended="2025-01-01")
        tid2 = kg.add_triple("Alice", "works_at", "Acme")
        assert tid1 != tid2  # new triple since old one was closed


class TestQueries:
    def test_query_outgoing(self, seeded_kg):
        results = seeded_kg.query_entity("Alice", direction="outgoing")
        predicates = {r["predicate"] for r in results}
        assert "parent_of" in predicates
        assert "works_at" in predicates

    def test_query_incoming(self, seeded_kg):
        results = seeded_kg.query_entity("Max", direction="incoming")
        assert any(r["subject"] == "Alice" and r["predicate"] == "parent_of" for r in results)

    def test_query_both_directions(self, seeded_kg):
        results = seeded_kg.query_entity("Max", direction="both")
        directions = {r["direction"] for r in results}
        assert "outgoing" in directions
        assert "incoming" in directions

    def test_query_as_of_filters_expired(self, seeded_kg):
        results = seeded_kg.query_entity("Alice", as_of="2023-06-01", direction="outgoing")
        employers = [r["object"] for r in results if r["predicate"] == "works_at"]
        assert "Acme Corp" in employers
        assert "NewCo" not in employers

    def test_query_as_of_shows_current(self, seeded_kg):
        results = seeded_kg.query_entity("Alice", as_of="2025-06-01", direction="outgoing")
        employers = [r["object"] for r in results if r["predicate"] == "works_at"]
        assert "NewCo" in employers
        assert "Acme Corp" not in employers

    def test_query_relationship(self, seeded_kg):
        results = seeded_kg.query_relationship("does")
        assert len(results) == 2  # swimming + chess


class TestInvalidation:
    def test_invalidate_sets_valid_to(self, seeded_kg):
        seeded_kg.invalidate("Max", "does", "chess", ended="2026-01-01")
        results = seeded_kg.query_entity("Max", direction="outgoing")
        chess = [r for r in results if r["object"] == "chess"]
        assert len(chess) == 1
        assert chess[0]["valid_to"] == "2026-01-01"
        assert chess[0]["current"] is False


class TestTimeline:
    def test_timeline_all(self, seeded_kg):
        tl = seeded_kg.timeline()
        assert len(tl) >= 4

    def test_timeline_entity(self, seeded_kg):
        tl = seeded_kg.timeline("Max")
        subjects_and_objects = {t["subject"] for t in tl} | {t["object"] for t in tl}
        assert "Max" in subjects_and_objects

    def test_timeline_global_has_limit(self, kg):
        # Add > 100 triples
        for i in range(105):
            kg.add_triple(f"entity_{i}", "relates_to", f"entity_{i + 1}")
        tl = kg.timeline()
        assert len(tl) == 100  # LIMIT 100

    def test_timeline_entity_has_limit(self, kg):
        # Add > 100 triples all connected to a single entity
        for i in range(105):
            kg.add_triple(
                "hub", "connects_to", f"spoke_{i}", valid_from=f"2025-01-{(i % 28) + 1:02d}"
            )
        tl = kg.timeline("hub")
        assert len(tl) == 100  # LIMIT 100 on entity-filtered branch


class TestWALMode:
    def test_wal_mode_enabled(self, kg):
        conn = kg._conn()
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        conn.close()
        assert mode == "wal"


class TestStats:
    def test_stats_empty(self, kg):
        stats = kg.stats()
        assert stats["entities"] == 0
        assert stats["triples"] == 0

    def test_stats_seeded(self, seeded_kg):
        stats = seeded_kg.stats()
        assert stats["entities"] >= 4
        assert stats["triples"] == 5
        assert stats["current_facts"] == 4  # 1 expired (Acme Corp)
        assert stats["expired_facts"] == 1


class TestFreshStartStructuredMemory:
    def test_scoped_typed_fact_recall_separates_answer_and_support(self, kg):
        fact_id = kg.add_fact(
            "MemPalace",
            "uses",
            "CUDA exact search",
            fact_type="implementation",
            scope="cuda",
            source_drawer_id="drawer_cuda_1",
            source_file="design.md",
            support_text="CUDA exact search passed parity tests.",
        )

        global_recall = kg.structured_recall("MemPalace", scope="global")
        scoped_recall = kg.structured_recall("MemPalace", scope="cuda")

        assert global_recall["answer_facts"] == []
        assert scoped_recall["answer_facts"][0]["id"] == fact_id
        assert scoped_recall["answer_facts"][0]["fact_type"] == "implementation"
        assert scoped_recall["answer_facts"][0]["scope"] == "cuda"
        assert scoped_recall["support"][0]["fact_id"] == fact_id
        assert scoped_recall["support"][0]["source_drawer_id"] == "drawer_cuda_1"
        assert "hidden answer-deciding truth" in scoped_recall["policy"]

    def test_supersede_closes_old_fact_and_links_replacement(self, kg):
        old_id = kg.add_fact("Alice", "works_at", "OldCo", scope="career")

        result = kg.supersede_fact(
            "Alice",
            "works_at",
            "OldCo",
            "NewCo",
            valid_from="2026-04-15",
            scope="career",
            support_text="Alice said she moved to NewCo.",
        )

        assert result["superseded_fact_ids"] == [old_id]
        current = kg.structured_recall("Alice", scope="career")["answer_facts"]
        assert [(fact["predicate"], fact["object"]) for fact in current] == [("works_at", "NewCo")]

        history = kg.query_entity("Alice", direction="outgoing", scope="career")
        old_fact = [fact for fact in history if fact["object"] == "OldCo"][0]
        new_fact = [fact for fact in history if fact["object"] == "NewCo"][0]
        assert old_fact["current"] is False
        assert old_fact["superseded_by"] == new_fact["id"]
        assert new_fact["supersedes"] == old_id

    def test_cleanup_removes_only_orphan_entities(self, kg):
        kg.add_entity("Orphan", entity_type="concept")
        kg.add_fact("Alice", "knows", "Bob")

        report = kg.maintenance_report()
        assert report["orphan_entities"] == 1

        dry_run = kg.cleanup(dry_run=True)
        assert dry_run["changed"] == 0
        assert "orphan" in dry_run["orphan_entities"]

        cleanup = kg.cleanup(dry_run=False)
        assert cleanup["changed"] == 1
        assert kg.maintenance_report()["orphan_entities"] == 0

    def test_consolidate_marks_duplicate_active_facts_without_deleting_truth(self, kg):
        fact_id = kg.add_fact("Alice", "likes", "coffee", scope="prefs")
        conn = kg._conn()
        conn.execute(
            """
            INSERT INTO triples (
                id, subject, predicate, object, fact_type, scope, status
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            ("duplicate_fact", "alice", "likes", "coffee", "relation", "prefs", "active"),
        )
        conn.commit()

        report = kg.maintenance_report()
        assert report["duplicate_active_fact_groups"] == 1

        result = kg.consolidate(dry_run=False)
        assert result["changed"] == 1
        assert result["duplicate_facts"] == ["duplicate_fact"]
        current = kg.structured_recall("Alice", scope="prefs")["answer_facts"]
        assert [fact["id"] for fact in current] == [fact_id]

    def test_replay_events_recovers_structured_facts(self, kg, tmp_path):
        kg.add_fact(
            "Project",
            "has_status",
            "diagnostic",
            scope="cuda",
            support_text="Parity first.",
        )
        events = kg.export_replay_events()

        recovered = KnowledgeGraph(db_path=str(tmp_path / "recovered.sqlite3"))
        result = recovered.replay_events(events, clear_first=True)

        assert result["applied"] == 1
        recall = recovered.structured_recall("Project", scope="cuda")
        assert recall["answer_facts"][0]["object"] == "diagnostic"
        assert recall["support"][0]["support_text"] == "Parity first."
