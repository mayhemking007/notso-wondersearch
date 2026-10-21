import json
import os
from dataclasses import replace
from app.indexing.sqlite_index import SQLiteIndex, stable_id
from app.retrieval.fusion import fuse
from app.retrieval.query_expander import QueryExpander, select_terms
from app.config import Settings


def test_legacy_duplicate_ids_migrate_without_reindexing(tmp_path):
    root = tmp_path / "docs"
    root.mkdir()
    source = root / "orders.md"
    source.write_text("idempotency", encoding="utf-8")
    path = tmp_path / "index.sqlite3"
    index = SQLiteIndex(path)
    canonical_root = os.path.normcase(str(root.resolve()))
    identity = stable_id(f"{canonical_root}:{os.path.normcase('orders.md')}")
    index.replace_document("old-id", str(root), str(source), "stale", [(1, "old contents")])
    index.replace_document(identity, canonical_root, str(source), "current", [(2, "idempotency")])
    # Mimic a database created before the migration existed.
    with index.connect() as db:
        db.execute("PRAGMA user_version=0")
    reopened = SQLiteIndex(path)
    assert reopened.counts() == {"documents": 1, "passages": 1}
    with reopened.connect() as db:
        row = db.execute("SELECT * FROM passages").fetchone()
        assert row["document_id"] == identity
        assert row["passage_text"] == "idempotency"
        assert row["page_number"] == 2
        assert db.execute("SELECT count(*) FROM passages_fts WHERE passages_fts MATCH 'old'").fetchone()[0] == 0
        assert db.execute("SELECT count(*) FROM passages_fts WHERE passages_fts MATCH 'idempotency'").fetchone()[0] == 1
    assert SQLiteIndex(path).counts() == reopened.counts()


def test_migration_preserves_official_corpus_ids(tmp_path):
    path = tmp_path / "index.sqlite3"
    index = SQLiteIndex(path)
    index.replace_document("official-42", "scifact", "scifact:official-42", "x", [(None, "science")])
    with index.connect() as db:
        db.execute("PRAGMA user_version=0")
    reopened = SQLiteIndex(path)
    with reopened.connect() as db:
        assert db.execute("SELECT document_id FROM documents").fetchone()[0] == "official-42"


def test_expansion_filters_fragments_echoes_and_invalid_weights():
    assert select_terms(
        ["[CLS]", "##ation", "the", "orders", "deduplication", "request", "Use a semicolons to prevent duplicate orders.", "nanword"],
        [20, 15, 10, 10, 3, 2, 100, float("nan")],
        "How do I prevent duplicate orders?", 3) == [("deduplication", 3.0), ("request", 2.0)]


def test_old_generated_sentence_cache_is_not_reused(tmp_path, monkeypatch):
    settings = Settings(data_dir=tmp_path)
    index = SQLiteIndex(tmp_path / "index.sqlite3")
    query = "How do I prevent duplicate orders?"
    old_key = stable_id(json.dumps([query, "google/flan-t5-small", "main", 3, "prompt-v1"]))
    index.cache_expansions(old_key, ["Use a semicolons to prevent duplicate orders."])
    expander = QueryExpander(settings, index)
    monkeypatch.setattr(expander, "_generate", lambda q: [("deduplication", 3), ("request", 1)])
    assert expander.expand_weighted(query) == [("deduplication", 0.75), ("request", 0.25)]


def test_model_weights_affect_fusion():
    rankings = [[{"passage_id": "original"}], [{"passage_id": "weak"}], [{"passage_id": "strong"}]]
    result = fuse(rankings, weights=[1.0, 0.1, 0.9])
    assert [r["passage_id"] for r in result] == ["original", "strong", "weak"]
