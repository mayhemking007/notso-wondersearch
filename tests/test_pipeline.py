from dataclasses import replace
from pathlib import Path
import pytest
from app.config import Settings
from app.indexing.sqlite_index import SQLiteIndex
from app.ingestion.chunker import chunk
from app.ingestion.extractor import normalize
from app.retrieval.bm25 import retrieve
from app.retrieval.fusion import fuse
from app.retrieval.query_expander import QueryExpander
from app.retrieval.search import SearchService


@pytest.fixture
def service(tmp_path):
    return SearchService(Settings(data_dir=tmp_path / "data", expansion_enabled=False, rerank_enabled=False))


def test_incremental_persistence_and_deletion(service, tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    file = docs / "orders.md"
    file.write_text("Idempotency keys prevent duplicate orders.", encoding="utf-8")
    assert service.index_directory(str(docs))["documents_indexed"] == 1
    initial = service.search("idempotency")["results"][0]
    assert service.index_directory(str(docs))["documents_skipped"] == 1
    restarted = SearchService(service.settings)
    assert restarted.search("idempotency")["results"][0]["passage_id"] == initial["passage_id"]
    file.write_text("Backups support recovery.", encoding="utf-8")
    service.index_directory(str(docs))
    assert not service.search("idempotency")["results"]
    assert service.search("recovery")["results"]
    file.unlink()
    assert service.index_directory(str(docs))["documents_deleted"] == 1
    assert service.index.counts() == {"documents": 0, "passages": 0}
    assert not service.search("recovery")["results"]


def test_failed_extraction_keeps_previous_document(service, tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    file = docs / "x.txt"
    file.write_text("persistent content", encoding="utf-8")
    service.index_directory(str(docs))
    file.write_bytes(b"\xff\xfe\xff")
    assert service.index_directory(str(docs))["errors"]
    assert service.search("persistent")["results"]


@pytest.mark.parametrize("query", ['C++ AND OR NOT "broken', "* : () - ^ NEAR /", "数据库", "'; DROP TABLE passages;--"])
def test_special_characters(service, query):
    service.search(query)
    assert service.index.counts()["passages"] == 0


def test_chunking():
    text = normalize("One  paragraph.\n\nSecond\nparagraph.")
    assert text == "One paragraph.\n\nSecond paragraph."
    assert "\n\n" in chunk(text)[0]
    pieces = chunk(" ".join(map(str, range(450))), 200, 30)
    assert [len(p.split()) for p in pieces] == [200, 200, 110]
    assert pieces[0].split()[-30:] == pieces[1].split()[:30]
    assert chunk("") == []


def test_rrf_merges_and_deduplicates():
    a, b = {"passage_id": "a"}, {"passage_id": "b"}
    output = fuse([[a, b], [b]])
    assert output[0]["passage_id"] == "b"
    assert output[0]["retrieval_score"] == pytest.approx(1/62 + 1/61)
    assert len(output) == 2


def test_expansion_cache_and_fallback(service, monkeypatch):
    settings = replace(service.settings, expansion_enabled=True, expansion_limit=2)
    expander = QueryExpander(settings, service.index)
    monkeypatch.setattr(expander, "_generate", lambda q: [("duplicate", 5), ("idempotency", 4), ("idempotency", 3), ("deduplication", 2)])
    assert expander.expand("duplicate orders") == ["idempotency", "deduplication"]
    assert expander.expand_weighted("duplicate orders") == [("idempotency", pytest.approx(2/3)), ("deduplication", pytest.approx(1/3))]
    def fail(query):
        raise RuntimeError("offline")
    monkeypatch.setattr(expander, "_generate", fail)
    assert expander.expand("duplicate orders") == ["idempotency", "deduplication"]
    assert expander.expand("uncached query") == []
    assert expander.error == "offline"


def test_semantic_branch_and_ranking(service, monkeypatch):
    service.index.replace_document("a", "fixture", "orders", "a", [(None, "Idempotency keys")])
    service.index.replace_document("b", "fixture", "backup", "b", [(None, "Backup keys")])
    monkeypatch.setattr(service.expander, "expand_weighted", lambda query: [("keys", 1.0)])
    def rank(query, candidates):
        assert query == "prevent duplicate purchases"
        for item in candidates:
            item["rerank_score"] = 9.0 if item["document_id"] == "a" else -1.0
        return sorted(candidates, key=lambda item: -item["rerank_score"])
    monkeypatch.setattr(service.reranker, "rank", rank)
    response = service.search("prevent duplicate purchases", 1)
    assert response["results"][0]["document_id"] == "a"
    assert response["results"][0]["rerank_score"] == 9.0


def test_indexing_does_not_load_models(service, tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text("hello", encoding="utf-8")
    service.index_directory(str(docs))
    assert service.expander.model is None
    assert service.reranker.model is None


def test_root_scoped_deletion(service, tmp_path):
    for name in ("one", "two"):
        root = tmp_path / name
        root.mkdir()
        (root / "a.txt").write_text(name, encoding="utf-8")
        service.index_directory(str(root))
    (tmp_path / "one" / "a.txt").unlink()
    service.index_directory(str(tmp_path / "one"))
    assert service.index.counts()["documents"] == 1
    assert service.search("two")["results"]


def test_failed_traversal_does_not_prune(service, tmp_path, monkeypatch):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text("retained", encoding="utf-8")
    service.index_directory(str(docs))
    def fail_walk(*args, **kwargs):
        kwargs["onerror"](PermissionError("inaccessible subtree"))
        return iter(())
    monkeypatch.setattr("os.walk", fail_walk)
    with pytest.raises(PermissionError):
        service.index_directory(str(docs))
    assert service.search("retained")["results"]
