import pytest
from fastapi.testclient import TestClient
from app.config import Settings
from app.main import create_app
from app.retrieval.search import SearchService
from app.benchmark.scifact import sample_ids


def test_api(tmp_path):
    service = SearchService(Settings(data_dir=tmp_path / "data", expansion_enabled=False, rerank_enabled=False))
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("Use idempotency keys.", encoding="utf-8")
    with TestClient(create_app(service)) as client:
        assert client.get("/health").json()["index"]["documents"] == 0
        assert client.post("/index", json={"docs_path": str(docs)}).json()["documents_indexed"] == 1
        assert client.post("/search", json={"query": "idempotency"}).json()["results"] == service.search("idempotency")["results"]
        assert client.post("/search", json={"query": "  "}).status_code == 422
        assert client.post("/search", json={"query": "x", "top_k": 21}).status_code == 422
        assert client.post("/index", json={"docs_path": str(tmp_path / "missing")}).status_code == 400


def test_sampling():
    ids = [str(i) for i in range(300)]
    assert sample_ids(ids, 30) == sample_ids(list(reversed(ids)), 30)
    assert sample_ids(ids, 5) == sample_ids(ids, 30)[:5]
    with pytest.raises(ValueError):
        sample_ids(ids, 301)


def test_pdf_page_metadata(tmp_path):
    import pymupdf
    from app.ingestion.extractor import extract
    pdf = tmp_path / "test.pdf"
    with pymupdf.open() as document:
        document.new_page().insert_text((72, 72), "Page one idempotency")
        document.new_page().insert_text((72, 72), "Page two recovery")
        document.save(pdf)
    pages = extract(pdf)
    assert pages[0][0] == 1 and "idempotency" in pages[0][1]
    assert pages[1][0] == 2 and "recovery" in pages[1][1]
