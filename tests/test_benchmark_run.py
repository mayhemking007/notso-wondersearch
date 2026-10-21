"""A tiny official-format fixture verifies evaluation IDs and metric wiring."""
import json
import zipfile
from dataclasses import replace
from app.benchmark.scifact import run
from app.config import Settings


def test_benchmark_official_ids_and_metrics(tmp_path):
    dataset_dir = tmp_path / "datasets"
    dataset_dir.mkdir()
    with zipfile.ZipFile(dataset_dir / "scifact.zip", "w") as archive:
        archive.writestr("scifact/corpus.jsonl", "\n".join(json.dumps(doc) for doc in [
            {"_id": "official-17", "title": "Biology", "text": "mitochondria produce energy"},
            {"_id": "official-42", "title": "Physics", "text": "photons carry light"}]))
        archive.writestr("scifact/queries.jsonl", json.dumps({"_id": "q1", "text": "mitochondria energy"}))
        archive.writestr("scifact/qrels/test.tsv", "query-id\tcorpus-id\tscore\nq1\tofficial-17\t1\n")
    settings = Settings(data_dir=tmp_path, expansion_enabled=False, rerank_enabled=False, offline=True)
    result = run(limit=1, settings=settings)
    assert result["nDCG@10"] == 1.0
    assert result["Recall@10"] == 1.0
    assert result["documents_indexed"] == 2
    assert result["query_ids"] == ["q1"]
    assert (tmp_path / "benchmark/results_1.json").exists()
    assert not (tmp_path / "index.sqlite3").exists()
