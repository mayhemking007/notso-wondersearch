"""Official BEIR SciFact evaluation using the production search service."""
from dataclasses import asdict, replace
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from urllib.request import urlopen
import csv
import importlib.metadata
import json
import os
import platform
import random
import sqlite3
import ssl
import zipfile

from app.config import Settings
from app.indexing.sqlite_index import SQLiteIndex
from app.retrieval.search import SearchService

DATASET_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"


def prepare_dataset(base: Path, offline: bool) -> tuple[Path, str]:
    archive = base / "scifact.zip"
    base.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        if offline:
            raise RuntimeError("SciFact is not cached; download it with SEARCH_OFFLINE=false first")
        temporary = archive.with_suffix(".partial")
        try:
            import certifi
            context = ssl.create_default_context()
            context.load_verify_locations(cafile=certifi.where())
            with urlopen(DATASET_URL, timeout=120, context=context) as response, temporary.open("wb") as output:
                while block := response.read(1024 * 1024):
                    output.write(block)
            # Validate before promoting a partial download into the cache.
            with zipfile.ZipFile(temporary) as zipped:
                if zipped.testzip() is not None:
                    raise ValueError("Corrupt dataset archive")
            temporary.replace(archive)
        finally:
            temporary.unlink(missing_ok=True)
    # Read only the three expected files; never extract arbitrary archive paths.
    folder = base / "scifact"
    with zipfile.ZipFile(archive) as zipped:
        for relative in ("corpus.jsonl", "queries.jsonl", "qrels/test.tsv"):
            target = folder / relative
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(zipped.read("scifact/" + relative))
    return folder, sha256(archive.read_bytes()).hexdigest()


def sample_ids(ids: list[str], limit: int) -> list[str]:
    if not 1 <= limit <= len(ids):
        raise ValueError(f"limit must be between 1 and {len(ids)}")
    ordered = sorted(ids)
    random.Random(42).shuffle(ordered)
    return ordered[:limit]


def run(limit: int = 30, settings: Settings | None = None) -> dict:
    from ir_measures import Qrel, ScoredDoc, nDCG, Recall, calc_aggregate
    total_start = perf_counter()
    original = settings or Settings()
    if original.candidates < 10:
        raise ValueError("Benchmark requires at least 10 candidates")
    dataset, checksum = prepare_dataset(original.data_dir / "datasets", original.offline)
    qrels = []
    with (dataset / "qrels/test.tsv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            qrels.append(Qrel(str(row["query-id"]), str(row["corpus-id"]), int(row["score"])))
    ids = sample_ids(sorted({q.query_id for q in qrels}), limit)
    queries = {str(q["_id"]): q["text"] for q in
               map(json.loads, (dataset / "queries.jsonl").read_text(encoding="utf-8").splitlines())}
    benchmark_dir = original.data_dir / "benchmark"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    (benchmark_dir / f"query_ids_{limit}.json").write_text(json.dumps(ids, indent=2), encoding="utf-8")
    index = SQLiteIndex(benchmark_dir / "scifact.sqlite3")
    with index.connect() as db:
        existing = {r["document_id"]: r["fingerprint"] for r in db.execute("SELECT * FROM documents")}
    corpus_ids = set()
    changed_documents = []
    with (dataset / "corpus.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            doc = json.loads(line)
            doc_id = str(doc["_id"])
            corpus_ids.add(doc_id)
            text = (doc.get("title", "") + "\n\n" + doc["text"]).strip()
            digest = sha256(text.encode()).hexdigest()
            if existing.get(doc_id) != digest:
                changed_documents.append((doc_id, "scifact", f"scifact:{doc_id}", digest, [(None, text)]))
    index.replace_documents(changed_documents)
    with index.connect() as db:
        db.executemany("DELETE FROM documents WHERE document_id=?", [(d,) for d in set(existing) - corpus_ids])
    # Separate cache keeps prior query expansion runs from distorting benchmark latency.
    with index.connect() as db:
        db.execute("DELETE FROM expansion_cache")
    service = SearchService(replace(original, candidates=min(original.candidates, 20)), index)
    warmup_start = perf_counter()
    service.search("scientific evidence", top_k=10)
    warmup_seconds = perf_counter() - warmup_start
    scored, latencies, warnings = [], [], []
    for query_id in ids:
        response = service.search(queries[query_id], top_k=10)
        latencies.append(response["latency_ms"])
        warnings.extend(response["warnings"])
        for rank, item in enumerate(response["results"], 1):
            # Preserve final pipeline order, including deterministic score ties.
            scored.append(ScoredDoc(query_id, item["document_id"], float(11 - rank)))
    metrics = calc_aggregate([nDCG@10, Recall@10], [q for q in qrels if q.query_id in ids], scored)
    config = asdict(service.settings)
    config["data_dir"] = str(config["data_dir"])
    revisions = {}
    for name, model in (("expansion", service.expander.model), ("reranker", service.reranker.model)):
        config_object = getattr(model, "config", None)
        if config_object is None:
            config_object = getattr(getattr(model, "model", None), "config", None)
        revisions[name] = getattr(config_object, "_commit_hash", None)
    result = {
        "dataset": "SciFact", "dataset_url": DATASET_URL, "dataset_sha256": checksum,
        "dataset_version": "BEIR SciFact archive identified by SHA256",
        "queries_evaluated": len(ids), "documents_indexed": index.counts()["documents"],
        "nDCG@10": metrics[nDCG@10], "Recall@10": metrics[Recall@10],
        "mean_latency_ms": sum(latencies) / len(latencies),
        "warmup_seconds": warmup_seconds, "total_runtime_seconds": perf_counter() - total_start,
        "query_ids": ids, "seed": 42, "settings": config, "resolved_model_revisions": revisions,
        "hardware": {"platform": platform.platform(), "processor": platform.processor(),
                     "python": platform.python_version(), "sqlite": sqlite3.sqlite_version,
                     "logical_cpu_count": os.cpu_count(), "device": "cpu"},
        "package_versions": {p: importlib.metadata.version(p) for p in
                             ("torch", "transformers", "sentence-transformers", "ir-measures")},
        "warnings": sorted(set(warnings)),
        "note": "Preliminary subset evaluation; not directly comparable to published full-test scores."
        if limit < len({q.query_id for q in qrels}) else "Full official test set evaluation.",
    }
    (benchmark_dir / f"results_{limit}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
