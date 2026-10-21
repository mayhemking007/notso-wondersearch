# Local semantic search without document embeddings

A backend-only experimental search engine. Documents become ordinary SQLite
FTS5 lexical postings. Neural processing happens only at query time: a small
local query encoder expands the query, BM25 retrieves passages, reciprocal rank
fusion merges the lists, and a cross-encoder reranks the best 20 candidates.

This is not a reproduction of WonderSearch or full SPLADE retrieval. There is
no training, document embedding generation, vector database, hosted inference,
frontend, or GPU requirement.

## Install

Use Python 3.11 or 3.12 (the pinned stack is validated on Python 3.12). Your
SQLite build must include FTS5. Create a virtual environment from the project
directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
# Install CPU-only Torch first, especially on Linux where the default wheel may pull CUDA libraries.
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev]"
```

On Linux/macOS activate with `source .venv/bin/activate`. On Apple Silicon use
`python -m pip install torch==2.6.0` from the default package index instead of
the CPU wheel index. All inference explicitly uses CPU.

The direct dependency versions are pinned in `requirements.txt`. They are not
a complete transitive lockfile. A standard Python installation is recommended;
the Windows Store execution alias is not a Python interpreter.

## Run

```powershell
python -m app.cli index --docs ./documents
python -m app.cli download-models
python -m app.cli search "How do I prevent duplicate order creation?"
python -m app.cli search "How does idempotency work?" --top-k 5
python -m app.cli serve
```

Indexing never loads either model. Models are loaded lazily once per process;
the API retains them between requests. Separate CLI invocations each load
their own model instances. The default API binds to `127.0.0.1:8000`; OpenAPI
documentation is at `http://127.0.0.1:8000/docs`.

Initial installation, model downloads, and dataset downloads require internet.
After downloading, set `SEARCH_OFFLINE=true` to use only cached resources:

```powershell
$env:SEARCH_OFFLINE="true"
python -m app.cli search "safe request retries"
```

For a fast model-free lexical check:

```powershell
$env:SEARCH_EXPANSION_ENABLED="false"
$env:SEARCH_RERANK_ENABLED="false"
python -m app.cli index --docs ./documents
python -m app.cli search "idempotency"
```

In bash use `export SEARCH_OFFLINE=true`, etc. Export configuration in your
shell; `.env.example` documents variables but is not automatically loaded.

## HTTP API

`POST /index`:

```json
{"docs_path": "./documents"}
```

Returns `documents_indexed`, `passages_indexed`, `documents_skipped`,
`documents_deleted`, per-file `errors`, and `elapsed_ms`. Counts indexed are
the documents and passages written during this run, not cumulative totals.

`POST /search`:

```json
{"query": "How do I prevent duplicate order creation?", "top_k": 5}
```

Returns the original query, expansions, ranked results, warnings, and latency.
Each result includes document ID, passage ID, source, page number, passage
text, weighted RRF retrieval score, and optional raw reranker score. The response
also includes `expansion_weights`, the normalized weights of the expanded terms.

`GET /health` returns document/passage counts and model load/error status.
Model readiness is reported separately from service availability. An enabled
reranker failure returns HTTP 503 rather than silently changing the pipeline.
An expansion failure logs a warning and falls back to the original query.
Restart after fixing a model loading error to retry expansion loading.

## Configuration

| Environment variable | Default |
| --- | --- |
| `SEARCH_DATA_DIR` | `./data` |
| `SEARCH_EXPANSION_ENABLED` | `true` |
| `SEARCH_RERANK_ENABLED` | `true` |
| `SEARCH_OFFLINE` | `false` |
| `SEARCH_EXPANSION_MODEL` | `naver/splade-cocondenser-ensembledistil` |
| `SEARCH_EXPANSION_REVISION` | `main` |
| `SEARCH_RERANK_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| `SEARCH_RERANK_REVISION` | `main` |
| `SEARCH_EXPANSION_LIMIT` | `3` (maximum 8) |
| `SEARCH_CANDIDATES` | `20` (maximum 200) |
| `SEARCH_MAX_LENGTH` | `512` tokens (maximum 512) |
| `SEARCH_BATCH_SIZE` | `8` |
| `SEARCH_TORCH_THREADS` | `4` |

Pin model revision variables to Hugging Face commit hashes for repeatable
experiments. The benchmark records configured and resolved model revisions.
Changing a model revision or expansion limit changes the expansion cache key.

## Design and limitations

- **Extraction:** UTF-8/UTF-8-BOM text and Markdown; PyMuPDF page text for PDF.
  Scanned/image-only PDFs need OCR, which is not included. Such PDFs may yield
  no passages. Per-page windows preserve accurate page numbers, so short PDF
  pages remain short passages.
- **Chunking:** approximately 200 words, overlapping by 30. Paragraph breaks
  survive within windows. Last passages may be shorter.
- **Incremental indexing:** stable path-based document IDs and positional
  passage IDs; SHA256 detects content/configuration changes. Each replacement
  is transactional, and FTS triggers keep postings consistent. Unreadable
  changed files retain their previous searchable content and report an error.
  Deleted files are removed only for the indexed root after a successful
  traversal. Renaming a file gives it a new document ID. Avoid overlapping
  indexed roots: each root owns its own documents. Symlinks are skipped.
- **Query expansion:** the retrieval-trained SPLADE model processes only the
  original query. Max-pooled `log(1 + ReLU(logits))` yields vocabulary weights.
  Select up to three positively weighted whole-word terms after removing
  original terms, stopwords, punctuation, and WordPiece fragments. This avoids
  instruction leakage from the former FLAN-T5-small rewriter. Term weights are
  normalized to sum to one and used in fusion; they are not probabilities.
  No document is passed through this encoder. Ordinary lexical postings remain
  unchanged, so this is not full SPLADE retrieval. English vocabulary filtering
  and discarding subwords can miss technical terms, and expansions may still
  drift. Quality must be measured for your corpus. The model's license is
  CC-BY-NC-SA-4.0; see its model card for terms.
- **Retrieval:** safely tokenized OR searches, one per original/expanded
  query. Raw FTS5 BM25 sorts ascending. Weighted RRF uses
  `sum(weight/(60 + rank))`, with original-query weight 1 and expansion weights
  totaling 1, then
  retains 20 passages by default. The API's `retrieval_score` is this RRF score,
  not BM25 or a probability. Technical punctuation is discarded safely, so
  terms such as C++ lose some lexical specificity.
- **Reranking:** the configured MiniLM cross-encoder sees the original query
  paired with each passage. Batched CPU prediction returns raw logits, which
  are meaningful for ranking rather than as calibrated probabilities. Long
  inputs are truncated. With reranking disabled, `rerank_score` is null and
  RRF determines the order. A reranker cannot recover missing candidates.
- **Concurrency:** single API process with serialized model inference and
  indexing requests. SQLite uses WAL and short-lived connections. A search
  during a long indexing run may see a partially updated corpus. This is a
  small local tool, not a distributed search service.

## SciFact benchmark

```powershell
# First run downloads the official BEIR archive and caches it.
python -m app.cli benchmark --dataset scifact --limit 5
python -m app.cli benchmark --dataset scifact --limit 30
# Explicit opt-in only:
python -m app.cli benchmark --dataset scifact --limit 300
```

The benchmark calls the same search service. Every corpus document is one
retrievable unit retaining its official ID; the user's passage index is
untouched. It uses official test qrels and a seeded, sorted-then-shuffled query
sample, with nested samples as the limit increases. It evaluates nDCG@10 and
Recall@10 through `ir-measures`, with at most 20 candidates and ten final results.

Results are printed as JSON and written to `data/benchmark/results_N.json`.
Sample IDs are also saved separately. Metadata includes archive checksum,
model names/revisions, package versions, retrieval configuration, hardware,
warm-up duration, mean latency, and total runtime including preparation.
Expansion cache is cleared for each benchmark; model loading and a separate
warm-up query precede measured queries. No metric values are fabricated.
A 30-query run is preliminary and cannot be directly compared to published
full-test benchmark scores.

Dataset: [official BEIR SciFact archive](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip).
Model documentation: [SPLADE query encoder](https://huggingface.co/naver/splade-cocondenser-ensembledistil),
[Sentence Transformers cross-encoders](https://www.sbert.net/docs/cross_encoder/pretrained_models.html).
SQLite ranking: [FTS5](https://www.sqlite.org/fts5.html).

## Tests

```powershell
python -m pytest -q
```

Routine tests use model stubs and require no downloads. They cover persistence,
changed/deleted files, failure retention, safe queries, PDF page references,
chunk overlap, expansion caching/fallback, RRF, ranking, API parity, and
benchmark sampling. Real model and benchmark checks use the CLI above.

## Verified local run

Validated on Windows 11, Python 3.12.14, CPU-only Torch 2.6.0:

- 24 tests passed, including API/CLI, PDF extraction, legacy duplicate-ID
  migration, expansion-cache invalidation, and benchmark evaluation.
- Both models downloaded successfully and loaded with `SEARCH_OFFLINE=true`.
- A retry-safety query ranked the example idempotency passage first.
- The replacement SPLADE query encoder passed the five-query offline SciFact
  smoke test: nDCG@10 **0.5333333333**, Recall@10 **0.80**, mean query latency
  **831.82 ms**, over all **5,183** corpus documents. Results and exact revisions
  are saved in `data/benchmark/results_5.json`. Rerun after changing models;
  scores from the former FLAN-T5 pipeline do not describe the new encoder.

These five-query numbers are smoke-test results, not a reliable quality estimate
or a comparison with published benchmarks. No 300-query run was performed.
The API tests need local loopback sockets; restricted execution sandboxes may
block Windows asyncio's socket pair even though TestClient opens no HTTP server.

### Updating an existing installation

The index automatically migrates legacy path-based IDs and merges duplicate
records on startup. It preserves the canonical document's passages and leaves
official benchmark IDs unchanged; rebuilding your index is unnecessary.
Old generated expansions remain unused because the new model and cache format
have different cache keys.

Stop the running server with Ctrl+C and download the replacement model once,
then restart. In Git Bash on Windows:

```bash
./.venv/Scripts/python.exe -m app.cli download-models
./.venv/Scripts/python.exe -m app.cli search "How do I prevent duplicate orders?"
./.venv/Scripts/python.exe -m app.cli serve
```

Remove any old `SEARCH_EXPANSION_MODEL=google/flan-t5-small` shell override.
The expansion model setting now expects a SPLADE-compatible masked-language
model. If `SEARCH_OFFLINE=true`, temporarily unset it for the initial download.

## Project layout

```text
app/
  config.py                 environment settings
  cli.py / main.py          CLI and HTTP adapters
  ingestion/                extraction and deterministic chunking
  indexing/                 SQLite storage and incremental indexing
  retrieval/                expansion, BM25, fusion, reranking, shared service
  benchmark/scifact.py      dataset caching and evaluation
tests/                      offline tests
documents/                  example corpus
data/                       generated local artifacts (ignored)
```
