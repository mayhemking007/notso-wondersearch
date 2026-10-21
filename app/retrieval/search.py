"""Single shared service for API, CLI, and benchmark execution."""
from threading import Lock
from time import perf_counter
from app.config import Settings
from app.indexing.sqlite_index import SQLiteIndex
from app.retrieval.bm25 import retrieve
from app.retrieval.fusion import fuse
from app.retrieval.query_expander import QueryExpander
from app.retrieval.reranker import Reranker


class SearchService:
    def __init__(self, settings: Settings | None = None, index: SQLiteIndex | None = None):
        self.settings = settings or Settings()
        self.index = index or SQLiteIndex(self.settings.data_dir / "index.sqlite3")
        self.expander = QueryExpander(self.settings, self.index)
        self.reranker = Reranker(self.settings)
        self.index_lock = Lock()

    def index_directory(self, path: str) -> dict:
        with self.index_lock:
            return self.index.index_directory(path, self.settings.chunk_words, self.settings.overlap_words)

    def search(self, query: str, top_k: int = 5) -> dict:
        query = query.strip()
        if not query or len(query) > 2000:
            raise ValueError("Query must contain 1–2000 characters")
        if not 1 <= top_k <= self.settings.candidates:
            raise ValueError(f"top_k must be between 1 and {self.settings.candidates}")
        started = perf_counter()
        weighted_expansions = self.expander.expand_weighted(query)
        expansions = [term for term, _ in weighted_expansions]
        rankings = [retrieve(self.index, text, self.settings.candidates) for text in [query, *expansions]]
        candidates = fuse(rankings, self.settings.candidates, self.settings.rrf_k,
                          [1.0, *(weight for _, weight in weighted_expansions)])
        ranked = self.reranker.rank(query, candidates)[:top_k]
        results = [{"document_id": x["document_id"], "passage_id": x["passage_id"],
                    "source": x["source_path"], "page_number": x["page_number"],
                    "text": x["passage_text"], "retrieval_score": x["retrieval_score"],
                    "rerank_score": x["rerank_score"]} for x in ranked]
        warnings = ["Expansion unavailable; used original query"] if self.expander.error else []
        return {"query": query, "expanded_queries": expansions, "results": results,
                "expansion_weights": dict(weighted_expansions),
                "latency_ms": round((perf_counter() - started) * 1000, 2), "warnings": warnings}

    def health(self) -> dict:
        return {"status": "ok", "index": self.index.counts(), "device": "cpu",
                "expansion": {"enabled": self.settings.expansion_enabled,
                              "loaded": self.expander.model is not None, "error": self.expander.error},
                "reranker": {"enabled": self.settings.rerank_enabled,
                             "loaded": self.reranker.model is not None, "error": self.reranker.error}}
