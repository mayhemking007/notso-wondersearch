"""CPU cross-encoder, loaded once and returning raw logits."""
from threading import Lock
from app.config import Settings


class Reranker:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.model = None
        self.error: str | None = None
        self.lock = Lock()

    def rank(self, query: str, candidates: list[dict]) -> list[dict]:
        if not self.settings.rerank_enabled or not candidates:
            return candidates
        with self.lock:
            try:
                import torch
                from sentence_transformers import CrossEncoder
                s = self.settings
                torch.set_num_threads(s.torch_threads)
                if self.model is None:
                    self.model = CrossEncoder(s.rerank_model, device="cpu", max_length=s.max_length,
                                              cache_folder=str(s.model_cache), revision=s.rerank_revision,
                                              local_files_only=s.offline)
                scores = self.model.predict(
                    [(query, item["passage_text"]) for item in candidates],
                    batch_size=s.batch_size, show_progress_bar=False,
                    activation_fn=torch.nn.Identity())
                for item, score in zip(candidates, scores):
                    item["rerank_score"] = float(score)
                self.error = None
                return sorted(candidates, key=lambda x: (-x["rerank_score"], x["passage_id"]))
            except Exception as exc:
                self.error = str(exc)
                raise RuntimeError("Reranker failed. Download models first or set SEARCH_RERANK_ENABLED=false. "
                                   + str(exc)) from exc
