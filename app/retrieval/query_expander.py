"""Query-only SPLADE vocabulary expansion over ordinary lexical postings."""
import json
import logging
import math
import re
from threading import Lock
from app.config import Settings
from app.indexing.sqlite_index import SQLiteIndex, stable_id

log = logging.getLogger(__name__)
STOPWORDS = set("a an and are as at be been but by can could did do does for from had has have how i if in into is it its may me my not of on or our should so than that the their them then there these they this those to was we were what when where which who why will with would you your".split())
CACHE_VERSION = "splade-query-v1"


def select_terms(tokens: list[str], scores: list[float], query: str,
                 limit: int) -> list[tuple[str, float]]:
    """Discard WordPieces, punctuation, stopwords and already-present terms."""
    original = set(re.findall(r"[^\W_]+", query.casefold()))
    selected = {}
    for token, score in zip(tokens, scores):
        token = token.casefold()
        if (not math.isfinite(score) or score <= 0 or token in STOPWORDS or token in original
                or not re.fullmatch(r"[a-z][a-z0-9]{2,}", token)):
            continue
        selected[token] = max(selected.get(token, 0.0), float(score))
    ranked = sorted(selected.items(), key=lambda pair: (-pair[1], pair[0]))
    return ranked[:limit]


class QueryExpander:
    def __init__(self, settings: Settings, index: SQLiteIndex):
        self.settings, self.index = settings, index
        self.model = self.tokenizer = None
        self.error: str | None = None
        self.lock = Lock()

    def _load(self) -> None:
        if self.model is not None:
            return
        if self.error:
            raise RuntimeError(self.error)
        import torch
        from transformers import AutoTokenizer, AutoModelForMaskedLM
        s = self.settings
        torch.set_num_threads(s.torch_threads)
        kwargs = dict(cache_dir=str(s.model_cache), revision=s.expansion_revision,
                      local_files_only=s.offline)
        self.tokenizer = AutoTokenizer.from_pretrained(s.expansion_model, **kwargs)
        self.model = AutoModelForMaskedLM.from_pretrained(s.expansion_model, **kwargs).to("cpu").eval()

    def _generate(self, query: str) -> list[tuple[str, float]]:
        import torch
        self._load()
        inputs = self.tokenizer(query, return_tensors="pt", truncation=True, max_length=256)
        with torch.inference_mode():
            logits = self.model(**inputs).logits
            weights = (torch.log1p(torch.relu(logits)) * inputs["attention_mask"].unsqueeze(-1)).amax(dim=1)[0]
        tokens = self.tokenizer.convert_ids_to_tokens(list(range(weights.shape[0])))
        return select_terms(tokens, weights.tolist(), query, self.settings.expansion_limit)

    def expand(self, query: str) -> list[str]:
        return [term for term, _ in self.expand_weighted(query)]

    def expand_weighted(self, query: str) -> list[tuple[str, float]]:
        s = self.settings
        if not s.expansion_enabled or not s.expansion_limit:
            return []
        key = stable_id(json.dumps([query, s.expansion_model, s.expansion_revision,
                                   s.expansion_limit, CACHE_VERSION]))
        with self.lock:
            cached = self.index.cached_expansions(key)
            if cached is not None:
                return [(str(term), float(weight)) for term, weight in cached]
            try:
                output = self._generate(query)
                selected = select_terms([term for term, _ in output],
                                        [weight for _, weight in output], query, s.expansion_limit)
                total = sum(weight for _, weight in selected)
                expansions = [(term, weight / total) for term, weight in selected] if total else []
                self.index.cache_expansions(key, expansions)
                self.error = None
                return expansions
            except Exception as exc:
                self.error = str(exc)
                log.warning("Query expansion unavailable; using original query: %s", exc)
                return []
