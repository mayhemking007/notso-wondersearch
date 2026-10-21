"""Reciprocal Rank Fusion uses ranks rather than incomparable BM25 magnitudes."""


def fuse(rankings: list[list[dict]], limit: int = 20, k: int = 60,
         weights: list[float] | None = None) -> list[dict]:
    weights = weights if weights is not None else [1.0] * len(rankings)
    if len(weights) != len(rankings) or any(w < 0 for w in weights):
        raise ValueError("Require one non-negative weight per ranking")
    merged: dict[str, dict] = {}
    for ranking, weight in zip(rankings, weights):
        seen = set()
        for rank, item in enumerate(ranking, 1):
            key = item["passage_id"]
            if key in seen:
                continue
            seen.add(key)
            if key not in merged:
                merged[key] = {**item, "retrieval_score": 0.0, "rerank_score": None}
            merged[key]["retrieval_score"] += weight / (k + rank)
    return sorted(merged.values(), key=lambda x: (-x["retrieval_score"], x["passage_id"]))[:limit]
