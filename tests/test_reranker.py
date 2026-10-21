import sys
from types import SimpleNamespace
from app.config import Settings
from app.retrieval.reranker import Reranker


def test_raw_scores_batching_and_model_reuse(tmp_path, monkeypatch):
    calls = []
    class FakeCrossEncoder:
        def __init__(self, name, **kwargs):
            calls.append(("load", name, kwargs))

        def predict(self, pairs, **kwargs):
            calls.append(("predict", pairs, kwargs))
            return [-2.5, 4.0]

    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(CrossEncoder=FakeCrossEncoder))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        set_num_threads=lambda count: None, nn=SimpleNamespace(Identity=lambda: "identity")))
    reranker = Reranker(Settings(data_dir=tmp_path, batch_size=2, max_length=256))
    candidates = [{"passage_id": "a", "passage_text": "irrelevant"},
                  {"passage_id": "b", "passage_text": "relevant"}]
    for _ in range(2):
        ranked = reranker.rank("original query", candidates)
        assert ranked[0]["passage_id"] == "b"
        assert ranked[0]["rerank_score"] == 4.0
    assert len([call for call in calls if call[0] == "load"]) == 1
    assert calls[0][2]["device"] == "cpu"
    assert calls[0][2]["max_length"] == 256
    assert calls[1][2]["activation_fn"] == "identity"
    assert calls[1][2]["batch_size"] == 2
