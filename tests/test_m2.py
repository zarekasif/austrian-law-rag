import csv
import json

import numpy as np
import pytest

from lawrag.chunk import chunk_corpus
from lawrag.experiments import build_retrievers, run
from lawrag.metrics import first_gold_rank
from lawrag.retrieve import (
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    RerankRetriever,
    reciprocal_rank_fusion,
)

RECORDS = [
    {"id": "AZG § 11", "law": "AZG", "law_title": "Arbeitszeitgesetz", "section": "§ 11", "title": "Ruhepausen",
     "path": [], "text": "§ 11. (1) Beträgt die Tagesarbeitszeit mehr als sechs Stunden, ist eine Ruhepause zu gewähren."},
    {"id": "MRG § 33", "law": "MRG", "law_title": "Mietrechtsgesetz", "section": "§ 33", "title": "Kündigung",
     "path": [], "text": "§ 33. (1) Der Mieter kann den Mietvertrag schriftlich kündigen."},
    {"id": "UrlG § 2", "law": "UrlG", "law_title": "Urlaubsgesetz", "section": "§ 2", "title": "Urlaubsausmaß",
     "path": [], "text": "§ 2. (1) Der Arbeitnehmer hat Anspruch auf einen bezahlten Urlaub von 30 Werktagen."},
]
CHUNKS = chunk_corpus(RECORDS)
QUESTIONS = [
    {"id": "q1", "question": "Muss mein Chef mir eine Pause geben?", "gold": ["AZG § 11"]},
    {"id": "q2", "question": "Wie viele Urlaubstage habe ich?", "gold": ["UrlG § 2"]},
]


class FakeEncoder:
    """Bag-of-letters embeddings: deterministic, no model download."""

    def __init__(self):
        self.calls = 0

    def encode(self, texts, normalize_embeddings=True, **_):
        self.calls += 1
        out = np.zeros((len(texts), 26))
        for i, t in enumerate(texts):
            for ch in t.lower():
                if "a" <= ch <= "z":
                    out[i, ord(ch) - 97] += 1
        return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)


class FakeCrossEncoder:
    """Scores a pair by whether the chunk contains a given keyword."""

    def __init__(self, keyword):
        self.keyword = keyword

    def predict(self, pairs):
        return [1.0 if self.keyword in text else 0.0 for _, text in pairs]


class Fixed:
    def __init__(self, name, ranking, chunk_ranking=None):
        self.name, self.ranking, self.chunk_ranking = name, ranking, chunk_ranking or []

    def __call__(self, query, k=100):
        return self.ranking[:k]

    def rank_chunks(self, query, k=100):
        return self.chunk_ranking[:k]


def test_weighted_rrf_lets_the_stronger_list_win():
    weak, strong = ["X", "G"], ["G", "X"]
    assert reciprocal_rank_fusion([weak, strong])[0] == "X"  # equal weights: tie, first seen wins
    assert reciprocal_rank_fusion([weak, strong], weights=[0.25, 1.0])[0] == "G"


def test_hybrid_name_shows_weights():
    h = HybridRetriever(Fixed("bm25-ngram", []), Fixed("e5", []), weights=[0.5, 1.0])
    assert h.name == "0.5·bm25-ngram + e5"
    assert HybridRetriever(Fixed("bm25", []), Fixed("e5", [])).name == "bm25 + e5"


def test_bm25_retriever_names():
    assert [BM25Retriever(CHUNKS, analyzer=a).name for a in ("word", "stem", "ngram")] == ["bm25", "bm25-stem", "bm25-ngram"]


def test_reranker_reorders_candidates_and_keeps_the_rest():
    first = Fixed("e5", [], chunk_ranking=[0, 1, 2])  # chunk 0 = AZG, 1 = MRG, 2 = UrlG
    reranked = RerankRetriever(first, CHUNKS, model=FakeCrossEncoder("Urlaub"), top_n=2)
    # only the top 2 are rescored (neither contains "Urlaub", order kept); chunk 2 follows
    assert reranked.rank_chunks("q") == [0, 1, 2]
    reranked = RerankRetriever(first, CHUNKS, model=FakeCrossEncoder("Mieter"), top_n=3)
    assert reranked("q")[0] == "MRG § 33"
    assert reranked.name == "e5 → rerank"


def test_dense_retriever_caches_embeddings(tmp_path):
    enc = FakeEncoder()
    DenseRetriever(CHUNKS, "e5", cache_dir=tmp_path, encoder=enc)
    assert enc.calls == 1 and len(list(tmp_path.glob("*.npy"))) == 1
    enc2 = FakeEncoder()
    d = DenseRetriever(CHUNKS, "e5", cache_dir=tmp_path, encoder=enc2)
    assert enc2.calls == 0  # loaded from cache
    assert len(d("Urlaub Werktage")) == 3 and enc2.calls == 1  # only the query was encoded


def test_first_gold_rank():
    assert first_gold_rank(["A", "B", "C"], {"C", "B"}) == 2
    assert first_gold_rank(["A"], {"Z"}) is None


def test_experiment_suite_lexical_only():
    retrievers = build_retrievers(CHUNKS, lexical_only=True)
    summary, per_q = run(QUESTIONS, retrievers)
    assert list(summary) == ["bm25", "bm25-stem", "bm25-ngram"]
    # the compound question is only solved by the n-gram analyzer
    assert per_q[0]["bm25"] == "" and per_q[0]["bm25-ngram"] == 1
    assert summary["bm25-ngram"]["recall@1"] >= summary["bm25"]["recall@1"]
    assert all("seconds" in m for m in summary.values())


def test_experiment_suite_with_fake_models(monkeypatch, tmp_path):
    import lawrag.retrieve as retrieve

    monkeypatch.setattr(retrieve.DenseRetriever, "__init__", _fake_dense_init)
    import lawrag.experiments as experiments

    retrievers = experiments.build_retrievers(CHUNKS, dense_models=("e5",), rerank=False)
    names = [r.name for r in retrievers]
    assert names == ["bm25", "bm25-stem", "bm25-ngram", "e5", "bm25 + e5",
                     "0.25·bm25-ngram + e5", "0.5·bm25-ngram + e5", "bm25-ngram + e5"]
    summary, _ = run(QUESTIONS, retrievers)
    assert set(summary) == set(names)


def _fake_dense_init(self, chunks, model="e5", cache_dir=None, batch_size=32, encoder=None):
    self.np, self.chunks, self.name, self.query_prefix = np, chunks, model, ""
    self.encoder = FakeEncoder()
    self.embeddings = self.encoder.encode([c["text"] for c in chunks])


def test_cli_writes_results(tmp_path, monkeypatch):
    import sys

    from lawrag import experiments

    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in RECORDS), encoding="utf-8")
    qfile = tmp_path / "q.jsonl"
    qfile.write_text("\n".join(json.dumps(q, ensure_ascii=False) for q in QUESTIONS), encoding="utf-8")
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["x", "--corpus", str(corpus), "--questions", str(qfile),
                                      "--out", str(out), "--lexical-only"])
    experiments.main()
    assert "| bm25-ngram |" in (out / "results.md").read_text(encoding="utf-8")
    rows = list(csv.DictReader((out / "per_question.csv").open(encoding="utf-8")))
    assert [r["id"] for r in rows] == ["q1", "q2"] and "bm25-stem" in rows[0]
