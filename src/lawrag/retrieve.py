"""Retrievers over the chunked corpus. Each returns a ranked list of § ids.

- ``BM25Retriever``: lexical, with ``word``, ``stem`` or ``ngram`` analysis. No dependencies.
- ``DenseRetriever``: sentence-transformers bi-encoder (multilingual E5, BGE-M3). Chunk
  embeddings are cached on disk, so a corpus is encoded once per model.
- ``HybridRetriever``: weighted reciprocal rank fusion of any retrievers.
- ``RerankRetriever``: a cross-encoder rescoring the top chunks of a first-stage retriever.

Dense and reranking models need the ``dense`` extra and are run in Colab.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .bm25 import BM25
from .chunk import chunk_corpus
from .metrics import sections_from_chunks

DENSE_MODELS = {
    # name: (Hugging Face id, query prefix, passage prefix)
    "e5": ("intfloat/multilingual-e5-base", "query: ", "passage: "),
    "bge-m3": ("BAAI/bge-m3", "", ""),
}
RERANKER = "BAAI/bge-reranker-v2-m3"


def load_corpus(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class BM25Retriever:
    def __init__(self, chunks: list[dict], analyzer: str = "word", k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.name = "bm25" if analyzer == "word" else f"bm25-{analyzer}"
        self.index = BM25([c["text"] for c in chunks], k1=k1, b=b, analyzer=analyzer)

    def rank_chunks(self, query: str, k: int = 100) -> list[int]:
        return [i for i, _ in self.index.top_k(query, k)]

    def __call__(self, query: str, k: int = 100) -> list[str]:
        return sections_from_chunks(self.chunks[i]["section_id"] for i in self.rank_chunks(query, k))


def _cache_key(model_id: str, texts: list[str]) -> str:
    h = hashlib.sha1(model_id.encode())
    for t in texts:
        h.update(b"\0" + t.encode("utf-8"))
    return h.hexdigest()[:16]


class DenseRetriever:
    """Bi-encoder retrieval with cosine similarity over normalised embeddings."""

    def __init__(self, chunks: list[dict], model: str = "e5", cache_dir: str | Path | None = None,
                 batch_size: int = 32, encoder=None):
        import numpy as np

        self.np = np
        self.chunks = chunks
        self.name = model
        model_id, self.query_prefix, passage_prefix = DENSE_MODELS.get(model, (model, "", ""))
        if encoder is None:
            from sentence_transformers import SentenceTransformer

            encoder = SentenceTransformer(model_id)
        self.encoder = encoder

        texts = [passage_prefix + c["text"] for c in chunks]
        cache = Path(cache_dir) / f"{model.replace('/', '_')}-{_cache_key(model_id, texts)}.npy" if cache_dir else None
        if cache and cache.exists():
            self.embeddings = np.load(cache)
        else:
            self.embeddings = np.asarray(
                encoder.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True)
            )
            if cache:
                cache.parent.mkdir(parents=True, exist_ok=True)
                np.save(cache, self.embeddings)

    def rank_chunks(self, query: str, k: int = 100) -> list[int]:
        q = self.np.asarray(self.encoder.encode([self.query_prefix + query], normalize_embeddings=True))[0]
        scores = self.embeddings @ q
        return [int(i) for i in self.np.argsort(-scores)[:k]]

    def __call__(self, query: str, k: int = 100) -> list[str]:
        return sections_from_chunks(self.chunks[i]["section_id"] for i in self.rank_chunks(query, k))


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60, weights: list[float] | None = None) -> list[str]:
    """Fuse ranked lists: score(d) = sum_i w_i / (k + rank_i(d)). Ties keep first-seen order."""
    weights = weights or [1.0] * len(rankings)
    scores: dict[str, float] = {}
    for ranking, w in zip(rankings, weights):
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + w / (k + rank)
    return sorted(scores, key=lambda item: scores[item], reverse=True)


class HybridRetriever:
    def __init__(self, *retrievers, weights: list[float] | None = None, rrf_k: int = 60):
        self.retrievers = retrievers
        self.weights = list(weights) if weights else [1.0] * len(retrievers)
        self.rrf_k = rrf_k
        parts = [r.name if w == 1 else f"{w:g}·{r.name}" for r, w in zip(retrievers, self.weights)]
        self.name = " + ".join(parts)

    def __call__(self, query: str, k: int = 100) -> list[str]:
        return reciprocal_rank_fusion([r(query, k) for r in self.retrievers], self.rrf_k, self.weights)


class RerankRetriever:
    """Rescore the first stage's top ``top_n`` chunks with a cross-encoder, then rank §§."""

    def __init__(self, first_stage, chunks: list[dict], model=None, top_n: int = 30, max_length: int = 512):
        if model is None:
            from sentence_transformers import CrossEncoder

            model = CrossEncoder(RERANKER, max_length=max_length)
        self.first_stage = first_stage
        self.chunks = chunks
        self.model = model
        self.top_n = top_n
        self.name = f"{first_stage.name} → rerank"

    def rank_chunks(self, query: str, k: int = 100) -> list[int]:
        first = self.first_stage.rank_chunks(query, max(k, self.top_n))
        candidates, rest = first[: self.top_n], first[self.top_n :]
        if not candidates:
            return []
        scores = self.model.predict([(query, self.chunks[i]["text"]) for i in candidates])
        order = sorted(range(len(candidates)), key=lambda j: float(scores[j]), reverse=True)
        # reranked candidates first, then the first stage's remaining order
        return ([candidates[j] for j in order] + rest)[:k]

    def __call__(self, query: str, k: int = 100) -> list[str]:
        return sections_from_chunks(self.chunks[i]["section_id"] for i in self.rank_chunks(query, k))


def build_bm25(corpus_path: str | Path, max_chars: int = 1200, analyzer: str = "word") -> BM25Retriever:
    return BM25Retriever(chunk_corpus(load_corpus(corpus_path), max_chars), analyzer=analyzer)
