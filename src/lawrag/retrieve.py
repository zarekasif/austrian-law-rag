"""Retrievers over the chunked corpus. Each returns a ranked list of § ids.

- ``BM25Retriever``: lexical baseline, no dependencies.
- ``DenseRetriever``: sentence-transformers bi-encoder (e.g. multilingual E5); needs the
  ``dense`` extra and is run in Colab.
- ``HybridRetriever``: reciprocal rank fusion of any retrievers.
"""

from __future__ import annotations

import json
from pathlib import Path

from .bm25 import BM25
from .chunk import chunk_corpus
from .metrics import sections_from_chunks


def load_corpus(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class BM25Retriever:
    name = "bm25"

    def __init__(self, chunks: list[dict], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.index = BM25([c["text"] for c in chunks], k1=k1, b=b)

    def rank_chunks(self, query: str, k: int = 100) -> list[int]:
        return [i for i, _ in self.index.top_k(query, k)]

    def __call__(self, query: str, k: int = 100) -> list[str]:
        return sections_from_chunks(self.chunks[i]["section_id"] for i in self.rank_chunks(query, k))


class DenseRetriever:
    """Bi-encoder retrieval with cosine similarity. Prefixes follow the model's convention
    (E5 models expect ``query: `` / ``passage: ``)."""

    def __init__(self, chunks: list[dict], model_name: str = "intfloat/multilingual-e5-base",
                 query_prefix: str = "query: ", passage_prefix: str = "passage: ", batch_size: int = 32):
        import numpy as np
        from sentence_transformers import SentenceTransformer

        self.np = np
        self.chunks = chunks
        self.name = model_name.split("/")[-1]
        self.model = SentenceTransformer(model_name)
        self.query_prefix = query_prefix
        self.embeddings = self.model.encode(
            [passage_prefix + c["text"] for c in chunks],
            batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True,
        )

    def rank_chunks(self, query: str, k: int = 100) -> list[int]:
        q = self.model.encode([self.query_prefix + query], normalize_embeddings=True)[0]
        scores = self.embeddings @ q
        top = self.np.argsort(-scores)[:k]
        return [int(i) for i in top]

    def __call__(self, query: str, k: int = 100) -> list[str]:
        return sections_from_chunks(self.chunks[i]["section_id"] for i in self.rank_chunks(query, k))


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> list[str]:
    """Fuse ranked lists: score(d) = sum over lists of 1 / (k + rank(d)). Ties keep first-seen order."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda item: scores[item], reverse=True)


class HybridRetriever:
    def __init__(self, *retrievers, rrf_k: int = 60):
        self.retrievers = retrievers
        self.rrf_k = rrf_k
        self.name = "+".join(r.name for r in retrievers)

    def __call__(self, query: str, k: int = 100) -> list[str]:
        return reciprocal_rank_fusion([r(query, k) for r in self.retrievers], self.rrf_k)


def build_bm25(corpus_path: str | Path, max_chars: int = 1200) -> BM25Retriever:
    return BM25Retriever(chunk_corpus(load_corpus(corpus_path), max_chars))
