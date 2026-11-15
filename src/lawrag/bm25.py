"""Okapi BM25 over German legal text, implemented from scratch as the lexical baseline.

Terms come from ``lawrag.text.analyze``: plain words, Snowball German stems, or character
4-grams of the stems (which lets "Pause" match the compound "Ruhepause").
"""

from __future__ import annotations

import math
from collections import Counter

from .text import GERMAN_STOPWORDS, analyze


def tokenize(text: str, stopwords: frozenset[str] = GERMAN_STOPWORDS) -> list[str]:
    """Casefolded words without stopwords (the ``word`` analyzer)."""
    return analyze(text, "word", stopwords)


class BM25:
    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75, analyzer: str = "word"):
        self.k1, self.b, self.analyzer = k1, b, analyzer
        self.docs = [Counter(analyze(d, analyzer)) for d in documents]
        self.lengths = [sum(c.values()) for c in self.docs]
        self.avg_len = sum(self.lengths) / len(self.lengths) if self.lengths else 0.0
        n = len(self.docs)
        df = Counter(term for doc in self.docs for term in doc)
        self.idf = {t: math.log((n - f + 0.5) / (f + 0.5) + 1.0) for t, f in df.items()}

    def scores(self, query: str) -> list[float]:
        terms = analyze(query, self.analyzer)
        out = []
        for doc, length in zip(self.docs, self.lengths):
            norm = self.k1 * (1 - self.b + self.b * length / self.avg_len) if self.avg_len else self.k1
            s = 0.0
            for t in terms:
                tf = doc.get(t, 0)
                if tf:
                    s += self.idf[t] * tf * (self.k1 + 1) / (tf + norm)
            out.append(s)
        return out

    def top_k(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        scored = sorted(enumerate(self.scores(query)), key=lambda x: x[1], reverse=True)
        return [(i, s) for i, s in scored[:k] if s > 0]
