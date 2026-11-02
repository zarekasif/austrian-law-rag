"""Okapi BM25 over German legal text, implemented from scratch as the lexical baseline.

Tokenisation: Unicode casefolding (so "ß" matches "ss"), split on non-word characters,
drop a small list of German function words. No stemming, so "Mietvertrag" and "Vertrag"
do not match; how much this costs on German compounds is one of the things the dense
retrievers are compared against.
"""

from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN = re.compile(r"\w+", re.UNICODE)

GERMAN_STOPWORDS = frozenset(
    """
    aber als also am an auch auf aus bei bin bis bist da damit dann das dass dem den denn der des die dies diese
    dieser dieses doch dort du durch ein eine einem einen einer eines er es etwa für gegen hat hatte hier ich ihr
    im in ist ja jedoch kann kein keine man mit muss nach nicht noch nur ob oder ohne sein sich sie sind so soll
    über um und uns unter vom von vor wann war was weil welche welcher wenn wer werden wie wir wird wo zu zum zur
    """.split()
)


def tokenize(text: str, stopwords: frozenset[str] = GERMAN_STOPWORDS) -> list[str]:
    return [t for t in _TOKEN.findall(text.casefold()) if t not in stopwords]


class BM25:
    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(d)) for d in documents]
        self.lengths = [sum(c.values()) for c in self.docs]
        self.avg_len = sum(self.lengths) / len(self.lengths) if self.lengths else 0.0
        n = len(self.docs)
        df = Counter(term for doc in self.docs for term in doc)
        self.idf = {t: math.log((n - f + 0.5) / (f + 0.5) + 1.0) for t, f in df.items()}

    def scores(self, query: str) -> list[float]:
        terms = tokenize(query)
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
