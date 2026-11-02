"""Retrieval metrics at § level.

Retrievers rank chunks; a § counts as retrieved at the rank of its best chunk. Gold labels
are § ids such as ``MRG § 1``; a question may have more than one relevant §.
"""

from __future__ import annotations

import math
from statistics import mean
from typing import Callable, Iterable


def sections_from_chunks(chunk_section_ids: Iterable[str]) -> list[str]:
    """Collapse a ranked chunk list to a ranked § list, keeping each § at its first rank."""
    seen: set[str] = set()
    ranked = []
    for sid in chunk_section_ids:
        if sid not in seen:
            seen.add(sid)
            ranked.append(sid)
    return ranked


def recall_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    return len(set(ranked[:k]) & gold) / len(gold) if gold else 0.0


def mrr_at_k(ranked: list[str], gold: set[str], k: int = 10) -> float:
    for i, sid in enumerate(ranked[:k], start=1):
        if sid in gold:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: list[str], gold: set[str], k: int = 10) -> float:
    dcg = sum(1.0 / math.log2(i + 1) for i, sid in enumerate(ranked[:k], start=1) if sid in gold)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(gold), k) + 1))
    return dcg / ideal if ideal else 0.0


def evaluate(
    questions: list[dict],
    retrieve: Callable[[str], list[str]],
    ks: tuple[int, ...] = (1, 3, 5, 10),
) -> dict:
    """Score ``retrieve(question) -> ranked § ids`` on questions ``{"question", "gold": [...]}``."""
    per_q = []
    for q in questions:
        ranked = retrieve(q["question"])
        gold = set(q["gold"])
        row = {f"recall@{k}": recall_at_k(ranked, gold, k) for k in ks}
        row["mrr@10"] = mrr_at_k(ranked, gold, 10)
        row["ndcg@10"] = ndcg_at_k(ranked, gold, 10)
        per_q.append(row)
    summary = {name: 100 * mean(r[name] for r in per_q) for name in per_q[0]} if per_q else {}
    summary["questions"] = len(per_q)
    return summary
