"""Evaluate retrievers on the question set and print a results table.

Usage:
    python -m lawrag.evaluate --corpus data/corpus.jsonl --questions data/questions.jsonl
    python -m lawrag.evaluate ... --retrievers bm25 bm25-stem bm25-ngram e5 hybrid

Search interactively (useful while writing questions):
    python -m lawrag.evaluate --corpus data/corpus.jsonl --search "Wie lange darf ich arbeiten?"

For the full comparison (all retrievers, weight sweep, reranking, per-question ranks) use
``python -m lawrag.experiments``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .chunk import chunk_corpus
from .metrics import evaluate
from .retrieve import DENSE_MODELS, BM25Retriever, DenseRetriever, HybridRetriever, load_corpus

LEXICAL = {"bm25": "word", "bm25-stem": "stem", "bm25-ngram": "ngram"}
CHOICES = [*LEXICAL, *DENSE_MODELS, "hybrid"]
COLUMNS = ["recall@1", "recall@3", "recall@5", "recall@10", "mrr@10", "ndcg@10"]


def load_questions(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        qs = [json.loads(line) for line in f if line.strip() and not line.lstrip().startswith("//")]
    for q in qs:
        if not q.get("question") or not q.get("gold"):
            raise ValueError(f"question needs 'question' and non-empty 'gold': {q}")
    return qs


def check_gold(questions: list[dict], records: list[dict]) -> list[str]:
    """Gold § ids that do not exist in the corpus (typos, or a § that was repealed)."""
    ids = {r["id"] for r in records}
    return sorted({g for q in questions for g in q["gold"] if g not in ids})


def format_table(rows: dict[str, dict]) -> str:
    lines = ["| retriever | " + " | ".join(COLUMNS) + " |", "|---" * (len(COLUMNS) + 1) + "|"]
    for name, m in rows.items():
        lines.append(f"| {name} | " + " | ".join(f"{m[c]:.1f}" for c in COLUMNS) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--corpus", type=Path, default=Path("data/corpus.jsonl"))
    ap.add_argument("--questions", type=Path, default=Path("data/questions.jsonl"))
    ap.add_argument("--retrievers", nargs="+", default=["bm25"], choices=CHOICES)
    ap.add_argument("--max-chars", type=int, default=1200)
    ap.add_argument("--cache", type=Path, help="directory for cached chunk embeddings")
    ap.add_argument("--search", help="print the top 10 § for one query and exit")
    ap.add_argument("--out", type=Path, help="write the results as JSON")
    args = ap.parse_args()

    records = load_corpus(args.corpus)
    by_id = {r["id"]: r for r in records}
    chunks = chunk_corpus(records, args.max_chars)

    if args.search:
        for rank, sid in enumerate(BM25Retriever(chunks)(args.search)[:10], start=1):
            print(f"{rank:2d}. {sid:14s} {by_id[sid]['title']}")
        return

    questions = load_questions(args.questions)
    missing = check_gold(questions, records)
    if missing:
        raise SystemExit(f"gold labels not in corpus: {missing}")

    built: dict[str, object] = {}

    def get(name: str):
        if name not in built:
            if name in LEXICAL:
                built[name] = BM25Retriever(chunks, analyzer=LEXICAL[name])
            elif name in DENSE_MODELS:
                built[name] = DenseRetriever(chunks, name, cache_dir=args.cache)
            else:  # "hybrid": the first-run setting, BM25 (words) + E5 with equal weights
                built[name] = HybridRetriever(get("bm25"), get("e5"))
        return built[name]

    results = {name: evaluate(questions, get(name)) for name in args.retrievers}
    print(f"{len(questions)} questions, {len(records)} §, {len(chunks)} chunks\n")
    print(format_table(results))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
