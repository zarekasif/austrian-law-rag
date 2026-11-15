"""Milestone 2: compare all retrievers on the question set in one run.

Runs lexical (BM25 on words, stems, stem 4-grams), dense (E5, BGE-M3), weighted hybrids and
cross-encoder reranking; writes ``results.md`` (table), ``results.json`` and ``per_question.csv``
(rank of the first correct § for every question and retriever, for error analysis).

Usage (GPU recommended for the dense and reranking parts):
    python -m lawrag.experiments --corpus data/corpus.jsonl --questions data/questions.jsonl \
        --out results/m2 --cache data/embeddings
    python -m lawrag.experiments ... --lexical-only      # fast, no models
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from .chunk import chunk_corpus
from .evaluate import check_gold, format_table, load_questions
from .metrics import evaluate, first_gold_rank
from .retrieve import BM25Retriever, DenseRetriever, HybridRetriever, RerankRetriever, load_corpus

HYBRID_WEIGHTS = (0.25, 0.5, 1.0)  # weight of the lexical ranking; dense always 1.0


def build_retrievers(chunks: list[dict], lexical_only: bool = False, cache: Path | None = None,
                     dense_models=("e5", "bge-m3"), rerank: bool = True) -> list:
    lexical = [BM25Retriever(chunks, analyzer=a) for a in ("word", "stem", "ngram")]
    retrievers = list(lexical)
    if lexical_only:
        return retrievers
    dense = [DenseRetriever(chunks, m, cache_dir=cache) for m in dense_models]
    retrievers += dense
    best_lexical = lexical[2]  # stem 4-grams, chosen a priori as the variant built for compounds
    for d in dense:
        retrievers.append(HybridRetriever(lexical[0], d))  # the first-run setting
        for w in HYBRID_WEIGHTS:
            retrievers.append(HybridRetriever(best_lexical, d, weights=[w, 1.0]))
    if rerank:
        model = None
        for d in dense:
            r = RerankRetriever(d, chunks, model=model)
            model = r.model  # load the cross-encoder once
            retrievers.append(r)
    return retrievers


def run(questions: list[dict], retrievers: list) -> tuple[dict, list[dict]]:
    summary, per_q = {}, [
        {"id": q["id"], "question": q["question"], "gold": " | ".join(q["gold"])} for q in questions
    ]
    for r in retrievers:
        start = time.time()
        rankings = {q["id"]: r(q["question"]) for q in questions}  # retrieve each question once
        by_text = {q["question"]: rankings[q["id"]] for q in questions}
        summary[r.name] = evaluate(questions, by_text.__getitem__)
        summary[r.name]["seconds"] = round(time.time() - start, 1)
        for row, q in zip(per_q, questions):
            rank = first_gold_rank(rankings[q["id"]], set(q["gold"]))
            row[r.name] = rank if rank is not None else ""
    return summary, per_q


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--corpus", type=Path, default=Path("data/corpus.jsonl"))
    ap.add_argument("--questions", type=Path, default=Path("data/questions.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("results/m2"))
    ap.add_argument("--cache", type=Path, help="directory for cached chunk embeddings")
    ap.add_argument("--lexical-only", action="store_true")
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--dense", nargs="+", default=["e5", "bge-m3"])
    args = ap.parse_args()

    records = load_corpus(args.corpus)
    chunks = chunk_corpus(records)
    questions = load_questions(args.questions)
    missing = check_gold(questions, records)
    if missing:
        raise SystemExit(f"gold labels not in corpus: {missing}")

    retrievers = build_retrievers(chunks, args.lexical_only, args.cache, tuple(args.dense), not args.no_rerank)
    summary, per_q = run(questions, retrievers)

    header = f"{len(questions)} questions, {len(records)} §, {len(chunks)} chunks"
    table = format_table(summary)
    print(header + "\n\n" + table)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.md").write_text(header + "\n\n" + table + "\n", encoding="utf-8")
    (args.out / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (args.out / "per_question.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_q[0]))
        writer.writeheader()
        writer.writerows(per_q)
    print(f"\nwritten to {args.out}/ (results.md, results.json, per_question.csv)")


if __name__ == "__main__":
    main()
