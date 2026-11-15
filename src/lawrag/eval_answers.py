"""Generate answers for the question set and score them.

Automatic checks per answer:
- ``gold_in_context``: the correct § was among the retrieved sections (retrieval upper bound)
- ``abstained``: the model said the context does not answer the question
- ``cites_gold``: the answer cites a correct §
- ``citation_precision``: share of cited §§ that were actually in the context (the rest are
  invented citations)
- ``fact_hit``: the answer contains the key fact(s) checked against the law text (questions
  with a ``facts`` field only; a rough proxy, not a full correctness judgement)

A review sheet (``review.csv``) lists every answer with an empty ``manual_correct`` column for
grading by hand.

Usage (GPU for the default open model):
    python -m lawrag.eval_answers --corpus data/corpus.jsonl --questions data/questions.jsonl \
        --cache data/embeddings --out results/m3
    python -m lawrag.eval_answers ... --backend anthropic --model claude-haiku-4-5
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from statistics import mean

from .answer import Answer, AnswerPipeline


def _contains(text: str, alternative: str) -> bool:
    if alternative.isdigit():
        return re.search(rf"(?<!\d){alternative}(?!\d)", text) is not None
    return alternative.casefold() in text.casefold()


def fact_hit(answer: str, facts: list[list[str]] | None) -> bool | None:
    """Every fact group must be matched by at least one of its alternatives."""
    if not facts:
        return None
    return all(any(_contains(answer, alt) for alt in group) for group in facts)


def score(question: dict, a: Answer) -> dict:
    gold = set(question["gold"])
    cited = a.citations
    in_context = [c for c in cited if c in a.context_sections]
    return {
        "gold_in_context": bool(gold & set(a.context_sections)),
        "abstained": a.abstained,
        "cites_any": bool(cited),
        "cites_gold": bool(gold & set(cited)),
        "citation_precision": len(in_context) / len(cited) if cited else None,
        "fact_hit": fact_hit(a.answer, question.get("facts")),
    }


def summarize(rows: list[dict]) -> dict:
    def pct(key):
        vals = [r[key] for r in rows if r[key] is not None]
        return round(100 * mean(vals), 1) if vals else None

    answered = [r for r in rows if not r["abstained"]]
    return {
        "questions": len(rows),
        "gold_in_context": pct("gold_in_context"),
        "abstained": pct("abstained"),
        "cites_gold": pct("cites_gold"),
        "cites_any_when_answering": round(100 * mean(r["cites_any"] for r in answered), 1) if answered else None,
        "citation_precision": pct("citation_precision"),
        "fact_hit": pct("fact_hit"),
        "fact_questions": sum(r["fact_hit"] is not None for r in rows),
    }


def run(questions: list[dict], pipeline: AnswerPipeline, progress: bool = True) -> tuple[list[dict], dict]:
    rows = []
    for q in questions:
        a = pipeline(q["question"])
        row = {"id": q["id"], "question": q["question"], "gold": q["gold"], "answer": a.answer,
               "citations": a.citations, "context_sections": a.context_sections, **score(q, a)}
        rows.append(row)
        if progress:
            flag = "ok " if row["cites_gold"] else ("-- " if not row["abstained"] else "?? ")
            print(f"{flag}{q['id']}  cites {a.citations}  gold {q['gold']}")
    return rows, summarize(rows)


def write_outputs(rows: list[dict], summary: dict, out: Path, meta: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with (out / "answers.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out / "summary.json").write_text(json.dumps({**meta, **summary}, indent=2, ensure_ascii=False), encoding="utf-8")
    with (out / "review.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "question", "gold", "answer", "citations", "cites_gold", "fact_hit", "manual_correct", "note"])
        for r in rows:
            w.writerow([r["id"], r["question"], " | ".join(r["gold"]), r["answer"], " | ".join(r["citations"]),
                        r["cites_gold"], "" if r["fact_hit"] is None else r["fact_hit"], "", ""])


def format_summary(summary: dict) -> str:
    labels = [
        ("gold_in_context", "correct § retrieved into the context"),
        ("cites_gold", "answer cites a correct §"),
        ("citation_precision", "citations that exist in the context"),
        ("fact_hit", f"key fact present ({summary['fact_questions']} questions with facts)"),
        ("abstained", "said it cannot answer"),
        ("cites_any_when_answering", "answers with at least one citation"),
    ]
    return "\n".join(f"{summary[k]!s:>6} %  {text}" for k, text in labels)


def main() -> None:
    from .chunk import chunk_corpus
    from .evaluate import check_gold, load_questions
    from .retrieve import DenseRetriever, RerankRetriever, load_corpus

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--corpus", type=Path, default=Path("data/corpus.jsonl"))
    ap.add_argument("--questions", type=Path, default=Path("data/questions.jsonl"))
    ap.add_argument("--cache", type=Path, help="directory for cached chunk embeddings")
    ap.add_argument("--out", type=Path, default=Path("results/m3"))
    ap.add_argument("--backend", choices=["hf", "anthropic"], default="hf")
    ap.add_argument("--model", help="model id (default: Qwen/Qwen2.5-7B-Instruct or claude-haiku-4-5)")
    ap.add_argument("--top-chunks", type=int, default=6)
    args = ap.parse_args()

    records = load_corpus(args.corpus)
    chunks = chunk_corpus(records)
    questions = load_questions(args.questions)
    missing = check_gold(questions, records)
    if missing:
        raise SystemExit(f"gold labels not in corpus: {missing}")

    retriever = RerankRetriever(DenseRetriever(chunks, "e5", cache_dir=args.cache), chunks)
    if args.backend == "hf":
        from .answer import HFGenerator

        generator = HFGenerator(args.model or "Qwen/Qwen2.5-7B-Instruct")
    else:
        from .answer import AnthropicGenerator

        generator = AnthropicGenerator(args.model or "claude-haiku-4-5")

    pipeline = AnswerPipeline(retriever, chunks, generator, top_chunks=args.top_chunks)
    rows, summary = run(questions, pipeline)
    meta = {"retriever": retriever.name, "generator": generator.name, "top_chunks": args.top_chunks}
    write_outputs(rows, summary, args.out, meta)
    print(f"\n{meta['retriever']}  →  {meta['generator']}, {summary['questions']} questions\n")
    print(format_summary(summary))
    print(f"\nwritten to {args.out}/ (summary.json, answers.jsonl, review.csv)")


if __name__ == "__main__":
    main()
