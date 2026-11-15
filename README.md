# Austrian Law RAG

Question answering over Austrian federal law, in German, with answers that cite the § they come
from. The corpus covers six laws that matter when you study, rent, work or live in Austria, taken
from the official legal information system (RIS) in their current version.

> *"Muss mein Chef mir eine Pause geben?"* → **AZG § 11 (Ruhepausen)**: after more than six hours
> of work, a break of at least half an hour.

The point of the project is the measurement: every retrieval method is scored on a hand-written
set of German questions with gold § labels, so each design choice is backed by numbers.

**Status:** milestone 1 done; first retrieval results below. Milestone 2 (stronger baselines, reranking, larger question set) in progress.

## Corpus

| Law | Abbreviation | Topic | §§ (Nov 2026) |
| --- | --- | --- | --- |
| Universitätsgesetz 2002 | UG | studying | 196 |
| Mietrechtsgesetz | MRG | renting | 94 |
| Niederlassungs- und Aufenthaltsgesetz | NAG | residence permits | 91 |
| Arbeitszeitgesetz | AZG | working hours | 79 |
| Ausländerbeschäftigungsgesetz | AuslBG | work permits | 67 |
| Urlaubsgesetz | UrlG | paid leave | 24 |

Counts are API documents. The built corpus has **545 §§ in 1,435 chunks** (the law's title record
is dropped); the longest § is UG § 143 (transitional provisions, ~40k characters, split into many chunks).

## How it works

1. **Download** (`lawrag.ris`, `lawrag.corpus`): the RIS OGD API lists every § of a law in the
   version in force on a given date (`Fassung.FassungVom`); each § is fetched as XML. Requests are
   one second apart and cached, as the RIS usage notes ask.
2. **Parse** (`lawrag.parse`): keeps only the norm text (elements marked `ct="text"`), numbered
   list items and the § title; carries section headings forward, since RIS prints them only on the
   first § of a section.
3. **Chunk** (`lawrag.chunk`): one chunk per § where possible; long §§ split at paragraph
   boundaries. Every chunk starts with a header such as
   `MRG § 1 – Geltungsbereich (Mietrechtsgesetz: I. Hauptstück › Miete)` so it can be cited.
4. **Retrieve** (`lawrag.retrieve`, `lawrag.text`): BM25 (own implementation) on words, on German
   stems (Snowball algorithm, own implementation) or on character 4-grams of stems for compounds;
   dense bi-encoders (multilingual E5, BGE-M3) with cached embeddings; weighted reciprocal rank
   fusion; cross-encoder reranking (BGE reranker v2 m3).
5. **Evaluate** (`lawrag.metrics`, `lawrag.evaluate`, `lawrag.experiments`): Recall@k, MRR@10 and
   nDCG@10 at § level, plus the rank of the correct § per question for error analysis.

## Quick start

```bash
pip install -e ".[dev]"
pytest                                    # 56 tests, offline (real RIS fixtures, fake models)

python -m lawrag.corpus --out data/corpus.jsonl            # needs internet; ~10 min first time
python -m lawrag.evaluate --search "Wie viel Urlaub steht mir zu?"
python -m lawrag.evaluate --retrievers bm25                # scores data/questions.jsonl
```

Full comparison of all retrievers: `python -m lawrag.experiments --cache data/embeddings` (GPU recommended).

Colab notebooks:
[01: corpus, questions, BM25](https://colab.research.google.com/github/zarekasif/austrian-law-rag/blob/main/notebooks/01_corpus_and_bm25.ipynb) ·
[02: compare all retrievers](https://colab.research.google.com/github/zarekasif/austrian-law-rag/blob/main/notebooks/02_compare_retrievers.ipynb).

## First results

22 questions (21 phrased the way a non-lawyer asks), 545 §§, 1,435 chunks. Retrieval at § level.

| Retriever | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 |
| --- | --- | --- | --- | --- | --- | --- |
| BM25 (own implementation, no stemming) | 22.7 | 27.3 | 27.3 | 27.3 | 25.0 | 25.6 |
| Dense: multilingual-e5-base | **50.0** | **72.7** | **81.8** | **90.9** | **62.7** | **69.5** |
| Hybrid: BM25 + E5, reciprocal rank fusion | 31.8 | 31.8 | 45.5 | 68.2 | 38.2 | 44.9 |

What this shows so far:

- **Lexical search fails on everyday wording.** BM25 finds only 6 of 22 §§ in its top 10. People ask
  "Muss mein Chef mir eine Pause geben?"; the law says *Ruhepause*. Without stemming or compound
  splitting, *Pause* never matches *Ruhepause*.
- **The dense model closes most of that gap**: the right § is in the top 10 for 20 of 22 questions.
- **Naive fusion hurts.** Equal-weight reciprocal rank fusion lets BM25's confident wrong hits push
  correct E5 results down. A hybrid only helps if the weak retriever is weighted down or improved.

Caveat: with 22 questions, one question is 4.5 points. The set is being extended to ~60 before
drawing firm conclusions. The questions were drafted with AI assistance and each gold § was checked
against the RIS text; layperson phrasing deliberately tests vocabulary mismatch, which favours dense
retrieval.

## Evaluation set

Hand-written German questions in [`data/questions.jsonl`](data/questions.jsonl), mostly phrased
the way a non-lawyer would ask. How they are written, and how pooling bias towards BM25 is
avoided, is described in [`docs/writing-questions.md`](docs/writing-questions.md).

## Roadmap

1. ~~Corpus, BM25 baseline, evaluation harness~~ (done)
2. Stronger baselines and reranking (code done: stemmed and 4-gram BM25, BGE-M3, weighted hybrid,
   cross-encoder reranking); results pending. Extend the question set to ~60 with a held-out part.
3. Answer generation with an LLM that must cite §§; check that every citation supports the answer.
4. FastAPI service, Docker image and a public demo.

## Data and disclaimer

Legal texts: Rechtsinformationssystem des Bundes (RIS), Bundeskanzleramt Österreich,
[ris.bka.gv.at](https://www.ris.bka.gv.at), licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Only the texts published in the
Bundesgesetzblatt are legally binding.

This is a research and portfolio project, **not legal advice**.

Code: [MIT](LICENSE).
