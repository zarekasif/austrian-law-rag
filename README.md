# Austrian Law RAG

Question answering over Austrian federal law, in German, with answers that cite the § they come
from. The corpus covers six laws that matter when you study, rent, work or live in Austria, taken
from the official legal information system (RIS) in their current version.

> *"Muss mein Chef mir eine Pause geben?"* → **AZG § 11 (Ruhepausen)**: after more than six hours
> of work, a break of at least half an hour.

The point of the project is the measurement: every retrieval method is scored on a hand-written
set of German questions with gold § labels, so each design choice is backed by numbers.

**Status:** milestone 1 of 4: corpus pipeline, BM25 baseline and evaluation harness, all tested.

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
4. **Retrieve** (`lawrag.retrieve`): BM25 (own implementation), a dense bi-encoder (multilingual
   E5) and a hybrid using reciprocal rank fusion.
5. **Evaluate** (`lawrag.metrics`, `lawrag.evaluate`): Recall@k, MRR@10 and nDCG@10 at § level.

## Quick start

```bash
pip install -e ".[dev]"
pytest                                    # 25 tests, offline, using real RIS fixtures

python -m lawrag.corpus --out data/corpus.jsonl            # needs internet; ~10 min first time
python -m lawrag.evaluate --search "Wie viel Urlaub steht mir zu?"
python -m lawrag.evaluate --retrievers bm25                # scores data/questions.jsonl
```

Or run everything in Colab: [`notebooks/01_corpus_and_bm25.ipynb`](notebooks/01_corpus_and_bm25.ipynb)
([open in Colab](https://colab.research.google.com/github/zarekasif/austrian-law-rag/blob/main/notebooks/01_corpus_and_bm25.ipynb)).

## Evaluation set

Hand-written German questions in [`data/questions.jsonl`](data/questions.jsonl), mostly phrased
the way a non-lawyer would ask. How they are written, and how pooling bias towards BM25 is
avoided, is described in [`docs/writing-questions.md`](docs/writing-questions.md).

## Roadmap

1. ~~Corpus, BM25 baseline, evaluation harness~~ (done)
2. Evaluation set of 60 questions; BM25 vs. dense (E5, BGE-M3) vs. hybrid vs. cross-encoder reranking.
   Hypothesis: lexical search misses everyday wording and German compounds (*Pause* vs. *Ruhepause*).
3. Answer generation with an LLM that must cite §§; check that every citation supports the answer.
4. FastAPI service, Docker image and a public demo.

## Data and disclaimer

Legal texts: Rechtsinformationssystem des Bundes (RIS), Bundeskanzleramt Österreich,
[ris.bka.gv.at](https://www.ris.bka.gv.at), licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Only the texts published in the
Bundesgesetzblatt are legally binding.

This is a research and portfolio project, **not legal advice**.

Code: [MIT](LICENSE).
