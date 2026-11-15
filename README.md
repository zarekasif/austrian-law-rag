# Austrian Law RAG

Question answering over Austrian federal law, in German, with answers that cite the § they come
from. The corpus covers six laws that matter when you study, rent, work or live in Austria, taken
from the official legal information system (RIS) in their current version.

> *"Muss mein Chef mir eine Pause geben?"* → **AZG § 11 (Ruhepausen)**: after more than six hours
> of work, a break of at least half an hour.

The point of the project is the measurement: every retrieval method is scored on a hand-written
set of German questions with gold § labels, so each design choice is backed by numbers.

**Status:** milestones 1–3 done. Retrieval: E5 + cross-encoder reranking finds the right § in the top 10 for all 22 questions. Answers: an open 7B model cites the right § for 77 %, but only 59 % of answers are fully correct on manual review, with errors concentrated in exception clauses. Next: service and demo.

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
5. **Evaluate retrieval** (`lawrag.metrics`, `lawrag.evaluate`, `lawrag.experiments`): Recall@k,
   MRR@10 and nDCG@10 at § level, plus the rank of the correct § per question for error analysis.
6. **Answer** (`lawrag.answer`): the top chunks go into a numbered context; an LLM (open
   Qwen2.5-7B-Instruct in 4-bit, or Claude via the API) answers in German, citing §§ as `[AZG § 11]`,
   or says the context does not answer the question.
7. **Evaluate answers** (`lawrag.eval_answers`): is the correct § cited, are all citations really in
   the context, does the answer contain the key fact checked against the law text; plus a sheet
   for grading answers by hand.

## Quick start

```bash
pip install -e ".[dev]"
pytest                                    # 68 tests, offline (real RIS fixtures, fake models)

python -m lawrag.corpus --out data/corpus.jsonl            # needs internet; ~10 min first time
python -m lawrag.evaluate --search "Wie viel Urlaub steht mir zu?"
python -m lawrag.evaluate --retrievers bm25                # scores data/questions.jsonl
```

Full comparison of all retrievers: `python -m lawrag.experiments --cache data/embeddings` (GPU recommended).

Colab notebooks:
[01: corpus, questions, BM25](https://colab.research.google.com/github/zarekasif/austrian-law-rag/blob/main/notebooks/01_corpus_and_bm25.ipynb) ·
[02: compare all retrievers](https://colab.research.google.com/github/zarekasif/austrian-law-rag/blob/main/notebooks/02_compare_retrievers.ipynb) ·
[03: answers with citations](https://colab.research.google.com/github/zarekasif/austrian-law-rag/blob/main/notebooks/03_answers_with_citations.ipynb).

## Results

22 questions (21 phrased the way a non-lawyer asks), 545 §§, 1,435 chunks, retrieval at § level.
Selected rows; all 15 configurations and per-question ranks are in [`results/m2/`](results/m2/).

| Retriever | Recall@1 | Recall@3 | Recall@10 | MRR@10 |
| --- | --- | --- | --- | --- |
| BM25, words | 22.7 | 27.3 | 27.3 | 25.0 |
| BM25, German stems | 36.4 | 36.4 | 45.5 | 38.1 |
| BM25, 4-grams of stems | 36.4 | 54.5 | 68.2 | 46.4 |
| Dense: multilingual-e5-base | 50.0 | 72.7 | 90.9 | 62.7 |
| Dense: BGE-M3 | 68.2 | 72.7 | 86.4 | 73.0 |
| Hybrid: BM25 words + E5, equal weights | 31.8 | 31.8 | 68.2 | 38.2 |
| Hybrid: 0.25 × BM25 4-grams + BGE-M3 | 54.5 | 77.3 | 90.9 | 68.1 |
| BGE-M3 → cross-encoder rerank | 68.2 | 81.8 | 90.9 | 76.1 |
| **E5 → cross-encoder rerank** | **72.7** | **86.4** | **100.0** | **82.4** |

Findings:

1. **Vocabulary mismatch breaks plain BM25**, and German morphology explains much of it. Stemming
   lifts Recall@10 from 27 to 45; character 4-grams of stems, which match parts of compounds, to 68.
   "Muss mein Chef mir eine Pause geben?" moves from rank 69 to rank 1 once *Pause* can match *Ruhepause*.
2. **Dense retrieval is far stronger** on everyday questions. BGE-M3 puts the right § first twice as
   often as E5 (68 vs. 50 Recall@1); E5 finds slightly more within the top 10.
3. **Hybrid search does not pay off here.** An equal-weight mix with plain BM25 is much worse than
   dense alone; down-weighting a better lexical ranking only brings it back to roughly dense-only level.
4. **Reranking is the largest gain.** A cross-encoder over E5's top 30 chunks finds the right § in
   the top 10 for every question and ranks it first for 73%. Its ceiling is the first stage: for
   one question BGE-M3 ranked the § too low for it to be among the 30 reranked chunks.
5. **Remaining hard cases need reasoning, not matching**: "Ich habe bis 23 Uhr gearbeitet. Darf ich
   um 7 Uhr wieder anfangen?" (rest period, AZG § 12) is at best rank 7; the question never says
   *Ruhezeit* or *elf Stunden*.

Caveats: with 22 questions one question is 4.5 points, so differences of a few points are noise.
The hybrid weights come from a three-value grid evaluated on the same questions; the reranker and
4-gram analyzer were chosen in advance. Questions were drafted with AI assistance and each gold §
was checked against the RIS text. The set will grow to ~60, with a held-out part, before the
choice of pipeline is final. The first run (BM25, E5, equal hybrid) reproduced exactly.

## Answers with citations

E5 → rerank → top 6 chunks → Qwen2.5-7B-Instruct (4-bit, Colab T4), German answers citing §§.
Every answer was also graded by hand against the law text
([run 1](results/m3/run1_review.md), [run 2](results/m3/run2_review.md)).

| | Run 1 | Run 2 |
| --- | --- | --- |
| Cites a correct § (automatic) | 54.5 % | 77.3 % |
| Contains the key fact (automatic, 18 questions) | 66.7 % | 83.3 % |
| **Fully correct (manual)** | **14 / 22** | **13 / 22** |
| Partly wrong / wrong (manual) | 2 / 4 | 4 / 2 |
| "No answer" | 2 | 3 |

Run 2 changed how `[1]`-style references are scored, the citation instruction and the context
budget, after reading run 1 on the same questions; most of its citation gain is a scoring fix.

Findings:

1. **Automatic citation checks overstate quality.** 17 of 22 run-2 answers cite the right §, but
   only 13 are fully correct. Two answers cite the right § and are still wrong.
2. **The model misreads exceptions.** Asked whether tenancy law covers a student dorm, it cites
   MRG § 1 and turns the list of *excluded* cases into a reason the law *applies*. Similar inversions
   occur for sick days during leave (UrlG § 5) and leave payout (UrlG § 10).
3. **Saying "no answer" works.** When the right passage was not in the context, the model mostly
   declined instead of guessing.

Next steps for answer quality: a stronger generator, quoting the exact sentence before answering,
and retrieving the specific Absatz of long §§.

## Evaluation set

German questions in [`data/questions.jsonl`](data/questions.jsonl), mostly phrased
the way a non-lawyer would ask. How they are written, and how pooling bias towards BM25 is
avoided, is described in [`docs/writing-questions.md`](docs/writing-questions.md). 18 of 22 questions
also carry a key fact taken from the § (e.g. *11 Stunden* for the rest period) to check answers.

## Roadmap

1. ~~Corpus, BM25 baseline, evaluation harness~~ (done)
2. ~~Stronger baselines and reranking~~ (done, results above). Still to do: extend the question
   set to ~60 with a held-out part.
3. ~~Answer generation with citations; automatic and manual evaluation~~ (done, results above).
   Open: a stronger generator, quote-then-answer prompting, Absatz-level retrieval.
4. FastAPI service, Docker image and a public demo.

## Data and disclaimer

Legal texts: Rechtsinformationssystem des Bundes (RIS), Bundeskanzleramt Österreich,
[ris.bka.gv.at](https://www.ris.bka.gv.at), licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Only the texts published in the
Bundesgesetzblatt are legally binding.

This is a research and portfolio project, **not legal advice**.

Code: [MIT](LICENSE).
