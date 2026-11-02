# Writing the evaluation questions

The question set is what makes this project more than a demo: every retriever and every
change is measured against it. Aim for **60 questions**, about 10 per law, written by hand.

## Format

One JSON object per line in `data/questions.jsonl`:

```json
{"id": "q001", "question": "Gilt das Mietrechtsgesetz auch für ein Zimmer im Studentenheim?", "gold": ["MRG § 1"], "law": "MRG", "type": "layperson"}
```

- `gold`: every § that answers the question, written exactly like the corpus ids (`"AZG § 11"`).
  Most questions have one; use two or three when the answer needs several §§.
- `type`: one of
  - `layperson`: how a student or tenant would actually ask, in everyday German. **Most questions should be this type.**
  - `legal`: phrased with the law's own terms ("Ruhepause", "Kündigungsfrist").
  - `multi`: needs more than one §.

## Rules that keep the evaluation honest

1. **Do not copy wording from the §.** If the question reuses the §'s rare words, BM25 wins by
   construction. Ask the way a person who has never read the law would ask
   ("Muss mein Chef mir eine Pause geben?", not "Wann ist die Tagesarbeitszeit durch eine Ruhepause zu unterbrechen?").
2. **Find the gold § by reading, not only by searching.** If gold labels are picked from BM25's
   top results, the test set favours BM25 (pooling bias). Use `browse("AZG")` in the notebook to
   read the law's §§ and pick the right one, then check what search returns.
3. **Check the answer is really in the §.** Open it with `show("AZG § 11")` before saving.
4. **Spread across laws and topics.** About 10 per law; within a law, cover different parts.
5. **Freeze the set before tuning.** Once you start changing chunk sizes or models, set aside
   20 questions as a held-out test set and only look at its score at the end.

## Writing in German at B1

That is fine and even useful: simple, slightly informal questions are realistic. Check grammar
with a dictionary if needed, but don't make questions more formal than a real user would.
