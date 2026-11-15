# Answer run 2: review (2026-11-16)

Changes from run 1: passage numbers like `[1]` count as citing that passage's §; the prompt asks for
`[LAW § n]` explicitly; the context budget fits all six chunks (9,000 characters); two key facts
corrected. These changes were made after reading run 1's answers, on the same 22 questions.

Setup otherwise unchanged: E5 → cross-encoder rerank, top 6 chunks, Qwen2.5-7B-Instruct in 4-bit.

## Automatic checks

| Check | Run 1 | Run 2 |
| --- | --- | --- |
| Answer cites a correct § | 54.5 % | 77.3 % |
| Key fact present (18 questions with facts) | 66.7 % | 83.3 % |
| Said it cannot answer | 9.1 % | 13.6 % |

Most of the citation gain comes from scoring `[1]`-style references, not from better answers.

## Manual grading against the law text

| Grade | Run 1 | Run 2 | Run 2 questions |
| --- | --- | --- | --- |
| Correct | 14 | 13 | q001, q002, q005, q007, q008, q011, q012, q013, q014, q016, q018, q020, q022 |
| Partly wrong | 2 | 4 | q004, q006, q015, q019 |
| Wrong | 4 | 2 | q003, q010 |
| Abstained | 2 | 3 | q009, q017, q021 |

- **Fewer wrong answers, not more correct ones.** Two wrong answers became abstentions or partly
  wrong ones; the share of fully correct answers stayed at about 60 %.
- **Automatic checks overstate quality.** q003 and q010 cite the right § and q003 contains the key
  fact, yet both answers are wrong. 17 of 22 answers cite a correct §; 13 are fully correct.
- **The dominant error is misreading exceptions.** q010 (twice, with different clauses) presents an
  exclusion from MRG § 1 Abs. 2 as a reason the law applies; q003 says leave days are lost while
  quoting the rule that they are not counted; q004 turns the repayment exception of UrlG § 10 into a
  condition for the payout. Exception lists ("fallen nicht", "außer bei") are where a 7B model fails.
- **Abstention works when retrieval falls short.** q009 (rest period) and q021 (student work hours,
  in Abs. 7 of the long AuslBG § 4) were answered with "no answer" rather than guesses; for q021 the
  right § was retrieved but probably not the chunk containing Abs. 7.
- Partly wrong: q006 adds a 13-hour exception attributed to AZG § 7 that is not checked; q019 garbles
  who pays tuition fees; q015 only covers fixed-term leases.
