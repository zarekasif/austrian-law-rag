# Answer run 1: review (2026-11-16)

Setup: E5 → cross-encoder rerank, top 6 chunks (context budget 6,000 characters), Qwen2.5-7B-Instruct
in 4-bit on a Colab T4, 22 questions.

## Automatic checks

| Check | Result |
| --- | --- |
| Answer cites a correct § | 12 / 22 (54.5 %) |
| Said it cannot answer | 2 / 22 (9.1 %) |
| Key fact present (18 questions with facts) | 12 / 18 (66.7 %) |

## Manual grading against the law text

| Grade | Questions |
| --- | --- |
| Correct | q001, q002, q005, q006, q007, q008, q011, q012, q013, q014, q016, q018, q020, q022 (14) |
| Partly wrong | q004 (presents the repayment exception of UrlG § 10 as a condition for the payout), q015 (answers only for fixed-term leases) |
| Wrong | q003, q010, q017, q019 (4) |
| Abstained | q009, q021 (2) |

What went wrong:

- **q010, the most important failure.** The answer says the Mietrechtsgesetz *applies* to a student
  dorm room, citing MRG § 1 Abs. 2 Z 1a. Abs. 2 lists what is *excluded*, and Z 1 excludes student
  dorms. The citation is correct; the reading is inverted. A correct citation does not make a
  correct answer.
- **q003.** "Ja, Sie verlieren diese vier Urlaubstage … werden … nicht angerechnet." The second half
  is right (illness of more than three days does not count as leave), the first half contradicts it.
  The key-fact check passed because "drei" appears, which shows the limits of keyword checks. The
  citation `[1 AZG § 5 (1)]` names the wrong law.
- **q019.** Cites UG § 62 (enrolment deadlines) instead of § 91 (when fees are due), and misses the
  actual rule.
- **q017.** Quotes ECTS caps from another part of UG § 78 and misses the main rule (recognition when
  there are no essential differences).

Evaluation problems found and fixed for run 2:

- 4 correct answers (q001, q011, q020, q022) cited the context passage number (`[1]`) instead of the §;
  passage numbers are now resolved to their §, and the prompt asks for `[LAW § n]` explicitly.
- The context budget of 6,000 characters could drop the sixth chunk; raised to 9,000 so all six fit.
- Two of my key facts were wrong: q008 did not match the inflected "einer halben Stunde"; q014 said
  three years, but the current MRG § 29 requires **five** years when the landlord is a company and three
  otherwise (the model had this right). Both corrected from the law text.
