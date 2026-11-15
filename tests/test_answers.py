import csv
import json

from lawrag.answer import NO_ANSWER, AnswerPipeline, build_context, is_abstention, parse_citations, user_prompt
from lawrag.chunk import chunk_corpus
from lawrag.eval_answers import fact_hit, run, score, summarize, write_outputs

RECORDS = [
    {"id": "AZG § 11", "law": "AZG", "law_title": "Arbeitszeitgesetz", "section": "§ 11", "title": "Ruhepausen",
     "path": [], "text": "§ 11. (1) Beträgt die Tagesarbeitszeit mehr als sechs Stunden, ist eine Ruhepause von mindestens einer halben Stunde zu gewähren."},
    {"id": "AZG § 12", "law": "AZG", "law_title": "Arbeitszeitgesetz", "section": "§ 12", "title": "Ruhezeiten",
     "path": [], "text": "§ 12. (1) Nach Beendigung der Tagesarbeitszeit ist eine ununterbrochene Ruhezeit von mindestens elf Stunden zu gewähren."},
    {"id": "MRG § 16b", "law": "MRG", "law_title": "Mietrechtsgesetz", "section": "§ 16b", "title": "Kaution",
     "path": [], "text": "§ 16b. (1) Die Kaution ist nach Ende des Mietvertrags samt Zinsen unverzüglich zurückzustellen."},
]
CHUNKS = chunk_corpus(RECORDS)


# --- citation parsing -------------------------------------------------------------------

def test_parse_citations_in_requested_format():
    assert parse_citations("Ja, nach sechs Stunden [AZG § 11]. Siehe auch [MRG § 16b].") == ["AZG § 11", "MRG § 16b"]


def test_parse_citations_in_austrian_order_with_details():
    text = "Gemäß § 11 Abs. 1 AZG und § 4 Abs. 7 Z 2 AuslBG sowie §16b MRG."
    assert parse_citations(text) == ["AZG § 11", "AuslBG § 4", "MRG § 16b"]


def test_parse_citations_deduplicates_and_ignores_unknown_laws():
    assert parse_citations("[AZG § 11] ... [AZG § 11] ... § 5 ABGB") == ["AZG § 11"]


def test_auslbg_is_not_read_as_another_law():
    assert parse_citations("[AuslBG § 19]") == ["AuslBG § 19"]


def test_abstention():
    assert is_abstention(NO_ANSWER)
    assert not is_abstention("Ja, [AZG § 11].")


# --- context ------------------------------------------------------------------------------

def test_build_context_numbers_chunks_and_respects_budget():
    context, sections = build_context(CHUNKS, [2, 0, 1], max_chars=10_000)
    assert context.startswith("[1] MRG § 16b – Kaution") and "\n\n[2] AZG § 11" in context
    assert sections == ["MRG § 16b", "AZG § 11", "AZG § 12"]
    _, short = build_context(CHUNKS, [2, 0, 1], max_chars=150)
    assert short == ["MRG § 16b"]  # the first chunk is always kept, later ones must fit
    _, repeated = build_context(CHUNKS, [0, 0], max_chars=10_000)
    assert repeated == ["AZG § 11", "AZG § 11"]  # one entry per numbered passage


def test_user_prompt_contains_question_and_context():
    p = user_prompt("Frage?", "[1] Text")
    assert "Frage: Frage?" in p and "[1] Text" in p


# --- pipeline and scoring with fake components -------------------------------------------

class FakeRetriever:
    name = "fake"

    def __init__(self, ranking):
        self.ranking = ranking

    def rank_chunks(self, query, k=100):
        return self.ranking[:k]


class ScriptedGenerator:
    """Returns canned answers in order and records the prompts it was given."""

    name = "scripted"

    def __init__(self, answers):
        self.answers = list(answers)
        self.prompts = []

    def __call__(self, system, user):
        self.prompts.append((system, user))
        return self.answers.pop(0)


QUESTIONS = [
    {"id": "q1", "question": "Muss mein Chef mir eine Pause geben?", "gold": ["AZG § 11"],
     "facts": [["halbe Stunde", "30 Minuten"]]},
    {"id": "q2", "question": "Bekomme ich meine Kaution zurück?", "gold": ["MRG § 16b"], "facts": [["Zinsen"]]},
    {"id": "q3", "question": "Wie viele Urlaubstage habe ich?", "gold": ["UrlG § 2"]},
]


def test_pipeline_passes_context_and_parses_answer():
    gen = ScriptedGenerator(["Ja, nach sechs Stunden eine halbe Stunde [AZG § 11]."])
    a = AnswerPipeline(FakeRetriever([0, 1]), CHUNKS, gen)("Muss mein Chef mir eine Pause geben?")
    assert a.citations == ["AZG § 11"] and a.context_sections == ["AZG § 11", "AZG § 12"]
    system, user = gen.prompts[0]
    assert "[AZG § 11]" in system and "Ruhepause" in user and not a.abstained


def test_score_detects_invented_citation_and_missing_fact():
    gen = ScriptedGenerator(["Ja, laut [AZG § 11] und [UrlG § 2]."])
    a = AnswerPipeline(FakeRetriever([0]), CHUNKS, gen)(QUESTIONS[0]["question"])
    s = score(QUESTIONS[0], a)
    assert s["cites_gold"] and s["citation_precision"] == 0.5 and s["fact_hit"] is False


def test_fact_hit_rules():
    assert fact_hit("Sie haben 30 Werktage", [["30"]]) is True
    assert fact_hit("Sie haben 300 Euro", [["30"]]) is False  # numbers match whole numbers only
    assert fact_hit("eine HALBE Stunde", [["halbe Stunde"]]) is True
    assert fact_hit("x", None) is None
    assert fact_hit("elf Stunden und schriftlich", [["11", "elf"], ["schriftlich"]]) is True


def test_run_and_summary(tmp_path):
    gen = ScriptedGenerator([
        "Ja, mindestens eine halbe Stunde [AZG § 11].",
        "Ja, samt Zinsen (§ 16b MRG).",
        NO_ANSWER,
    ])
    pipeline = AnswerPipeline(FakeRetriever([0, 2]), CHUNKS, gen)
    rows, summary = run(QUESTIONS, pipeline, progress=False)
    assert [r["cites_gold"] for r in rows] == [True, True, False]
    assert summary["cites_gold"] == 66.7 and summary["abstained"] == 33.3
    assert summary["fact_hit"] == 100.0 and summary["fact_questions"] == 2
    assert summary["citation_precision"] == 100.0 and summary["cites_any_when_answering"] == 100.0
    assert summary["gold_in_context"] == 66.7  # UrlG § 2 was never retrieved

    write_outputs(rows, summary, tmp_path, {"generator": "scripted"})
    assert json.loads((tmp_path / "summary.json").read_text())["generator"] == "scripted"
    review = list(csv.DictReader((tmp_path / "review.csv").open(encoding="utf-8")))
    assert review[1]["citations"] == "MRG § 16b" and review[0]["manual_correct"] == ""


def test_summarize_handles_all_abstentions():
    rows = [{"gold_in_context": False, "abstained": True, "cites_any": False, "cites_gold": False,
             "citation_precision": None, "fact_hit": None}]
    s = summarize(rows)
    assert s["cites_any_when_answering"] is None and s["citation_precision"] is None


def test_chat_prompt_asks_the_template_for_text():
    from lawrag.answer import chat_prompt

    class Tok:
        def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False, **kw):
            assert tokenize is False and add_generation_prompt is True and "return_tensors" not in kw
            return "|".join(f"{m['role']}:{m['content']}" for m in messages) + "|assistant:"

    assert chat_prompt(Tok(), "S", "U") == "system:S|user:U|assistant:"


def test_context_numbers_count_as_citations():
    numbered = ["UrlG § 2", "UrlG § 5", "AZG § 11"]
    assert parse_citations("30 Werktage [1].", numbered) == ["UrlG § 2"]
    assert parse_citations("Siehe [1][3].", numbered) == ["UrlG § 2", "AZG § 11"]
    # passage number plus an invented law name: both are recorded, so precision drops
    assert parse_citations("nicht angerechnet [1 AZG § 5 (1)].", numbered) == ["UrlG § 2", "AZG § 5"]
    assert parse_citations("[9]", numbered) == []  # no such passage
    assert parse_citations("30 Werktage [1].") == []  # without the mapping, numbers are ignored


def test_pipeline_resolves_passage_numbers():
    gen = ScriptedGenerator(["Ja, eine halbe Stunde [1]."])
    a = AnswerPipeline(FakeRetriever([0, 1]), CHUNKS, gen)("Pause?")
    assert a.citations == ["AZG § 11"] and a.extra["numbered"] == ["AZG § 11", "AZG § 12"]
    assert "Schreibe nicht nur [1]" in gen.prompts[0][0]
