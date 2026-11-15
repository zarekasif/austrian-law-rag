import pytest

from lawrag.bm25 import BM25
from lawrag.text import analyze, char_ngrams, german_stem


@pytest.mark.parametrize(
    "word, stem",
    [
        # reference pairs from the Snowball German sample vocabulary
        ("aufeinanderfolgenden", "aufeinanderfolg"),
        ("kategorischen", "kategor"),
        ("häuser", "haus"),
        ("straße", "strass"),
        # inflections that must collapse
        ("pause", "paus"),
        ("pausen", "paus"),
        ("kündigung", "kundig"),
        ("kündigen", "kundig"),
        ("urlaubstage", "urlaubstag"),
        ("urlaubstagen", "urlaubstag"),
        ("arbeitszeiten", "arbeitszeit"),
        ("wiederholung", "wiederhol"),
        ("wiederholen", "wiederhol"),
        ("ergebnisse", "ergebnis"),  # -niss rule
        ("möglichkeit", "moglich"),  # -keit, then -ig/-lich rule
        ("bauen", "bau"),  # u between vowels is a consonant
    ],
)
def test_german_stem(word, stem):
    assert german_stem(word) == stem


def test_stemming_does_not_split_compounds():
    assert german_stem("ruhepause") == "ruhepaus" != german_stem("pause")


def test_char_ngrams():
    assert char_ngrams("paus") == ["paus"]
    assert char_ngrams("ruhepaus") == ["ruhe", "uhep", "hepa", "epau", "paus"]


def test_analyzer_modes():
    text = "Die Pausen und die Ruhepause"
    assert analyze(text, "word") == ["pausen", "ruhepause"]
    assert analyze(text, "stem") == ["paus", "ruhepaus"]
    assert "paus" in analyze("Ruhepause", "ngram")
    with pytest.raises(ValueError):
        analyze(text, "nonsense")


def test_ngram_bm25_matches_compounds_that_word_bm25_misses():
    docs = [
        "Beträgt die Tagesarbeitszeit mehr als sechs Stunden, ist eine Ruhepause zu gewähren.",
        "Der Mieter kann den Mietvertrag schriftlich kündigen.",
    ]
    query = "Muss mein Chef mir eine Pause geben?"
    assert BM25(docs, analyzer="word").top_k(query) == []
    assert BM25(docs, analyzer="ngram").top_k(query)[0][0] == 0


def test_stem_bm25_matches_inflections():
    docs = ["Kündigung des Mietvertrags durch den Mieter", "Urlaubsausmaß pro Arbeitsjahr"]
    assert BM25(docs, analyzer="word").top_k("Wie kann ich kündigen?") == []
    assert BM25(docs, analyzer="stem").top_k("Wie kann ich kündigen?")[0][0] == 0
