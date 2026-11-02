import pytest

from lawrag.bm25 import BM25, tokenize
from lawrag.chunk import chunk_record, header
from lawrag.evaluate import check_gold, format_table
from lawrag.metrics import evaluate, mrr_at_k, ndcg_at_k, recall_at_k, sections_from_chunks
from lawrag.retrieve import BM25Retriever, HybridRetriever, reciprocal_rank_fusion

RECORDS = [
    {"id": "UrlG § 2", "law": "UrlG", "law_title": "Urlaubsgesetz", "section": "§ 2", "title": "Urlaubsausmaß",
     "path": [], "text": "§ 2. (1) Der Arbeitnehmer hat für jedes Arbeitsjahr Anspruch auf einen ununterbrochenen bezahlten Urlaub."},
    {"id": "AZG § 11", "law": "AZG", "law_title": "Arbeitszeitgesetz", "section": "§ 11", "title": "Ruhepausen",
     "path": ["Abschnitt 3", "Ruhepausen und Ruhezeiten"], "text": "§ 11. (1) Beträgt die Gesamtdauer der Tagesarbeitszeit mehr als sechs Stunden, so ist die Arbeitszeit durch eine Ruhepause zu unterbrechen."},
    {"id": "MRG § 1", "law": "MRG", "law_title": "Mietrechtsgesetz", "section": "§ 1", "title": "Geltungsbereich",
     "path": ["I. Hauptstück", "Miete"], "text": "§ 1. (1) Dieses Bundesgesetz gilt für die Miete von Wohnungen.\n(2) Nicht erfasst sind Heime für Schüler oder Studenten."},
]


# --- tokenizer and BM25 ---------------------------------------------------------------

def test_tokenize_casefolds_and_drops_stopwords():
    assert tokenize("Die Straße und der Urlaub") == ["strasse", "urlaub"]


def test_bm25_ranks_the_matching_document_first():
    index = BM25(["Urlaub Anspruch Arbeitsjahr", "Ruhepause Arbeitszeit Stunden", "Miete Wohnungen Studenten"])
    assert index.top_k("Wie viel Urlaub habe ich?", 3)[0][0] == 0
    assert index.top_k("Pause nach sechs Stunden Arbeitszeit", 3)[0][0] == 1


def test_bm25_rare_terms_weigh_more_than_common_ones():
    index = BM25(["miete wohnung", "miete wohnung", "miete heim"])
    assert index.idf["heim"] > index.idf["miete"]


def test_bm25_returns_nothing_for_unknown_words():
    assert BM25(["miete wohnung"]).top_k("quantenphysik", 5) == []


# --- chunking ---------------------------------------------------------------------------

def test_header_names_law_section_title_and_path():
    assert header(RECORDS[2]) == "MRG § 1 – Geltungsbereich (Mietrechtsgesetz: I. Hauptstück › Miete)"


def test_short_record_is_one_chunk_with_header():
    chunks = chunk_record(RECORDS[0])
    assert len(chunks) == 1 and chunks[0]["section_id"] == "UrlG § 2"
    assert chunks[0]["text"].startswith("UrlG § 2 – Urlaubsausmaß (Urlaubsgesetz)\n§ 2. (1)")


def test_long_record_splits_at_blocks_and_keeps_all_text():
    blocks = [f"({i}) " + "Wort " * 60 for i in range(1, 8)]
    rec = dict(RECORDS[2], text="\n".join(blocks))
    chunks = chunk_record(rec, max_chars=700)
    assert len(chunks) > 1
    assert all(len(c["text"]) <= 700 + len(header(rec)) + 1 for c in chunks)
    joined = "\n".join(c["text"].split("\n", 1)[1] for c in chunks)
    assert joined == rec["text"]
    assert [c["chunk_id"] for c in chunks][:2] == ["MRG § 1#0", "MRG § 1#1"]


def test_a_single_overlong_block_is_split_by_sentence():
    rec = dict(RECORDS[0], text=" ".join(f"Satz Nummer {i} endet hier." for i in range(80)))
    chunks = chunk_record(rec, max_chars=400)
    assert len(chunks) > 1 and all(c["text"].split("\n", 1)[1].endswith(".") for c in chunks)


# --- metrics ----------------------------------------------------------------------------

def test_sections_from_chunks_keeps_first_rank():
    assert sections_from_chunks(["A", "B", "A", "C", "B"]) == ["A", "B", "C"]


def test_rank_metrics():
    ranked = ["X", "G1", "Y", "G2"]
    gold = {"G1", "G2"}
    assert recall_at_k(ranked, gold, 1) == 0.0
    assert recall_at_k(ranked, gold, 3) == 0.5
    assert recall_at_k(ranked, gold, 4) == 1.0
    assert mrr_at_k(ranked, gold) == 0.5
    assert ndcg_at_k(["G1", "G2"], gold) == pytest.approx(1.0)
    assert 0 < ndcg_at_k(ranked, gold) < 1


def test_evaluate_averages_in_percent():
    qs = [{"question": "a", "gold": ["A"]}, {"question": "b", "gold": ["B"]}]
    result = evaluate(qs, lambda q: ["A", "B"])
    assert result["recall@1"] == pytest.approx(50.0)
    assert result["mrr@10"] == pytest.approx(75.0)
    assert result["questions"] == 2


# --- retrievers and evaluation helpers -------------------------------------------------

def test_bm25_retriever_end_to_end():
    from lawrag.chunk import chunk_corpus

    retriever = BM25Retriever(chunk_corpus(RECORDS))
    assert retriever("Gilt das Mietrecht für ein Studentenheim? Heime Studenten")[0] == "MRG § 1"
    assert retriever("Wann muss ich eine Ruhepause machen?")[0] == "AZG § 11"
    qs = [{"question": "Anspruch auf bezahlten Urlaub", "gold": ["UrlG § 2"]}]
    assert evaluate(qs, retriever)["recall@1"] == 100.0


def test_reciprocal_rank_fusion_rewards_agreement():
    fused = reciprocal_rank_fusion([["A", "B", "C"], ["B", "A", "D"]])
    assert set(fused[:2]) == {"A", "B"} and fused[-1] in {"C", "D"}
    assert reciprocal_rank_fusion([["A", "B"], ["B", "C"]])[0] == "B"


def test_hybrid_retriever_combines_names_and_rankings():
    class Fixed:
        def __init__(self, name, ranking):
            self.name, self.ranking = name, ranking

        def __call__(self, query, k=100):
            return self.ranking

    hybrid = HybridRetriever(Fixed("a", ["X", "Y"]), Fixed("b", ["Y", "Z"]))
    assert hybrid.name == "a+b" and hybrid("q")[0] == "Y"


def test_check_gold_reports_unknown_sections():
    qs = [{"question": "q", "gold": ["MRG § 1", "MRG § 999"]}]
    assert check_gold(qs, RECORDS) == ["MRG § 999"]


def test_format_table_has_one_row_per_retriever():
    m = {"recall@1": 10, "recall@3": 20, "recall@5": 30, "recall@10": 40, "mrr@10": 15, "ndcg@10": 25}
    table = format_table({"bm25": m, "e5": m})
    assert table.count("\n") == 3 and "| bm25 | 10.0 |" in table
