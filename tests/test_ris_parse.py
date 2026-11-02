import json
from pathlib import Path

from lawrag.corpus import build_record, section_key
from lawrag.parse import carry_headings, parse_paragraph_xml
from lawrag.ris import parse_search, search_url

FIX = Path(__file__).parent / "fixtures"


def test_parse_search_reads_real_response():
    total, refs = parse_search(json.loads((FIX / "search_mrg.json").read_text(encoding="utf-8")))
    assert total == 94
    assert [r.section for r in refs] == ["§ 0", "§ 1", "§ 2"]
    assert [r.doc_type for r in refs] == ["Norm", "Paragraph", "Paragraph"]
    one = refs[1]
    assert one.law == "MRG" and one.law_title == "Mietrechtsgesetz"
    assert one.xml_url.endswith("NOR40105099.xml")
    assert one.url == "https://ris.bka.gv.at/eli/bgbl/1981/520/P1/NOR40105099"
    assert one.index == ["20/05 Wohn- und Mietrecht"]


def test_parse_search_handles_single_hit_as_object():
    payload = json.loads((FIX / "search_mrg.json").read_text(encoding="utf-8"))
    results = payload["OgdSearchResult"]["OgdDocumentResults"]
    results["OgdDocumentReference"] = results["OgdDocumentReference"][1]  # API returns a dict for 1 hit
    results["Hits"]["#text"] = "1"
    total, refs = parse_search(payload)
    assert total == 1 and len(refs) == 1 and refs[0].section == "§ 1"


def test_search_url_asks_for_current_version_and_paging():
    url = search_url("10002531", "2026-11-02", page=3)
    assert "Fassung.FassungVom=2026-11-02" in url and "Seitennummer=3" in url and "Applikation=BrKons" in url


def test_parse_paragraph_extracts_title_headings_and_text_only():
    p = parse_paragraph_xml((FIX / "NOR40105099.xml").read_bytes())
    assert p.title == "Geltungsbereich"
    assert p.headings == {1: "I. Hauptstück", 2: "Miete"}
    assert p.blocks[0].startswith("§ 1. (1) Dieses Bundesgesetz gilt für die Miete von Wohnungen")
    text = "\n".join(p.blocks)
    # list items keep their numbering, separated from the text
    assert "1. Mietgegenstände, die im Rahmen des Betriebes" in text
    assert "1a. Wohnungen oder Wohnräume" in text
    # nested items (ebene 2) are indented
    assert "\n  a) eine Geschäftsräumlichkeit oder" in text
    # page headers/footers and metadata are not part of the text
    assert "Bundesrecht konsolidiert" not in text and "Seite" not in text and "Schlagworte" not in text
    assert p.meta["abkuerzung"] == "MRG" and p.meta["artikel_anlage"] == "§ 1"


def test_carry_headings_replaces_level_and_drops_deeper_levels():
    path = {1: "I. Hauptstück", 2: "Miete", 3: "Erhaltung"}
    assert carry_headings(path, {}) == path
    assert carry_headings(path, {2: "Mietzins"}) == {1: "I. Hauptstück", 2: "Mietzins"}
    assert carry_headings(path, {1: "II. Hauptstück"}) == {1: "II. Hauptstück"}


def test_build_record_from_real_documents():
    _, refs = parse_search(json.loads((FIX / "search_mrg.json").read_text(encoding="utf-8")))
    record, path = build_record(refs[1], (FIX / "NOR40105099.xml").read_bytes(), {}, topic="Wohnen")
    assert record["id"] == "MRG § 1" == section_key("MRG", "§  1")
    assert record["path"] == ["I. Hauptstück", "Miete"] and path == {1: "I. Hauptstück", 2: "Miete"}
    assert record["title"] == "Geltungsbereich" and record["topic"] == "Wohnen"
    assert "Studentenheim" in record["keywords"]


class _OfflineClient:
    """Serves the fixtures instead of the network; counts search pages requested."""

    def __init__(self, pages):
        from lawrag.ris import RisClient

        self.pages = pages
        self.search_calls = 0
        self.law_documents = lambda gesnr, date: RisClient.law_documents(self, gesnr, date)

    def get(self, url):
        if "Seitennummer=" in url:
            page = int(url.rsplit("Seitennummer=", 1)[1])
            self.search_calls += 1
            return json.dumps(self.pages[min(page, len(self.pages)) - 1]).encode()
        return (FIX / "NOR40105099.xml").read_bytes()


def _page(refs_slice, total):
    payload = json.loads((FIX / "search_mrg.json").read_text(encoding="utf-8"))
    res = payload["OgdSearchResult"]["OgdDocumentResults"]
    res["OgdDocumentReference"] = res["OgdDocumentReference"][refs_slice]
    res["Hits"]["#text"] = str(total)
    return payload


def test_law_documents_pages_until_all_hits_are_collected():
    client = _OfflineClient([_page(slice(0, 2), 3), _page(slice(2, 3), 3)])
    refs = client.law_documents("10002531", "2026-11-02")
    assert [r.section for r in refs] == ["§ 0", "§ 1", "§ 2"] and client.search_calls == 2


def test_law_documents_stops_if_a_page_repeats():
    same = _page(slice(0, 2), 50)  # claims 50 hits but keeps returning the same page
    client = _OfflineClient([same, same])
    assert len(client.law_documents("10002531", "2026-11-02")) == 2 and client.search_calls == 2


def test_build_law_skips_the_norm_record():
    from lawrag.corpus import build_law

    client = _OfflineClient([_page(slice(0, 3), 3)])
    records = build_law(client, "MRG", "2026-11-02")
    assert [r["section"] for r in records] == ["§ 1", "§ 2"]  # § 0 ("Norm") dropped
    assert records[0]["path"] == ["I. Hauptstück", "Miete"]


def test_build_record_uses_our_abbreviation_when_ris_has_none():
    _, refs = parse_search(json.loads((FIX / "search_mrg.json").read_text(encoding="utf-8")))
    ref = refs[1]
    ref.law = "Urlaubsgesetz"  # what RIS returns for a law without Abkuerzung
    record, _ = build_record(ref, (FIX / "NOR40105099.xml").read_bytes(), {}, law="UrlG")
    assert record["law"] == "UrlG" and record["id"].startswith("UrlG § ")
