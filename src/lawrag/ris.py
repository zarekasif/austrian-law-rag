"""Client for the RIS Open Government Data API (consolidated Austrian federal law).

Only the standard library is used. Requests are spaced out and cached on disk, following
the RIS OGD usage notes: no key needed, pause between requests, avoid repeated mass
downloads. Data licence: CC BY 4.0, Bundeskanzleramt Österreich (ris.bka.gv.at).
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path

API_URL = "https://data.bka.gv.at/ris/api/v2.6/Bundesrecht"
USER_AGENT = "austrian-law-rag/0.1 (+https://github.com/zarekasif/austrian-law-rag)"

# Laws in the corpus: abbreviation -> (Gesetzesnummer, short title, topic).
LAWS: dict[str, tuple[str, str, str]] = {
    "UG": ("20002128", "Universitätsgesetz 2002", "Studium"),
    "MRG": ("10002531", "Mietrechtsgesetz", "Wohnen"),
    "NAG": ("20004242", "Niederlassungs- und Aufenthaltsgesetz", "Aufenthalt"),
    "AZG": ("10008238", "Arbeitszeitgesetz", "Arbeit"),
    "AuslBG": ("10008365", "Ausländerbeschäftigungsgesetz", "Arbeit"),
    "UrlG": ("10008376", "Urlaubsgesetz", "Arbeit"),
}


@dataclass
class DocRef:
    """One search hit: a single § (or article/annex) in its current version."""

    doc_id: str
    law: str
    law_title: str
    doc_type: str
    section: str
    in_force: str
    url: str
    xml_url: str
    keywords: str = ""
    index: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _as_list(value) -> list:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def search_url(gesetzesnummer: str, date: str, page: int = 1, page_size: str = "OneHundred") -> str:
    params = {
        "Applikation": "BrKons",
        "Gesetzesnummer": gesetzesnummer,
        "Fassung.FassungVom": date,
        "DokumenteProSeite": page_size,
        "Seitennummer": str(page),
    }
    return f"{API_URL}?{urllib.parse.urlencode(params)}"


def parse_search(payload: dict) -> tuple[int, list[DocRef]]:
    """Return (total hits, document references) from one search response page."""
    results = payload.get("OgdSearchResult", {}).get("OgdDocumentResults", {})
    hits_field = results.get("Hits", {})
    total = int(hits_field.get("#text", 0)) if isinstance(hits_field, dict) else int(hits_field or 0)

    refs = []
    for item in _as_list(results.get("OgdDocumentReference")):
        data = item.get("Data", {})
        meta = data.get("Metadaten", {})
        br = meta.get("Bundesrecht", {})
        kons = br.get("BrKons", {})
        urls = _as_list(
            data.get("Dokumentliste", {}).get("ContentReference", {}).get("Urls", {}).get("ContentUrl")
        )
        xml_url = next((u.get("Url", "") for u in urls if u.get("DataType") == "Xml"), "")
        refs.append(
            DocRef(
                doc_id=meta.get("Technisch", {}).get("ID", ""),
                law=kons.get("Abkuerzung") or br.get("Kurztitel", ""),
                law_title=br.get("Kurztitel", ""),
                doc_type=kons.get("Dokumenttyp", ""),
                section=(kons.get("ArtikelParagraphAnlage") or "").strip(),
                in_force=kons.get("Inkrafttretensdatum", ""),
                url=br.get("Eli") or meta.get("Allgemein", {}).get("DokumentUrl", ""),
                xml_url=xml_url,
                keywords=kons.get("Schlagworte", "") or "",
                index=[str(x) for x in _as_list((kons.get("Indizes") or {}).get("item"))],
            )
        )
    return total, refs


class RisClient:
    """Polite, caching HTTP client. ``delay`` seconds pass between network requests."""

    def __init__(self, cache_dir: Path | None = None, delay: float = 1.0, timeout: float = 30.0):
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.delay = delay
        self.timeout = timeout
        self._last = 0.0
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get(self, url: str) -> bytes:
        cache_file = None
        if self.cache_dir:
            cache_file = self.cache_dir / hashlib.sha1(url.encode()).hexdigest()
            if cache_file.exists():
                return cache_file.read_bytes()
        wait = self.delay - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            body = resp.read()
        self._last = time.monotonic()
        if cache_file:
            cache_file.write_bytes(body)
        return body

    def law_documents(self, gesetzesnummer: str, date: str) -> list[DocRef]:
        """All documents of one law in the version in force on ``date`` (YYYY-MM-DD)."""
        refs: list[DocRef] = []
        seen: set[str] = set()
        page = 1
        while True:
            total, page_refs = parse_search(json.loads(self.get(search_url(gesetzesnummer, date, page))))
            new = [r for r in page_refs if r.doc_id not in seen]
            if not new:
                break
            refs.extend(new)
            seen.update(r.doc_id for r in new)
            if len(refs) >= total:
                break
            page += 1
        return refs
