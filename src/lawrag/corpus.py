"""Build the paragraph corpus: one record per § in force on a given date.

Usage (needs internet access to RIS, e.g. in Colab):
    python -m lawrag.corpus --out data/corpus.jsonl --cache data/raw
    python -m lawrag.corpus --laws MRG UrlG --date 2026-11-02 --out data/corpus.jsonl
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

from .parse import carry_headings, parse_paragraph_xml
from .ris import LAWS, DocRef, RisClient


def section_key(law: str, section: str) -> str:
    """Stable identifier used as the gold label in evaluation, e.g. ``MRG § 1``."""
    return f"{law} {' '.join(section.split())}"


def build_record(
    ref: DocRef, xml: bytes, path: dict[int, str], topic: str = "", law: str | None = None
) -> tuple[dict, dict[int, str]]:
    """``law`` overrides the abbreviation from RIS, which some laws (e.g. the Urlaubsgesetz) lack."""
    parsed = parse_paragraph_xml(xml)
    path = carry_headings(path, parsed.headings)
    law = law or ref.law
    record = {
        "id": section_key(law, ref.section),
        "law": law,
        "law_title": ref.law_title,
        "topic": topic,
        "section": ref.section,
        "title": parsed.title,
        "path": [path[k] for k in sorted(path)],
        "text": "\n".join(parsed.blocks),
        "keywords": ref.keywords,
        "in_force": ref.in_force,
        "url": ref.url,
        "doc_id": ref.doc_id,
    }
    return record, path


def build_law(client: RisClient, law: str, date: str) -> list[dict]:
    gesnr, _, topic = LAWS[law]
    records: list[dict] = []
    path: dict[int, str] = {}
    for ref in client.law_documents(gesnr, date):
        if ref.doc_type != "Paragraph" or not ref.xml_url:
            continue  # skip the law's title record ("Norm") and anything without text
        record, path = build_record(ref, client.get(ref.xml_url), path, topic, law=law)
        if record["text"]:
            records.append(record)
    return records


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--laws", nargs="+", default=list(LAWS), choices=list(LAWS))
    ap.add_argument("--date", default=dt.date.today().isoformat(), help="version in force on this date")
    ap.add_argument("--out", type=Path, default=Path("data/corpus.jsonl"))
    ap.add_argument("--cache", type=Path, default=Path("data/raw"))
    ap.add_argument("--delay", type=float, default=1.0, help="seconds between requests")
    args = ap.parse_args()

    client = RisClient(cache_dir=args.cache, delay=args.delay)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with args.out.open("w", encoding="utf-8") as f:
        for law in args.laws:
            records = build_law(client, law, args.date)
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            total += len(records)
            print(f"{law:7s} {len(records):4d} paragraphs")
    print(f"total   {total:4d} paragraphs, version of {args.date} -> {args.out}")


if __name__ == "__main__":
    main()
