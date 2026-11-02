"""Split § records into retrieval chunks.

Most §§ are short and become a single chunk. Long ones are split at block boundaries
(Absätze and list items), never mid-sentence unless one block alone is too long. Every
chunk starts with a header naming the law, §, title and section path, so a chunk taken
out of context still says where it comes from; this helps both retrieval and the answer
step, which has to cite the §.
"""

from __future__ import annotations

import re

_SENTENCE = re.compile(r"(?<=[.;:])\s+(?=[(A-ZÄÖÜ0-9§])")


def header(record: dict) -> str:
    parts = [f"{record['law']} {record['section']}"]
    if record.get("title"):
        parts.append(record["title"])
    head = " – ".join(parts)
    if record.get("path"):
        head += f" ({record['law_title']}: {' › '.join(record['path'])})"
    else:
        head += f" ({record['law_title']})"
    return head


def _split_long(block: str, max_chars: int) -> list[str]:
    pieces, current = [], ""
    for sentence in _SENTENCE.split(block):
        if current and len(current) + 1 + len(sentence) > max_chars:
            pieces.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        pieces.append(current)
    return pieces


def chunk_record(record: dict, max_chars: int = 1200) -> list[dict]:
    """Return chunks ``{"chunk_id", "section_id", "text"}`` covering the whole § text."""
    blocks: list[str] = []
    for block in record["text"].split("\n"):
        blocks.extend(_split_long(block, max_chars) if len(block) > max_chars else [block])

    groups: list[list[str]] = [[]]
    size = 0
    for block in blocks:
        if groups[-1] and size + len(block) > max_chars:
            groups.append([])
            size = 0
        groups[-1].append(block)
        size += len(block) + 1

    head = header(record)
    return [
        {"chunk_id": f"{record['id']}#{i}", "section_id": record["id"], "text": head + "\n" + "\n".join(g)}
        for i, g in enumerate(groups)
        if g
    ]


def chunk_corpus(records: list[dict], max_chars: int = 1200) -> list[dict]:
    return [c for r in records for c in chunk_record(r, max_chars)]
