"""Extract the legal text and headings from a RIS consolidated-law XML document.

Only elements marked ``ct="text"`` carry the norm itself; everything else is metadata,
page headers or footers. Structural headings (``ueberschrift typ="g1"``, ``"g2"``, ...)
appear only in the first § of a section, so callers carry them forward across §§.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

NS = "{http://www.bka.gv.at}"
_WS = re.compile(r"[ \t ]+")


@dataclass
class ParsedParagraph:
    title: str = ""
    headings: dict[int, str] = field(default_factory=dict)  # level -> heading in this document
    blocks: list[str] = field(default_factory=list)
    meta: dict[str, str] = field(default_factory=dict)


def _local(tag: str) -> str:
    return tag[len(NS):] if tag.startswith(NS) else tag


def _text(el: ET.Element) -> str:
    """Element text with list symbols separated from the item ("1." + space) and tabs dropped."""
    parts = [el.text or ""]
    for child in el:
        name = _local(child.tag)
        if name == "symbol":
            parts.append((child.text or "").strip() + " ")
        elif name != "tab":
            parts.append(_text(child))
        parts.append(child.tail or "")
    return _WS.sub(" ", "".join(parts)).strip()


def _list_depth(el: ET.Element, parents: dict) -> int:
    node = parents.get(el)
    while node is not None:
        if _local(node.tag) == "aufzaehlung":
            return max(int(node.get("ebene", "1")), 1)
        node = parents.get(node)
    return 0


def parse_paragraph_xml(xml: bytes | str) -> ParsedParagraph:
    root = ET.fromstring(xml)
    parents = {child: parent for parent in root.iter() for child in parent}
    out = ParsedParagraph()

    for el in root.iter():
        ct = el.get("ct")
        if ct is None:
            continue
        name = _local(el.tag)
        if ct != "text":
            if name == "absatz" and ct not in out.meta:
                out.meta[ct] = _text(el)
            continue
        text = _text(el)
        if not text:
            continue
        if name == "ueberschrift":
            typ = el.get("typ", "")
            if typ == "para":
                out.title = text
            elif typ.startswith("g") and typ[1:].isdigit():
                out.headings[int(typ[1:])] = text
            else:
                out.blocks.append(text)
        elif name == "listelem":
            out.blocks.append("  " * (_list_depth(el, parents) - 1) + text)
        else:
            out.blocks.append(text)
    return out


def carry_headings(current: dict[int, str], new: dict[int, str]) -> dict[int, str]:
    """Update the running section path: a new level-n heading replaces n and drops deeper levels."""
    if not new:
        return dict(current)
    top = min(new)
    carried = {lvl: h for lvl, h in current.items() if lvl < top}
    carried.update(new)
    return carried
