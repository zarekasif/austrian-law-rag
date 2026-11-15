"""Text analysis for lexical retrieval: German stemming and character n-grams.

``german_stem`` implements the Snowball German stemming algorithm
(https://snowballstem.org/algorithms/german/stemmer.html) in plain Python, so the lexical
baseline has no dependencies. Stemming maps inflections together ("Pausen" and "Pause" both
become "paus"), but not compounds: "Ruhepause" stays "ruhepaus". Character n-grams of the stems
handle that, because "paus" is also a 4-gram inside "ruhepaus".
"""

from __future__ import annotations

import re

VOWELS = set("aeiouyäöü")
S_ENDING = set("bdfghklmnrt")
ST_ENDING = S_ENDING - {"r"}

_TOKEN = re.compile(r"\w+", re.UNICODE)

GERMAN_STOPWORDS = frozenset(
    """
    aber als also am an auch auf aus bei bin bis bist da damit dann das dass dem den denn der des die dies diese
    dieser dieses doch dort du durch ein eine einem einen einer eines er es etwa für gegen hat hatte hier ich ihr
    im in ist ja jedoch kann kein keine man mit muss nach nicht noch nur ob oder ohne sein sich sie sind so soll
    über um und uns unter vom von vor wann war was weil welche welcher wenn wer werden wie wir wird wo zu zum zur
    """.split()
)


def _regions(w: str) -> tuple[int, int]:
    """Start of R1 and R2 (Snowball definition), R1 adjusted to leave at least 3 letters before it."""

    def after_vc(start: int) -> int:
        for i in range(start + 1, len(w)):
            if w[i] not in VOWELS and w[i - 1] in VOWELS:
                return i + 1
        return len(w)

    r1 = after_vc(0)
    r2 = after_vc(r1) if r1 < len(w) else len(w)
    return max(r1, 3), r2


def german_stem(word: str) -> str:
    w = word.lower().replace("ß", "ss")
    # Prelude: u and y between vowels are consonants for this algorithm.
    chars = list(w)
    for i in range(1, len(chars) - 1):
        if chars[i] in "uy" and chars[i - 1] in VOWELS and chars[i + 1] in VOWELS:
            chars[i] = chars[i].upper()
    w = "".join(chars)
    r1, r2 = _regions(w)

    # Step 1
    for suffix in ("ern", "em", "er", "en", "es", "e", "s"):
        if w.endswith(suffix):
            start = len(w) - len(suffix)
            if start >= r1:
                if suffix in ("ern", "em", "er"):
                    w = w[:start]
                elif suffix in ("en", "es", "e"):
                    w = w[:start]
                    if w.endswith("niss"):
                        w = w[:-1]
                elif start > 0 and w[start - 1] in S_ENDING:
                    w = w[:start]
            break

    # Step 2
    for suffix in ("est", "en", "er", "st"):
        if w.endswith(suffix):
            start = len(w) - len(suffix)
            if start >= r1:
                if suffix != "st":
                    w = w[:start]
                elif start >= 4 and w[start - 1] in ST_ENDING:
                    w = w[:start]
            break

    # Step 3 (d-suffixes)
    for suffix in ("isch", "lich", "heit", "keit", "end", "ung", "ig", "ik"):
        if not w.endswith(suffix):
            continue
        start = len(w) - len(suffix)
        if start < r2:
            break
        if suffix in ("end", "ung"):
            w = w[:start]
            if w.endswith("ig") and len(w) - 2 >= r2 and not w[:-2].endswith("e"):
                w = w[:-2]
        elif suffix in ("ig", "ik", "isch"):
            if not w[:start].endswith("e"):
                w = w[:start]
        elif suffix in ("lich", "heit"):
            w = w[:start]
            for pre in ("er", "en"):
                if w.endswith(pre) and len(w) - 2 >= r1:
                    w = w[:-2]
                    break
        else:  # keit
            w = w[:start]
            for pre in ("lich", "ig"):
                if w.endswith(pre) and len(w) - len(pre) >= r2:
                    w = w[: -len(pre)]
                    break
        break

    # Postlude
    return w.replace("U", "u").replace("Y", "y").replace("ä", "a").replace("ö", "o").replace("ü", "u")


def char_ngrams(token: str, n: int = 4) -> list[str]:
    return [token] if len(token) <= n else [token[i : i + n] for i in range(len(token) - n + 1)]


def analyze(text: str, mode: str = "word", stopwords: frozenset[str] = GERMAN_STOPWORDS) -> list[str]:
    """Turn text into index terms.

    - ``word``: casefolded words without stopwords (no normalisation of inflection)
    - ``stem``: Snowball German stems of those words
    - ``ngram``: character 4-grams of the stems, so parts of compounds match
    """
    words = [t for t in _TOKEN.findall(text.casefold()) if t not in stopwords]
    if mode == "word":
        return words
    stems = [german_stem(t) for t in words]
    if mode == "stem":
        return stems
    if mode == "ngram":
        return [g for s in stems for g in char_ngrams(s)]
    raise ValueError(f"unknown analyzer mode: {mode}")
