"""Answer a question from retrieved §§, citing the § each statement comes from.

Pipeline: retrieve the top chunks (default E5 → cross-encoder rerank), put them in a numbered
context, and ask an LLM for a short German answer that cites §§ as ``[AZG § 11]``. If the
context does not contain the answer, the model must say so instead of guessing.

Two generator backends:
- ``HFGenerator``: an open model run locally (default Qwen2.5-7B-Instruct in 4-bit, fits a free
  Colab T4). No API key, no cost.
- ``AnthropicGenerator``: a Claude model through the Anthropic API (needs ``ANTHROPIC_API_KEY``).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

from .ris import LAWS

NO_ANSWER = "Dazu finde ich in den vorliegenden Bestimmungen keine Antwort."

SYSTEM_PROMPT = f"""Du beantwortest Fragen zum österreichischen Recht ausschließlich anhand der unten \
angegebenen Gesetzesstellen.

Regeln:
- Antworte auf Deutsch, in einfachen Worten, in höchstens vier Sätzen.
- Belege jede Aussage mit der Fundstelle in eckigen Klammern, genau in der Form [AZG § 11].
- Nenne nur Paragraphen, die in den angegebenen Gesetzesstellen vorkommen.
- Wenn die Gesetzesstellen die Frage nicht beantworten, antworte genau: "{NO_ANSWER}"
- Gib keine Rechtsberatung über den Wortlaut hinaus und erfinde nichts dazu."""

_LAW_NAMES = "|".join(sorted(LAWS, key=len, reverse=True))
_SECTION = r"(\d+[a-z]?)"
# [AZG § 11] / AZG § 11  and the Austrian order § 11 (Abs. 1 Z 2) AZG
_CITE_LAW_FIRST = re.compile(rf"\b({_LAW_NAMES})\s*§+\s*{_SECTION}\b")
_CITE_SECTION_FIRST = re.compile(
    rf"§+\s*{_SECTION}(?:\s+Abs\.?\s*\d+[a-z]?)?(?:\s+(?:Z|Ziffer)\s*\d+[a-z]?)?(?:\s+lit\.?\s*[a-z]\))?\s+({_LAW_NAMES})\b"
)


def parse_citations(text: str) -> list[str]:
    """§ ids cited in an answer, in order of first appearance, e.g. ``["AZG § 11"]``."""
    found: list[tuple[int, str]] = []
    for m in _CITE_LAW_FIRST.finditer(text):
        found.append((m.start(), f"{m.group(1)} § {m.group(2)}"))
    for m in _CITE_SECTION_FIRST.finditer(text):
        found.append((m.start(), f"{m.group(2)} § {m.group(1)}"))
    seen, out = set(), []
    for _, cid in sorted(found):
        if cid not in seen:
            seen.add(cid)
            out.append(cid)
    return out


def is_abstention(text: str) -> bool:
    return "keine antwort" in text.casefold()


def build_context(chunks: list[dict], chunk_ids: list[int], max_chars: int = 6000) -> tuple[str, list[str]]:
    """Numbered context from retrieved chunks; returns the text and the § ids it contains."""
    parts, sections, used = [], [], 0
    for n, i in enumerate(chunk_ids, start=1):
        text = chunks[i]["text"]
        if parts and used + len(text) > max_chars:
            break
        parts.append(f"[{n}] {text}")
        used += len(text)
        if chunks[i]["section_id"] not in sections:
            sections.append(chunks[i]["section_id"])
    return "\n\n".join(parts), sections


def user_prompt(question: str, context: str) -> str:
    return f"Gesetzesstellen:\n\n{context}\n\nFrage: {question}\n\nAntwort:"


@dataclass
class Answer:
    question: str
    answer: str
    citations: list[str]
    context_sections: list[str]
    abstained: bool
    extra: dict = field(default_factory=dict)


class AnswerPipeline:
    def __init__(self, retriever, chunks: list[dict], generator, top_chunks: int = 6, max_context_chars: int = 6000):
        self.retriever = retriever
        self.chunks = chunks
        self.generator = generator
        self.top_chunks = top_chunks
        self.max_context_chars = max_context_chars

    def __call__(self, question: str) -> Answer:
        chunk_ids = self.retriever.rank_chunks(question, self.top_chunks)
        context, sections = build_context(self.chunks, chunk_ids, self.max_context_chars)
        text = self.generator(SYSTEM_PROMPT, user_prompt(question, context)).strip()
        return Answer(question, text, parse_citations(text), sections, is_abstention(text))


class HFGenerator:
    """Local open model through transformers; 4-bit quantisation so a 7B model fits a T4."""

    def __init__(self, model_id: str = "Qwen/Qwen2.5-7B-Instruct", load_in_4bit: bool = True, max_new_tokens: int = 300):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.torch = torch
        self.name = model_id.split("/")[-1]
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        kwargs = {"device_map": "auto"}
        if load_in_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4"
            )
        else:
            kwargs["torch_dtype"] = torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        self.max_new_tokens = max_new_tokens

    def __call__(self, system: str, user: str) -> str:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)
        with self.torch.no_grad():
            out = self.model.generate(inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        return self.tokenizer.decode(out[0, inputs.shape[1]:], skip_special_tokens=True)


class AnthropicGenerator:
    """Claude through the Anthropic API. Reads the key from ANTHROPIC_API_KEY."""

    def __init__(self, model: str = "claude-haiku-4-5", max_tokens: int = 400, api_key: str | None = None):
        import anthropic

        self.client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self.model = model
        self.name = model
        self.max_tokens = max_tokens

    def __call__(self, system: str, user: str) -> str:
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(block.text for block in msg.content if getattr(block, "type", "") == "text")
