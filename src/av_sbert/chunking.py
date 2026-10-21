"""Splitting long documents into encoder-sized chunks and pairing the chunks of two documents.

all-MiniLM-L12-v2 reads at most K = 256 word pieces, while PAN texts (essays, interview transcripts) are
often much longer. Each document is cut into chunks of at most K tokens and the chunks of the two
documents are paired position by position: (A1, B1), (A2, B2), ... When one document has fewer chunks,
its chunks are reused cyclically, so a pair yields ``M = max(len(A), len(B))`` chunk pairs and no text is
dropped.

Two chunkers are provided:

* ``hard``: fixed windows of K tokens. Used for training and for the two main submissions.
* ``soft``: whole sentences are packed into a chunk until the next one would overflow it, so a sentence
  is only cut when it alone is longer than K. Used for the third submission (golden-ottoman).
"""
from __future__ import annotations

import re
from typing import Callable, List, Sequence, Tuple

CountFn = Callable[[str], int]

_SENT = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9<])")


def split_sentences(text: str) -> List[str]:
    """Sentence split with NLTK's punkt when installed, otherwise a punctuation-based fallback."""
    try:
        import nltk

        return [s for s in nltk.sent_tokenize(text) if s.strip()]
    except (ImportError, LookupError):
        return [s for s in _SENT.split(text) if s.strip()]


def hard_chunks(text: str, tokenize: Callable[[str], List], detokenize: Callable[[List], str],
                chunk_len: int) -> List[str]:
    tokens = tokenize(text)
    if len(tokens) <= chunk_len:
        return [text]
    return [detokenize(tokens[i:i + chunk_len]) for i in range(0, len(tokens), chunk_len)]


def _split_long_sentence(sentence: str, count: CountFn, chunk_len: int) -> List[str]:
    """Greedy word packing for a sentence longer than one chunk (an over-long single word stays alone)."""
    pieces, current, size = [], [], 0
    for word in sentence.split():
        n = count(word)
        if current and size + n > chunk_len:
            pieces.append(" ".join(current))
            current, size = [], 0
        current.append(word)
        size += n
    if current:
        pieces.append(" ".join(current))
    return pieces


def soft_chunks(text: str, count: CountFn, chunk_len: int) -> List[str]:
    units: List[Tuple[str, int]] = []
    for sentence in split_sentences(text):
        n = count(sentence)
        if n > chunk_len:
            units += [(p, count(p)) for p in _split_long_sentence(sentence, count, chunk_len)]
        else:
            units.append((sentence, n))
    chunks, current, size = [], [], 0
    for unit, n in units:
        if current and size + n > chunk_len:
            chunks.append(" ".join(current))
            current, size = [], 0
        current.append(unit)
        size += n
    if current:
        chunks.append(" ".join(current))
    return chunks or [text]


def pair_chunks(chunks_a: Sequence[str], chunks_b: Sequence[str]) -> List[Tuple[str, str]]:
    """Positional pairing; the shorter list wraps around."""
    if not chunks_a or not chunks_b:
        return []
    m = max(len(chunks_a), len(chunks_b))
    return [(chunks_a[i % len(chunks_a)], chunks_b[i % len(chunks_b)]) for i in range(m)]


class Chunker:
    """Chunker bound to a Hugging Face tokenizer (the encoder's own tokenizer by default)."""

    def __init__(self, tokenizer, chunk_len: int = 256, mode: str = "hard"):
        if mode not in ("hard", "soft"):
            raise ValueError("mode must be 'hard' or 'soft'")
        self.tokenizer = tokenizer
        self.chunk_len = chunk_len
        self.mode = mode

    def count(self, text: str) -> int:
        return len(self.tokenizer.tokenize(text))

    def __call__(self, text: str) -> List[str]:
        if self.mode == "soft":
            return soft_chunks(text, self.count, self.chunk_len)
        return hard_chunks(text, lambda t: self.tokenizer.encode(t, add_special_tokens=False),
                           lambda ids: self.tokenizer.decode(ids), self.chunk_len)

    def pair(self, text_a: str, text_b: str) -> List[Tuple[str, str]]:
        return pair_chunks(self(text_a), self(text_b))
