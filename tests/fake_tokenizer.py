"""Fake tokenizers for CPU-only tests.

Both implement exactly the surface the pipeline uses:
__call__ → {"input_ids": [...]}, decode(ids, skip_special_tokens=...),
all_special_ids.
"""

from __future__ import annotations

import re


class FakeCharTokenizer:
    """One token per character (id = codepoint).

    Deterministic and trivially invertible, so alignment and length-matching
    logic can be tested without HF.
    """

    all_special_ids: list[int] = []

    def __call__(self, text: str, add_special_tokens: bool = True, **kwargs):
        return {"input_ids": [ord(c) for c in text]}

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        return "".join(chr(i) for i in ids)


class FakeCodeTokenizer:
    """A byte-BPE-*shaped* tokenizer: leading whitespace joins the next piece.

    `FakeCharTokenizer` cannot exercise the E11 generator at all, because
    nothing there is a single token — and "the answer must be a single token"
    is one of the constraints under test. This splitter reproduces the one
    property that matters for those constraints: `" 3"`, `" x"` and `" =="`
    are each ONE token, exactly as under deepseek-coder's tokenizer, so
    single-token checks, one-token mutations and answer alignment behave the
    way they do on the real thing.

    Ids are assigned on first sight and cached, so `decode` round-trips.
    """

    _PATTERN = re.compile(r"\s*\w+|\s*[^\s\w]|\s+")

    def __init__(self):
        self.all_special_ids: list[int] = []
        self._vocab: dict[str, int] = {}
        self._inverse: dict[int, str] = {}

    def _id_for(self, piece: str) -> int:
        if piece not in self._vocab:
            token_id = len(self._vocab) + 1000
            self._vocab[piece] = token_id
            self._inverse[token_id] = piece
        return self._vocab[piece]

    def tokenize(self, text: str) -> list[str]:
        return self._PATTERN.findall(text)

    def __call__(self, text: str, add_special_tokens: bool = True, **kwargs):
        pieces = self.tokenize(text)
        if "".join(pieces) != text:              # the split must be lossless
            raise AssertionError(f"tokenizer lost characters on {text!r}")
        return {"input_ids": [self._id_for(p) for p in pieces]}

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        return "".join(self._inverse[int(i)] for i in ids)


class FakeDigitTokenizer(FakeCodeTokenizer):
    """A deepseek-coder-*shaped* tokenizer: digit-level numbers, ` == ` merged.

    The real tokenizer differs from `FakeCodeTokenizer` in two ways that both
    change what E11 has to do, and both were found by running it rather than by
    reading it:

      * numbers are tokenized **one digit at a time**, so `15` has no single
        unembedding row and `' 3'` is a space token followed by a digit — the
        atomic form is the bare `'3'`;
      * an operator absorbs the spaces around it, so `x = 3` is `['x', ' = ',
        '3']` and the answer after `assert f() ==` is emitted *bare*, meaning
        the prompt must end with the trailing space.

    Keeping a fake with these properties means the CPU tests cover the branch
    the real run actually takes.
    """

    _PATTERN = re.compile(r"\s*[=+\-*%<>!]+ |\d|\s*[A-Za-z_]\w*|\s*[^\s\w]|\s+")
