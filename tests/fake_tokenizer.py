"""A character-level fake tokenizer for CPU-only tests.

Implements exactly the tokenizer surface the pipeline uses:
__call__ → {"input_ids": [...]}, decode(ids, skip_special_tokens=...),
all_special_ids. One token per character (id = codepoint), so alignment
and length-matching logic can be tested deterministically without HF.
"""

from __future__ import annotations


class FakeCharTokenizer:
    all_special_ids: list[int] = []

    def __call__(self, text: str, add_special_tokens: bool = True, **kwargs):
        return {"input_ids": [ord(c) for c in text]}

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        return "".join(chr(i) for i in ids)
