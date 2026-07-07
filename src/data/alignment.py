"""AST-span → token-index alignment.

Probe examples must be built from the hidden state of the *actual* def/use
occurrence, not from the first token whose string happens to match the
variable name. This module maps an AST event's (line, col) span to the
token indices that cover it, using the tokenizer's offset mapping.

The offset mapping is computed once per example (at extraction time) and
saved alongside the activations, so probe training never re-tokenizes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence


def line_col_to_char(source: str, line: int, col: int) -> int:
    """Convert a 1-based line / 0-based column position to a character offset."""
    lines = source.splitlines(keepends=True)
    if line < 1 or line > len(lines):
        raise ValueError(f"line {line} out of range for source with {len(lines)} lines")
    return sum(len(l) for l in lines[: line - 1]) + col


def compute_offsets(
    source: str,
    tokenizer,
    input_ids: Optional[Sequence[int]] = None,
) -> list[tuple[int, int]]:
    """Character (start, end) offsets for each token, computed by incremental
    prefix decoding.

    `return_offsets_mapping` is NOT used: converted fast tokenizers (e.g.
    deepseek-coder's) return overlapping/shifted spans that misalign every
    downstream label. Decoding prefixes ids[:i] with skip_special_tokens=True
    is tokenizer-agnostic and exact — the concatenation is verified against
    the source. Special tokens get empty (start, start) spans and are never
    matched by char_span_to_tokens.

    Pass `input_ids` to reuse an existing encoding (must come from this
    tokenizer and source, possibly truncated).
    """
    if input_ids is None:
        input_ids = tokenizer(source)["input_ids"]
    ids = list(input_ids)

    # Fast path: the tokenizer's own offset mapping, accepted ONLY if its
    # non-special spans tile the source exactly (broken tokenizers produce
    # overlapping/shifted spans — those fall through to the exact slow path).
    fast = _validated_fast_offsets(source, tokenizer, ids)
    if fast is not None:
        return fast

    offsets: list[tuple[int, int]] = []
    prev_len = 0
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    for i in range(1, len(ids) + 1):
        if ids[i - 1] in special_ids:
            offsets.append((prev_len, prev_len))
            continue
        cur = tokenizer.decode(ids[:i], skip_special_tokens=True)
        offsets.append((prev_len, len(cur)))
        prev_len = len(cur)

    decoded = tokenizer.decode(ids, skip_special_tokens=True)
    if decoded != source[: len(decoded)]:
        raise ValueError(
            "Tokenizer round-trip does not reproduce the source; "
            "offset alignment would be wrong for this tokenizer."
        )
    return offsets


def _validated_fast_offsets(
    source: str,
    tokenizer,
    ids: list[int],
) -> Optional[list[tuple[int, int]]]:
    """Offsets via return_offsets_mapping, or None if unsupported/invalid."""
    try:
        enc = tokenizer(source, return_offsets_mapping=True, truncation=True,
                        max_length=len(ids))
        fast_ids = enc["input_ids"]
        offsets = [tuple(o) for o in enc["offset_mapping"]]
    except (TypeError, ValueError, KeyError, NotImplementedError):
        return None
    if list(fast_ids) != ids or len(offsets) != len(ids):
        return None
    special = set(getattr(tokenizer, "all_special_ids", []) or [])
    recon = "".join(source[a:b] for (a, b), i in zip(offsets, ids) if i not in special)
    decoded = tokenizer.decode(ids, skip_special_tokens=True)
    if recon != decoded or decoded != source[: len(decoded)]:
        return None
    # normalize special-token spans to empty so they are never matched
    return [
        ((a, a) if i in special else (a, b))
        for (a, b), i in zip(offsets, ids)
    ]


def char_span_to_tokens(
    offsets: Sequence[tuple[int, int]],
    start: int,
    end: int,
) -> list[int]:
    """Token indices whose spans overlap the character range [start, end)."""
    hits = []
    for i, (tok_start, tok_end) in enumerate(offsets):
        if tok_start == tok_end:  # special token / empty span
            continue
        if tok_start < end and tok_end > start:
            hits.append(i)
    return hits


@dataclass
class AlignedEvent:
    """A source-level event (def/use/guard/...) resolved to token indices."""

    name: str
    kind: str                # "def" | "use" | "guard" | "sink_arg" | ...
    line: int
    col: int
    token_indices: list[int]

    @property
    def anchor(self) -> int:
        """The token index used as the probing position (last covering token).

        For decoder-only models the last token of a span is the first position
        whose hidden state can integrate the whole span.
        """
        return self.token_indices[-1]


class TokenAligner:
    """Aligns AST events to token positions for one source string."""

    def __init__(self, source: str, offsets: Sequence[tuple[int, int]]):
        self.source = source
        self.offsets = list(offsets)

    @classmethod
    def from_tokenizer(cls, source: str, tokenizer) -> "TokenAligner":
        return cls(source, compute_offsets(source, tokenizer))

    def align(
        self,
        name: str,
        kind: str,
        line: int,
        col: int,
        end_line: Optional[int] = None,
        end_col: Optional[int] = None,
    ) -> Optional[AlignedEvent]:
        """Resolve an AST span to tokens; returns None if the span is not covered
        (e.g. truncated away)."""
        start = line_col_to_char(self.source, line, col)
        if end_line is not None and end_col is not None:
            end = line_col_to_char(self.source, end_line, end_col)
        else:
            end = start + len(name)
        toks = char_span_to_tokens(self.offsets, start, end)
        if not toks:
            return None
        return AlignedEvent(name=name, kind=kind, line=line, col=col, token_indices=toks)

    def align_var_event(self, event) -> Optional[AlignedEvent]:
        """Align a graphs.dfg_extractor.VarEvent."""
        return self.align(
            name=event.name,
            kind=event.kind,
            line=event.line,
            col=event.col,
            end_line=event.end_line,
            end_col=event.end_col,
        )
