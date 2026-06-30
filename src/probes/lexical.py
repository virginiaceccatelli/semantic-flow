"""Lexical and binding probes.

Phase 1 tasks:
  - LexicalProbe: classify token type (keyword, identifier, literal, operator, ...)
  - BindingProbe: given two token hidden states, predict whether they refer to the
    same variable declaration (binary pairwise classification).

These establish the baseline: can the model distinguish names from meaning?
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .base import LinearProbe, ProbeConfig, ProbeResult, cross_validate_probe


# Token type taxonomy
TOKEN_TYPES = [
    "keyword",
    "identifier",
    "string_literal",
    "numeric_literal",
    "operator",
    "delimiter",
    "comment",
    "whitespace",
    "unknown",
]
TOKEN_TYPE_TO_IDX = {t: i for i, t in enumerate(TOKEN_TYPES)}


@dataclass
class LexicalExample:
    """Single (hidden_state, token_type_label) example."""
    hidden: np.ndarray      # shape (d_model,)
    token_str: str
    token_type: str         # one of TOKEN_TYPES
    layer: int
    position: int


@dataclass
class BindingExample:
    """Pairwise example: do token A and token B refer to the same variable?"""
    hidden_a: np.ndarray    # shape (d_model,)
    hidden_b: np.ndarray
    token_str_a: str
    token_str_b: str
    same_binding: bool      # ground-truth label
    layer: int
    pos_a: int
    pos_b: int


class LexicalProbe:
    """Multiclass probe for token-type classification."""

    def __init__(self, config: Optional[ProbeConfig] = None):
        self.config = config or ProbeConfig()
        self._probe = LinearProbe(config=self.config)

    def build_dataset(
        self, examples: list[LexicalExample], layer: int
    ) -> tuple[np.ndarray, np.ndarray]:
        filtered = [e for e in examples if e.layer == layer]
        X = np.stack([e.hidden for e in filtered])
        y = np.array([TOKEN_TYPE_TO_IDX.get(e.token_type, TOKEN_TYPE_TO_IDX["unknown"])
                      for e in filtered])
        return X, y

    def run(
        self, examples: list[LexicalExample], layer: int
    ) -> ProbeResult:
        X, y = self.build_dataset(examples, layer)
        return cross_validate_probe(
            LinearProbe, X, y, layer=layer, task="lexical_token_type", config=self.config
        )


class BindingProbe:
    """Binary pairwise probe: same variable binding or not?

    Input: concatenation of the two token hidden states [h_a; h_b; h_a - h_b].
    This difference vector captures relational information.

    Critical experimental variant: test with identically-named tokens that
    refer to *different* declarations (shadowing, renamed copies) to check
    whether the probe is tracking surface identity or true binding.
    """

    def __init__(self, config: Optional[ProbeConfig] = None):
        self.config = config or ProbeConfig()

    def _features(self, examples: list[BindingExample], layer: int) -> tuple[np.ndarray, np.ndarray]:
        filtered = [e for e in examples if e.layer == layer]
        feats = []
        labels = []
        for ex in filtered:
            diff = ex.hidden_a - ex.hidden_b
            feat = np.concatenate([ex.hidden_a, ex.hidden_b, diff])
            feats.append(feat)
            labels.append(int(ex.same_binding))
        return np.stack(feats), np.array(labels)

    def run(
        self, examples: list[BindingExample], layer: int
    ) -> ProbeResult:
        X, y = self._features(examples, layer)
        return cross_validate_probe(
            LinearProbe, X, y, layer=layer, task="variable_binding", config=self.config
        )

    def run_lexical_decoy_split(
        self,
        examples: list[BindingExample],
        layer: int,
    ) -> dict[str, ProbeResult]:
        """Split examples into same-name and different-name pairs,
        and run probes separately to diagnose lexical vs semantic tracking."""
        same_name = [e for e in examples if e.token_str_a == e.token_str_b]
        diff_name = [e for e in examples if e.token_str_a != e.token_str_b]
        results = {}
        for split_name, split in [("same_surface_name", same_name),
                                   ("different_surface_name", diff_name)]:
            if len(split) < 10:
                continue
            X, y = self._features(split, layer)
            if len(np.unique(y)) < 2:
                continue
            results[split_name] = cross_validate_probe(
                LinearProbe, X, y, layer=layer,
                task=f"binding_{split_name}", config=self.config
            )
        return results
