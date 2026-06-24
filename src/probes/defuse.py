"""Def-use edge probes.

Phase 2 tasks:
  - DefUseEdgeProbe: binary pairwise classification — is there a def-use edge
    between token positions i and j?
  - NodeRoleProbe: classify token role as source/sink/sanitizer/def-site/use-site.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .base import LinearProbe, ProbeConfig, ProbeResult, cross_validate_probe


NODE_ROLES = ["def_site", "use_site", "source", "sink", "sanitizer", "other"]
NODE_ROLE_TO_IDX = {r: i for i, r in enumerate(NODE_ROLES)}


@dataclass
class DefUseExample:
    """Pairwise example: is there a def-use edge from position i to position j?"""
    hidden_i: np.ndarray    # hidden state at definition position
    hidden_j: np.ndarray    # hidden state at use position
    has_edge: bool          # ground-truth label
    layer: int
    pos_i: int
    pos_j: int
    distance: int           # |pos_j - pos_i| — for distance-stratified analysis
    same_function: bool = True
    name: str = ""


@dataclass
class NodeRoleExample:
    """Single token with a semantic role label."""
    hidden: np.ndarray
    role: str               # one of NODE_ROLES
    layer: int
    position: int
    name: str = ""


class DefUseEdgeProbe:
    """Binary pairwise probe for def-use edges.

    Feature: [h_i; h_j; h_i - h_j; |h_i - h_j|]

    Negative examples: randomly sampled non-edge pairs from the same function
    (hard negatives) plus cross-function pairs (easy negatives).
    The ratio of positives to negatives should be logged.
    """

    def __init__(self, config: Optional[ProbeConfig] = None):
        self.config = config or ProbeConfig()

    def _features(
        self, examples: list[DefUseExample], layer: int
    ) -> tuple[np.ndarray, np.ndarray]:
        filtered = [e for e in examples if e.layer == layer]
        feats, labels = [], []
        for ex in filtered:
            diff = ex.hidden_i - ex.hidden_j
            feat = np.concatenate([ex.hidden_i, ex.hidden_j, diff, np.abs(diff)])
            feats.append(feat)
            labels.append(int(ex.has_edge))
        return np.stack(feats), np.array(labels)

    def run(
        self, examples: list[DefUseExample], layer: int
    ) -> ProbeResult:
        X, y = self._features(examples, layer)
        return cross_validate_probe(
            LinearProbe, X, y, layer=layer, task="defuse_edge", config=self.config
        )

    def run_by_distance(
        self,
        examples: list[DefUseExample],
        layer: int,
        buckets: list[tuple[int, int]] = [(0, 10), (10, 50), (50, 200), (200, 10000)],
    ) -> dict[str, ProbeResult]:
        """Run separate probes for each distance bucket to track degradation."""
        results = {}
        for lo, hi in buckets:
            bucket = [e for e in examples if lo <= e.distance < hi]
            if len(bucket) < 20:
                continue
            X, y = self._features(bucket, layer)
            label = f"defuse_edge_dist_{lo}_{hi}"
            results[label] = cross_validate_probe(
                LinearProbe, X, y, layer=layer, task=label, config=self.config
            )
        return results


class NodeRoleProbe:
    """Multiclass probe for semantic node role classification."""

    def __init__(self, config: Optional[ProbeConfig] = None):
        self.config = config or ProbeConfig()

    def _features(
        self, examples: list[NodeRoleExample], layer: int
    ) -> tuple[np.ndarray, np.ndarray]:
        filtered = [e for e in examples if e.layer == layer]
        X = np.stack([e.hidden for e in filtered])
        y = np.array([NODE_ROLE_TO_IDX.get(e.role, NODE_ROLE_TO_IDX["other"])
                      for e in filtered])
        return X, y

    def run(
        self, examples: list[NodeRoleExample], layer: int
    ) -> ProbeResult:
        X, y = self._features(examples, layer)
        return cross_validate_probe(
            LinearProbe, X, y, layer=layer, task="node_role", config=self.config
        )
