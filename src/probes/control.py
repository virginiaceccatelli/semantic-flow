"""Control-dependency and branch membership probes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .base import LinearProbe, ProbeConfig, ProbeResult, cross_validate_probe


@dataclass
class ControlDepExample:
    """Pairwise: is statement B control-dependent on branch/statement A?"""
    hidden_a: np.ndarray    # branch/guard statement
    hidden_b: np.ndarray    # body statement
    is_dependent: bool
    layer: int
    pos_a: int
    pos_b: int
    branch_type: str = "if"   # "if", "while", "for", "try"


@dataclass
class BranchMembershipExample:
    """Single token: which branch body does it belong to?"""
    hidden: np.ndarray
    branch_id: int          # integer id of the enclosing branch
    layer: int
    position: int
    branch_type: str = "if"


class ControlDepProbe:
    """Binary pairwise probe for control dependency."""

    def __init__(self, config: Optional[ProbeConfig] = None):
        self.config = config or ProbeConfig()

    def _features(
        self, examples: list[ControlDepExample], layer: int
    ) -> tuple[np.ndarray, np.ndarray]:
        filtered = [e for e in examples if e.layer == layer]
        feats, labels = [], []
        for ex in filtered:
            diff = ex.hidden_a - ex.hidden_b
            feat = np.concatenate([ex.hidden_a, ex.hidden_b, diff, np.abs(diff)])
            feats.append(feat)
            labels.append(int(ex.is_dependent))
        return np.stack(feats), np.array(labels)

    def run(
        self, examples: list[ControlDepExample], layer: int
    ) -> ProbeResult:
        X, y = self._features(examples, layer)
        return cross_validate_probe(
            LinearProbe, X, y, layer=layer, task="control_dep", config=self.config
        )


class BranchMembershipProbe:
    """Classify which branch body a token belongs to.

    This tests whether the model encodes 'which conditional guards this code'
    as part of its internal representation.
    """

    def __init__(self, config: Optional[ProbeConfig] = None):
        self.config = config or ProbeConfig()

    def run(
        self, examples: list[BranchMembershipExample], layer: int
    ) -> ProbeResult:
        filtered = [e for e in examples if e.layer == layer]
        X = np.stack([e.hidden for e in filtered])
        y = np.array([e.branch_id for e in filtered])
        return cross_validate_probe(
            LinearProbe, X, y, layer=layer, task="branch_membership", config=self.config
        )
