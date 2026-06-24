"""Tests for probe classes."""

import numpy as np
import pytest

from src.probes.base import LinearProbe, ProbeConfig, ProbeResult, cross_validate_probe
from src.probes.lexical import BindingExample, BindingProbe, LexicalExample, LexicalProbe
from src.probes.defuse import DefUseEdgeProbe, DefUseExample


RNG = np.random.default_rng(0)
D = 64   # small hidden dim for tests


def _rand_hidden(n: int, d: int = D) -> np.ndarray:
    return RNG.standard_normal((n, d)).astype(np.float32)


class TestLinearProbe:
    def test_fit_predict(self):
        X = _rand_hidden(200)
        y = (RNG.random(200) > 0.5).astype(int)
        probe = LinearProbe()
        probe.fit(X, y)
        preds = probe.predict(X)
        assert preds.shape == (200,)

    def test_evaluate_returns_dict(self):
        X = _rand_hidden(200)
        y = (RNG.random(200) > 0.5).astype(int)
        probe = LinearProbe()
        probe.fit(X, y)
        m = probe.evaluate(X, y)
        assert "accuracy" in m
        assert 0.0 <= m["accuracy"] <= 1.0

    def test_cross_validate(self):
        X = _rand_hidden(100)
        y = (RNG.random(100) > 0.5).astype(int)
        cfg = ProbeConfig(cv_folds=3, run_selectivity_control=True)
        result = cross_validate_probe(LinearProbe, X, y, layer=0, task="test", config=cfg)
        assert isinstance(result, ProbeResult)
        assert result.layer == 0
        assert 0.0 <= result.accuracy <= 1.0
        assert result.control_accuracy >= 0.0

    def test_selectivity_is_accuracy_minus_control(self):
        X = _rand_hidden(100)
        y = (RNG.random(100) > 0.5).astype(int)
        cfg = ProbeConfig(cv_folds=3, run_selectivity_control=True)
        result = cross_validate_probe(LinearProbe, X, y, layer=0, task="test", config=cfg)
        assert abs(result.selectivity - (result.accuracy - result.control_accuracy)) < 1e-6

    def test_probe_not_fitted_raises(self):
        probe = LinearProbe()
        with pytest.raises(AssertionError):
            probe.predict(_rand_hidden(10))


class TestLexicalProbe:
    def _make_examples(self, n=100, n_layers=3):
        from src.probes.lexical import TOKEN_TYPES
        examples = []
        for layer in range(n_layers):
            for i in range(n):
                examples.append(LexicalExample(
                    hidden=RNG.standard_normal(D).astype(np.float32),
                    token_str="x",
                    token_type=TOKEN_TYPES[i % len(TOKEN_TYPES)],
                    layer=layer,
                    position=i,
                ))
        return examples

    def test_run_returns_result(self):
        examples = self._make_examples()
        probe = LexicalProbe(config=ProbeConfig(cv_folds=3))
        result = probe.run(examples, layer=0)
        assert isinstance(result, ProbeResult)
        assert result.task == "lexical_token_type"


class TestBindingProbe:
    def _make_examples(self, n=120, layer=0):
        # Four cases so both name-splits contain both binding classes:
        #   i%4==0: same name, same binding
        #   i%4==1: same name, diff binding (shadowed variable)
        #   i%4==2: diff name, same binding (alias)
        #   i%4==3: diff name, diff binding
        examples = []
        for i in range(n):
            case = i % 4
            same = case in (0, 2)
            h_a = RNG.standard_normal(D).astype(np.float32)
            h_b = h_a + RNG.standard_normal(D).astype(np.float32) * (0.01 if same else 1.0)
            token_b = "x" if case in (0, 1) else "y"
            examples.append(BindingExample(
                hidden_a=h_a, hidden_b=h_b,
                token_str_a="x", token_str_b=token_b,
                same_binding=same,
                layer=layer, pos_a=i, pos_b=i + 1,
            ))
        return examples

    def test_run_returns_result(self):
        examples = self._make_examples()
        probe = BindingProbe(config=ProbeConfig(cv_folds=3))
        result = probe.run(examples, layer=0)
        assert isinstance(result, ProbeResult)

    def test_decoy_split(self):
        examples = self._make_examples(n=100)
        probe = BindingProbe(config=ProbeConfig(cv_folds=3))
        results = probe.run_lexical_decoy_split(examples, layer=0)
        assert isinstance(results, dict)


class TestDefUseEdgeProbe:
    def _make_examples(self, n=80, layer=0):
        examples = []
        for i in range(n):
            has_edge = i % 3 == 0
            h_i = RNG.standard_normal(D).astype(np.float32)
            h_j = h_i * 0.9 + RNG.standard_normal(D).astype(np.float32) * (0.05 if has_edge else 1.0)
            examples.append(DefUseExample(
                hidden_i=h_i, hidden_j=h_j,
                has_edge=has_edge,
                layer=layer, pos_i=i, pos_j=i + 5,
                distance=5,
            ))
        return examples

    def test_run_returns_result(self):
        examples = self._make_examples()
        probe = DefUseEdgeProbe(config=ProbeConfig(cv_folds=3))
        result = probe.run(examples, layer=0)
        assert isinstance(result, ProbeResult)
        assert result.task == "defuse_edge"

    def test_run_by_distance(self):
        examples = []
        for dist in [5, 25, 100, 300]:
            for _ in range(30):
                h_i = RNG.standard_normal(D).astype(np.float32)
                h_j = RNG.standard_normal(D).astype(np.float32)
                examples.append(DefUseExample(
                    hidden_i=h_i, hidden_j=h_j,
                    has_edge=bool(RNG.integers(0, 2)),
                    layer=0, pos_i=0, pos_j=dist, distance=dist,
                ))

        probe = DefUseEdgeProbe(config=ProbeConfig(cv_folds=3))
        results = probe.run_by_distance(examples, layer=0)
        assert isinstance(results, dict)
