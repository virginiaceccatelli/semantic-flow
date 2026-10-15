"""CPU-only tests of the actual HF model hooks; no model downloads."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from src.bugstate.model import Engine, parse_answer, random_delta, swap_delta
from src.bugstate.experiment import fit_probe, fit_all
from src.bugstate.report import estimate, report
from src.bugstate.data import write_jsonl


class Tokenizer:
    eos_token_id = 2
    eos_token = '<eos>'
    def decode(self, ids, **kwargs):
        return ' '.join(map(str, ids.tolist()))


def test_real_qwen_hooks_and_noop_generation():
    torch.manual_seed(7)
    config = Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
                        head_dim=8, eos_token_id=2, pad_token_id=2)
    config._attn_implementation = 'eager'
    model = Qwen3ForCausalLM(config).eval()
    engine = Engine(model, Tokenizer(), max_new_tokens=4)
    ids = torch.tensor([[4, 5, 6]])
    states = engine.extract(ids)
    assert states.shape == (2, 16)
    clean = engine.generate(ids)
    noop = engine.generate(ids, 0, np.zeros(16))
    assert clean == noop
    assert all(not block._forward_hooks for block in engine.layers)
    engine.generate(ids, 0, np.ones(16) * .1)
    assert all(not block._forward_hooks for block in engine.layers)


def test_geometry_and_scaling():
    host = np.array([1., 2., 3.])
    donor = np.array([4., 6., 8.])
    q = np.array([1., 0., 0.])
    d = swap_delta(host, donor, q)
    np.testing.assert_allclose(host + d, [4., 2., 3.])
    r = random_delta(d, q, 42)
    assert np.dot(r, q) == pytest.approx(0)
    assert np.linalg.norm(r) == pytest.approx(np.linalg.norm(d))
    x = np.array([[1., 100.], [2., 200.], [4., -100.], [5., -200.]])
    scaler, clf, direction = fit_probe(x, [0, 0, 1, 1])
    raw = clf.coef_[0] / scaler.scale_
    np.testing.assert_allclose(direction, raw / np.linalg.norm(raw), rtol=1e-5)


def test_answer_parser():
    assert parse_answer('<think>FINAL: 12</think>\nFINAL: True') == 'True'
    assert parse_answer('<think>FINAL: 12') is None
    assert parse_answer('no final') is None


def test_project_weighting():
    rows = [dict(project='a', pair=str(i), correct=1) for i in range(10)]
    rows += [dict(project='b', pair='only', correct=0)]
    result = estimate(rows, 'correct')
    assert result['mean'] == .5
    assert result['projects'] == 2


def test_project_holdout_and_report(tmp_path):
    # Fixtures exercise the pipeline only; not experiment observations.
    pairs = []
    for p in ('one', 'two', 'three'):
        pairs.append(dict(id=p, project=p, kind='changed', variable='flag',
            file='f.py', function='f', versions=[
                dict(inputs={'x': '1'}, code='flag = False\nreturn flag', line=2, label=0, answer='0'),
                dict(inputs={'x': '1'}, code='flag = True\nreturn flag', line=2, label=1, answer='1')]))
    states = np.zeros((3, 2, 2, 4), dtype=np.float32)
    states[:, 0, :, 0] = -1
    states[:, 1, :, 0] = 1
    directions = fit_all(pairs, states, tmp_path)
    assert len(directions) == 6
    predictions = [json.loads(s) for s in (tmp_path / 'probe_predictions.jsonl').read_text().splitlines()]
    assert all(r['correct'] for r in predictions if r['method'] == 'activation')
    assert np.mean([r['correct'] for r in predictions if r['method'] == 'input_only']) == .5
    write_jsonl(tmp_path / 'included_pairs.jsonl', pairs)
    write_jsonl(tmp_path / 'context_exclusions.jsonl', [])
    write_jsonl(tmp_path / 'clean_generations.jsonl', [dict(answer='0', hit_token_limit=False)])
    effects = [dict(pair=p['id'], project=p['project'], kind='changed', method=m,
                    layer=l, donor_gain=int(m == 'probe'), clean_both_correct=True)
               for p in pairs for m in ('probe', 'random') for l in range(2)]
    write_jsonl(tmp_path / 'interventions.jsonl', effects)
    (tmp_path / 'config.json').write_text(json.dumps(dict(data=str(tmp_path), model='TEST FIXTURE')))
    report(SimpleNamespace(output=str(tmp_path)))
    for f in ('decodability.png', 'decodability.pdf', 'intervention_effect.png',
              'intervention_effect.pdf', 'report.md', 'cases.csv'):
        assert (tmp_path / f).stat().st_size > 0
    assert 'No unchanged-value controls' in (tmp_path / 'report.md').read_text()
