"""Stage 293: tracer, counterfactual pairs, protocol, claims, E-Space algebra, freeze and an
end-to-end CPU analysis on a synthetic run with a planted E-Space signal."""
import gzip
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.espace import pairs, protocol
from src.espace.claims import find
from src.espace.load import Run
from src.espace.space import Space
from src.espace.tracing import profile, trace
from src.execution_uncertainty import reasoning
from src.execution_uncertainty.uspace import basis_from_vectors, unit

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/execsem_reasoning.yaml"


@pytest.fixture(autouse=True)
def no_global_test_manifests(monkeypatch):
    monkeypatch.setattr("src.cruxeval.artifacts.write_manifest", lambda *args, **kwargs: None)


@pytest.fixture(scope="module")
def cases():
    return pairs.generate(per_dimension=12, seed=7)[0]


# ---- execution ----

def test_tracer_profile_counts_branches_loops_and_raises():
    code = ("def g(x):\n    return int(x)\n"
            "def f(xs):\n    out = []\n    for x in xs:\n        try:\n            out.append(g(x))\n"
            "        except ValueError:\n            out.append(0)\n    if len(out) > 9:\n        out.sort()\n"
            "    return out\n")
    t = trace(code, "f(['3', 'a', '1', 'b'])")
    assert t["value"] == "[3, 0, 1, 0]"
    p = profile(code, t)
    assert p["exceptions_raised"] == 2          # each raise propagates through two frames
    assert p["max_loop_iterations"] == 4 and p["if_tests"] == 1 and p["if_bodies"] == 0
    assert t["values"]["out"][-1] == "[3, 0, 1, 0]"


def test_pairs_are_verified_minimal_and_split(cases):
    by_pair = {}
    for c in cases:
        by_pair.setdefault(c["pair_id"], []).append(c)
    assert len(by_pair) == 48
    for pid, (pos, neg) in by_pair.items():
        assert pos["label"] == 1 and neg["label"] == 0 and pos["split"] == neg["split"]
        same_code = pos["code"] == neg["code"]
        assert same_code != (pos["call"] == neg["call"])   # exactly one of code / call differs
        a, b = (pos["code"], neg["code"]) if not same_code else (pos["call"], neg["call"])
        assert len(a.splitlines()) == len(b.splitlines())
        for c in (pos, neg):                               # labels re-derived from a fresh trace
            observed, t = pairs.observed(c["dimension"], c["code"], c["call"], c["marker_line"], c["marker_var"])
            assert observed == bool(c["label"]) and t["value"] == c["expected"]
    assert {c["dimension"] for c in cases} == set(pairs.DIMENSIONS)
    assert {c["template"] for c in cases if c["dimension"] == "branch"} == \
        {t.__name__ for t in pairs.TEMPLATES["branch"]}


# ---- prompt and answers ----

ITEM = {"code": "def f(x):\n    return {x: [1.5]}", "call": "f(2)", "gold": "{2: [1.5]}",
        "prompt_style": protocol.STYLE, "ds": "cruxeval"}


def test_prompt_contains_program_and_format_but_not_answer():
    p = protocol.build_prompt(ITEM)
    assert "```python\ndef f(x):\n    return {x: [1.5]}\n\nf(2)\n```" in p
    assert p.endswith("Result: <the returned value as a Python literal>")
    assert "{2: [1.5]}" not in p
    s, c, e = protocol.program_span(ITEM)
    assert p[s:c] == ITEM["code"] and p[c:e] == "\n\nf(2)"


@pytest.mark.parametrize("text, status, correct", [
    ("Explanation: builds a dict.\nResult: {2: [1.5]}", "parsed", True),
    ("**Explanation:** a.\n\n**Result:** `{2: [1.5]}`", "parsed", True),
    ("Explanation: x\nResult:\n```python\n{2: [1.5]}\n```", "parsed", True),
    ("Result: {2:\n [1.5]}\ntrailing words", "parsed", True),
    ("Explanation: x\nResult: {2: [1.4]}", "parsed", False),
    ("Explanation: x\nResult: f(2)", "invalid", False),
    ("The answer is {2: [1.5]}", "no_result", None),
])
def test_result_extraction_and_grading(text, status, correct):
    v = protocol.grade_text(ITEM, text)
    assert v["status"] == status and v["correct"] == correct


def test_adapter_dispatches_on_prompt_style_and_keeps_stage_292(tmp_path):
    assert reasoning.build_prompt(ITEM) == protocol.build_prompt(ITEM)
    assert reasoning.grade(ITEM, "Result: {2: [1.5]}") == ("{2: [1.5]}", True)
    old = {k: v for k, v in ITEM.items() if k != "prompt_style"}
    old["gold"] = "[1, 2]"
    assert reasoning.build_prompt(old).startswith("You are given a Python function")
    case = {"case_id": "p:pos", "dataset": "cfpairs", "code": "def f():\n    return 1", "call": "f()",
            "expected": "1", "source_group": "p", "has_loop": False, "has_branch": True,
            "pair_id": "p", "label": 1, "values": {"x": ["1"]}}
    (tmp_path / "cases.jsonl").write_text(json.dumps(case) + "\n")
    rel = tmp_path.relative_to(ROOT) if tmp_path.is_relative_to(ROOT) else tmp_path
    item = reasoning.load_items({"prepared": str(rel), "prompt_style": protocol.STYLE})[0]
    assert item["prompt_style"] == protocol.STYLE and item["pair_id"] == "p" and item["values"] == {"x": ["1"]}


def test_claims_flag_values_the_variable_never_took():
    values = {"total": ["0", "1", "3", "6"], "xs": ["[1, 2, 3]"]}
    found = find("total = 0, then total becomes 1. Wait, total = 5? No: total = 3 + 3 = 6. "
                 "So total is 6. xs = [1, 2, 3]. Also total = 7. n = 4.", values)
    assert [(c["var"], c["value"], c["wrong"]) for c in found] == [
        ("total", "0", False), ("total", "1", False), ("total", "5", True), ("total", "6", False),
        ("xs", "[1, 2, 3]", False), ("total", "7", True)]


# ---- E-Space algebra ----

def test_space_from_cosines_equals_published_construction():
    rng = np.random.default_rng(0)
    d, terms = 32, {c: {"anchors": [f"{c}a{i}" for i in range(3)], "poles": [f"{c}p{i}" for i in range(4)]}
                    for c in ("one", "two", "three", "four")}
    words = [w for c in terms.values() for s in c.values() for w in s]
    ids = {w: k for k, w in enumerate(words)}
    D = unit(rng.standard_normal((len(words), d))).astype(np.float64)
    basis, _, _ = basis_from_vectors(dict(enumerate(D)), ids, terms)
    rows = {c: {s: [ids[w] for w in terms[c][s]] for s in terms[c]} for c in terms}
    space = Space(D, rows)
    h = rng.standard_normal((5, d))
    C = unit(h) @ D.T
    assert np.allclose(space.orth(C), unit(h) @ basis, atol=1e-5)
    b = np.stack([unit(D[rows[c]["anchors"]].mean(0) - D[rows[c]["poles"]].mean(0)) for c in terms], 1)
    assert np.allclose(space.bipolar(C), unit(h) @ b, atol=1e-5)


# ---- synthetic end-to-end run ----

def _fake_run(tmp, cases, d=24, seeds=(41, 42), signal=3.0):
    rng = np.random.default_rng(1)
    cfg = yaml.safe_load(CONFIG.read_text())
    cfg["generation"]["seeds"] = list(seeds)
    cfg["models"] = {"qwen": cfg["models"]["qwen"]}
    cfg["espace"]["word_controls"] = cfg["espace"]["gaussian_controls"] = 4
    config = tmp / "config.yaml"
    config.write_text(yaml.safe_dump(cfg))
    data = tmp / "data"
    terms = yaml.safe_load((ROOT / cfg["espace"]["terms"]).read_text())
    words = [w for c in terms for s in ("anchors", "poles") for w in terms[c][s] if w != "unrepeated"]
    pool = [f"w{i}" for i in range(400)]
    controls = []
    for k in range(4):
        picks = iter(rng.choice(pool, 40, replace=False))
        controls.append({c: {s: [str(next(picks)) for _ in terms[c][s]] for s in ("anchors", "poles")}
                         for c in terms})
    U = np.linalg.qr(rng.standard_normal((d, 4)))[0]
    E = unit(rng.standard_normal((len(words), d)))
    C_rows = [(f"C{k}:{w}", unit(rng.standard_normal(d))) for k, ct in enumerate(controls)
              for c in ct.values() for s in ("anchors", "poles") for w in c[s]]
    G = [np.linalg.qr(rng.standard_normal((d, 4)))[0] for _ in range(4)]
    names = [f"U:{c}" for c in ("ambiguity", "incompleteness", "conflicting_evidence", "general_uncertainty")]
    names += [f"E:{w}" for w in words] + [n for n, _ in C_rows] + [f"G{k}:{j}" for k in range(4) for j in range(4)]
    D = np.concatenate([U.T, E, np.stack([v for _, v in C_rows])] + [g.T for g in G])
    bdir = data / "espace" / "qwen"
    bdir.mkdir(parents=True)
    np.savez(bdir / "bank.npz", D=D.astype(np.float32), names=np.array(names), n_main=4 + len(words))
    (bdir / "bank.json").write_text(json.dumps({"terms": terms, "controls": controls,
                                                "uspace_categories": [n[2:] for n in names[:4]]}))
    (data / "uspace").mkdir(parents=True)
    np.savez(data / "uspace" / "qwen.npz", U=U, categories=np.array([n[2:] for n in names[:4]]))
    idx = {w: k for k, w in enumerate(words)}
    dirs = [unit(E[[idx[w] for w in terms[c]["anchors"] if w in idx]].mean(0) -
                 E[[idx[w] for w in terms[c]["poles"] if w in idx]].mean(0)) for c in terms]
    for ds in ("cruxeval", "cfpairs"):
        items = [dict(id=c["case_id"], ds=ds, code=c["code"], call=c["call"], gold=c["expected"],
                      group=c["source_group"], has_loop=False, has_branch=False, stratum=None,
                      prompt_style=protocol.STYLE, runtime=profile(c["code"], trace(c["code"], c["call"])),
                      **{k: c[k] for k in reasoning.EXTRA if k in c}) for c in cases]
        (data / "items").mkdir(exist_ok=True)
        (data / "items" / f"{ds}.jsonl").write_text("".join(json.dumps(i) + "\n" for i in items))
        state = {}
        for it in items:
            h = rng.standard_normal(d)
            k = pairs.DIMENSIONS.index(it["dimension"])
            state[it["id"]] = h + signal * (1 if it["label"] else -1) * dirs[k]
        M = len(names)

        def cos(h):
            return unit(np.atleast_2d(h)) @ D.T

        P = np.stack([np.repeat(cos(state[it["id"]]), 4, 0) for it in items])
        np.savez(bdir / f"{ds}_prompt.npz", id=np.array([i["id"] for i in items]),
                 slots=np.array(["code_end", "program_end", "prompt_end", "program_mean"]), summ=P,
                 cos=np.zeros((0, 4 + len(words)), np.float16), n=np.zeros(len(items), int),
                 off=np.zeros(len(items), int))
        for seed in seeds:
            labels, rows, texts, summ, tok, ent, states = [], [], [], [], [], [], []
            for j, it in enumerate(items):
                if j == 0:
                    labels.append(dict(id=it["id"], tlen=99, eot_tok=None, pred=None, correct=None,
                                       excluded="truncated"))
                    continue
                n_think, n_ans = 30, 6
                wrong = bool(rng.random() < 0.4)
                values = it.get("values") or {}
                var = next(iter(values), "x")
                claim = f"{var} = 999" if wrong else f"{var} = {values.get(var, ['0'])[0]}"
                think = (f"Let me trace. Wait, {claim}. " + "step " * 40)[:n_think * 4]
                ends = [min(4 * (t + 1), len(think)) for t in range(n_think)]
                labels.append(dict(id=it["id"], tlen=n_think + n_ans, eot_tok=n_think, pred="1", correct=not wrong,
                                   excluded=None))
                rows.append(it["id"])
                H = state[it["id"]] + 0.5 * rng.standard_normal((n_think + n_ans, d))
                tok.append(cos(H)[:, :4 + len(words)].astype(np.float16))
                summ.append(np.stack([cos(H[n_think])[0], cos(H[:n_think + 1].mean(0))[0],
                                      cos(H[n_think + 1:].mean(0))[0], cos(H[n_think + 1:].mean(0))[0]]))
                states.append(H[n_think])
                ent.append(rng.random(n_think + n_ans) + (0.3 if wrong else 0))
                texts.append(dict(id=it["id"], think=think, think_ends=ends, answer="Result: 1",
                                  explanation="x", result="1", status="parsed", correct=not wrong,
                                  invalid_reason=None, explanation_tokens=None))
            n = np.array([len(e) for e in ent])
            off = np.concatenate([[0], np.cumsum(n)[:-1]])
            err = np.array([int(not l["correct"]) for l in labels if l["correct"] is not None])
            for sub in ("labels", "forward", "texts"):
                (data / sub / "qwen").mkdir(parents=True, exist_ok=True)
            (data / "labels" / "qwen" / f"{ds}_s{seed}.jsonl").write_text("".join(json.dumps(l) + "\n" for l in labels))
            np.savez(data / "forward" / "qwen" / f"{ds}_s{seed}.npz", id=np.array(rows), error=err,
                     eot_tok=np.full(len(rows), 30), n=n, off=off, tok_entropy=np.concatenate(ent).astype(np.float32),
                     eot_state=np.stack(states).astype(np.float32))
            np.savez(bdir / f"{ds}_s{seed}.npz", id=np.array(rows), slots=np.array(["eot", "think_mean", "answer_mean",
                                                                                      "explanation_mean"]),
                     summ=np.stack(summ), cos=np.concatenate(tok), n=n, off=off, eot_tok=np.full(len(rows), 30))
            with gzip.open(data / "texts" / "qwen" / f"{ds}_s{seed}.jsonl.gz", "wt") as f:
                f.writelines(json.dumps(t) + "\n" for t in texts)
            sdir = data / "scores" / "qwen" / f"{ds}_s{seed}"
            sdir.mkdir(parents=True)
            for name in ("ulens", "a_cone", "predictive_entropy", "gen_length"):
                np.savez(sdir / f"{name}.npz", id=np.array(rows), error=err, gen_length=n.astype(float),
                         score=err + rng.random(len(rows)) + 0.1)
    return Run(config, data)


def test_end_to_end_freeze_and_analysis(tmp_path, cases):
    from src.espace.freeze import freeze
    from src.espace.report import analyze
    run = _fake_run(tmp_path, cases)
    out = freeze(run)
    frozen = json.loads(out.read_text())
    assert "unrepeated" in frozen["dropped"] and frozen["dropped"]["unrepeated"]["rule"] == "tokenization"
    with pytest.raises(SystemExit):
        freeze(run)                                      # never refreezes silently
    report = analyze(run, tmp_path / "analysis")
    s = json.loads((report / "summary.json").read_text())
    m = s["models"]["qwen"]
    e1 = m["experiment1"]["cruxeval"]
    assert e1["seeds"]["41"]["excluded"] == {"truncated": 1}
    assert e1["methods"]["ulens"]["auroc"] > 0.7 and e1["eot_projection_max_abs_diff"] < 1e-3
    pairs_res = m["experiment2"]["pairs"]
    for dim in pairs.DIMENSIONS:                          # the planted signal, recovered with its sign
        e = pairs_res["dimensions"][dim]["positions"]["prompt_end"]
        assert e["E_orth"]["dz"] > 1 and e["E_bipolar"]["dz"] > 1 and e["E_orth"]["pair_accuracy"] > 0.8
    t = m["tokens"]["cruxeval"]
    assert t["n_wrong_claims"] > 0 and "combined" in t
    assert "Counterfactual pairs" in (report / "report.md").read_text()


def test_claims_skip_numbers_that_are_not_python_literals():
    assert [c["value"] for c in find("x = 05, then x = 5 and x = 007.", {"x": ["5"]})] == ["5"]
