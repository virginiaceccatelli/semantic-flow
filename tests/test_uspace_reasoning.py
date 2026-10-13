"""Stage 292: CRUXEval prompt, answer extraction, and the upstream bootstrap."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from src.execution_uncertainty.data import prepare
from src.execution_uncertainty.reasoning import build_prompt, extract, grade

ROOT = Path(__file__).resolve().parents[1]
ITEM = {"ds": "cruxeval", "code": "def f(x):\n    return [x, x + 1]", "call": "f(1)", "gold": "[1, 2]"}


@pytest.fixture(autouse=True)
def no_global_test_manifests(monkeypatch):
    monkeypatch.setattr("src.cruxeval.artifacts.write_manifest", lambda *args, **kwargs: None)


def test_prompt_is_cruxeval_cot_format_without_answer():
    prompt = build_prompt(ITEM)
    assert prompt.startswith("You are given a Python function and an assertion")
    assert prompt.endswith("[PYTHON]\ndef f(x):\n    return [x, x + 1]\nassert f(1) == ??\n[/PYTHON]\n[THOUGHT]\n")
    assert "[1, 2]" not in prompt


@pytest.mark.parametrize("text, expected", [
    ("Steps...\n[ANSWER]\nassert f(1) == [1, 2]\n[/ANSWER]", "[1, 2]"),
    ("[ANSWER]\nassert f(1) == [0]\n[/ANSWER] then [ANSWER]\nassert f(1) == [1, 2]\n[/ANSWER]", "[1, 2]"),
    ("[ANSWER]\n```python\nassert f(1) == [1,\n 2]\n```\n[/ANSWER]", "[1,\n 2]"),
    ("So the result is:\nassert f(1) == [1, 2]\nDone.", "[1, 2]"),
    ("[ANSWER]\n[1, 2]\n[/ANSWER]", "[1, 2]"),
    ("The answer is a list.", None),
])
def test_extract_follows_cruxeval_rule(text, expected):
    assert extract(text) == expected


def test_grade_uses_python_assert_semantics():
    assert grade(ITEM, "[ANSWER]\nassert f(1) == [1, 2]\n[/ANSWER]") == ("[1, 2]", True)
    assert grade(ITEM, "[ANSWER]\nassert f(1) == [1, 3]\n[/ANSWER]")[1] is False
    assert grade(ITEM, "[ANSWER]\nassert f(1) == f(0)\n[/ANSWER]")[1] is False
    assert grade(ITEM, "no answer here") == (None, None)


UPSTREAM = os.environ.get("USPACE_PIPELINE", str(ROOT / "third_party/u-space-pipeline"))


@pytest.mark.skipif(not (Path(UPSTREAM) / "pipeline").is_dir(), reason="no upstream checkout")
def test_bootstrap_runs_upstream_sampling_with_code_items(tmp_path):
    raw = tmp_path / "raw.jsonl"
    raw.write_text("\n".join(json.dumps(dict(id=f"s{i}", code=f"def f(x):\n    return x + {i}",
                                               input="1", output=str(1 + i))) for i in range(5)))
    prepared = prepare("cruxeval", tmp_path / "prepared", input_jsonl=raw)
    config = tmp_path / "config.yaml"
    config.write_text(json.dumps({"data_dir": str(tmp_path / "data"), "items": {"n": 1000, "seed": 42},
                                  "datasets": {"cruxeval": {"prepared": str(prepared)}},
                                  "uspace_terms": "data/uspace_terms.yaml",
                                  "generation": {"seeds": [42], "prompt_headroom": 4096}, "models": {}}))
    script = f"""
import json, runpy, sys
from pathlib import Path
sys.path.insert(0, {str(ROOT)!r})
from src.execution_uncertainty import reasoning
root = reasoning.bootstrap(Path({UPSTREAM!r}))
import src
assert Path(src.__file__).resolve().parent == (root / "src").resolve()
sys.argv = ["01", "--config", {str(config)!r}]
runpy.run_path(str(root / "pipeline/01_sample_items.py"), run_name="__main__")
from src import prompts, grading, data
items = data.read_items(Path({str(tmp_path / 'data/items/cruxeval.jsonl')!r}))
assert len(items) == 5 and items[0]["ds"] == "cruxeval"
assert prompts.build(items[0]).endswith("assert f(1) == ??\\n[/PYTHON]\\n[THOUGHT]\\n")
it = [i for i in items if i["id"] == "s2"][0]
assert grading.grade(it, "[ANSWER]\\nassert f(1) == 3\\n[/ANSWER]") == ("3", True)
assert grading.grade({{"ds": "triviaqa", "aliases": ["paris"], "gold": "Paris"}}, "It is Paris.")[1] is True
print("ok")
"""
    env = dict(os.environ, PYTHONPATH="")
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, env=env, cwd=ROOT)
    assert out.returncode == 0, out.stderr[-3000:]
    assert out.stdout.strip().endswith("ok")
