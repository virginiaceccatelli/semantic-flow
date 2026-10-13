"""Stage 292: the published U-Space pipeline on code-execution prediction.

The upstream pipeline (s2labres/U-Space, pinned commit in third_party/u-space/COMMIT)
is run unmodified: sampling, vLLM generation, labelling, teacher-forced forward pass,
basis, scores and tables. Its steps dispatch per benchmark in three places only, and
this module supplies code-execution versions of exactly those three:

* ``data.LOADERS``  — items are the stage-290 execution-verified CRUXEval / MBPP calls;
* ``prompts.build`` — CRUXEval's official chain-of-thought output-prediction prompt
  (facebookresearch/cruxeval ``prompts.py::make_cot_output_prompt``, commit
  190faf16d175b5847b0af05d937872b1fb395942), used verbatim for both benchmarks;
* ``grading.grade`` — CRUXEval's answer rule (text after ``[ANSWER]``, then after the
  first ``==``) on the post-thinking text, graded by the stage-291 grader (Python
  ``assert`` semantics, constant expressions, never executing model text).

Upstream and this repository both name their package ``src``. ``bootstrap`` imports
everything needed from this repository first, keeps it alive under ``semflow_src``
aliases, and only then lets ``src`` resolve to the upstream checkout.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

from src.execution_uncertainty.answers import answer_end, grade as grade_answer

ROOT = Path(__file__).resolve().parents[2]
PINNED = (ROOT / "third_party/u-space/COMMIT").read_text().strip()
UPSTREAM = ROOT / "third_party/u-space-pipeline"
CODE_DATASETS = ("cruxeval", "mbpp")
CRUXEVAL_COMMIT = "190faf16d175b5847b0af05d937872b1fb395942"

# Verbatim from cruxeval prompts.py::make_cot_output_prompt at CRUXEVAL_COMMIT, with
# ``f({input})`` generalised to the published call (identical text for CRUXEval).
COT_OUTPUT_PROMPT = '''You are given a Python function and an assertion containing an input to the function. Complete the assertion with a literal (no unsimplified expressions, no function calls) containing the output when executing the provided code on the given input, even if the function is incorrect or incomplete. Do NOT output any extra information. Execute the program step by step before arriving at an answer, and provide the full assertion with the correct output in [ANSWER] and [/ANSWER] tags, following the examples.

[PYTHON]
def f(s):
    s = s + s
    return "b" + s + "a"
assert f("hi") == ??
[/PYTHON]
[THOUGHT]
Let's execute the code step by step:

1. The function f is defined, which takes a single argument s.
2. The function is called with the argument "hi", so within the function, s is initially "hi".
3. Inside the function, s is concatenated with itself, so s becomes "hihi".
4. The function then returns a new string that starts with "b", followed by the value of s (which is now "hihi"), and ends with "a".
5. The return value of the function is therefore "bhihia".
[/THOUGHT]
[ANSWER]
assert f("hi") == "bhihia"
[/ANSWER]

[PYTHON]
{code}
assert {call} == ??
[/PYTHON]
[THOUGHT]
'''


def build_prompt(item):
    return COT_OUTPUT_PROMPT.format(code=item["code"].rstrip(), call=item["call"])


def load_items(src, raw_dir=None):
    """Upstream item dicts from a stage-290 prepared directory (``src['prepared']``)."""
    path = ROOT / src["prepared"] / "cases.jsonl"
    items = []
    for line in path.read_text().splitlines():
        if line.strip():
            case = json.loads(line)
            items.append(dict(id=case["case_id"], ds=case["dataset"], code=case["code"],
                              call=case["call"], gold=case["expected"],
                              group=case["source_group"], has_loop=case["has_loop"],
                              has_branch=case["has_branch"], stratum=None))
    return items


def extract(text):
    """CRUXEval's rule: text after the last ``[ANSWER]`` (up to ``[/ANSWER]``), then after
    the first ``==``. Without an ``[ANSWER]`` block, the last line holding an assertion
    is used; with neither, there is no answer (upstream then excludes the item)."""
    blocks = re.findall(r"\[ANSWER\](.*?)(?:\[/ANSWER\]|$)", text, re.S)
    if blocks:
        body = blocks[-1]
    else:
        lines = [l for l in text.splitlines() if "assert" in l and "==" in l]
        if not lines:
            return None
        body = lines[-1]
    body = re.sub(r"```(?:python)?", "", body).strip()
    if "==" in body:
        body = body.split("==", 1)[1]
    body = body.lstrip()
    end = answer_end(body + "\n")
    return body[:end] if end is not None else body


def grade(item, answer_text):
    """(prediction, correct); correct is None when no answer can be extracted."""
    answer = extract(answer_text)
    if answer is None:
        return None, None
    result = grade_answer(item["call"], answer, item["gold"])
    return answer[:300], bool(result["correct"])


def check_checkout(root=UPSTREAM):
    root = Path(root)
    if not (root / "pipeline").is_dir():
        raise FileNotFoundError(f"Upstream checkout missing at {root}; run "
                                "`bash jobs/uspace_reasoning.sh setup`")
    head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True,
                          text=True, check=True).stdout.strip()
    if head != PINNED:
        raise ValueError(f"Upstream checkout is at {head}, expected pinned {PINNED}")
    return root


def bootstrap(root=UPSTREAM):
    """Hand the name ``src`` to the upstream checkout and install the three adapters."""
    root = check_checkout(root)
    for name in [n for n in sys.modules if n == "src" or n.startswith("src.")]:
        sys.modules["semflow_" + name] = sys.modules.pop(name)
    sys.path.insert(0, str(root))
    import src as upstream
    if Path(upstream.__file__).resolve().parent != (root / "src").resolve():
        raise ImportError(f"`src` resolved to {upstream.__file__}, not the upstream checkout")
    from src import data, grading, prompts
    original_build, original_grade = prompts.build, grading.grade

    def build(it):
        return build_prompt(it) if it["ds"] in CODE_DATASETS else original_build(it)

    def grade_any(it, answer):
        return grade(it, answer) if it["ds"] in CODE_DATASETS else original_grade(it, answer)

    for ds in CODE_DATASETS:
        data.LOADERS[ds] = load_items
    prompts.build, grading.grade = build, grade_any
    return root

