"""Stage 293 prompt and answer rule: predict the returned value and explain the execution.

One prompt for every program (CRUXEval, MBPP, counterfactual pairs). The worked example
is the one in the project brief; it is not from either benchmark. The answer is the
text after the last ``Result:`` line of the post-thinking text, graded by the stage-291
grader (Python ``==`` as in ``assert <call> == <answer>``; constant expressions are
evaluated by a whitelisted AST interpreter; model text is never executed).

Outcomes are kept apart so none are silently dropped:

* no ``Result:`` line -> ``(None, None)``; upstream step 03 excludes it as ``no_answer``,
  and the stage-293 report counts it as a parsing failure;
* a ``Result:`` that is not a constant -> graded incorrect, reported as ``invalid``.
"""
from __future__ import annotations

import ast
import re

from src.execution_uncertainty.answers import answer_end, grade as grade_assertion

STYLE = "explain_result_v1"

PROMPT = '''Predict the exact output of the Python program below: the value returned by the call on its last line. Then explain how the program executes in 2-3 sentences.

For example, for

```python
def f(x):
    a = [x, x + 1]
    a[0] += a[1]
    return a[0]

f(3)
```

the answer is

Explanation: The list initially contains [3, 4]. The first element is increased by the second element, giving [7, 4].
Result: 7

Now the program:

```python
{program}
```

Answer in exactly this format:
Explanation: <2-3 sentences describing how the program executes>
Result: <the returned value as a Python literal>'''

RESULT = re.compile(r"^[ \t>*_#-]*\**result\**[ \t]*\**:\**[ \t]*", re.I | re.M)
EXPLANATION = re.compile(r"^[ \t>*_#-]*\**explanation\**[ \t]*\**:\**[ \t]*", re.I | re.M)


def program_text(item):
    return item["code"].replace("\r\n", "\n").rstrip() + "\n\n" + item["call"]


def build_prompt(item):
    return PROMPT.format(program=program_text(item))


def program_span(item):
    """(start, code_end, end) character offsets of the program inside the prompt."""
    prompt = build_prompt(item)
    text = program_text(item)
    start = prompt.rindex(text)
    code_end = start + len(item["code"].replace("\r\n", "\n").rstrip())
    return start, code_end, start + len(text)


def split_answer(text):
    """{'explanation', 'result'} from the post-thinking text; each None when absent."""
    results = list(RESULT.finditer(text))
    out = {"explanation": None, "result": None, "result_start": None}
    if not results:
        return out
    last = results[-1]
    body = text[last.end():]
    body = re.sub(r"^\s*```(?:python)?[ \t]*\n?", "", body)
    body = body.split("```")[0]
    end = answer_end(body.strip("\n") + "\n")
    value = body.strip("\n")[:end] if end is not None else body
    value = value.strip().strip("`").strip()
    value = re.sub(r"^\*+|\*+$", "", value).strip()
    out["result"], out["result_start"] = value, last.start()
    explanations = [m for m in EXPLANATION.finditer(text) if m.start() < last.start()]
    if explanations:
        out["explanation"] = text[explanations[-1].end():last.start()].strip()
    return out


def parse_expected(text):
    return ast.literal_eval(text)


def grade_text(item, text):
    """Full verdict for one post-thinking text."""
    parts = split_answer(text)
    if parts["result"] is None:
        return dict(parts, status="no_result", correct=None, prediction=None)
    verdict = grade_assertion(item["call"], parts["result"], item["gold"], parse=parse_expected)
    status = "parsed" if verdict["answer_status"] == "parsed" else "invalid"
    return dict(parts, status=status, correct=bool(verdict["correct"]), prediction=parts["result"][:300],
                invalid_reason=verdict["invalid_reason"], answer_form=verdict["answer_form"])


def grade(item, text):
    """Upstream ``grading.grade`` contract: (prediction, correct), correct None = no answer."""
    v = grade_text(item, text)
    return v["prediction"], v["correct"]
