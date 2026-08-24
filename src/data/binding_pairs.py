"""E13 data: a binding counterfactual crossed with the value assignment.

E11 intervened on the *value* a use resolves to, and what survived was a claim
about output-aligned token directions — because the two candidate values are
literals in the text, a direction that pushes toward the token `7` and a
direction that carries "the inner definition is in scope" are indistinguishable.
E12 tried to escape that by making the tracked value text-absent, which forced
the value to be *computed*, which made two-step arithmetic the load-bearing
capability. It failed there (`docs/ARCHIVE.md §1.4`, and the 1.3b triage),
and the failure was about arithmetic rather than about binding.

E13 takes the cheaper escape. It intervenes on the **binding** and lets the
value assignment do the disambiguating work, in a 2x2:

    ARM "ab"  (v_outer, v_inner) = (a, b)      ARM "ba"  = (b, a)

    source (outer binding)  target (inner binding)
        x = a                   x = a
        def f():                def f():
            y = b                   x = b        <- ONE differing token
            return x                return x
        -> a                    -> b

Within an arm, source and target are a one-token counterfactual: the inner
definition's *name*. Across arms, the two literals swap.

**The falsification.** Install the target run's state into the source run at the
marked use. In arm `ab` the answer must move a -> b; in arm `ba` the very same
intervention must move it b -> a. A subspace encoding "the token b" gets arm
`ab` right and arm `ba` exactly backwards. A subspace encoding "the answer"
does the same. Only something encoding *which definition is in scope* moves
both arms toward the value the installed binding selects.

So the alignment is fitted on arm `ab` alone and tested on arm `ba`, which no
prior design here could do: E11's `answers_disjoint_from_values` invariant
exists precisely because it had no way to break that circularity, and it paid
for it by requiring an arithmetic operation between the value and the answer.
Here the answer IS the bound value — deliberately — and the arm swap breaks the
circularity instead. **No arithmetic anywhere**: the model has to return a
variable, nothing more.

Reuse is total on the alignment side: the template is E11's `global_shadow`
with the operation removed, so `counterfactual_pairs._ast_spans` and
`_positions_for` resolve every anchor unchanged.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import jsonlines

from src.data.counterfactual_pairs import (
    NAME_POOL,
    VALUE_POOL,
    _appends_one_token,
    _positions_for,
    choose_answer_suffix,
    encode_prompt,
    execute_program,
    number_token,
)

logger = logging.getLogger(__name__)

ARMS = ("ab", "ba")
BINDINGS = ("source", "target")          # source = outer binding, target = inner

# The four programs of one base, as (arm, binding) keys.
CELLS = tuple((arm, binding) for arm in ARMS for binding in BINDINGS)

# Minimum tokens between the mutated name and the marked use. E11 required 2
# (never adjacent); E13 keeps a wider margin so no local window around the use
# can see which name the inner definition carries.
MIN_MUTATION_DISTANCE = 4


def render(outer: str, inner: str, v_outer: int, v_inner: int) -> str:
    """E11's `global_shadow`, with the downstream operation removed.

    `inner == outer` renders the shadowing (target) program; any other name
    renders the non-shadowing (source) one. The two differ at exactly one token.
    """
    return (f"{outer} = {v_outer}\n"
            f"def f():\n"
            f"    {inner} = {v_inner}\n"
            f"    return {outer}")


@dataclass
class BindingFactorial:
    """One base: four programs, two arms x two bindings, one shared anchor set."""

    pair_id: str
    base_id: str
    outer_name: str
    inner_name: str                 # the non-shadowing inner name
    v_a: int
    v_b: int

    programs: dict                  # "{arm}_{binding}" -> source
    answers: dict                   # "{arm}_{binding}" -> int
    positions: dict                 # shared by all four programs
    token_ids: dict                 # "v_a" / "v_b" -> single token id
    n_tokens: int
    mutation_index: int
    answer_suffix: str
    split: str = "unassigned"
    metadata: dict = field(default_factory=dict)

    # -- accessors -----------------------------------------------------------
    @staticmethod
    def key(arm: str, binding: str) -> str:
        return f"{arm}_{binding}"

    def program(self, arm: str, binding: str) -> str:
        return self.programs[self.key(arm, binding)]

    def prompt(self, arm: str, binding: str) -> str:
        return self.program(arm, binding) + self.answer_suffix

    def answer(self, arm: str, binding: str) -> int:
        """The value the use resolves to in this cell."""
        return self.answers[self.key(arm, binding)]

    def other_answer(self, arm: str, binding: str) -> int:
        """The answer under the OPPOSITE binding, same arm.

        This is what an interchange from the other binding should produce, and
        it is the quantity whose *token* flips between arms while its *meaning*
        ("the value the installed binding selects") does not.
        """
        other = "target" if binding == "source" else "source"
        return self.answers[self.key(arm, other)]

    def answer_token(self, arm: str, binding: str) -> int:
        value = self.answer(arm, binding)
        return self.token_ids["v_a" if value == self.v_a else "v_b"]

    def other_answer_token(self, arm: str, binding: str) -> int:
        value = self.other_answer(arm, binding)
        return self.token_ids["v_a" if value == self.v_a else "v_b"]

    def to_dict(self) -> dict:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, d: dict) -> "BindingFactorial":
        d = dict(d)
        d["positions"] = {k: int(v) for k, v in d["positions"].items()}
        d["token_ids"] = {k: int(v) for k, v in d["token_ids"].items()}
        d["answers"] = {k: int(v) for k, v in d["answers"].items()}
        return cls(**d)


def _finalize(
    base_index: int,
    outer: str,
    inner: str,
    v_a: int,
    v_b: int,
    tokenizer,
    answer_suffix: str,
) -> Optional[BindingFactorial]:
    """Build one base if every invariant holds, else None (a normal rejection)."""
    if v_a == v_b or outer == inner:
        return None

    sources = {
        "ab_source": render(outer, inner, v_a, v_b),
        "ab_target": render(outer, outer, v_a, v_b),
        "ba_source": render(outer, inner, v_b, v_a),
        "ba_target": render(outer, outer, v_b, v_a),
    }

    # 1. execution ground truth, and the factorial's defining property: the
    #    binding flip moves the answer in OPPOSITE token directions per arm.
    try:
        answers = {key: execute_program(src) for key, src in sources.items()}
    except Exception:                                        # unrunnable candidate
        return None
    expected = {"ab_source": v_a, "ab_target": v_b, "ba_source": v_b, "ba_target": v_a}
    if answers != expected:
        logger.warning("base_%04d: execution disagrees with the template (%s vs %s)",
                       base_index, answers, expected)
        return None

    # 2. both values need their own single-token row
    token_ids = {}
    for name, value in (("v_a", v_a), ("v_b", v_b)):
        number = number_token(tokenizer, value)
        if number is None:
            return None
        token_ids[name] = number[0]
    if token_ids["v_a"] == token_ids["v_b"]:
        return None

    # 3. all four prompts share a token length, and each arm's source/target
    #    differ at exactly one token — the inner definition's name.
    prompts = {key: src + answer_suffix for key, src in sources.items()}
    ids = {key: encode_prompt(tokenizer, prompt) for key, prompt in prompts.items()}
    if len({len(i) for i in ids.values()}) != 1:
        return None
    mutation_index = None
    for arm in ARMS:
        diffs = [i for i, (a, b) in enumerate(zip(ids[f"{arm}_source"], ids[f"{arm}_target"]))
                 if a != b]
        if len(diffs) != 1:
            return None
        if mutation_index is None:
            mutation_index = diffs[0]
        elif mutation_index != diffs[0]:
            return None                       # the arms must mutate the same slot

    # 4. anchors resolve identically in all four programs
    positions = {}
    for key, src in sources.items():
        resolved = _positions_for(prompts[key], tokenizer, src, outer)
        if resolved is None:
            return None
        positions[key] = resolved
    shared = positions["ab_source"]
    if any(positions[key] != shared for key in sources):
        return None

    # 5. the mutation is the inner name, well before the use, and the use token
    #    itself is identical everywhere
    if mutation_index != shared["mutation"]:
        return None
    if shared["use"] - shared["mutation"] < MIN_MUTATION_DISTANCE:
        return None
    use_tokens = {ids[key][shared["use"]] for key in sources}
    if len(use_tokens) != 1:
        return None

    # 6. the answer is exactly one appended token in every cell
    for key in sources:
        if not _appends_one_token(tokenizer, prompts[key], answer_suffix, answers[key]):
            return None

    return BindingFactorial(
        pair_id=f"base_{base_index:04d}", base_id=f"base_{base_index:04d}",
        outer_name=outer, inner_name=inner, v_a=v_a, v_b=v_b,
        programs=sources, answers=answers, positions=shared, token_ids=token_ids,
        n_tokens=len(ids["ab_source"]), mutation_index=int(mutation_index),
        answer_suffix=answer_suffix,
        metadata={"use_minus_mutation": shared["use"] - shared["mutation"],
                  "cells": {f"{arm}_{binding}": answers[f"{arm}_{binding}"]
                            for arm, binding in CELLS}},
    )


def generate_binding_factorials(
    tokenizer,
    n_bases: int = 400,
    seed: int = 42,
    max_attempts_per_base: int = 40,
) -> list[BindingFactorial]:
    """`n_bases` verified factorials, one record each (four programs inside)."""
    rng = random.Random(seed)
    answer_suffix = choose_answer_suffix(tokenizer)
    out: list[BindingFactorial] = []
    attempts = 0
    while len(out) < n_bases and attempts < n_bases * max_attempts_per_base:
        attempts += 1
        outer, inner = rng.sample(NAME_POOL, 2)
        v_a, v_b = rng.sample(VALUE_POOL, 2)
        record = _finalize(len(out), outer, inner, v_a, v_b, tokenizer, answer_suffix)
        if record is not None:
            out.append(record)
    if len(out) < n_bases:
        logger.warning("generated %d/%d bases in %d attempts", len(out), n_bases, attempts)
    return out


# -- splits -------------------------------------------------------------------

def split_pairs(
    records: Sequence[BindingFactorial],
    calib_frac: float = 0.3,
    seed: int = 42,
) -> list[BindingFactorial]:
    """Assign calib/test by whole base, written into the data file.

    Note what is NOT split here: the two arms. Both arms of a base stay
    together, because the held-out *arm* is a fixed experimental factor rather
    than a sample — the alignment trains on `ab` and is tested on `ba` within
    the same test bases, so arm transfer and example generalization are not
    confounded.
    """
    bases = sorted({r.base_id for r in records})
    rng = random.Random(seed)
    rng.shuffle(bases)
    n_calib = max(1, int(round(calib_frac * len(bases)))) if bases else 0
    calib = set(bases[:n_calib])
    for record in records:
        record.split = "calib" if record.base_id in calib else "test"
    return list(records)


def assert_disjoint(records: Sequence[BindingFactorial]) -> None:
    """No base in both splits. Unit tested against a deliberate leak."""
    by_split: dict[str, set[str]] = {}
    for record in records:
        by_split.setdefault(record.split, set()).add(record.base_id)
    overlap = by_split.get("calib", set()) & by_split.get("test", set())
    if overlap:
        raise AssertionError(
            f"{len(overlap)} bases appear in both splits: {sorted(overlap)[:5]}. "
            "An alignment fitted on calibration would be evaluated on states it "
            "was fitted to.")


def save_pairs(records: Sequence[BindingFactorial], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(path, mode="w") as writer:
        for record in records:
            writer.write(record.to_dict())
    return path


def load_pairs(path: str | Path) -> list[BindingFactorial]:
    with jsonlines.open(path) as reader:
        return [BindingFactorial.from_dict(obj) for obj in reader]


def resolve_pairs_path(model: str, pairs: str | Path | None = None) -> Path:
    """The E13 pair file for `model`, or an error naming the actual problem."""
    path = Path(pairs) if pairs else Path("data/synthetic") / f"binding_pairs_{model}.jsonl"
    if path.exists():
        return path
    named = path.stem[len("binding_pairs_"):] if path.stem.startswith("binding_pairs_") else ""
    hint = ""
    if named and named != model:
        hint = (f"\n  The path names model '{named}' but --model is '{model}'. "
                f"A leftover shell $MODEL is the usual cause (jobs/common.csh "
                f"defaults it to deepseek-coder-6.7b): omit --pairs.")
    available = sorted(p.name for p in Path("data/synthetic").glob("binding_pairs_*.jsonl"))
    raise FileNotFoundError(
        f"No E13 pair file at {path}.{hint}\n"
        f"  Available: {available or 'none — stage 100 has not run'}\n"
        f"  Generate:  python scripts/100_binding_pairs.py --model {model}")


def dataset_summary(records: Sequence[BindingFactorial]) -> dict:
    """What stage 100 prints and the manifest records."""
    distances = [r.metadata.get("use_minus_mutation", 0) for r in records]
    return {
        "n_bases": len(records),
        "n_programs": 4 * len(records),
        "splits": {s: sum(1 for r in records if r.split == s)
                   for s in sorted({r.split for r in records})},
        "min_use_minus_mutation": min(distances) if distances else 0,
        "n_tokens_mean": (sum(r.n_tokens for r in records) / len(records)) if records else 0,
        "distinct_value_pairs": len({(r.v_a, r.v_b) for r in records}),
    }
