"""Synthetic code generation with controlled semantic structure.

Generates Python functions with known ground-truth semantic graphs,
including adversarial variants that test lexical vs semantic disambiguation:
  - renamed variables (same structure, different names)
  - shadowed identifiers (same name, different binding)
  - dead code insertion (name appears but is never used on the taint path)
  - semantically equivalent rewrites (different surface form, same semantics)
  - minimal pairs for Phase 5 causal tests
  - behavioral tasks for Phase 4 t_latent / t_failure experiments
"""

from __future__ import annotations

import random
import string
import textwrap
from dataclasses import dataclass, field
from typing import Optional

from .dataset import ProbeExample


@dataclass
class MinimalPair:
    """A clean/corrupted pair differing by a single semantically critical edit.

    Used in Phase 5 to test whether restoring activations from the clean example
    into the corrupted one improves the model's answer.
    """
    pair_id: str
    clean: ProbeExample           # correct semantic state (e.g. sanitized taint path)
    corrupted: ProbeExample       # broken semantic state (e.g. sanitizer applied to wrong var)
    relation_type: str            # "taint" | "binding" | "guard"
    corruption_description: str   # human-readable description of what changed
    target_line: int              # line where the semantic decision is made (sink / use / op)


@dataclass
class BehavioralTask:
    """A cloze-style probing task that evaluates model behaviour, not just latent state.

    The model is presented with `code_prefix + prompt_suffix` and asked to continue
    with one of the `choices`. `correct_idx` is the index of the correct continuation.

    Used in Phase 4 to compute:
      t_latent  — prefix length at which the probe first fails to decode the relation
      t_failure — prefix length at which the model first picks the wrong choice
      lead_time = t_failure - t_latent
    """
    task_id: str
    task_type: str                # see BEHAVIORAL_TASK_TYPES
    code_prefix: str              # code context shown to model (growing prefix)
    prompt_suffix: str            # question stem appended after code_prefix
    choices: list[str]            # candidate completions (short tokens/words)
    correct_idx: int              # index into choices of the correct answer
    semantic_relation: str        # "taint" | "binding" | "def_use" | "control" | "guard"
    metadata: dict = field(default_factory=dict)


BEHAVIORAL_TASK_TYPES = [
    "next_variable",       # which variable should be used at this point?
    "taint_at_sink",       # is the value reaching this sink tainted?
    "branch_reachable",    # which branch executes for this concrete input?
    "return_value",        # what does this function return?
    "def_reaches_use",     # which definition reaches this use?
    "guard_dominates",     # does this guard ensure safety at the operation below?
    "constrained_api",     # what is the correct argument to this call?
]


@dataclass
class SyntheticSpec:
    """Specification for a synthetic code example."""

    n_vars: int = 3                         # number of variables
    n_statements: int = 5                   # number of statements
    has_shadow: bool = False                # include a shadowed variable
    has_dead_def: bool = False              # include a definition never used
    has_decoy_name: bool = False            # rename a variable to match another
    chain_length: int = 2                   # def-use chain depth
    has_branch: bool = False                # include conditional
    has_taint: bool = False                 # include a source→sink path
    seed: Optional[int] = None


class SyntheticCodeGenerator:
    """Generate synthetic Python functions with annotated semantic structure."""

    SAFE_NAMES = list("abcdefghijklmnopqrstuvwxyz")
    SOURCES = ["input()", "request.GET.get('q')", "os.environ.get('VAR')", "sys.argv[1]"]
    SINKS = ["eval({})", "exec({})", "os.system({})", "subprocess.call({})"]
    SANITIZERS = ["html.escape({})", "shlex.quote({})"]

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def _fresh_name(self, existing: set[str]) -> str:
        for c in self.SAFE_NAMES:
            if c not in existing:
                return c
        # Fall back to random 2-char names
        while True:
            name = self.rng.choice(string.ascii_lowercase) + self.rng.choice(string.ascii_lowercase)
            if name not in existing:
                return name

    def generate_binding(self, spec: Optional[SyntheticSpec] = None) -> ProbeExample:
        """Generate a function with clear def-use binding structure."""
        spec = spec or SyntheticSpec()
        rng = random.Random(spec.seed) if spec.seed is not None else self.rng

        used_names: set[str] = set()
        vars_ = []
        for _ in range(spec.n_vars):
            name = self._fresh_name(used_names)
            used_names.add(name)
            vars_.append(name)

        lines = ["def func():"]
        # Initial definitions
        for v in vars_:
            val = rng.randint(0, 100)
            lines.append(f"    {v} = {val}")

        # Chain of uses
        for i in range(spec.chain_length):
            v_src = vars_[i % len(vars_)]
            v_dst = vars_[(i + 1) % len(vars_)]
            lines.append(f"    {v_dst} = {v_dst} + {v_src}")

        # Dead definition (if requested)
        if spec.has_dead_def:
            dead_name = self._fresh_name(used_names)
            lines.append(f"    {dead_name} = 999")

        # Shadow (if requested)
        if spec.has_shadow and vars_:
            shadow_name = vars_[0]
            lines.append(f"    {shadow_name} = 0  # shadows earlier def")

        lines.append(f"    return {vars_[-1]}")
        source = "\n".join(lines)

        return ProbeExample(
            example_id=f"synthetic_binding_{rng.randint(0, 999999)}",
            source=source,
            metadata={"type": "binding", "vars": vars_, "spec": vars(spec)},
        )

    def generate_taint(
        self,
        sanitized: bool = False,
        chain_length: int = 2,
        seed: Optional[int] = None,
    ) -> ProbeExample:
        """Generate a function with a taint source → sink path.

        If sanitized=False, the taint reaches the sink (vulnerable).
        If sanitized=True, a sanitizer is applied (safe).
        """
        rng = random.Random(seed) if seed is not None else self.rng
        source_expr = rng.choice(self.SOURCES)
        sink_tmpl = rng.choice(self.SINKS)
        sanitizer_tmpl = rng.choice(self.SANITIZERS)

        lines = ["def func():"]
        lines.append(f"    x = {source_expr}")

        # Propagation chain
        prev = "x"
        for i in range(chain_length):
            nxt = f"v{i}"
            lines.append(f"    {nxt} = {prev}")
            prev = nxt

        if sanitized:
            lines.append(f"    safe = {sanitizer_tmpl.format(prev)}")
            lines.append(f"    {sink_tmpl.format('safe')}")
        else:
            lines.append(f"    {sink_tmpl.format(prev)}")

        source = "\n".join(lines)
        return ProbeExample(
            example_id=f"synthetic_taint_{int(sanitized)}_{rng.randint(0, 999999)}",
            source=source,
            label=0 if sanitized else 1,
            metadata={
                "type": "taint",
                "sanitized": sanitized,
                "chain_length": chain_length,
                "source_expr": source_expr,
                "sink": sink_tmpl,
            },
        )

    def generate_shadow(self) -> ProbeExample:
        """Generate a function where two occurrences of the same name have different bindings.

        Key adversarial case: a probe relying on lexical identity will falsely predict
        same binding; a probe tracking semantic structure should distinguish them.
        """
        source = textwrap.dedent("""\
            def func(x):
                result = x * 2
                if result > 10:
                    x = result - 5   # shadows parameter x
                    result = x + 1   # this 'x' refers to the reassigned x
                return result
        """)
        return ProbeExample(
            example_id="synthetic_shadow_0",
            source=source,
            metadata={"type": "shadow", "shadowed_var": "x"},
        )

    def generate_renamed(self, original: ProbeExample, rename_map: dict[str, str]) -> ProbeExample:
        """Produce a renamed copy of a function with identical semantics but different names.

        Used to test whether probes track meaning or surface form.
        """
        new_source = original.source
        for old, new in rename_map.items():
            new_source = new_source.replace(old, new)
        return ProbeExample(
            example_id=original.example_id + "_renamed",
            source=new_source,
            label=original.label,
            metadata={**original.metadata, "type": "renamed", "rename_map": rename_map},
        )

    def generate_batch(
        self,
        n_binding: int = 100,
        n_taint: int = 100,
        n_shadow: int = 50,
    ) -> list[ProbeExample]:
        examples = []

        for i in range(n_binding):
            spec = SyntheticSpec(
                n_vars=self.rng.randint(2, 5),
                chain_length=self.rng.randint(1, 4),
                has_dead_def=self.rng.random() < 0.3,
                has_shadow=self.rng.random() < 0.2,
                seed=i,
            )
            examples.append(self.generate_binding(spec))

        for i in range(n_taint // 2):
            examples.append(self.generate_taint(sanitized=False, chain_length=self.rng.randint(1, 4), seed=i))
            examples.append(self.generate_taint(sanitized=True, chain_length=self.rng.randint(1, 4), seed=i))

        for _ in range(n_shadow):
            examples.append(self.generate_shadow())

        self.rng.shuffle(examples)
        return examples

    def generate_minimal_pair(
        self,
        pair_id: str = "pair_0",
        chain_length: int = 2,
        seed: Optional[int] = None,
    ) -> "MinimalPair":
        """Generate a clean/corrupted taint pair differing by a single sanitizer placement.

        Clean  : sanitizer applied to the tainted variable before the sink.
        Corrupted: sanitizer applied to a different (irrelevant) variable;
                   the tainted value still reaches the sink.
        """
        rng = random.Random(seed) if seed is not None else self.rng
        source_expr = rng.choice(self.SOURCES)
        sink_tmpl = rng.choice(self.SINKS)
        sanitizer_tmpl = rng.choice(self.SANITIZERS)

        # Propagation chain: x → v0 → v1 → ...
        chain_vars = ["x"] + [f"v{i}" for i in range(chain_length)]
        sink_var = chain_vars[-1]

        def _chain_lines() -> list[str]:
            lines = [f"    x = {source_expr}"]
            for i in range(chain_length):
                lines.append(f"    {chain_vars[i + 1]} = {chain_vars[i]}")
            return lines

        # Clean: sanitize the sink variable
        clean_lines = ["def func():"] + _chain_lines()
        clean_lines.append(f"    safe = {sanitizer_tmpl.format(sink_var)}")
        clean_lines.append(f"    {sink_tmpl.format('safe')}")
        clean_source = "\n".join(clean_lines)

        # Corrupted: sanitize a decoy variable instead
        corrupted_lines = ["def func():"] + _chain_lines()
        corrupted_lines.append(f"    _decoy = {sanitizer_tmpl.format('x')}")
        corrupted_lines.append(f"    {sink_tmpl.format(sink_var)}")
        corrupted_source = "\n".join(corrupted_lines)

        target_line = len(clean_lines) - 1  # sink line index (0-based)

        clean_ex = ProbeExample(
            example_id=f"{pair_id}_clean",
            source=clean_source,
            label=0,
            metadata={"type": "taint", "sanitized": True, "pair_id": pair_id},
        )
        corrupted_ex = ProbeExample(
            example_id=f"{pair_id}_corrupted",
            source=corrupted_source,
            label=1,
            metadata={"type": "taint", "sanitized": False, "pair_id": pair_id},
        )
        return MinimalPair(
            pair_id=pair_id,
            clean=clean_ex,
            corrupted=corrupted_ex,
            relation_type="taint",
            corruption_description=(
                f"Sanitizer applied to decoy variable 'x' instead of sink variable "
                f"'{sink_var}'; taint reaches sink unchanged."
            ),
            target_line=target_line,
        )

    def generate_behavioral_task(
        self,
        task_type: str = "taint_at_sink",
        seed: Optional[int] = None,
    ) -> "BehavioralTask":
        """Generate a cloze-style behavioral task for Phase 4.

        Returns a BehavioralTask where the model must predict the correct
        continuation given a code prefix.
        """
        rng = random.Random(seed) if seed is not None else self.rng
        task_id = f"{task_type}_{rng.randint(0, 999999)}"

        if task_type == "taint_at_sink":
            source_expr = rng.choice(self.SOURCES)
            sink_tmpl = rng.choice(self.SINKS)
            sanitized = rng.random() < 0.5
            chain_length = rng.randint(1, 3)
            chain_vars = ["x"] + [f"v{i}" for i in range(chain_length)]
            sink_var = chain_vars[-1]

            lines = [f"x = {source_expr}"]
            for i in range(chain_length):
                lines.append(f"{chain_vars[i + 1]} = {chain_vars[i]}")
            if sanitized:
                lines.append(f"safe = {rng.choice(self.SANITIZERS).format(sink_var)}")
                lines.append(f"# Is the value passed to sink tainted?")
                correct_idx = 1  # "no"
            else:
                lines.append(f"# Is the value passed to sink tainted?")
                correct_idx = 0  # "yes"

            return BehavioralTask(
                task_id=task_id,
                task_type=task_type,
                code_prefix="\n".join(lines),
                prompt_suffix=f"\n# Answer (yes/no): ",
                choices=["yes", "no"],
                correct_idx=correct_idx,
                semantic_relation="taint",
                metadata={"sanitized": sanitized, "chain_length": chain_length},
            )

        elif task_type == "next_variable":
            n_vars = rng.randint(2, 4)
            used: set[str] = set()
            vars_ = []
            for _ in range(n_vars):
                name = self._fresh_name(used)
                used.add(name)
                vars_.append(name)
            target = vars_[-1]
            correct_idx = 0
            distractors = [v for v in vars_ if v != target][:1]
            choices = [target] + distractors

            lines = []
            for v in vars_[:-1]:
                lines.append(f"{v} = {rng.randint(0, 100)}")
            lines.append(f"{target} = {vars_[-2]} + {rng.randint(1, 10)}")
            lines.append(f"result = ")

            return BehavioralTask(
                task_id=task_id,
                task_type=task_type,
                code_prefix="\n".join(lines),
                prompt_suffix="# Complete: result = ?",
                choices=choices,
                correct_idx=correct_idx,
                semantic_relation="binding",
            )

        elif task_type == "def_reaches_use":
            var = self._fresh_name(set())
            val_a = rng.randint(1, 50)
            val_b = rng.randint(51, 100)
            use_first = rng.random() < 0.5
            if use_first:
                lines = [f"{var} = {val_a}", f"result = {var}", f"{var} = {val_b}"]
                correct_idx = 0
            else:
                lines = [f"{var} = {val_a}", f"{var} = {val_b}", f"result = {var}"]
                correct_idx = 1

            return BehavioralTask(
                task_id=task_id,
                task_type=task_type,
                code_prefix="\n".join(lines),
                prompt_suffix=f"\n# Which value of {var} reaches `result`?",
                choices=[str(val_a), str(val_b)],
                correct_idx=correct_idx,
                semantic_relation="def_use",
            )

        else:
            # Fallback: simple return_value task
            var = self._fresh_name(set())
            val = rng.randint(0, 100)
            wrong = val + rng.randint(1, 10)
            lines = [f"def func():", f"    {var} = {val}", f"    return {var}"]
            return BehavioralTask(
                task_id=task_id,
                task_type="return_value",
                code_prefix="\n".join(lines),
                prompt_suffix="\n# func() returns: ",
                choices=[str(val), str(wrong)],
                correct_idx=0,
                semantic_relation="binding",
            )

    def generate_behavioral_batch(
        self,
        n_per_type: int = 20,
        seed: int = 42,
    ) -> list["BehavioralTask"]:
        """Generate a mix of behavioral tasks for Phase 4."""
        rng = random.Random(seed)
        implemented = ["taint_at_sink", "next_variable", "def_reaches_use", "return_value"]
        tasks = []
        for i, ttype in enumerate(implemented * n_per_type):
            tasks.append(self.generate_behavioral_task(task_type=ttype, seed=rng.randint(0, 999999)))
        rng.shuffle(tasks)
        return tasks

    def generate_minimal_pair_batch(
        self,
        n: int = 20,
        seed: int = 42,
    ) -> list["MinimalPair"]:
        """Generate a batch of minimal pairs for Phase 5."""
        rng = random.Random(seed)
        return [
            self.generate_minimal_pair(
                pair_id=f"pair_{i}",
                chain_length=rng.randint(1, 3),
                seed=rng.randint(0, 999999),
            )
            for i in range(n)
        ]
