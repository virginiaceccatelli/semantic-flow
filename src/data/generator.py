"""Synthetic code generation with controlled semantic structure.

Generates Python functions with known ground-truth semantic graphs,
including adversarial variants that test lexical vs semantic disambiguation:
  - renamed variables (same structure, different names)
  - shadowed identifiers (same name, different binding)
  - dead code insertion (name appears but is never used on the taint path)
  - semantically equivalent rewrites (different surface form, same semantics)
"""

from __future__ import annotations

import random
import string
import textwrap
from dataclasses import dataclass, field
from typing import Optional

from .dataset import ProbeExample


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
