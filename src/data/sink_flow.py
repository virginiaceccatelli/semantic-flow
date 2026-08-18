"""E15 data: a controlled source → sensitive-sink benchmark (the audit corpus).

E9 asks whether *binding* and *def-use* survive the obfuscation ladder. E15 asks
the security question the ladder was built for: **does the value that reaches a
code-bearing, security-sensitive argument come from untrusted input — and does a
frozen readout of that fact survive obfuscation?**

Three sink families, four flow structures, twenty base seeds, two labels:

    3 x 4 x 20 x 2 = 480 clean programs

    command execution   untrusted request/CLI input  -> os.system / subprocess(shell)
    SQL execution       untrusted request input      -> cursor.execute
    dynamic execution   untrusted request/stdin      -> eval / exec

**No sanitizers.** The generic sanitizer list in `generator.py`
(`html.escape`, `shlex.quote`) is deliberately not reused: `html.escape` before
`exec` and `shlex.quote` before `eval` are not security mitigations, and a
benchmark whose "safe" class is built from them would label a vulnerable program
safe. The safe member here carries an **independently trusted value** — a
literal that never touches the source — and reaches the same sink through the
same propagation code.

**The pair.** Every base seed yields a matched unsafe/safe pair holding the same
source, the same propagation, the same trusted alternative and the same sink;
the two members differ **only in the sink-argument span**, which is checked
character-exactly (`pair_diff_is_confined_to_sink_arg`) rather than asserted.
Which of the two chain names is the tainted one alternates with the base index,
so the anchor token identity is uninformative about the label across the corpus
and the measured surface baseline is not handed the answer for free.

**Ground truth is never taken from the generator.** Two independent readings
must agree with each other and with the intended label, in the discipline
`store_semantics.py` established for E12/E13:

  * `observe_program` — execute the program under **stubs**, with a provenance
    -carrying string standing in for the untrusted input. `os.system`,
    `subprocess.*`, `cursor.execute`, `eval` and `exec` are recorders; the
    module runs with `__builtins__ = {}`, so no dangerous API is reachable even
    in principle. This reading is flow-sensitive because it *is* the execution.
  * `static_sink_label` — a flow-insensitive taint fixpoint over the AST,
    written against call shapes (attribute paths, not variable names) so it
    still reads a renamed, flattened variant. Interprocedural to one level via
    per-function parameter->return summaries, which is what the `helper`
    structure needs.

Being flow-insensitive, the static reading is an over-approximation: it would
call a program unsafe if *any* assignment could carry the source to the sink
variable. That is exactly why it is paired with execution rather than trusted
alone — the two disagree precisely on the programs whose label nobody could
defend, and those are refused (see `recover_label`).

Obfuscation is E9's ladder, unchanged and unextended
(`src/data/obfuscation.py::ObfuscationLadder`), applied to **held-out programs
only**, with label preservation verified per variant by the same two readings.
"""

from __future__ import annotations

import ast
import hashlib
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

import jsonlines

from src.data.alignment import char_span_to_tokens, line_col_to_char
from src.data.dataset import ProbeExample
from src.data.obfuscation import OBFUSCATION_LEVELS, ObfuscationLadder

logger = logging.getLogger(__name__)

# ── the design constants (mirrored in configs/experiments.yaml) ──────────────

FAMILIES: tuple[str, ...] = ("command_exec", "sql_exec", "dynamic_exec")
STRUCTURES: tuple[str, ...] = ("direct", "assign_chain", "branch_merge", "helper")
ROLES: tuple[str, ...] = ("unsafe", "safe")

N_BASE_SEEDS = 20          # base seeds per (family, structure)
N_TRAIN_SEEDS = 14         # of those, used for clean probe training
N_HELDOUT_SEEDS = 6        # held out, and the only ones ever obfuscated

OBF_LEVELS: tuple[int, ...] = tuple(level for level, _ in OBFUSCATION_LEVELS)
OBF_NAMES: dict[int, str] = {level: name for level, name in OBFUSCATION_LEVELS}


# ── conditions: the atomic arms and the cumulative ladder ────────────────────
#
# The first E15 run evaluated the cumulative ladder only, so its level-4 result
# ("flattening breaks the readout") was a *marginal* claim: level 4 contains
# renaming, opaque predicates and MBA encoding as well as the dispatch loop, and
# nothing in the data could say which of the four did the damage. The atomic
# arms fix exactly that and nothing else — no new transformation is added, and
# no arbitrary pairwise combination is generated. Each atomic condition applies
# exactly ONE of the ladder's existing rewrites to the clean program; each
# cumulative condition applies exactly the declared prefix.
#
#   atomic      what does this transformation do ON ITS OWN?
#   cumulative  what does an adversary who composes them achieve?
#   difference  the interaction: cumulative minus the atomic that is in it
#
# `rename_only` and `rename_cumulative` apply the identical transformation set
# (the cumulative prefix of length one *is* renaming). They are kept as two
# conditions because their transformation draws are independent, which makes
# their difference a measured DRAW-NOISE FLOOR for every other row of the
# atomic-vs-cumulative table: an interaction smaller than that row is not an
# interaction. This is the one row of that table which is a control by
# construction, and the report says so.

@dataclass(frozen=True)
class Condition:
    """One evaluated condition: a named subset of the ladder's transformations."""

    name: str
    steps: tuple[str, ...]          # names from obfuscation.STEP_ORDER
    kind: str                       # clean | baseline | atomic | cumulative
    order: int                      # sort key, and the value of `obf_level`
    legacy_level: Optional[int] = None   # the old ladder level it reproduces
    predecessor: Optional[str] = None    # for the marginal cumulative delta
    atomic_counterpart: Optional[str] = None  # for the interaction column

    @property
    def n_steps(self) -> int:
        return len(self.steps)

    @property
    def is_transformed(self) -> bool:
        return self.kind != "clean"


CLEAN_CONDITION = Condition(
    name="clean_heldout", steps=(), kind="clean", order=-1)

CONDITIONS: tuple[Condition, ...] = (
    CLEAN_CONDITION,
    # the shared-formatting baseline: an ast round-trip and nothing else, so
    # unparse artifacts are never confounded with a transformation
    Condition("normalize", (), "baseline", 0, legacy_level=0),
    # atomic: exactly one transformation applied to the clean program
    Condition("rename_only", ("rename",), "atomic", 11),
    Condition("opaque_only", ("opaque",), "atomic", 12),
    Condition("encode_only", ("encode",), "atomic", 13),
    Condition("flatten_only", ("flatten",), "atomic", 14),
    # cumulative: exactly the declared prefix of the ladder
    Condition("rename_cumulative", ("rename",), "cumulative", 21,
              legacy_level=1, predecessor="normalize",
              atomic_counterpart="rename_only"),
    Condition("rename_opaque", ("rename", "opaque"), "cumulative", 22,
              legacy_level=2, predecessor="rename_cumulative",
              atomic_counterpart="opaque_only"),
    Condition("rename_opaque_encode", ("rename", "opaque", "encode"),
              "cumulative", 23, legacy_level=3, predecessor="rename_opaque",
              atomic_counterpart="encode_only"),
    Condition("rename_opaque_encode_flatten",
              ("rename", "opaque", "encode", "flatten"), "cumulative", 24,
              legacy_level=4, predecessor="rename_opaque_encode",
              atomic_counterpart="flatten_only"),
)

CONDITIONS_BY_NAME: dict[str, Condition] = {c.name: c for c in CONDITIONS}
TRANSFORMED_CONDITIONS: tuple[Condition, ...] = tuple(
    c for c in CONDITIONS if c.is_transformed)
ATOMIC_CONDITIONS: tuple[str, ...] = tuple(
    c.name for c in CONDITIONS if c.kind == "atomic")
CUMULATIVE_CONDITIONS: tuple[str, ...] = tuple(
    c.name for c in CONDITIONS if c.kind == "cumulative")
DEFAULT_CONDITIONS: tuple[str, ...] = tuple(c.name for c in TRANSFORMED_CONDITIONS)

# What the old five-level ladder called each condition. Result CSVs written
# before the atomic arms existed carry `obf_level` 0-4; this is how a reader
# (and stage 124) maps one onto the other without rerunning anything.
LEGACY_LEVEL_TO_CONDITION: dict[int, str] = {
    c.legacy_level: c.name for c in CONDITIONS if c.legacy_level is not None}


def condition_for(name: str) -> Condition:
    """The condition of that name, or an error listing the ones that exist."""
    try:
        return CONDITIONS_BY_NAME[name]
    except KeyError:
        raise ValueError(
            f"unknown E15 condition '{name}'. Known: "
            f"{sorted(CONDITIONS_BY_NAME)}") from None


def resolve_conditions(names: Sequence[str] = DEFAULT_CONDITIONS) -> list[Condition]:
    """Named conditions in canonical order, deduplicated, clean excluded.

    The clean held-out shard is not a transformation and is never generated as
    a variant, so asking for it here is a mistake worth naming.
    """
    wanted = []
    for name in names:
        condition = condition_for(name)
        if condition.kind == "clean":
            raise ValueError(
                "'clean_heldout' is the untransformed held-out shard, not a "
                "variant condition; it is evaluated from the heldout shard")
        if condition not in wanted:
            wanted.append(condition)
    return sorted(wanted, key=lambda c: c.order)


def expected_variant_count(n_heldout_programs: int,
                           conditions: Sequence[Condition]) -> int:
    """Exactly one variant per (held-out program, condition). No exceptions."""
    return n_heldout_programs * len(conditions)


def expected_clean_programs(
    families: Sequence[str] = FAMILIES,
    structures: Sequence[str] = STRUCTURES,
    n_seeds: int = N_BASE_SEEDS,
) -> int:
    """3 x 4 x 20 x 2 = 480 under the canonical configuration."""
    return len(families) * len(structures) * n_seeds * len(ROLES)


# ── family definitions ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class SinkFamily:
    """One sink family: where untrusted data comes from, and where it must not go."""

    name: str
    description: str
    sources: tuple[str, ...]          # untrusted-input expressions
    alt_sources: tuple[str, ...]      # a second untrusted expression (branch/merge)
    trusted: tuple[str, ...]          # independently trusted literals
    alt_trusted: tuple[str, ...]
    sinks: tuple[str, ...]            # "{}"-templated, sensitive argument first
    sink_names: tuple[str, ...]       # canonical name of each sink, same order


FAMILY_SPECS: dict[str, SinkFamily] = {
    "command_exec": SinkFamily(
        name="command_exec",
        description="untrusted request/CLI input reaching a shell command string",
        sources=('request.args.get("cmd")', 'request.args.get("target")'),
        alt_sources=("sys.argv[1]", 'request.form.get("cmd")'),
        trusted=('"systemctl status"', '"uptime"'),
        alt_trusted=('"df -h"', '"free -m"'),
        sinks=("os.system({})", "subprocess.call({}, shell=True)"),
        sink_names=("os.system", "subprocess.call"),
    ),
    "sql_exec": SinkFamily(
        name="sql_exec",
        description="untrusted request input reaching the SQL text of cursor.execute",
        sources=('request.args.get("uid")', 'request.args.get("name")'),
        alt_sources=('request.form.get("uid")', 'request.form.get("name")'),
        trusted=('"SELECT id FROM users WHERE id = 1"', '"SELECT count(*) FROM users"'),
        alt_trusted=('"SELECT id FROM users LIMIT 1"', '"SELECT 1"'),
        sinks=("cursor.execute({})",),
        sink_names=("cursor.execute",),
    ),
    "dynamic_exec": SinkFamily(
        name="dynamic_exec",
        description="untrusted request/stdin input reaching eval or exec",
        sources=('request.form.get("expr")', 'request.args.get("expr")'),
        alt_sources=("input()", 'request.form.get("code")'),
        trusted=('"2 * 3"', '"7 + 1"'),
        alt_trusted=('"4 - 2"', '"9 // 3"'),
        sinks=("eval({})", "exec({})"),
        sink_names=("eval", "exec"),
    ),
}

# Chain-variable names. Two are drawn per base; which one carries the tainted
# chain alternates with the base index, so "this identifier means tainted" is
# uninformative across the corpus.
CHAIN_NAMES: tuple[str, ...] = (
    "value", "entry", "param", "field", "item", "chunk",
    "record", "handle", "buffer", "detail", "packet", "segment",
)

HELPER_NAME = "relay"
HELPER_PARAM = "carrier"
COUNTER_NAME = "count"          # an int local: what the opaque/MBA levels bite on


# ── program records ──────────────────────────────────────────────────────────


@dataclass
class FlowProgram:
    """One program: a member of a matched pair, or an obfuscated variant of one."""

    program_id: str
    base_id: str
    family: str
    structure: str
    role: str                    # "unsafe" | "safe"
    label: int                   # 1 = the sink argument is source-derived
    split: str                   # "train" | "heldout"
    source: str
    obf_level: int = -1          # -1 = clean
    obf_name: str = "clean"
    metadata: dict = field(default_factory=dict)

    def to_example(self) -> ProbeExample:
        """The standard `ProbeExample` the stage-10 extraction contract takes."""
        return ProbeExample(
            example_id=self.program_id,
            source=self.source,
            label=self.label,
            metadata={
                "type": "sinkflow",
                "base_id": self.base_id,
                "pair_id": self.base_id,
                "family": self.family,
                "structure": self.structure,
                "role": self.role,
                "label": self.label,
                "split": self.split,
                "obf_level": self.obf_level,
                "obf_name": self.obf_name,
                **self.metadata,
            },
        )

    def to_dict(self) -> dict:
        return dict(self.__dict__)

    @classmethod
    def from_example(cls, example: ProbeExample) -> "FlowProgram":
        md = dict(example.metadata)
        return cls(
            program_id=example.example_id,
            base_id=md["base_id"],
            family=md["family"],
            structure=md["structure"],
            role=md["role"],
            label=int(md["label"]),
            split=md["split"],
            source=example.source,
            obf_level=int(md.get("obf_level", -1)),
            obf_name=md.get("obf_name", "clean"),
            metadata={k: v for k, v in md.items()
                      if k not in {"type", "base_id", "pair_id", "family", "structure",
                                   "role", "label", "split", "obf_level", "obf_name"}},
        )


@dataclass
class SinkFlowBase:
    """One base seed: the matched unsafe/safe pair and what they share."""

    base_id: str
    family: str
    structure: str
    seed: int
    split: str
    unsafe: FlowProgram
    safe: FlowProgram
    metadata: dict = field(default_factory=dict)

    def programs(self) -> list[FlowProgram]:
        return [self.unsafe, self.safe]


# ── reading 1: instrumented execution under stubs ────────────────────────────


class Tainted(str):
    """A string carrying source provenance through the propagation we generate.

    A `str` subclass rather than a wrapper so that a program which merely moves
    the value around behaves identically to one holding a plain string; `+` is
    overridden because taint that vanished on concatenation would silently
    relabel a vulnerable program safe.
    """

    __slots__ = ()

    def __add__(self, other):
        return Tainted(str.__add__(self, other))

    def __radd__(self, other):
        return Tainted(str(other) + str(self))


@dataclass
class SinkCall:
    name: str
    value: str
    tainted: bool


@dataclass
class Observation:
    """What executing a program under stubs revealed about its sink."""

    ok: bool
    sink: str = ""
    value: str = ""
    tainted: bool = False
    n_sink_calls: int = 0
    error: str = ""

    @property
    def label(self) -> Optional[int]:
        return int(self.tainted) if self.ok else None

    def key(self) -> tuple:
        """What must be identical between a base and an obfuscated variant."""
        return (self.ok, self.sink, self.value, self.tainted, self.n_sink_calls)


class _Params:
    """`request.args` / `request.form` / `request.GET` — every read is untrusted."""

    def get(self, key, default=None):
        return Tainted(f"<untrusted:{key}>")

    def __getitem__(self, key):
        return self.get(key)


class _Request:
    def __init__(self):
        self.args = _Params()
        self.form = _Params()
        self.GET = _Params()
        self.POST = _Params()
        self.cookies = _Params()
        self.headers = _Params()


def _stub_namespace() -> tuple[dict, list[SinkCall]]:
    """Globals for executing a generated program with every dangerous API stubbed.

    `__builtins__` is emptied, so `eval`/`exec` inside the program can only
    resolve to the recorders installed here: the benchmark's sinks are never
    executed, in any obfuscation level, even if a generated program were wrong.
    """
    calls: list[SinkCall] = []

    def recorder(name: str):
        def sink(argument, *rest, **kwargs):
            calls.append(SinkCall(name=name, value=str(argument),
                                  tainted=isinstance(argument, Tainted)))
            return None
        return sink

    class _Namespace:
        def __init__(self, **members):
            self.__dict__.update(members)

    namespace = {
        "__builtins__": {},
        "request": _Request(),
        "os": _Namespace(system=recorder("os.system"), popen=recorder("os.popen")),
        "subprocess": _Namespace(call=recorder("subprocess.call"),
                                 run=recorder("subprocess.run"),
                                 Popen=recorder("subprocess.Popen")),
        "cursor": _Namespace(execute=recorder("cursor.execute")),
        "sys": _Namespace(argv=["prog", Tainted("<untrusted:argv1>")]),
        "input": lambda *a: Tainted("<untrusted:stdin>"),
        "eval": recorder("eval"),
        "exec": recorder("exec"),
    }
    return namespace, calls


def observe_program(source: str, entry: str = "func") -> Observation:
    """Execute `source` under stubs and report what reached the sink.

    Only sources this repository generated are ever passed here, and they are
    executed with no builtins and with every sensitive API replaced by a
    recorder — the standard the E9 ladder already uses for its equivalence
    check, tightened because these programs are *about* dangerous calls.
    """
    namespace, calls = _stub_namespace()
    try:
        exec(compile(source, "<sinkflow>", "exec"), namespace)   # noqa: S102
        namespace[entry](namespace["request"])
    except Exception as exc:                                     # noqa: BLE001
        return Observation(ok=False, error=f"{type(exc).__name__}: {exc}")
    if not calls:
        return Observation(ok=False, error="no sink call was reached")
    last = calls[-1]
    return Observation(ok=True, sink=last.name, value=last.value,
                       tainted=last.tainted, n_sink_calls=len(calls))


# ── reading 2: a flow-insensitive taint fixpoint over the AST ────────────────

_SOURCE_ATTRS = {"args", "form", "GET", "POST", "cookies", "headers"}
_SINK_ATTRS = {("os", "system"), ("os", "popen"), ("subprocess", "call"),
               ("subprocess", "run"), ("subprocess", "Popen"), ("cursor", "execute")}
_SINK_BARE = {"eval", "exec"}


def _attr_path(node: ast.AST) -> Optional[tuple[str, ...]]:
    """`os.system` -> ("os", "system"); None for anything not a plain dotted name."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return tuple(reversed(parts))
    return None


def is_source_expr(node: ast.AST) -> bool:
    """An untrusted-input expression, recognised by shape rather than by name.

    Alpha-renaming (obfuscation level 1) rewrites locals, including the request
    parameter, so `request.args.get(...)` may arrive as `bq3.args.get(...)`.
    The attribute chain and the builtin names survive, and that is what is
    matched here.
    """
    if isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "get" \
                and isinstance(func.value, ast.Attribute) \
                and func.value.attr in _SOURCE_ATTRS:
            return True
        if isinstance(func, ast.Name) and func.id == "input":
            return True
    if isinstance(node, ast.Subscript):
        path = _attr_path(node.value)
        if path is not None and path[-1] == "argv":
            return True
    return False


def sink_name_of(node: ast.AST) -> Optional[str]:
    """The canonical sink name of a call node, or None if it is not a sink."""
    if not isinstance(node, ast.Call):
        return None
    if isinstance(node.func, ast.Name) and node.func.id in _SINK_BARE:
        return node.func.id
    path = _attr_path(node.func)
    if path is not None and len(path) >= 2 and tuple(path[-2:]) in _SINK_ATTRS:
        return ".".join(path[-2:])
    return None


def find_sink_call(tree: ast.AST) -> ast.Call:
    """The one sensitive call in the program; an error if there is not exactly one."""
    hits = [n for n in ast.walk(tree) if sink_name_of(n) is not None]
    if len(hits) != 1:
        raise ValueError(f"expected exactly one sensitive sink call, found {len(hits)}")
    call = hits[0]
    if not call.args:
        raise ValueError("the sink call has no sensitive argument")
    return call


def _function_defs(tree: ast.Module) -> list[ast.FunctionDef]:
    return [n for n in tree.body if isinstance(n, ast.FunctionDef)]


def _summaries(tree: ast.Module) -> dict[str, list[bool]]:
    """For each module-level function, which parameters can reach its return value.

    One fixpoint pass per parameter is enough for the depth this benchmark
    generates (a single helper boundary), and the outer loop re-runs until the
    summaries stop changing so a helper calling a helper would still resolve.
    """
    functions = {fn.name: fn for fn in _function_defs(tree)}
    summaries: dict[str, list[bool]] = {
        name: [False] * len(fn.args.args) for name, fn in functions.items()
    }
    for _ in range(len(functions) + 1):
        changed = False
        for name, fn in functions.items():
            for i, arg in enumerate(fn.args.args):
                reaches = _returns_taint(fn, {arg.arg}, summaries)
                if reaches != summaries[name][i]:
                    summaries[name][i] = reaches
                    changed = True
        if not changed:
            break
    return summaries


def _taint_fixpoint(fn: ast.FunctionDef, seeds: set[str],
                    summaries: dict[str, list[bool]]) -> set[str]:
    """Names in `fn` that can hold source-derived data (flow-insensitive)."""
    tainted = set(seeds)
    assigns = [n for n in ast.walk(fn) if isinstance(n, ast.Assign)]
    for _ in range(len(assigns) + 2):
        changed = False
        for node in assigns:
            if not _expr_tainted(node.value, tainted, summaries):
                continue
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id not in tainted:
                    tainted.add(target.id)
                    changed = True
        if not changed:
            break
    return tainted


def _returns_taint(fn: ast.FunctionDef, seeds: set[str],
                   summaries: dict[str, list[bool]]) -> bool:
    tainted = _taint_fixpoint(fn, seeds, summaries)
    return any(
        node.value is not None and _expr_tainted(node.value, tainted, summaries)
        for node in ast.walk(fn) if isinstance(node, ast.Return)
    )


def _expr_tainted(node: ast.AST, tainted: set[str],
                  summaries: dict[str, list[bool]]) -> bool:
    if is_source_expr(node):
        return True
    if isinstance(node, ast.Name):
        return node.id in tainted
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id in summaries:
            reaches = summaries[node.func.id]
            return any(reaches[i] and _expr_tainted(arg, tainted, summaries)
                       for i, arg in enumerate(node.args) if i < len(reaches))
        return any(_expr_tainted(arg, tainted, summaries) for arg in node.args)
    if isinstance(node, ast.BinOp):
        return (_expr_tainted(node.left, tainted, summaries)
                or _expr_tainted(node.right, tainted, summaries))
    if isinstance(node, ast.IfExp):
        return (_expr_tainted(node.body, tainted, summaries)
                or _expr_tainted(node.orelse, tainted, summaries))
    if isinstance(node, (ast.Attribute, ast.Subscript)):
        return _expr_tainted(node.value, tainted, summaries)
    if isinstance(node, (ast.JoinedStr, ast.FormattedValue)):
        return any(_expr_tainted(child, tainted, summaries)
                   for child in ast.iter_child_nodes(node))
    return False


def static_sink_label(source: str, entry: str = "func") -> int:
    """1 if the sink's sensitive argument can be source-derived, else 0.

    Deliberately flow-insensitive, so it reads a control-flow-flattened variant
    (level 4) the same way it reads the original: every assignment is a possible
    edge. The price is over-approximation, which is why `recover_label` requires
    it to agree with execution rather than believing it on its own.
    """
    tree = ast.parse(source)
    summaries = _summaries(tree)
    target = next((fn for fn in _function_defs(tree) if fn.name == entry), None)
    if target is None:
        raise ValueError(f"no function `{entry}` in source")
    tainted = _taint_fixpoint(target, set(), summaries)
    call = find_sink_call(target)
    return int(_expr_tainted(call.args[0], tainted, summaries))


def recover_label(source: str, entry: str = "func") -> int:
    """The label, recomputed from the program alone — never read from metadata.

    Both readings must agree. A disagreement is a program whose label nobody can
    defend, and it is refused rather than reconciled (the rule
    `store_semantics.cross_check` applies to E12/E13).
    """
    static = static_sink_label(source, entry=entry)
    observed = observe_program(source, entry=entry)
    if not observed.ok:
        raise ValueError(f"instrumented execution failed: {observed.error}")
    if int(observed.tainted) != static:
        raise ValueError(
            f"the two readings disagree: static taint analysis says "
            f"{static}, instrumented execution says {int(observed.tainted)}")
    return static


# ── anchors ──────────────────────────────────────────────────────────────────

ANCHOR_KINDS = ("source_call", "sink_call", "sink_arg")


def find_anchors(source: str, entry: str = "func") -> dict[str, tuple[int, int, int, int]]:
    """(line, col, end_line, end_col) spans of the source, the sink and its argument.

    Recomputed from whichever source is passed — a variant's anchors come from
    the variant, exactly as E5/E9 rebuild their ground truth per variant.
    """
    tree = ast.parse(source)
    target = next((fn for fn in _function_defs(tree) if fn.name == entry), None)
    if target is None:
        raise ValueError(f"no function `{entry}` in source")

    sources = [n for n in ast.walk(target) if is_source_expr(n)]
    if not sources:
        raise ValueError("no untrusted-source expression found")
    first = min(sources, key=lambda n: (n.lineno, n.col_offset))
    call = find_sink_call(target)
    argument = call.args[0]

    def span(node: ast.AST) -> tuple[int, int, int, int]:
        return (node.lineno, node.col_offset, node.end_lineno, node.end_col_offset)

    return {"source_call": span(first), "sink_call": span(call), "sink_arg": span(argument)}


def anchor_char_span(source: str, span: Sequence[int]) -> tuple[int, int]:
    line, col, end_line, end_col = span
    return (line_col_to_char(source, line, col), line_col_to_char(source, end_line, end_col))


def anchor_token_span(
    source: str,
    offsets: Sequence[tuple[int, int]],
    span: Sequence[int],
) -> Optional[list[int]]:
    """Token indices covering an anchor, **only if they cover it exactly**.

    "Exactly" means two things, and neither is negotiable for a probe that reads
    the last covering token (`alignment.AlignedEvent.anchor`):

      * the last covering token ends **exactly** where the AST span ends, so the
        anchor state integrates the whole span and nothing after it. A tokenizer
        that merged `value)` into one token would fold the call's closing
        syntax into the argument, and a probe there would be reading the merge;
      * no covering token starts inside the span with non-whitespace before it.
        Byte-BPE tokenizers universally absorb *leading whitespace* into the
        following piece (`" request"` is one token), which shifts a span's start
        by the indentation and nothing else; that is accepted, and anything else
        is refused.
    """
    start, end = anchor_char_span(source, span)
    hits = char_span_to_tokens(offsets, start, end)
    if not hits:
        return None
    if offsets[hits[-1]][1] != end:
        return None
    lead = source[offsets[hits[0]][0]:start]
    if lead.strip():
        return None
    return hits


# ── rendering ────────────────────────────────────────────────────────────────


def _propagation(structure: str, taint: str, trust: str, order_swap: bool,
                 alt_source: str, alt_trusted: str) -> tuple[list[str], str, str]:
    """Body lines for one flow structure, plus each chain's final variable."""
    def ordered(a: str, b: str) -> list[str]:
        return [b, a] if order_swap else [a, b]

    if structure == "direct":
        return [], taint, trust

    if structure == "assign_chain":
        lines = (ordered(f"{taint}_1 = {taint}", f"{trust}_1 = {trust}")
                 + ordered(f"{taint}_2 = {taint}_1", f"{trust}_2 = {trust}_1"))
        return [f"    {line}" for line in lines], f"{taint}_2", f"{trust}_2"

    if structure == "branch_merge":
        # Two definitions of each chain variable reach the sink; both carry the
        # same disposition, so the label is a property of the merge rather than
        # of a path a probe could guess from the guard.
        then_body = ordered(f"{taint}_1 = {taint}", f"{trust}_1 = {trust}")
        else_body = ordered(f"{taint}_1 = {alt_source}", f"{trust}_1 = {alt_trusted}")
        lines = ([f"    if {COUNTER_NAME} > 2:"]
                 + [f"        {line}" for line in then_body]
                 + ["    else:"]
                 + [f"        {line}" for line in else_body])
        return lines, f"{taint}_1", f"{trust}_1"

    if structure == "helper":
        lines = ordered(f"{taint}_1 = {HELPER_NAME}({taint})",
                        f"{trust}_1 = {HELPER_NAME}({trust})")
        return [f"    {line}" for line in lines], f"{taint}_1", f"{trust}_1"

    raise ValueError(f"unknown flow structure '{structure}'")


def render_pair(
    structure: str,
    taint_name: str,
    trust_name: str,
    source_expr: str,
    alt_source: str,
    trusted_expr: str,
    alt_trusted: str,
    sink_tmpl: str,
    order_swap: bool,
) -> tuple[str, str, str, str]:
    """The matched pair: (unsafe source, safe source, taint final, trust final).

    Both members hold the *same* source, propagation, trusted alternative and
    sink; only the sink argument differs.
    """
    head = [f"    {COUNTER_NAME} = 3"]
    heads = [f"{taint_name} = {source_expr}", f"{trust_name} = {trusted_expr}"]
    if order_swap:
        heads.reverse()
    head += [f"    {line}" for line in heads]

    body, taint_final, trust_final = _propagation(
        structure, taint_name, trust_name, order_swap, alt_source, alt_trusted)

    prelude = ([f"def {HELPER_NAME}({HELPER_PARAM}):",
                f"    return {HELPER_PARAM}", ""] if structure == "helper" else [])
    tail = [f"    {COUNTER_NAME} = {COUNTER_NAME} + 1"]

    def program(argument: str) -> str:
        lines = (prelude + ["def func(request):"] + head + body + tail
                 + [f"    {sink_tmpl.format(argument)}"])
        return "\n".join(lines) + "\n"

    return program(taint_final), program(trust_final), taint_final, trust_final


# ── pair invariants ──────────────────────────────────────────────────────────


def pair_diff_is_confined_to_sink_arg(unsafe: str, safe: str) -> tuple[bool, str]:
    """Do the two members differ *only* inside their sink-argument spans?

    Checked on characters, not on a claim: everything before each member's sink
    argument must be identical, and so must everything after it.
    """
    try:
        u_start, u_end = anchor_char_span(unsafe, find_anchors(unsafe)["sink_arg"])
        s_start, s_end = anchor_char_span(safe, find_anchors(safe)["sink_arg"])
    except Exception as exc:                                    # noqa: BLE001
        return False, f"anchors could not be resolved: {exc}"
    if unsafe[:u_start] != safe[:s_start]:
        return False, "the members differ before the sink argument"
    if unsafe[u_end:] != safe[s_end:]:
        return False, "the members differ after the sink argument"
    if unsafe[u_start:u_end] == safe[s_start:s_end]:
        return False, "the members do not differ at the sink argument at all"
    return True, (f"differ only at the sink argument: "
                  f"{unsafe[u_start:u_end]!r} vs {safe[s_start:s_end]!r}")


def pair_token_diff(unsafe: str, safe: str, tokenizer) -> dict:
    """Token-level view of the same invariant, for the record.

    Length matching is reported rather than required: the permitted span is the
    sink argument, and two identifiers of different token lengths still differ
    only there.
    """
    ids_u = tokenizer(unsafe, add_special_tokens=False)["input_ids"]
    ids_s = tokenizer(safe, add_special_tokens=False)["input_ids"]
    common = min(len(ids_u), len(ids_s))
    diffs = [i for i in range(common) if ids_u[i] != ids_s[i]]
    return {
        "n_tokens_unsafe": len(ids_u),
        "n_tokens_safe": len(ids_s),
        "token_length_matched": len(ids_u) == len(ids_s),
        "first_diff": diffs[0] if diffs else -1,
        "n_diff_tokens": len(diffs) + abs(len(ids_u) - len(ids_s)),
    }


# ── generation ───────────────────────────────────────────────────────────────


def _base_id(family: str, structure: str, index: int) -> str:
    return f"{family}_{structure}_{index:02d}"


def _build_base(
    family: str,
    structure: str,
    index: int,
    tokenizer,
    seed: int,
    max_attempts: int = 12,
) -> SinkFlowBase:
    """One base seed, or a ValueError naming the invariant that could not be met."""
    spec = FAMILY_SPECS[family]
    rng = random.Random(seed)
    base_id = _base_id(family, structure, index)
    # Which chain name is the tainted one, and which chain is declared first,
    # alternate on independent bits of the index: neither the anchor token nor
    # the declaration order carries the label across the corpus.
    role_swap = index % 2 == 1
    order_swap = (index // 2) % 2 == 1

    failures: list[str] = []
    for attempt in range(max_attempts):
        first, second = rng.sample(CHAIN_NAMES, 2)
        taint_name, trust_name = (second, first) if role_swap else (first, second)
        source_expr = spec.sources[attempt % len(spec.sources)]
        alt_source = spec.alt_sources[attempt % len(spec.alt_sources)]
        trusted_expr = spec.trusted[attempt % len(spec.trusted)]
        alt_trusted = spec.alt_trusted[attempt % len(spec.alt_trusted)]
        sink_pos = index % len(spec.sinks)
        sink_tmpl, sink_name = spec.sinks[sink_pos], spec.sink_names[sink_pos]

        unsafe_src, safe_src, taint_final, trust_final = render_pair(
            structure, taint_name, trust_name, source_expr, alt_source,
            trusted_expr, alt_trusted, sink_tmpl, order_swap)

        try:
            for src, expected in ((unsafe_src, 1), (safe_src, 0)):
                ast.parse(src)
                if recover_label(src) != expected:
                    raise ValueError(
                        f"the independent readings recovered "
                        f"{recover_label(src)} where the template intends {expected}")
        except Exception as exc:                                # noqa: BLE001
            failures.append(f"attempt {attempt}: {exc}")
            continue

        confined, detail = pair_diff_is_confined_to_sink_arg(unsafe_src, safe_src)
        if not confined:
            failures.append(f"attempt {attempt}: {detail}")
            continue

        anchors = {role: find_anchors(src)
                   for role, src in (("unsafe", unsafe_src), ("safe", safe_src))}
        if tokenizer is not None:
            bad = _unaligned_anchors(unsafe_src, safe_src, anchors, tokenizer)
            if bad:
                failures.append(f"attempt {attempt}: anchors not token-exact ({bad})")
                continue

        token_info = pair_token_diff(unsafe_src, safe_src, tokenizer) if tokenizer else {}
        shared = {
            "seed": seed, "source_expr": source_expr, "alt_source": alt_source,
            "trusted_expr": trusted_expr, "alt_trusted": alt_trusted,
            "sink": sink_name, "sink_template": sink_tmpl,
            "taint_name": taint_name, "trust_name": trust_name,
            "taint_final": taint_final, "trust_final": trust_final,
            "role_swap": role_swap, "order_swap": order_swap,
            **{f"pair_{k}": v for k, v in token_info.items()},
        }
        members = {}
        for role, src, label in (("unsafe", unsafe_src, 1), ("safe", safe_src, 0)):
            members[role] = FlowProgram(
                program_id=f"{base_id}_{role}", base_id=base_id, family=family,
                structure=structure, role=role, label=label, split="unassigned",
                source=src,
                metadata={**shared, "anchors": {k: list(v) for k, v in anchors[role].items()},
                          "sink_arg_name": taint_final if role == "unsafe" else trust_final},
            )
        return SinkFlowBase(base_id=base_id, family=family, structure=structure,
                            seed=seed, split="unassigned", unsafe=members["unsafe"],
                            safe=members["safe"], metadata=shared)

    raise ValueError(
        f"could not build base {base_id} in {max_attempts} attempts.\n  "
        + "\n  ".join(failures[-4:]))


def _unaligned_anchors(unsafe_src: str, safe_src: str, anchors: dict, tokenizer) -> str:
    """Anchors that do not land exactly on tokenizer boundaries, as a message."""
    from src.data.alignment import compute_offsets

    problems = []
    for role, src in (("unsafe", unsafe_src), ("safe", safe_src)):
        offsets = compute_offsets(src, tokenizer)
        for kind, span in anchors[role].items():
            if anchor_token_span(src, offsets, span) is None:
                start, end = anchor_char_span(src, span)
                problems.append(f"{role}/{kind}={src[start:end]!r}")
    return ", ".join(problems)


def generate_benchmark(
    tokenizer,
    families: Sequence[str] = FAMILIES,
    structures: Sequence[str] = STRUCTURES,
    n_seeds: int = N_BASE_SEEDS,
    n_train_seeds: int = N_TRAIN_SEEDS,
    seed: int = 42,
) -> list[SinkFlowBase]:
    """Every base of the benchmark, split by base id and stratified by cell.

    Splitting happens here rather than downstream so that the file on disk is
    the record of which bases a probe is allowed to see; both members and every
    obfuscated variant of a base inherit its split.
    """
    if n_train_seeds >= n_seeds:
        raise ValueError(f"n_train_seeds ({n_train_seeds}) must be < n_seeds ({n_seeds})")

    bases: list[SinkFlowBase] = []
    for family in families:
        for structure in structures:
            made = [
                _build_base(family, structure, index, tokenizer,
                            seed=seed + 1000 * FAMILIES.index(family)
                            + 100 * STRUCTURES.index(structure) + index)
                for index in range(n_seeds)
            ]
            # Stratified split: whole bases move, 14 train / 6 held out per cell.
            # The per-cell shuffle seed is derived with a *stable* digest —
            # Python's `hash` on strings is salted per process, so using it here
            # would give a different split on every run.
            order = list(range(n_seeds))
            cell_seed = int(base_ids_digest([family, structure])[:8], 16)
            random.Random(seed + cell_seed).shuffle(order)
            train = set(order[:n_train_seeds])
            for index, base in enumerate(made):
                base.split = "train" if index in train else "heldout"
                base.unsafe.split = base.safe.split = base.split
            bases.extend(made)
    return bases


# ── obfuscation of held-out programs only ────────────────────────────────────


# ── which transformations a program actually carries ─────────────────────────
#
# A condition called `flatten_only` that quietly also renamed would make the
# atomic attribution worthless, and a condition called `encode_only` whose draw
# happened to rewrite nothing would dilute the arm it names. Neither is
# detectable from the condition label, so both are *measured* from the variant's
# own AST, by signatures each transformation leaves and the others cannot:
#
#   rename    the program's own integer counter `count` is gone (the ladder's
#             fresh names are two consonants and a digit, so they never collide)
#   opaque    an opaque guard: `<expr> % k == c`, a shape the generator never
#             emits (its only comparison is `count > 2`) and which survives
#             flattening as the test of a state-transition IfExp
#   encode    a bitwise operator (^ & << ~) — the MBA identities are the only
#             thing in this repository that introduces one into these programs
#   flatten   a `while` dispatch loop; the generator emits no loops at all

_ENCODE_OPS = (ast.BitXor, ast.BitAnd, ast.BitOr, ast.LShift, ast.RShift, ast.Invert)


def detect_transformations(source: str) -> set[str]:
    """The transformations a variant carries, read off the variant itself."""
    tree = ast.parse(source)
    found: set[str] = set()
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    names |= {node.arg for node in ast.walk(tree) if isinstance(node, ast.arg)}
    if COUNTER_NAME not in names:
        found.add("rename")
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare) and len(node.ops) == 1 \
                and isinstance(node.ops[0], ast.Eq) \
                and isinstance(node.left, ast.BinOp) \
                and isinstance(node.left.op, ast.Mod):
            found.add("opaque")
        elif isinstance(node, (ast.BinOp, ast.UnaryOp)) and isinstance(node.op, _ENCODE_OPS):
            found.add("encode")
        elif isinstance(node, ast.While):
            found.add("flatten")
    return found


def transform_seed_for(base_id: str, condition: str, seed: int) -> int:
    """The transformation draw for one (base, condition).

    Derived from the BASE id, not the program id, so the unsafe and safe member
    of a pair are transformed with the SAME draw: they hold the same locals in
    the same structure, so an identical draw produces an identical renaming map,
    identical opaque slots and identical dispatch state ids, and the pair still
    differs only at the sink argument after the rewrite. A per-program draw
    would break the matching and silently turn every pair metric into noise.
    """
    return int(base_ids_digest([base_id, condition, str(seed)])[:8], 16)


def transform_heldout(
    bases: Sequence[SinkFlowBase],
    conditions: Sequence[str] = DEFAULT_CONDITIONS,
    seed: int = 42,
    max_draws: int = 8,
) -> list[FlowProgram]:
    """Every held-out program under every requested condition, verified per variant.

    Both members of a base are transformed **together, under one draw**, and a
    draw is accepted only if the variant it produced is the condition it claims
    to be:

      * it parses, and both independent readings still recover the base's label;
      * the instrumented run observes the same sink call with the same argument
        and the same provenance;
      * the transformations detectable in the variant are **exactly** the ones
        the condition declares (`detect_transformations`);
      * the pair still differs only inside the sink-argument span.

    The redraw exists because two of the ladder's rewrites are probabilistic:
    the MBA encoder rewrites an addition with p=0.6 and an int constant with
    p=0.5, so roughly a fifth of `encode` draws would leave the program
    textually unchanged. Keeping those would report an arm diluted with
    untransformed programs under the name of the transformation. This is not
    repair-by-dropping: nothing is discarded, the number of draws used is
    recorded per variant, and a base that never satisfies its own condition is
    emitted marked as failing so the S0 gate refuses the run.
    """
    ladder = ObfuscationLadder(seed=seed)
    wanted = resolve_conditions(conditions)
    variants: list[FlowProgram] = []
    for base in bases:
        if base.split != "heldout":
            continue
        baselines = {p.role: observe_program(p.source) for p in base.programs()}
        for condition in wanted:
            attempt: dict = {}
            for draw_index in range(max_draws):
                draw = transform_seed_for(
                    base.base_id,
                    condition.name if draw_index == 0
                    else f"{condition.name}#{draw_index}", seed)
                attempt = _transform_pair(base, condition, ladder, draw,
                                          baselines, draw_index + 1)
                if attempt["ok"]:
                    break
            for role, payload in attempt["members"].items():
                program = base.unsafe if role == "unsafe" else base.safe
                variants.append(FlowProgram(
                    program_id=f"{program.program_id}__{condition.name}",
                    base_id=program.base_id, family=program.family,
                    structure=program.structure, role=program.role,
                    label=program.label, split=program.split,
                    source=payload["source"],
                    obf_level=condition.order, obf_name=condition.name,
                    metadata={**program.metadata,
                              "anchors": payload["anchors"],
                              "condition": condition.name,
                              "condition_kind": condition.kind,
                              "condition_steps": ",".join(condition.steps),
                              "n_steps": condition.n_steps,
                              "legacy_level": condition.legacy_level,
                              "transform_seed": attempt["draw"],
                              "n_draws": attempt["n_draws"],
                              "detected_steps": ",".join(sorted(payload["detected"])),
                              "label_preserved": attempt["ok"],
                              "recovered_label": payload["recovered"],
                              "preservation_error": attempt["error"]},
                ))
    return variants


def _transform_pair(base: SinkFlowBase, condition: Condition,
                    ladder: ObfuscationLadder, draw: int,
                    baselines: dict, n_draws: int) -> dict:
    """One draw applied to both members of a base; what it produced and whether
    it is the condition it claims to be."""
    members: dict[str, dict] = {}
    problems: list[str] = []
    declared = set(condition.steps)
    for program in base.programs():
        source, detected, recovered = "", set(), -1
        try:
            source = ladder.apply_steps(program.source, condition.steps,
                                        rng=random.Random(draw))
            ast.parse(source)
            recovered = recover_label(source)
            observed = observe_program(source)
            detected = detect_transformations(source)
            if recovered != program.label:
                problems.append(f"{program.program_id}: label {recovered} "
                                f"vs {program.label}")
            if observed.key() != baselines[program.role].key():
                problems.append(f"{program.program_id}: observation "
                                f"{observed.key()} vs {baselines[program.role].key()}")
            if detected != declared:
                problems.append(
                    f"{program.program_id}: carries {sorted(detected) or ['nothing']} "
                    f"but the condition declares {sorted(declared) or ['nothing']}")
        except Exception as exc:                                # noqa: BLE001
            problems.append(f"{program.program_id}: {type(exc).__name__}: {exc}")
        members[program.role] = {
            "source": source, "detected": detected, "recovered": recovered,
            "anchors": ({k: list(v) for k, v in find_anchors(source).items()}
                        if source else {}),
        }
    if all(m["source"] for m in members.values()):
        confined, detail = pair_diff_is_confined_to_sink_arg(
            members["unsafe"]["source"], members["safe"]["source"])
        if not confined:
            problems.append(f"{base.base_id}/{condition.name}: {detail}")
    return {"ok": not problems, "error": "; ".join(problems), "members": members,
            "draw": draw, "n_draws": n_draws}


def obfuscate_heldout(
    bases: Sequence[SinkFlowBase],
    levels: Sequence[int] = OBF_LEVELS,
    seed: int = 42,
) -> list[FlowProgram]:
    """The cumulative ladder alone, by level — kept for callers that predate the
    atomic arms. `transform_heldout` is the general entry point."""
    return transform_heldout(
        bases, conditions=[LEGACY_LEVEL_TO_CONDITION[int(level)] for level in levels],
        seed=seed)


# ── validity gates ───────────────────────────────────────────────────────────


@dataclass
class GateViolation:
    """One failed validity check, with everything needed to act on it."""

    gate: str
    expected: str
    observed: str
    offenders: list[str] = field(default_factory=list)
    fix: str = ""

    def message(self) -> str:
        offenders = (f"\n  offending ids: {', '.join(self.offenders[:5])}"
                     + (f" (+{len(self.offenders) - 5} more)" if len(self.offenders) > 5 else "")
                     ) if self.offenders else ""
        return (f"GATE {self.gate} FAILED\n"
                f"  expected: {self.expected}\n"
                f"  observed: {self.observed}{offenders}\n"
                f"  rerun:    {self.fix}")

    def to_dict(self) -> dict:
        return {"gate": self.gate, "expected": self.expected, "observed": self.observed,
                "offenders": list(self.offenders[:20]), "fix": self.fix}


def validate_benchmark(
    bases: Sequence[SinkFlowBase],
    variants: Sequence[FlowProgram],
    tokenizer=None,
    families: Sequence[str] = FAMILIES,
    structures: Sequence[str] = STRUCTURES,
    n_seeds: int = N_BASE_SEEDS,
    n_train_seeds: int = N_TRAIN_SEEDS,
    conditions: Sequence[str] = DEFAULT_CONDITIONS,
    rerun: str = "python scripts/120_sinkflow_generate.py --model MODEL",
) -> list[GateViolation]:
    """Every generation-time validity gate, as a list of failures (empty = pass).

    Nothing here is skipped when something earlier fails: a caller that stops on
    the first violation would fix one problem per run, and a caller that
    silently dropped the offending programs would report a smaller benchmark as
    if it were the designed one.
    """
    from src.data.alignment import compute_offsets

    violations: list[GateViolation] = []
    programs = [p for b in bases for p in b.programs()]

    def fail(gate: str, expected: str, observed: str, offenders: Sequence[str] = ()):
        violations.append(GateViolation(gate, expected, observed, list(offenders), rerun))

    # 1. exact size
    wanted = expected_clean_programs(families, structures, n_seeds)
    if len(programs) != wanted:
        fail("clean_program_count", f"exactly {wanted} clean programs "
             f"({len(families)} families x {len(structures)} structures x "
             f"{n_seeds} seeds x 2 labels)", f"{len(programs)}")

    # 2. exact balance across family, structure and label
    unbalanced = []
    for family in families:
        for structure in structures:
            for role in ROLES:
                n = sum(1 for p in programs if p.family == family
                        and p.structure == structure and p.role == role)
                if n != n_seeds:
                    unbalanced.append(f"{family}/{structure}/{role}={n}")
    if unbalanced:
        fail("cell_balance", f"exactly {n_seeds} programs in every "
             f"(family, structure, label) cell", "; ".join(unbalanced[:8]), unbalanced)

    # 3. no base or pair leakage across splits, and the designed split sizes
    by_base: dict[str, set[str]] = {}
    for base in bases:
        by_base.setdefault(base.base_id, set()).add(base.split)
    leaked = sorted(b for b, s in by_base.items() if len(s) > 1)
    member_leak = sorted(b.base_id for b in bases
                         if {b.unsafe.split, b.safe.split} != {b.split})
    if leaked or member_leak:
        fail("split_leakage", "every base — and both members of every pair — in "
             "exactly one split",
             f"{len(leaked)} bases in both splits, {len(member_leak)} pairs whose "
             f"members disagree with their base", leaked + member_leak)
    wrong_sizes = []
    for family in families:
        for structure in structures:
            n_train = sum(1 for b in bases if b.family == family
                          and b.structure == structure and b.split == "train")
            if n_train != n_train_seeds:
                wrong_sizes.append(f"{family}/{structure} train={n_train}")
    if wrong_sizes:
        fail("split_sizes", f"{n_train_seeds} training bases in every "
             f"(family, structure) cell", "; ".join(wrong_sizes[:8]), wrong_sizes)

    # 4. every program parses
    unparsable = []
    for program in programs:
        try:
            ast.parse(program.source)
        except SyntaxError as exc:
            unparsable.append(f"{program.program_id} ({exc})")
    if unparsable:
        fail("programs_parse", "every clean program parses",
             f"{len(unparsable)} do not", unparsable)

    # 5. anchors land exactly on tokenizer positions
    if tokenizer is not None:
        misaligned = []
        for program in programs:
            try:
                offsets = compute_offsets(program.source, tokenizer)
                for kind in ANCHOR_KINDS:
                    span = program.metadata["anchors"][kind]
                    if anchor_token_span(program.source, offsets, span) is None:
                        misaligned.append(f"{program.program_id}/{kind}")
            except Exception as exc:                            # noqa: BLE001
                misaligned.append(f"{program.program_id} ({exc})")
        if misaligned:
            fail("anchor_alignment", "source and sink anchors covered exactly by "
                 "tokenizer positions in every program",
                 f"{len(misaligned)} anchors are not", misaligned)

    # 6. labels recomputed from the program, never read from the record
    mislabelled = []
    for program in programs:
        try:
            recovered = recover_label(program.source)
        except Exception as exc:                                # noqa: BLE001
            mislabelled.append(f"{program.program_id} ({exc})")
            continue
        if recovered != program.label:
            mislabelled.append(f"{program.program_id} (recovered {recovered}, "
                               f"record says {program.label})")
    if mislabelled:
        fail("independent_labels", "the static taint fixpoint and instrumented "
             "execution agree with each other and with every stored label",
             f"{len(mislabelled)} programs disagree", mislabelled)

    # 7. the matched pair differs only at the sink argument
    unmatched = []
    for base in bases:
        ok, detail = pair_diff_is_confined_to_sink_arg(base.unsafe.source, base.safe.source)
        if not ok:
            unmatched.append(f"{base.base_id}: {detail}")
    if unmatched:
        fail("pair_diff_confined", "each pair's two members differ only inside "
             "the sink-argument span", f"{len(unmatched)} pairs differ elsewhere",
             unmatched)

    # 8-14. the transformed conditions: atomic arms and the cumulative ladder
    if variants:
        wanted = resolve_conditions(conditions)
        heldout_programs = {p.program_id for b in bases if b.split == "heldout"
                            for p in b.programs()}

        # 8. the security label survives every transformation, under BOTH readings
        broken = [v.program_id for v in variants if not v.metadata.get("label_preserved")]
        if broken:
            details = [f"{v.program_id}: {v.metadata.get('preservation_error', '')}"
                       for v in variants if not v.metadata.get("label_preserved")]
            fail("transformation_label_preserved", "every transformed variant parses "
                 "and keeps its base's security label under both readings",
                 f"{len(broken)} variants do not", details)

        # 9. exact counts, per condition — a missing cell is a silently smaller
        #    benchmark reported as if it were the designed one
        wrong_counts = []
        for condition in wanted:
            n = sum(1 for v in variants if v.obf_name == condition.name)
            if n != len(heldout_programs):
                wrong_counts.append(f"{condition.name}={n}")
        expected_total = expected_variant_count(len(heldout_programs), wanted)
        if wrong_counts or len(variants) != expected_total:
            fail("condition_counts", f"exactly {len(heldout_programs)} variants in "
                 f"each of {len(wanted)} conditions ({expected_total} in total)",
                 f"{len(variants)} variants; wrong cells: "
                 f"{'; '.join(wrong_counts[:8]) or 'none'}", wrong_counts)

        # 10. every held-out program is present in every requested condition
        by_program: dict[str, set[str]] = {}
        for variant in variants:
            root = variant.program_id.split("__")[0]
            by_program.setdefault(root, set()).add(variant.obf_name)
        missing = []
        for program_id in sorted(heldout_programs):
            absent = sorted({c.name for c in wanted} - by_program.get(program_id, set()))
            if absent:
                missing.append(f"{program_id} missing {absent}")
        if missing:
            fail("conditions_complete",
                 f"conditions {[c.name for c in wanted]} present for all "
                 f"{len(heldout_programs)} held-out programs",
                 f"{len(missing)} programs incomplete", missing)

        # 11. only held-out bases are ever transformed
        train_bases = set(split_base_ids(bases, "train"))
        stray = sorted({v.program_id for v in variants if v.base_id in train_bases})
        if stray:
            fail("transformation_heldout_only", "only held-out programs are "
                 "transformed", f"{len(stray)} variants come from training bases",
                 stray)

        # 12. each condition carries EXACTLY the transformations it declares —
        #     read off the variant's own AST, never from its label
        mislabelled = []
        for variant in variants:
            if not variant.source:
                continue
            declared = set(condition_for(variant.obf_name).steps)
            try:
                detected = detect_transformations(variant.source)
            except SyntaxError as exc:
                mislabelled.append(f"{variant.program_id} (unparsable: {exc})")
                continue
            if detected != declared:
                mislabelled.append(
                    f"{variant.program_id}: carries {sorted(detected) or ['nothing']}, "
                    f"declares {sorted(declared) or ['nothing']}")
        if mislabelled:
            fail("condition_transformation_isolation",
                 "each atomic condition contains only its named transformation and "
                 "each cumulative condition exactly its declared prefix",
                 f"{len(mislabelled)} variants disagree with their condition",
                 mislabelled)

        # 13. both members of a pair share one draw, and the transformed pair
        #     still differs only inside the sink-argument span
        pairs: dict[tuple[str, str], dict[str, FlowProgram]] = {}
        for variant in variants:
            pairs.setdefault((variant.base_id, variant.obf_name), {})[variant.role] = variant
        seed_mismatch, unconfined = [], []
        for (base_id, condition_name), members in sorted(pairs.items()):
            if set(members) != {"unsafe", "safe"}:
                unconfined.append(f"{base_id}/{condition_name}: only "
                                  f"{sorted(members)} present")
                continue
            seeds = {m.metadata.get("transform_seed") for m in members.values()}
            if len(seeds) != 1:
                seed_mismatch.append(f"{base_id}/{condition_name}: draws {sorted(seeds)}")
            if not (members["unsafe"].source and members["safe"].source):
                continue
            ok, detail = pair_diff_is_confined_to_sink_arg(
                members["unsafe"].source, members["safe"].source)
            if not ok:
                unconfined.append(f"{base_id}/{condition_name}: {detail}")
        if seed_mismatch:
            fail("pair_transformation_seed", "both members of a pair are "
                 "transformed under the same draw",
                 f"{len(seed_mismatch)} pairs were not", seed_mismatch)
        if unconfined:
            fail("transformed_pair_diff_confined", "each transformed pair's two "
                 "members differ only inside the sink-argument span",
                 f"{len(unconfined)} transformed pairs differ elsewhere", unconfined)

        # 14. the split survives the transformation: a variant inherits its base
        stray_split = [v.program_id for v in variants if v.split != "heldout"]
        unknown_base = [v.program_id for v in variants
                        if v.base_id not in {b.base_id for b in bases}]
        if stray_split or unknown_base:
            fail("variant_split_integrity",
                 "every variant is held out and belongs to a known base",
                 f"{len(stray_split)} variants are not held out, "
                 f"{len(unknown_base)} have no base", stray_split + unknown_base)

    return violations


# ── persistence ──────────────────────────────────────────────────────────────


def sinkflow_paths(model: str, root: str | Path = "data/synthetic") -> dict[str, Path]:
    """The three shards on disk. Model-specific because the tokenizer verifies them."""
    root = Path(root)
    return {
        "train": root / f"sinkflow_{model}_train.jsonl",
        "heldout": root / f"sinkflow_{model}_heldout.jsonl",
        "heldout_obf": root / f"sinkflow_{model}_heldout_obf.jsonl",
    }


def save_programs(programs: Iterable[FlowProgram], path: str | Path) -> Path:
    """Write programs in the `ProbeExample` jsonl contract stage 10/121 reads."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(path, mode="w") as writer:
        for program in programs:
            writer.write(program.to_example().to_dict())
    return path


def load_programs(path: str | Path) -> list[FlowProgram]:
    from src.data.dataset import CodeProbeDataset

    return [FlowProgram.from_example(ex) for ex in CodeProbeDataset.load(path).examples]


def resolve_sinkflow_path(model: str, shard: str, path: str | Path | None = None) -> Path:
    """A shard's path, or an error that names the actual problem and the fix."""
    resolved = Path(path) if path else sinkflow_paths(model)[shard]
    if resolved.exists():
        return resolved
    available = sorted(p.name for p in Path("data/synthetic").glob("sinkflow_*.jsonl"))
    raise FileNotFoundError(
        f"No E15 '{shard}' shard at {resolved}.\n"
        f"  Available: {available or 'none — stage 120 has not run'}\n"
        f"  Generate:  python scripts/120_sinkflow_generate.py --model {model}")


def split_base_ids(bases: Sequence[SinkFlowBase], split: str) -> list[str]:
    return sorted({b.base_id for b in bases if b.split == split})


def base_ids_digest(base_ids: Sequence[str]) -> str:
    """A stable fingerprint of a split, so a frozen probe can prove what it saw."""
    joined = "\n".join(sorted(base_ids)).encode()
    return hashlib.sha256(joined).hexdigest()[:16]


def dataset_summary(bases: Sequence[SinkFlowBase],
                    variants: Sequence[FlowProgram] = ()) -> dict:
    """What stage 120 prints and the manifest records."""
    programs = [p for b in bases for p in b.programs()]
    by_cell: dict[str, int] = {}
    for program in programs:
        key = f"{program.family}/{program.structure}/{program.role}"
        by_cell[key] = by_cell.get(key, 0) + 1
    return {
        "n_bases": len(bases),
        "n_clean_programs": len(programs),
        "n_obf_variants": len(variants),
        "families": sorted({b.family for b in bases}),
        "structures": sorted({b.structure for b in bases}),
        "splits": {s: sum(1 for b in bases if b.split == s)
                   for s in sorted({b.split for b in bases})},
        "labels": {str(l): sum(1 for p in programs if p.label == l) for l in (0, 1)},
        "cells": by_cell,
        "conditions": sorted({v.obf_name for v in variants},
                             key=lambda name: condition_for(name).order),
        "condition_counts": {name: sum(1 for v in variants if v.obf_name == name)
                             for name in sorted({v.obf_name for v in variants},
                                                key=lambda n: condition_for(n).order)},
        "condition_kinds": {name: condition_for(name).kind
                            for name in sorted({v.obf_name for v in variants},
                                               key=lambda n: condition_for(n).order)},
        "n_redraws": sum(1 for v in variants if int(v.metadata.get("n_draws", 1)) > 1),
        "obf_levels": sorted({v.obf_level for v in variants}),
        "train_digest": base_ids_digest(split_base_ids(bases, "train")),
        "heldout_digest": base_ids_digest(split_base_ids(bases, "heldout")),
    }
