"""Semantics-preserving obfuscation ladder (E9).

Tigress-inspired (https://tigress.wtf) but implemented natively for Python:
Tigress itself is a C source-to-source obfuscator and cannot be applied to
this project's Python corpus, so the relevant transformation classes are
re-implemented as ast-level rewrites. Semantics preservation is not assumed —
every variant is VERIFIED observationally equivalent to its base by executing
both and comparing `func(...)` results (only trusted, generated sources are
ever executed).

Levels are cumulative, in increasing difficulty:

  0 normalize   ast round-trip only — shared formatting baseline, so that
                unparse artifacts are never confounded with obfuscation
  1 rename      consistent alpha-renaming of every local variable
  2 opaque      + dead branches guarded by opaque predicates (provably false
                for all ints, e.g. v*v % 4 == 3) with decoy assignments
  3 encode      + mixed boolean-arithmetic encoding of int expressions
                (a+b → (a^b) + ((a&b)<<1); a-b → a + ~b + 1; c → (c^m)^m)
  4 flatten     + control-flow flattening into a while/state-machine dispatch
                with shuffled, non-contiguous state ids

Probing ground truth is never carried over from the base program: the E9
harness rebuilds it from each variant's own source, exactly like the E5
context variants. Level 1 isolates lexical reliance (RQ3); levels 2–4 add
control-flow and dataflow surface noise while the underlying relations that
remain are recomputed per variant.
"""

from __future__ import annotations

import ast
import copy
import itertools
import random
from typing import Optional

from .dataset import ProbeExample

OBFUSCATION_LEVELS: list[tuple[int, str]] = [
    (0, "normalize"),
    (1, "rename"),
    (2, "opaque"),
    (3, "encode"),
    (4, "flatten"),
]

_NAME_ALPHA = "bcdfghjklmnpqrstvwxz"


def _fresh_opaque_name(rng: random.Random, taken: set[str]) -> str:
    while True:
        name = (rng.choice(_NAME_ALPHA) + rng.choice(_NAME_ALPHA)
                + str(rng.randint(0, 9)))
        if name not in taken and not name[0].isdigit():
            taken.add(name)
            return name


def _collect_local_names(tree: ast.AST) -> set[str]:
    """Names bound by assignment / params / for-targets (not function names)."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            names.add(node.id)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
    return names


def _all_names(tree: ast.AST) -> set[str]:
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(node.name)
    return names


# ── T1: consistent renaming ──────────────────────────────────────────────────

class _Renamer(ast.NodeTransformer):
    def __init__(self, mapping: dict[str, str]):
        self.mapping = mapping

    def visit_Name(self, node: ast.Name):
        if node.id in self.mapping:
            node.id = self.mapping[node.id]
        return node

    def visit_arg(self, node: ast.arg):
        if node.arg in self.mapping:
            node.arg = self.mapping[node.arg]
        return node


def _rename(tree: ast.AST, rng: random.Random) -> None:
    """Injective, consistent renaming of all locally bound names.

    Renaming every occurrence of a name to the same fresh name preserves
    semantics even under shadowing (the binding structure is untouched)."""
    local = _collect_local_names(tree)
    taken = set(_all_names(tree))
    mapping = {name: _fresh_opaque_name(rng, taken) for name in sorted(local)}
    _Renamer(mapping).visit(tree)


# ── T2: opaque-predicate dead code ───────────────────────────────────────────

def _int_vars_before(body: list[ast.stmt], idx: int) -> list[str]:
    """Names assigned at the top level of `body[:idx]` whose values are
    provably int (int constants, or arithmetic over already-known ints)."""
    known: set[str] = set()
    for stmt in body[:idx]:
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 \
                and isinstance(stmt.targets[0], ast.Name):
            if _is_int_expr(stmt.value, known):
                known.add(stmt.targets[0].id)
            else:
                known.discard(stmt.targets[0].id)
    return sorted(known)


def _is_int_expr(node: ast.expr, int_names: set[str]) -> bool:
    if isinstance(node, ast.Constant):
        return isinstance(node.value, int) and not isinstance(node.value, bool)
    if isinstance(node, ast.Name):
        return node.id in int_names
    if isinstance(node, ast.BinOp) and not isinstance(node.op, ast.Div):
        return _is_int_expr(node.left, int_names) and _is_int_expr(node.right, int_names)
    if isinstance(node, ast.UnaryOp):
        return _is_int_expr(node.operand, int_names)
    return False


def _opaque_false_guard(var: str, rng: random.Random) -> ast.expr:
    """A predicate that is False for every int value of `var`."""
    v = ast.Name(var, ast.Load())
    if rng.random() < 0.5:
        # v*v % 4 == 3  — squares are 0 or 1 mod 4
        test = ast.BinOp(ast.BinOp(v, ast.Mult(), copy.deepcopy(v)), ast.Mod(),
                         ast.Constant(4))
        return ast.Compare(test, [ast.Eq()], [ast.Constant(3)])
    # (v*v + v) % 2 == 1  — v(v+1) is always even
    test = ast.BinOp(
        ast.BinOp(ast.BinOp(v, ast.Mult(), copy.deepcopy(v)), ast.Add(),
                  copy.deepcopy(v)),
        ast.Mod(), ast.Constant(2))
    return ast.Compare(test, [ast.Eq()], [ast.Constant(1)])


def _insert_opaque_dead_code(tree: ast.AST, rng: random.Random) -> None:
    taken = set(_all_names(tree))
    for func in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
        # candidate insertion points: after >=1 statement, before the final return
        last = len(func.body) - 1 if isinstance(func.body[-1], ast.Return) \
            else len(func.body)
        slots = [i for i in range(1, last + 1) if _int_vars_before(func.body, i)]
        if not slots:
            continue
        n_blocks = rng.randint(1, min(2, len(slots)))
        for idx in sorted(rng.sample(slots, n_blocks), reverse=True):
            var = rng.choice(_int_vars_before(func.body, idx))
            d1 = _fresh_opaque_name(rng, taken)
            d2 = _fresh_opaque_name(rng, taken)
            v = lambda: ast.Name(var, ast.Load())
            dead_body = [
                ast.Assign([ast.Name(d1, ast.Store())],
                           ast.BinOp(v(), ast.Add(), ast.Constant(rng.randint(2, 40)))),
                ast.Assign([ast.Name(d2, ast.Store())],
                           ast.BinOp(ast.Name(d1, ast.Load()), ast.Mult(), v())),
            ]
            block = ast.If(test=_opaque_false_guard(var, rng), body=dead_body,
                           orelse=[])
            func.body.insert(idx, block)


# ── T3: mixed boolean-arithmetic expression encoding ─────────────────────────

def _pure(node: ast.expr) -> bool:
    if isinstance(node, (ast.Name, ast.Constant)):
        return True
    if isinstance(node, ast.BinOp):
        return _pure(node.left) and _pure(node.right)
    if isinstance(node, ast.UnaryOp):
        return _pure(node.operand)
    return False


class _ExprEncoder(ast.NodeTransformer):
    """Integer-exact rewrites; operands must be pure (evaluated twice).

    Identities hold for ALL Python ints (arbitrary precision, two's
    complement bitwise semantics):
        a + b == (a ^ b) + ((a & b) << 1)
        a - b == a + ~b + 1
        c     == (c ^ m) ^ m
    """

    def __init__(self, rng: random.Random):
        self.rng = rng

    def visit_BinOp(self, node: ast.BinOp):
        self.generic_visit(node)
        if not (_pure(node.left) and _pure(node.right)):
            return node
        if isinstance(node.op, ast.Add) and self.rng.random() < 0.6:
            l2, r2 = copy.deepcopy(node.left), copy.deepcopy(node.right)
            xor = ast.BinOp(node.left, ast.BitXor(), node.right)
            carry = ast.BinOp(ast.BinOp(l2, ast.BitAnd(), r2), ast.LShift(),
                              ast.Constant(1))
            return ast.BinOp(xor, ast.Add(), carry)
        if isinstance(node.op, ast.Sub) and self.rng.random() < 0.6:
            inv = ast.UnaryOp(ast.Invert(), node.right)
            return ast.BinOp(ast.BinOp(node.left, ast.Add(), inv), ast.Add(),
                             ast.Constant(1))
        return node

    def visit_Assign(self, node: ast.Assign):
        self.generic_visit(node)
        node.value = self._maybe_encode_const(node.value)
        return node

    def _maybe_encode_const(self, node: ast.expr) -> ast.expr:
        if (isinstance(node, ast.Constant) and isinstance(node.value, int)
                and not isinstance(node.value, bool) and node.value >= 2
                and self.rng.random() < 0.5):
            m = self.rng.randint(1, 255)
            return ast.BinOp(ast.Constant(node.value ^ m), ast.BitXor(),
                             ast.Constant(m))
        return node


def _encode_int_exprs(tree: ast.AST, rng: random.Random) -> None:
    _ExprEncoder(rng).visit(tree)


# ── T4: control-flow flattening ──────────────────────────────────────────────

_EXIT = -1


def _flatten_body(body: list[ast.stmt], st: str, ret: str,
                  rng: random.Random) -> list[ast.stmt]:
    """Lower a statement list to a while/state-machine dispatch.

    If-statements become conditional state transitions; returns store to a
    result variable and jump to the exit state; every other statement is an
    atomic case. State ids are shuffled and non-contiguous."""
    counter = itertools.count()
    # sid -> (stmts, jump) where jump is ("jump", target) or
    # ("cond", test, then_target, else_target)
    cases: dict[int, tuple[list[ast.stmt], tuple]] = {}

    def lower(stmts: list[ast.stmt], next_state: int) -> int:
        entry = next_state
        for stmt in reversed(stmts):
            sid = next(counter)
            if isinstance(stmt, ast.If):
                then_entry = lower(stmt.body, entry)
                else_entry = lower(stmt.orelse, entry) if stmt.orelse else entry
                cases[sid] = ([], ("cond", stmt.test, then_entry, else_entry))
            elif isinstance(stmt, ast.Return):
                store = ast.Assign([ast.Name(ret, ast.Store())],
                                   stmt.value or ast.Constant(None))
                cases[sid] = ([store], ("jump", _EXIT))
            else:
                cases[sid] = ([stmt], ("jump", entry))
            entry = sid
        return entry

    entry = lower(body, _EXIT)
    if not cases:
        return body

    # non-contiguous shuffled state ids (exit id stays -1)
    ids = sorted(cases)
    new_ids = rng.sample(range(0, 3 * len(ids) + 7), len(ids))
    remap = dict(zip(ids, new_ids))
    remap[_EXIT] = _EXIT

    def jump_assign(jump: tuple) -> ast.Assign:
        target = ast.Name(st, ast.Store())
        if jump[0] == "jump":
            return ast.Assign([target], ast.Constant(remap[jump[1]]))
        _, test, then_t, else_t = jump
        return ast.Assign([target], ast.IfExp(test, ast.Constant(remap[then_t]),
                                              ast.Constant(remap[else_t])))

    # dispatch chain in shuffled order
    order = sorted(cases, key=lambda s: remap[s])
    rng.shuffle(order)
    dispatch: Optional[ast.If] = None
    for sid in reversed(order):
        stmts, jump = cases[sid]
        case_body = [copy.deepcopy(s) for s in stmts] + [jump_assign(jump)]
        test = ast.Compare(ast.Name(st, ast.Load()), [ast.Eq()],
                           [ast.Constant(remap[sid])])
        dispatch = ast.If(test=test, body=case_body,
                          orelse=[dispatch] if dispatch else [])

    loop_test = ast.Compare(ast.Name(st, ast.Load()), [ast.NotEq()],
                            [ast.Constant(_EXIT)])
    return [
        ast.Assign([ast.Name(ret, ast.Store())], ast.Constant(None)),
        ast.Assign([ast.Name(st, ast.Store())], ast.Constant(remap[entry])),
        ast.While(test=loop_test, body=[dispatch], orelse=[]),
        ast.Return(ast.Name(ret, ast.Load())),
    ]


def _flatten_control_flow(tree: ast.AST, rng: random.Random) -> None:
    taken = set(_all_names(tree))
    for node in tree.body if isinstance(tree, ast.Module) else []:
        if isinstance(node, ast.FunctionDef):
            st = _fresh_opaque_name(rng, taken)
            ret = _fresh_opaque_name(rng, taken)
            node.body = _flatten_body(node.body, st, ret, rng)


# ── Ladder + verification ────────────────────────────────────────────────────

class ObfuscationLadder:
    """Apply cumulative obfuscation levels to a (trusted, generated) source."""

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def obfuscate(self, source: str, level: int,
                  rng: Optional[random.Random] = None) -> str:
        rng = rng or self.rng
        tree = ast.parse(source)
        if level >= 1:
            _rename(tree, rng)
        if level >= 2:
            _insert_opaque_dead_code(tree, rng)
        if level >= 3:
            _encode_int_exprs(tree, rng)
        if level >= 4:
            _flatten_control_flow(tree, rng)
        ast.fix_missing_locations(tree)
        return ast.unparse(tree)


def _run_func(source: str, arg_tuples: list[tuple]) -> list:
    ns: dict = {}
    exec(compile(source, "<obf>", "exec"), ns)  # trusted generated code only
    fn = ns["func"]
    return [fn(*args) for args in arg_tuples]


def semantically_equivalent(base_source: str, variant_source: str,
                            n_arg_samples: int = 4, seed: int = 0) -> bool:
    """Observational equivalence of `func` (same standard Tigress uses:
    identical I/O behavior). Zero-arg functions are called once; functions
    with parameters are called on sampled int argument tuples."""
    try:
        tree = ast.parse(base_source)
        func = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
        n_params = len(func.args.args) + len(func.args.posonlyargs)
        rng = random.Random(seed)
        if n_params == 0:
            arg_tuples = [()]
        else:
            arg_tuples = [tuple(rng.randint(-100, 100) for _ in range(n_params))
                          for _ in range(n_arg_samples)]
        return _run_func(base_source, arg_tuples) == \
            _run_func(variant_source, arg_tuples)
    except Exception:
        return False


def generate_obfuscation_batch(
    n_base: int = 40,
    levels: Optional[list[int]] = None,
    seed: int = 42,
) -> list[ProbeExample]:
    """Binding programs × cumulative obfuscation levels, execution-verified.

    All levels of a base are kept or dropped together, so per-level curves
    are always computed over the identical set of base programs. Metadata:
    base_example_id, obf_level, obf_name, verified."""
    from .generator import SyntheticCodeGenerator, SyntheticSpec

    rng = random.Random(seed)
    gen = SyntheticCodeGenerator(seed=seed)
    ladder = ObfuscationLadder(seed=seed)
    wanted = levels if levels is not None else [lv for lv, _ in OBFUSCATION_LEVELS]
    variants: list[ProbeExample] = []

    n_made = 0
    attempt = 0
    while n_made < n_base and attempt < n_base * 5:
        attempt += 1
        spec = SyntheticSpec(
            n_vars=rng.randint(2, 4),
            chain_length=rng.randint(2, 4),
            has_branch=rng.random() < 0.6,
            has_dead_def=rng.random() < 0.3,
            has_shadow=rng.random() < 0.2,
            seed=rng.randint(0, 999999),
        )
        base = gen.generate_binding(spec)
        level_sources: dict[tuple[int, str], str] = {}
        ok = True
        for level, name in OBFUSCATION_LEVELS:
            if level not in wanted:
                continue
            try:
                v_src = ladder.obfuscate(base.source, level,
                                         rng=random.Random(rng.randint(0, 999999)))
            except Exception:
                ok = False
                break
            if not semantically_equivalent(base.source, v_src):
                ok = False
                break
            level_sources[(level, name)] = v_src
        if not ok or not level_sources:
            continue
        n_made += 1
        for (level, name), src in level_sources.items():
            variants.append(ProbeExample(
                example_id=f"{base.example_id}_obf{level}",
                source=src,
                metadata={
                    "type": "obfuscation_variant",
                    "base_example_id": base.example_id,
                    "obf_level": level,
                    "obf_name": name,
                    "verified": True,
                },
            ))
    return variants
