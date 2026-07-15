"""Single source of truth for constructing probe datasets.

Builders turn (source, token offsets) into lightweight *records* — token
positions, labels, strata, and the source example id. Feature matrices are
assembled from records + hidden states on demand (`assemble_token_features`
/ `assemble_pair_features`), so hidden vectors are never duplicated per
Python object and every row carries its group id for leak-free CV.

All def/use/guard positions come from AST spans via
src.data.alignment.TokenAligner — never from token-string matching.

Tasks:
  lexical_token_type — token-type classification (E1)
  binding            — same-binding pair classification with negative strata (E2)
  defuse_edge        — def→use edge prediction with distance strata (E3)
  control_dep        — guard→statement control dependence (E4)
  taint_state        — is the value at the decision point tainted? (E6/E7 probe)
"""

from __future__ import annotations

import ast
import random
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from src.data.alignment import TokenAligner
from src.graphs.dfg_extractor import DefUseExtractor, VarEvent

# ── Record types ──────────────────────────────────────────────────────────────

@dataclass
class TokenRecord:
    """A single-position classification example."""
    example_id: str
    pos: int
    label: int
    label_name: str = ""


@dataclass
class PairRecord:
    """A pairwise classification example between two token positions."""
    example_id: str
    pos_i: int
    pos_j: int
    label: int
    stratum: str            # "positive" | "same_name_diff_binding" | "diff_name"
                            # | "distance_matched" | "context_matched"
    distance: int = 0
    name_i: str = ""
    name_j: str = ""


# ── Feature assembly ──────────────────────────────────────────────────────────

def assemble_token_features(
    hidden: np.ndarray,                 # (seq_len, d_model) for ONE example, one layer
    records: Sequence[TokenRecord],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """X, y, groups for token records of a single example."""
    rows = [r for r in records if r.pos < hidden.shape[0]]
    X = np.stack([hidden[r.pos] for r in rows]).astype(np.float32)
    y = np.array([r.label for r in rows], dtype=np.int64)
    groups = np.array([r.example_id for r in rows])
    return X, y, groups


def pair_feature(h_i: np.ndarray, h_j: np.ndarray) -> np.ndarray:
    diff = h_i - h_j
    return np.concatenate([h_i, h_j, diff, np.abs(diff)])


def assemble_pair_features(
    hidden: np.ndarray,                 # (seq_len, d_model) for ONE example, one layer
    records: Sequence[PairRecord],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[PairRecord]]:
    """X, y, groups, kept-records for pair records of a single example."""
    rows = [r for r in records if r.pos_i < hidden.shape[0] and r.pos_j < hidden.shape[0]]
    X = np.stack([
        pair_feature(hidden[r.pos_i].astype(np.float32), hidden[r.pos_j].astype(np.float32))
        for r in rows
    ])
    y = np.array([r.label for r in rows], dtype=np.int64)
    groups = np.array([r.example_id for r in rows])
    return X, y, groups, rows


# ── E1: lexical token type ────────────────────────────────────────────────────

TOKEN_TYPES = [
    "keyword", "identifier", "string_literal", "numeric_literal",
    "operator", "delimiter", "unknown",
]
TOKEN_TYPE_TO_IDX = {t: i for i, t in enumerate(TOKEN_TYPES)}

_KEYWORDS = {
    "def", "class", "if", "else", "elif", "for", "while", "return",
    "import", "from", "with", "as", "try", "except", "finally",
    "pass", "break", "continue", "lambda", "yield", "and", "or", "not",
    "in", "is", "True", "False", "None", "async", "await",
}


def classify_token(tok: str) -> str:
    tok = tok.strip()
    if tok in _KEYWORDS:
        return "keyword"
    if tok.startswith(("'", '"')):
        return "string_literal"
    if tok and tok.replace(".", "").replace("-", "").replace("_", "").isdigit():
        return "numeric_literal"
    if tok.isidentifier():
        return "identifier"
    if tok and all(c in "+-*/%=<>!&|^~@" for c in tok):
        return "operator"
    if tok and all(c in "()[]{}:,;." for c in tok):
        return "delimiter"
    return "unknown"


def build_lexical_records(
    token_strings: Sequence[str],
    example_id: str,
) -> list[TokenRecord]:
    records = []
    for pos, tok in enumerate(token_strings):
        t = classify_token(tok)
        records.append(TokenRecord(
            example_id=example_id, pos=pos,
            label=TOKEN_TYPE_TO_IDX[t], label_name=t,
        ))
    return records


# ── Binding resolution (shared by E2/E3 builders) ────────────────────────────

@dataclass
class ResolvedEvents:
    """All identifier events of one program with binding ids and token anchors."""
    events: list[VarEvent] = field(default_factory=list)
    anchors: list[int] = field(default_factory=list)        # token anchor per event
    binding_ids: list[int] = field(default_factory=list)    # index of the defining event
    edge_pairs: set[tuple[int, int]] = field(default_factory=set)  # (def_ev_idx, use_ev_idx)


def resolve_events(source: str, aligner: TokenAligner) -> Optional[ResolvedEvents]:
    """Extract def/use events, resolve each use to its reaching definition,
    and align every event to a token anchor. Events that can't be aligned
    (truncation) are dropped."""
    dfg = DefUseExtractor().extract(source)
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None

    # Collect all events in source order via the extractor's tracker output:
    # reconstruct from dfg edges plus standalone defs.
    events: list[VarEvent] = []
    seen: set = set()

    def _add(ev: VarEvent):
        key = (ev.name, ev.kind, ev.line, ev.col)
        if key not in seen:
            seen.add(key)
            events.append(ev)

    for edge in dfg.edges:
        _add(edge.definition)
        _add(edge.use)
    events.sort(key=lambda e: (e.line, e.col, e.kind))

    resolved = ResolvedEvents()
    index_of: dict = {}
    for ev in events:
        aligned = aligner.align_var_event(ev)
        if aligned is None:
            continue
        index_of[(ev.name, ev.kind, ev.line, ev.col)] = len(resolved.events)
        resolved.events.append(ev)
        resolved.anchors.append(aligned.anchor)

    # Binding id: def events bind to themselves; uses bind to their reaching def.
    reaching: dict[int, int] = {}
    for edge in dfg.edges:
        d_key = (edge.definition.name, "def", edge.definition.line, edge.definition.col)
        u_key = (edge.use.name, "use", edge.use.line, edge.use.col)
        if d_key in index_of and u_key in index_of:
            reaching[index_of[u_key]] = index_of[d_key]
            resolved.edge_pairs.add((index_of[d_key], index_of[u_key]))

    for i, ev in enumerate(resolved.events):
        if ev.kind == "def":
            resolved.binding_ids.append(i)
        else:
            resolved.binding_ids.append(reaching.get(i, -1))

    return resolved


# ── E2: binding pairs with explicit negative strata ──────────────────────────

def _matched_event_pair(ev, metadata: Optional[dict]) -> Optional[tuple[int, int]]:
    """Locate the designed (def, use) event indices of a context-matched
    program (generator.generate_matched_binding_pair) from its metadata."""
    m = (metadata or {}).get("matched")
    if not m:
        return None
    di = next((k for k, e in enumerate(ev)
               if e.kind == "def" and e.name == m["var"] and e.line == m["def_line"]), None)
    ui = next((k for k, e in enumerate(ev)
               if e.kind == "use" and e.name == m["var"] and e.line == m["use_line"]), None)
    if di is None or ui is None:
        return None
    return di, ui


def _pair_group(example_id: str, metadata: Optional[dict]) -> str:
    """CV group id: both programs of a context-matched pair share a group so
    grouped CV never splits a pair across train/test."""
    m = (metadata or {}).get("matched")
    return m["pair_id"] if m else example_id


def build_binding_records(
    source: str,
    aligner: TokenAligner,
    example_id: str,
    rng: random.Random,
    neg_per_pos: int = 3,
    metadata: Optional[dict] = None,
) -> list[PairRecord]:
    """Same-binding pair classification.

    Positives: two identifier occurrences with the same binding id.
    Negative strata:
      same_name_diff_binding — same surface name, different binding (hard)
      diff_name              — different names, different bindings
      distance_matched       — random identifier pairs, distance-matched to positives
      context_matched        — the designed pair of a matched program pair:
                               windows and distance identical across the two
                               programs, label flipped by one rebinding token
                               (positives and negatives both carry this stratum)
    """
    resolved = resolve_events(source, aligner)
    if resolved is None or len(resolved.events) < 2:
        return []

    ev = resolved.events
    anchors = resolved.anchors
    bid = resolved.binding_ids
    n = len(ev)
    matched = _matched_event_pair(ev, metadata)
    group = _pair_group(example_id, metadata)

    def _rec(i: int, j: int, label: int, stratum: str) -> PairRecord:
        return PairRecord(
            example_id=group,
            pos_i=anchors[i], pos_j=anchors[j],
            label=label, stratum=stratum,
            distance=abs(anchors[j] - anchors[i]),
            name_i=ev[i].name, name_j=ev[j].name,
        )

    positives, hard_negs, diff_negs = [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            if anchors[i] == anchors[j] or bid[i] == -1 or bid[j] == -1:
                continue
            if bid[i] == bid[j]:
                stratum = "context_matched" if (i, j) == matched else "positive"
                positives.append(_rec(i, j, 1, stratum))
            elif ev[i].name == ev[j].name:
                stratum = "context_matched" if (i, j) == matched else "same_name_diff_binding"
                hard_negs.append(_rec(i, j, 0, stratum))
            else:
                diff_negs.append(_rec(i, j, 0, "diff_name"))

    if not positives:
        return []

    # Cap easy negatives; keep ALL hard negatives (they are the point).
    rng.shuffle(diff_negs)
    n_easy = min(len(diff_negs), neg_per_pos * len(positives))
    records = positives + hard_negs + diff_negs[:n_easy]

    # Distance-matched random negatives: sample identifier-anchor pairs whose
    # token distance matches the positive distance distribution.
    pos_dists = [p.distance for p in positives]
    id_anchors = sorted(set(anchors))
    dm = []
    attempts = 0
    while len(dm) < len(positives) and attempts < 50 * len(positives):
        attempts += 1
        target = rng.choice(pos_dists)
        a = rng.choice(id_anchors)
        b_candidates = [x for x in id_anchors if abs(x - a) == target and x != a]
        if not b_candidates:
            continue
        b = rng.choice(b_candidates)
        i = anchors.index(a)
        j = anchors.index(b)
        if bid[i] != -1 and bid[j] != -1 and bid[i] != bid[j]:
            dm.append(_rec(i, j, 0, "distance_matched"))
    records += dm
    return records


# ── E3: def-use edge prediction ───────────────────────────────────────────────

def build_defuse_records(
    source: str,
    aligner: TokenAligner,
    example_id: str,
    rng: random.Random,
    neg_per_pos: int = 3,
    metadata: Optional[dict] = None,
) -> list[PairRecord]:
    """Directed def→use edge prediction.

    Positives: (def, use) pairs connected in the DFG.
    Negative strata:
      same_name_diff_binding — (def, use) same name but a different def reaches
      diff_name              — (def of x, use of y)
      distance_matched       — random (def, use) pairs, distance-matched
      context_matched        — designed pair of a matched program pair
                               (see build_binding_records)
    """
    resolved = resolve_events(source, aligner)
    if resolved is None or not resolved.edge_pairs:
        return []

    ev = resolved.events
    anchors = resolved.anchors
    n = len(ev)
    def_idx = [i for i in range(n) if ev[i].kind == "def"]
    use_idx = [i for i in range(n) if ev[i].kind == "use"]
    matched = _matched_event_pair(ev, metadata)
    group = _pair_group(example_id, metadata)

    def _rec(i: int, j: int, label: int, stratum: str) -> PairRecord:
        return PairRecord(
            example_id=group,
            pos_i=anchors[i], pos_j=anchors[j],
            label=label, stratum=stratum,
            distance=abs(anchors[j] - anchors[i]),
            name_i=ev[i].name, name_j=ev[j].name,
        )

    positives, hard_negs, diff_negs = [], [], []
    for d in def_idx:
        for u in use_idx:
            if anchors[d] == anchors[u]:
                continue
            if (d, u) in resolved.edge_pairs:
                stratum = "context_matched" if (d, u) == matched else "positive"
                positives.append(_rec(d, u, 1, stratum))
            elif ev[d].name == ev[u].name:
                stratum = "context_matched" if (d, u) == matched else "same_name_diff_binding"
                hard_negs.append(_rec(d, u, 0, stratum))
            else:
                diff_negs.append(_rec(d, u, 0, "diff_name"))

    if not positives:
        return []

    rng.shuffle(diff_negs)
    n_easy = min(len(diff_negs), neg_per_pos * len(positives))
    records = positives + hard_negs + diff_negs[:n_easy]

    pos_dists = [p.distance for p in positives]
    dm = []
    attempts = 0
    while len(dm) < len(positives) and attempts < 50 * len(positives):
        attempts += 1
        target = rng.choice(pos_dists)
        d = rng.choice(def_idx)
        candidates = [u for u in use_idx
                      if abs(anchors[u] - anchors[d]) == target
                      and (d, u) not in resolved.edge_pairs
                      and anchors[u] != anchors[d]]
        if candidates:
            dm.append(_rec(d, rng.choice(candidates), 0, "distance_matched"))
    records += dm
    return records


DISTANCE_BUCKETS = [(0, 10), (10, 50), (50, 200), (200, 100000)]


def bucket_label(distance: int) -> str:
    for lo, hi in DISTANCE_BUCKETS:
        if lo <= distance < hi:
            return f"dist_{lo}_{hi}"
    return "dist_other"


# ── E4: control dependence ────────────────────────────────────────────────────

class _GuardCollector(ast.NodeVisitor):
    """Collect (guard-expression span, dependent-statement spans) via AST walk.

    The guard anchor is the *test/iter expression* span (not the whole If/While
    statement, whose AST span includes the body)."""

    def __init__(self):
        self.guards: list[dict] = []       # {expr: (l,c,el,ec), body_stmts: [stmt spans], type}
        self.all_stmts: list[tuple] = []   # spans of every simple statement

    def _stmt_span(self, stmt: ast.stmt) -> tuple:
        return (stmt.lineno, stmt.col_offset, stmt.end_lineno, stmt.end_col_offset)

    def _collect_body(self, body: list[ast.stmt]) -> list[tuple]:
        spans = []
        for s in body:
            spans.append(self._stmt_span(s))
            # nested bodies belong to the outer guard too
            for attr in ("body", "orelse", "finalbody"):
                inner = getattr(s, attr, None)
                if isinstance(inner, list) and inner and isinstance(inner[0], ast.stmt):
                    spans.extend(self._collect_body(inner))
        return spans

    def visit(self, node):
        if isinstance(node, (ast.If, ast.While)):
            expr = node.test
            self.guards.append({
                "expr": (expr.lineno, expr.col_offset, expr.end_lineno, expr.end_col_offset),
                "body": self._collect_body(node.body),
                "orelse": self._collect_body(node.orelse),
                "type": type(node).__name__,
            })
        elif isinstance(node, ast.For):
            expr = node.iter
            self.guards.append({
                "expr": (expr.lineno, expr.col_offset, expr.end_lineno, expr.end_col_offset),
                "body": self._collect_body(node.body),
                "orelse": self._collect_body(node.orelse),
                "type": "For",
            })
        if isinstance(node, ast.stmt) and not isinstance(
            node, (ast.If, ast.While, ast.For, ast.FunctionDef,
                   ast.AsyncFunctionDef, ast.ClassDef, ast.Try)
        ):
            self.all_stmts.append(self._stmt_span(node))
        super().generic_visit(node)


def build_control_dep_records(
    source: str,
    aligner: TokenAligner,
    example_id: str,
    rng: random.Random,
    neg_per_pos: int = 3,
) -> list[PairRecord]:
    """(guard expression, statement) control-dependence classification.

    Positives: statement inside the guard's body (or orelse).
    Negatives: statements from the same program outside that guard's bodies.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    collector = _GuardCollector()
    collector.visit(tree)
    if not collector.guards:
        return []

    def _anchor(span: tuple) -> Optional[int]:
        aligned = aligner.align("", "stmt", span[0], span[1], span[2], span[3])
        return aligned.anchor if aligned else None

    records = []
    for guard in collector.guards:
        g_anchor = _anchor(guard["expr"])
        if g_anchor is None:
            continue
        dependent = set(guard["body"]) | set(guard["orelse"])
        positives, negatives = [], []
        for span in collector.all_stmts:
            s_anchor = _anchor(span)
            if s_anchor is None or s_anchor == g_anchor:
                continue
            label = 1 if span in dependent else 0
            rec = PairRecord(
                example_id=example_id,
                pos_i=g_anchor, pos_j=s_anchor,
                label=label,
                stratum="positive" if label else "non_dependent",
                distance=abs(s_anchor - g_anchor),
            )
            (positives if label else negatives).append(rec)
        rng.shuffle(negatives)
        records += positives + negatives[: max(1, neg_per_pos * max(1, len(positives)))]
    return records


# ── E6/E7: taint-state probe ─────────────────────────────────────────────────

def build_taint_records(
    source: str,
    aligner: TokenAligner,
    example_id: str,
    label: int,
) -> list[TokenRecord]:
    """Taint-state classification at the decision point.

    Anchor: the argument token of the final sink call (last Call in the
    program whose argument is a bare Name), falling back to the last token.
    `label` is the example-level ground truth (1 = tainted reaches sink).
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    sink_arg_span = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and node.args:
            arg = node.args[0]
            if isinstance(arg, ast.Name):
                sink_arg_span = (arg.lineno, arg.col_offset, arg.end_lineno, arg.end_col_offset)

    records = []
    if sink_arg_span is not None:
        aligned = aligner.align("", "sink_arg", *sink_arg_span)
        if aligned is not None:
            records.append(TokenRecord(
                example_id=example_id, pos=aligned.anchor,
                label=label, label_name="sink_arg",
            ))

    n_tokens = len(aligner.offsets)
    records.append(TokenRecord(
        example_id=example_id, pos=n_tokens - 1,
        label=label, label_name="last_token",
    ))
    return records
