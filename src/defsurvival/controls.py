"""Controls that keep shallow cues from solving the task.

`surface`  stage 20's local reader: token ids within 3 of d and u, bucketed distance.
`shallow`  the strongest non-semantic reader we can state: `surface`, plus the bag of
           every token before the use, the bag of tokens between d and u, the number
           and indentation of redefinitions, the indentation heuristic, self-update
           flags and recency of the last redefinition.
`certify`  a pair counts as controlled only if every input to `shallow` except recency
           is identical across its two members while the label differs. Any reader of
           those cues (keyword presence, token counts, indentation, distance) then
           scores exactly one member of every pair correctly.
`swap_twin` builds such a pair from any program: it swaps the assigned names of one
           redefinition of `v` and one assignment to another name at the same
           indentation between d and u, and keeps the swap only if the label flips
           and the pair certifies. What changes is only which construct encloses the
           redefinition, e.g. a `with` body (always runs) versus an `if` body.
"""
from __future__ import annotations

import ast
import math
from collections import Counter
from types import SimpleNamespace

from src.cruxeval.metrics import surface_features

# Covariates that must match within a controlled pair.
MATCHED = ("n_killers", "heuristic", "d_indent", "u_indent", "killer_indents", "self_update", "use_in_update")


def surface(ids, row):
    """Stage 20's lexical reader: token ids within 3 of d and u, plus bucketed distance."""
    return surface_features(ids, SimpleNamespace(pos_i=row["d_anchor"], pos_j=row["u_anchor"],
                                                 distance=row["token_distance"]))


def shallow(ids, row):
    features = surface(ids, row)
    for prefix, tokens in (("p", ids[:row["u_anchor"] + 1]), ("b", ids[row["d_anchor"] + 1:row["u_anchor"]])):
        for token, count in Counter(tokens).items():
            features[f"{prefix}:{token}"] = float(count)
    for key in ("n_killers", "heuristic", "d_indent", "u_indent", "self_update", "use_in_update"):
        features[key] = float(row[key])
    for value, count in Counter(str(row["killer_indents"]).split("+")).items():
        features[f"killer_indent:{value}"] = float(count)
    features["log_distance"] = math.log1p(row["token_distance"])
    features["log_last_killer_distance"] = math.log1p(row["last_killer_distance"])
    return features


def certify(ids_a, a, ids_b, b):
    """Raise AssertionError unless (a, b) is a controlled pair with opposite labels."""
    assert a["label"] != b["label"], "labels do not differ"
    assert len(ids_a) == len(ids_b) and (a["d_anchor"], a["u_anchor"]) == (b["d_anchor"], b["u_anchor"]), \
        "pair not token-aligned"
    u, d = a["u_anchor"], a["d_anchor"]
    assert Counter(ids_a[:u + 1]) == Counter(ids_b[:u + 1]), "tokens before the use differ"
    assert Counter(ids_a[d + 1:u]) == Counter(ids_b[d + 1:u]), "tokens between d and u differ"
    assert surface(ids_a, a) == surface(ids_b, b), "pair differs near d or u"
    for key in MATCHED:
        assert str(a[key]) == str(b[key]), f"{key} differs"


def _name_spans(source, node):
    """Character span of a single-line assignment's target name."""
    line = source.splitlines(keepends=True)[node.lineno - 1]
    start = sum(len(x) for x in source.splitlines(keepends=True)[:node.lineno - 1])
    col = len(line.encode("utf-8")[:node.col_offset].decode("utf-8"))
    return start + col, start + col + len(node.id)


def swap_twin(source, ids, row, label, rng, max_tries=20):
    """A certified twin of (source, row), or None.

    `label(source, only) -> (ids, rows)` tokenizes and labels a program, restricted to the
    candidate `only=(name, d_line, u_line)`, which a single-line swap leaves in place.
    """
    tree = ast.parse(source)
    v = row["name"]
    assigns = [n for n in ast.walk(tree)
               if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
               and n.lineno == n.end_lineno and row["d_line"] < n.lineno < row["u_line"]]
    killers = [n for n in assigns if n.targets[0].id == v]
    others = [n for n in assigns if n.targets[0].id != v]
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    options = [(k, o) for k in killers for o in others if k.col_offset == o.col_offset
               and parents[k] is not parents[o]]  # Same block, same execution condition: never flips.
    rng.shuffle(options)
    for killer, other in options[:max_tries]:
        spans = sorted([(_name_spans(source, killer.targets[0]), other.targets[0].id),
                        (_name_spans(source, other.targets[0]), v)], reverse=True)
        text = source
        for (start, end), name in spans:
            text = text[:start] + name + text[end:]
        twin_ids, rows = label(text, (v, row["d_line"], row["u_line"]))
        match = [r for r in rows if r["name"] == v and (r["d_line"], r["u_line"]) == (row["d_line"], row["u_line"])]
        if twin_ids is None or len(match) != 1:
            continue
        try:
            certify(ids, row, twin_ids, match[0])
        except AssertionError:
            continue
        return text, twin_ids, match[0]
    return None
