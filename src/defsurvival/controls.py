"""Controls that keep shallow cues from solving the task.

`surface`  stage 20's local reader: token ids within 3 of d and u, bucketed distance.
`shallow`  the strongest non-semantic reader we can state: `surface`, plus the bag of
           every token before the use, the bag of tokens between d and u, the number
           and indentation of redefinitions, the indentation heuristic, self-update
           flags and recency of the last redefinition.
`certify`  a pair counts as controlled only if every input to `shallow` is identical
           across its two members while the label differs. Any reader of those cues
           (keyword presence, token counts, indentation, recency) then scores exactly
           one member of every pair correctly.
"""
from __future__ import annotations

import math
from collections import Counter
from types import SimpleNamespace

from src.cruxeval.metrics import surface_features

# Covariates that must match within a controlled pair.
MATCHED = ("n_killers", "heuristic", "d_indent", "u_indent", "killer_indents", "self_update", "use_in_update",
           "last_killer_distance")


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
