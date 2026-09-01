"""The semantic-concept vocabulary panel — a separate question from value recovery.

E19's readout asks what a position is poised to *say*, over the whole
vocabulary, and its value-carrying families score a specific runtime value. This
panel asks a different question of the same instrument:

    does the residual stream, read through the published J-lens or R-lens,
    surface the LANGUAGE OF BINDING — `local`, `global`, `shadowed`, `scope` —
    at the positions where a binding has been resolved?

It is kept separate from runtime-value recovery on purpose. A value is a fact
about the program; a concept word is a claim about what the model has named. The
two nulls mean different things, and pooling them would let a hit on one cover a
miss on the other.

## What makes a positive supportable

A single word ranking well is not evidence. `local` is a common Python token: it
appears in `locals()`, in `local_var`, in comments, and any position inside a
function body has some prior on it. So the panel is built so that a positive
requires ALL of:

  1. the PREDECLARED binding concepts move *with the binding* — the shadowed and
     unshadowed arms of the same program differ in the direction the semantics
     demand, not merely in magnitude;
  2. the crossed VALUE arms agree — the same movement appears whether the outer
     definition holds `v_a` and the inner `v_b` or the reverse, so nothing rides
     on which literal is in scope;
  3. the movement is stronger than the MATCHED CONTROLS below;
  4. it replicates across prompts, and preferably across models.

Nothing is redefined after the fact. The concept sets are fixed in this module,
the read positions are E19's four, and the aggregation is the same
`readout.rank_of` used everywhere else.

## The controls, and what each one rules out

    generic_code      unrelated code vocabulary of comparable frequency and
                      tokenization (`return`, `import`, `value`... ). Rules out
                      "any code-ish word ranks well inside a function body".
    positional        `earlier`/`later`, `kept`/`replaced`. These are CONFOUND
                      DIAGNOSTICS, not binding semantics: the shadowing program
                      also differs in which statement came last, so a reader
                      tracking recency would move these. Labelled as such
                      everywhere, and never counted as a binding positive.
    random_concepts   size-matched random draws from the model's own vocabulary,
                      restricted to tokens of comparable frequency band. The
                      floor for "a set of this size ranks this well by chance".

Two further contrasts come from the panel's own programs rather than from a
word list, and are the ones conditions 1 and 2 read:

    value-changed     same binding structure, different literals. A concept that
                      tracks binding must be invariant to them.
    binding-flipped   the two arms of one pair: token-identical at the read
                      position, opposite definitions live.

## A concept is a set of token ids, and never a truncation

`local` may be one token bare, one token space-prefixed, capitalised, or split.
Every single-token spelling is accepted and the concept's score is the best rank
over them, exactly as `evalsuite.target_token_ids` does for value targets. A
concept the tokenizer *splits* is recorded as unavailable and scored on nothing:
reducing `shadowed` to `shad` would report the BPE merge table as a finding
about the model. Accepted ids and their decoded spellings are written into the
manifest and the report, so a reader can see precisely what was scored.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Sequence

logger = logging.getLogger(__name__)

#: The four E19 read positions, in program order. Same names, same anchors.
READS = ("use", "post_use", "call", "answer")

#: Predeclared, in this order, before any number was looked at. Each entry is a
#: concept name and the surface spellings that count as that concept. Casing
#: variants are members of the same concept: which case a BPE tokenizer keeps
#: whole is a fact about the merge table, not about the model's semantics.
BINDING_CONCEPTS: dict[str, tuple[str, ...]] = {
    "local": ("local", "Local", "LOCAL"),
    "global": ("global", "Global", "GLOBAL"),
    "inner": ("inner", "Inner"),
    "outer": ("outer", "Outer"),
    "scope": ("scope", "Scope"),
    "scoped": ("scoped", "Scoped"),
    "shadow": ("shadow", "Shadow"),
    "shadowed": ("shadowed", "Shadowed"),
    "binding": ("binding", "Binding"),
    "bound": ("bound", "Bound"),
    "active": ("active", "Active"),
    "inactive": ("inactive", "Inactive"),
    "definition": ("definition", "Definition", "def"),
    "variable": ("variable", "Variable", "var"),
    "value": ("value", "Value", "val"),
}

#: Unrelated code vocabulary. Chosen for comparable frequency and tokenization —
#: ordinary Python identifiers and keywords a code model sees constantly — so
#: "binding concepts rank well" cannot just mean "code words rank well here".
GENERIC_CODE_CONCEPTS: dict[str, tuple[str, ...]] = {
    "return": ("return", "Return"),
    "import": ("import", "Import"),
    "class": ("class", "Class"),
    "print": ("print", "Print"),
    "range": ("range", "Range"),
    "index": ("index", "Index", "idx"),
    "result": ("result", "Result"),
    "buffer": ("buffer", "Buffer", "buf"),
    "string": ("string", "String", "str"),
    "number": ("number", "Number", "num"),
    "file": ("file", "File"),
    "path": ("path", "Path"),
    "error": ("error", "Error", "err"),
    "count": ("count", "Count"),
    "total": ("total", "Total"),
}

#: CONFOUND DIAGNOSTICS. The shadowing construction also differs in recency and
#: in "which assignment survived", so a model tracking either would move these
#: without representing scope. Reported in their own block, and a movement here
#: is a warning about the interpretation, never a binding positive.
POSITIONAL_CONCEPTS: dict[str, tuple[str, ...]] = {
    "earlier": ("earlier", "Earlier", "first", "First"),
    "later": ("later", "Later", "last", "Last"),
    "kept": ("kept", "Kept", "keep"),
    "replaced": ("replaced", "Replaced", "replace"),
}

#: Which family a concept set belongs to, and how a movement in it is to be read.
FAMILY_ROLE = {
    "binding_concept": "predeclared binding semantics — the hypothesis",
    "generic_code": "matched unrelated code vocabulary — control",
    "positional": "recency / survival wording — CONFOUND DIAGNOSTIC, not semantics",
    "random_concepts": "size- and frequency-matched random draw — chance floor",
}


@dataclass
class Concept:
    """One concept, resolved against one tokenizer."""

    name: str
    family: str
    requested: tuple[str, ...]
    token_ids: list[int] = field(default_factory=list)
    spellings: list[str] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)

    @property
    def available(self) -> bool:
        return bool(self.token_ids)

    def as_dict(self) -> dict:
        return {"concept": self.name, "family": self.family,
                "requested": list(self.requested), "token_ids": list(self.token_ids),
                "spellings": list(self.spellings), "rejected": list(self.rejected),
                "n_token_ids": len(self.token_ids), "available": self.available}


def resolve_concept(tokenizer, name: str, spellings: Sequence[str],
                    family: str) -> Concept:
    """Every single-token spelling of `name`, space-prefixed and bare.

    A spelling the tokenizer splits is REJECTED, never truncated. The rejection
    is kept on the concept so the report can say `shadowed` was unavailable
    rather than silently scoring `shad`.
    """
    concept = Concept(name=name, family=family, requested=tuple(spellings))
    for word in spellings:
        for form in (" " + word, word):
            enc = tokenizer(form, add_special_tokens=False)["input_ids"]
            if len(enc) == 1:
                if int(enc[0]) not in concept.token_ids:
                    concept.token_ids.append(int(enc[0]))
                    concept.spellings.append(form)
            elif form not in concept.rejected:
                concept.rejected.append(form)
    return concept


def random_concepts(tokenizer, like: Sequence[Concept], n_sets: int = 3,
                    seed: int = 0, vocab_size: Optional[int] = None
                    ) -> list[Concept]:
    """Size- and frequency-band-matched random concept sets.

    "Frequency-matched" is approximated by token id: BPE merge tables are built
    in descending corpus frequency, so a token drawn from the same id decile as
    a real concept's tokens sits in a comparable frequency band. That is a
    coarse proxy and is labelled as one — it is enough to floor "a set of this
    size, made of tokens this common, ranks this well by chance", which is the
    objection it exists to answer, and it needs no corpus counts this repository
    does not have.

    Deterministic given `seed`, so the floor is the same number on every run and
    on every machine.
    """
    import random

    available = [c for c in like if c.available]
    if not available:
        return []
    vocab = int(vocab_size or getattr(tokenizer, "vocab_size", 0) or 0)
    if vocab <= 0:
        return []
    rng = random.Random(seed)
    out: list[Concept] = []
    for index in range(n_sets):
        template = available[index % len(available)]
        # Draw from the same id decile as the template's own tokens.
        low = max(min(template.token_ids) - vocab // 10, 0)
        high = min(max(template.token_ids) + vocab // 10, vocab - 1)
        if high <= low:
            low, high = 0, vocab - 1
        ids = sorted({rng.randint(low, high) for _ in range(len(template.token_ids) * 3)})
        ids = ids[:len(template.token_ids)]
        concept = Concept(name=f"random_{index}", family="random_concepts",
                          requested=(f"matched to '{template.name}' "
                                     f"({len(template.token_ids)} ids, id range "
                                     f"{low}-{high})",),
                          token_ids=[int(i) for i in ids])
        concept.spellings = [tokenizer.decode([i]) for i in concept.token_ids]
        out.append(concept)
    return out


def resolve_all(tokenizer, seed: int = 0, n_random_sets: int = 3,
                vocab_size: Optional[int] = None) -> list[Concept]:
    """Every predeclared concept plus every control, in a fixed order."""
    concepts: list[Concept] = []
    for family, table in (("binding_concept", BINDING_CONCEPTS),
                          ("generic_code", GENERIC_CODE_CONCEPTS),
                          ("positional", POSITIONAL_CONCEPTS)):
        for name, spellings in table.items():
            concepts.append(resolve_concept(tokenizer, name, spellings, family))
    concepts += random_concepts(
        tokenizer, [c for c in concepts if c.family == "binding_concept"],
        n_sets=n_random_sets, seed=seed, vocab_size=vocab_size)
    unavailable = [c.name for c in concepts if not c.available]
    if unavailable:
        logger.warning("no single-token spelling for %d concept(s): %s — scored "
                       "on nothing rather than on a truncation",
                       len(unavailable), unavailable)
    return concepts


# ── the panel's own programs ─────────────────────────────────────────────────

_HEADER = "# utility helpers\nimport os\nimport sys\n\n"

_NAMES = ("helper", "process", "collect", "gather", "resolve",
          "compute_all", "select", "combine", "expand", "reduce_all")
_OTHER = ("y", "z", "acc", "tmp", "buf", "idx", "cur", "prev", "res", "val")


@dataclass
class ConceptItem:
    """One (program, read position) the concept panel is scored at."""

    item_id: str
    base_id: str            # the program construction, shared by all four cells
    value_arm: str          # "ab" | "ba" — which literal the outer binding holds
    binding_arm: str        # "outer" | "inner" — which definition is live
    read: str               # one of READS
    prompt: str
    anchor: str
    answer_value: int       # the value in scope, for the value-invariance check
    other_value: int

    @property
    def cell(self) -> str:
        return f"{self.value_arm}_{self.binding_arm}"

    def as_dict(self) -> dict:
        return {"item_id": self.item_id, "base_id": self.base_id,
                "value_arm": self.value_arm, "binding_arm": self.binding_arm,
                "cell": self.cell, "read": self.read, "anchor": self.anchor,
                "answer_value": self.answer_value, "other_value": self.other_value}


def _programs(base: int, v_a: int, v_b: int, fname: str, other: str,
              suffix: Optional[str]) -> list[ConceptItem]:
    """Four cells of one base, each read at up to four positions.

    The construction is E19's `binding` family, crossed on the VALUE assignment
    as well as on the binding — which is E13's design, and what makes condition
    2 ("the crossed value arms agree") askable at all. In arm `ab` the outer
    definition holds `v_a` and the inner holds `v_b`; in `ba` they are swapped.
    Both arms are token-identical at the read position within a binding arm, and
    the four cells share one base id.
    """
    items: list[ConceptItem] = []
    for value_arm, (outer, inner) in (("ab", (v_a, v_b)), ("ba", (v_b, v_a))):
        common = f"{_HEADER}x = {outer}\n\n\ndef {fname}():\n"
        cells = {
            "outer": (common + f"    {other} = {inner}\n    return x", outer, inner),
            "inner": (common + f"    x = {inner}\n    return x", inner, outer),
        }
        for binding_arm, (program, in_scope, shadowed) in cells.items():
            stem = f"concept-{base}-{value_arm}-{binding_arm}"
            prompt = program + (suffix or "")
            use_anchor = "    return x"
            candidates = [("use", use_anchor)]
            if suffix:
                candidates += [("post_use", use_anchor + "\nassert"),
                               ("call", suffix.split("==")[0].rstrip()),
                               ("answer", suffix)]
            for read, anchor in candidates:
                if prompt.count(anchor) != 1:
                    continue          # ambiguous here; drop rather than guess
                items.append(ConceptItem(
                    item_id=f"{stem}-{read}", base_id=f"concept-{base}",
                    value_arm=value_arm, binding_arm=binding_arm, read=read,
                    prompt=prompt, anchor=anchor,
                    answer_value=in_scope, other_value=shadowed))
    return items


def build_panel(tokenizer, n_bases: int = 10, seed: int = 0) -> tuple[list[ConceptItem], dict]:
    """The panel's programs, built against THIS tokenizer.

    Literals are drawn from the integers the tokenizer keeps whole — the same
    constraint `evalsuite` documents, and for the same reason: both code models
    here segment every multi-digit number digit by digit, so the usable pool is
    2-9. The panel does not score values, but the programs must still be the
    ones E19's value families use, or the comparison between panels is between
    two different corpora.
    """
    import random

    from src.workspace_lens.evalsuite import (_answer_suffix, _single_token_values,
                                              _value_pairs)

    values = _single_token_values(tokenizer)
    if len(values) < 2:
        raise RuntimeError(
            f"only {len(values)} single-token integer literals for this "
            "tokenizer; the concept panel's programs cannot be built.")
    pairs = _value_pairs(values, n_bases, seed=seed)
    try:
        suffix_template = _answer_suffix(tokenizer, "f")
        answer_reads = "built"
    except Exception as exc:                                    # noqa: BLE001
        suffix_template, answer_reads = None, (
            f"unavailable: {type(exc).__name__}: {str(exc)[:120]}")
        logger.warning("no answer-position reads for this tokenizer (%s); the "
                       "three earlier reads are unaffected", answer_reads)

    items: list[ConceptItem] = []
    for index, (v_a, v_b) in enumerate(pairs):
        fname, other = _NAMES[index % len(_NAMES)], _OTHER[index % len(_OTHER)]
        suffix = (suffix_template.replace("f()", f"{fname}()")
                  if suffix_template else None)
        items += _programs(index, v_a, v_b, fname, other, suffix)
    meta = {"n_bases": len(pairs), "n_items": len(items),
            "answer_reads": answer_reads, "reads": sorted({i.read for i in items}),
            "seed": seed}
    return items, meta


# ── metrics ──────────────────────────────────────────────────────────────────

#: Predeclared thresholds. `pass@k` is the share of items whose concept is
#: inside the top k of the FULL vocabulary — the same rank the value families
#: use, from the same `readout.rank_of`.
PASS_AT_K = (1, 5, 10, 50, 100)


def summarise(rows, k_values: Sequence[int] = PASS_AT_K):
    """Per (lens, layer, read, family, concept): pass@k, median rank, n.

    Never pooled over `read` or over `lens`: the whole point of four read
    positions is that they are different questions, and the whole point of three
    readouts is that they can disagree.
    """
    import numpy as np
    import pandas as pd

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    out = []
    for (lens, layer, read, family, concept), grp in frame.groupby(
            ["lens", "layer", "read", "family", "concept"]):
        record = {"lens": lens, "layer": int(layer), "read": read,
                  "family": family, "concept": concept, "n": len(grp),
                  "median_rank": float(grp["rank"].median()),
                  "mean_rank": float(grp["rank"].mean())}
        for k in k_values:
            record[f"pass@{k}"] = float(np.mean(grp["rank"].to_numpy() < k))
        out.append(record)
    return pd.DataFrame(out).sort_values(
        ["lens", "read", "family", "concept", "layer"])


def earliest_entries(rows, k_values: Sequence[int] = PASS_AT_K):
    """First layer each concept enters the top k, per (lens, read, concept).

    "Never" is kept as a missing layer rather than as the last one: averaging a
    concept that never surfaces as though it surfaced at the final layer turns a
    failure into a late success, which is the mistake `readout.earliest_layer`
    exists to prevent. This uses that function.
    """
    import pandas as pd

    from src.workspace_lens.readout import earliest_layer

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    out = []
    for (lens, read, family, concept), grp in frame.groupby(
            ["lens", "read", "family", "concept"]):
        by_layer = grp.groupby("layer")["rank"].median().to_dict()
        record = {"lens": lens, "read": read, "family": family,
                  "concept": concept, "n_items": int(grp["item_id"].nunique())}
        for k in k_values:
            layer = earliest_layer({int(l): float(r) for l, r in by_layer.items()}, k)
            record[f"earliest@{k}"] = layer
        out.append(record)
    return pd.DataFrame(out)


def binding_contrasts(rows, n_boot: int = 2000, seed: int = 42):
    """Does the concept move WITH the binding, and does it agree across arms?

    Two paired differences per (lens, layer, read, concept), both clustered on
    the base program because the four cells of a base are one construction:

      `binding_delta`  inner-arm score minus outer-arm score, within a value
                       arm. This is the hypothesis: a concept that tracks which
                       definition is live must separate the two arms, which are
                       token-identical at the read position.
      `value_delta`    the same binding contrast computed on `ab` minus the one
                       computed on `ba`. A binding effect must be INVARIANT to
                       which literal is in scope, so this should be ~0 with an
                       interval containing zero; a large value means the concept
                       is riding on the answer token, not on the binding.

    Scores rather than ranks for the difference: a rank is a monotone but
    non-linear function of the score, and differencing two ranks near the tail
    of a 32k vocabulary is dominated by the tail's density. Ranks are still
    reported per cell; the CONTRAST is on the lens score, which is what the
    edit-free comparison actually has.
    """
    import numpy as np
    import pandas as pd

    from src.analysis.bootstrap import paired_cluster_bootstrap_ci

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    keys = ["lens", "layer", "read", "family", "concept"]
    out = []
    for key, grp in frame.groupby(keys):
        wide = grp.pivot_table(index="base_id", columns="cell", values="score")
        record = dict(zip(keys, key))
        record["n_bases"] = int(len(wide))
        for arm in ("ab", "ba"):
            inner, outer = f"{arm}_inner", f"{arm}_outer"
            if inner not in wide.columns or outer not in wide.columns:
                continue
            paired = wide[[inner, outer]].dropna()
            if len(paired) < 3:
                continue
            ci = paired_cluster_bootstrap_ci(
                paired[inner].to_numpy(), paired[outer].to_numpy(),
                paired.index.to_numpy(), n_boot=n_boot, seed=seed)
            record[f"binding_delta_{arm}"] = ci.point
            record[f"binding_delta_{arm}_lo"] = ci.lo
            record[f"binding_delta_{arm}_hi"] = ci.hi
        cols = {"ab_inner", "ab_outer", "ba_inner", "ba_outer"}
        if cols.issubset(wide.columns):
            paired = wide[sorted(cols)].dropna()
            if len(paired) >= 3:
                ci = paired_cluster_bootstrap_ci(
                    (paired["ab_inner"] - paired["ab_outer"]).to_numpy(),
                    (paired["ba_inner"] - paired["ba_outer"]).to_numpy(),
                    paired.index.to_numpy(), n_boot=n_boot, seed=seed)
                record["value_delta"] = ci.point
                record["value_delta_lo"] = ci.lo
                record["value_delta_hi"] = ci.hi
                # Both arms must move the SAME way for the crossing to agree.
                a = record.get("binding_delta_ab", float("nan"))
                b = record.get("binding_delta_ba", float("nan"))
                record["crossed_agreement"] = bool(
                    np.isfinite(a) and np.isfinite(b) and np.sign(a) == np.sign(b)
                    and a != 0)
        out.append(record)
    return pd.DataFrame(out)


def verdict(contrasts, alpha_family: str = "binding_concept") -> dict:
    """The four-condition read, applied — never a post-hoc "which word won".

    A supported positive needs every one of:

      * a predeclared binding concept whose binding contrast excludes zero,
      * in BOTH value arms, with the same sign (`crossed_agreement`),
      * with a value-invariance interval that CONTAINS zero (the effect does not
        depend on which literal is in scope),
      * and a movement larger than the best matched generic/positional control
        at the same (lens, layer, read).

    Anything less is reported as a null, and a null here says only that the
    published linear token-indexed lenses do not surface these concepts. It is
    not evidence against the probe or DAS results, which read a different object
    by a different method.
    """
    import numpy as np
    import pandas as pd

    if contrasts is None or len(contrasts) == 0:
        return {"supported": False, "reason": "no contrasts computed",
                "n_candidates": 0}

    frame = pd.DataFrame(contrasts)
    needed = {"binding_delta_ab", "binding_delta_ab_lo", "binding_delta_ba",
              "binding_delta_ba_lo", "crossed_agreement", "value_delta_lo",
              "value_delta_hi"}
    if not needed.issubset(frame.columns):
        return {"supported": False, "n_candidates": 0,
                "reason": f"contrast table is missing {sorted(needed - set(frame.columns))}"}

    def excludes_zero(row, arm):
        lo, hi = row[f"binding_delta_{arm}_lo"], row[f"binding_delta_{arm}_hi"]
        return bool(np.isfinite(lo) and np.isfinite(hi) and (lo > 0 or hi < 0))

    candidates = []
    for _, row in frame[frame["family"] == alpha_family].iterrows():
        if not (excludes_zero(row, "ab") and excludes_zero(row, "ba")):
            continue
        if not bool(row.get("crossed_agreement")):
            continue
        lo, hi = row.get("value_delta_lo"), row.get("value_delta_hi")
        if not (np.isfinite(lo) and np.isfinite(hi) and lo <= 0 <= hi):
            continue
        peers = frame[(frame["lens"] == row["lens"]) & (frame["layer"] == row["layer"])
                      & (frame["read"] == row["read"])
                      & (frame["family"].isin(("generic_code", "positional",
                                               "random_concepts")))]
        control = (float(np.nanmax(np.abs(peers["binding_delta_ab"].to_numpy())))
                   if len(peers) else 0.0)
        if abs(float(row["binding_delta_ab"])) <= control:
            continue
        candidates.append({"lens": row["lens"], "layer": int(row["layer"]),
                           "read": row["read"], "concept": row["concept"],
                           "binding_delta_ab": float(row["binding_delta_ab"]),
                           "binding_delta_ba": float(row["binding_delta_ba"]),
                           "best_control": control})
    return {
        "supported": bool(candidates),
        "n_candidates": len(candidates),
        "candidates": candidates[:20],
        "reason": ("predeclared binding concepts separate the arms in both value "
                   "arms, agree in sign, are invariant to the literal, and beat "
                   "every matched control"
                   if candidates else
                   "no predeclared binding concept met all four conditions. This "
                   "is a null about the published linear token-indexed J/R "
                   "lenses at these positions, and says nothing about the probe "
                   "or DAS evidence, which read a different object."),
    }
