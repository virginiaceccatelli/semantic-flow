# E15 — Do code models see through obfuscation?

**Auditing a security-relevant semantic representation: does the value that
reaches a code-bearing, sensitive argument come from untrusted input, and does a
*frozen* readout of that fact survive the obfuscation ladder?**

Status: **active**. Built, gated, and smoke-tested end to end on
`deepseek-coder-1.3b`; **not run at canonical scale**, so nothing here is a
result. `results/STATUS.yaml` is the registry of record.

---

## 1. Threat model

The auditor is a defender reading code that someone else may have written to be
hard to read. They want one bit per program:

> Is the value passed to this security-sensitive, **code-bearing** argument
> derived from untrusted input?

Three sink families, each with a real CWE behind it:

| family | untrusted source | sensitive argument |
|---|---|---|
| `command_exec` | `request.args.get(...)`, `sys.argv[1]` | `os.system(·)`, `subprocess.call(·, shell=True)` |
| `sql_exec` | `request.args.get(...)`, `request.form.get(...)` | `cursor.execute(·)` |
| `dynamic_exec` | `request.form.get(...)`, `input()` | `eval(·)`, `exec(·)` |

The adversary in the threat model is **the surface form, not the model**: the
program may be renamed, padded with opaque dead branches, have its arithmetic
rewritten, and have its control flow dissolved into a dispatch loop — all
meaning-preserving. What must not change is the answer to the question above.

**What counts as safe.** Not a sanitizer. The generic sanitizer list used
elsewhere in this repository (`html.escape`, `shlex.quote`) is deliberately not
reused here, because `html.escape` before `exec` and `shlex.quote` before `eval`
are not mitigations, and a benchmark whose "safe" class was built from them
would be labelling vulnerable programs safe. The safe member instead passes an
**independently trusted literal** — a constant that never touches the source —
to the same sink.

**What this is not.** No causal claim. E15 is observational: it asks what a
linear readout of the model's states can recover and what breaks it. E13's
interchange is the causal instrument and is strictly stronger for that purpose.

## 2. The benchmark

```
3 sink families x 4 flow structures x 20 base seeds x 2 labels = 480 clean programs
```

Flow structures, each exercising a different thing a reader has to do:

| structure | what it adds |
|---|---|
| `direct` | the source's value goes straight to the sink |
| `assign_chain` | two aliasing steps between source and sink |
| `branch_merge` | two definitions reach the sink through a join point |
| `helper` | one function-call boundary (`relay(x) -> x`) |

Every base seed is a **matched pair**:

```python
def func(request):
    count = 3
    param = request.args.get("cmd")        # the source
    detail = "systemctl status"            # the independently trusted value
    param_1 = relay(param)
    detail_1 = relay(detail)
    count = count + 1
    os.system(param_1)                     # unsafe   ← the only difference
    os.system(detail_1)                    # safe     ←
```

Both members hold the same source, the same propagation, the same trusted
alternative and the same sink. `pair_diff_is_confined_to_sink_arg` checks
character-exactly that everything before and after the sink-argument span is
identical — the invariant is *verified*, not asserted, and the same check is
re-run on every obfuscated variant (both members of a base are obfuscated with
the same draw, so the pair stays matched at every level).

**The name cue is balanced away.** Which of the two chain names carries the
tainted value alternates with the base index, and so does the declaration order.
The token at the anchor is therefore uninformative about the label across the
corpus, which is what keeps the measured surface baseline near chance instead of
handing it the answer. `count` exists so the opaque-predicate and
mixed-boolean-arithmetic levels of the ladder have an integer to bite on.

## 3. Labels: recomputed, never trusted

Two readings, computed differently, must agree with each other *and* with the
intended label, or the program is refused (`sink_flow.recover_label`):

* **instrumented execution** (`observe_program`) — the module runs under stubs
  with a provenance-carrying `str` subclass standing in for untrusted input.
  `os.system`, `subprocess.*`, `cursor.execute`, `eval` and `exec` are
  recorders, and the module executes with `__builtins__ = {}`. **No dangerous
  API is ever executed**, at any obfuscation level, even if a generated program
  were wrong. This reading is flow-sensitive because it is the execution.
* **a static taint fixpoint** (`static_sink_label`) — flow-insensitive over the
  AST, matched on *call shapes* (attribute chains, `input`, `sys.argv`) rather
  than variable names, so it still reads a renamed and control-flow-flattened
  variant. Interprocedural to one level via parameter→return summaries, which is
  what the `helper` structure needs.

Flow-insensitivity is an over-approximation, and that is the point of pairing it
with execution: the two disagree exactly on the programs whose label nobody
could defend, and those are refused rather than reconciled. This is the
discipline `src/data/store_semantics.py` established for E12/E13 and
`tests/test_ground_truth_crosscheck.py` for def–use.

## 4. Split, training, and the frozen evaluation

* 14 of the 20 seeds per (family, structure) → **clean training**;
  6 → **held out**. Whole bases move; both members and every variant of a base
  inherit the split. Stratification is per cell, so no family or structure is
  over-represented in either side.
* Only **held-out** programs are obfuscated, with E9's existing ladder, unchanged
  and unextended: `0 normalize · 1 rename · 2 opaque · 3 encode · 4 flatten`.
* The readout is fitted **once**, on clean training programs, and then frozen —
  never refitted on a variant, so a change in accuracy across the ladder is a
  change in the model's state rather than in the probe (E5/E9's rule).

Reported separately by **family, structure, obfuscation level, model and layer**.
The pooled row is present but is not the finding: a readout that holds on
`direct` flows and fails across the helper boundary is a different result from
one that degrades evenly, and only the per-cell rows can tell them apart.

## 5. Controls

| control | what it kills |
|---|---|
| **measured surface baseline** — ±3 token ids around the anchor, no hidden states, **frozen and transferred through the ladder** | "the identifier gives it away". Because it is frozen too, level 1 (rename) measures what renaming does to a lexical shortcut instead of leaving it as an argument |
| **embedding layer (−1)** | token identity before any computation happened |
| **selectivity control** — the same probe on labels shuffled within each base | accuracy from class priors or per-base regularities |
| **grouped CV by base** | the two members of a pair share hidden structure; ungrouped folds leak one into the other |
| **role/order balancing in the generator** | a corpus-wide "this name means tainted" shortcut |
| **`last_token` reported separately from `sink_arg`** | a headline averaged over two sites that answer differently |

## 6. Validity gates

Every stage refuses to run on a failed prerequisite (exit 2), through the same
registry as E12/E13 (`src/experiments/store_gates.py::SINKFLOW`,
`results/sinkflow/{model}/gates.yaml`). `--override-gate REASON` is permitted and
is recorded permanently in the gate file and in the manifest.

| gate | stage | asserts |
|---|---|---|
| **S0** | 120 | exactly 480 clean programs; exact balance across family × structure × label; no base or pair leakage across splits, and 14 training bases per cell; every program parses; source and sink anchors covered exactly by tokenizer positions; every label independently recovered by both readings; every pair differs only in the sink-argument span; every obfuscated variant parses and preserves its label; all requested levels present for every held-out base; only held-out bases obfuscated |
| **S1** | 121 | activations exist for every program in every shard, with no skips, and every anchor lands on a token boundary **in the encoding that was stored** (truncation is the step that can silently move one) |
| **S2** | 122 | the readout saw the clean training split and nothing else; the selectivity control, the embedding layer and the no-hidden-state surface baseline all actually ran; the probe beats its own shuffled-label control somewhere |
| **S3** | 123 | the probe's provenance record shows training bases disjoint from every evaluated base and a training digest matching the shard on disk; the result row count equals the count the design predicts; both classes are present in every reported cell; the surface arm produced rows |

A failed gate prints which gate failed, the expected and observed values, the
offending ids and the exact command to rerun. **Nothing is repaired by dropping
the offending programs** — that would report a smaller benchmark as if it were
the designed one — and no partial headline is emitted as valid.

## 7. Commands

```bash
# CPU only — no model, no GPU
python -m pytest tests/test_sink_flow.py -q
python scripts/120_sinkflow_generate.py --model deepseek-coder-1.3b     # S0

# GPU (MPS is fine for 1.3b)
python scripts/121_sinkflow_extract.py --model deepseek-coder-1.3b      # S1

# CPU
python scripts/122_sinkflow_probe.py       --model deepseek-coder-1.3b  # S2
python scripts/123_sinkflow_obfuscation.py --model deepseek-coder-1.3b  # S3
python scripts/124_sinkflow_report.py      --model deepseek-coder-1.3b

# all of it
make sinkflow MODEL=deepseek-coder-1.3b
make sinkflow-smoke                  # 96 programs, 3 layers, minutes on a laptop
```

Outputs land in `results/sinkflow/{model}/`: `benchmark.csv`, `gates.yaml`,
`sinkflow_clean.csv`, `sinkflow_obfuscation.csv`, `sinkflow_predictions.csv`,
`e15_report.{md,yaml}`, `probes/{site}/{layer_XX,surface}.pkl` and
`probes/provenance.json`; figures in `results/figures/sinkflow_*.png`.

## 8. Limitations, stated before any number is read

1. **The floor is not pinned to chance by construction the way E2's is.** It is
   pinned only against the *declared* surface family (a ±3 token window at the
   anchor). A predictor with the whole program text could recover the label by
   performing the taint analysis itself. E15 is therefore an audit of a
   readout's transfer, not a representation claim of E2's kind — and the
   surface baseline is reported beside every number rather than in a footnote.
2. **Synthetic programs, one language, four flow structures.** The structures
   are the ones a taint analysis has to handle, not a sample of real code. E8's
   caveat applies here too: transfer to naturalistic code is untested.
3. **The sink families are the common ones, not a taxonomy.** Three families
   with two sink spellings each; nothing here says anything about sinks not in
   the list.
4. **The static reading is flow-insensitive.** It is sound *for this generator*
   because no chain variable is ever assigned the other chain's value, and that
   property is what the execution reading independently checks. It is not a
   general-purpose taint analyser and must not be reused as one.
5. **Level 4 changes what "the source anchor" means.** After flattening, the
   first source expression in source order is whichever dispatch case the
   shuffle put first. The `sink_arg` anchor is unaffected, which is why it, not
   the source anchor, is the headline site.
6. **Nothing causal.** A frozen readout surviving a transformation says the
   information is still linearly present, not that the model uses it.
