# E15 — Do code models see through obfuscation?

**Auditing a security-relevant semantic representation: does the value that
reaches a code-bearing, sensitive argument come from untrusted input, does a
*frozen* readout of that fact survive obfuscation, and is the difference
expressed in the model's own vocabulary?**

Three experiments over one benchmark, one set of matched pairs and one set of
activations. **All three have run at canonical scale on `deepseek-coder-1.3b`,
`deepseek-coder-6.7b` and `starcoder2-3b`, with all six gates passing and no
overrides recorded.**

| | question | stages | result |
|---|---|---|---|
| **A** atomic + cumulative robustness | which transformation breaks the frozen readout *on its own*, and what does composing them add? | 120–124 | **Flattening alone accounts for the entire collapse.** Opaque predicates and MBA encoding cost *exactly* nothing; the interaction is inside the draw-noise floor in all three models (§8.2) |
| **B** whole-program lexical baseline | could a reader of the entire program text recover the label with no hidden state? | 122–124 | **No** — 0.465–0.535 in every condition and model, against 1.000 for the hidden state (§8.1) |
| **C** vocabulary-space contrast | after mapping the sink state into the model's own output basis, which vocabulary directions separate unsafe from its matched safe counterfactual? | 125–127 | **A null.** No security-vocabulary concept in any model; 1.3B's contrast is significantly *inverted*. All three lenses agree, so it is not a lens artifact (§8.6) |

**The one-sentence result.** The property is decodable at ceiling over two
measured floors; **control-flow flattening alone destroys the readout**, while
renaming costs little and opaque predicates and arithmetic rewriting cost
nothing; what survives flattening is each model's class prior rather than flow
information; and the distinction is *not* expressed in the models' own output
vocabulary.

`results/STATUS.yaml` is the registry of record, and the limitations in §9 are
not optional reading — the floor here is weaker than E2's by construction, and
"flattening breaks the readout" is a claim about a frozen linear readout, not
about what the model retains.

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
* Only **held-out** programs are transformed, with E9's existing rewrites,
  unchanged and unextended.
* The readout is fitted **once**, on clean training programs, and then frozen —
  never refitted on a variant, so a change in accuracy across a condition is a
  change in the model's state rather than in the probe (E5/E9's rule).

### 4.1 The ten conditions

The first run evaluated the cumulative ladder only, which made "flattening
breaks the readout" a *marginal* claim: level 4 contains four transformations
and nothing in the data could say which one did the damage. The atomic arms fixed
exactly that, and the answer is in §8.2: **flattening alone**. **No new
obfuscation algorithm is introduced, and no
arbitrary pairwise combination is generated** — the same four rewrites are simply
made addressable one at a time (`ObfuscationLadder.apply_steps`).

| condition | kind | transformations |
|---|---|---|
| `clean_heldout` | clean | — (the untransformed held-out shard) |
| `normalize` | baseline | ast round-trip only |
| `rename_only` | atomic | rename |
| `opaque_only` | atomic | opaque |
| `encode_only` | atomic | encode |
| `flatten_only` | atomic | flatten |
| `rename_cumulative` | cumulative | rename |
| `rename_opaque` | cumulative | rename → opaque |
| `rename_opaque_encode` | cumulative | rename → opaque → encode |
| `rename_opaque_encode_flatten` | cumulative | rename → opaque → encode → flatten |

`144 held-out programs × 9 transformed conditions = 1296 variants` per model.

Three differences are computed on every reported cell, and they answer three
different questions:

* `delta_clean` — change from clean held-out. What the condition costs.
* `delta_previous` — for a cumulative condition, the change from the condition
  one step shorter. **The only column that supports "adding X costs Y".**
* `delta_atomic` — cumulative minus its atomic counterpart: the **interaction**,
  the part of a cumulative failure the transformation does not produce alone.

**`rename_only` and `rename_cumulative` apply the identical transformation set**
— the cumulative prefix of length one *is* renaming — under two independent
draws. That row of the interaction table is therefore a measured **draw-noise
floor**, and an interaction smaller than it is not an interaction. This is
stated in the report itself, not left for a reader to notice.

**Attribution rule.** Flattening is named as the cause only where `flatten_only`
supports it. Everywhere else the result is described as a cumulative effect.

Reported separately by **family, structure, condition, model, layer, site and
class**. The pooled row is present but is not the finding: a readout that holds
on `direct` flows and fails across the helper boundary is a different result
from one that degrades evenly, and only the per-cell rows can tell them apart.

### 4.2 Verified conditions, not asserted ones

A condition that quietly did more than it claims would make every attribution
wrong, and a draw that happened to do *nothing* would dilute the arm that names
it — two of the rewrites are probabilistic (the MBA encoder rewrites an addition
with p=0.6 and an int constant with p=0.5, so about a fifth of `encode` draws
change nothing). So the transformations a variant carries are **read off its own
AST** (`detect_transformations`) by signatures the others cannot produce:

| detected | signature |
|---|---|
| rename | the program's own counter `count` is gone |
| opaque | an opaque guard `<expr> % k == c` — a shape the generator never emits |
| encode | a bitwise operator (`^ & << ~`), which only the MBA identities introduce |
| flatten | a `while` dispatch loop; the generator emits no loops at all |

Both members of a base are transformed **together under one draw**, and the draw
is redrawn (up to 8 times, recorded per variant) until the variant is the
condition it claims to be *and* the pair still differs only at the sink argument.
Nothing is dropped: a base that never satisfies its own condition is emitted
marked as failing, and S0 refuses the run.

## 5. Controls

| control | what it kills |
|---|---|
| **local surface baseline** — ±3 token ids around the anchor, no hidden states, **frozen and transferred through every condition** | "the identifier gives it away". Because it is frozen too, `rename_only` measures what renaming does to a lexical shortcut instead of leaving it as an argument |
| **whole-program lexical baseline** (§12) — token uni/bigrams and char 3–5-grams over the ENTIRE program, no hidden states, frozen the same way | "the generator left something in the text". This is the floor the ±3 window structurally cannot see, and it is the one E15's limitation §9.1 has always been about |
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
| **S0** | 120 | exactly 480 clean programs; exact balance across family × structure × label; no base or pair leakage across splits, and 14 training bases per cell; every program parses; source and sink anchors covered exactly by tokenizer positions; every label independently recovered by both readings; every pair differs only in the sink-argument span. **And per condition:** exact counts (144 per condition, 1296 in total); every held-out program present in every condition; only held-out bases transformed; each variant carries **exactly** the transformations its condition declares (read off the AST, §4.2); both members of a pair transformed under the same draw; the transformed pair still differs only at the sink argument; every variant inherits its base's split |
| **S1** | 121 | activations exist for every program in every shard **and every condition**, with no skips, and every anchor lands on a token boundary **in the encoding that was stored** (truncation is the step that can silently move one) |
| **S2** | 122 | the readout saw the clean training split and nothing else; the selectivity control, the embedding layer, the local surface baseline and the whole-program lexical baseline all actually ran; all four arms exist; the lexical vectorizer's provenance records that it was fitted on clean training text alone; the probe beats its own shuffled-label control somewhere |
| **S3** | 123 | the probe's provenance record shows training bases disjoint from every evaluated base and a training digest matching the shard on disk; the result row count equals the count the design predicts; both classes are present in every reported cell; all four arms produced rows; every atomic and cumulative condition the design declares is present; the per-class and matched-pair metrics exist in every pooled hidden-state cell |
| **J0** | 125 | the LRP instrumentation leaves the forward logits unchanged within the relative tolerance; all three lenses exist at every requested layer; every lens carries exactly the frozen candidate token ids in order; every lens artifact belongs to this model and checkpoint; no lens vector is NaN or infinite; at least one unsafe- and one safe-oriented concept token survived the tokenizer check |
| **J1** | 126 | one row per (lens, layer, site, condition, base) with each safe member matched to exactly one distinct unsafe member; one recorded orientation everywhere; discovery ran on the training split and its digest is recorded and differs from the evaluated split's; the frozen token set exists; every declared cell is present; no contrast or token delta is NaN; the permutation and mismatched-pair controls ran |

A failed gate prints which gate failed, the expected and observed values, the
offending ids and the exact command to rerun. **Nothing is repaired by dropping
the offending programs** — that would report a smaller benchmark as if it were
the designed one — and no partial headline is emitted as valid.

**A gate validates the experiment, not the hypothesis.** Only mechanical and
data-integrity failures block: instrumentation that moved the forward logits,
missing pairs/layers/anchors/vocabulary rows, an inconsistent orientation,
train/held-out overlap, inconsistent token ids or vocabulary dimensions, NaN or
infinity, a result count that does not match the design, or a transformed
program that failed label validation. **No gate anywhere requires a positive
security-token result**, and J0/J1 must pass when the semantic result is null.

Lens *quality* is a different thing and is never blocking. Weak next-token
recovery, weak agreement with the final-layer distribution and poor relevance
conservation are measured per (layer, lens), emit warnings, and leave the layer
in the experiment — refusing there would restrict the study to the layers where
the instrument is comfortable, and early and middle layers are the target. The
report separates the four outcomes explicitly: **mechanically invalid**,
**mechanically valid with weak lens fidelity**, **valid null semantic result**,
and **positive semantic result above controls**.

## 7. Commands

The complete, ordered, copy-pasteable pipeline — CPU/GPU marked, and the gate
each stage must pass before the next one may run — is **§14**. The short version:

```bash
# CPU only — no model, no GPU
python -m pytest tests/test_sink_flow.py tests/test_sinkflow_vocab.py -q
python scripts/120_sinkflow_generate.py --model deepseek-coder-1.3b     # S0

# GPU (MPS is fine for 1.3b)
python scripts/121_sinkflow_extract.py --model deepseek-coder-1.3b      # S1

# CPU
python scripts/122_sinkflow_probe.py       --model deepseek-coder-1.3b  # S2
python scripts/123_sinkflow_obfuscation.py --model deepseek-coder-1.3b  # S3
python scripts/124_sinkflow_report.py      --model deepseek-coder-1.3b

# E15-C: GPU once, then CPU
python scripts/125_sinkflow_vocab_discover.py --model deepseek-coder-1.3b  # J0
python scripts/126_sinkflow_vocab_contrast.py --model deepseek-coder-1.3b  # J1
python scripts/127_sinkflow_vocab_report.py   --model deepseek-coder-1.3b

# all of it
make sinkflow MODEL=deepseek-coder-1.3b
make sinkflow-vocab-all MODEL=deepseek-coder-1.3b
make sinkflow-smoke                  # 96 programs, 3 layers, minutes on a laptop
make sinkflow-vocab-smoke            # 2 layers, 24 candidate tokens

# cross-model reading: always at MATCHED RELATIVE DEPTH, never at layer index
python scripts/124_sinkflow_report.py --model deepseek-coder-1.3b --depth 0.48
python scripts/124_sinkflow_report.py --model deepseek-coder-6.7b --depth 0.48
python scripts/124_sinkflow_report.py --model starcoder2-3b       --depth 0.48
```

Outputs land in `results/sinkflow/{model}/`: `benchmark.csv`, `gates.yaml`,
`sinkflow_clean.csv`, `sinkflow_obfuscation.csv`, `sinkflow_predictions.csv`,
`e15_report.{md,yaml}`, `probes/{site}/{layer_XX,surface,whole_program_lexical}.pkl`
and `probes/provenance.json`; the E15-C artifacts under
`results/sinkflow/{model}/vocab/`; figures in `results/figures/sinkflow_*.png`.

## 8. Results — three models, all six gates, no overrides

480 clean programs, 336 training / **144 held-out (72 bases) per condition**,
1296 transformed variants, 1776 programs extracted per model. S0–S3 and J0/J1
pass in every run with **no overrides recorded**. Intervals are
cluster-bootstrapped over base programs.

Headline site is `sink_arg`, read at **the layer nearest 48% of network depth**:
1.3B layer 11 (48%), 6.7B layer 15 (48%), starcoder2-3b layer 15 (52% — its grid
has nothing closer). In all three that layer is also the argmax of clean-training
CV, so the depth match and the best-layer choice agree.

> **These numbers supersede the earlier cumulative-ladder-only run.** Adding the
> atomic conditions regenerated the benchmark, which redraws every
> transformation, so figures moved — 1.3B's flattening result reads 0.729 for the
> cumulative condition where the old shard read 0.632. Do not mix generations.

### 8.1 The property is decodable, and it is not the text

Clean training programs, grouped CV at `sink_arg`. Two floors now, not one:

| | 1.3B | 6.7B | starcoder2-3b |
|---|---:|---:|---:|
| **local surface** (±3 token ids, no hidden states) | 0.491 | 0.491 | 0.488 |
| **whole-program lexical** (token n-grams + char 3–5-grams, whole file) | 0.464 | 0.464 | 0.464 |
| −1 embedding | 0.482 | 0.482 | 0.482 |
| **~48% depth** | **1.000** | **1.000** | **0.997** |

Held out, across all ten conditions, the floors stay at chance: local surface
0.431–0.521, whole-program lexical 0.465–0.535, embedding 0.451–0.569.

The second floor is the one that matters for §9.1. The old limitation said the
floor was pinned only against a *local* feature family and that "a predictor with
the whole program text could recover the label". Half of that objection is now
answered: a frozen reader of the **entire program** never leaves chance. The
other half stands — a reader that runs the taint analysis would score 1.0, and
nothing here bounds it.

### 8.2 Flattening alone accounts for the entire collapse

`sink_arg` at matched depth, 144 programs / 72 bases per row:

| condition | kind | 1.3B (L11) | 6.7B (L15) | starcoder2-3b (L15) |
|---|---|---:|---:|---:|
| clean held-out | — | **1.000** | **1.000** | **1.000** |
| normalize | baseline | 1.000 | 1.000 | 1.000 |
| `rename_only` | atomic | 0.938 [0.889, 0.972] | 0.986 [0.965, 1.000] | 0.882 [0.833, 0.931] |
| `opaque_only` | atomic | **1.000** | **1.000** | **1.000** |
| `encode_only` | atomic | **1.000** | **1.000** | **1.000** |
| **`flatten_only`** | atomic | **0.688** [0.618, 0.750] | **0.667** [0.597, 0.729] | **0.660** [0.583, 0.736] |
| `rename_cumulative` | cumulative | 0.958 | 0.951 | 0.910 |
| `rename_opaque` | cumulative | 0.944 | 0.965 | 0.931 |
| `rename_opaque_encode` | cumulative | 0.951 | 0.965 | 0.938 |
| **full ladder (R+O+E+F)** | cumulative | **0.729** [0.660, 0.799] | **0.653** [0.590, 0.715] | **0.674** [0.604, 0.743] |

Three readings, in order of importance:

1. **Two of the four transformations are free.** `opaque_only` and `encode_only`
   are *exactly* 1.000 in every model. Dead branches under opaque predicates and
   mixed-boolean-arithmetic rewriting cost the readout nothing at all.
2. **Flattening alone costs 0.312 / 0.333 / 0.340** — within 0.03 of what the
   whole four-transformation composition costs (0.271 / 0.347 / 0.326).
3. **The interaction is null.** Cumulative minus its atomic counterpart is
   **+0.042 / −0.014 / +0.014**, against a *measured* draw-noise floor of
   **0.021 / 0.035 / 0.028** (`rename_only` vs `rename_cumulative`: identical
   transformation, independent draws). No model shows an interaction
   distinguishable from noise.

This is the attribution §9.5 has been waiting for. The sentence the design may
now write is **"control-flow flattening breaks the readout"**, not "the
cumulative ladder does". Composition is not the threat; one transformation is.

### 8.3 What survives flattening is class bias, not flow

| at `flatten_only` | 1.3B | 6.7B | starcoder2-3b |
|---|---:|---:|---:|
| accuracy | 0.688 | 0.667 | 0.660 |
| unsafe / safe | 0.625 / 0.750 | **0.833 / 0.500** | 0.667 / 0.653 |
| matched pairs given the same label | 0.514 | 0.556 | 0.458 |

| at the full ladder | 1.3B | 6.7B | starcoder2-3b |
|---|---:|---:|---:|
| unsafe / safe | 0.667 / 0.792 | **0.861 / 0.444** | **0.569 / 0.778** |
| matched pairs given the same label | 0.431 | 0.583 | 0.486 |

About half the matched pairs collapse to a single label, and the residual
accuracy biases in **opposite directions** across models — 6.7B toward "unsafe",
starcoder2-3b toward "safe". A constant predictor of either class scores exactly
0.500 on this balanced set, so accuracies within 0.08 of each other produced by
opposite biases are each model's prior, not retained flow information. Quoting
"≈0.67 retained" from the accuracy column alone would be wrong three times over.

### 8.4 The dangerous errors arrive before any structural change

| `rename_only`, matched depth | accuracy | on unsafe | on safe | pairs same |
|---|---:|---:|---:|---:|
| 1.3B | 0.938 | 0.917 | 0.958 | 0.097 |
| 6.7B | 0.986 | 0.972 | 1.000 | 0.028 |
| **starcoder2-3b** | 0.882 | **0.764** | 1.000 | 0.236 |

Starcoder2's entire renaming loss is **false negatives**: it keeps calling safe
programs safe and starts calling vulnerable programs safe too — after nothing but
consistent identifier renaming, with the control flow untouched. For an audit
readout that is the failure direction that matters, and pooled accuracy hides it.

### 8.5 Where the degradation lives: structure, not sink family

Per structure at matched depth (36 programs per cell), `rename_only` / `flatten_only`:

| structure | 1.3B | 6.7B | starcoder2-3b |
|---|---:|---:|---:|
| `direct` | 1.000 / 0.722 | 1.000 / 0.639 | 0.972 / 0.639 |
| `branch_merge` | 1.000 / 0.694 | 1.000 / 0.833 | 1.000 / 0.806 |
| `helper` | 0.972 / 0.639 | 0.972 / 0.667 | 0.917 / 0.556 |
| `assign_chain` | **0.778** / 0.694 | 0.972 / **0.528** | **0.639** / 0.639 |

The **assignment chain is the fragile structure under renaming** in all three
models, with the helper boundary next; `branch_merge` is untouched. A merge point
being *more* robust than a two-step alias chain is the opposite of what "longer
chain = harder" predicts, and it now has three replications. By sink family the
picture is flat with no ordering that reproduces — the null the design wanted:
the readout tracks flow, not which dangerous API is at the end of it.

### 8.6 E15-C — no security vocabulary, in any model

The observational contrast (§13) is **a null**, and a clean one.

| at the reported layer, `sink_arg`, clean held-out | 1.3B | 6.7B | starcoder2-3b |
|---|---:|---:|---:|
| concept tokens that survive the tokenizer | `" vulnerable"` | `" vulnerable"` | `" unsafe"` |
| held-out sign consistency (R-lens) | **0.153** | 0.403 | 0.694 |
| permutation p | 0.000 | 0.004 | 0.008 |
| verdict | **inverted** | stable non-security | stable non-security |

The security lexicon does not carry the contrast in any model, and the direction
is not even consistent: 1.3B is significantly **inverted** — 85% of pairs put
*less* unsafe-pole mass on the unsafe member — while starcoder2-3b leans the
hypothesised way without reaching the declared 0.70 threshold.

Four things make this a real null rather than a failed measurement:

* **The three lenses agree.** Pairwise cosine of their mean vocabulary-difference
  vectors is 0.90/0.96/0.97 (1.3B), 0.91/0.96/0.97 (6.7B), 0.75/0.96/0.77
  (starcoder2-3b). The null is not an artifact of choosing the R-lens.
* **It is not token identity.** At the embedding layer the contrast is null
  (p = 0.81 / 0.80 / 0.71), and **75% of pairs share the same anchor token** at
  `sink_arg` anyway.
* **Something does replicate — it just is not security.** Frozen
  training-discovered tokens reappear in the held-out top-k at 0.875 / 0.750 /
  0.875 against 0.000 / 0.000 / 0.031 for random control tokens. The tokens
  themselves are `" ?"`, `"?."`, `"??"` (1.3B), `" liber"`, `"clean"`, `"tbl"`
  (6.7B), `"OrNull"`, `"displayMode"`, `"fuchsia"` (starcoder2-3b).
* **Both readouts fail together under flattening.** The vocabulary contrast
  degrades with the probe (sign consistency 0.389 / 0.472 / 0.583, p = 0.014 /
  0.782 / 0.392), so this is the design's "loss of both trained and output-aligned
  auditability" outcome, not a dissociation.

**What this licenses:** *linear decodability and expression in the model's own
output vocabulary are different properties, and E15 exhibits the first without
the second.* What it does not license is any sentence containing "the model
represents unsafe".

### 8.7 Diagnostics — measured, warned about, never blocking

R-lens relevance conservation is **1.0001** (1.3B) and **0.9993** (6.7B) —
essentially exact, reproducing E14's gate-R target — but **0.154** on
starcoder2-3b, so the LRP rules do *not* conserve relevance on that architecture
and its R-lens numbers carry a fidelity caveat. Agreement with the final-layer
vocabulary distribution runs 0.18–0.47, below the 0.30 warning threshold in
several cells. Next-token recovery is unmeasurable (`n_eval = 0`): a 196-token
candidate vocabulary rarely coincides with the true next token.

None of this blocked anything, by design (§13.8). The experiment is mechanically
valid; the instrument is a stated caveat on one model.

### 8.8 The companion E9 run — the boundary is general

E9 is now complete on all three models, which settles the specificity question
§10.1 left open. At each model's best layer:

| | rename | flatten |
|---|---:|---:|
| binding | 0.708 – 0.883 | 0.527 – 0.615 |
| def–use | 0.689 – 0.864 | 0.402 – 0.545 |
| **E15 security flow** | **0.882 – 0.986** | **0.660 – 0.688** |

The same transformations break binding and def–use the same way, and the security
readout is **at least as robust** as the primitives it rests on. The supported
claim is *"structural obfuscation breaks frozen linear readouts of program
relations, security ones included"* — **not** "security representations are
specifically fragile". That caveat is now retired on evidence.
## 9. Limitations, stated before any number is read

1. **The floor is pinned only against declared feature families.** Two are now
   measured — a ±3-token window and a whole-program lexical reader — and both sit
   at chance in every condition (§8.1). But a predictor that ran the taint
   analysis itself would score 1.0, and nothing here bounds it. E15 is an audit
   of a readout's *transfer*, not a construction-pinned representation claim of
   E2's kind.
2. **Synthetic programs, one language, four flow structures.** The structures are
   the ones a taint analysis has to handle, not a sample of real code. Transfer
   to naturalistic code is untested.
3. **The sink families are the common ones, not a taxonomy.**
4. **The static reading is flow-insensitive.** Sound *for this generator*, and the
   execution reading independently checks that. Not a general-purpose taint
   analyser and must not be reused as one.
5. **Eight arms, not the full lattice.** Four atomic and four cumulative
   conditions — not all 15 combinations. An interaction between, say, opaque
   predicates and flattening *without* renaming is not measurable here, and no
   sentence may imply it is.
6. **"Flattening breaks the readout" is a claim about a FROZEN LINEAR readout at
   one position.** A probe that fails says the information is not linearly
   present there for that probe — not that the model has lost it. §8.6 shows the
   output-aligned readout fails alongside it, which is consistent with real loss
   but does not establish it.
7. **The selectivity control is weak here by construction.** Two rows per base, so
   shuffling can only swap them. The load-bearing floors are the two measured
   baselines and the embedding layer; do not quote selectivity as the margin.
8. **The models were probed on different layer grids** (8 for 1.3B, 10 for the
   other two) because `ModelConfig` computes its own default and
   `configs/models.yaml`'s `probe_layers` are not read. That is a genuine repo
   defect, still open, affecting every experiment. Its consequence here is
   closed: `relative_depth` is a column on every row and every cross-model number
   in §8 is read at matched depth.
9. **The embedding control is one measurement, not three.** At layer −1 the probe
   reduces to a lookup on the anchor token, and the benchmark's fixed identifier
   pool induces the same partition under both tokenizers, so the −1 predictions
   are byte-identical across models. It is a real control; it is not three.
10. **E15-C is observational, and its search is restricted.** No intervention of
    any kind. The J/R candidate pool is logit-lens-selected, so a direction only
    those lenses would surface, on a token outside the pool, could not be
    discovered. Security-lexicon coverage is model-specific (`" vulnerable"` on
    deepseek, `" unsafe"` on starcoder2-3b), which weakens the cross-model
    comparison. And the R-lens fidelity caveat on starcoder2-3b (§8.7) applies.
11. **Nothing causal.** A frozen readout surviving a transformation says the
    information is still linearly present, not that the model uses it. E13's
    interchange is the causal instrument, and it covers *binding*, not this.

## 10. Next, in order

Everything this design set out to measure has been measured, on three models,
with no overrides. What is left is not more of E15:

1. **Explain the `assign_chain` fragility** (§8.5), which has now replicated three
   times under renaming *alone*. Starcoder2-3b is the sharpest case: 0.639 in that
   structure against 1.000 on `branch_merge`. Diagnose on the existing
   `sinkflow_predictions.csv` first — which member fails, at which alias step, and
   whether the failing programs share a renamed identifier — before spending GPU.
2. **Fix `configs/models.yaml` ↔ `MODEL_REGISTRY`** so declared `probe_layers` are
   the ones that actually run (§9.8). Repo-wide, affects every experiment, and
   would remove the residual 52%-vs-48% mismatch for starcoder2-3b.
3. **Decide whether the R-lens is usable on starcoder2-3b at all** (§8.7): relevance
   conservation of 0.154 there, against ~1.000 on both deepseek models, is an
   architecture-level finding about the LRP rules and belongs in E14's track, not
   this one.
4. **Naturalistic transfer** is the honest boundary (§9.2), and it is a project,
   not a next step: real programs have no matched pair, so the surface floor would
   stop being pinned and E8's caveat would apply in full.

---

## 11. Experiment A — atomic and cumulative obfuscation

The ten conditions are §4.1; the verification that each one is what it claims to
be is §4.2. What this section fixes is **how the result may be read**.

### 11.1 The five tables stage 124 produces

1. **atomic robustness** — clean, `normalize`, and the four atomic arms, with
   cluster-bootstrap intervals, per-class accuracy, false-negative and
   false-positive rates, predicted-unsafe fraction, matched-pair agreement, and
   `delta_clean`.
2. **cumulative robustness** — the same columns along the ladder, plus
   `delta_previous`, the marginal cost of the step that condition adds.
3. **atomic versus cumulative** — one row per (atomic, cumulative) pair the
   design declares, with `interaction = cumulative − atomic` and the rename row
   labelled as the draw-noise floor.
4. **per-class accuracy and matched-pair collapse** — every condition, because
   pooled accuracy conceals the failure the threat model is about.
5. **the four arms** — hidden state at the reported layer against its three
   floors (§12).

Every one of these is also a row in `sinkflow_obfuscation.csv`, broken down by
family and by structure as well as pooled, so nothing in the report is a number
that cannot be recomputed from the CSV.

### 11.2 What the three differences license

| column | question it answers | claim it supports |
|---|---|---|
| `delta_clean` on an **atomic** row | what does this transformation do alone? | "renaming costs X" / "flattening costs Y" — an **independent transformation effect** |
| `delta_previous` on a **cumulative** row | what did adding this step cost, given the ones before it? | "adding flattening to an already-renamed, opaque, encoded program costs Z" — a **marginal** effect |
| `delta_atomic` | how much of the cumulative failure is not the transformation itself? | an **interaction**: the cost of composition |

**The attribution rule, stated before the numbers:** flattening is named as the
cause of a failure only if `flatten_only` supports that conclusion. If the
readout survives `flatten_only` and fails at `rename_opaque_encode_flatten`, the
finding is a **cumulative** effect and must be written that way — the previous
draft of §9.5 is exactly the limitation this arm exists to close.

### 11.3 What it cannot say

The design deliberately does **not** generate all pairwise or higher-order
combinations: 4 atomic + 4 cumulative is 8 arms, the full lattice is 15, and the
extra 7 would triple the extraction cost to answer a question nobody asked. So
an interaction between, say, `opaque` and `flatten` *without* renaming is not
measurable here, and no sentence in the report may imply it is.

## 12. Experiment B — the whole-program lexical baseline

Limitation §9.1 has always said the same thing: the floor is pinned only against
the **declared** surface family, a ±3-token window at the anchor, and "a
predictor with the whole program text could recover the label by performing the
taint analysis itself." That sentence conflates two very different predictors,
and the difference matters:

* a predictor that *runs the taint analysis* — still out of scope, still the
  honest boundary, and no baseline in this repository bounds it;
* a predictor that reads the **text** for n-grams the generator happened to
  correlate with the label. That one is cheap to build and, until now, unmeasured.

So: token unigrams and bigrams (plus character 3–5-grams) over the complete
program, TF-IDF, a linear classifier, **fitted only on clean training programs**
and then frozen and transferred to every held-out condition exactly like the
hidden-state probes. CPU-only. Grouped by base, and the vectorizer is refitted
inside every CV fold, because a vocabulary fitted on all the training text and
then cross-validated would leak the held-out fold's terms and idf weights into
the features — precisely the leak this arm exists to detect elsewhere.

**Deliberately not given** AST, graph, taint-analysis or any other
program-analysis features. It is a bound on the textual shortcut, not a competing
program analysis, and adding those would turn it into an unrelated experiment.

What it can conclude:

* **near chance** → the whole program text does not carry the label under this
  feature family, and the hidden-state result is not a generator artifact that a
  wider window would have caught. It still does **not** establish that no
  whole-program predictor could succeed.
* **high** → the benchmark is lexically solvable and every hidden-state number
  needs that caveat attached, in the report and in the paper.

Reported as its own arm (`whole_program_lexical`) beside `local_surface`,
`embedding` and `hidden_state`, in every condition, so that "what renaming does
to a whole-program lexical shortcut" is a measurement rather than an argument.

## 13. Experiment C — the vocabulary-space contrast (observational)

> After mapping the sink-site state into the model's **own** vocabulary space,
> which vocabulary directions distinguish an unsafe program from its matched
> safe counterfactual?

The probe experiments say what a *fitted* direction can recover. They cannot say
whether the model's output-aligned coordinates carry the distinction, because a
probe chooses its own basis and will find any linearly available direction
whether or not the model ever uses it. E15-C asks about **format**, not use.

**It is observational, and it stays observational.** No J-space coordinate
intervention, no interchange, no swap. E13 is the causal result and is strictly
stronger for that purpose. A vocabulary direction that separates the two members
is *not* evidence that the model uses it.

### 13.1 Three readouts, one set of states

| readout | vector for candidate `w` |
|---|---|
| logit lens | `g ⊙ W_U[w]` — no layer-to-layer correction |
| J-lens | `E[J_l]ᵀ (g ⊙ W_U[w])` — the averaged first-order causal effect (E10/E11) |
| R-lens | the same estimator under the LRP rules of `src/models/lrp.py` (E14) |

**R-lens is the primary readout, declared in code (`PRIMARY_LENS`) and here,
before any result was produced**, because the target includes early and middle
layers — exactly where E14's gate R showed raw autograd is least faithful (it
inverts sign at depth; the LRP backward holds 0.945–1.005). The other two are
comparisons. Choosing the primary after seeing which produced the strongest
security-token result would make every number a selection artifact.

### 13.2 Why the candidate vocabulary is restricted

A J-lens or R-lens vector is one vector-Jacobian product **per candidate token**,
so a full 32k-row lens at every layer is not expensive, it is infeasible. The
logit lens has no such constraint. Discovery is therefore two-phase:

1. **full vocabulary, logit lens, clean TRAINING pairs only** — every vocabulary
   token ranked by mean paired delta; the top ±`n_pool` become the candidate pool;
2. **within the pool** — J-lens and R-lens vectors are built, and each lens ranks
   the pool by *its own* mean paired training delta.

**Limitation, recorded in the discovery provenance itself:** the J/R candidate
pool is logit-lens-selected, so a direction only they would have surfaced, on a
token outside the pool, cannot be discovered here.

The frozen security lexicon and a random control set are added unconditionally,
so neither depends on discovery.

### 13.3 The security lexicon, and the tokenizer check

Fixed before any held-out number: unsafe-oriented `unsafe`, `untrusted`,
`tainted`, `vulnerable`; safe-oriented `safe`, `trusted`, `clean`. Small on
purpose — a long hand-written list turns "does a security word carry the
contrast" into a multiple-comparisons exercise.

A word is included **only if it is one stable vocabulary token under this
model's tokenizer**, checked two ways: it encodes to exactly one token in one of
the prompt-space variants (`" word"`, `"word"`, `" word\n"`), and that token
decodes back to the variant that produced it. Every omitted word is recorded with
its reason, and **nothing is substituted** — the first token of a split word is a
prefix, not the word, and a table that pretended otherwise would be measuring the
prefix. This is per model and the omissions differ: on `deepseek-coder-1.3b`
only `" vulnerable"` survives on the unsafe pole (`unsafe`, `untrusted`,
`tainted` all split), while on `starcoder2-3b` `" unsafe"` survives and
`vulnerable` does not.

### 13.4 Orientation, and the three score conventions

For every matched pair, at every (lens, layer, site, condition):

```
delta(pair, token) = score_unsafe(token) - score_safe(token)
```

fixed in one place and recorded on every row, because a per-cell orientation
choice would make every sign statistic meaningless.

`JLens.scores` drops a positive per-position factor (see `src/models/lens.py`),
and a paired contrast compares two *different* positions. So every statistic is
carried in three conventions:

| convention | meaning | exactness |
|---|---|---|
| `score` | the raw lens score | exact for the logit lens; scale-carrying for J/R |
| `z` | z-scored across the candidate set at that position | **exactly** invariant to the dropped factor, so two positions are comparable |
| `prob` | softmax over the candidate set — the "probability mass" | exact for the logit lens; inherits the dropped factor for J/R |

The design names probability mass as the primary measurement, so
`mean_delta_contrast_prob` is reported as such — and `..._z` beside it
everywhere, because for the J/R lenses that is the one whose sign is exact. Sign
consistency is reported in both.

### 13.5 Controls

| control | what it kills |
|---|---|
| **permutation** — re-orient each base at random | keeps every pair and magnitude, destroys only the safe→unsafe alignment: is it the *direction* that carries the effect? |
| **mismatched pairs** — unsafe and safe from different bases, matched on family and structure | keeps the orientation, destroys the pairing: is it the safe/unsafe difference or any difference between two programs of this kind? |
| **embedding layer (−1)** | at `sink_arg` the state *is* the anchor token's embedding, so this is exactly the token-identity contrast |
| **`last_token` beside `sink_arg`** | at the last token both members carry the same token id (recorded per row as `anchor_token_same`), so an effect that is only the differing sink-argument token cannot appear there |
| **identifier-role strata** (`role_swap`) | the generator alternates which chain name is tainted; a token-identity account predicts the contrast flips with it |
| **random and Gram-matched lenses** | matched norms, and matched norms *and angles*: the only thing left that differs is which residual-stream directions the lens points at |
| **renamed and every other condition** | survival of the contrast under the same transformations the probe was tested on |

### 13.6 What licenses a semantic reading

All of these, or none of it:

1. tokens discovered on **training pairs only**, frozen to disk before held-out
   scoring (the freeze is a filesystem boundary: stage 126 reads a file it did
   not write);
2. replication on held-out pairs;
3. one consistent safe→unsafe orientation;
4. an effect above **both** the permutation and the mismatched-pair controls;
5. stability across the generator's identifier-role assignment;
6. not reducible to the differing sink-argument token.

If they do not all hold, the output is the descriptive table of top vocabulary
directions and the conclusion that **no stable vocabulary-aligned security
concept was found**. Stage 127 decides this by checklist and prints the checklist.

**"The token `unsafe` appeared in a top-k list" is not a result** and is not
allowed to become one.

### 13.7 The interpretations this design can support

| outcome | reading |
|---|---|
| stable held-out mass on the concept tokens, above controls | **explicit security vocabulary** |
| stable non-security tokens carry the contrast instead | **output-aligned flow information without explicit verbalisation** |
| probe succeeds, all three lenses null | **linearly decodable without demonstrated vocabulary alignment** — the two claims come apart, which is why this experiment exists |
| R-lens separates where the J-lens does not | the **J-lens intermediate-layer limitation** E14 measured, showing up on a task |
| every readout collapses under structural obfuscation | loss of **both** trained and output-aligned auditability |

### 13.8 Lens fidelity is a diagnostic

Every layer is measured with every lens. Next-token recovery, agreement with the
final-layer distribution, R-lens relevance conservation and the random /
Gram-matched floors are recorded per (layer, lens) with warning thresholds, and
**none of them is consulted by a gate** (there is a test that asserts this). A
weak layer earns a warning and stays in the experiment: refusing there would
restrict the study to the layers where the instrument is comfortable, and the
early and middle layers are the target.

---

## 14. The complete pipeline, in order

Every command, labelled CPU or **GPU**, with the gate that must pass before the
next one may run. A stage refuses to start when a prerequisite gate has not
passed; `--override-gate REASON` is permitted and is recorded permanently.

### 14.1 CPU tests and dataset validation (no model, no GPU)

```bash
# the CPU test suites for this track
python -m pytest tests/test_sink_flow.py tests/test_sinkflow_vocab.py \
                 tests/test_obfuscation.py tests/test_lens.py tests/test_lrp.py -q
python -m pytest tests/ -q                       # the whole suite

# S0 — the benchmark, all ten conditions, tokenizer-verified anchors
python scripts/120_sinkflow_generate.py --model deepseek-coder-1.3b
```

**Gate: S0 must pass** before stage 121. It also has to be re-run per model,
because the anchors are verified against that model's tokenizer.

### 14.2 Smoke (a few minutes on a laptop, MPS or CPU)

```bash
make sinkflow-smoke MODEL=deepseek-coder-1.3b        # A + B: 96 programs, 3 layers
make sinkflow-vocab-smoke MODEL=deepseek-coder-1.3b  # C: 1 layer, 15 candidates
```

The smoke writes to `data/smoke/` and `results/smoke/` and touches nothing in the
canonical trees. `sinkflow-smoke` is a few minutes; `sinkflow-vocab-smoke` is
~15 minutes on an M-series laptop, almost all of it in the two lens builds (see
§14.6's MPS caveat).

### 14.3 Per model — DeepSeek-Coder 1.3B

```bash
python scripts/120_sinkflow_generate.py       --model deepseek-coder-1.3b   # CPU  → S0
python scripts/121_sinkflow_extract.py        --model deepseek-coder-1.3b   # GPU  → S1
python scripts/122_sinkflow_probe.py          --model deepseek-coder-1.3b   # CPU  → S2
python scripts/123_sinkflow_obfuscation.py    --model deepseek-coder-1.3b   # CPU  → S3
python scripts/124_sinkflow_report.py         --model deepseek-coder-1.3b   # CPU
python scripts/125_sinkflow_vocab_discover.py --model deepseek-coder-1.3b   # GPU  → J0
python scripts/126_sinkflow_vocab_contrast.py --model deepseek-coder-1.3b   # CPU  → J1
python scripts/127_sinkflow_vocab_report.py   --model deepseek-coder-1.3b   # CPU
```

### 14.4 Per model — DeepSeek-Coder 6.7B and StarCoder2-3B

Identical, with `--model deepseek-coder-6.7b` / `--model starcoder2-3b`. On the
GPU host the two GPU stages run under `screen`:

```bash
screen -dmS sinkflow-extract-6.7b env MODEL=deepseek-coder-6.7b jobs/sinkflow_extract.csh
screen -dmS sinkflow-vocab-6.7b   env MODEL=deepseek-coder-6.7b jobs/sinkflow_vocab.csh
```

### 14.5 Cross-model reading

Always at **matched relative depth**, never at a common layer index (§9.7):

```bash
for M in deepseek-coder-1.3b deepseek-coder-6.7b starcoder2-3b; do
    python scripts/124_sinkflow_report.py       --model $M --depth 0.48
    python scripts/127_sinkflow_vocab_report.py --model $M --depth 0.48
done
```

### 14.6 What each stage costs and produces

| stage | where | gate | output |
|---|---|---|---|
| 120 | CPU, ~2 min | **S0** | `data/synthetic/sinkflow_{model}_{train,heldout,heldout_obf}.jsonl` (336 / 144 / **1296**), `benchmark.csv`, `gates.yaml` |
| 121 | **GPU**, ~25 min (1.3b) | **S1** | `results/activations/{model}/sinkflow_{train,heldout,heldout_obf}/` — 1776 programs |
| 122 | CPU, minutes | **S2** | `sinkflow_clean.csv`, `probes/{site}/{layer_XX,surface,whole_program_lexical}.pkl`, `probes/provenance.json` |
| 123 | CPU, minutes | **S3** | `sinkflow_obfuscation.csv`, `sinkflow_predictions.csv` |
| 124 | CPU, seconds | — | `e15_report.{md,yaml}`, `results/figures/sinkflow_*.png` |
| 125 | **GPU**, hours | **J0** | `vocab/vocab_discovery.json`, `vocab/vocab_train_deltas.csv`, `vocab/vocab_lens_diagnostics.csv`, `vocab/lenses/*.pkl` |
| 126 | CPU, minutes | **J1** | `vocab/vocab_{pairs,pair_tokens,tokens,summary,controls,condition_similarity,lens_agreement}.csv` |
| 127 | CPU, seconds | — | `vocab/e15c_report.{md,yaml}` |

Stage 125 is the expensive one and its cost is `n_candidates × n_build × n_tprime`
backward passes per (layer, lens): the candidate cap (`--max-candidates`), the
number of build triples (`--n-build`), `--n-tprime` and the layer list are the
four knobs.

**A measured MPS caveat.** On Apple silicon the fp16 backward through this path
returns non-finite gradients at *every* scale in `compute_lens_vectors`' retry
ladder, so `--dtype float32` is required there — and fp32 backward on MPS is slow
enough that a single (candidate, t') pair costs seconds. Measured on
`deepseek-coder-1.3b` at layer 11: ~60 s for 8 candidates × 1 t' × 1 sample. That
is why `make sinkflow-vocab-smoke` uses one layer, 8 candidates, 2 samples and one
readout position, and why the canonical runs belong on the CUDA host with the
stage's defaults. The two deepseek layers nearest the embedding are the most
expensive of all: the VJP at layer −1 traverses every decoder block.

### 14.7 Expected row counts

With the canonical benchmark and a model probed at `L` layers (including −1):

| file | rows |
|---|---|
| `benchmark.csv` | `480 + 1296 = 1776` |
| `sinkflow_predictions.csv` | `2 sites × (L + 2 arms) × 1440 held-out programs` |
| `sinkflow_obfuscation.csv` | `2 × (L + 2) × 10 conditions × 8 breakdowns` — e.g. 1920 rows at L = 10 |
| `vocab/vocab_pairs.csv` | `3 lenses × L × 2 sites × 10 conditions × 72 bases` |
| `vocab/vocab_tokens.csv` | `3 × L × 2 × 10 × n_candidates` |
| `vocab/vocab_pair_tokens.csv` | `3 × L × 2 × 10 × 72 × n_raw_tokens` (`--raw-tokens`, concept tokens by default) |
| `vocab/vocab_summary.csv` | `3 × L × 2 × 10` |

S3 and J1 check these against the design and refuse on a mismatch, so a silently
missing condition cannot reach a report.

### 14.8 What warns without blocking

* lens fidelity per (layer, lens): next-token recovery, agreement with the
  final-layer distribution, R-lens relevance conservation, and the random /
  Gram-matched floors (§13.8);
* the number of variants that needed a redraw to carry exactly their declared
  transformation (stage 120 prints it);
* record problems while assembling pairs (stage 126 prints the count and the
  first few) — as long as the cells the gate requires are all present.

### 14.9 Not run locally

The full-scale runs of the atomic arms, the whole-program lexical baseline at
canonical size, and E15-C on any model are **GPU-host work and have not been run
here**. What has run locally is the smoke pipeline for all three experiments
(`make sinkflow-smoke`, `make sinkflow-vocab-smoke`) and the CPU test suites. The
three-model numbers in §8 are the **cumulative-ladder** runs that predate the
atomic arms and are unchanged by this work — but they were produced from a
720-variant shard, so re-running stage 120 regenerates a 1296-variant shard and
stage 121 must be re-run before 122/123 are meaningful for the new conditions.
