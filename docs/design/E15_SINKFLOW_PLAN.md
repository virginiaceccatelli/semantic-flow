# E15 — Do code models see through obfuscation?

**Auditing a security-relevant semantic representation: does the value that
reaches a code-bearing, sensitive argument come from untrusted input, does a
*frozen* readout of that fact survive obfuscation, and is the difference
expressed in the model's own vocabulary?**

Three experiments over one benchmark, one set of matched pairs and one set of
activations:

| | question | stages | status |
|---|---|---|---|
| **A** atomic + cumulative robustness | which transformation breaks the frozen readout *on its own*, and what does composing them add? | 120–124 | ladder run on three models (§8); the atomic arms are implemented and smoke-tested, the full runs are the pending GPU work (§11) |
| **B** whole-program lexical baseline | could a reader of the entire program text recover the label with no hidden state? | 122–124 | implemented, CPU-only, runs inside the same stages (§12) |
| **C** vocabulary-space contrast | after mapping the sink state into the model's own output basis, which vocabulary directions separate unsafe from its matched safe counterfactual? | 125–127 | implemented and smoke-tested; observational only (§13) |

Status: A's **cumulative ladder** has run at canonical scale on three models —
`deepseek-coder-1.3b`, `deepseek-coder-6.7b` and `starcoder2-3b` — with S0–S3
passing in every run and no overrides (§8). The **atomic arms**, the
**whole-program lexical baseline** and the **vocabulary-space contrast** are
implemented, gated and smoke-tested; the full-scale runs are the commands in
§14. `results/STATUS.yaml` is the registry of record, and the limitations in §9
are not optional reading — the floor here is weaker than E2's by construction.

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
and nothing in the data could say which one did the damage (§9.5). The atomic
arms fix exactly that. **No new obfuscation algorithm is introduced, and no
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


All three models: 480 clean programs, 336 training / **144 held-out (72 bases)
per condition**, all four gates passing with **no overrides recorded**. 1.3B
probes 8 layers, 6.7B and starcoder2-3b probe 10 (see the caveat in §9.7).
Intervals are cluster-bootstrapped over base programs.

Headline site is `sink_arg`, read at **the layer nearest 48% of network depth**:
1.3B layer 11 (48%), 6.7B layer 15 (48%), starcoder2-3b layer 15 (52% — its grid
has nothing closer; layer 11 is 38%). In all three that layer is also the argmax
of clean-training CV, so the depth match and the best-layer choice agree.

### 8.1 The property is decodable, and it is not the identifier

Clean training programs, grouped CV at `sink_arg`. Layer indices are not
comparable across 24/32/30-layer models, so each column is labelled with its
depth:

| | 1.3B (24L) | 6.7B (32L) | starcoder2-3b (30L) |
|---|---:|---:|---:|
| **surface baseline** (±3 token ids, no hidden states) | **0.491** | **0.491** | **0.488** |
| −1 embedding | 0.482 | 0.482 | 0.482 |
| layer 0 | 0.473 | 0.461 | 0.462 |
| ~10% depth | 0.777 (L3) | 0.758 (L3) | **0.896** (L3) |
| ~25% depth | 0.991 (L7) | 0.979 (L7) | 0.952 (L7) |
| **~48% depth** | **1.000** (L11) | **1.000** (L15) | **0.997** (L15) |
| last layer | 0.988 (L23) | 1.000 (L31) | 0.988 (L29) |

Chance at the input, built by the first quarter of the network, at ceiling near
half depth and held to the output — E2's binding profile reproduced on a security
label, in **three** models across two architecture families and two pretraining
corpora. AUC is ≥0.99 from 25% depth on in all three.

The one difference worth naming: **starcoder2-3b has most of the property by 10%
depth** (0.896 at layer 3, against 0.76–0.78 for both deepseek models). Whether
that is architecture, corpus or tokenisation is not something this design can
separate, and nothing downstream rests on it.

**A note on the embedding row.** The −1 predictions are *byte-identical* across
all three models — every program, every condition. That looks like a plumbing
bug and is not one. At layer −1 the feature is the raw embedding of the anchor
token, so the probe degenerates into a per-token lookup, and the anchor tokens
induce **the same 22-way partition of the 1200 programs under both tokenizers**
(verified by recomputing anchor token ids under each: deepseek splits `chunk`
into `ch`+`unk` where starcoder2 keeps it whole, but the *grouping* of programs
is identical, because the benchmark draws its identifiers from one fixed pool).
Identical groups and identical training labels give identical held-out
predictions regardless of embedding geometry. The control is doing exactly what
it claims — token identity before any computation carries nothing (0.472–0.521
in every condition) — and its model-independence is a property of the benchmark,
not evidence that the stores were confused.

### 8.2 Frozen transfer: renaming is cheap, the ladder's last rung is the boundary

`sink_arg` at matched depth, 144 programs / 72 bases per row. **These are the
cumulative-ladder runs**, so level 4 contains all four transformations and the
row below reads as "the composition", not "flattening" — that attribution needs
the `flatten_only` arm of §4.1, which has not run at this scale yet:

| condition | 1.3B (L11) | 6.7B (L15) | starcoder2-3b (L15) | surface |
|---|---:|---:|---:|---:|
| clean held-out | **1.000** [1.000, 1.000] | **1.000** [1.000, 1.000] | **1.000** [1.000, 1.000] | 0.444–0.451 |
| 0 normalize | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.444–0.458 |
| 1 rename | 0.931 [0.889, 0.972] | **0.965** [0.938, 0.993] | 0.868 [0.812, 0.917] | 0.479–0.493 |
| 2 opaque | 0.951 [0.917, 0.986] | 0.979 [0.951, 1.000] | 0.938 [0.903, 0.972] | 0.458–0.500 |
| 3 encode | 0.938 [0.889, 0.972] | 0.958 [0.924, 0.986] | 0.924 [0.882, 0.959] | 0.479–0.521 |
| **4 flatten** | **0.632** [0.556, 0.708] | **0.562** [0.500, 0.625] | **0.569** [0.479, 0.653] | 0.500–0.507 |

The shape is the same in all three: **ceiling on clean text, a small loss to
levels 1–3, and a collapse at level 4.** The surface arm never leaves chance in
any condition or any model, so none of this is the identifier — including at
level 1, where renaming destroys the identifiers outright and the hidden-state
readout loses 0.03–0.13.

Ordering across models is *not* stable and should not be read as a scale effect:
6.7B is best at levels 1–3 and worst at level 4. Starcoder2-3b is the weakest at
renaming (0.868) but the difference from 1.3B is a 0.06 gap between overlapping
intervals, and it is the same 0.868 at its layer 11 (38% depth) and 0.910 at
layer 19 (66%), so the reading is not an artifact of the 52%-vs-48% mismatch.

### 8.3 What survives flattening is class bias, and the third model proves it

This is the finding that accuracy alone would have got wrong, and the reason
`pairs_same_label` and `frac_predicted_unsafe` are columns rather than a
footnote. At level 4, matched depth:

| | 1.3B (L11) | 6.7B (L15) | starcoder2-3b (L15) |
|---|---:|---:|---:|
| accuracy | 0.632 | 0.562 | 0.569 |
| accuracy on unsafe / safe | 0.583 / 0.681 | **0.792 / 0.333** | **0.417 / 0.722** |
| predicted unsafe | 0.451 | **0.729** | **0.347** |
| matched pairs given the same label | 0.514 | 0.653 | 0.444 |

Three numbers within 0.07 of each other, produced three different ways. **1.3B**
half-loses the distinction — 51% of pairs get one label — with no class
preference. **6.7B** collapses toward "unsafe", calling 73% of programs
vulnerable and getting two thirds of the safe ones wrong. **starcoder2-3b
collapses the other way**, toward "safe": it calls only 35% of programs
vulnerable and misses 58% of the genuinely unsafe ones.

The third model is what turns this from an observation into an argument. A
constant predictor of either class scores exactly 0.500 on this balanced set, so
if the residual 0.56–0.63 were retained flow information the three models would
have to be retaining it *and* biasing in opposite directions. The parsimonious
reading is that **level 4 destroys the readout in all three, and what is left is
each model's own prior**. Quoting "≈60% retained" from the accuracy column alone
would have been wrong three times over.

The `last_token` site makes the same point at its limit. Outside a narrow band
around half depth, the flattened `last_token` readout is **exactly 0.500 with a
zero-width bootstrap interval and `pairs_same_label` 1.000** in all three models
— and at most of those layers the answer is literally constant: "safe" at 1.3B's
layers 7/19/23 and 6.7B's layers 23–31 (positive rate 0.000), "unsafe" at
starcoder2's layers 11 and 15 (positive rate 1.000). Dead readouts pointing in
opposite directions. (A zero-width cluster bootstrap over 72 bases is the same
signature that diagnosed E12's behavioural failure.)

Starcoder2 also loses `last_token` far earlier than either deepseek model: it is
already at 0.632 under mere *normalisation* (level 0), where 1.3B holds 0.986 and
6.7B 0.965. The `sink_arg` site replicates across all three models; the
`last_token` site does not, which is the empirical justification for reporting
them separately rather than pooling.

### 8.4 The errors are not symmetric, and they run the auditor's dangerous way

Renaming costs starcoder2-3b 0.13 pooled, which sounds survivable. The
per-class split says otherwise:

| level 1 (rename), matched depth | accuracy | on unsafe | on safe | predicted unsafe |
|---|---:|---:|---:|---:|
| 1.3B (L11) | 0.931 | 0.931 | 0.931 | 0.500 |
| 6.7B (L15) | 0.965 | 0.931 | 1.000 | 0.465 |
| **starcoder2-3b (L15)** | 0.868 | **0.750** | 0.986 | 0.382 |

Starcoder2's entire renaming loss is **false negatives**: it keeps calling safe
programs safe and starts calling vulnerable programs safe too. In the
`assign_chain` structure this is severe — accuracy 0.583, of which **0.222 on
unsafe against 0.944 on safe**: after nothing but consistent identifier renaming,
the frozen readout misses **78% of the vulnerable assignment-chain programs**
while remaining almost perfect on the safe ones.

For an audit readout that is the failure direction that matters, and pooled
accuracy hides it completely — 0.868 reads like "mostly fine". Any use of a
readout like this one has to report per-class rates, and any comparison across
models has to as well: 1.3B loses the same 0.07 symmetrically, which is a
different failure with the same headline number.

### 8.5 Where the degradation lives: structure, not sink family

Per structure at matched depth (36 programs per cell), rename / flatten:

| structure | 1.3B (L11) | 6.7B (L15) | starcoder2-3b (L15) |
|---|---:|---:|---:|
| `direct` | 1.000 / 0.611 | 1.000 / 0.611 | 1.000 / 0.806 |
| `branch_merge` | 1.000 / 0.806 | 1.000 / 0.750 | 0.972 / 0.694 |
| `helper` | 0.917 / 0.556 | 1.000 / 0.528 | 0.917 / 0.444 |
| `assign_chain` | **0.806** / 0.556 | **0.861** / 0.361 | **0.583** / 0.333 |

The ordering now reproduces in **three** models: `direct` and `branch_merge` are
untouched or nearly untouched by renaming, and the **assignment chain is the
fragile structure** in every model under both renaming and flattening, with the
helper boundary next. A merge point being *at least as* robust as a two-step
alias chain is the opposite of what "longer chain = harder" would predict, and
three independent replications make it the most interesting open question in the
track. (At 6.7B's layer 11 — 35% depth — `assign_chain` reads 0.639, and an
earlier draft quoted that against 1.3B's 48%-depth number. The ordering it
implied was an artifact of the layer mismatch, which is why §9.7 is now closed
rather than outstanding.)

By sink family the picture is flat in all three models (0.83–0.98 at levels 1–3,
0.52–0.73 at level 4, no ordering that reproduces), which is the null the design
wanted: the readout tracks flow, not which dangerous API is at the end of it.

## 9. Limitations, stated before any number is read

1. **The floor is not pinned to chance by construction the way E2's is.** It is
   pinned only against *declared* feature families. Two are now measured — the
   ±3-token local window and the whole-program lexical reader (§12) — but a
   predictor that ran the taint analysis itself would still score 1.0, and
   nothing here bounds that. E15 is therefore an audit of a readout's transfer,
   not a representation claim of E2's kind, and both baselines are reported
   beside every number rather than in a footnote.
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
5. **The three-model numbers in §8 are cumulative-only, so "flattening breaks
   it" is still a marginal claim *for those runs*.** Level 4 there contains
   renaming, opaque predicates and MBA encoding as well as the dispatch loop.
   What §8 supports is that levels 1–3 together cost ≤ 0.13 and adding
   flattening costs a further ~0.30. The atomic arms (§4.1, §11) are what turn
   that into an attribution, and until they have run at canonical scale on a
   model the correct wording is "a cumulative effect", not "flattening".
6. **The selectivity control is weak here by construction.** A base has exactly
   two rows with opposite labels, so shuffling within it can only *swap* them,
   never destroy the signal — which is why the control sits at 0.52–0.61 rather
   than 0.500 and selectivity (0.43–0.48) understates the effect. The
   load-bearing floors in E15 are the measured surface baseline and the embedding
   layer, both at chance in all three models; do not quote selectivity as the
   margin.
7. **The models were probed on different layer grids** — 8 layers for 1.3B, 10
   for 6.7B and starcoder2-3b — because `ModelConfig` computes its own default
   from `MODEL_REGISTRY` and the `probe_layers` in `configs/models.yaml` are not
   read by anything. *That* is a genuine repo defect and it is still open (it
   affects every experiment, not just E15). Its consequence for E15 is
   **closed**: `relative_depth` is a column on every result row,
   `124_sinkflow_report.py --depth 0.48` reports at matched depth, and §8 is
   written from those numbers. It mattered: at layer index 11 the two deepseek
   models' rename robustness reads 0.931 vs 0.910, and at matched depth it reads
   0.931 vs 0.965 — the ordering reverses. The residual mismatch is
   starcoder2-3b, whose grid has no layer between 38% and 52%; §8.2 checks the
   neighbours and the reading does not depend on the choice.
8. **The embedding control is model-independent, and that is expected here.**
   Its predictions are identical across all three models because at layer −1 the
   probe reduces to a lookup on the anchor token and the benchmark's fixed
   identifier pool induces the same partition under both tokenizers (§8.1). It
   is a real control — it says token identity carries nothing — but it is *not*
   three independent measurements, and it should not be cited as if the three
   models had each been shown to lack the information at their input.
9. **Level 4 changes what "the source anchor" means.** After flattening, the
   first source expression in source order is whichever dispatch case the
   shuffle put first. The `sink_arg` anchor is unaffected, which is why it, not
   the source anchor, is the headline site.
10. **Nothing causal.** A frozen readout surviving a transformation says the
   information is still linearly present, not that the model uses it. And a
   readout *failing* says the information is not linearly present at that
   position for this probe — not that the model has lost it.

## 10. Next, in order

Done since the first draft: **6.7B at matched relative depth** (§9.7, closed),
**`starcoder2-3b`** (three-model replication, §8), and — implemented, gated and
smoke-tested, awaiting their full runs — the **atomic arms** (§4.1, §11), the
**whole-program lexical baseline** (§12) and the **vocabulary-space contrast**
(§13). What remains:

1. **Run the atomic arms at canonical scale** (limitation §9.5). Everything is
   built: the four `*_only` conditions, S0's isolation check, and the three
   difference columns. It costs one CPU regeneration plus one GPU re-extraction
   per model — the shard grows from 720 to 1296 variants — and it is what turns
   "level 4 breaks the readout" into an attribution. §14 has the commands.
2. **Run E15-C at canonical scale** (§13). One GPU stage per model (stage 125,
   the lens build) and then CPU. Its outcome is genuinely open, and a null is a
   reportable result: the report's verdict machinery is built to say so rather
   than to hunt for a security word in a top-k list.
3. **Explain the `assign_chain` fragility** (§8.5), now that it has replicated
   three times. Starcoder2-3b is the sharpest case: 0.583 under renaming alone,
   0.222 of it on the unsafe member. Diagnose on the existing
   `sinkflow_predictions.csv` first — which member fails, at which alias step,
   and whether the failing programs share a renamed identifier — before spending
   any GPU.
4. **Per-class reporting wherever this readout is quoted** (§8.4) — *done*: the
   report's table 4 carries `acc_unsafe`, `acc_safe`, the false-negative and
   false-positive rates and `pairs_same_label` for every condition, so the
   pooled number can no longer be quoted alone from the report itself.
5. **Fix `configs/models.yaml` ↔ `MODEL_REGISTRY`** so declared `probe_layers`
   are the ones that actually run (§9.7). Repo-wide, affects every experiment,
   and would remove the residual 52%-vs-48% mismatch for starcoder2-3b.
6. **Naturalistic transfer** is the honest boundary (§9.2), and it is a project,
   not a next step: real programs have no matched pair, so the surface floor
   would stop being pinned and E8's caveat would apply in full.

### 10.1 Companion runs still outstanding

E15 says flattening breaks a *security* readout. E9 says what the same ladder
does to *binding* on the same model. Without the companion, a security-specific
reading cannot be separated from a general one — E9 exists at 1.3B and 6.7B, and
the starcoder2-3b arm is running. Until it lands, the three-model E15 table
should be read as "the same ladder, the same boundary, three models", not as
"security representations are specifically fragile".

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
