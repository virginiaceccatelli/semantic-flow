# E15 — Do code models see through obfuscation?

**Auditing a security-relevant semantic representation: does the value that
reaches a code-bearing, sensitive argument come from untrusted input, and does a
*frozen* readout of that fact survive the obfuscation ladder?**

Status: **run at canonical scale on three models** — `deepseek-coder-1.3b`,
`deepseek-coder-6.7b` and `starcoder2-3b` — all four gates passing in every run
with no overrides. The results are in §8. `results/STATUS.yaml` is the registry
of record, and the limitations in §9 are not optional reading — the floor here is
weaker than E2's by construction.

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

# cross-model reading: always at MATCHED RELATIVE DEPTH, never at layer index
python scripts/124_sinkflow_report.py --model deepseek-coder-1.3b --depth 0.48
python scripts/124_sinkflow_report.py --model deepseek-coder-6.7b --depth 0.48
python scripts/124_sinkflow_report.py --model starcoder2-3b       --depth 0.48
```

Outputs land in `results/sinkflow/{model}/`: `benchmark.csv`, `gates.yaml`,
`sinkflow_clean.csv`, `sinkflow_obfuscation.csv`, `sinkflow_predictions.csv`,
`e15_report.{md,yaml}`, `probes/{site}/{layer_XX,surface}.pkl` and
`probes/provenance.json`; figures in `results/figures/sinkflow_*.png`.

## 8. Results — canonical runs, 1.3B, 6.7B and starcoder2-3b

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

### 8.2 Frozen transfer: renaming is cheap, flattening is the boundary

`sink_arg` at matched depth, 144 programs / 72 bases per row:

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
5. **Level 4 is cumulative, so "flattening breaks it" is a marginal claim.**
   Level 4 contains renaming, opaque predicates and MBA encoding as well as the
   dispatch loop. What the data supports is that levels 1–3 together cost
   ≤ 0.10 and adding flattening costs a further ~0.30; a flatten-only arm would
   turn that into a clean attribution, and the ladder can express one without
   any new transformation.
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

Done since the first draft: **6.7B at matched relative depth** (§9.7, closed) and
**`starcoder2-3b`** (three-model replication, §8). What remains:

1. **A flatten-only arm** (limitation §9.5). The one thing the current data
   cannot separate: level 4 bundles four transformations, and the interesting
   claim — "the readout is anchored on control structure, and dissolving it is
   what breaks the audit" — needs the dispatch loop applied *without* renaming,
   opaque predicates or MBA. No new transformation; the ladder already has the
   pieces, only the composition is cumulative. This is the highest-value next
   run: CPU plus one GPU extraction per model, and it has been verified to
   produce all 144 held-out programs.
2. **Explain the `assign_chain` fragility** (§8.5), now that it has replicated
   three times. Starcoder2-3b is the sharpest case: 0.583 under renaming alone,
   0.222 of it on the unsafe member. Diagnose on the existing
   `sinkflow_predictions.csv` first — which member fails, at which alias step,
   and whether the failing programs share a renamed identifier — before spending
   any GPU.
3. **Per-class reporting wherever this readout is quoted** (§8.4). The pooled
   number hid a 0.75-on-unsafe failure in one model and a symmetric 0.93/0.93 in
   another. `acc_unsafe`/`acc_safe` are already columns; the report table should
   carry them, not just the CSV.
4. **Fix `configs/models.yaml` ↔ `MODEL_REGISTRY`** so declared `probe_layers`
   are the ones that actually run (§9.7). Repo-wide, affects every experiment,
   and would remove the residual 52%-vs-48% mismatch for starcoder2-3b.
5. **Naturalistic transfer** is the honest boundary (§9.2), and it is a project,
   not a next step: real programs have no matched pair, so the surface floor
   would stop being pinned and E8's caveat would apply in full.

### 10.1 Companion runs still outstanding

E15 says flattening breaks a *security* readout. E9 says what the same ladder
does to *binding* on the same model. Without the companion, a security-specific
reading cannot be separated from a general one — E9 exists at 1.3B and 6.7B, and
the starcoder2-3b arm is running. Until it lands, the three-model E15 table
should be read as "the same ladder, the same boundary, three models", not as
"security representations are specifically fragile".
