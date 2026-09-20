# CruxEval J/R-lens and value-DAS cluster runbook

Stages 235–238 run first. They reuse the published E19 J/R pair fitted on the
independent Pile corpus by stage 201. Stages 239–241 then construct and test
real-code value interchange. Stages 242–243 are an exploratory inspection pass
over the same stage-236 readout: which early/middle layers surface the output
tokens, and what the lenses literally put at the top of the vocabulary there.
Stages 244–246 restrict the J-lens to coherent words, select a held-out
(read, layer) by how program-specific those words are, and replicate the
readout on execution-verified obfuscated variants. None of these stages consumes the probe activation
stores; they reuse `prepared_all` for source, graph, tokenizer, and provenance.

## 0. Tests and paths

Run from a bash shell in the repository:

```bash
PY=.venv/bin/python
PROBES=results/cruxeval/deepseek-coder-6.7b/probes_n500
MECH=results/cruxeval/deepseek-coder-6.7b/mechanistic_n500
LENS=results/workspace_lens/deepseek-coder-6.7b
CORPUS=data/lens_corpus/pile10k-n100.jsonl

$PY -m pytest tests/test_cruxeval.py tests/test_cruxeval_pipeline.py \
  tests/test_cruxeval_lens_top20.py tests/test_cruxeval_lens_semantic.py -q
test -f "$PROBES/prepared_all/gates.json"
```

The small tests construct lens targets and execution-grounded value pairs with
a fake tokenizer, and run stages 242/243 end to end against a real J/R pair
fitted on an 8-wide toy decoder. None of them loads model weights.

## 1. Fit the independent J/R pair only when absent

```bash
test -f "$LENS/j-lens/lens.pt" && test -f "$LENS/r-lens/lens.pt"
```

If that succeeds, continue to stage 235. If either file is absent:

```bash
$PY -m pip install --no-deps -e third_party/jacobian-lens
$PY scripts/201_lens_fit.py --model deepseek-coder-6.7b \
  --corpus "$CORPUS" --check-env
$PY scripts/201_lens_fit.py --model deepseek-coder-6.7b \
  --corpus "$CORPUS" --dim-batch 16 --dry-run
$PY scripts/201_lens_fit.py --model deepseek-coder-6.7b \
  --corpus "$CORPUS" --dim-batch 16 --dtype bfloat16
```

The final command is the expensive full-Jacobian fit. It checkpoints and
resumes automatically. J and R are fitted in one invocation with identical
model, corpus, recipe, and aggregation provenance.

## 2. Prepare and read the lenses on CruxEval

```bash
$PY scripts/235_cruxeval_lens_prepare.py \
  --prepared "$PROBES/prepared_all" \
  --output "$MECH/lens_targets"

$PY scripts/236_cruxeval_lens_read.py \
  --prepared "$MECH/lens_targets" --lens-dir "$LENS" --corpus "$CORPUS" \
  --output "$MECH/lens_readout" --model deepseek-coder-6.7b \
  --dtype bfloat16 --device cuda --checkpoint-every 5 \
  --unembed-batch-size 32 --resume
```

Stage 236 first reruns the E19 mechanical gates. It then produces token rank,
pass@k, sequence pass@k, and target-versus-real-output-distractor margins for J,
R, and the ordinary logit lens at every fitted layer and all four read sites.
Answer tokens are teacher-forced one at a time. Checkpoint rows are resumable.

## 3. Causal erasure and lens report

The released 6.7B fit normally contains source layers 4–30. Confirm them with
`meta.json` after stage 236. The causal panel uses a seeded 50-program sample,
the latest use and its post-use token, the call, and the first answer token:

```bash
$PY scripts/237_cruxeval_lens_ablate.py \
  --prepared "$MECH/lens_targets" --readout "$MECH/lens_readout" \
  --lens-dir "$LENS" --output "$MECH/lens_ablation_L4-12-20-28_n50" \
  --layers 4,12,20,28 --model deepseek-coder-6.7b \
  --dtype bfloat16 --device cuda --limit 50 --seed 42

$PY scripts/238_cruxeval_lens_report.py \
  --prepared "$MECH/lens_targets" --readout "$MECH/lens_readout" \
  --ablation "$MECH/lens_ablation_L4-12-20-28_n50" \
  --output "$MECH/lens_report"
```

Stage 237 edits the live model at representative use, post-use, call, and answer
sites and measures the model's own output logits. It does not score the lens
after editing it. The causal target is the first output token; the observational
stage still reads every output token and every use site.

## 4. Construct execution-grounded DAS pairs

```bash
$PY scripts/239_cruxeval_das_prepare.py \
  --prepared "$PROBES/prepared_all" --output "$MECH/das_pairs_multivariant" \
  --model deepseek-coder-6.7b --seed 42 --min-pairs 20 --max-pairs 200 \
  --max-variants-per-source 5

cat "$MECH/das_pairs_multivariant/meta.json"
```

This CPU stage may reject most CruxEval functions. It fails unless at least 20
certified pairs survive and calibration/test source groups are disjoint. Review
the acceptance count before allocating a GPU.

## 5. Fit and evaluate DAS across layers

Start with layer 1, where the synthetic-transfer probe was strongest. Stage 240
first evaluates clean behavior for every pair in both directions. Under
`paired_clean`, a pair enters DAS only when the clean model prefers the correct
continuation in both directions. The gate requires at least eight distinct
source functions in each split and records the raw and retained coverage. This
selection uses no intervention outcome. Do not run the cross-depth sweep until
layer 1 passes:

```bash
$PY scripts/240_cruxeval_value_das.py \
  --prepared "$MECH/das_pairs_multivariant" \
  --output "$MECH/das_L1_r1_paired" \
  --layer 1 --rank 1 --model deepseek-coder-6.7b \
  --dtype bfloat16 --device cuda --steps 200 --batch-size 8 \
  --lr 0.01 --seed 42 --behavior-policy paired_clean \
  --min-qualified-groups 8 --bootstrap 1000

cat "$MECH/das_L1_r1_paired/behavior_gate.json"
```

If that stage passes, run the remaining declared depths with the same pair
population and policy:

```bash
for LAYER in 5 16 31; do
  $PY scripts/240_cruxeval_value_das.py \
    --prepared "$MECH/das_pairs_multivariant" \
    --output "$MECH/das_L${LAYER}_r1_paired" \
    --layer "$LAYER" --rank 1 --model deepseek-coder-6.7b \
    --dtype bfloat16 --device cuda --steps 200 --batch-size 8 \
    --lr 0.01 --seed 42 --behavior-policy paired_clean \
    --min-qualified-groups 8 --bootstrap 1000
done

$PY scripts/241_cruxeval_value_das_report.py \
  --runs "$MECH/das_L1_r1_paired" "$MECH/das_L5_r1_paired" \
         "$MECH/das_L16_r1_paired" "$MECH/das_L31_r1_paired" \
  --output "$MECH/das_report_r1"
```

Stage 240 requires the clean model to distinguish the factual and
counterfactual answer tokens in both directions on the retained held-out
examples. It learns DAS and the
answer-only actuator on calibration functions, freezes both, and evaluates
both directions on test functions. The magnitude-matched random arm is asserted
never to move less state than DAS; the no-op must move exactly zero.

Run a rank sweep only after inspecting rank 1. Use new directories such as
`das_L16_r4`; never resume a completed result with changed hyperparameters.

## 6. Exploratory top-token inspection (stages 242–243)

Stages 235–238 ask a pre-declared yes/no question: is the recorded output token
rankable at a causal read position? Stages 242–243 ask the open one: *what do
the lenses actually put at the top of the vocabulary there*, in literal
tokenizer tokens, at shallow and middle depth. Nothing is refitted. Stage 242
reuses the stage-236 ranks; stage 243 reuses the same J/R checkpoints, the same
stage-235 positions, the same BOS adaptation and the same E19 validation.

This is an inspection experiment. A null, noisy, punctuation-heavy or
uninterpretable list is a **result**, not a failure. Only mechanical corruption
fails a stage: bad provenance, a selected layer no lens was fitted at, a
position outside the encoding, a malformed rank block, a tampered input
artifact.

### What the exploratory layer score measures

Stage 242 scores each `(lens, layer, read)` by reciprocal rank of the
ground-truth output tokens:

```
rr = 1 / (rank + 1)                 # rank is the 0-based full-vocabulary rank
per-program  = mean of rr within one (dataset_id, read)
score        = mean over programs, each program weighted equally
combined     = unweighted mean of the scores at use, post_use and call
consensus    = unweighted mean of the J and R combined scores
```

The two averaging steps exist because a CruxEval program with eleven use sites
and a four-token output contributes 44 rows at `use` while a short one
contributes 1; a row-level mean would let a handful of long programs set the
layer curve. J and R scores are combined without renormalization because they
are already directly comparable — the same reciprocal ranks over the same
vocabulary, programs, positions and target tokens. The exact formula is
recorded in `selected_layers.json` under `primary_statistic`.

`answer` is computed and reported at every layer as a **positive control** and
never selects a layer. A lens reading a teacher-forced answer token at the
answer position has been told the answer.

**This selection is exploratory and same-dataset.** Stage 242 picks layers from
the same clean CruxEval population that stage 243 then reads. It is not
held-out model selection, and `selection_status` says so in every artifact. Do
not report a discovered layer as if it had been validated out of sample.

### How requested percentages map to actual layers

Layer `l` is the residual stream *after* transformer block `l + 1`, so:

```
actual_depth_fraction = (layer + 1) / n_layers
```

The nominal checkpoints 10%, 15%, 20% and 25% are always included. Each is
mapped to the **nearest fitted source layer**, because the released recipe does
not fit every block: on the 6.7B stack the sources start at layer 4, so 10% of
32 blocks (layer 2) is unavailable and resolves to layer 4 at 15.625%. Every
checkpoint records `requested_percent`, `layer`, `actual_percent`,
`distance_percent` and `exact_fitted_layer_available`. These are inspection
points, not a prediction that semantics live at 15–25% depth.

Stage 242 additionally selects the J best pre-answer layer, the R best
pre-answer layer, the J/R consensus layer, and (by default) one fitted layer on
each side of the consensus. The final list is deduplicated.

### What positions are read

Stage 243 reads the verified stage-235 positions, one deterministic
representative site per program per read — the same rule the stage-237 causal
panel uses, so the lists and the erasure rows describe the same places:

| read       | position                                            |
|------------|-----------------------------------------------------|
| `use`      | the **latest** verified use site before the call boundary |
| `post_use` | that same site's post-use token                     |
| `call`     | the verified call position                          |
| `answer`   | the first teacher-forced answer step                 |

Nothing is averaged across positions: every printed list belongs to one token
position in one encoding. `--all-sites` widens the run to every use site and
every answer step; the default stays compact enough to finish.

### Tokens are not words

Every row is a **tokenizer vocabulary item**, not a word. Under deepseek-coder's
byte-BPE a vocabulary item is frequently a word fragment, a bare space, a
newline, a single digit or an indent run. `token_repr` is the Python `repr` of
the decoded text, so `' '`, `'\n'`, `' return'` and `'42'` stay distinguishable;
`token_text` is the raw decoding. The primary top-20 is **unfiltered**:
punctuation, whitespace and special tokens are all retained, flagged by the
`is_punctuation_only`, `is_whitespace_only` and `is_special_token` columns.
`top20_lexical_lists.csv.gz` drops those three classes and is a convenience
view only — it never replaces the primary list.

### Cluster commands

Stage 242 is CPU-only and takes seconds. Stage 243 needs the GPU.

```bash
$PY scripts/242_cruxeval_lens_layer_discovery.py \
  --readout "$MECH/lens_readout" --prepared "$MECH/lens_targets" \
  --output "$MECH/lens_layer_discovery" \
  --depth-percents 10,15,20,25 --consensus-neighbours 1

cat "$MECH/lens_layer_discovery/selected_layers.json"

$PY scripts/243_cruxeval_lens_top20.py \
  --prepared "$MECH/lens_targets" --readout "$MECH/lens_readout" \
  --discovery "$MECH/lens_layer_discovery" \
  --lens-dir "$LENS" --corpus "$CORPUS" \
  --output "$MECH/lens_top20_clean" --model deepseek-coder-6.7b \
  --dtype bfloat16 --device cuda --top-k 20 \
  --checkpoint-every 5 --unembed-batch-size 32 \
  --example-programs 20 --resume
```

Inspect `selected_layers.json` before spending the GPU: it lists exactly which
layers stage 243 will run and why each was chosen. Add `--limit 50` for a
smoke run, or `--all-sites` for the wide variant (several times the rows and
the runtime). Stage 243 checkpoints every `--checkpoint-every` programs and
`--resume` skips completed programs without recomputing or duplicating them.

### How to inspect the resulting lists

```bash
$PY - <<'PY'
import pandas as pd
d = "results/cruxeval/deepseek-coder-6.7b/mechanistic_n500/lens_top20_clean"
lists = pd.read_csv(f"{d}/top20_lists.csv.gz", keep_default_na=False)
print(lists.query("read == 'use'")[
    ["dataset_id", "lens", "layer", "actual_depth_percent",
     "source_token_repr", "top20_tokens", "target_ranks"]].head(30).to_string())
PY
```

Always pass `keep_default_na=False`. `token_text` holds raw decoded text, and a
vocabulary item spelled `NA`, `null`, `None` or `nan` is otherwise silently
turned into a missing value by pandas; `token_repr` is escaped and safe either
way.

`examples.md` renders a deterministic sample of about 20 programs with the
source excerpt, input, ground-truth output, read position, layer, relative
depth and all three top-20 lists side by side. `diagnostics.json` records how
often a list contained a ground-truth output token and what fraction of the
surfaced tokens were whitespace, punctuation or special — as measurements, not
gates.

The lists are observational. A recognisable token at a pre-answer position is
not evidence that the model uses it there; that claim needs the stage-237
erasure rows or the stage-240 interchange. The obfuscated replication is not
implemented yet: stage 243's outputs are shaped so the identical readout can
later be applied to execution-verified obfuscated variants.

## 7. J-lens semantic words, and the obfuscation replication (stages 244–246)

Stage 243 read the unfiltered vocabulary and found a program-independent format
prior: brackets, digits and whitespace, plus an R-lens that returned the same
function words for every program at every depth. Stages 244–246 ask the
narrower question — **when the lens is restricted to coherent words, does it
surface words about _this_ program?** — and then ask what obfuscation does to
the answer.

These stages use the **J-lens only**. The R-lens is loaded so the matched-pair
provenance gate still runs, and is never transported. The ordinary logit lens
rides along because it comes from the same forward pass at no cost; without it,
"J surfaces semantic words" is unfalsifiable.

### The filter cannot be the finding

Masking the vocabulary to words guarantees words come out. A list of
plausible-looking code vocabulary is therefore not evidence on its own. The
selection statistic is **specificity**:

```
specificity = overlap(top words, THIS program's word set)
            - mean overlap(top words, the OTHER programs' word sets)
```

A lens returning one fixed word list for every program scores 0 by
construction, however meaningful that list looks. That is a built-in
permutation control, and it is what the R-lens would have failed on the
unfiltered pass. Abundance-style scores — a predeclared execution lexicon, a
matched non-execution control lexicon, and the repeat rate of each word across
programs — are written to the same table, so a different objective can be
chosen later without another GPU run.

Each program's word set is parsed from its own AST and recorded output, never
from the lens, and split three ways because obfuscation acts on them
differently:

| kind | contents | what the ladder does to it |
|---|---|---|
| `lexical` | the program's own identifier names | destroyed by level 1 (rename), by construction |
| `operational` | executed methods, builtins, control constructs | untouched by rename; rewritten by level 4 (flatten) |
| `type` | words naming the output's type | unchanged at every level |

So `specificity_lexical` collapsing at level 1 is expected and means nothing on
its own. The informative outcome is whether `specificity_operational` and
`specificity_type` survive rename and fail at flattening — which is the
signature a readout tracking execution would have, and matches the E9/R4
finding that renaming is survivable mid-layer while flattening is not.

### Selection is held out

Stage 244 splits programs into calibration and test by `source_group`, the same
group-disjoint rule stages 234 and 240 use. The `(read, layer)` cell is chosen
on calibration by `specificity_semantic` (= operational + type, ties to the
shallower layer) and reported on the disjoint test groups. `answer` is computed
at every layer as a positive control and can never be selected: a teacher-forced
answer position has been told the answer.

Stage 246 then **freezes** that cell. Nothing is re-selected per obfuscation
condition, so the contrast cannot be quietly re-tuned, and every condition is
scored against the **clean** program's word sets — the question is whether the
obfuscated state still surfaces the original semantics.

### Cluster commands

```bash
$PY scripts/244_cruxeval_jlens_semantic_sweep.py \
  --prepared "$MECH/lens_targets_rebuilt" \
  --lens-dir "$LENS" --corpus "$CORPUS" \
  --output "$MECH/jlens_semantic_L4-25" --model deepseek-coder-6.7b \
  --dtype bfloat16 --device cuda --first-layer 4 --last-layer 25 \
  --top-k 20 --seed 42 --checkpoint-every 10 --resume

cat "$MECH/jlens_semantic_L4-25/selected_site.json"

$PY scripts/245_cruxeval_obfuscate.py \
  --prepared "$MECH/lens_targets_rebuilt" \
  --output "$MECH/obfuscated_L0-4" --levels 0,1,2,3,4 \
  --model deepseek-coder-6.7b --seed 42 --min-programs 20

cat "$MECH/obfuscated_L0-4/meta.json"     # check acceptance per level first

$PY scripts/246_cruxeval_obfuscation_semantic.py \
  --prepared "$MECH/lens_targets_rebuilt" \
  --obfuscated "$MECH/obfuscated_L0-4" \
  --sweep "$MECH/jlens_semantic_L4-25" \
  --lens-dir "$LENS" --corpus "$CORPUS" \
  --output "$MECH/obfuscation_semantic" --model deepseek-coder-6.7b \
  --dtype bfloat16 --device cuda --top-k 20 --resume
```

Stage 244 and 246 need the GPU; 245 is CPU-only and runs the obfuscated code,
so it is the one to inspect before allocating anything. Stage 244 sweeps 22
layers × 4 reads but pays for only one forward pass per encoding, so its cost
is close to stage 243's. Stage 246 reads a single layer at a single position
and is the cheapest of the three.

Check `obfuscated_L0-4/meta.json` before running 246. Levels 3 (`encode`) and 4
(`flatten`) rewrite integer expressions and control flow, and a CruxEval
function operating on strings or dicts will often be rejected by execution
verification. `variants_by_level` and `audit.csv` record how many survived and
why the rest did not; a level with few survivors constrains what can be
concluded from it, and that is a result about coverage, not about the lens.

### Reading the outputs

```bash
$PY - <<'PY'
import pandas as pd, json
d = "results/cruxeval/deepseek-coder-6.7b/mechanistic_n500/jlens_semantic_L4-25"
print(json.dumps(json.load(open(f"{d}/selected_site.json")), indent=2))
s = pd.read_csv(f"{d}/semantic_scores.csv")
s = s[(s.lens == "j-lens") & (s.read != "answer") & (s.split == "test")]
print(s.sort_values("specificity_semantic", ascending=False)[
    ["read","layer","specificity_operational","specificity_type",
     "specificity_lexical","execution_lexicon_rate","control_lexicon_rate",
     "list_repeat_rate"]].head(15).to_string(index=False))
PY
```

`examples.md` shows ~20 programs at the selected site with each program's own
operational and type words listed, and the surfaced words that hit them in
bold. The same bold rate in the logit-lens row is the chance rate for this
vocabulary — compare against it before reading anything into a J-lens list.

Interpretation limits that apply to every number here: the word mask makes
words appear whether or not anything was computed, so only the specificity
columns and the logit-lens control carry information; the readout is
observational, so a program-specific word is not evidence the model *uses* it
there; and a clean-arm specificity at or below 0 means there was nothing for
obfuscation to remove, and no obfuscation conclusion follows.

## Outputs

- `lens_readout/lens_rows.csv.gz`: one row per program, site, output token,
  lens, and layer.
- `lens_readout/lens_summary.csv`: token and whole-output pass@k summaries.
- `lens_ablation_*/ablation_rows.csv`: live-model edits and exact edit doses.
- `lens_report/`: certified tables, figure, and claim-scoped report.
- `das_pairs_multivariant/pairs.jsonl`: source-disjoint executable
  counterfactual pairs, with up to five variants per function.
- `das_L*/interchange_rows.csv`: held-out two-direction causal outcomes.
- `das_L*/interchange_summary.csv`: source-bootstrap intervals per control arm.
- `das_report_r1/`: combined cross-layer table, rows, figure, and report.
- `lens_layer_discovery/alignment_by_layer.csv`: program-equal MRR, median
  rank and pass@1/5/10/20 per lens, layer and read, plus a
  `combined_pre_answer` row.
- `lens_layer_discovery/best_layers.csv`: best layer per (lens, read), per
  lens combined pre-answer, and the J/R consensus; `selects_a_layer` is false
  on the `answer` rows.
- `lens_layer_discovery/selected_layers.json`: the layers stage 243 will read,
  the depth-checkpoint mapping, and the exact selection formula.
- `lens_top20_clean/top20_long.csv.gz`: one row per vocabulary token rank, with
  logit, full-vocabulary log probability, and the punctuation/whitespace/
  special/in-input flags.
- `lens_top20_clean/top20_lists.csv.gz`: one row per program, read, lens and
  layer, with `top20_tokens`, `top20_token_ids` and `top20_logits`.
- `lens_top20_clean/top20_lexical_lists.csv.gz`: convenience view only.
- `lens_top20_clean/examples.md`: a deterministic ~20-program sample.
- `jlens_semantic_L4-25/semantic_scores.csv`: specificity, lexicon rates and
  repeat rate per lens, read, layer and split.
- `jlens_semantic_L4-25/selected_site.json`: the held-out (read, layer) and its
  test-split numbers, J-lens and logit-lens control.
- `jlens_semantic_L4-25/semantic_word_lists.csv.gz`: the word list per program,
  read, layer and lens.
- `obfuscated_L0-4/lens_programs.jsonl`: execution-verified variants with every
  anchor rebuilt from the variant's own source; `audit.csv` says why each
  rejected variant was rejected.
- `obfuscation_semantic/obfuscation_scores.csv`: the frozen site's specificity
  per obfuscation level; `report.md` renders the same table.

Every stage writes `gates.json`. A scientific null can pass; malformed
provenance, alignment, execution, split leakage, failed optimizer convergence,
weak clean forced-choice behavior, or broken controls fail the stage.
