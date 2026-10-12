# Binding in effect: following control flow from synthetic to real code (stage 280)

## Question

Does a code model track which assignment to a variable is in effect when the
variable is used, across redefinitions, branches, loops and `with` blocks, in
a way that cannot be explained by word matching? And where does a linear
readout of that information stop working as code moves from controlled
synthetic programs to benchmark code to real repositories?

## The task

Take a use `u` of a local variable `v`, and an earlier assignment `d` to `v`
followed, before `u`, by at least one more assignment to `v`. Label 1 if
some execution path from `d` to `u` avoids every later assignment (`d` may
still be the binding in effect), 0 otherwise.

```python
x = load()          x = load()          x = load()          if c:
if cached:          with lock:          x = x.strip()           x = a()
    x = fetch()         x = fetch()                         else:
                                                                x = b()
use(x)   # 1        use(x)   # 0        use(x)   # 0             use(x)   # 0
```

A closer assignment always exists, so "the nearest assignment" never answers
the question. Each candidate is assigned the mechanism behind its label:

| category | meaning |
|---|---|
| `survives` | some path from `d` to `u` avoids every redefinition |
| `overwritten` | `d` reaches `u`'s position, but every such path redefines `v` |
| `exclusive` | no path leads from `d` to `u` at all (mutually exclusive branches) |

`exclusive` comes from a counterfactual: the labeler is rerun with every other
assignment to `v` renamed. Self-updates (`x = x.strip()`, `x += 1`) count as
redefinitions and are flagged.

**Why this property and not lexical shadowing.** Shadowing is rare inside
single functions. CruxEval has almost none and CodeSearchNet very little.
Binding-in-effect occurs in about 14% of CodeSearchNet functions and in about
220 of CruxEval's 800.

## What rules out word matching

Natural code can't be controlled, so the claim rests on **controlled pairs**.
These are two programs whose members agree on every shallow cue, but with
opposite labels. `controls.certify` admits a pair only if its members have:

- exactly the same multiset of tokens before the use, and between `d` and `u`;
- the same tokens within ±3 of `d` and of `u`, at the same token positions;
- the same number of redefinitions, the same indentation of `d`, `u` and
  every redefinition, the same indentation-heuristic prediction, and the same
  self-update flags;
- opposite labels.

The only thing that differs is **which of two same-indentation assignments
targets `v`**, and so which construct encloses the redefinition. In the pair
below, the two members contain the same words:

```python
x = load()                       x = load()
with lock:                       with lock:
    y = fetch()                      x = fetch()       # always runs
if cached:                       if cached:
    x = parse()                      y = parse()       # may not run
use(x)    # 1 (survives)         use(x)    # 0 (overwritten)
```

So the following can't separate the members, and each scores exactly one
member of every pair correctly:

- keyword detection ("is there a `with`?");
- token or assignment counts;
- indentation;
- the local context of `d` and `u`.

A probe that gets **both** members right must read how the constructs are
arranged and what they do: that a `with` body always runs, that an `if` body,
a loop body or a single branch may not, and that `if`/`else` arms assigning
`v` on every arm amount to an unconditional redefinition.

The pairs come from three sources, one per controlled domain:

| domain | how pairs are built |
|---|---|
| `synthetic` | counterbalanced templates (below) |
| `hard` | `swap_twin` on random compositional programs |
| `real_twins` | `swap_twin` on CodeSearchNet functions from test-split repositories |

`swap_twin` swaps the assigned names of one redefinition of `v` and one
assignment to another name at the same indentation between `d` and `u`. It
keeps the result only if the label flips and the pair certifies. Swaps
inside the same block are never tried: both statements run under the same
condition, so the label can't flip.

What this does not rule out: a probe could read "the construct enclosing the
last assignment to `v` is `with`" as a learned configuration, not by
simulating execution. That configuration *is* the control-flow fact being
tested, but it's a syntactic-semantic association, not proof of step-by-step
execution. Linear decodability also shows the information is present at
`d`/`u`, not that the model uses it.

## Domains

| domain | what it is | role |
|---|---|---|
| `synthetic` | counterbalanced template pairs | train + test, paired |
| `hard` | swap twins of compositional programs | train + test, paired |
| `cruxeval` | CruxEval's short benchmark functions (`code` only) | test only, natural |
| `real` | CodeSearchNet Python, split by repository | train + test, natural |
| `real_twins` | swap twins of CodeSearchNet test-repository functions | test only, paired |

**synthetic.** Both members contain the same two constructs and the same
statements:

| family | label 1 (survives) | label 0 (overwritten) |
|---|---|---|
| `guard_swap` | `if/while/for c: v = A` · `with m: w = B` | `if/while/for c: w = A` · `with m: v = B` |
| `else_swap` | `if a: v=A else: w=B` · `if b: w=C else: v=D` | `if a: v=A else: v=B` · `if b: w=C else: w=D` |
| `target_swap` | `for v in s:` · `with m as w:` | `for w in s:` · `with m as v:` |

- **Variation:** block order is random, and `else_swap` optionally has an
  `elif` arm.
- **Self-updates:** in about 40% of pairs every assignment in the constructs
  reads `v`.
- **Use form:** in about 30% the use is itself an update (`v = f(a, b, v)`).
- **Certification:** on the deepseek tokenizer, all generated template pairs
  certify.

**hard.** Random functions built from:
- 1–8 assignments to `v`;
- 2–12 assignments to three distractor names;
- if/if-else/for/while/with blocks nested up to depth 4;
- self-updates off, or on at rate 0.5.

The final line uses `v`. About a quarter of programs yield a certified twin,
and `prepare` scans until it has `--hard-pairs` pairs. The knobs are stored
with each candidate.

**cruxeval.** Every supported CruxEval function, test-only and natural. Most
of its candidates are single-redefinition self-update chains.

**real.** The CodeSearchNet Python train split, pinned to an immutable parquet
commit and scanned in seeded random order until 3,000 functions contain a
candidate. Duplicates (identical `ast.dump`) are dropped, functions over
1,024 tokens are excluded, and splits are 60/20/20 by repository.

**real_twins.** Only about 3% of real functions with a candidate admit a
certified twin. After the natural quota is filled, `prepare` keeps scanning
**test-split repositories only** until it has `--real-twins` pairs. No real
twin shares a repository with any real training or validation program.

## Labels

Labels are beniget may-reach chains, through `src/data/cruxeval_graph.py`.
Beniget is exact for sequences, `if`/`elif`/`else`, loops, `with` and walrus.
Unit tests found it wrong in three places: after early exits, for loop `else`
clauses, and around `try`. A candidate is kept only if the region between `d`
and `u` contains none of the following:

| excluded | why |
|---|---|
| `return`, `raise`, `break`, `continue`, `del` | jumps; beniget over-approximates reachability |
| `try`, `match`, loop `else` | beniget treats guaranteed paths as optional |
| `while True`, `if 0` (constant tests) | path feasibility depends on the value |
| nested `def`/`class`/`lambda`; `global`/`nonlocal` anywhere | changes which variable a name denotes |

Every definition precedes the use, so a decoder has read everything that
decides the label by the time it reaches `u`. `tests/test_def_survival.py`
pins the semantics with hand-labelled cases, every excluded construct, every
category, the certification of every template family, and the rejection of
an `if`-versus-`with` keyword pair.

## Probe, baselines and analysis

- **Read point:** the residual stream at the last token of `d` and of `u`, for
  the embedding (layer −1) and every block output before the final norm.
  Features are `[h_d, h_u, h_d − h_u, |h_d − h_u|]`.
- **Probe:** logistic regression, C = 0.1, balanced class weights.
- **Training:** each domain with a train split (synthetic, hard, real) trains
  a probe on the same number of candidates. Each probe is scored on all five
  domains.
- **Layer choice:** on the probe's **own** domain's validation split.

| baseline | what it rules out |
|---|---|
| **shallow**: logistic regression on the bag of every token before the use, the bag of tokens between `d` and `u`, ±3 surface windows, redefinition count and indentations, the heuristic, self-update flags and recency of the last redefinition | everything word matching and shallow structure can do; the main comparison is probe − shallow |
| **surface**: ±3 token windows plus bucketed distance | local lexical shortcuts |
| **heuristic**: killed iff a redefinition is indented no deeper than the use | reading indentation; exactly 0.5 on paired domains |
| **embedding**: the probe at layer −1 | token identity without context |
| **shuffled**: the probe trained on permuted labels | probe capacity (Hewitt & Liang control task) |

**Pair accuracy** (paired domains) is the fraction of pairs with both members
right. Chance is 0.25. A reader of the matched cues scores 0.

To locate degradation:

1. **`breakdown.csv`**: accuracy of each probe on each domain within every
   covariate bin, next to all baselines. The covariates are:
   - `category`;
   - number of redefinitions;
   - self-update, and whether the use is itself an update;
   - depth of the use and blocks between `d` and `u`;
   - other assignments between;
   - token distance;
   - whether the heuristic is right.
2. **`standardized.csv`**: each domain's accuracy reweighted to real test
   code's mix over `category × n_killers × self_update × depth_u`, with
   group-bootstrap intervals.
   - If hard-synthetic accuracy standardized to real's mix falls to real's
     raw accuracy, the measured structure explains the gap.
   - If it stays high, something else does: naturalness, identifier
     semantics, length or a shifted representation.
3. **`layers.csv`**: balanced accuracy per layer for every train → test pair.

These comparisons separate three explanations:
- **Synthetic → real twins vs. synthetic → real:** whether the gap comes from
  real code's uncontrolled cues or from the representation itself.
- **Real → real vs. real → real twins:** whether real code's own probe reads
  control flow or shallow cues.
- **Probes trained on different domains, within one covariate bin:** whether
  information is missing at the read point (every probe fails) or the
  readout doesn't transfer (only the transferred probe fails).

## Run on the cluster

The commands are bash, from the repository root. `beniget==0.5.0` and
`gast==0.7.0` are in `requirements-cruxeval.txt`. Set `MODEL` and `RUN`
together every time.

```bash
MODEL=deepseek-coder-1.3b
RUN="results/binding_flow/${MODEL}/seed42"

# 1. CPU, needs network for CodeSearchNet, CruxEval and the tokenizer. The twin
#    searches take roughly 30-60 minutes; run in a screen.
.venv/bin/python -m pytest tests/test_def_survival.py tests/test_cruxeval.py -q
.venv/bin/python scripts/280_def_survival.py prepare --model "$MODEL" --output "$RUN/prepared"
cat "$RUN/prepared/report.md"

# 2. GPU, in a screen. Stores only the d/u read points.
.venv/bin/python scripts/280_def_survival.py extract \
  --prepared "$RUN/prepared" --output "$RUN/activations" --resume

# 3. CPU, in a screen.
.venv/bin/python scripts/280_def_survival.py evaluate \
  --prepared "$RUN/prepared" --activations "$RUN/activations" --output "$RUN/evaluation"
cat "$RUN/evaluation/report.md"
```

The defaults are 3,000 real functions, 300 real twins, 3,000 template pairs
and 3,000 hard pairs. Repeat for `deepseek-coder-6.7b` and `starcoder2-3b` in
their own `RUN` directories, one model at a time. For seed sensitivity, rerun
all three stages with `--seed 43` and `--seed 44`.

- **Overwrites:** every stage refuses to write into a non-empty directory;
  only `extract --resume` continues a run with identical arguments.
- **Reproducibility:** `prepared/meta.json` records the CodeSearchNet and
  CruxEval revisions.
- **Offline:** pass `--input-jsonl` (CodeSearchNet rows) and `--cruxeval-jsonl`
  (rows with `id`, `code`).

The `results/def_survival/` runs from 2026-10-09 used an earlier version of
this task: uncontrolled keyword pairs, no self-updates, and only synthetic and
real domains. Don't mix their numbers with this version's.

## Outputs

| file | content |
|---|---|
| `prepared/coverage.csv` | programs, groups, candidates, labels, `exclusive`, self-updates, mean redefinitions and heuristic accuracy per domain × split |
| `prepared/exclusions.csv` | excluded programs and candidates by domain and reason (incl. hard programs without a certified twin) |
| `prepared/candidates.csv`, `programs.jsonl` | every candidate with anchors and covariates (hard: plus knobs); exact sources and token ids. Paired program ids end in `_a`/`_b` or `_1`/`_0` |
| `evaluation/summary.csv` | every train → test cell at the selected layer: probe, pair accuracy, all baselines, intervals |
| `evaluation/breakdown.csv` | accuracy by covariate bin, per train → test cell |
| `evaluation/standardized.csv` | accuracy standardized to real code's structural mix |
| `evaluation/layers.csv` | probe and shuffled-control balanced accuracy per layer and split |
| `evaluation/predictions.csv.gz` | per-candidate predictions for every layer and training domain (gitignored) |
| `evaluation/fits.csv` | convergence of every fit |

## Limits

- **Labels:** static may-reach on a supported subset of Python, not dynamic
  traces.
- **Real twins are counterfactual:** one assignment target is swapped, so the
  code is realistic in context but no longer the original program.
- **Coverage:** CodeSearchNet functions lack their repository context, every
  accepted subset is conditioned on containing a candidate, and CruxEval is
  too small to train on.
- **Standardization** only compares domains on the cells they share.
