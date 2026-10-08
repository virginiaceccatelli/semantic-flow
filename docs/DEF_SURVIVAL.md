# Definition survival: synthetic minimal pairs versus CodeSearchNet (stage 280)

## Question

Linear probes read data-flow facts from code models almost perfectly on
synthetic programs. Does a probe trained on controlled synthetic programs read
the same fact from human-written code, and does a probe trained on human code
recover more? Stage 280 asks this for one property, with the same labeler,
read positions, probe and baselines in both domains.

## The property

Take a use `u` of a local variable `v`, and an earlier assignment `d` to `v`
that is followed, before `u`, by at least one more assignment to `v`. **Does
`d` still reach `u`?** Label 1 if some execution path from `d` to `u` avoids
every later assignment, 0 if every path overwrites `d`.

```python
x = load()            x = load()            x = load()
if cached:            with lock:            if cached:
    x = fetch()           x = fetch()           x = fetch()
                                            else:
                                                x = parse()
use(x)   # 1          use(x)   # 0          use(x)   # 0
```

Why this property:

- **It needs control flow.** In every candidate a closer redefinition exists
  by construction, so "the nearest assignment" never answers it. The model
  has to tell an optional overwrite (`if` with no `else`, a loop body, a loop
  variable) from an unconditional one (sequence, `with`, both branches).
- **It is common in human code.** About 14% of sampled CodeSearchNet
  functions contain a supported candidate, e.g. `if x is None: x = []`. Lexical
  shadowing, the obvious alternative, is rare in single functions.
- **It has one-token minimal pairs.** `if ctx:` ↔ `with ctx:` flips the label
  while the definition, the use and the distance between them stay identical.
- **The answer comes before the read point.** Every definition precedes `u`
  in the text, so a decoder has read everything that decides the label by the
  time it reaches `u`.

## Labels

Labels are beniget may-reach chains, through the existing
`src/data/cruxeval_graph.py` reference graph. Beniget is exact for sequences,
`if`/`elif`/`else`, loops, `with` and walrus. Unit tests found it wrong in
three places: after early exits (`if c: x = 2` / `else: return` still reports
the first `x` as reaching), for loop `else` clauses, and around `try`.
Candidates are therefore kept only when the region between `d` and `u`
contains none of the following:

| excluded | why |
|---|---|
| `return`, `raise`, `break`, `continue`, `del` | jumps; beniget over-approximates reachability |
| `try`, `match`, loop `else` | beniget treats guaranteed paths as optional |
| `while True`, `if 0` (constant tests) | path feasibility depends on the value |
| nested `def`/`class`/`lambda`; `global`/`nonlocal` anywhere | changes which variable a name denotes |
| a redefinition that reads `v` (`x = x + 1`, `x += 1`) | so "killed" means discarded, not transformed |

Exclusions are counted by reason in `exclusions.csv`. The hand-labelled
cases in `tests/test_def_survival.py` define the intended semantics. Each
excluded construct has a test.

## Data

**Real.** CodeSearchNet Python train split, pinned to an immutable parquet
commit. Functions are scanned in seeded random order until 3,000 contain at
least one candidate. Formatting-insensitive duplicates (identical `ast.dump`)
are dropped. Functions longer than 1,024 tokens are excluded, never truncated.
Splits are 60/20/20 by repository hash. A 40k-function sample yields roughly
3.2 candidates per accepted function, 62% of them surviving.

**Synthetic.** Paired functions from `src/defsurvival/synthetic.py`, four
construction families:

| family | label 1 (survives) | label 0 (killed) |
|---|---|---|
| `if_vs_with` | `if ctx: v = …` | `with ctx: v = …` |
| `else_branch` | `if c: v = …` / `else: w = …` | `if c: v = …` / `else: v = …` |
| `loop_target` | `for v in seq: …` | `with seq as v: …` |
| `indent_swap` | `if c:` / `    v = …` / `w = …` | `if c:` / `    w = …` / `v = …` |

Names, expressions, filler statements, the use form, and whether `d` is a
parameter are randomized. Each pair is kept only if the reference labeler
reproduces both intended labels, both members have the same token length, `d`
and `u` sit at the same token positions, and the lexical features around `d`
and `u` are identical. Splits are by pair, so a pair never straddles train and
test.

## Probe and baselines

The read point is the residual stream at the last token of `d` and of `u`, for
the embedding (layer −1) and every block output before the final norm. The
pair representation is stage 20's `[h_d, h_u, h_d − h_u, |h_d − h_u|]`.
The probe is logistic regression with C = 0.1 and balanced class weights.

Each domain trains on the same number of candidates (`--max-train-rows`,
capped by the smaller pool), and every probe is scored on both domains'
held-out splits: four train → test cells. The layer is chosen on the probe's
**own** domain's validation split, so synthetic → real transfer never sees a
real label.

| baseline | what it rules out |
|---|---|
| **surface**: logistic regression on token ids within ±3 of `d` and `u`, plus bucketed distance, fitted on the same training rows | local lexical shortcuts; exactly 0.5 on synthetic by construction |
| **heuristic**: killed iff some redefinition is indented no deeper than the use | reading indentation, not semantics; ≈0.83 on real, 0.625 on synthetic |
| **embedding**: the probe at layer −1 | token identity without context |
| **shuffled**: the probe trained on permuted labels (Hewitt & Liang control task) | probe capacity |

The main metric is balanced accuracy, because real labels are about 62/38.
Intervals are 95% bootstrap intervals that resample whole repositories (real)
or pairs (synthetic). Differences such as probe − surface are bootstrapped
jointly. `strata.csv` splits test accuracy into **heuristic_wrong** cases
(the semantically hard ones) and **heuristic_right**, and synthetic further
by family.

## Run on the cluster

The commands are bash, from the repository root. `beniget==0.5.0` and
`gast==0.7.0` are in `requirements-cruxeval.txt`.

```bash
MODEL=deepseek-coder-1.3b
RUN="results/def_survival/${MODEL}/seed42"

# 1. CPU, needs network for CodeSearchNet and the tokenizer (~5 min).
.venv/bin/python -m pytest tests/test_def_survival.py -q
.venv/bin/python scripts/280_def_survival.py prepare --model "$MODEL" --output "$RUN/prepared"
cat "$RUN/prepared/report.md"

# 2. GPU. Stores only the d/u read points (~2 GB for 1.3b, ~6 GB for 6.7b).
.venv/bin/python scripts/280_def_survival.py extract \
  --prepared "$RUN/prepared" --output "$RUN/activations" --resume

# 3. CPU.
.venv/bin/python scripts/280_def_survival.py evaluate \
  --prepared "$RUN/prepared" --activations "$RUN/activations" --output "$RUN/evaluation"
cat "$RUN/evaluation/report.md"
```

Repeat with `MODEL=deepseek-coder-6.7b` and `MODEL=starcoder2-3b`, each in its
own directory, since tokenization changes the populations. For seed
sensitivity, rerun all three stages with `--seed 43` and `--seed 44` and
report all of them. Every stage refuses to write into a non-empty directory.
Only `extract --resume` continues a run with identical arguments. To reproduce a dataset
exactly, pass `--revision <commit>` from `prepared/meta.json`. Offline, pass
`--input-jsonl` with `repository_name`, `func_code_url` and `whole_func_string`
per row.

## Outputs

| file | content |
|---|---|
| `prepared/coverage.csv` | programs, groups, candidates, label balance and heuristic accuracy per domain × split |
| `prepared/exclusions.csv` | excluded programs and candidates by reason |
| `prepared/candidates.csv`, `programs.jsonl` | every labelled candidate with anchors; exact sources and token ids |
| `evaluation/summary.csv` | **main result**: four cells, selected layer, all baselines, intervals |
| `evaluation/strata.csv` | heuristic-right vs heuristic-wrong cases; synthetic families |
| `evaluation/layers.csv` | probe and shuffled-control balanced accuracy for every layer and split |
| `evaluation/predictions.csv.gz` | per-candidate predictions for every layer and training domain |
| `evaluation/fits.csv` | convergence of every fit; `report.md` flags any that did not converge |

## Limits

Labels are static may-reach on a supported subset of Python, not dynamic
traces. They hold on the excluded constructs only to the extent stated above.
CodeSearchNet functions lack their repository context, and the accepted subset
is conditioned on containing a candidate, so it does not represent
CodeSearchNet as a whole. Synthetic generalization is measured across new
instances of four families, not unseen families. A decodable label shows the
information is linearly present at `d`/`u`, not that the model uses it. A
causal test is separate work.
