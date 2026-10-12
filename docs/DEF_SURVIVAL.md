# Binding in effect: synthetic, CruxEval and CodeSearchNet (stage 280)

## Question

Does a code model track which assignment to a variable is in effect when the
variable is used, across redefinitions, branches, loops and `with` blocks? Is
the direction a probe finds the same in synthetic code, benchmark code
(CruxEval) and real repositories (CodeSearchNet)? What does that direction do
when added to the model, and what does the published J-lens read at the end of
the code?

## The task

Take a use `u` of a local variable `v`, and an earlier assignment `d` to `v`
followed, before `u`, by at least one more assignment to `v`. Label 1 if some
execution path from `d` to `u` avoids every later assignment (`d` may still be
the binding in effect), 0 otherwise.

```python
x = load()          x = load()          x = load()          if c:
if cached:          with lock:          x = x.strip()           x = a()
    x = fetch()         x = fetch()                         else:
                                                                x = b()
use(x)   # 1        use(x)   # 0        use(x)   # 0             use(x)   # 0
```

A closer assignment always exists, so "the nearest assignment" never answers
the question. Each candidate is assigned a mechanism:

| category | meaning |
|---|---|
| `survives` | some path from `d` to `u` avoids every redefinition |
| `overwritten` | `d` reaches `u`'s position, but every such path redefines `v` |
| `exclusive` | no path leads from `d` to `u` at all (mutually exclusive branches) |

Self-updates (`x = x.strip()`, `x += 1`) count as redefinitions and are
flagged. Lexical shadowing was rejected as the task because it barely occurs
inside single functions.

## Domains

| domain | what it is | split |
|---|---|---|
| `synthetic` | counterbalanced template pairs (below) | 60/20/20 by pair |
| `cruxeval` | CruxEval's benchmark functions (`code` only), ≈410 candidates in ≈220 functions | 60/20/20 by program |
| `real` | CodeSearchNet Python, 3,000 functions with a candidate | 60/20/20 by repository |

Every domain trains its own probe, and every probe is scored on every domain's
test split. CruxEval is small (about 250 training candidates, 80 test), so its
cells have wide intervals. By default each domain trains on up to
`--max-train-rows` candidates. `--match-budget` instead trains all three on
CruxEval's size.

## Ruling out word matching on synthetic code

Each synthetic pair is two functions built from the **same tokens**: the same
two constructs, the same statements, the same final optional redefinition.
They differ only in which of two same-indentation assignments targets `v`, so
whether `d` survives depends on what the enclosing constructs do:

| family | label 1 (survives) | label 0 (overwritten) |
|---|---|---|
| `guard_swap` | `if/while/for c: v = A` · `with m: w = B` | `if/while/for c: w = A` · `with m: v = B` |
| `else_swap` | `if a: v=A else: w=B` · `if b: w=C else: v=D` | `if a: v=A else: v=B` · `if b: w=C else: w=D` |
| `target_swap` | `for v in s:` · `with m as w:` | `for w in s:` · `with m as v:` |

`controls.certify` admits a pair only if both members have:
- the same multiset of tokens before the use, and between `d` and `u`;
- the same tokens within ±3 of both anchors, at the same positions;
- the same redefinition count, indentations, distance to the last
  redefinition, indentation-heuristic prediction and self-update flags;
- opposite labels.

So keyword detection, token counts, indentation and recency each score exactly
one member of every pair. **Pair accuracy** (both members right) has chance
0.25 and is 0 for any reader of those cues. On the deepseek tokenizer every
generated pair certifies.

CruxEval and CodeSearchNet can't be controlled this way. There, the comparison
is against the strongest shallow reader below.

## Labels

Labels are beniget may-reach chains, through `src/data/cruxeval_graph.py`.
Beniget was found wrong after early exits, for loop `else` clauses and around
`try`, so a candidate is kept only if the region between `d` and `u` contains
none of the following:

| excluded | why |
|---|---|
| `return`, `raise`, `break`, `continue`, `del` | jumps; beniget over-approximates reachability |
| `try`, `match`, loop `else` | beniget treats guaranteed paths as optional |
| `while True`, `if 0` (constant tests) | path feasibility depends on the value |
| nested `def`/`class`/`lambda`; `global`/`nonlocal` anywhere | changes which variable a name denotes |

`tests/test_def_survival.py` pins the semantics with hand-labelled cases.

## Probe, baselines and directions

- **Inputs:** deepseek-coder declares a BOS token that the repository's fast
  tokenizer doesn't add, so every stage prepends it (`pipeline.encode`). That
  matches how the J-lens was fitted. For StarCoder2 no BOS is added.
- **Read point:** the residual stream at the last token of `d` and of `u`, for
  the embedding (layer −1) and every block output before the final norm.
  Probe features are `[h_d, h_u, h_d − h_u, |h_d − h_u|]`.
- **Probe:** logistic regression, C = 0.1, balanced class weights. The layer is
  chosen on the training domain's own validation split.

| baseline | what it is |
|---|---|
| **shallow** | logistic regression on the bag of every token before the use, the bag of tokens between `d` and `u`, ±3 surface windows, redefinition count, indentations and recency, the heuristic and self-update flags; C chosen on validation |
| **best shallow** | shallow or the indentation heuristic, whichever is better on the training domain's validation split; the main comparison is probe − best shallow |
| **surface** | ±3 token windows plus bucketed distance |
| **heuristic** | killed iff a redefinition is indented no deeper than the use |
| **embedding** | the probe at layer −1 |
| **shuffled** | the probe trained on permuted labels (control task) |

**Directions.** For each domain, the residual direction is the difference of
class means of the use-token state (survives minus overwritten) at its probe's
layer:
- `direction_grid.csv` tests each direction as a threshold classifier on
  every domain's test split.
- `cosines.csv` compares the three directions at every layer.

**Breakdown.** `breakdown.csv` gives each probe's accuracy on each domain
within bins of:
- mechanism;
- redefinition count;
- self-update, and whether the use is itself an update;
- depth of the use, and blocks crossed between `d` and `u`;
- other assignments between;
- token distance;
- whether the heuristic is right.

## Steering and J-lens (`steer`)

For each direction and for 40 test snippets per domain (`--n-prompts`):

- **Code section:** the program up to the end of the line holding the use.
  Its last token is where the decoder has read everything that decides the
  label.
- **J-lens:** the published Jacobian lens fitted by stage 201
  (`results/workspace_lens/<model>/j-lens/lens.pt`, or `--lens-dir`). It gives
  the top-10 tokens at the last token, at the direction's probe layer,
  unsteered and steered. The lens is fitted for blocks 0 to n−3; a probe layer
  above that is read with the logit lens and labelled so.
- **Steering:** `c × (distance between the class means) × direction` is added
  to the residual at that layer, at the last token and at every generated
  token, for `c` in `--coefficients` (default −8, −4, 4, 8). Positive `c`
  pushes toward "survives". The stage records:
  - the next-token top-10;
  - a 24-token greedy continuation next to the unsteered one;
  - the tokens whose log-probability the direction raises and lowers on
    average, per test domain.
- **The bare direction:** the J-lens and logit-lens reading of `±` the
  direction itself.

## Run on the cluster

The commands are bash, from the repository root. Set `MODEL` and `RUN`
together, and run one model at a time.

```bash
MODEL=deepseek-coder-1.3b
RUN="results/binding_flow/${MODEL}/seed42"

.venv/bin/python -m pytest tests/test_def_survival.py tests/test_cruxeval.py -q
.venv/bin/python scripts/280_def_survival.py prepare --model "$MODEL" --output "$RUN/prepared"   # CPU, network
cat "$RUN/prepared/report.md"
.venv/bin/python scripts/280_def_survival.py extract --prepared "$RUN/prepared" --output "$RUN/activations" --resume   # GPU
.venv/bin/python scripts/280_def_survival.py evaluate --prepared "$RUN/prepared" \
  --activations "$RUN/activations" --output "$RUN/evaluation"                                          # CPU
.venv/bin/python scripts/280_def_survival.py steer --prepared "$RUN/prepared" \
  --evaluation "$RUN/evaluation" --output "$RUN/steer"                                                 # GPU, needs the stage-201 lens
cat "$RUN/evaluation/report.md" "$RUN/steer/report.md"
```

Run long stages in a screen. For seed sensitivity, rerun every stage with
`--seed 43` and `--seed 44`. Every stage refuses to write into a non-empty
directory; only `extract --resume` continues a run with identical arguments.

Runs in `results/def_survival/` (2026-10-09) and the five-domain
`results/binding_flow/` run (2026-10-12) used earlier versions of this stage.
Those versions had no BOS token, no recency control in the pairs, and an
untuned shallow baseline. Don't mix their numbers with this version's.

## Outputs

| file | content |
|---|---|
| `prepared/coverage.csv`, `exclusions.csv` | sizes, labels, mechanisms and heuristic accuracy per domain × split; exclusions by reason |
| `prepared/candidates.csv`, `programs.jsonl` | every candidate with anchors and covariates; sources, token ids and offsets |
| `evaluation/summary.csv` | every train → test cell at the selected layer: probe, pair accuracy, all baselines, intervals |
| `evaluation/baselines.csv` | the shallow reader's chosen C and which reader is "best shallow" per domain |
| `evaluation/direction_grid.csv`, `cosines.csv`, `directions.npz`, `directions.json` | the directions, their cross-domain accuracy and cosines |
| `evaluation/breakdown.csv`, `layers.csv` | accuracy by covariate bin; per-layer curves |
| `steer/jlens_top10.csv` | J-lens top-10 at each snippet's last token, per direction layer |
| `steer/steering.jsonl` | per snippet × direction × coefficient: next-token top-10, continuation, steered J-lens |
| `steer/promoted.csv`, `direction_lens.csv` | tokens each direction raises and lowers; the bare direction's lens reading |

## Limits

- **Labels:** static may-reach on a supported subset of Python, not dynamic
  traces.
- **Controls:** only the synthetic domain is token-controlled. On CruxEval and
  CodeSearchNet, "beyond shallow" means beyond the best shallow reader, not
  beyond every possible cue.
- **Steering:** it shows what a direction does to the model's output, not that
  the model uses it to decide the binding.
- **J-lens:** it shows what an activation is disposed to make the model say.
  Expect generic code tokens unless the direction happens to be verbalisable.
