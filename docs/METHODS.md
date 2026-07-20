# Methods

This file explains the methodology behind every number in `results/tables/`.
Each section states **what** we do, **why** it is necessary, and **how** it
works, so a reader can judge the claims without reading the code. It is written
to be lifted into the paper's Methods section.

The overall logic: we extract a model's internal hidden states while it reads a
program, and train a simple linear classifier ("probe") to predict a semantic
fact about that program (e.g. "are these two tokens the same variable?"). If a
*simple* probe can recover the fact, the model must already represent it. The
hard part is making sure the probe is reading the *model's computation* and not
some shortcut in the text — most of this document is about closing those
loopholes.

---

## 1. The probes

**What.** Every result uses a single, deliberately weak classifier: **logistic
regression** (`C=0.1`, class-balanced) on standardized features. No neural-net
probes.

**Why weak on purpose.** A high-capacity probe (e.g. an MLP) can *learn* the
semantic relation itself from raw activations. If that happens, "the fact is
decodable" tells you about the probe's power, not the model's representation. A
linear probe can only read features the model has already made linearly
available — so "decodable" stays a statement about the model.

**How the inputs are built.**
- **Single-position tasks** (token type, taint state) probe one token's hidden
  state directly, `h_i`.
- **Pairwise tasks** (binding, def-use, control dependence) ask about a *relation
  between two tokens*, so the probe sees a feature vector that captures both
  tokens and their interaction:
  `[h_i ; h_j ; h_i − h_j ; |h_i − h_j|]` (concatenation, difference, and
  absolute difference).

**Convergence bookkeeping.** Fits use the `saga` solver with `max_iter=2000`,
`tol=1e-3`. Whether each fit actually converged is recorded and appears in every
results row (`converged`). Stage 20 fails its sanity check if any *reported*
fit did not converge, so a non-converged probe can never quietly become a
headline number. (The shuffled-label control fits — Section 5 — routinely hit
the iteration cap by design and are tracked separately as `control_converged`.)

---

## 2. Where in the model we read (layers and positions)

**Layers.** Hidden states are captured at a fixed set of transformer blocks
(spanning input to output) plus one special layer:

- **Layer −1 = the embedding output.** This is the token representation *before
  any attention or context mixing*. It encodes token identity only. We use it as
  the **context-free reference**: anything decodable here is a property of the
  token string, not of the model's reasoning.
- **Layer 0 and up = decoder-block outputs.** Note that "layer 0" is the output
  of the first transformer block, which has *already* mixed context once. This is
  why we extract layer −1 separately — without it, there is no truly context-free
  baseline in the layer sweep.

**Position.** For a task about a source-code event (a variable use, a guard, a
sink argument), we read the hidden state at the event's **last covering token**
— the first position whose state can "see" the whole event under causal
(left-to-right) attention. Reading earlier would miss part of the event; reading
later would fold in unrelated downstream tokens.

---

## 3. Ground truth and token alignment

**What.** The labels the probe is trained against come from **static analysis of
the same program**, not from the model:
- **Def-use chains** from reaching-definition analysis.
- **Control dependence** from the AST (guard nesting, with join points resolved
  exactly so control does not "leak" past where branches merge).
- **Taint state per line**, known because the generator produced the program.

**The alignment problem.** Static analysis speaks in source coordinates ("line
5, column 8"); the model speaks in subword tokens. We have to translate one to
the other exactly, or every label is attached to the wrong hidden state.

**How.** Each source event is located by its **AST span** and mapped to token
indices through a **verified offset table**. Offsets are computed by incremental
prefix decoding — decode the first *n* tokens, see how many characters they
cover, repeat — and the result is checked to **reproduce the source exactly**
before it is used (`src/data/alignment.py`).

**Why not just match token strings.** With a subword vocabulary a variable name
may not be any single token, and string matching silently mislabels *shadowed*
names (two different variables spelled the same) — which is precisely the
phenomenon E2 is trying to measure. String matching would build the shortcut
into the ground truth.

### 3a. Independent cross-check of the ground truth

Because every downstream label depends on our extractor, its def-use edges are
**differentially tested against `beniget`**, a mature, independently written
reaching-definitions analysis (`tests/test_ground_truth_crosscheck.py`). This is
the same "validate the program graph against a second implementation" discipline
that code-property-graph tools (Joern, llvm2cpg) use.

The two analyses answer slightly different questions — ours resolves each use to
the *single most-recent* reaching definition, while beniget returns *all*
possibly-reaching definitions across branches — so the sound comparison is
**set inclusion** (our edges ⊆ beniget's), with **exact equality on
straight-line code**. This check caught a real bug: uses in self-referential
updates like `b = b + a` were being linked to the *same-line* target definition
instead of the prior one. The extractor now resolves reaching definitions in
execution order (right-hand side before the assignment target).

### 3b. Tokenizer integrity

`AutoTokenizer` on transformers 5.x silently resolves deepseek-coder to a slow
SentencePiece path that **mis-tokenizes code** — `def func` becomes
`['de','ff','unc']` with whitespace dropped. Any activations or labels built with
it are garbage. `src/models/loader.py::load_tokenizer` therefore loads via
`PreTrainedTokenizerFast` and **rejects any tokenizer that fails an exact
code round-trip**. All results predating this guard are invalid.

---

## 4. Cross-validation without leakage

**The problem.** Many probe examples come from the *same program* and therefore
share overlapping hidden-state vectors. Ordinary random k-fold cross-validation
would put some rows of a program in training and others in test, letting the
probe memorize program-specific quirks and inflate test accuracy — leakage.

**How we prevent it.** All cross-validation is **`StratifiedGroupKFold` grouped
by source-example id**: every row from one program stays entirely within one
fold. When we cap dataset size for tractability (`max_samples=20000` per
task × layer), we drop **whole groups (programs), never individual rows**, so a
program is either fully in or fully out.

---

## 5. Selectivity control (guarding against "easy" accuracy)

**The problem.** A probe can score high accuracy for boring reasons: class
imbalance, or per-program regularities that correlate with the label. High
accuracy alone is not evidence of a *semantic* representation.

**How.** For every probe we retrain the **identical** classifier on **shuffled
labels** and report:

> `selectivity = accuracy − control_accuracy`

If the real structure matters, the true-label probe should beat the shuffled one;
if the "signal" was just priors and regularities, the shuffled probe matches it
and selectivity ≈ 0. **Claims are made on selectivity, not raw accuracy.**

The shuffle is done carefully:
- For **pairwise/per-token tasks**, labels are shuffled *within* each program, so
  each program's label mix is preserved and only the token→label pairing is
  destroyed.
- For **example-level tasks** where the label is constant across the whole
  program (taint_state), a within-program shuffle would do nothing, so instead
  the **program→label assignment is permuted across programs**.

---

## 6. Negative-sampling strata (the honest headline)

**The problem.** For a relation like binding, most "negative" pairs are trivially
separable from the text alone (two differently-named variables). A probe scoring
well on those is not demonstrating semantic understanding. So we break the
negatives into **strata** and report held-out accuracy for each, from easiest to
hardest:

| Stratum | What it is | What it controls for |
|---|---|---|
| `diff_name` | different variable names (capped at 3× positives) | trivial baseline |
| `distance_matched` | negatives at the same token distance as positives | positional shortcuts |
| `same_name_diff_binding` | same name, different actual binding | the name-identity shortcut |
| **`context_matched`** | **two token-identical programs differing by one binding-flipping character** | **every surface cue at once** |

**Why `context_matched` is the one that matters.** Its two programs are identical
token-for-token except the single character that flips the correct label; the
anchor windows and token distance are identical, and the pair shares one CV
group. By construction, *no* feature of the text can separate the labels — only
something the model computed can. **The honest headline number is
`context_matched` accuracy measured against the surface baseline (Section 7),
not the pooled accuracy across strata.**

---

## 7. The surface-shortcut baseline

**What.** A probe that sees **no hidden states at all** — only the ±3-token
window of token ids around each anchor plus the bucketed distance between them —
fit with the same grouped CV and reported per stratum (`features="surface"`).

**Why it exists.** The first full 1.3b run scored ~0.98 on *every* task and layer,
including the supposedly hard `same_name_diff_binding` stratum at the earliest
layer. This no-model baseline reproduced that ~0.98 — proving the templated
corpus was **leaking labels through local token context**, so the "semantic"
result was a mirage. The baseline is now a permanent floor: **a hidden-state
result only counts if it beats the surface baseline on the same stratum.** By
construction the surface probe scores exactly 0.5 on `context_matched`, which is
why that stratum is the clean one.

---

## 8. Frozen-probe evaluation (E5 context degradation, E9 obfuscation)

**What.** For robustness experiments we take the probes trained in stage 20 and
**evaluate them, unchanged, on transformed programs** — we never retrain per
condition.

**Why not retrain.** Retraining on each condition would measure how *learnable*
the relation is under that condition, which is a different and easier question.
Freezing the probe measures whether the **representation it already found still
holds up** when the input is stressed — which is the actual research question.

**How truth stays correct.** Some transformations genuinely change the program
graph (competing updates in E5; inserted opaque branches and flattened control
flow in E9). Ground truth is therefore **recomputed for every variant**, so the
frozen probe is always scored against the transformed program's real labels.

**E9's equivalence guarantee.** "Same semantics" is never assumed. Every
obfuscated variant is **executed and checked to be observationally equivalent**
to its base program (the same I/O-equivalence standard Tigress uses). All levels
of a given base program are kept or dropped together, so per-level comparisons
always hold the set of base programs fixed.

---

## 9. Calibration and signal independence (E6 lead time)

E6 compares *when the model's internal taint state goes wrong* against *when its
behavior goes wrong*, to ask whether the latent failure comes first.

- **Threshold calibration.** The taint probe's decision threshold is chosen on a
  held-out calibration split (the cutoff that maximizes balanced accuracy) and
  **fixed before any test example is seen** — so the threshold cannot be tuned to
  manufacture a lead time.
- **Independent signals.** The **latent** signal (the probe's linear readout) and
  the **behavioral** signal (the model's own forced-choice log-probabilities)
  come from different mechanisms, and the latent signal is **never derived from
  the behavioral one**. This independence is what makes a lead time meaningful
  rather than circular.

---

## 10. Causal claims (E7 activation patching)

Probes show a fact is *present*; they cannot show it is *used*. E7 tests use by
**activation patching**: run the model on one program, swap in the hidden state
from a minimally different program at a chosen (layer, position), and measure how
much the output flips.

- **Minimal pairs** are verified **token-length-matched with the only difference
  confined to the sink argument**, so a patched position in one program
  corresponds exactly to the same position in the other — no misalignment.
- **Recovery** is the fraction of the output logit-difference restored by the
  patch, reported per (layer, position).
- The **last-token position** is quarantined as the trivial case (patching the
  final position can force the answer mechanically), so it is never counted as
  evidence of an internal mechanism.
- A relation counts as **"used"** when recovery of the answer logit-diff exceeds
  0.5 at a **non-readout** position *while the frozen probe still decodes the
  relation on both sides* — i.e. the information is both present and causal.

---

## 11. Reproducibility

- **Seed 42 everywhere** by default (generator, CV splits, subsampling,
  bootstrap).
- **Every stage writes a manifest** (`results/manifests/`) recording the git
  SHA, the arguments, and wall-clock time.
- **All figures and tables regenerate from the tidy CSVs alone** (stage 90), so
  the entire chain from raw data to published figure is auditable end to end.
