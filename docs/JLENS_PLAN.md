# J-Lens Integration Plan (proposed — not yet implemented)

**Status: planning document only. No code has been changed.** This is a
design spec for a new experiment track, written so it can be picked up and
built incrementally, the same way `CLAUDE.md`'s restructure section tracks
in-progress work. Once (if) implemented, its content is meant to be
distributed into `docs/EXPERIMENTS.md` (new `E10` entries), `docs/METHODS.md`
(new `§12`), `docs/PIPELINE.md` (new stages 60–63), and `docs/RESULTS.md` (a
new status row) — it lives here standalone until that happens.

## TL;DR verdict

**Feasible, and unusually well-suited to this project specifically.** The
source technique — the "J-lens" / verbalizable-workspace method from
Gurnee, Lindsey et al. (Anthropic, July 2026),
[*Verbalizable Representations Form a Global Workspace in Language
Models*](https://transformer-circuits.pub/2026/workspace/index.html) — needs
three things this repo already has cleanly factored out: a raw
autograd-capable HF model object, exact token-to-source alignment, and an
existing hook/patching abstraction to extend. The one part of the original
method that looked like it might be a compute blocker (scoring against the
full ~32k-token vocabulary) turns out to be a non-issue here: the synthetic
generator draws every identifier from a **fixed 26-letter pool**
(`SAFE_NAMES` in `src/data/generator.py:122`), so the code-domain analogue of
the paper's "France" / "spider" concept vectors is a **closed, ~30-token
vocabulary**, not an open 32k one. That turns an expensive, corpus-scale
Jacobian estimation into something that fits inside the compute budget this
project already spends on E7 (activation patching, an 8h SGE job).

The rest of this document works out exactly what that integration looks
like, reusing as much of the existing pipeline as possible and adding the
minimum new surface area.

---

## 1. What the source method actually does

(Summarized from the paper; see it directly for full detail.)

The **J-lens** operationalizes "verbalizability" — representations poised to
be reported or flexibly reused — as a **causal, corpus-averaged linear
readout**:

```
J_ℓ = E_{prompt, t, t'} [ ∂h_final,t' / ∂h_ℓ,t ]
lens(h_ℓ) = softmax( W_U · norm( J_ℓ h_ℓ ) )
```

- `J_ℓ` is the average *first-order causal effect* of a layer-`ℓ` activation
  on the final-layer state, averaged over source position `t`, later
  positions `t'`, and a corpus of prompts. It is a genuine gradient-based
  (VJP/JVP) quantity, not a fitted regression — that's what distinguishes it
  from the **tuned lens** (correlational) and from the plain **logit lens**
  (`softmax(W_U · h_ℓ)`, no correction for the layer-to-layer map at all).
- Applying `W_U` (unembedding) after `J_ℓ` turns any hidden state into a
  ranked list over vocabulary tokens — "what the model is, on average,
  disposed to eventually say" if this activation were surfaced.
- **J-space** is defined as the sparse-nonnegative span of these per-token
  "J-lens vectors" — an interpretable, low-capacity (~6–10% of activation
  variance) subspace that the paper shows is where flexible reasoning,
  verbal report, and cross-task reuse of a concept all live; "automatic"
  processing (text continuation, anomaly detection) largely happens outside
  it.
- The causal claim is tested by **coordinate-space patching**: swap only the
  J-space component of a hidden state (`h + V(σ(c) − c)`), leaving the
  orthogonal complement untouched, and show the swap moves what the model
  says.

## 2. Why this maps cleanly onto semantic-flow

| J-lens paper needs | This project already has |
|---|---|
| Raw model with autograd, access to `W_U` and final norm | `ModelLoader.model` (`src/models/loader.py:113-117`) returns a plain `AutoModelForCausalLM`, `.eval()`-only, gradients not disabled — a `requires_grad_` + backward pass works unmodified. `W_U`/final-norm accessors don't exist yet but are one-line HF calls (§6). |
| A concept that recurs across many contexts, so a "readout direction" is well-defined | Natural-language concepts ("France") ↔ code identifiers here. Because `SAFE_NAMES` (`generator.py:122`) is a **fixed 26-letter pool** reused across almost every generated program, the token `"x"` (or `"counter"`, etc.) has the *same stable vocabulary identity* everywhere it appears — exactly the property that makes a corpus-averaged lens vector meaningful. |
| Token-position alignment to build the corpus of `(t, t')` pairs | `TokenAligner.align_var_event` (`src/data/alignment.py`) already maps every def/use/guard/sink event to an exact token index — this *is* the position-sampling machinery the J-lens needs, already built for the probes. |
| A hook abstraction to extend for coordinate-space patching | `src/models/hooks.py`'s `HookManager`/`_get_decoder_layers`/`patch_positions` (whole-vector swap under `@torch.no_grad()`) is the direct template — coordinate patching is a new sibling function using the same layer-discovery + forward-hook pattern, not a modification of the existing one. |
| A frozen-artifact contract for expensive-to-build, reusable readouts | Stage 20's probe-freeze → stage 30/31 frozen-evaluation pattern (`METHODS.md §8`) is exactly the shape a frozen J-lens dictionary needs: build once (GPU), evaluate many times (CPU) on context/obfuscation variants. |
| A "does the model do this, or is it a shortcut" honesty culture | The project already has the *exact* right control for this: `context_matched` binding pairs (`METHODS.md §6-7`) — two token-identical programs differing by one binding-flipping character. J-lens readouts on that pair are the cleanest possible test of whether "verbalizability" tracks the *actual binding*, not surface text. |

Net: this is not a bolt-on integration in name only — the paper's core
methodological ingredients (position alignment, frozen/reusable readouts,
hard-negative controls) are ones this project independently built for its
own rigor reasons, and they transfer almost unchanged.

## 3. New research question and experiment

Add to the RQ table in `CLAUDE.md`:

| RQ | Question | Experiment |
|----|----------|-----------|
| RQ6 | Do the semantic relations shown to be linearly decodable (RQ1) also belong to a small, causally-privileged, *verbalizable* subspace — and does membership in it predict where causal use (RQ5/E7) and robustness (RQ3/E5/E9) actually hold? | E10 |

RQ6 is deliberately framed as **extending RQ5**, not duplicating it: E7
already asks "is this relation causally used?" via binary activation
patching at hand-picked positions. The J-lens gives a second, independent,
*continuous* and *unsupervised* (no probe training) operationalization of
the same question, and — because it's built from the model's own output
head rather than fit against static-analysis labels — it can diverge from
what the probes find, which is itself informative (e.g., "present per the
probe, but never verbalizable" vs. "probe finds it late, but it's
verbalizable early").

Sub-experiments, mirroring the E1–E9 house style (hypothesis → method →
control → output):

| ID | Name | Mirrors | One-line description |
|----|------|---------|----|
| E10a | J-lens construction & validation | E1 (sanity baseline) | Build and sanity-check the per-layer J-lens dictionary; must recover trivial cases (e.g. copy tasks) before anything else is trusted. |
| E10b | Workspace membership of E2–E4 relations | E2/E3/E4 | At a def/guard/sink anchor, does the J-lens readout put elevated mass on the *correct* downstream identifier vs. distractors — reusing the exact same negative strata (`same_name_diff_binding`, `diff_name`, `distance_matched`, `context_matched`). |
| E10c | Coordinate-space causal patching | E7 | Swap only the J-space component of a hidden state (not the whole vector) and measure answer movement — a more surgical version of E7's patch. |
| E10d | Workspace robustness | E5 / E9 | Frozen E10a dictionary evaluated (never rebuilt) on the existing context-degradation and obfuscation variant sets. |
| E10e *(stretch)* | Sparse J-space subspace analysis | — | The paper's sparse-nonnegative-decomposition definition of "J-space" as a subspace, and its variance-capacity claim (6–10%). Not required for the headline result; flagged as optional deeper analysis. |

## 4. Mathematical adaptation for code

Three design choices need to be made explicitly to port the definition;
each is resolved below by reusing an existing project asset rather than
inventing a new one.

### 4.1 What corpus / position pairs `(prompt, t, t')` to average over

**Decision: reuse the existing anchor positions, not raw token positions.**
Instead of the paper's "all subsequent positions `t'` in the context,"
restrict `t` and `t'` to the def/use/guard/statement/sink anchors already
computed by `TokenAligner` for `core.jsonl` (the same positions E2–E4's
probes are trained on), plus the sequence-final position (the natural
"has this been surfaced by generation time" readout). This:
- ties `J_ℓ` construction directly to positions with known ground truth
  (no fresh corpus needed — reuse `data/synthetic/core.jsonl`, ~500
  examples),
- keeps the number of `(t, t')` pairs per program small (a handful of
  anchors, not every token),
- makes E10a/b directly comparable to E1–E4's stratified structure.

### 4.2 What "vocabulary" to build lens vectors for

**Decision: a bounded task vocabulary, not the full ~32k-token vocabulary.**
Build lens vectors only for:
- the 26 single-letter identifiers in `SAFE_NAMES` (`generator.py:122`) —
  covers the synthetic corpus almost completely (the generator falls back to
  2-char names only once 26 are exhausted in one program, `generator.py:130-138`),
- a small fixed set of structurally relevant tokens: the taint-relevant
  keywords/sink names, `True`/`False`, comparison/arithmetic operator
  tokens used by control-dependence guards.

This is a real, principled scope reduction (not an approximation of the
paper's method) precisely *because* the synthetic corpus's identifier space
is closed by construction — for E8's real-code CodeSearchNet set this
assumption breaks (§8) and the vocabulary must be widened or the analysis
scoped to E8 as a stretch goal.

### 4.3 How to compute `J_ℓ^T w` for each candidate token `w` without
materializing a full `d_model × d_model` matrix

**Decision: batch the cotangents.** For a fixed layer `ℓ` and example, do
one backward pass with a *batch of cotangent vectors* (one row of `W_U` per
candidate token, ~30 rows) propagated through the sub-network from the
final layer back to layer `ℓ`'s output — this is a single VJP call with a
`(V_task, d_model)` cotangent rather than `V_task` separate backward passes
(PyTorch's `torch.func.vjp` + `vmap`, or an equivalent manual batched
`autograd.grad`, both support this). Average the resulting
`(V_task, d_model)` matrix across the sampled `(prompt, t, t')` triples to
get the frozen, reusable `J_ℓ`-derived lens vectors for that layer.

Full mathematical statement of what gets frozen and reused:

```
lens_vectors[ℓ] : (V_task, d_model) matrix, one row per candidate token
                 = average over sampled (prompt, t, t') of
                   VJP( h_final,t'  wrt  h_ℓ,t ,  cotangent = W_U[candidate_tokens] )
readout(h) = softmax( lens_vectors[ℓ] @ norm(h) )   # (V_task,) scores
```

This is precomputed once per `(model, layer)` — exactly parallel to how
stage 20 fits a probe once per `(task, layer)` and freezes it.

## 5. Reuse inventory — nothing here needs to change

| Existing asset | Role in E10 |
|---|---|
| `data/synthetic/core.jsonl`, `context.jsonl`, `obfuscation.jsonl`, `minimal_pairs.jsonl` | Corpus for J-lens construction (E10a) and all downstream sub-experiments (E10b–d) — no new data generation needed. |
| `src/models/loader.py` (`ModelConfig`, `ModelLoader`) | Unchanged. Same model/tokenizer loading path as every GPU stage. |
| `src/data/alignment.py` (`TokenAligner`, `align_var_event`) | Unchanged. Supplies every `(t, t')` anchor pair and every E10b probe position. |
| `src/models/hooks.py` (`HookManager`, `_get_decoder_layers`) | The layer-discovery logic (`_get_decoder_layers`) and the forward-hook registration pattern are the template for the new gradient-enabled capture hook and the coordinate-patch hook — reused as a pattern, not edited. |
| `src/experiments/context_degradation.py::load_frozen_probes` | Direct template for a new `load_frozen_lens` — same "load pickled artifact indexed by layer" contract. |
| `src/utils.write_manifest` | Unchanged; every new stage writes a manifest exactly like stages 00–90 do. |
| `src/analysis/tables.py` / `visualization.py`, `scripts/90_make_paper_assets.py` | Unchanged; add new table/figure functions following the existing "read only from `results/tables/*.csv`" contract. |
| Negative strata (`same_name_diff_binding`, `diff_name`, `distance_matched`, `context_matched`) | Reused verbatim as the E10b evaluation strata — no new negative-sampling logic. |
| `docs/EXPERIMENTS.md` / `METHODS.md` / `PIPELINE.md` / `RESULTS.md` house style | Followed for eventual doc integration (§7). |

## 6. New components needed

All new — nothing here modifies an existing file, matching how E5/E9 were
added as new modules alongside the frozen-probe contract rather than
changes to stage 20.

| New file | Purpose | Sketch |
|---|---|---|
| `src/models/lens.py` | Core J-lens machinery. | `get_output_unembedding(model)` / `get_final_norm(model)` — small per-architecture-family accessors (Llama-family: `model.lm_head`, `model.model.norm`; needs a try/except-by-attr-name fallback exactly like `HookManager._get_decoder_layers` already does for decoder layers, since starcoder2's final-norm naming isn't repo-verified, see §8). `compute_lens_vectors(model, tokenizer, examples, layer, candidate_token_ids) -> np.ndarray (V_task, d_model)` — batched-VJP estimator per §4.3. `apply_lens(h, lens_vectors) -> np.ndarray (V_task,)` — softmax readout. `save_lens/load_lens` — pickle, mirrors `LinearProbe.save/load`. |
| `src/models/lens_patch.py` (or extend `hooks.py`'s sibling module) | Coordinate-space patching. | `patch_subspace(model, input_ids, layer, position, lens_vectors, source_coef, target_coef) -> logits` — new forward hook computing `hidden[:, pos, :] += V @ (target_coef - source_coef)`, gradient-free at *application* time (only construction needs grad) — same hook-registration pattern as `patch_positions`, new function. |
| `src/experiments/jlens_membership.py` | E10b. | Reuses `build_binding_records`/`build_defuse_records`/`build_control_dep_records` (`src/probes/builders.py`) to get the same anchor pairs and strata; for each, scores whether `apply_lens` at the source anchor ranks the *target* token/identifier above distractors. Reports per-stratum top-1/top-k accuracy and score margin — same shape as `static_probes.py`'s per-stratum table. |
| `src/experiments/jlens_patching.py` | E10c. | Parallels `causal_patching.py`: length-matched minimal pairs, sweep (layer, position), but intervention is `patch_subspace` instead of `patch_positions`; same logit-diff-recovery metric and causal-class scheme. |
| `src/experiments/jlens_robustness.py` | E10d. | Parallels `context_degradation.py`/`obfuscation_robustness.py`: frozen `lens_vectors` (never rebuilt) evaluated on `context.jsonl`/`obfuscation.jsonl` variants; ground truth rebuilt per variant, same contract as `METHODS.md §8`. |
| `scripts/60_compute_jlens.py` | New GPU stage building/freezing `lens_vectors` per `(model, layer)`. | CLI shape mirrors `scripts/50_causal_patching.py`: `--model`, `--dataset` (default `core.jsonl`), `--layers`, `--n-samples`, `--candidate-tokens` (default: `SAFE_NAMES` + fixed keyword set), `--output` (default `results/lens/{model}`), writes a manifest. **Needs GPU/autograd — cannot reuse stage 10's `@torch.no_grad()` activation store**, so this is a genuinely new extraction-adjacent stage, not a consumer of existing `.npz` files. |
| `scripts/61_jlens_membership.py` | E10b, CPU. | Consumes `results/lens/{model}` + the existing activation store (stage 10 output) — same cost profile as stage 20. |
| `scripts/62_jlens_patching.py` | E10c, GPU. | Same shape as stage 50; consumes `results/lens/{model}` + `minimal_pairs.jsonl`. |
| `scripts/63_jlens_robustness.py` | E10d, CPU. | Same shape as stage 30/31; consumes frozen lens + context/obfuscation activation stores. |
| `tests/test_lens.py` | Unit tests, following the per-module convention already used (`test_probes.py`, `test_graphs.py`, …). | At minimum: (a) lens-vector shape/determinism given seed, (b) sanity case — a trivial copy-task program where the J-lens at the copied variable's def token should rank the copied identifier highly, (c) `context_matched` divergence test (§8 control), (d) coordinate patch leaves the orthogonal component numerically unchanged. |

## 7. Pipeline integration

Extends the existing stage diagram — new stages run **after** stage 20 (need
frozen taint/binding probes for causal-class labeling, same as E7) and are
independent of 30/31/40/50:

```
00 → 10 → 20 → { 30, 31, 40, 50, 60 } → { 61, 62, 63 } → 90
CPU   GPU   CPU    CPU CPU GPU GPU GPU     CPU  GPU  CPU     CPU
```

- **`configs/experiments.yaml`**: new blocks `stage60_jlens` (candidate
  tokens, n_samples, layers), `stage61_jlens_membership`,
  `stage62_jlens_patching`, `stage63_jlens_robustness` — same style as the
  existing `stage30_context`/`stage31_obfuscation` blocks.
- **`Makefile`**: new targets `jlens`, `jlens-membership`, `jlens-patching`,
  `jlens-robustness`, following the `extract`/`probes`/`context` pattern
  exactly (`$(PY) scripts/6N_*.py --model $(MODEL) ...`).
- **`jobs/`**: `jobs/compute_jlens.csh`, `jobs/jlens_patching.csh` — copy
  `jobs/patching.csh`'s SGE header (`h_rt=08:00:00`, 1 GPU, `source
  jobs/common.csh`) — see §9 for whether 8h is enough.
- **`make smoke`**: extend with a tiny stage-60→63 pass (2–3 layers, ~5
  candidate tokens, ~10 examples) exactly like the existing smoke target's
  scale-down of every other stage, so the whole new track gets an
  end-to-end correctness check before any real compute is spent.
- **Docs** (once implemented, not now): `EXPERIMENTS.md` gets an `## E10`
  block per §3's table; `METHODS.md` gets a `## 12. The J-lens (E10)`
  section explaining the VJP construction, the bounded-vocabulary
  adaptation, and the coordinate-patching intervention, written at the same
  "what/why/how" level as the existing `§10` causal-patching section;
  `RESULTS.md` gets a new row in the status table.

## 8. Controls and validity checks

The project's existing discipline (surface baseline, selectivity control,
`context_matched` hard negatives, frozen-probe evaluation) has direct
analogues here — carrying them over is what keeps this an honest addition
rather than a second pipeline with looser standards:

| Control | Purpose | How |
|---|---|---|
| **Logit-lens floor** | Is `J_ℓ`'s correction over the plain unembedding doing anything? | Compute `softmax(W_U · norm(h_ℓ))` (no `J_ℓ`) as a zero-cost baseline row for every E10b/d result — directly parallels the paper's own comparison (§4 of the fetched summary) and costs nothing extra since it needs no backward pass. |
| **Random-direction floor** | Are the lens vectors doing anything beyond "some direction correlates with rank"? | Replace `lens_vectors` with random vectors of matched per-row norm; top-1/top-k accuracy should collapse to chance. |
| **`context_matched` divergence test** | The cleanest causal test available: two token-identical programs, one binding-flipping character, label flips. | The J-lens readout at the *same* anchor position across the pair must favor the *different* correct downstream identifier despite identical preceding text — if it doesn't, "verbalizability" here is riding surface form, exactly the failure mode `METHODS.md §7` already documents for the surface-shortcut baseline. This is the E10 equivalent of E2's headline stratum. |
| **Shuffled-corpus control** | Guards against `J_ℓ` accidentally encoding position/index regularities rather than a real content-dependent map. | Rebuild `J_ℓ` from `(t, t')` pairs sampled across *unrelated* programs (mismatched prompt for `t` vs. `t'`) and confirm ranking accuracy drops to the random-direction floor. |
| **E10a sanity gate** | Mirrors E1's role as "pipeline bug vs. finding." | A trivial copy/echo synthetic case (e.g., a program that assigns then immediately prints the same variable) must show near-ceiling top-1 lens accuracy before any E10b/c/d number is trusted — same "must-pass-or-it's-a-bug" logic as E1 in `EXPERIMENTS.md`. |
| **Coordinate-patch orthogonality check** | The whole point of coordinate patching over whole-vector patching is that it leaves the rest of the state alone. | Unit test (`tests/test_lens.py`) verifying `patch_subspace` output differs from the unpatched hidden state only within the `lens_vectors` span, to machine precision. |

## 9. Compute cost and staged rollout

**Why this is tractable here specifically:** with `V_task ≈ 30` (§4.2) and
anchor-restricted `(t, t')` sampling (§4.1) against `N ≈ 300–500` examples
reused from `core.jsonl`, across `n_layers ≈ 6–10` probe layers, the total
work is on the order of `V_task × N × n_layers ≈ 30 × 400 × 8 ≈ 10⁵`
batched-VJP-equivalent backward passes for the 1.3b dev model — the same
order of magnitude as the existing E7 causal-patching stage, which already
runs as an 8-hour SGE job. Scaling to 6.7b (`d_model` 2048→4096, `n_layers`
24→32) is roughly a 3–5× slowdown per pass, which should still fit an 8h
job or a two-job split (e.g., half the layers per submission) without
needing a fundamentally different estimator.

This estimate is a **planning approximation, not a benchmark** — it has not
been measured. If it's wrong, the levers to pull, in order of preference
(cheapest to lose the least fidelity first):
1. Reduce `N` (fewer sampled anchor pairs) before reducing `V_task` — the
   average is what makes this "verbalizable" rather than per-instance.
2. Reduce the number of probed layers (mirror E1–E4's practice of probing a
   sparse layer subset, not every block).
3. Reduce `V_task` further (drop the operator/keyword tokens, keep only
   the 26 identifiers) — this is the least preferred cut since it's exactly
   the part motivating the whole design.
4. Split the SGE job by layer instead of trying to raise `h_rt` past
   cluster policy.

**Suggested rollout, matching the project's existing dev→cluster
discipline** (`configs/experiments.yaml`'s `dev_model`/`main_model` split,
`make smoke` before any real run):

1. **Smoke scale** (local, MPS, 1.3b): 2–3 layers, ~5 candidate tokens,
   ~10 examples — verify E10a produces a sane lens dictionary and the E10a
   copy-task sanity gate passes, before spending any real compute. Wire
   into `make smoke`.
2. **Dev-model full validation** (local or single GPU, 1.3b): full
   `V_task`/`N`/layer set from §9's estimate. This is the checkpoint to
   actually look at `context_matched` divergence and the logit-lens/
   random-direction floors before deciding the method works at all — same
   role the 1.3b "VERDICT POSITIVE" run played for E2/E3 (`CLAUDE.md`
   remaining-item 1).
3. **Cluster 6.7b run**: only after step 2 shows a real, control-passing
   signal — mirrors the project's existing policy of not running the
   expensive main-results model until the dev model has validated the
   method (E2/E3/E4 all followed this order).

## 10. Risks, open questions, and disanalogies

- **Real-code (E8) identifier vocabulary is not closed.** CodeSearchNet
  functions use arbitrary, often multi-token identifier names
  (`load_codesearchnet_sample`, `src/data/dataset.py:118`) — the §4.2
  bounded-vocabulary trick doesn't transfer as-is. Options if E10 is
  extended to E8: widen `V_task` to the most-frequent identifier tokens in
  the sample, or restrict E10 to the synthetic corpus only and flag
  real-code J-lens analysis as a stretch goal alongside E10e.
- **Final-norm/unembedding accessor is architecture-family-specific.**
  Confirmed for the Llama family (deepseek-coder, codellama) as
  `model.lm_head` / `model.model.norm`; **not repo-verified** for
  starcoder2's GPTBigCode-derived trunk. `get_final_norm`/
  `get_output_unembedding` (§6) need the same defensive multi-attribute
  try/except `_get_decoder_layers` already uses, and should raise clearly
  (matching the project's `load_tokenizer` philosophy: fail loud rather
  than silently use the wrong module) rather than guessing.
- **Multi-token identifiers.** A `SAFE_NAMES` fallback name (2-char, only
  when 26 letters are exhausted in one program) may tokenize as >1 token
  under the deepseek-coder tokenizer — `V_task` candidates should be
  restricted to *single-token* strings (verifiable at build time via the
  same round-trip-checked tokenizer from `load_tokenizer`), with multi-token
  names simply excluded from the candidate set rather than handled specially.
- **This is a genuinely new, unvalidated estimator.** Unlike E5/E9 (frozen
  *existing* probes evaluated on new data) or E7 (reuses the *existing*
  patch-position mechanism), E10a is new measurement machinery end to end.
  The controls in §8 exist specifically because this method has no prior
  validation on this codebase or on code models at all — treat the dev-model
  checkpoint in §9 as a real go/no-go gate, not a formality.
- **E10e (sparse J-space subspace / variance-capacity analysis)** is
  explicitly out of scope for an MVP — it requires a sparse-coding/pursuit
  step over the lens-vector dictionary that adds real complexity for a
  claim (the workspace occupies ~6–10% of variance) that isn't needed to
  answer RQ6's core question. Revisit only if E10a–d land a clear positive
  result and the sparse-subspace framing adds something E10b/c don't already
  show.

## 11. Suggested implementation order

1. `src/models/lens.py`: accessors (`get_output_unembedding`,
   `get_final_norm`) + `compute_lens_vectors` + `apply_lens` +
   save/load. Unit-test against the copy-task sanity case (§8) before
   anything else.
2. `scripts/60_compute_jlens.py` + smoke-scale wiring into `make smoke`.
3. `src/experiments/jlens_membership.py` + `scripts/61_jlens_membership.py`
   (E10b) — cheapest sub-experiment to reach a first real result, since it
   needs no new patching code, just the frozen lens + existing builders.
4. Run the dev-model (1.3b) full validation (§9 step 2); evaluate against
   all four controls in §8. **Stop and reassess here if `context_matched`
   divergence doesn't clear the random-direction floor** — that would mean
   the method isn't finding anything beyond what E2's probes already show,
   and E10c/d/cluster-scale work should not proceed until that's understood.
5. `src/models/lens_patch.py` + `jlens_patching.py`/`scripts/62_*.py` (E10c).
6. `jlens_robustness.py`/`scripts/63_*.py` (E10d) — cheapest to add once
   E10a's frozen dictionary exists, since it's pure evaluation against
   already-generated `context.jsonl`/`obfuscation.jsonl`.
7. Cluster 6.7b run, docs integration into `EXPERIMENTS.md`/`METHODS.md`/
   `PIPELINE.md`/`RESULTS.md`, `CLAUDE.md` RQ table update.
