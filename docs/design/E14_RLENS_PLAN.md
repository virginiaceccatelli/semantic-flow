# E14 — R-lens: the shape of the binding decision across layers

**Status: PLANNED. Nothing built.**

> **While the binding is being constructed (layers 0–15), what does the state at
> the use site look like in the model's own output basis — and is the direction
> that reads it the same direction E13 shows is causally load-bearing?**

This plan replaces the J-lens with a **more faithful backward pass** (R-lens,
[LessWrong, 2026](https://www.lesswrong.com/posts/nv8oedrnLXKRzNEL9/r-lens-making-j-lens-more-faithful-on-early-layers))
and points the resulting instrument at E13's factorial.

> **E14 is observational and belongs entirely to the representational track.**
> It reads states; it never intervenes. It therefore *cannot* establish causal
> use, and no part of it is an alternative to E13. For causal use, an
> interchange intervention (E13/DAS) is strictly stronger and is already done —
> H0–H5 pass. What E14 adds is on the other axis: not *whether* the model uses
> the binding, but *what form the binding is in* while it is being built.

---

## Contents

- [§0 What this adds that the probes do not](#0-what-this-adds-that-the-probes-do-not)
- [§1 Why the current instrument cannot answer this](#1-why-the-current-instrument-cannot-answer-this)
- [§2 What R-lens changes, exactly](#2-what-r-lens-changes-exactly)
- [§3 Why it is applicable here](#3-why-it-is-applicable-here)
- [§4 Gate R — validating the instrument](#4-gate-r--validating-the-instrument)
- [§5 E14 — the experiment](#5-e14--the-experiment)
- [§6 What isolates semantics from the confounds](#6-what-isolates-semantics-from-the-confounds)
- [§7 The three shape descriptors](#7-the-three-shape-descriptors)
- [§8 Outcomes, both branches](#8-outcomes-both-branches)
- [§9 Implementation map and cost](#9-implementation-map-and-cost)
- [§10 Risks](#10-risks)
- [§11 Do not claim](#11-do-not-claim)

---

# 0. What this adds that the probes do not

E2/E3/E5/E9 already establish that binding is linearly decodable, with a floor
pinned at 0.500. So the first question to answer is not "why R-lens" but **"why
a lens at all"**.

A probe tells you the information is **present**. It cannot tell you what
**form** it is in, because the probe chooses its own direction.

| | trained probe (E2) | R-lens (E14) |
|---|---|---|
| who picks the direction | the probe's optimizer, against your labels | the model's own `W_U`, pushed back through the network |
| fitted to anything? | yes — task labels | no — a derivative, nothing fitted |
| units of the answer | a coordinate in a learned basis | a token in the model's vocabulary |
| what a positive result means | the relation is *decodable* here | the state here is *disposed toward* one value over the other |
| resolution per layer | one accuracy | two separate token traces |

Three consequences, all representational:

1. **Presence ≠ format.** Probe accuracy 0.984 at L11 says a linear direction
   exists. It says nothing about whether that direction is one the model's own
   downstream machinery is aligned with. *Decodable* vs. *output-aligned* is a
   claim about the shape of the code, not about intervention — and no probe,
   however accurate, can separate them.
2. **No fitted readout.** Every existing representational claim in the project
   rests on a supervised decoder. An unfitted, model-owned readout that agrees
   with it is independent evidence of a different kind, not a re-run.
3. **Shape needs two traces.** A probe returns one accuracy per layer, which
   cannot distinguish "the correct value rising" from "the wrong value being
   suppressed" (§7.2). A vocabulary readout returns both.

### 0.1 Why R-lens rather than the alternatives

| readout | unfitted? | usable in layers 0–15? | |
|---|:--:|:--:|---|
| logit lens | ✓ | ✗ | 0.000 top-1 at layers −1/0 in your own run — no layer-to-layer correction at all |
| J-lens | ✓ | ✗ | the trough at L7/L11 (§1) — a *biased* backward pass, so more samples cannot fix it |
| tuned lens | ✗ | ✓ | fitted to reproduce the model's final distribution, so it can manufacture the alignment it is measuring |
| **R-lens** | ✓ | ✓ | the only readout that is both |

That is the entire case. If the target were layers 19+, plain J-lens would do
and this plan would be unnecessary. The target is the construction window.

---

# 1. Why the current instrument cannot answer this

E10-0 established that the J-lens implementation is *correct* (V1 cosine
1.0000) and that the Jacobian correction is *live* (+0.18 top-1 over the logit
lens pre-final). What it did not establish is that the readout is **faithful at
every depth**. The run's own numbers say it is not — `results/tables/jlens_validation_deepseek-coder-6.7b.csv`,
V2 next-token top-1:

| layer | −1 | 0 | 3 | 7 | 11 | 15 | 19 | 23 | 27 |
|---|---|---|---|---|---|---|---|---|---|
| J-lens | 0.033 | 0.033 | 0.283 | **0.167** | **0.117** | 0.300 | 0.567 | 0.633 | 0.650 |
| logit  | 0.000 | 0.000 | 0.150 | 0.300 | 0.183 | 0.183 | 0.383 | 0.550 | 0.633 |

The J-lens curve is **non-monotonic** and near-blind through layer 11 — at
layers 7 and 11 it is *worse than at layer 3*, and worse than the plain logit
lens. Nothing about the model justifies that shape; a deeper state contains
strictly more about the next token than a shallower one. It is the error
accumulation the R-lens post describes, measured on this model.

Now overlay E2 (`docs/RESULTS.md`), `context_matched` binding accuracy:

| layer | emb (−1) | block 0 | 3 | peak (11–15) | last (31) |
|---|---|---|---|---|---|
| probe | 0.500 | 0.531 | 0.914 | **0.984** | 0.914 |

**The two windows are complements.** The binding is built in layers 0–15 and
partly shed after; the J-lens is trustworthy from layer 19 up. Every question
about *how the representation is shaped as it forms* falls in the region where
the current instrument is noise. That is the gap, and it is the whole reason to
do this.

---

# 2. What R-lens changes, exactly

R-lens is J-lens with three
[LRP](https://proceedings.mlr.press/v235/achtibat24a.html) rules installed in
the backward pass. **The forward pass is untouched** — every rule is written so
the returned value is algebraically identical and only the local derivative
changes. That property is load-bearing here (§4, R0): it means R-lens reads the
*same* model, and the same hidden states, as stages 10/20/103.

deepseek-coder is Llama-architecture; on `transformers 5.12.1` the two target
modules are exactly:

```python
# LlamaMLP.forward
down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

# LlamaRMSNorm.forward
variance = hidden_states.pow(2).mean(-1, keepdim=True)
hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
return self.weight * hidden_states.to(input_dtype)
```

### 2.1 LN-rule — detach the normalization denominator

```python
scale = torch.rsqrt(variance + self.variance_epsilon).detach()   # ← the rule
return self.weight * (h * scale).to(input_dtype)
```

*Why it is the one that matters most.* RMSNorm's true Jacobian is
`(1/rms)·(I − h hᵀ/(d·rms²))·diag(g)`. The second term **subtracts the component
along `h` itself** — the direction the residual stream actually carries. Applied
once it is a mild shrink; composed over 20–30 blocks it is the "relevance
collapse" the post names, and it is the most likely cause of the layer-7/11
trough above. Detaching the denominator makes the norm a plain diagonal map and
removes the cancellation.

### 2.2 identity-rule — detach the nonlinear factor of SiLU

```python
g = self.gate_proj(x)
a = g * torch.sigmoid(g).detach()      # value == silu(g), exactly
```

`silu(g) = g·σ(g)`, so detaching `σ(g)` preserves the value bit-for-bit and
replaces the true derivative `σ(g)(1 + g(1−σ(g)))` — which is negative for
`g < −1.28` and grows without bound — with a bounded, positive, per-element
scalar. Fail loudly if `act_fn` is not SiLU rather than guessing a ratio form
(the `get_output_unembedding` philosophy).

### 2.3 half-rule — split relevance evenly across the gate

```python
b = self.up_proj(x)
prod = 0.5 * (a * b.detach()) + 0.5 * (a.detach() * b)   # value == a * b, exactly
return self.down_proj(prod)
```

Under ordinary autograd a product double-counts: `⟨∂(ab)/∂a, a⟩ + ⟨∂(ab)/∂b, b⟩
= 2ab`. The half-rule makes it `ab`. This is what makes conservation testable
(§4, R2).

### 2.4 What is left alone

Linear layers (the LRP 0-rule *is* autograd's gradient there), attention, and
q/k norms — following the post. Attention therefore remains the **only**
non-conserving element in the downstream path, which is not a defect of the plan
but the thing R2 measures.

### 2.5 How they are installed

A context manager that rebinds `forward` on the *instances* and restores in a
`finally`, with modules found by duck-typing (`gate_proj`/`up_proj`/`down_proj`/
`act_fn`; `weight` + `variance_epsilon`), name-excluding `q_norm`/`k_norm`. No
class-level monkeypatching — a leaked patch would silently corrupt every later
stage in the same process. Per-rule flags (`ln`, `identity`, `half`) exist for
the R2 ablation.

Cost: three `.detach()` calls. Negligible, as the post states.

---

# 3. Why it is applicable here

| R-lens needs | This project has |
|---|---|
| A gradient-based output-aligned readout to improve on | `src/models/lens.py` — VJP path, fp16 retry ladder, frozen artifact, `logit_lens`/`random_lens`/`gram_matched_random_lens` controls, all validated |
| Llama-family RMSNorm + gated SiLU MLP | deepseek-coder 1.3b/6.7b, confirmed: `W_U (32256, 4096)`, `LlamaRMSNorm` |
| A concept with a stable vocabulary identity | single-digit values, and `SAFE_NAMES` (26/26 single tokens, Phase 0.1) |
| Something specific to *ask* in early layers | E2's construction window, currently described only by a probe |
| A floor no surface feature can beat | `context_matched` pairs — exactly 0.500 by construction |

The last row is what makes this stronger than the post's own evaluation. R-lens
was validated with `pass@10` on intermediate concepts — a judgement call about
whether a plausible token appeared. Here the floor is pinned by construction,
and §6's arm cross refutes the readout that E11 died on.

**One structural fit worth stating plainly.** E13's answer *is* the bound value,
and it is a single token. An output-aligned vocabulary readout is therefore the
natural instrument for E13's data — no auxiliary decoder, no fitted probe, no
choice of basis. The lens asks the model's own output head what the state at the
use site is disposed to say.

---

# 4. Gate R — validating the instrument

Stage 110. Three checks; 111 refuses to run unless R0 and R2 pass.

### R0 — forward invariance (required, exact)

With rules installed, logits must equal the unmodified model's. Required:
`max|Δlogit| < 1e-4` in fp32 over ≥5 programs. This is what licenses reusing
every other stage's hidden states, and it fails loudly if a rule was written
with the wrong algebra.

### R1 — last-layer identity preserved (required, regression guard only)

At the last decoder layer `J` is the identity, so R-lens must still equal the
logit lens (cosine ≥ 0.99). **Be honest about what this does not test:** at
`layer == last_layer` the code differentiates a tensor against itself and no
decoder module is traversed, so the LRP path is never exercised. R1 catches a
broken install, not a wrong rule.

### R2 — conservation, per layer (required; this is the real gate)

With the rules installed and **no biases in any projection** (Llama sets
`attention_bias=False`, `mlp_bias=False` — the script asserts this rather than
assuming it), every downstream module is degree-1 homogeneous in its input.
Therefore for the scalar score `s = (g⊙W_U[w])·h_final,t'`:

```
ρ_ℓ  =  Σ_t ⟨ ∇_{h_ℓ,t} s , h_ℓ,t ⟩  /  s        →  1
```

exactly, up to (a) attention's softmax and AV product, deliberately unmodified,
and (b) RMSNorm's ε. Report `ρ_ℓ` for **both** raw autograd (J-lens) and the LRP
backward (R-lens), at every probe layer. Requires no labels, no candidate set,
and one extra backward per (example, layer).

- **Required:** R-lens `|ρ−1| <` J-lens `|ρ−1|` at every layer, and R-lens
  median `|ρ−1| < 0.1` over layers 0–15.
- **Reported:** the residual R-lens gap *is* attention's contribution to
  unfaithfulness, isolated. If it is large, the optional AttnLRP extension
  (freeze the softmax pattern; half-rule on the AV product) becomes worth
  building — as an ablation arm, not a baseline.

R2 is the check the J-lens never had. V1 validated the *plumbing* against a
closed form at one layer; R2 validates the *faithfulness* at every layer, in
closed form, and quantifies on this model the error accumulation the post
asserts.

### R2b — rule ablation (reported)

`ρ_ℓ` with each rule individually disabled. **Prediction stated in advance:** the
LN-rule dominates and its effect grows with the number of traversed blocks
(§2.1); identity and half matter less and roughly uniformly. If that ordering
does not hold, the mechanism story in §2.1 is wrong and should be corrected
before §7's interpretation is trusted.

---

# 5. E14 — the experiment

**Data.** E13's factorial, unchanged — 2 arms (`ab`, `ba`) × 2 bindings
(inner/outer) × ~400 bases, with `record.positions` already carrying named
anchors (`pre_def`, `def_source`, `def_target`, `mutation`, `use`, `answer`) and
`record.token_ids` the two value tokens. No new data generation.

**Candidate vocabulary.** The ten single-digit tokens. `V_task = 10` — two rows
carry the comparison, the other eight give a scale-invariant denominator (§5.1).

**Lens build corpus.** `data/synthetic/core.jsonl` programs — **not** the binding
pairs. A lens fitted without ever seeing the factorial, then applied to it,
removes any circularity between the readout and the thing being read. ~100
programs, anchor-restricted `(t, t')` sampling as in stage 60.

**Measurement.** At each probe layer ℓ and anchor, score the two value tokens
and record:

- **sign accuracy** — is `s(bound value) > s(other value)`? Floor exactly
  0.500. Primary statistic; rank-based, so the module's scale caveat does not
  bite.
- **normalized margin** — `(s_bound − s_other) / ‖s − mean(s)‖₂` over the ten
  digits. Numerator and denominator both carry the unknown positive factor, so
  it is scale-invariant and comparable across layers. Secondary; for shape only.

**Arms are reported separately, never pooled** (§6).

**Readout arms:** R-lens · J-lens · logit lens · `gram_matched_random_lens`.

**Anchors:** `use` (the site), `pre_def` (before the binding exists), `answer`
(where the model demonstrably produces the value — E13's H1 already established
this behaviourally).

---

# 6. What isolates semantics from the confounds

Four nested controls. The second is the one that makes this a claim about
*semantics* rather than about an answer direction — the failure that retired
E11.

**1. Floor pinned by construction.** Within one arm, the two programs are
token-identical except one character; both values appear in both. Position,
distance, and every neighbouring token are identical. A surface predictor is
right exactly 0.500 of the time, by construction, not by estimate.

**2. The arm cross.** In arm `ab` the correct binding-flip moves the readout
`v_a → v_b`; in arm `ba` the *same* flip moves it `v_b → v_a`. So:

| what the readout tracks | arm `ab` | arm `ba` | pooled |
|---|---|---|---|
| the binding | > 0.5 | > 0.5 | > 0.5 |
| a fixed answer/value direction | > 0.5 | **< 0.5** | 0.5 |
| position or recency | > 0.5 | **< 0.5** | 0.5 |

A readout must clear 0.500 **in both arms** to count. This imports E13's
falsification structure onto the representational side at zero extra cost, and
it is why per-arm reporting is mandatory.

**3. Readout floors.** `gram_matched_random` (every length and angle matched, so
only the *directions* differ), logit lens (the unembedding alone), J-lens (does
the LRP backward actually buy anything).

**4. Matched-in-kind site controls — the E10-3 lesson.** E10-3 was archived
because its positive control was an *identity* control where the test was
*relational*, so "not verbalizable" and "this readout cannot express relations
here" stayed indistinguishable. E14 fixes that by construction: the same
instrument, the same relational question, and the same candidate set are read at
a site where the answer **must be chance** (`pre_def` — the binding does not yet
exist) and at a site where it **must approach ceiling** (`answer`, late layers).
Both controls are relational. A flat curve at `use` is interpretable only if
`answer` is at ceiling; if both are flat, the instrument — not the model — is
the finding, and the report must say so.

Cluster bootstrap over base programs throughout, controls paired on the same
rows, per `docs/METHODS.md`.

---

# 7. The three shape descriptors

All scale-invariant. Each is one accuracy or one cosine.

### 7.1 Onset — when does *represented* become *output-aligned*?

First layer where per-arm sign accuracy clears 0.500 with a bootstrap CI
excluding it. Plot against two curves the project already owns:

```
decodable            E2 probe            0.531 @ L0 → 0.914 @ L3 → 0.984 @ L11–15
output-aligned       E14 R-lens          ← the new curve
causally transported E13 interchange     ← the site that already passed H0–H5
```

If the R-lens onset coincides with the probe's, the binding is output-aligned
from the moment it exists — one computation, read directly by the output head.
If it lags by several blocks, there are **two stages**: the relation is computed
first and routed into output-aligned coordinates later. Both are real findings
about shape, and neither is currently answerable.

### 7.2 Construction or suppression — what is the state doing?

Track the normalized scores of the bound and the competing value *separately*
across layers. Does the correct value rise from nothing (construction), or does
the wrong one start high and get suppressed (a competition being resolved)? This
is the most direct answer to "what does the representation look like as it
forms", and it is readable **only** with a faithful early-layer lens — under the
J-lens both traces are in the layer-7/11 noise trough.

### 7.3 Direction geometry — is it one object or two?

Let `d_ℓ = v_{v_a}^(ℓ) − v_{v_b}^(ℓ)`, the readout direction separating the two
candidates at layer ℓ.

- **`cos(d_ℓ, d_ℓ')` across layers.** A stable direction means the binding is
  written into a fixed channel and the later blocks amplify it; a rotating one
  means it is re-encoded block by block. Floor: the same cosine under
  `gram_matched_random`.
- **`cos(d_ℓ, w_ℓ)` against E13's rank-1 interchange direction — secondary,
  descriptive.** This measurement adds **no causal evidence**. E13 has already
  established, by intervention, that `w_ℓ` is causally load-bearing; all this
  cosine does is *characterize that already-established direction in
  representational terms* — is the direction the model acts on the same one its
  output head reads? Agreement means the two tracks describe one object;
  near-orthogonality means E13 transports something the output head does not
  read, which is the more interesting result. Neither reading licenses a causal
  claim from E14, and the report must not phrase it as corroboration of E13.
  Floor is mandatory here: random cosine in d=4096 is ≈0.016, so even small
  values look large — compare against the `gram_matched_random` distribution,
  never against zero.

---

# 8. Outcomes, both branches

| | If it works | If it does not |
|---|---|---|
| **R2** | R-lens `ρ→1` at all depths; J-lens diverges with depth → the post's claim replicates on a code model, in closed form, and the instrument is licensed | R-lens no better than J-lens → **stop and report**. Code models may differ; do not run 111. This is a publishable methods note either way |
| **7.1 onset** | R-lens onset ≈ probe onset → binding is output-aligned as soon as it is computed | Onset lags → a two-stage compute-then-route picture, currently invisible to every existing stage |
| **7.2 shape** | A clean construction *or* suppression signature, monotone in depth | Both traces flat at `use` while `answer` is at ceiling → the use site holds the binding in a form the output head does not read; that is the "decodable but not output-aligned" dissociation E10-3 could not license, now with a relational positive control |
| **7.3 geometry** | `cos(d_ℓ, w_ℓ)` above the `gram_random` floor → the direction E13 showed is causal is also the one the output head reads | Near orthogonal → E13's causal direction is not the output-aligned one. Genuinely new, and it constrains how E13's direction should be *described* — not whether it is causal |

Every branch is reportable. The only outcome that stops the track is R2 failing,
and that is by design.

---

# 9. Implementation map and cost

Built (2026-08-14) — the instrument and its gate:

- [x] `src/models/lrp.py` — the three rules, `lrp_rules(model, ln, identity, half)`
      context manager, duck-typed discovery, value-preservation checked
      numerically (`_assert_silu`), `strict=True` refusal when nothing matched.
- [x] `src/models/lens.py` — `compute_lens_vectors(..., lrp=False)` → `kind="rlens"`,
      and `conservation_ratio(..., lrp_flags=...)`. The retry ladder, the vmapped
      VJP, the frozen artifact and all four controls are untouched.
- [x] `src/experiments/rlens_validate.py` — R0 / R1 / R2 / R2b.
- [x] `scripts/110_rlens_validate.py`, `jobs/rlens_validate.csh`, `Makefile`
      target, `configs/experiments.yaml` block, `results/STATUS.yaml` entry.
- [x] `tests/test_lrp.py` — 19 CPU-only tests against a *real* 3-layer Llama
      rather than a stand-in, so the duck-typing is exercised: per-rule value
      preservation, the backward *does* differ, the half-rule halves both branch
      gradients, the identity-rule's factor is exactly `sigmoid(z)`, the LN-rule
      restores the Euler identity, conservation holds only with the rules, and
      the context manager restores on normal exit **and** on exception.

**One deviation from the design.** `conservation_ratio` lives in `lens.py`, not
`lrp.py`: it needs `_layer_module` / `last_layer_index` / the leaf-replacement
hook, and `lens.py` already imports `lrp.py`. Keeping it in `lrp.py` would make
the import circular. `lrp.py` stays pure — rules and installation only.

Not built: `src/experiments/rlens_trajectory.py`, `scripts/111_rlens_trajectory.py`,
`scripts/112_rlens_report.py` — by design, behind stage 110's gate.

**Cost.** `V_task = 10`, ~10 probe layers. Lens build ~100 programs × 10 layers;
eval 400 bases × 4 cells × 3 anchors × 10 layers, one backward each. Comparable
to stage 60, which already ran on 6.7b. Dev on 1.3b first; R2 alone is a few
minutes and settles whether the rest is worth running.

**Run order.** 110 on 1.3b → read R2 → 110 on 6.7b → 111 on 6.7b (the model with
E13's result to connect to) → 112. Hard gate between 110 and 111 in
`STATUS.yaml`, per the repo's refusal convention.

---

# 10. Risks

- **The scale caveat carries over unchanged.** R-lens scores are defined up to a
  positive per-`h` factor. Sign accuracy and the normalized margin are both
  invariant; raw magnitudes across layers are not, and must not be plotted.
- **R-lens is a *designed* attribution, not the model's true gradient.** It is
  deliberately not the derivative. It measures output-alignment; it cannot
  support a causal claim on its own. E13 does the causal work. Stating this
  wrongly is exactly the overclaim `docs/ARCHIVE.md` records for E10.
- **Conservation is internal consistency, not meaning.** `ρ ≈ 1` says the
  backward pass is a coherent decomposition, not that the readout is
  informative. That is why R2 is a gate and §6's floors carry the claim.
- **Attention is untouched.** The residual non-conservation is real and
  measured, not assumed away.
- **fp16 backward stability.** The existing retry ladder covers it, and the LRP
  rules should *help* — detaching the RMSNorm denominator and the SiLU factor
  removes the two largest sources of gradient blow-up. Falsifiable via the
  `n_dropped` / `n_rescaled` counters already in `JLens.metadata`.

---

# 11. Do not claim

- **Nothing causal, at all.** E14 is observational end to end. "Output-aligned"
  means *the output head reads this state as favouring one value* — it does not
  mean the model's computation uses it. Interchange (E13) is what settles that,
  and it is strictly stronger for the purpose. If a sentence in the write-up
  would survive replacing "R-lens shows" with "an intervention shows", it is
  wrong.
- Not that R-lens shows the model "verbalizes" or "is aware of" the binding.
- Not that a high `cos(d_ℓ, w_ℓ)` corroborates E13. E13 stands on its own
  intervention; the cosine only describes the direction E13 already established.
- Not that E14 revives E10-2 or E10-3. Those were retired for reasons about
  their *targets* and *controls*, not their instrument; a better lens does not
  repair either.
- Not a scale claim from 1.3b vs 6.7b unless both arms are fully controlled —
  see the retirement of the E6 scale split.
