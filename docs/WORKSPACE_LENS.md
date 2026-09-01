# E19 — the published J-lens and R-lens on code models

**Status: complete at canonical scale on DeepSeek-Coder 1.3B and 6.7B and
StarCoder2-3B.** Both lenses were fitted on 100 independent prompts, passed the
required gates, and were evaluated observationally and causally. StarCoder2 is
also reported with a paper-minimal sensitivity R-lens that omits the unpublished
LayerNorm analogue.

This document is the technical report the implementation was asked for. It says
what was built, which published choices it follows, which models can carry the
method faithfully, and where a deviation was unavoidable.

---

## 1. What this replaces, and why the old code was renamed

The repository previously contained modules called "J-lens" and "R-lens" that
cited the 2026 workspace paper and the R-lens post. They implement a different
method. The differences are not stylistic:

| | archived method | published method |
|---|---|---|
| what is averaged | `E[J_l^T (g · W_U[w])]` — a vector-Jacobian product against a **fixed candidate vocabulary** | `E[∂h_target/∂h_l]` — the **full `d_model × d_model` Jacobian** |
| normalization | the `1/rms` factor is **dropped**, so scores are defined only up to a per-position positive scale | the model's own final norm is applied: `softmax(W_U · norm(J_l h))` |
| readout | ranks within a hand-chosen candidate list (often two tokens) | rank over the **entire vocabulary**; top-k, pass@k |
| target | the **final** block's residual stream | the **penultimate** block (`n_layers − 2`), per the released artifacts |
| fitting corpus | the task programs themselves (binding pairs, taint programs) | an independent **pretraining-like** corpus (`NeelNanda/pile-10k`) |
| positions | task-chosen `t` and a handful of `t'` | every position after `skip_first`, with cotangents summed over all `t' ≥ t` |
| R-lens rules | LN + identity + half **plus a q/k-detaching "attn-rule"** | LN + identity + half; attention, linear layers and q/k norms **unmodified** |

The candidate-vocabulary restriction is the load-bearing one. Dropping the
normalizer is what forces it — without `norm`, scores are only comparable within
a position, so a top-k over the vocabulary is not defined, and the whole class of
questions the paper asks ("what is this activation poised to say?") cannot be put.

Those modules and their results have therefore been renamed rather than deleted.
Nothing was thrown away; the claims they support are unchanged and still
reproducible. The new names are:

| was | is now |
|---|---|
| `src/models/lens.py`, class `JLens` | `src/models/cotangent_lens.py`, class `CotangentLens` |
| `src/models/lrp.py` | `src/models/cotangent_lrp.py` |
| `src/experiments/jlens_*.py` | `src/experiments/clens_*.py` |
| `src/experiments/rlens_validate.py` | `src/experiments/clrp_validate.py` |
| `scripts/60,61,62_jlens_*.py` | `scripts/60,61,62_clens_*.py` |
| `scripts/110_rlens_validate.py` | `scripts/110_clrp_validate.py` |
| `results/jlens/`, `results/rlens/` | `results/clens/`, `results/clrp/` |
| `make jlens*`, `make rlens*` | `make clens*`, `make clrp*` |

The rename reaches the label strings inside the archived CSVs as well, so a table
from the old method cannot be read as if it came from the new one. The full
existing test suite (682 tests) passes unchanged after the rename.

---

## 2. The implementation

### 2.1 The estimator is the released code, not a reimplementation

`third_party/jacobian-lens` is the reference implementation released with the
paper, vendored verbatim at commit `581d398` (Apache-2.0; `LICENSE` retained,
commit recorded in `third_party/jacobian-lens.COMMIT`). It is installed as the
`jlens` package and `jlens.fitting.fit` does the actual work. Its own 32-test
suite is run by `make lens-smoke`.

This matters for faithfulness in a specific way. The paper's estimator is not
simply "the average Jacobian"; it is a particular reduction —

```
J_l = E_prompt [ mean_p ( sum_{p' >= p} ∂h_target[p'] / ∂h_l[p] ) ]
```

cotangents injected at **every valid target position at once** and backpropagated,
then averaged over source positions. The reference docstring notes that a
per-position estimator gives a *different* `J_l`. Calling the released function
removes the possibility of silently picking the other one.

### 2.2 The R-lens is the same call under different backward rules

`src/workspace_lens/relp.py` implements the three published dense rules:

* **LN-rule** — detach the normalization denominator, making the norm linear.
* **identity-rule** — detach the nonlinear factor of SiLU/GELU:
  `silu(x) = x·σ(x)` with `σ(x)` detached; `gelu(x) = x·Φ(x)` with `Φ(x)` detached.
* **half-rule** — write a gate's product as `½(a·b̄) + ½(ā·b)`, so each branch's
  local derivative is half of the product rule's, instead of the gate
  contributing `2ab` of relevance where it produced `ab`.

Attention, all linear layers, and query/key norms are **unmodified**, as
published. The archived `cotangent_lrp.py` additionally detached q and k; that
rule is *not* carried over, and its absence is the reason `relp.py` makes no
conservation claim about attention.

The rules bind to module *instances* inside a context manager entered around the
whole fit — including the forward pass, since that is the graph every backward
walks — and are removed in a `finally`.

### 2.3 Value preservation is enforced, not asserted

Every rewrite is checked numerically against the module it replaces *before it is
allowed to bind* (`_verify_value_preserving`), and gate W4 re-checks the whole
network end to end, on the logits and on the residual stream at every block. The
identity is algebraic rather than bitwise — the half-rule replaces one fused
multiply with two multiplies and an add, and the norm rules recompute statistics
in float32 — so both checks use a tolerance and both **record the measured
deviation** rather than only a verdict.

The two are held to deliberately different standards, because they can be. The
per-module check (**W5e**) compares one rewrite against the module it replaces
with no accumulation, on a float32 *copy* for normalization layers, so it
resolves to ~1e-7 whatever dtype the fit runs in — a rule half a percent wrong is
refused and never binds. The end-to-end check (**W4**) cannot do that: `x·σ(x)`
and a fused `silu(x)` differ in the last ulp, and at bfloat16 (eps ≈ 8e-3, the
dtype the fits use) that compounds through 24–32 blocks. W4's tolerance therefore
scales as `eps·√depth`, making it a check on *material* change — the right
question for it, since a rule leaking into the network as a whole is an O(1)
effect, not an O(eps) one.

### 2.4 Activation identification is numeric

`transformers` spells SiLU three ways and GELU at least four. `identify_activation`
evaluates the module on a probe grid and matches it against each candidate
factorisation, so a renamed class cannot silently leave the identity-rule
unbound, and a wrong guess cannot silently change the forward pass.

---

## 3. Configuration

Every value below follows the released artifacts
(`camilablank/workspace-lenses`) unless marked otherwise, and every one of them is
written into each lens's `provenance` and its `lens_meta.json` sidecar.

| setting | value | source |
|---|---|---|
| target layer | `n_layers − 2` (penultimate block) | released artifacts |
| source layers | every block below the target | reference default |
| identity anchor | `J[target_layer] = I`, stored | released artifacts |
| `skip_first` | 4 | released artifacts (reference default is 16) |
| `max_seq_len` | 128 tokens | paper |
| fitting corpus | `NeelNanda/pile-10k` | released artifacts |
| `n_prompts` | **100** | paper §9.3: quality saturates, "~100 prompts is usable" |
| readout | `softmax(W_U · norm(J_l h))` using the model's own final norm and LM head, including `final_logit_softcapping` where present | paper / reference `HFLensModel.unembed` |
| storage dtype | fp16 on disk, fp32 in use | reference `JacobianLens.save` |
| compute dtype | bfloat16 | see §5 |

`n_prompts = 100` rather than the released 25 or the paper's 1000: 25 is what the
released *artifacts* used, 1000 is what the paper's own lenses used, and the paper
states quality saturates in between. 100 sits at the saturation point and keeps
the 6.7B fit affordable.

The corpus is a **prefix in dataset order**, not a shuffled sample, and the
resulting file is committed at `data/lens_corpus/pile10k-n100.jsonl` (digest
`483c1e1743d1`). Both choices are about reproducibility. Three different loaders
can supply pile-10k rows depending on what a host has installed — `datasets`, a
parquet read, or the HF datasets-server API — and only document order is
guaranteed identical across all three; a shuffle would make the lens depend on
which loader happened to be available, with a quietly differing digest as the
only symptom. All three paths were checked to return byte-identical rows.
Committing the file means a fitting run needs no network, and `n = 25` is a
strict prefix of `n = 100` by construction.

### 3.1 The corpus is independent, and that is checked

`J_l` is an expectation over prompts, so *which* prompts is part of the lens. A
lens averaged over the binding programs it is then used to read would be fitted to
the structure it is meant to detect, and no readout number would reveal it. The
corpus is therefore pretraining-like web text, disjoint from the probe suite by
construction, and the disjointness is **tested** at build time and again as gate
W1 — by normalised exact match and by any shared 120-character run at any offset.

A second lens fitted on held-out Python from CodeSearchNet is available
(`--corpus code`) as a *sensitivity arm*: every model here is a code model, so
"pretraining-like" arguably means code. That is a real question about the method,
so it is answered as one — the primary lens uses the published corpus, and the
code-corpus lens only measures how much that choice moves the readout. Results
from it are labelled as such and never substituted for the primary.

---

## 4. Compatibility, per model

The published dense recipe is LN-rule + identity-rule + half-rule, with attention
and linear layers untouched. Module structure was inspected directly
(`relp.describe_architecture`, which is also gate W5).

### 4.1 DeepSeek-Coder 1.3B and 6.7B — fully compatible, no deviations

`LlamaForCausalLM`: `LlamaRMSNorm`, `SiLUActivation`, gated SwiGLU
(`gate_proj`/`up_proj`/`down_proj`), **no biases anywhere**, no q/k norms. This is
the exact dense configuration the released artifacts describe, and all three
published rules apply verbatim. These are the primary models.

### 4.2 StarCoder2-3B — supported, with two documented adaptations

`Starcoder2ForCausalLM` differs in three ways that matter:

* **`nn.LayerNorm` instead of RMSNorm.** The published LN-rule is stated for
  RMSNorm. The adaptation detaches the *same quantity* — the normalization
  denominator — and changes nothing else: the mean subtraction is already a
  linear map and needs no rule, and the bias is an additive constant. The
  resulting Jacobian is `diag(g)/s · (I − 11ᵀ/d)`, verified against that closed
  form in `tests/test_workspace_relp.py`. **This is a deviation and is labelled
  as one.**
* **GELU-tanh instead of SiLU.** No deviation: the post names GELU explicitly.
  The factorisation used is exact for `gelu_pytorch_tanh`.
* **A non-gated MLP** (`c_fc → act → c_proj`). The half-rule has **no
  multiplicative gate to split**. This is recorded as `half_rule: "n/a"`, never
  as `"off"` — a report must not be able to read an architecturally absent rule
  as a deliberately disabled one, and a test pins that distinction.

StarCoder2 also carries biases throughout, which breaks the exact degree-1
homogeneity that makes LRP relevance conserve. That is a property of the
architecture, not a rule change, and it is reported rather than corrected.

This is a materially better outcome than the archived method managed on the same
model: `cotangent_lrp.py`'s LN-rule matched RMSNorm only and skipped LayerNorm
entirely, so on StarCoder2 both homogenising rules bound to nothing and stage 140
refuses to run at all. The LayerNorm analogue is what brings StarCoder2 back into
the experiment.

### 4.3 What was *not* done

No released lens artifact could be reused. `camilablank/workspace-lenses` covers
Qwen3.5/3.6, Gemma-3 and DeepSeek-V4-Flash; a `J_l` is a `d_model × d_model` map
in one specific model's residual basis and is meaningless applied to another.
Lenses must be fitted for these models, which is what stage 201 does, following
the released recipe field for field.

---

## 5. Cost, and why bfloat16

Per prompt per lens: one forward pass and `ceil(d_model / dim_batch)` backward
passes. `dim_batch` trades memory for pass count and leaves total FLOPs unchanged
(a test pins this), so the work is about `2 · d_model` forward passes per prompt.

| model | `d_model` | target | backwards/prompt @ `dim_batch=16` | ≈ PFLOP/prompt | fp32 checkpoint | `lens.pt` |
|---|---|---|---|---|---|---|
| DeepSeek-Coder 1.3B | 2048 | L22/24 | 128 | 1.4 | 0.37 GB | 0.18 GB |
| StarCoder2-3B | 3072 | L28/30 | 192 | 4.7 | 1.06 GB | 0.53 GB |
| DeepSeek-Coder 6.7B | 4096 | L30/32 | 256 | 14 | 2.01 GB | 1.01 GB |

At 100 prompts and ~100 TFLOP/s effective, that is roughly 0.4 h, 1.3 h and 3.9 h
per lens; double it for the J/R pair. `make lens-fit-dry` prints this for the
actual configuration without loading any weights.

**Host RAM is the trap, not VRAM.** Jacobians accumulate on the CPU in float32:
the running sum, the per-prompt matrices and the checkpoint write each cost the
checkpoint size, so budget ~8 GB of host RAM at 6.7B. Stage 201 checkpoints every
10 prompts and resumes automatically.

**bfloat16, not float16.** This is a backward pass through up to 30 blocks and
fp16 gradients underflow (a failure this repository has already recorded once, on
MPS). bfloat16 is the checkpoints' native dtype and keeps float32's exponent
range. Gate W4 and check W5f are where a dtype problem would surface.

---

## 6. The evaluation

The question is the paper's, put to a code model: **at a given layer and token
position, which vocabulary tokens is the residual stream poised to be verbalised
as, and do they name the program-semantic intermediate the model needs there?**

That is deliberately *not* the archived experiments' question. Those fixed a
two-word candidate vocabulary in advance and compared margins inside it. With a
full-vocabulary readout the honest metric is a rank against a target the
program's semantics determines, and a distractor the program's *surface* would
produce instead.

Eight families, each pairing a program-semantic intermediate with a surface
distractor (`src/workspace_lens/evalsuite.py`): `binding` (the repository's own
shadowing construction, both arms token-identical at the read position),
`defuse`, `alias`, `call`, `typeof`, `arith`, `loopvar`, and `scopeword` — the
same shadowing programs scored against scope vocabulary rather than values.

Half the items have **targets that appear nowhere in the prompt** (`arith`,
`typeof`, `loopvar`, `scopeword`), so a hit there cannot be attention copying an
input token forward. That subset is plotted separately.

Value programs are read at four predeclared positions in the same prompt: the
variable use, the following token, the call site, and the answer position. The
first three test the value while it is used; the last is the positive control
where it is about to be emitted. This replaces the initial single-position
design.

Reported per (lens, layer, family, read position): top-k tokens, target-concept rank over the
full vocabulary, pass@k for k ∈ {1, 10, 25}, and the earliest layer at which the
concept enters the top k — the quantity the R-lens post claims to improve. Ranks
rather than probabilities, because a lens fitted as an average carries no
calibration. An item that never surfaces its concept is kept distinct from one
that surfaces it late; averaging the first as "the last layer" would turn a
failure into a success.

### 6.1 A tokenizer constraint worth stating

Both DeepSeek-Coder and StarCoder2 segment **every multi-digit number digit by
digit**. The single-token integer pool is exactly 2–9 for both. A
vocabulary-rank readout can therefore only ask a code model about single-digit
values, which is why the suite selects its literals per tokenizer, why `arith`
uses operands whose sum stays single-digit, and why literals are reused across
items rather than being unique to one. Concepts that a tokenizer splits are
dropped rather than scored on their first piece — scoring a fragment would report
the tokenizer's segmentation as a finding about the model — and the count of what
was dropped is printed by stage 200 and stored on the suite.

### 6.2 Causal ablation

A rank is a readout, and the paper's framing is a causal claim. Stage 204 erases
the lens's own read direction `u_w = J_lᵀ(g · W_U[w])` from the residual stream at
the read position and measures the change in the **model's own** answer logit
difference — not the lens's score, which would be a tautology. Four controls
answer distinct objections: the logit-lens direction (`J = I`), a stable-seeded
random projection, a random displacement matched exactly to the J-lens erase
magnitude, and a distractor-token direction constructed separately with the J
and R lenses. Effects use paired 95% cluster-bootstrap intervals over programs,
and the fraction of state norm moved is recorded per example.

---

## 7. The gate

`scripts/202_lens_validate.py` runs seven checks and exits non-zero on a failed
required one. Stages 203–205 are not interpretable until it passes, and stage 205
reproduces the gate table at the top of the report, so a reader never sees a
pass@k table without seeing what certified it.

| | check | required | what it rules out |
|---|---|---|---|
| W1 | corpus independence | yes | a lens fitted to what it is meant to read |
| W2 | matched pair | yes | J and R differing in anything but the backward graph |
| W3 | identity anchor reproduces the model's logits | yes | transport orientation, normalization or unembedding wired wrongly |
| W3b | the readout's `unembed` *is* the model's tail | yes | a readout that is not the model's own head |
| W4 | RelP forward invariance (tolerance scales as `eps·√depth`) | yes | an R-lens that is a lens on a different model |
| W5a–e | rules bind to the right modules and only those; each rewrite value-checked to ~1e-7 | yes | rules that silently bound to nothing, or bound wrongly |
| W5f | the R Jacobian actually differs from the J Jacobian | yes | an "R-lens" that is a J-lens — the quiet failure |
| W6 | disjoint-half fits agree (worst per-layer cosine) | yes | an estimator that has not converged |
| W7 | the papers' own qualitative example reproduces | no | reported, not required — see below |

W6 reports the **worst** per-layer cosine, not the mean: the estimator's variance
is largest at the earliest layers, which is exactly where the R-lens claims its
advantage. It needs the `--halves` lenses from stage 201; without them it reports
**skipped**, not passed.

W7 (the reference implementation's ASCII-face example, where the lens should read
out "nose" at a `^` that is never named) is reported rather than required: these
are code models being asked a natural-language question, so a miss is evidence
about transfer, while a hit is strong evidence the pipeline is right. The ranks
are recorded either way.

---

## 8. Deviations from the published methods

Complete list. Everything not here follows the published choices.

1. **LayerNorm LN-rule on StarCoder2-3B** (§4.2). The published rule is stated
   for RMSNorm, so the default StarCoder2 R-lens is **not** an exact
   implementation of the published dense recipe. The analogue detaches the same
   denominator and leaves the centring and the bias alone; it is labelled in
   provenance as `ln_rule: "layernorm-adaptation"`. Because "how much does the
   adaptation change?" is a measurable question rather than an arguable one,
   `make lens-fit-paperminimal MODEL=starcoder2-3b` fits the arm that IS exact —
   identity-rule only, LN-rule off — into a `-paperminimal` directory, so the
   two can be read side by side. DeepSeek-Coder is unaffected: all three
   published rules apply to it verbatim.
2. **The half-rule is inapplicable to StarCoder2-3B** (§4.2). No gate exists.
   Recorded as `"n/a"`, distinct from `"off"`.
   The paper-minimal StarCoder2 sensitivity fit disables the LayerNorm analogue
   and retains only the exact GELU identity-rule. Its results do not change the
   substantive conclusion.
3. **`n_prompts = 100`** rather than the released 25 or the paper's 1000 (§3),
   at the paper's own stated saturation point. `--n-prompts` reproduces either,
   and 25 is a strict prefix of 100.
4. **BOS is prepended when — and only when — the checkpoint asks for one.**
   `jlens.from_hf(force_bos=True)` sets `tokenizer.add_bos_token`, which the
   reference implementation itself warns "may have no effect for some
   fast-tokenizer configurations". DeepSeek-Coder is one: the first cluster run
   recorded `bos_prepended: False` even though its `tokenizer_config.json` sets
   `add_bos_token: true` and the reference warns raw-text prompts are "degraded
   without an attention-sink BOS", so the adapter prepends the id in the encode
   path and re-measures. StarCoder2 declares no `add_bos_token` at all — its
   `bos_token` is `<|endoftext|>`, a document *separator* — so nothing is added
   there, because prepending a token the model never sees at the start of raw
   text would be a deviation dressed up as fidelity. Provenance carries
   `bos_declared`, `bos_prepended` and `bos_forced`; gate W2 requires the J and R
   lenses to agree on all three. This makes the checkpoint's own declaration
   hold rather than overriding it, but it is a line of our code rather than of
   the release's, so it is listed here.

5. **Forward invariance is algebraic, not bitwise.** The released artifacts say
   "bit-identical"; that holds when the rules are fused into the kernels. Here the
   half-rule reassociates one multiply and the norm rules recompute statistics in
   float32, so both checks use a relative tolerance and record the measured
   deviation. W5e stays exact (~1e-7, on a float32 copy of each norm); W4's bound
   scales with the compute dtype's rounding floor and the model's depth.
6. **Not a deviation, but a difference from this repository's past:** the
   archived q/k-detaching "attn-rule" is not used. Attention is unmodified, as
   published.

---

## 9. Reproducing

```bash
make lens-smoke                                    # toy models + the release's own tests
make lens-check   MODEL=deepseek-coder-6.7b        # can this host run the fit? no weights
make lens-fit-dry MODEL=deepseek-coder-6.7b        # size the run, no weights loaded
make lens        MODEL=deepseek-coder-1.3b LENS_HALVES=--halves
make lens        MODEL=starcoder2-3b
make lens        MODEL=deepseek-coder-6.7b
```

On the GPU host, run `jobs/workspace_lens.csh` in a screen session instead, one
model at a time (`device_map="auto"` will otherwise offload a co-resident model's
tail to meta placeholders). Stage-by-stage commands are in `docs/PIPELINE.md`.

---

## 10. Results

All required gates pass on the three canonical models. The disjoint-half build
check also passes on DeepSeek-Coder 1.3B (worst layer cosine 0.915 for J and
0.977 for R).

The answer-position positive control is perfect: every value family reaches
pass@10 = 1.000 under J, R and logit lenses. At the three earlier positions,
however, the needed program values are essentially absent, including arithmetic
targets that never occur in the prompt. DeepSeek 6.7B has a few isolated J-lens
hits on prompt-present values (at most 0.15), but they do not replicate in R or
the other models. The representation-versus-verbalizability dissociation
therefore survives the expanded positional test.

R often reaches answer concepts earlier than J, but the logit lens is usually
equally early or earlier. Target-absent `loopvar` and `typeof` concepts surface
under all three lenses, generally first under the logit lens: they validate the
evaluation without supplying a J-space-specific result.

Late-layer erasures survive separate distractor controls and exactly
magnitude-matched random edits, but are close to the output head. In the
predeclared L12/L16/L20 sweep, StarCoder2 has no coherent effect that reliably
beats the magnitude-matched control. DeepSeek 6.7B has a small L20 effect (J
minus matched random −0.018, 95% CI [−0.033, −0.002]; R −0.024
[−0.037, −0.008]), but neither transport beats the logit direction there.
DeepSeek 1.3B has a later L20 effect, yet the logit direction is substantially
stronger than J (`J − logit = +0.204` [0.149, 0.270]).

This is a complete, gated negative workspace result on these code models: the
published J-lens does not surface needed program-semantic values while they are
used, and Jacobian transport supplies no consistent causal or earliest-layer
advantage over the logit lens. R provides modest local improvements over J on
DeepSeek, but not the qualitative early-layer recovery reported in the R-lens
post. Exact tables and intervals are in the generated model reports, with the
StarCoder2 paper-minimal sensitivity report stored alongside them.
