# E13 — what a working instrument must be pointed at

**Read after `docs/design/E12_PLAN.md`.** E12 validates an apparatus: it asks
whether a computed, text-absent program value can be identified and
interchanged such that downstream computation transforms it. That is a check on
the measurement, and a passing E12 is a methods section, not a contribution.

The reason is unglamorous: **scalar interchange is already done.** DAS supplies
the formalism and the metric; Othello-GPT and the chess world-model work supply
the "install a state, watch the rules apply" template; Wu, Geiger & Millière
([arXiv:2505.20896](https://arxiv.org/abs/2505.20896)) already run interchange
interventions on variable-binding chains in symbolic programs. A paper whose
result is "we did that too, in Python, on a pretrained model" has no
identification strategy that the field does not have.

What this project owns that the field does not is the **construction-pinned
floor** — a counterfactual where no surface feature is informative *by
construction* rather than by estimate. Combining that with a magnitude-free
interchange is the contribution. E12 builds half of it. This document is about
the other half: which semantic object to point it at.

---

## 1. The bar a successful extension has to clear

A one-scalar store is the simplest possible program state, which is why it is
the right thing to validate on and the wrong thing to publish. An extension
earns a contribution when the object being interchanged is one that:

1. **static analysis considers hard** — the interesting cases are where a
   sound analyser must approximate, because that is where "does the model do
   something analysis-like?" stops being a rhetorical question;
2. **cannot be read off the text** — no token, no AST node, no fixed-offset
   window carries it, so the pinned floor is real rather than nominal;
3. **has a transition law with more structure than `+3`** — a join, a strong
   update, a fixpoint — so that "transformed" is a substantive prediction and
   not arithmetic;
4. **fails informatively** — a null says something specific about the model's
   representation rather than about the instrument.

The four candidates below are ranked on those criteria.

---

## 2. Candidate A — path-sensitive abstract state and control-flow joins

**Object.** The abstract store at a merge point: a lattice element of ℘(ℤ)
rather than a scalar. When the path condition is decidable from earlier text
the state should be a **singleton** (selection); when it is not, it should be a
**set** that soundly contains both feasible branch values and — the part that
makes it a claim — **excludes a provably infeasible one**.

**Why it is the strongest.** It imports static analysis's own evaluation
criteria, soundness and precision, into an interpretability measurement, and it
yields a within-example false-positive rate for a "neural analyser" — a
quantity nobody has measured. It also repairs E4, which was demoted precisely
because control dependence has a 0.927 surface baseline; this asks the
*semantic* control question (what the branch does to the value state) rather
than the syntactic one.

**The trap, and the repair.** The obvious design — mutate a digit in the guard,
`a = 7` vs `a = 2` against `if a > 5` — has **no pinned floor**: the label is a
threshold on one one-hot token at a fixed offset, which a logistic regression
represents exactly. That is E4's defect rebuilt. The repair is small and makes
the claim true: mutate the guard's **operand name** (`if a > 5` vs `if b > 5`)
with the name→value assignment **permuted across examples**, so the label is a
conjunction of two tokens that no fixed-offset linear model can represent.
Build it that way or not at all.

**Second risk.** Transformers linearly represent distributions over hidden
states (Shai et al., [arXiv:2405.15943](https://arxiv.org/abs/2405.15943)), so
finding a two-element mixture at a merge point is closer to the expected result
than to evidence of a lattice join. Only two auxiliary predictions separate
"join" from "uncertainty": exclusion of the infeasible value, and collapse
under a strong update (`x = 5` after the join). Pre-register that failing
either converts the headline from "the model computes a join" to "the model is
uncertain", automatically.

**Interchange target.** The path condition itself: install the guard state from
a decidable run into an undecidable one and require the join to collapse to the
corresponding branch's value — with the cross-operation falsification E12
already implements.

## 3. Candidate B — aliasing and heap state

**Object.** The points-to map ρ: Var → Loc and the induced may-alias relation;
whether a mutation through one name is visible through another.

**Appeal.** Points-to is the canonical hard static-analysis relation, it is not
computable from the AST, and no prior work probes it in LM activations. The
**destruction test** — `r = list(r)` must kill the alias, and a probe that keeps
firing is convicted of recording co-occurrence rather than a relation — is a
genuinely sharp discriminator, especially against a lexical-cue-matched twin
(`r = p; r = list(r); r[0] = 9` vs `r = p; s = list(r); r[0] = 9`).

**Why it is not first.** In a straight-line fragment with explicit copies,
may-alias *is* syntactic def-use — the relation E3 already established as
decodable — with the copy spellings acting as a three-token lexical marker. The
novelty claim does not survive contact with its own data. It becomes real only
with conditional aliasing (`if input(): r = p else: r = q`), a function
boundary, or a container — at which point it inherits Candidate A's
set-vs-uncertainty problem and should be built as the same line of work.

**Second cost.** Alias is a *relation between two positions*, so it wants a pair
probe, and pair probes are this repository's most expensive CPU workload:
16,384-dimensional features, 30.4 h for a stage-20 run at 6.7b, against minutes
for the single-position probes E12 uses. Budget accordingly.

**Third cost.** Behavioural competence. Base code models are not trained to
answer `assert p[0] ==` after an in-place mutation through an alias. That is a
10-GPU-minute check and it should be run before any engineering.

## 4. Candidate C — semantic equivalence

**Object.** Whether two syntactically different programs compute the same
function — an equivalence class rather than a state.

**Appeal.** It is the question the obfuscation ladder (E9) already gestures at,
with execution-verified equivalence already implemented in
`src/data/obfuscation.py`. And E9 left a real unexplained number: frozen probes
survive renaming and opaque predicates but drop to 0.750 under control-flow
flattening.

**Why it is weak as an interchange target.** Equivalence is a property of a
program *pair*, not a state at a position, so there is nothing to install. It
can be probed, and it can be tested behaviourally, but it does not fit the
apparatus E12 validates. Best treated as a **robustness axis over whichever
object is chosen** — run Candidate A's join experiment on flattened and
MBA-encoded variants — rather than as an object in its own right.

## 5. Candidate D — causal localisation of the first incorrect semantic transition

**Object.** Not a new semantic object but a new *use* of E12's machinery: given
a program the model answers wrongly, find the earliest statement at which its
internal state diverges from the true trace, and verify the localisation
causally by interchanging the correct state in at that point and checking the
answer is repaired.

**Appeal, and it is the most practical.** This is the only candidate with a
direct tool story: a semantic failure localiser that says *which statement* a
model's reasoning went wrong at, verified by repair rather than by correlation.
It also converts E12's apparatus from a demonstration into a measurement
instrument — the trichotomy readout becomes a per-statement divergence detector.

**Why it is not first.** It presupposes a working instrument on a non-trivial
object; run it *after* A. It also needs programs the model gets wrong for
*semantic* rather than arithmetic reasons, which the E11 competence profile
(three of five families at 0.567–0.640 on a single operation) suggests is hard
to arrange in this model family.

**But it is the best second experiment**, and it is nearly free once A works.

---

## 6. Recommendation

**Candidate A (path-sensitive joins), with the name-mutation repair, and
Candidate D layered on top of it once it runs.** A is the only candidate that
is simultaneously hard for static analysis, unreadable from the text, equipped
with a non-trivial transition law (join, then strong update), and able to fail
informatively. B merges into A once it is made genuinely may-alias. C is a
robustness axis, not an object.

---

## 7. Tooling

### 7.1 vLLM for behavioural evaluation — yes for G1, no for the interventions

**What it buys.** Stage 82 is pure forced-choice scoring: one forward pass per
prompt, no hooks, no gradients, prompts ~37 tokens. That is exactly vLLM's
strength — continuous batching and paged attention would turn a several-minute
stage into a several-second one, and it would matter much more for the larger
behavioural sweeps a Candidate-A design implies (three-way branches × two
decidability arms × operation families multiplies the prompt count by ~6).

**What it cannot do.** vLLM's execution path is built for throughput, not for
introspection: there is no supported way to register a forward hook on a
decoder layer, edit one position's residual stream mid-pass, and read the
resulting states — and no autograd at all, which stage 87's DAS optimisation
requires. Any attempt to bolt hooks onto it would depend on internals that
change between releases, and this repository has already been bitten once by an
upstream change it did not control (`AutoTokenizer` on transformers 5.x
silently mis-tokenising deepseek-coder).

**Recommendation.** Optional acceleration for stage 82 **only**, behind a
`--backend vllm` flag, with a mandatory equivalence check: score the same 200
prompts through both backends and require identical argmax tokens and
log-probability agreement to ~1e-3 before trusting it. Stages 83, 86 and 87
stay on plain HF forward passes with hooks. Do not adopt it as the default
until that check is a test.

### 7.2 Tigress for a later C study — plausible, not now

[Tigress](https://tigress.wtf) is a source-to-source C obfuscator, and E9's
ladder is explicitly Tigress-inspired but re-implemented natively for Python in
`src/data/obfuscation.py` because Tigress is C-only. If the work ever moves to
C, Tigress becomes attractive: it supplies control-flow flattening, opaque
predicates, virtualisation and MBA encoding as a maintained, well-specified
tool, and its transformation set is exactly the semantics-preserving ladder E9
approximates by hand.

Two caveats. Tigress's guarantees are observational-equivalence guarantees on
the transformed program, which is what E9 already verifies by execution, so it
adds convenience rather than rigour. And it introduces a licensing and
reproducibility dependency for a corpus that currently regenerates from a
single Python module. **Not now; revisit only if the target language changes.**

### 7.3 Language and IR choice

| Target | For | Against | Verdict |
|---|---|---|---|
| **Python** | The whole pipeline is built on it: AST spans → verified token offsets, `beniget` cross-checks, execution-verified obfuscation, `sys.settrace` ground truth. Code models are strongest here, which matters because every gate is capability-limited first. | Semantics are dynamic and messy; aliasing and integer behaviour are idiosyncratic; nothing is in SSA. | **Keep, for the instrument and for Candidate A.** |
| **C** | Tigress; a mature analysis ecosystem; the language most static-analysis results are stated in. | Undefined behaviour makes "the true state" genuinely ambiguous in exactly the corner cases an adversarial design gravitates to; code models are weaker on C than on Python; the entire alignment and ground-truth stack would be rebuilt. | Only if Tigress becomes essential. |
| **LLVM IR** | **SSA is the reason to care.** Every value has exactly one definition, so "the value of `%7`" is unambiguous in a way `c` in Python never quite is; φ-nodes make control-flow joins *explicit syntactic objects*, so a join has a named location to anchor on and a ground truth that is read off the IR rather than inferred; `opt` supplies verified semantics-preserving passes for free; and the analysis vocabulary (reaching definitions, dominance, alias sets) is native. | Code models see far less IR than source, so the behavioural gate is the binding risk; verbose token counts; `clang -O0` output is noisy and needs canonicalisation. | **The right target for Candidate A specifically**, and worth a 10-GPU-minute behavioural probe before committing. A φ-node is a control-flow join with a name and an address — precisely the object §2 has to construct by hand in Python. |
| **Assembly** | Closest to what security tooling actually inspects. | No SSA, no types, register reuse destroys the value identity the whole design depends on, and model competence is worst here. | No. |

**Recommendation.** Keep the instrument validation in Python — the alignment
stack, the ground truth and the model's competence are all strongest there, and
there is no technical reason to move. For Candidate A, run the cheap
behavioural probe on LLVM IR before building: if a code model can answer
questions about φ-node values at all, IR gives the join a ground truth that is
*read off the program* rather than reconstructed, which is a materially
stronger identification story than the Python version can offer. If it cannot,
build Candidate A in Python with the name-mutation repair and say why.

---

## 8. References

- Geiger et al., DAS. https://arxiv.org/abs/2303.02536
- Huang et al., RAVEL. https://aclanthology.org/2024.acl-long.470/
- Wu, Geiger & Millière, ICML 2025. https://arxiv.org/abs/2505.20896
- Li et al., Othello-GPT. https://arxiv.org/abs/2210.13382 · Karvonen. https://arxiv.org/abs/2403.15498
- Shai et al., belief-state geometry. https://arxiv.org/abs/2405.15943
- Liu et al., shortcuts to automata. https://arxiv.org/abs/2210.10749
- Tigress. https://tigress.wtf
