# E12 — Latent store transitions: an INSTRUMENT VALIDATION study

**Status: IMPLEMENTED, not run.** Stages 80–88 exist (`src/data/store_programs.py`,
`src/data/store_semantics.py`, `src/models/das.py`, `src/experiments/store_*.py`,
`scripts/8*.py`, `tests/test_store.py`, `jobs/store_*.csh`). No GPU stage has
been executed and no result is claimed. `results/STATUS.yaml` carries E12 as
`active` with `claim: none`.

**E12 is not the research contribution.** Causal state interchange on a learned
low-rank subspace is established method — DAS, Othello-GPT, and variable-binding
work in symbolic programs all do a version of it (§12). What E12 asks is
narrower and entirely about the apparatus:

> Can we reliably identify and interchange a computed, **text-absent** program
> value in a pretrained code model, such that downstream computation correctly
> **transforms** the installed value?

A pass licenses the next experiment. It is not a finding, and §16 exists to
stop it being written up as one. The contribution this instrument is being
built *for* is in [E13_DIRECTIONS.md](E13_DIRECTIONS.md).

**Provenance.** Designed 2026-08-08 from a two-agent review (creative proposal
pass over five candidate directions, then a hostile reviewer pass with a
literature search), with every decision-critical claim verified directly
against this repository and the cited papers — see §3, which records three
facts the project's own documentation under-states. Reframed from "research
direction" to "instrument validation" on 2026-08-09, and implemented against
that framing.

---

## Contents

- [§0 The project, in plain terms](#0-the-project-in-plain-terms) — start here
- [§1 Why the current program is not yet convincing](#1-why-the-current-program-is-not-yet-convincing)
- [§2 The gap left by E2/E3 and E11](#2-the-gap-left-by-e2e3-and-e11)
- [§3 Verified facts that constrain the design](#3-verified-facts-that-constrain-the-design)
- [§4 What E12 asks, and what it does not](#4-what-e12-asks-and-what-it-does-not)
- [§5 The programs](#5-the-programs)
- [§6 Identification strategy](#6-identification-strategy)
- [§7 The gate sequence](#7-the-gate-sequence)
- [§8 The factorial design](#8-the-factorial-design)
- [§9 Controls](#9-controls)
- [§10 Thresholds and decision rules](#10-thresholds-and-decision-rules)
- [§11 What each outcome means](#11-what-each-outcome-means)
- [§12 Position in the literature](#12-position-in-the-literature)
- [§13 Implementation map](#13-implementation-map)
- [§14 Cost](#14-cost)
- [§15 Risks that remain](#15-risks-that-remain)
- [§16 Do not claim](#16-do-not-claim)
- [§17 References](#17-references)

---

# 0. The project, in plain terms

*This section assumes nothing. It re-explains the whole project from scratch,
then says what E12 adds. Everything after it is detail.*

## 0.1 The question

A code model can look like it understands a program when it is really just
following the text. Identifiers usually keep their meaning; things that belong
together are usually written near each other; indentation usually shows the
structure. A model that tracks only those regularities will get most
predictions right — and will fail exactly where they break down: a shadowed
name, a stale value, a branch that was never taken.

So the project asks one question:

> **Does a code model actually work out what a program *means*, or does it only
> track what the program *looks like*? And if it does work it out, does it then
> *use* what it worked out?**

## 0.2 How you tell the difference

The trick the project is built on is a pair of programs that differ in **one
character**, where that character changes the meaning while leaving everything
else identical:

```python
x = 3                  x = 3
def f():               def f():
    y = 7                  x = 7
    return x               return x
```

On the left, `return x` refers to the outer `x = 3`. On the right the inner
line has captured the name, so it refers to `x = 7`. The token we ask about
sits at the *same position* in both programs, the words around it are the same,
and the distance to everything is the same. Only the answer flips.

A "cheating" predictor — one that looks only at nearby words and distances — is
therefore right exactly half the time. Not approximately: **exactly 0.500, by
construction.** Anything above that is something the model computed rather than
read off the page. That construction-pinned floor is the project's real asset,
and it is unusual: most interpretability testbeds have to *estimate* how much a
shortcut explains rather than making it explain nothing.

## 0.3 What the project has found so far

| | Finding | Confidence |
|---|---|---|
| **Built** | Which definition a name refers to is *absent* at the input (0.500), is constructed within the first few transformer blocks, peaks near 0.98 mid-network, and is partly shed before the output. Same in a 1.3B and a 6.7B model. | Foundation |
| **Made of** | Survives 1,000 tokens of inert filler (0.921) and survives renaming every identifier in the middle layers (0.85–0.90); collapses under filler that reuses the same names (0.570) and under control-flow flattening (0.750). It breaks when the *problem* gets harder, not when the text gets longer or is spelled differently. | Supporting |
| **Used** | Partly, and late. Exchanging two output-aligned directions near the *readout* position moves the answer; the same edit where the binding is *resolved* does nothing — but a control showed no edit that small does anything there, so that is an underpowered question, not an answer. | Preliminary, and formally a no-go (§3.1) |

Three earlier interpretations were withdrawn (`docs/LEGACY_RESULTS.md`), all for
one reason: they were claims that something was *absent*, and none had a
positive control good enough to tell "absent" from "our instrument cannot see it
here".

## 0.4 What is missing

Three separate claims you could make about a model and a piece of program
meaning:

1. **It represents it** — the fact is in there and you can read it out.
2. **It updates it** — as the program proceeds, the model changes that fact the
   way the program says it should.
3. **It uses it** — the model's own later computation actually reads it.

The project has (1), robustness results *about* (1), and a partial, late,
contested version of (3). **It has never tested (2).** And every attempt at (3)
hit the same wall: the intervention is either a whole-state transplant (too
blunt — it carries the text difference along with the meaning) or a
two-dimensional nudge (too small to register at the site that matters). No
instrument in the repository has a size set by anything but the experimenter's
choice.

## 0.5 What E12 is

**E12 is a check on the measuring apparatus, not an experiment about code
models.** Before asking a new scientific question, we need one instrument that
demonstrably works. E12 builds it and tests it, and it does so on the hardest
version of the easiest case.

It studies a number that **never appears anywhere in the program**.

```python
a = 1              # a = 2  in the counterfactual — one token differs
b = 4
c = a + 4          # c is 5 here, 6 there — the digits 5 and 6 appear nowhere
d = c + 3          # d is 8 here, 9 there
assert d ==
```

The value of `c` is not written down, so it cannot be copied from a token. If
the model is going to answer, it has to hold that number somewhere.

E12 then does the only kind of intervention whose size is not chosen by hand.
It runs *both* programs and **writes the second run's coordinates for that
number into the first run's state** — an *interchange intervention*. There is
no "how hard should I push"; you install what the other program actually has.

Then the question that separates carrying a number from computing with one.
After installing the other run's value, read the model's internal state at the
*next* statement and sort the outcome into three bins:

| bin | what the state now says `d` is | what it would mean |
|---|---|---|
| **stale** | 8 — the original | the edit did nothing |
| **copied** | 6 — the number we installed, unchanged | it carries values but does not compute with them |
| **transformed** | 9 — the installed value *plus 3* | it applied the program's own next step to what we installed |

Only the third is a program state being updated. An intervention that merely
steers the answer token cannot produce it, and neither can a model that just
shuttles numbers around. To make that airtight, each base program carries
several different downstream operations over the same `c`, so **one edit has to
produce a different correct answer in each**.

## 0.6 Why this is validation and not a result

Installing a state and watching the model compute with it is a known technique.
It has been done on a board game (Othello-GPT), on chess, on entity tracking,
and on variable binding in symbolic programs; the formal version has a name
(causal abstraction) and a metric (interchange-intervention accuracy). E12's
only novelty is the *object* — a value with no token, in a pretrained
production code model, with a construction-pinned counterfactual — and that is
not enough to carry a paper.

What E12 buys is the right to run the next experiment on something harder
without wondering whether a null means the model lacks the structure or the
instrument cannot see it. §11 says what each outcome licenses;
[E13_DIRECTIONS.md](E13_DIRECTIONS.md) says what to do with a pass.

---

# 1. Why the current program is not yet convincing

**The foundation is correlational, and the project says so.** E2/E3 establish
decodability above a construction-pinned floor. Decodable is compatible with
the representation being a faithful shadow of a computation happening
elsewhere, and `docs/RESULTS.md` already treats "decodable" as insufficient for
a headline.

**Every causal attempt has been indexed by a magnitude the experimenter chose.**

| Attempt | Intervention | Failure mode |
|---|---|---|
| E7 | whole-state replacement at `sink_arg` | too coarse — that position is the only place the two programs differ, so the patch transports the surface difference; claim retired |
| E11, use position | rank-2 additive coordinate swap | too fine — the site's dose-response is 18× convex; a push along the *known-correct* direction at 2% of ‖h‖ gives 0.002 nats with an interval covering zero; null retracted |
| E11, readout position | same edit | registers, but the specificity check against the plain logit lens **fails** (§3.1), so what is shown is output-aligned value directions in general |

Two arms of one question, failing in opposite directions along a single axis
the design never controlled.

**Three retractions share one structure.** E6, E10-2 and E10-3 were all
absence-of-evidence claims whose positive controls were not matched *in kind*
(an identity control cannot license a relational null) or *in scale* (a rank-2
edit cannot license a null at a site with a convex response). The repository has
internalised the first lesson thoroughly. The second is recent.

**And there is no test of the update law anywhere.** E5's `competing_update`
filler is the nearest thing, and it is a frozen-probe robustness measurement,
not a transition test.

---

# 2. The gap left by E2/E3 and E11

E2/E3 show a **relation between two token positions** is decodable at a site.
E11 shows a **value present as a literal in the text** is causally read near the
readout position, through directions the plain unembedding explains at least as
well.

Neither establishes that the model holds a semantic quantity **that appears
nowhere in the program text**, transforms it according to the program's own
transition function, and routes the result onward.

That is two gaps at once — a **content** gap (no semantic quantity without a
token has been studied, so every result so far is compatible with sophisticated
token bookkeeping) and an **instrument** gap (every intervention the repo owns
is a whole-state transplant or a hand-scaled low-rank edit). E12 closes the
instrument gap using the content gap as its test case.

---

# 3. Verified facts that constrain the design

Checked directly against this repository and against the cited papers. Each
changed the design.

## 3.1 E11's own pre-registered gate is a NO-GO on both positions

`results/jspace/6.7b-5fam/go_no_go.md` (use) and `go_no_go_answer.md` (answer)
both read **Verdict: NO-GO**.

| Check | use | answer |
|---|---|---|
| `behavioural_balanced_accuracy` (≥ 0.75) | FAIL 0.706 | FAIL 0.706 |
| `readout_beats_random_control` | FAIL +0.056 [−0.007, +0.117] | PASS +0.257 [+0.218, +0.297] |
| `swap_moves_logits_toward_swapped_value` | FAIL +0.001 [−0.002, +0.004] | PASS +0.141 [+0.111, +0.175] |
| `swap_is_specific_to_the_value_subspace` | FAIL | **FAIL −0.016 [−0.024, −0.009]** vs `logit_value` |
| cross-operation `all positive` | False | False |

`results/STATUS.yaml` and `docs/RESULTS.md` report the readout-position effect
and the Jacobian caveat, but not that the run is formally a no-go under its own
pre-registration, and not that the specificity check failed. **That should be
stated in the paper as a failed check, not only as a caveat**, and E12 must not
be built on an assumption that E11 passed. It is not: E12 shares no instrument
with E11 beyond hooks and the bootstrap.

## 3.2 The E11 corpus is small

`data/synthetic/jspace_pairs_6.7b_5fam.jsonl`: 290 pairs, **18 calibration and
42 test base programs**. Every E11 cluster bootstrap has 42 clusters. E12
therefore pre-registers `n_bases = 400`.

## 3.3 `VALUE_POOL` has six elements

`src/data/counterfactual_pairs.py:89` — `VALUE_POOL = tuple(range(2, 8))`, so
the stage-72 value probe has six coefficient rows and any "rank sweep to 32" on
that instrument is undefined past six. E12's rank axis is built from an
orthonormal basis learned directly (`src/models/das.py`), which is defined at
any rank ≤ d.

## 3.4 Pair probes are the expensive path; single-position probes are not

From `results/manifests/`: stage 20 on 6.7b `core` (five tasks, includes pair
probes) took 109,463 s ≈ **30.4 h**; the same stage on `taint_state` alone
(single-position) took **344 s**. The driver is the 16,384-dimensional
`pair_feature`. **E12 uses single-position multiclass decoders throughout.**

## 3.5 GPU throughput

Stage 73's last run: 76,560 interventions in 2,464.7 s ≈ **31 single-sequence
forward passes per second** at 6.7b fp16, mean prompt ≈ 49 tokens. All budgets
in §14 use this constant. E12's prompts are ~37 tokens.

## 3.6 A same-answer placebo returns a structural zero

`src/experiments/jspace_swap.py:356-358` computes the endpoint as a logit
difference between the bound and the *other* answer, so an irrelevance control
built with `answer_source == answer_target` gives `delta_ld ≡ 0` by
construction. E12's irrelevant twin therefore differs in a **literal no
statement reads**, keeping the two-alternative structure intact.

## 3.7 The 0.500 floor is pinned against a local-window baseline

The surface baseline is ±3 token ids plus bucketed distance
(`docs/METHODS.md` §7), which cannot represent cross-position token equality.
The floor may well hold, but "0.500 by construction" currently means "0.500
against a local-window baseline". Adding a string-equality baseline to stage 20
is an open item independent of E12.

## 3.8 The closest prior work exists and was uncited

Wu, Geiger & Millière, *How Do Transformers Learn Variable Binding in Symbolic
Programs?*, ICML 2025 ([arXiv:2505.20896](https://arxiv.org/abs/2505.20896)) —
verified from the full text. A 37.8M from-scratch transformer on dereference
chains to depth 4 with distractors; **interchange interventions** and
residual-stream patching; linear probes reach 30.87% on variable values and
8.90% on complete program state; the authors conclude the model *"does not
maintain a complete program state in a linearly decodable format at any single
vector location"*, implementing binding through dynamic routing.

This is why E12 is validation rather than a contribution: the technique and a
close version of the question are already in print. It also supplies a
published, sharp alternative outcome to hold the instrument against (§11).

---

# 4. What E12 asks, and what it does not

**Asks.** Can this apparatus identify and interchange a computed, text-absent
program value such that downstream computation transforms it?

**Does not ask.** Whether code models represent program semantics; whether that
representation is causally used in general; anything about scale, architecture,
or real code. Those need the extensions in
[E13_DIRECTIONS.md](E13_DIRECTIONS.md), and they need this instrument first.

**The operational claim, per gate.** Each of G0–G5 is a property of the
apparatus:

| gate | the apparatus can… |
|---|---|
| G0 | generate programs whose ground truth two independent readings agree on |
| G1 | pose them to a model that can actually answer them |
| G2 | read the text-absent value out of the states, above measured controls |
| G3 | watch that value change across statements, with a live transfer measurement |
| G4 | report the *transformed* outcome when the state truly came from the counterfactual |
| G5 | achieve the same with a low-rank, learned, magnitude-free edit that survives six controls |

---

# 5. The programs

## 5.1 The template

One template, every statement the same AST shape
(`src/data/store_programs.py::render`):

```python
def f():
    a = 1          # head literal — the one differing token
    b = 4          # irrelevant variable; no statement reads it
    c = a + 4      # INJECTION SITE. c = 5 / 6, absent from all text
    d = c + 3      # READ SITE. d = 8 / 9
    return d
assert f() ==
```

Each base emits three programs:

| program | differs from base by | role |
|---|---|---|
| `base` | — | the run being intervened on |
| `counter` | one token: the head literal | the donor, and the counterfactual |
| `irrelevant` | one token: `b`'s literal | the "you moved something, but not the thing that matters" control |

and one record per **operation family** over the same `c`: `add`, `sub_from`,
`double_sub`, `mod`. A base is kept only if ≥3 families verify — two for the
cross-operation falsification and one to hold out in G5.

## 5.2 Invariants, enforced at generation and re-checked in tests

| invariant | what it closes |
|---|---|
| exactly one differing token per pair, at the head literal | everything else is held identical |
| equal token length across the triple | every anchor is the same index in all three |
| mutation ≥ 6 tokens before the injection site, never adjacent | no local window can carry the label |
| `{intermediates} ∩ {literals in ANY program} = ∅` | **the load-bearing one** — the tracked value has no token |
| `{answers} ∩ ({literals} ∪ {intermediates}) = ∅` | an answer token and a value token are never the same row |
| **stale, copied, transformed pairwise distinct** | otherwise the causal readout cannot tell the three apart |
| all tracked quantities single-token, verified | the logits are read at the right row |
| answer appends exactly one token | the logits are read at the right position |
| the irrelevant twin's answer equals the base's | the control is actually irrelevant |

A note on economy, because it is not obvious: everything above has to fit in
ten single-token digits at once — two head values, an offset, the operation's
parameter, two intermediates and two answers, all mutually disjoint. Drawing
the irrelevant literal from a fresh pool spent two more digits and dropped the
generator's acceptance rate to ~2%; drawing it from digits the program already
contains costs nothing and brought the rate to ~100% (60/60 bases in 8 s).

## 5.3 Ground truth (stage 81, gate G0)

Three readings, required to agree at every statement:

1. **execution** — `execute_program`, plus the operation's own Python function;
2. **trace** — `sys.settrace` capturing `frame.f_locals` after each line;
3. **reference interpreter** — a small AST evaluator written independently of
   the rendering path (`src/data/store_semantics.py`).

Disagreements are dropped and counted, never reconciled. This is the discipline
already applied to def-use against `beniget`, which caught a real mislabelling
of `b = b + a`.

---

# 6. Identification strategy

| Confound | How it is removed |
|---|---|
| **Lexical identity** | The intervened quantity is absent from all three texts, so no token-presence direction can carry it. One-token counterfactual; equal token length; mutation ≥6 tokens from the injection anchor; anchor windows token-identical; names permuted across bases. |
| **Syntax / AST shape** | Every statement is `Assign(Name, BinOp(Name, Constant))`. The irrelevant statement sits between the mutation and the injection site at matched distance. |
| **Position and distance** | Anchors are the same token indices in all three programs by construction. `pre_def` is a position-matched negative where nothing is bound yet. |
| **Output/answer information** | Answers are single tokens, distinct, disjoint from every literal *and* every intermediate. The decisive control: one alignment must work on an operation family **held out of its training**, and a direction encoding the answer cannot transfer across families that map the same value to different answers. |
| **Intervention magnitude** | No α. The interchange installs whatever the counterfactual run holds. Scored between two *measured* bounds — the whole-state ceiling (the rank-d limit of the same operator) and a random subspace at matched **removed norm**, not merely matched rank, because for an orthogonal projector only the span matters. |
| **Probe capacity / memorisation** | Linear decoders only; alignment fitted on a disjoint calibration split (`assert_disjoint`, unit-tested against a deliberate leak); a Hewitt–Liang control task must fail; the smallest rank achieving the effect is the reported number. |

## 6.1 One honest limit, stated up front

`c` is a deterministic function of the visible text, so a baseline that can
execute the program scores 1.0. **G2 is a precondition, not a result, and has
no construction-pinned floor.** It is reported against the *measured* lexical
window baseline, whose reach is bounded by construction (the head literal is
≥6 tokens away), which makes it informative about this decoder rather than
about decodability in principle. The claim-bearing gates are G3–G5.

---

# 7. The gate sequence

Each stage declares its prerequisites in
`src/experiments/store_gates.py::STAGE_REQUIREMENTS` and **refuses to run**
(exit 2) unless they have passed. `--override-gate REASON` is permitted — a
diagnostic often needs to see what a downstream stage does after an upstream
failure — and is recorded in `gates.yaml`, in the manifest, and in every output
row, so a number produced under an override can never later be mistaken for one
produced under a passing gate.

| gate | stage | asserts | threshold |
|---|---|---|---|
| **G0** | 81 | trace, interpreter and stored labels agree; every invariant holds | ≥ 0.999 of records |
| **G1** | 82 | the model solves the programs | balanced accuracy ≥ 0.75 overall, ≥ 0.70 per retained family, ≥ 2 families retained |
| **G2** | 84 | the text-absent value is decodable above measured controls | margin ≥ 0.05 over the better of surface / control-task |
| **G3** | 85 | the natural transition is measurable | tracked transfer retention ≥ 0.60 **and** text-present control retention ≥ 0.90 **and** reversal ≥ 0.50 with CI above zero |
| **G4** | 86 | whole-state interchange yields the TRANSFORMED state | transformed ≥ 0.50, CI above the copied rate |
| **G5** | 87 | low-rank interchange clears every control | ≥ 50% of the G4 ceiling, all control contrasts CI > 0, every retained family positive, held-out family transfers |

G4 is doing double duty and is the cheapest form of the control whose absence
retired E10-3: if the readout cannot report `transformed` when the state
genuinely came from the counterfactual program, no G5 null is interpretable.

---

# 8. The factorial design

| Level | Held fixed | Varied |
|---|---|---|
| Within a triple | every token but one; token length; anchor indices; AST shape; names; formatting; operation family | head literal (→ intermediate → answer), or the irrelevant literal |
| Across families of one base | head literal; intermediate; all anchors | operation family (≥3) → **a different correct answer from the same intermediate** |
| Across the corpus | all §5.2 invariants | names; head/offset draw; which family is held out |
| Intervention | programs, anchors, decoder | variant (7 arms), rank (1/2/4/8…), injection layer |
| Models | everything | 1.3b pilot → 6.7b; `starcoder2-3b` optional |

---

# 9. Controls

| Class | Control | Rules out |
|---|---|---|
| **Ceiling + aliveness** | `whole_state` (rank-d limit of the same operator) | a dead readout; sets the normalizer |
| **Positive (measurement alive)** | text-present head-value transfer matrix | "transfer is unmeasurable here" masquerading as decay |
| **Random subspace** | `random_rank`, and `random_norm` matched on removed norm | any subspace of this rank / any edit of this size |
| **No-op** | donor is the state itself — provably the zero vector | numerical noise; a structural zero kept in the output |
| **Irrelevant variable** | donor is the twin that differs in an unread literal | installing *any* other run's state |
| **Position** | injection at `pre_def` | the position rather than the subspace; a second structural zero |
| **Answer-steering** | `held_out_family` transfer | the subspace encoding the answer instead of the value |
| **Lexical / capacity** | measured surface window; Hewitt–Liang control task; selectivity | text features; name memorisation; priors |
| **Magnitude** | `edit_fraction` logged per condition and used in the rule | the E11 dose failure, in either direction |

---

# 10. Thresholds and decision rules

All thresholds are in `configs/experiments.yaml` and in the module constants,
and are echoed into every gate's `detail` string. Changing one is a change to
the experiment, not a reporting choice.

- Intervals are **cluster bootstraps over base programs** throughout
  (`src/analysis/bootstrap.py`); control comparisons are **paired on the same
  rows**.
- The alignment is fitted on **calibration bases only** and evaluated on
  disjoint test bases; the split lives in the data file so every stage agrees
  by construction.
- Families failing G1 are **retained in the CSV and excluded from the retained
  set**, never silently dropped.
- `n_bases = 400` (≈280 test bases), against E11's 42.

---

# 11. What each outcome means

| Outcome | Reading | What it licenses |
|---|---|---|
| **All gates pass** | The apparatus can identify and interchange a text-absent value, and downstream computation transforms it. | Run a semantic extension from E13_DIRECTIONS. **Not** a paper on its own. |
| **G5 fails, G4 passes** | The state is transportable but not in any small learned subspace at this site. | Consistent with Wu, Geiger & Millière's routing conclusion (§3.8) on a much smaller model — worth reporting as a methods note, and a reason to prefer a design that does not need low-rank localisation. |
| **G4 fails** | The readout cannot report the transformation even under a whole-state interchange. | An instrument failure, not a model result. Fix the readout; nothing below is interpretable. |
| **G3 fails with the control alive** | The value is decodable but the format is not position-invariant. | The transition must be read some other way; the interchange design needs rethinking before use. |
| **G3 fails with the control dead** | The transfer measurement does not work here. | Says nothing about the model. This is exactly the E10-3 ambiguity, caught before it reaches a write-up. |
| **G2 fails** | Not linearly available at the anchors tried. | Try other anchors/layers, or accept that this instrument needs a non-linear readout — which would change what "decodable" means downstream. |
| **G1 fails** | A capability limit. | Not a representation result. Reduce chain difficulty or change model. |

Every branch is informative, and none of them is a scientific claim about code
models.

---

# 12. Position in the literature

E12 is a **new combination of known methods on a slightly new object**, and the
write-up must say so.

| Work | What it already does | E12's only delta |
|---|---|---|
| Geiger et al., DAS; Wu et al., Boundless DAS; Huang et al., RAVEL | interchange interventions on a learned subspace; interchange-intervention accuracy; the isolation criterion | applied to a program store in a pretrained code model |
| Li et al. (Othello-GPT); Nanda et al.; Karvonen | intervene on a world state, model's subsequent rule-governed behaviour follows | from-scratch models, one game, update rule is a lookup; here the transition is arithmetic over a latent value |
| Wu, Geiger & Millière, ICML 2025 | interchange interventions on dereference chains; program state *not* linearly decodable at a single location | 37.8M from-scratch vs pretrained 1.3B/6.7B; text-absent value |
| Jin & Rinard, ICML 2024 | intermediate state becomes decodable over training | no activation intervention, no transition test |
| Feng & Steinhardt; Kim & Schuster; Prakash et al.; Oh & Demberg | binding IDs, entity state, rebinding circuits | retrieval of a stored attribute vs application of a transformation to it |
| Code probing: Wan et al.; AST-Probe; Troshin & Chirkova; Ma et al.; GraphCodeBERT | recoverability of AST/DFG-computable relations | those relations are computable from the text; this value is not |

---

# 13. Implementation map

| Stage | Script | Module | Gate | Where |
|---|---|---|---|---|
| 80 | `scripts/80_store_pairs.py` | `src/data/store_programs.py` | — | CPU |
| 81 | `scripts/81_store_verify.py` | `src/data/store_semantics.py` | **G0** | CPU |
| 82 | `scripts/82_store_behaviour.py` | `src/experiments/store_behaviour.py` | **G1** | GPU |
| 83 | `scripts/83_store_extract.py` | `src/experiments/store_decode.py` (storage) | — | GPU |
| 84 | `scripts/84_store_decode.py` | `src/experiments/store_decode.py` | **G2** | CPU |
| 85 | `scripts/85_store_transition.py` | `src/experiments/store_decode.py` | **G3** | CPU |
| 86 | `scripts/86_store_ceiling.py` | `src/experiments/store_interchange.py` | **G4** | GPU |
| 87 | `scripts/87_store_interchange.py` | `src/models/das.py` + above | **G5** | GPU |
| 88 | `scripts/88_store_report.py` | `src/experiments/store_gates.py` | — | CPU |

Reused unchanged: `src/data/alignment.py`, `src/data/counterfactual_pairs.py`
(tokenizer helpers, `execute_program`), `src/probes/base.py`,
`src/analysis/bootstrap.py`, `src/utils.py`.
Added to `src/models/hooks.py`: two **siblings**, `transform_and_capture` and
`transform_positions_with_grad`; no existing function was modified.

Operational rules inherited from E11's failures: run the decoder stage before
the intervention stages so frozen decoders exist on disk (the recorded reason
`probe_basis` silently never ran), and record the selected cell in the manifest
before reading test numbers.

---

# 14. Cost

At the measured 31 passes/s (§3.5), 400 bases × ~3.4 families × 3 variants:

| Stage | Cost |
|---|---|
| 80 + 81 | ~2 min CPU (60 bases took 8 s) |
| 82 behaviour | ~5 GPU-min |
| 83 extract | ~10–15 GPU-min |
| 84 + 85 | minutes CPU (single-position probes; §3.4) |
| 86 ceiling | ~20–30 GPU-min |
| 87 DAS | ~1–3 GPU-h (dominated by the alignment's backward passes) |
| **Total** | **≈ 2–4 GPU-hours** for the full 6.7b run |

The 1.3b pilot is roughly a quarter of that and should always be run first.

---

# 15. Risks that remain

1. **Chain competence (G1).** The repo's own `jspace_behaviour.csv` shows 6.7b
   at 0.567–0.640 on three of five families for a *single* operation. E12 uses
   depth-2 chains of small-digit arithmetic for exactly this reason, but G1 may
   still fail. It is the cheapest gate and it runs first.
2. **DAS is expressive enough to find structure that is not there.** Mitigated
   by the disjoint split, the control-label alignment, the smallest-rank rule,
   and decisively by held-out-family transfer — but this is the risk that would
   most damage a naive reading, and it is why §16 exists.
3. **fp16 backward stability.** Stage 87 is the only E12 stage that
   backpropagates. If the loss goes non-finite, re-run with `--dtype float32`;
   the pattern is the same one `src/models/lens.py` documents.
4. **Anchor choice.** The injection site is the last covering token of `c`'s
   RHS. If the value is written to the stream later than that, G4 fails for a
   reason that is about anchoring, not about the model — the `pre_def` and
   `noop` structural zeros are what distinguish a real failure from a
   mis-anchored one.
5. **Synthetic, one model family.** Unchanged from E8's limitation. Nothing in
   E12 transfers to real code, and E12 does not claim it does.

---

# 16. Do not claim

- Not that E12 is a finding. It is instrument validation; the technique is
  established (§12). Any write-up must present it as a methods section or an
  appendix, never as a contribution.
- Not "the model understands", "reasons about", or "implements an interpreter".
- Not that G2 is evidence of anything on its own — no pinned floor, and an
  executing baseline scores 1.0.
- Not "the value is not represented" from a G5 null. The licensed statement is
  bounded by rank and site: *"no ≤r-dimensional interchangeable subspace here"*.
- Not "low-dimensional" if the effect needs rank 16. Report the smallest rank.
- Not any transfer to real code, other languages, or other model families.
- Not a scale claim from 1.3b vs 6.7b unless behavioural competence is matched —
  the E6 constant-responder failure was exactly this.
- Not that E11's readout-position effect passed. Both go/no-go files read NO-GO
  and the specificity check failed (§3.1).

---

# 17. References

- Geiger, Wu, Potts, Icard & Goodman. *Finding Alignments Between Interpretable Causal Variables and Distributed Neural Representations* (DAS). CLeaR 2024. https://arxiv.org/abs/2303.02536
- Huang, Wu, Potts, Geiger et al. *RAVEL.* ACL 2024. https://aclanthology.org/2024.acl-long.470/
- Wu, Geiger & Millière. *How Do Transformers Learn Variable Binding in Symbolic Programs?* ICML 2025. https://arxiv.org/abs/2505.20896
- Jin & Rinard. *Emergent Representations of Program Semantics in Language Models Trained on Programs.* ICML 2024. https://arxiv.org/abs/2305.11169
- Li, Hopkins, Bau, Viégas & Wattenberg. *Emergent World Representations.* ICLR 2023. https://arxiv.org/abs/2210.13382
- Nanda, Lee & Wattenberg. *Emergent Linear Representations in World Models.* BlackboxNLP 2023. https://arxiv.org/abs/2309.00941
- Karvonen. *Emergent World Models and Latent Variable Estimation in Chess-Playing Language Models.* 2024. https://arxiv.org/abs/2403.15498
- Feng & Steinhardt. *How do Language Models Bind Entities in Context?* ICLR 2024. https://arxiv.org/abs/2310.17191
- Hewitt & Liang. *Designing and Interpreting Probes with Control Tasks.* EMNLP 2019.
- Zhang & Nanda. *Towards Best Practices of Activation Patching.* ICLR 2024. https://arxiv.org/abs/2309.16042
- Nikankin, Reusch, Mueller & Belinkov. *Arithmetic Without Algorithms.* 2024. https://arxiv.org/abs/2410.21272
- Liu, Ash, Goel, Krishnamurthy & Zhang. *Transformers Learn Shortcuts to Automata.* ICLR 2023. https://arxiv.org/abs/2210.10749
