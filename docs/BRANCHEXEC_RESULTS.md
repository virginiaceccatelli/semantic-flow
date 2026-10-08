# BranchExec results: four code models

**Runs:** `results/branchexec/{deepseek-coder-1.3b,deepseek-coder-6.7b,starcoder2-3b,starcoder2-7b}/full`
(stages 250–260); cross-model table `results/branchexec/compare_20261007_0923` (stage 258).
**Design and runbook:** [BRANCHEXEC.md](BRANCHEXEC.md).

**Revision note (2026-10-07).** Two claims from the previous version are withdrawn:

- **"The answers do not use the branch outcome."** This holds only when the input is
  written before the code. In the CruxEval order every model follows the branch above
  chance (Finding 2).
- **"With size, the usable leverage moves from the `if` to the answer."** The
  single-block sweep shows the split is between model families, not sizes. The earlier
  contrast came from where the 5-block band was placed (Finding 4).

The first 6.7B `report/report.md` was also superseded earlier: its "online execution"
label used too loose a rule, and its error analysis was confounded. Every number below
comes from the committed CSVs.

---

## The question

> When a code model reads a **real** program whose input it already knows, does it
> compute which branch an `if` will take, and does its predicted output **use** that?

The question separates *representation* (the information is in the hidden state) from
*utilisation* (the answer depends on it). For variable binding both hold in this
repository: probes reach 0.98, and rank-1 DAS installs the binding on 100% of held-out
cases (R1, R10). Here the same question is asked of an execution fact, on real code, in
two model families at two sizes each.

## The experiment

From MBPP, HumanEval and CruxEval we take every `if` that runs exactly once on a test
input. We then search for a one-literal change to that input that makes the same `if`
take the other branch and changes the output; execution verifies each candidate. The two
programs in such a **pair** are token-identical apart from that literal.

Two prompt orders are used:

```python
# input first: the input is in the signature, read before the body
def f(xs=[3, 1]):
    if len(xs) > 2: ...
assert f() ==

# CruxEval order: the input comes after the code
def f(xs):
    if len(xs) > 2: ...
assert f([3, 1]) ==
```

In the CruxEval order the `if` is read before the input exists, so the state there
cannot hold the outcome; the state at the `if` is identical across a pair (|Δh| = 0).
**Readout, link and steering therefore use the input-first order. Behaviour is measured
in both orders.**

| | question | instrument | chance / control |
|---|---|---|---|
| **Readout** (252) | is the branch outcome in the state at the `if`? | direction learned **only on synthetic programs** (difference of means), frozen | exactly 0.5 (identical code within a pair) |
| **Behaviour** (253, 259) | does the predicted output follow the branch that runs? | greedy answer vs the true output and the other branch's output, in both orders | the model's own branch bias |
| **Link** (256) | in pairs the answer treats identically, is the outcome still computed? | readout split by what the answer does | readout in pairs whose answer follows the input |
| **Repair** (257, 260) | does pushing that state toward the true branch fix wrong answers, specifically? | h ← h + α‖h‖v, on a 5-block band (257) or one block at a time (260), at the `if`, the first body token or the answer | away-pushes, random directions; an answer-token push that knows the target |

## Data

| model | real pairs | real members |
|---|---:|---:|
| DeepSeek-Coder 1.3B | 735 | 1,168 |
| DeepSeek-Coder 6.7B | 735 (MBPP 520 / HumanEval 142 / CruxEval 73) | 1,168 |
| StarCoder2 3B | 743 | 1,184 |
| StarCoder2 7B | 743 | 1,184 |

Pairs are rebuilt per tokenizer. Each model also gets about 1,300 synthetic pairs, which
supply only directions, layers and doses.

---

## Finding 1: every model computes the branch outcome at the `if`

Frozen synthetic direction, real pairs, input first, `if` colon:

| model | layer chosen on synthetic (depth) | readout at `if` [95% CI] | real-trained ceiling | best real layer |
|---|---|---|---:|---|
| DeepSeek 1.3B | 8 (0.38) | 0.595 [0.551, 0.638] | 0.682 | — |
| DeepSeek 6.7B | 16 (0.53) | **0.746** [0.700, 0.791] | 0.796 | 15: 0.759 |
| StarCoder2 3B | 13 (0.47) | 0.678 [0.639, 0.721] | 0.673 | 14: 0.685 |
| StarCoder2 7B | 21 (0.69) | 0.680 [0.637, 0.726] | 0.651 | 15: 0.727 |

Controls (6.7B shown; the others behave the same way):

| control | pair accuracy |
|---|---:|
| `if` before the input (CruxEval order), \|Δh\| = 0 | 0.500 |
| model-free reader of the input tokens | 0.473 |
| shuffled-label direction | 0.482 |
| end of signature (input read, body not yet) | 0.487 |

- **Synthetic transfers to real code.** In every model the synthetic direction reaches or
  nearly reaches the real-trained ceiling.
- **Location.** The outcome appears where the condition is read and fades by the answer
  position (0.48–0.59).
- **Size.** It grows within DeepSeek (0.60 → 0.75) but not within StarCoder2 (0.68 → 0.68).
- **Layer choice.** For StarCoder2 7B the synthetic choice of layer (21) is later than
  the best real layer (15).

## Finding 2: whether the answer follows the branch depends on the prompt order

Among answers that equal one of the two branch outputs, the share that follow the branch
that actually runs:

| model | input first | CruxEval order | pairs whose answer follows the input (first / CruxEval) | same branch for both inputs, CruxEval order (bias alone predicts) |
|---|---:|---:|---|---|
| DeepSeek 1.3B | 0.533 | 0.613 | 0.055 / 0.293 | 0.684 (0.531) |
| DeepSeek 6.7B | 0.511 | **0.721** | 0.110 / **0.532** | **0.450** (0.513) |
| StarCoder2 3B | 0.508 | 0.626 | 0.066 / 0.337 | 0.624 (0.519) |
| StarCoder2 7B | 0.514 | 0.648 | 0.096 / 0.390 | 0.579 (0.527) |

Exact accuracy, input first / CruxEval order: 0.315 / 0.375, 0.311 / 0.485, 0.312 / 0.413
and 0.331 / 0.441, in the same model order. No member was dropped in the CruxEval order.

**Input first: branch choice at chance in all four models.** Answers default to the `if`
body (63–69%), and about 9 in 10 pairs get the same branch for both inputs, against 53–58%
expected from that bias alone.

The wrong answers are informative. A wrong answer here is the **exact** output of the
other branch, computed on the real input, so the model uses the input values to compute
a branch body. It just doesn't use them to choose the branch.

**CruxEval order: branch-following above chance in every model.**
- It is strongest in DeepSeek 6.7B: 0.72 overall, 0.83 on HumanEval, and 53% of pairs
  answered correctly for both inputs.
- DeepSeek 6.7B gives the same branch to both inputs *less* often than its body bias
  predicts (0.45 vs 0.51), so it is following the input.
- Within DeepSeek, branch-following grows with size (0.61 → 0.72). Within StarCoder2 it
  grows only slightly (0.63 → 0.65).
- The same branch is chosen in both orders for only 61–74% of members decided in both.

So the branch decision the models act on is formed in the CruxEval order, where the input
arrives after the code: **at or after the call**. When the input comes first and the
outcome is already available at the `if` (Finding 1), the answer does not use it.

## Finding 3: input first, the outcome is computed even when the answer ignores it

Readout pair accuracy (input first, `if` colon), split by what the model's input-first
answers do:

| model | pairs whose answer follows the input | pairs given one branch for both inputs | within-branch AUROC |
|---|---|---|---:|
| DeepSeek 1.3B | 0.591 (n = 22) | 0.598 (n = 373) | 0.565 |
| DeepSeek 6.7B | 0.791 [0.643, 0.927] (n = 43) | 0.755 [0.690, 0.813] (n = 343) | 0.707 |
| StarCoder2 3B | 0.714 [0.538, 0.889] (n = 28) | 0.661 [0.606, 0.719] (n = 387) | 0.522 |
| StarCoder2 7B | 0.791 [0.622, 0.944] (n = 43) | 0.667 [0.607, 0.729] (n = 396) | 0.640 |

In every model the state at the `if` separates the two inputs whether or not the answer
does, and every pair of intervals overlaps. When the input-first answer ignores the
input, the outcome was still computed. The failure is in using it.

The point estimates are higher in "follows the input" pairs for both StarCoder2 models,
but those categories are small.

The within-branch AUROC asks whether, among members with the same true branch, the
projection predicts which branch the model answers. It is clearly above chance only for
DeepSeek 6.7B (0.71) and StarCoder2 7B (0.64). Within pairs the input-driven state
changes while the answer does not, so this must come from a component that varies
between programs. It is correlational.

## Finding 4: the causal handle is at the `if` in StarCoder2 and at the answer in DeepSeek

Stage 260 edits **one block at a time** across the whole depth, on real members whose
input-first answer is the wrong branch's output (322–371 per model). Each edit pushes
toward the true branch, away from it, or in a random direction, at α = 1.0.
**Margin = repair(toward) − repair(away)**, where repair means the answer becomes
exactly correct.

| model | best `if` block: margin [95% CI] (toward / away / random) | best answer block: margin (toward / away / random) | first body token |
|---|---|---|---|
| StarCoder2 3B | **+0.204** [0.143, 0.266] @ depth 0.37 (0.209 / 0.006 / 0.039) | +0.075 @ 0.23 | ≤ +0.034 |
| StarCoder2 7B | **+0.124** [0.060, 0.192] @ depth 0.41 (0.137 / 0.013 / 0.032) | +0.070 @ 0.66 | ≤ +0.027 |
| DeepSeek 1.3B | +0.053 @ 0.29 | **+0.245** [0.108, 0.370] @ 0.71 (0.376 / 0.130 / 0.102) | ≤ +0.006 |
| DeepSeek 6.7B | +0.063 @ 0.34 | **+0.294** [0.147, 0.439] @ 0.53 (0.441 / 0.147 / 0.046) | ≤ +0.006 |

**StarCoder2: the state at the `if` is a directional handle.**
- Toward-pushes repair; away-pushes almost never do (≤ 1.3%).
- The effect spans several adjacent blocks at 0.3–0.5 depth: 3B is +0.18 to +0.20 at
  0.37–0.43; 7B is +0.10 to +0.12 at 0.28–0.41.
- That depth is earlier than where the readout is strongest. For StarCoder2 7B, the
  5-block band placed at the synthetic readout layer (21) repaired nothing (0.022), while
  block 12 repairs 13.7%.

**DeepSeek: the `if` state hardly moves the answer** (margins ≤ 0.063 in both sizes).
Wrong answers can be repaired from the answer position, but only at isolated blocks:
- 6.7B at block 16, with the neighbouring blocks near zero;
- 1.3B at block 16, with a smaller peak at block 8.

Both effects are directional (away 0.13–0.15, random 0.05–0.10), but a one-block spike is
a narrow handle and should be treated as such.

**Answer-position edits are not a clean branch direction.**
- In StarCoder2, late answer edits are *anti*-directional: margins down to −0.24, so
  pushing "toward" repairs less than pushing away.
- Random edits at the answer increasingly repair at late blocks (up to 0.14–0.21 at the
  last block), because disrupting the answer sometimes dislodges a wrong output.

**Nothing is steerable at the first body token, in any model.**

The 5-block band of stage 257 is consistent once its placement is known.
- At α = 0.4 it repaired at the `if` in StarCoder2 3B (21.5%), whose band covers the
  causal blocks, and in DeepSeek 1.3B (13%).
- It repaired at the answer in DeepSeek 6.7B (43.8%), whose band covers block 16.
- It repaired almost nothing in StarCoder2 7B, whose band at the readout layer misses
  both.

The earlier reading that size moves the handle from the `if` to the answer is withdrawn.
Both DeepSeek models put it at the answer, both StarCoder2 models at the `if`.

## What the findings say together

1. **Code models compute branch outcomes while reading real code.** When the input is
   known before the body, a direction learned only on synthetic programs reads which
   branch a real `if` takes. This holds in four models from two families (0.60–0.75,
   chance 0.5).
2. **That early computation is not what their answers rely on.** With the input first,
   answers follow the true branch at chance, even in pairs where the outcome was computed
   just as well. With the input after the code, where the `if` cannot know the outcome,
   answers follow the branch well above chance (up to 0.72). The decision the models act
   on is made late, at or after the call, not at the `if`.
3. **Whether the early state can be made to matter depends on the architecture.** In
   StarCoder2, pushing the `if` state toward the true branch at about 0.4 depth fixes up
   to 21% of wrong-branch answers, specifically. In DeepSeek it fixes almost none, and the
   only handle is a single block at the answer position.

For variable binding, the represented state is the one the answer uses (R10). For branch
outcomes, the state is computed early, then largely bypassed by a later decision.

## Limits

- **Order and site are confounded by design.**
  - The readout and every intervention use the input-first order, the only order in which
    the `if` can hold the outcome.
  - The behaviour that does follow the branch (CruxEval order) has no `if`-site state to
    compare against.
  - The claim that the decision is made late rests on this contrast and on the
    answer-site results. No readout of the branch outcome at the call position in the
    CruxEval order exists yet.
- **The input-first format is unnatural.** It lowers accuracy by 6–17 points relative to
  the CruxEval order. Part of the chance-level branch choice may be format-specific.
- **Coverage.** Only `if`s that run once, with inputs a single literal edit can flip.
  CruxEval contributes few pairs.
- **Correlational parts.** The readout shows linear presence. The within-branch AUROC is
  correlational and program-level.
- **Narrow handles.** The DeepSeek answer-site handles are single blocks. Single-block
  edits use α up to 1.0 (relative to ‖h‖), and late answer edits degrade outputs.
- **Two families, two sizes each.** Size effects are read within a family only.

## Next

1. **Read the decision where it is made.** In the CruxEval order, read the branch outcome
   at the call tokens (`f(<input>)`) and at the answer position, with directions trained
   at those positions on synthetic CruxEval-order pairs. Steer there. This closes the
   order/site confound and tests point 2 above directly.
2. **Recover the StarCoder2 handle as a single direction.** Fit one rank-1 direction at
   the causal blocks (0.37–0.43 depth) rather than the readout layer, and report repair
   in both families.
3. **A third size per family** to separate family from scale in Finding 4.
