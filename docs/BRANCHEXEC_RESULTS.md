# BranchExec results: DeepSeek-Coder 6.7B

**Run:** `results/branchexec/deepseek-coder-6.7b/full` (stages 250–255) · **Design and runbook:** [BRANCHEXEC.md](BRANCHEXEC.md)

This document replaces the reading in that run's generated `report/report.md`. That file's
closing "online execution" label came from a labelling rule that was too loose
(2 of 36 flips). Its error-analysis section was also confounded by the model's branch
bias (§3). Those parts are fixed in stage 255 v2, and the confounded analysis moved to
stage 256. The numbers below come from the run's committed CSVs.

---

## The question

> When a code model reads a **real** program whose input it already knows, does it
> compute which branch an `if` will take, and does its predicted output **use** that?

The question separates two properties that earlier work in this repository showed can
come apart: *representation* (is the information in the hidden state?) and
*utilisation* (does the answer depend on it?). For variable binding both hold: the
probes reach 0.98, and rank-1 DAS installs the binding on 100% of held-out cases (R1,
R10). Here the same question is asked of an execution fact, on real code.

## The experiment, in one paragraph

From CruxEval, MBPP and HumanEval we take every `if` that runs exactly once on a test
input. We then search for a one-literal change to that input that makes the same `if`
take the other branch and changes the output; execution verifies each candidate. The
two programs in such a **pair** are token-identical; only the input literal differs.
The input is written into the signature (`def f(xs=[3, 1]):`), so the model reads it
before the body. We ask three things:

1. **Readout.** Can a direction learned **only from synthetic programs** tell the two
   members of a real pair apart at the `if`? Chance is exactly 0.5, since the code is
   identical.
2. **Behaviour.** Does the model's predicted output follow the branch that truly runs?
3. **Steering.** Does pushing the `if` state toward the other branch move the output to
   that branch's output, which execution defines exactly?

## Data

| | programs | sites | pairs |
|---|---:|---:|---:|
| MBPP | 972 | 296 | 520 |
| HumanEval | 164 | 93 | 142 |
| CruxEval | 800 | 44 | 73 |
| **real total** | | **433** | **735** (1,168 members) |
| synthetic (directions, layer and dose only) | 1,200 | 723 | 1,302 |

CruxEval contributes little: most of its programs branch only inside loops, and those
`if`s run more than once.

---

## Finding 1: the branch outcome is computed at the `if` (representation)

Frozen synthetic direction, layer 16 (chosen on synthetic validation), real pairs:

| read position | pair accuracy | 95% CI |
|---|---:|---|
| end of signature (input read, body not yet) | 0.487 | [0.445, 0.529] |
| last token of the condition | 0.671 | [0.621, 0.720] |
| **`if` colon** | **0.746** | **[0.700, 0.791]** |
| first token of the body | 0.712 | [0.671, 0.751] |
| answer position | 0.582 | [0.534, 0.632] |

| control | pair accuracy |
|---|---:|
| same pairs, CruxEval order (`if` precedes the input): states identical, \|Δh\| = 0 | 0.500 exactly |
| model-free reader of the input tokens (grouped CV) | 0.473 |
| shuffled-label synthetic direction | 0.482 |
| frozen synthetic logistic probe | 0.731 [0.684, 0.776] |
| **real-trained direction, grouped CV (ceiling)** | **0.796** |

- **Synthetic transfers to real code.** A direction learned only from templates
  recovers about 83% of the above-chance signal a real-trained direction finds
  (0.246 of 0.296).
- **The outcome is computed where the condition is read, not when the input is read.**
  It is absent at the end of the signature at every layer (≈ 0.49), appears on the
  condition, and peaks at the colon.
- **Depth.** It emerges around layer 9 (0.59) and peaks at layers 13–17 (≈ 0.75), out
  of 32. The layer chosen on synthetic data (16) sits next to the best layer on real
  data (15, 0.759).
- **It fades by the answer.** At the answer position it reads only about 0.58, in both
  prompt orders.

## Finding 2: the answer does not use it (utilisation)

Unsteered greedy answers, real members, input first:

| | value |
|---|---:|
| exact accuracy, input first / input last (CruxEval order) | 0.311 / 0.485 |
| answers equal to one of the two branch outputs | 710 of 1,168 |
| **of those, answers following the branch that actually runs** | **0.511** (363/710) |
| by source: MBPP / HumanEval / CruxEval | 0.506 / 0.532 / 0.508 |
| of those, answers that are the `if`-body branch's output | 0.694 |
| same branch for both inputs of a pair | 0.84 (bias alone predicts 0.58) |

When the model produces one of the two branch outputs, it picks the branch that runs
**at chance**, in all three datasets. It defaults to the `if` body, and mostly gives
both inputs of a pair the same branch. That is far more often than its body bias
alone would produce, so the answer mostly ignores the input.

Together with Finding 1: **the model computes which branch runs, at the `if`, from a
direction learned without any real labels, and its predicted output is at chance with
respect to that branch.**

## Finding 3: steering at the `if` has a real but small effect

Edit h ← h + α‖h‖·v on blocks 14–18, pushing each member toward the branch it does
not take. Mean log-odds shift toward the other branch's output, all 1,168 real members:

| α | semantic at `if` | reverse at `if` | random | wrong site | answer-token push at `if` | semantic at answer |
|---|---:|---:|---:|---:|---:|---:|
| 0.05 | +0.020 | −0.021 | −0.001 | +0.001 | −0.019 | +0.129 |
| 0.1 | +0.040 | −0.039 | −0.001 | +0.002 | −0.036 | +0.265 |
| 0.2 | +0.075 | −0.069 | 0.000 | +0.003 | −0.078 | +0.503 |
| 0.4 | +0.100 | −0.089 | +0.006 | 0.000 | −0.320 | +0.883 |

- **At the `if`: specific but small.** Semantic and reverse move in opposite directions,
  grow with the dose, and do so for both true branches. Random, wrong-site and
  answer-token pushes stay flat or go the wrong way, so it is not generic damage or a
  token push.
  - The effect is small: +1.7 points of flip rate at α = 0.4.
  - Of the 36 members the model handles correctly in both directions, 5 flip at
    α = 0.4, against 0 for random and 2 for the shuffled direction.
- **At the answer: about 9× more leverage.** The same kind of direction flips 18 of 36,
  but the shuffled direction also starts to act at high dose there.
- **Synthetic steering does not replicate.** On synthetic validation the `if` direction
  moves slightly the wrong way (−0.03), and the primary dose α = 0.2 was chosen on a
  one-member difference. Read the full dose curves, not that dose.

## What the three findings say together

The branch outcome is **computed** at the `if` (Finding 1). It is **not reflected** in
the answer (Finding 2). Pushing the `if` state gives only weak control over the answer
(Finding 3). So the model works out which branch runs while reading the code, but that
state barely reaches its prediction. For binding, by contrast, the represented state is
the one the answer uses.

## Limits

- **One model.** DeepSeek-Coder 6.7B base, greedy decoding, prompts with the input
  written into the signature. That format lowers accuracy compared with CruxEval order
  (0.31 vs 0.49). The structural-zero control requires it, but it is not the model's
  native format.
- **Coverage.** Only `if`s that run once, and inputs a single literal edit can flip.
- **Representation is observational.** Pair accuracy shows the outcome is linearly
  present. Whether the model's own wrong answers go with a wrong state at the `if` is
  the job of stage 256; the pooled version in `report.md` is confounded.
- **Underpowered gated steering.** Only 36 of 1,168 members pass the capability gate,
  because the model rarely gets a program *and* its negated version right.

## Next (implemented; run with `jobs/branchexec_paper.csh`)

| Stage | Question | What would change the reading |
|---|---|---|
| 256 link | Is the readout equally good in pairs where the answer follows the input (`tracks`) and pairs given one branch for both inputs (`same_branch`)? Within each true branch, does the state predict which branch the model answers? | Equal readout across categories means the failure is in using the state, not computing it. |
| 257 repair | On the 347 answers giving the wrong branch's output, does pushing toward the **true** branch make the answer exactly correct, beyond random and away-pushes, at the `if` and at the answer? | Repair at the `if` means the state can be used; repair only at the answer locates where the decision is taken. |
| 258 compare | Same pipeline on DeepSeek 1.3B and StarCoder2-3B. | A constant readout with changing branch-following separates computing from using across models. |
