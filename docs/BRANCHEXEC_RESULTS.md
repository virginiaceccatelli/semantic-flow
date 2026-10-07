# BranchExec results: three code models

**Runs:** `results/branchexec/{deepseek-coder-1.3b,starcoder2-3b,deepseek-coder-6.7b}/full` (stages 250–257),
cross-model table `results/branchexec/compare_20261006_2109` (stage 258) · **Design and runbook:** [BRANCHEXEC.md](BRANCHEXEC.md)

This document replaces the reading in the 6.7B run's first generated `report/report.md`. Its
"online execution" label came from a labelling rule that was too loose, and its error
analysis was confounded by the model's branch bias. Both are fixed in `report_v2/` and
stage 256. Every number below comes from the committed CSVs.

---

## The question

> When a code model reads a **real** program whose input it already knows, does it
> compute which branch an `if` will take, and does its predicted output **use** that?

The question separates *representation* (the information is in the hidden state) from
*utilisation* (the answer depends on it). For variable binding both hold in this
repository: probes reach 0.98, and rank-1 DAS installs the binding on 100% of
held-out cases (R1, R10). Here the same question is asked of an execution fact, on real
code, in three models.

## The experiment

From MBPP, HumanEval and CruxEval we take every `if` that runs exactly once on a test
input. We then search for a one-literal change to that input that makes the same `if`
take the other branch and changes the output; execution verifies each candidate. The
two programs in such a **pair** are token-identical; only the input literal differs.
The input is written into the signature (`def f(xs=[3, 1]):`), so it is read before the
body.

| | question | instrument | chance / control |
|---|---|---|---|
| **Readout** | is the branch outcome in the state at the `if`? | direction learned **only on synthetic programs** (difference of means), frozen | exactly 0.5 (identical code within a pair) |
| **Behaviour** | does the predicted output follow the branch that runs? | greedy answer vs. the true output and the other branch's output | the model's own branch bias |
| **Link** | in pairs the answer treats identically, is the outcome still computed? | readout split by what the answer does | readout in pairs whose answer follows the input |
| **Steering / repair** | does pushing that state change the answer, specifically? | h ← h + α‖h‖v on 5 blocks around the readout layer, at the `if` or at the answer | away-pushes, random and shuffled directions, an answer-token push that knows the target |

Synthetic programs supply only the direction, the layer and (for stage 254) the dose.
Every claim below is about real programs.

## Data

| model | real pairs (MBPP / HumanEval / CruxEval) | real members |
|---|---|---:|
| DeepSeek-Coder 1.3B | 735 | 1,168 |
| StarCoder2 3B | 743 | 1,184 |
| DeepSeek-Coder 6.7B | 735 (520 / 142 / 73) | 1,168 |

Pairs are rebuilt per model because the token checks depend on the tokenizer. The
synthetic set per model is about 1,300 pairs from 1,200 programs.

---

## Finding 1: all three models compute the branch outcome at the `if`

Frozen synthetic direction, real pairs, at the `if` colon:

| model | layer (depth) | readout at `if` [95% CI] | real-trained ceiling | readout at answer (input first / CruxEval order) |
|---|---|---|---:|---|
| DeepSeek 1.3B | 8 (0.38) | 0.595 [0.551, 0.638] | 0.682 | 0.556 / 0.512 |
| StarCoder2 3B | 13 (0.47) | 0.678 [0.639, 0.721] | 0.673 | 0.556 / 0.483 |
| DeepSeek 6.7B | 16 (0.53) | **0.746 [0.700, 0.791]** | 0.796 | 0.582 / 0.581 |

Controls (6.7B; the other models behave the same way):

| control | pair accuracy |
|---|---:|
| `if` before the input (CruxEval order) | 0.500 (\|Δh\| = 0) |
| model-free reader of the input tokens | 0.473 |
| shuffled-label direction | 0.482 |
| end of signature (input read, body not yet) | 0.487 |

- **Synthetic transfers to real code.** The synthetic direction recovers most of what a
  direction trained on real pairs finds: 83% of the above-chance signal for 6.7B, and
  for StarCoder2 it matches the real ceiling.
- **The outcome appears where the condition is read**, not when the input is read, and
  it fades by the answer position.
- **It strengthens with size within DeepSeek:** 0.595 (1.3B) → 0.746 (6.7B).

## Finding 2: none of the three models' answers use it

Unsteered greedy answers on real members, input first:

| model | accuracy (input first / CruxEval order) | answer follows the branch that runs* | `if`-body share* | same branch for both inputs of a pair (bias alone predicts) | pairs whose answer follows the input |
|---|---|---:|---:|---|---:|
| DeepSeek 1.3B | 0.315 / 0.375 | **0.533** | 0.626 | 0.926 (0.532) | 0.055 |
| StarCoder2 3B | 0.312 / 0.416 | **0.508** | 0.651 | 0.913 (0.545) | 0.066 |
| DeepSeek 6.7B | 0.311 / 0.485 | **0.511** | 0.694 | 0.877 (0.576) | 0.110 |

\* among answers that equal one of the two branch outputs (6.7B: 710 of 1,168; similar
counts for the others).

- **Chance branch-following.** When an answer is one of the two branch outputs, it
  follows the branch that runs at chance, in every model and in every dataset (6.7B:
  MBPP 0.506, HumanEval 0.532, CruxEval 0.508).
- **Input ignored.** Models default to the `if` body. About 9 in 10 pairs get the same
  branch for both inputs, far above what that bias alone predicts, so the answer mostly
  ignores the input.
- **Size raises representation, not use.** Within DeepSeek, going from 1.3B to 6.7B
  raises the readout from 0.60 to 0.75 but leaves branch-following at chance (0.53 →
  0.51).

## Finding 3: the outcome is computed even when the answer ignores it

Readout pair accuracy split by what the model's answers do (stage 256, readout layer, `if` colon):

| model | pairs whose answer follows the input | pairs given one branch for both inputs |
|---|---|---|
| DeepSeek 1.3B | 0.591 (n = 22) | 0.598 (n = 373) |
| StarCoder2 3B | 0.714 [0.538, 0.889] (n = 28) | 0.661 [0.606, 0.719] (n = 387) |
| DeepSeek 6.7B | 0.791 [0.643, 0.927] (n = 43) | 0.755 [0.690, 0.813] (n = 343) |

In all three models the state at the `if` separates the two inputs about equally well
whether the answer follows the input or treats both inputs alike; every pair of
intervals overlaps. **The failure is in using the computed outcome, not in computing
it.**

**A second, correlational result (6.7B only):** among members whose true branch is the
same, the projection at the `if` also predicts which branch the model will *answer*.

| model | within-branch AUROC |
|---|---|
| DeepSeek 6.7B | **0.707** (not taken 0.714 [0.621, 0.795]; taken 0.699 [0.578, 0.779]) |
| DeepSeek 1.3B | 0.565 (CIs include 0.5) |
| StarCoder2 3B | 0.522 (CIs include 0.5) |

The input-driven part of the state differs within same-branch pairs while the answer
does not change. So this link must come from a component that varies *between
programs* (for example, how strongly a program invites its `if` body), not from the
input-driven outcome itself.

## Finding 4: the state is causally connected to the answer, at a model-specific site

**Flip steering** (stage 254): mean log-odds shift toward the other branch's output, all
real members, α = 0.4.

| model | at `if`: semantic / reverse / random | at answer |
|---|---|---:|
| DeepSeek 1.3B | +0.080 / −0.063 / −0.001 | +0.044 |
| StarCoder2 3B | **+0.330 / −0.287** / −0.002 | +0.214 |
| DeepSeek 6.7B | +0.100 / −0.089 / +0.006 | **+0.883** |

**Repair** (stage 257): members whose answer is the wrong branch's output are pushed
toward the **true** branch. Share whose answer becomes exactly correct, α = 0.4:

| model | n | at `if`: toward true / away / random / shuffled | at answer: toward true / away / random | answer-token push (knows the target) |
|---|---:|---|---|---:|
| DeepSeek 1.3B | 322 | **0.130** [0.082, 0.186] / 0.040 / 0.034 / 0.047 | 0.289 / 0.258 / 0.093 | 0.832 |
| StarCoder2 3B | 358 | **0.215** [0.150, 0.290] / 0.006 / 0.028 / 0.115 | 0.162 / 0.179 / 0.089 | 0.722 |
| DeepSeek 6.7B | 347 | 0.020 / 0.014 / 0.003 / 0.003 | **0.438** [0.342, 0.535] / 0.199 / 0.040 | 0.648 |

Log-odds shift toward the true output at α = 0.4, with 95% CIs:

| model | site | toward true | away |
|---|---|---|---|
| StarCoder2 | `if` | +0.41 [0.32, 0.50] | −0.29 [−0.37, −0.21] |
| DeepSeek 1.3B | `if` | +0.14 [0.08, 0.19] | −0.01 [−0.06, 0.04] |
| DeepSeek 6.7B | answer | +0.82 [0.33, 1.29] | −1.13 [−1.75, −0.52] |
| DeepSeek 1.3B | answer | +0.38 | +0.38 (not directional) |
| StarCoder2 | answer | +0.01 | −0.26 (not directional) |

- **At the `if`, steering is directional in every model.** Pushing toward a branch
  moves the answer toward that branch's output and the reverse push moves it away,
  monotonically with dose. Random, wrong-site and answer-token pushes at the `if` do not.
- **Where the edit becomes usable depends on the model.**
  - In **StarCoder2** and **DeepSeek 1.3B**, the `if` state repairs wrong-branch answers
    (21.5% and 13.0%, against 0.6–4.0% for away and random). At the answer, those two
    models respond the same to "toward" and "away" pushes, so that effect is generic
    disruption.
  - In **DeepSeek 6.7B**, the `if` state repairs almost nothing (2.0%), while the answer
    position repairs 43.8%, directionally (away 19.9%, random 4.0%).
- **Within DeepSeek, size moves the leverage later.** From 1.3B to 6.7B, repair at the
  `if` drops (13.0% → 2.0%) and directional repair appears at the answer. This is one
  family and two sizes, so it is suggestive, not established.
- **Two cautions.**
  - StarCoder2's shuffled direction recovers about half of its `if` repair (11.5% vs
    21.5%), so part of that effect is not specific to the branch labels. The away
    control (0.6%) is the cleaner comparison.
  - All answer-site edits break down at α = 0.8 (repair drops to 0–1%, outputs stop
    being answers). The qualitative `repair/examples.md` files were generated at α = 0.8
    and should not be quoted.

## What the findings say together

Across two families and three sizes, code models **compute which branch a real `if`
takes at the `if`** (Finding 1). They **compute it equally whether or not the answer
uses it** (Finding 3). Yet their **answers follow the true branch at chance** (Finding 2).
The state is causally connected to the answer: pushing it moves the answer in the
pushed direction in every model, and in the smaller models it repairs a share of wrong
answers. In 6.7B, however, the usable leverage sits at the answer position, not at the
`if` (Finding 4).

For binding, the represented state is the one the answer uses. For branch outcomes, the
state is computed but largely bypassed. In the larger DeepSeek model it is computed more
strongly, and the decision site appears to move later.

## Limits

- **The main open caveat is the input order.** Branch-following was scored only with the
  input written into the signature. That order is needed for the clean readout control,
  but it is not the models' native format. Accuracy in the CruxEval order is higher and
  grows with size (0.375 → 0.416 → 0.485), while input-first accuracy is flat at 0.31.
  Whether answers also ignore the branch in the natural order is untested.
- **Coverage.** Only `if`s that run once, with inputs a single literal edit can flip;
  CruxEval contributes few pairs.
- **What the readout and link show.** The readout shows the outcome is linearly present.
  The within-branch AUROC (6.7B) is correlational and program-level.
- **Underpowered gated steering.** The capability-gated flip analysis has only 36–51
  members per model, because models rarely answer both a program and its negated
  version correctly. Repair (stage 257) does not need that gate, which is why it is the
  causal result reported here.
- **Scale.** Three models from two families. The DeepSeek pair (1.3B, 6.7B) is the only
  within-family size contrast.

## Next (implemented: stages 259, 260 and StarCoder2-7B; run with `jobs/branchexec_paper.csh`)

1. **Score the other branch in the CruxEval order** (`o_flip` after `assert f(<input>) ==`).
   This is one extra continuation per member in stage 253: a GPU behaviour rerun with no
   new extraction. It tells whether Finding 2 holds in the models' native format, and it
   is the most important remaining check.
2. **Locate the decision site in 6.7B.** Repeat repair at intermediate positions (body
   tokens, the `return` line) and at single blocks instead of a 5-block band, to see
   where the branch decision that reaches the answer is actually formed.
3. **A second within-family size point**, e.g. StarCoder2 7B (already in the registry),
   to test whether leverage moving from the `if` to the answer is a size effect.
