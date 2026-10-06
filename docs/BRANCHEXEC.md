# BranchExec: is the branch decided at the `if`, and does that decision drive the answer?

Stages 250–258 · `src/branchexec/` · `jobs/branchexec.csh`, `jobs/branchexec_paper.csh` · tests: `tests/test_branchexec.py`

**Results so far:** [BRANCHEXEC_RESULTS.md](BRANCHEXEC_RESULTS.md) (DeepSeek-Coder 6.7B).

## The question

> When a code model reads a **real** program whose input it already knows, does
> it decide which branch an `if` will take *at the `if` itself*, and does that
> internal decision control the output it predicts?

In short: does the model execute code while reading it, or only when asked for
the answer?

Every claim is about real programs (CruxEval, MBPP, HumanEval, optionally
MBPP+/HumanEval+ or any JSONL source). Synthetic programs are used for exactly
three things, all frozen before real data is touched: the steering/readout
**direction**, the **layer**, and the **dose**.

## Why this design (what earlier results it builds on)

| Earlier result | What it implies here |
|---|---|
| Binding probe: 0.500 floor → 0.984 (R1) | Token-identical pairs with a flipped label give an exact chance floor. Here the pairs are **real** programs; only the input literal changes. |
| Control dependence retired: surface floor 0.927 (ARCHIVE §4.3) | A guard *relation* is syntactic. A guard *outcome* given the input is not: the code is identical across the two outcomes. |
| DAS needs donor/host pairs; mean difference moved 68% (R10) | No DAS. A synthetic difference-in-means direction is the actuator, which is plausible from R10's mean-direction result. |
| J-lens: values and operations are not verbalised (E19, CruxEval stage 244) | No vocabulary anywhere. Readout and steering only. |
| Coordinate swap: dose convexity (ARCHIVE §1.3) | Full dose curves; the primary dose is chosen on synthetic data only. |
| Latent store: capability confound (ARCHIVE §1.4) | Capability gate: only members where the model produces both outcomes unsteered are primary. |
| CruxEval puts the input *after* the code | Under causal attention the `if` cannot know the outcome there. That order becomes the **structural zero**; the main order puts the input first. |

## Design

### Units

- **Site:** an `if` in the entry function (not in a nested function, class or lambda) whose test runs **exactly once** on the input, so one token position stands for one evaluation. `if`s inside loops are allowed when they still run once (`in_loop` flag), but on CruxEval none survive.
- **Member:** (program, input, site), with ground truth from instrumented execution: `taken`, output `o`, and the counterfactual `o_flip` = the output of the program with that one test negated, run on **the same input**.
- **Pair:** two members of one site whose inputs differ in **one literal**, whose branch differs, and whose outputs differ. The base input comes from the dataset's own tests; the edit is found by search and verified by execution.

### Two prompt orders per member

```python
# first (main): input in the signature, read before the body
def f(nums=[3, 1, 4], k=2):
    ...
    if len(nums) > k:      # <- read / steer here
        ...
assert f() ==

# last (CruxEval order, structural zero): the `if` precedes every input token
def f(nums, k):
    ...
assert f([3, 1, 4], 2) ==
```

Token checks, both orders: equal length within a pair, every differing token inside the input span, and every read/steer position identical. Answers must be prefix-stable continuations.

### Read positions

`cond_last` (last token of the test), `colon` (**primary**), `body_first`, `def_colon` (end of signature: has seen the input, is not the branch), `answer` (last prompt token). Input-last order: `cond_last`, `colon`, `answer`.

### Directions (stage 252)

`v = mean(h_taken − h_not_taken)` over synthetic training pairs, per (order, position, layer). The layer `l*` is chosen by synthetic **validation** pair accuracy at `first/colon`. Also kept: a shuffled-label direction (random sign per synthetic pair) and a logistic probe at `l*`.

### Readout metric

**Pair accuracy**: the fraction of pairs where the taken member projects higher on `v`. The code is token-identical within a pair, so any code-only reader scores exactly 0.5.

| Row | Role |
|---|---|
| real pairs, `first/colon`, `l*` | **primary readout** |
| real pairs, `last/colon` | structural zero: states identical, asserted in stage 252 |
| real in-domain grouped-CV DiM | ceiling: how much a real-trained direction gets |
| bag of input tokens, grouped CV | model-free floor from the input literal itself |
| shuffled direction, probe | direction controls |
| all layers × positions | where the outcome appears (descriptive) |

### Behaviour (stage 253)

Teacher-forced greedy checks: is `o` the model's exact greedy answer (input first; input last), and is `o_flip` its answer to the **negated program**? A member is **capable** when both hold. "Exact" means every continuation token is the argmax and the next argmax is a newline, comment or EOS. That is identical to greedy decoding, without generating.

### Steering (stage 254)

Edit `h ← h + α‖h‖·v_l` on blocks `l*−2 … l*+2`, with α ∈ {0.02, 0.05, 0.1, 0.2, 0.4}, pushed toward the branch the member does **not** take. Score:
- **flip rate**: greedy output is exactly `o_flip`;
- **keep rate**: greedy output is exactly `o`;
- **other**: neither;
- **Δ log-odds** of `o_flip` vs `o` relative to unsteered.

| Condition | Where | Rules out |
|---|---|---|
| `semantic` | `if` condition span | — (the test) |
| `reverse` | `if` condition | non-directional damage (should reinforce `o`) |
| `random` | `if` condition | generic perturbation at equal dose |
| `shuffled` | `if` condition | "any synthetic difference" |
| `wrong_site` | def-line colon | site-unspecific effects |
| `answer_site` | answer token | **online vs lazy**: is the decision used at the `if` or at the answer? |
| `actuator_cond` / `actuator_answer` | `if` / answer | token pushing: γ⊙(W_U[o_flip token] − W_U[o token]) at the first differing answer token, so this control *knows the answer* |

A single program-agnostic direction producing each program's own multi-token `o_flip` cannot be a token push; the actuator conditions quantify that.

The primary α maximises flip(semantic) − flip(random) on **synthetic validation**. Primary population: real, capable members; all real members are reported alongside.

### Stage 255

`report.md` combines everything, plus the model's own errors: when the unsteered answer is `o_flip`, does the readout at the `if` already point to the wrong branch?

## Reading the outcome

| Readout at `if` (first order) | Steering at `if` | Steering at answer | Reading |
|---|---|---|---|
| > 0.5, zero control exact | flips beyond random, shuffled and actuator | — | **online execution**: the model decides the branch while reading, the answer uses that decision, and the synthetic direction is the one real code uses |
| > 0.5 | fails | works | decided online, **recomputed** at the answer |
| ≈ 0.5 at `if`, > 0.5 at answer | — | works | **lazy execution** |
| synthetic ≈ 0.5, real ceiling > 0.5 | — | — | a measured **synthetic-to-real gap** |

Stage 255 prints which row the numbers resemble, using loose labelling thresholds (0.6 readout, +0.05 flip margin); the tables are the evidence. Small effects are still reported.

## Runbook (cluster)

### 0. Once: caches

```tcsh
source jobs/common.csh
$PYTHON -c "from datasets import load_dataset as L; L('cruxeval-org/cruxeval'); L('google-research-datasets/mbpp','full'); L('openai/openai_humaneval')"
$PYTHON -c "from src.models.loader import ModelConfig, ModelLoader; l = ModelLoader(ModelConfig.from_registry('deepseek-coder-6.7b', device='cpu')); l.tokenizer"
# optional, more inputs per program:
$PYTHON -c "from datasets import load_dataset as L; L('evalplus/mbppplus'); L('evalplus/humanevalplus')"
$PYTHON -m pytest tests/test_branchexec.py -q     # CPU, ~10 s
```

The model weights download on the first GPU stage. Afterwards, `setenv HF_HUB_OFFLINE 1`.

### 1. Smoke run (do this first)

```tcsh
screen -S bx_smoke
setenv SMOKE 1; jobs/branchexec.csh
```

This uses 150 synthetic programs, 60 programs per real source, 30+30 steered members and 2 doses. Check `results/branchexec/deepseek-coder-6.7b/smoke/build/summary.json` (real pairs > 0) and `audit.csv`.

### 2. Full run

```tcsh
screen -S bx_full
unsetenv SMOKE; jobs/branchexec.csh
```

Add MBPP+/HumanEval+ by editing the stage-250 line in the job, or run it directly:

```tcsh
$PYTHON scripts/250_branch_build.py --output results/branchexec/deepseek-coder-6.7b/plus/build \
    --real cruxeval,mbpp,humaneval,mbppplus,humanevalplus
```

Then run stages 251–255 with the same paths (each stage's flags are in `jobs/branchexec.csh`). Run one model at a time. Replicate with `setenv MODEL deepseek-coder-1.3b` or `starcoder2-3b`.

### Per stage

| Stage | Device | Reads | Writes |
|---|---|---|---|
| 250 build | CPU (subprocess execution, `--workers`) | datasets, tokenizer | `members.jsonl`, `pairs.jsonl`, `audit.csv`, `summary.json`, `examples.md` |
| 251 extract | GPU | 250 | `acts_{order}_{position}.npy` (members × layers × d, fp16), `meta.json` |
| 252 readout | CPU | 250, 251 | `directions.npz`, `readout_layers.csv`, `ceiling.csv`, `selection.json`, `report.md` |
| 253 behaviour | GPU | 250 | `behaviour.csv`, `behaviour_summary.csv` |
| 254 steer | GPU, resumable (`--resume`) | 250, 252, 253 | `steer_long.csv`, `selection.json`, `examples.md`, `parts/` |
| 255 report | CPU | all | `report.md`, `dose_curves.csv`, `branchexec.png` (the job writes v2 to `report_v2/`) |
| 256 link | CPU | 250–253 | `link_pairs.csv`, `link_within_branch.csv`, `link.json`, `report.md` |
| 257 repair | GPU, resumable | 250, 252, 253 | `repair_long.csv`, `repair_summary.csv`, `repair.json`, `report.md`, `examples.md` |
| 258 compare | CPU | several runs | `compare.csv`, `compare.md`, `compare.png` |

Every stage writes `gates.json` with artifact digests, refuses a non-empty output directory, and refuses inputs whose upstream gate did not pass or whose files changed.

### Paper run: link, repair, three models

```tcsh
screen -S bx_paper
jobs/branchexec_paper.csh
```

This runs DeepSeek 6.7B, DeepSeek 1.3B and StarCoder2-3B (bfloat16), one after another.
`jobs/branchexec.csh` skips every stage whose `gates.json` says `passed`, so the existing
6.7B run only gains stage 256 (link), stage 257 (repair) and the v2 report. The other
models run the full pipeline. The job ends with stage 258 into
`results/branchexec/compare_<timestamp>/`. A subset can be run with
`setenv MODELS "deepseek-coder-1.3b"`.

What to bring back (all small):
- per model: `results/branchexec/<model>/full/{readout,behaviour,link,repair,report_v2}/`
  and `steer/{steer_long.csv,selection.json,examples.md}`;
- `results/branchexec/compare_*/`.

Leave out `extract/acts_*.npy` (gigabytes) and `steer/parts/`, `repair/parts/`.

**Stage 256** compares the frozen readout with the model's own branch choice without
the branch-bias confound:
- pair accuracy within pair categories (`tracks`: right branch for both inputs;
  `same_branch`: one branch for both; `inverted`: wrong branch for both);
- an AUROC computed separately within taken and within not-taken members.

**Stage 257** takes every real member whose unsteered answer is the wrong branch's
output and pushes toward the **true** branch, at the `if` and at the answer. Controls:
away-pushes, random directions, the shuffled direction, and an answer-token actuator
that knows the right token. Repair means the greedy answer becomes exactly correct.
Doses are 0.05–0.8, nothing is selected, and stage 258 quotes α = 0.4, fixed in advance.

### Resources (deepseek-coder-6.7b, fp16)

- **Disk, stage 251:** about 2.2 MB per member (8 reads × 33 layers × 4096 × 2 bytes). For example, 4,000 members ≈ 9 GB.
- **Stage 254 cost:** about 82 teacher-forced rows per member (8 conditions × 5 doses × 2 continuations + baseline). That is roughly 30–60 min on one A100 for ~1,500 members; `--gated-only` steers only capable members.
- **Stage 250:** a few minutes with 8 workers. It executes dataset code in isolated `python -I` subprocesses with a 2 s per-call alarm and a 2 GB address-space limit. Run it on a node where executing benchmark code is acceptable.

### Expected yield and what to do if real pairs are few

On a local sample of 50 CruxEval programs, 3 sites (4 pairs) survived. CruxEval is loop-heavy (29/50 programs have no usable `if`), and flipping a branch often crashes or leaves the output unchanged (`member:negated_program_invalid_or_same_output`). The full 800 should give roughly 50 sites. MBPP and HumanEval inputs branch more often. If `summary.json` shows fewer than ~150 real pairs, add `mbppplus,humanevalplus` or any JSONL source:

```json
{"id": "p1", "code": "def g(xs, k):\n    ...", "entry": "g", "inputs": ["[1, 2, 3], 2"], "outputs": ["[1, 2]"]}
```

`outputs` is optional; when present, it must match execution or the input is dropped (`input:recorded_output_mismatch`).

## What this cannot claim

- **Observational vs causal.** The readout rows show the outcome is *represented*. Only the steering rows, against their controls, speak to *use*.
- **Scope.** Only `if` statements that run exactly once, and inputs a single literal edit can flip. Branches inside loops that run many times are out of scope by construction.
- **Pinned floor.** Pair accuracy is pinned at 0.5 for code-only readers. The model-free input-token reader measures how much the input literal alone predicts; read the primary result against it.
- **What a null means.** A steering null at the tested band and doses is not proof that the decision is unused elsewhere. The answer-site condition and the layer profile are there to locate it.
- **Model.** DeepSeek-Coder 6.7B base by default. Other registry models run unchanged.
