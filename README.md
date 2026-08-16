# Tracing Semantic State in Code LLMs

**Do code language models build representations of program *semantics* —
which definition a name refers to, where a value flows, what guards a statement —
that are distinct from lexical and syntactic regularity? And does the model's own
computation *causally use* them?**

---

## 1. The problem, and the one idea that makes it tractable

A code model that tracks only surface form can look competent: identifiers
usually keep their meaning, related statements are usually written near each
other, and indentation usually exposes control structure. So a probe that
recovers binding or data flow from hidden states may be recovering the model's
computation — or may be recovering the text, which the hidden states also
contain.

The usual response is to estimate how much a shortcut could explain and subtract
it. Code allows something better: **make the shortcut carry no information at
all.** Consider two programs that differ in one character:

```python
x = 3                  x = 3
def f():               def f():
    y = 7                  x = 7          ← one character
    return x               return x
#   → 3                    → 7
```

The token we ask about is at the same index in both. The words around it are the
same. The distance to everything is unchanged. Only the answer flips. A
predictor with access to nearby tokens and distances is therefore right
**exactly 0.500 of the time — by construction, not by estimate.** Everything
above that floor is something the model computed.

That construction is the project's central asset. It is what E2, E3, E11 and E13
are built on, and its absence is why E4 was demoted.

## 2. The three questions, and why they need different evidence

| | Question | What settles it | Status |
|---|---|---|---|
| **Representation** | Is the relation *present*, beyond what the text predicts? | a decoder beating a floor no surface feature can exceed | **established** |
| **Robustness** | What is it made *of*? What destroys it? | change form with meaning fixed, and the reverse | **established** |
| **Causal use** | Does the model's own computation *read* it? | an intervention changing the relation and nothing else | **open** |

Conflating these is the source of every claim this project has withdrawn.

## 3. What we found

**Representation — yes, and it is built with depth.** Binding is *absent* at the
input (exactly 0.500), constructed within the first few transformer blocks,
peaks near **0.984** in the middle of the network, and is partly shed before the
output. Replicated at 1.3B and 6.7B at the same relative depth. Def–use follows
the same profile with mild distance decay.

**Robustness — it breaks on difficulty, not on distance or spelling.** A
thousand tokens of inert comments costs almost nothing (0.921); the same length
of filler that *reuses the tracked names* drives it to chance. Renaming every
identifier leaves middle layers at 0.85–0.90 while pushing the embedding layer
*below* chance. Control-flow flattening is the real limit (0.750).

**Causal use — open, after four attempts.** This is where the work is:

```
E7  whole-state patching   → transports the input too          [claim retired]
E10 output-aligned readout → instrument fine, both uses failed  [archived]
E11 rank-2 coordinate swap → below the site's causal dose       [NO-GO, retracted]
E12 latent store transfer  → bottlenecked on arithmetic         [parked]
E13 binding interchange    → H0–H3 pass; H4/H5 running
```

Each failed for a *different, nameable* reason, and each reason constrained the
next design. That sequence is the project's methodological content, not an
embarrassment to be hidden — `docs/ARCHIVE.md` records all four.

**E13, the current experiment**, removes both earlier failure modes at once: it
needs no arithmetic (the model returns a variable), and its intervention has no
dose parameter (an interchange installs whatever the donor run holds). Its
falsification is a 2×2 in which the *same* binding flip demands **opposite token
movements** in the two value assignments — so a subspace encoding the answer is
refuted rather than merely unsupported. So far H0–H3 pass on 6.7B with 400 base
programs; H4/H5 await a re-run after a bug in one control.

## 4. Where to read what

| Document | What is in it |
|---|---|
| **[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)** | the catalogue — every experiment, its question, method, controls and result, in the order the research took |
| **[docs/RESULTS.md](docs/RESULTS.md)** | what is established, what is not, and what this project does *not* claim |
| **[docs/ARCHIVE.md](docs/ARCHIVE.md)** | every retired or abandoned design, with the reason and the lesson |
| **[docs/METHODS.md](docs/METHODS.md)** | how — starting with §0, what "semantic" means here |
| **[docs/PIPELINE.md](docs/PIPELINE.md)** | every stage, its command, its outputs |
| **[docs/RUNBOOK_E13.md](docs/RUNBOOK_E13.md)** | the active experiment, step by step, with what to inspect at each gate |
| [docs/SETUP.md](docs/SETUP.md) | installation and known pitfalls (the tokenizer!) |
| [docs/design/](docs/design/) | design rationale per track; `archive/` holds the parked ones |

## 5. How it works

```
generate programs      run the model once     read the state back    intervene
with exact ground  →   save hidden states  →  train linear probes →  on frozen
truth (static AST      at chosen layers       + honest controls      artifacts
 or execution)
```

Small Python programs whose semantic structure is **known exactly by
construction**; ground truth from static analysis or from *executing* them;
every def/use/guard event mapped to its exact token position through verified
AST-span→offset alignment. A frozen model reads each program once and the hidden
states are saved. Then deliberately **low-capacity linear probes** decode the
relations, with controls that kill the cheap ways to score high.

*Why synthetic, why linear, why one pass:* per-token labels must be exact
(static analysis on real code is approximate, and its errors become label
noise); a linear readout is the standard operationalisation of "explicitly
represented" — a stronger probe would measure the probe, not the model; and one
GPU pass decouples scarce GPU time from frequent CPU-only analysis. Full
rationale in [docs/METHODS.md](docs/METHODS.md).

## 6. The pipeline

Numbered stages, one CLI each in `scripts/`, each writing a manifest (git SHA,
args, wall time) to `results/manifests/`. Extract on GPU once; everything else
is CPU and re-runnable.

```
00 generate data      CPU   programs + context variants + minimal pairs + obfuscation ladder + real sample
10 extract            GPU   one forward pass per (model, dataset) → activation store
20 static probes      CPU   E1–E4 (+E8): grouped CV, controls, frozen probe checkpoints
30 context            CPU   E5: frozen probes on filler variants
31 obfuscation        CPU   E9: frozen probes on the execution-verified ladder
50 causal patching    GPU   E7: patch clean→corrupted, layer × position
60 J-lens validation  GPU   E10-0: instrument check — a GATE
70–74 J-space         CPU/GPU  E11: counterfactual pairs → frozen lens → readout → swap → go/no-go
90 paper assets       CPU   every table and figure, regenerated from CSVs alone

── archived tracks, still runnable ──
40 lead time          GPU   E6         · 61/62 J-lens taint / control-dep   E10-2, E10-3
80–89 store           mixed E12 — parked at its behavioural gate

── the active experiments ──
100–108 binding interchange   E13: factorial → verify → behaviour → extract → decode
                              → ceiling → interchange → report → diagnose
                              Six gates (H0–H5); each stage refuses to run on a failed one.
120–124 source→sink audit     E15: 480-program controlled benchmark → extract →
                              clean frozen readout → the E9 ladder, held out → report
                              Four gates (S0–S3). Built and smoke-tested; NOT RUN.
```

Stage status lives in `results/STATUS.yaml`; stage 90 reads it and skips
archived experiments unless run with `--include-archived`.

## 7. What every experiment defends against

| Commitment | Shortcut it kills |
|---|---|
| **Grouped CV** — folds split by source program | rows from one program share hidden vectors; random folds leak train into test |
| **Selectivity control** — identical probe on shuffled labels | accuracy from class priors or per-program regularities |
| **Negative strata reported separately** | a headline averaged over easy negatives |
| **Measured surface baseline** — no hidden states | claiming a result beats "the text" without checking |
| **Verified token alignment** — AST spans → offsets, checked against source | string-matching a name silently mislabels shadows, which is what E2 measures |
| **Cross-validated ground truth** — `beniget`, execution, an independent interpreter | labels wrong the same way in train and test look like signal |
| **Cluster bootstrap** over programs; controls paired on the same rows | intervals too narrow, in the direction that makes a null look like a finding |
| **Hard gates** — a stage refuses to run on a failed prerequisite | a control silently skipped, which is how E11 lost `probe_basis` |

> One non-obvious hazard: `AutoTokenizer` on transformers 5.x silently
> mis-tokenizes deepseek-coder (`def func` → `['de','ff','unc']`), relabelling
> *every* example without crashing. The loader refuses any tokenizer that fails
> an exact code round-trip. See [docs/METHODS.md](docs/METHODS.md) §3b.

## 8. Repository map

```
src/
  data/        generators (E1–E9 programs, E11 pairs, E13 factorials), alignment,
               obfuscation ladder, execution + reference-interpreter ground truth
  graphs/      AST / def-use / CFG / control-dependence extraction (ground truth)
  models/      loading (round-trip guard), hooks, patching, J-lens, DAS interchange
  probes/      linear probe, grouped CV + controls, dataset builders
  experiments/ one module per experiment family
  analysis/    metrics, tables, figures, cluster bootstrap
scripts/       numbered stage CLIs (00–108)
jobs/          csh scripts per GPU stage (run under screen; no scheduler)
configs/       model registry + canonical experiment settings
results/       STATUS.yaml (what each experiment currently claims) + tables, figures, manifests
docs/          EXPERIMENTS · RESULTS · ARCHIVE · METHODS · PIPELINE · RUNBOOK_E13 · SETUP · design/
tests/         362 CPU-only tests (alignment exactness, CV leakage, strata, invariants,
               interchange algebra, gate refusal, obfuscation semantics,
               source→sink label recovery, …)
```

## 9. Quickstart

```bash
conda create -n semflow python=3.11 -y && conda activate semflow
pip install -e ".[dev]"
make test                     # 362 CPU-only tests
make smoke                    # tiny end-to-end run on this machine (~15 min, MPS)

# the foundation (Phase I + II)
python scripts/00_generate_data.py --model deepseek-coder-1.3b --real
make extract probes context obfuscation assets MODEL=deepseek-coder-1.3b

# the active experiment (Phase III)
make binding-pilot            # then read results/binding/*/e13_report.md
```

## 10. Models

| Model | Role | Why |
|---|---|---|
| `deepseek-coder-1.3b-base` | development, smoke, pilots | runs on Apple-Silicon MPS; full pipeline in minutes |
| `deepseek-coder-6.7b-base` | main results | strong open code model; one cluster GPU in fp16 |
| `starcoder2-3b` | optional replication | different corpus and architecture family |

Base (non-instruct) models on purpose: the object of study is the representation
built during code pretraining, not chat behaviour.

## 11. Intended contributions

1. **Layer-resolved maps** of where binding and def–use are linearly decodable,
   against a floor pinned to chance *by construction* rather than estimated.
2. **A failure surface** for that representation: robust to distance and to
   identifier spelling in the middle layers, fragile under scope interference
   and control-flow flattening — it breaks when the underlying problem gets
   harder, not when the text gets longer.
3. **A causal test** at the site where a binding is resolved, whose
   falsification refutes an answer-direction account rather than assuming it
   away (E13, in progress).
4. **Methodology**, from four failed interventions: floors pinned by
   construction; positive controls matched in kind *and* in scale; dose-response
   for any low-rank edit; and hard gates so a missing control is refused rather
   than skipped.

Out of scope here: reasoning-trajectory probing on instruct models, multi-language
extension. Forward options are laid out in
[docs/design/E13_DIRECTIONS.md](docs/design/E13_DIRECTIONS.md).
