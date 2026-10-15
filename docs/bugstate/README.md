# BugsInPy: intermediate Boolean values inside a reasoning model

This is the small experiment discussed in the chat: execution labels → layer-wise
linear probes → swap the probe component → two plots and a report. It uses real
BugsInPy tests and code. No locally generated examples enter the research dataset.
Small artificial fixtures exist only in software tests.

## Run on the cluster

From the repository root, use Python 3.10 or newer for this pipeline:

```bash
bash jobs/bugstate.sh setup
bash jobs/bugstate.sh test

# CPU allocation with network access. Historical Python versions are fetched by uv.
bash jobs/bugstate.sh prepare

# Download before entering a network-isolated GPU allocation.
bash jobs/bugstate.sh download

# Inside your GPU allocation:
bash jobs/bugstate.sh run
```

Default model: `Qwen/Qwen3-4B`, BF16 on CUDA, thinking enabled. Start with a GPU with
24 GB or more memory. This is a planning estimate, not a measured memory guarantee;
long prompts/answers use additional memory. The script does not request resources
or choose a scheduler partition for you. CPU preparation should run in a CPU job,
not on a login node; it installs historical dependencies and can take substantial
time/disk space. Git objects are shared per project, but virtualenvs are separate.

The preparation scans all BugsInPy projects in deterministic order. To try a small
preparation pilot first:

```bash
bash jobs/bugstate.sh prepare --max-bugs-per-project 5
# Then extend the same dataset; completed bugs are not repeated:
bash jobs/bugstate.sh prepare
```

A five-bug pilot is only a setup check and may yield no eligible examples. The GPU
stage requires changed-value pairs from at least **three projects**. A real sample
size cannot be promised before reproducing the historical environments. If the
strict Boolean/function subset is too small, the code stops or explicitly calls
the report a pilot; it does not substitute synthetic programs or silently broaden
the experiment. Review `data/bugstate/preparation.jsonl` and each linked log.
To retry previously excluded bugs after fixing an environment issue, use a fresh
`--data` directory; successful/failed attempts are both cached in the original ledger.

Model/settings can be changed explicitly:

```bash
bash jobs/bugstate.sh run --model Qwen/Qwen3-4B \
  --revision main --max-new-tokens 4096 \
  --output results/bugstate/qwen3-4b
```

Use a model commit hash instead of `main` to pin weights. The model must expose
Hugging Face's `model.layers` decoder layout and support a thinking chat template;
the supplied default is Qwen3. Resuming requires the same model, data, and settings.
Use a new output directory when changing them. Activations and completed generations
are reused on resume. Run only one preparation/run process per output directory.

## Exactly what happens

1. Clone the official BugsInPy metadata and each project's source. Check out both
   recorded commits and give both the fixed-version tests specified by the dataset.
   Install the recorded requirements/setup in separate environments using the
   recorded Python minor version. No modifications to your shell startup files.
2. Run the dataset's existing test commands with `sys.settrace`. Require a failing
   buggy test command and a passing fixed command. Capture literal call arguments,
   intermediate Boolean locals immediately before each line, and literal returns.
3. Find the same function and identical arguments in both runs. Require different
   return values and an unchanged source line visited exactly once in each run.
   Select the first eligible changed Boolean local, excluding function arguments.
   Also select an unchanged Boolean local from that same call when available.
   Keep at most one primary pair and one control pair per bug.
4. Extract the function and its same-module helpers/literal globals/stdlib imports.
   Replay the original captured call twice independently. Require the original
   value and return in both replays. Unsupported dependencies, object arguments,
   methods, generators, and exceptional executions are excluded and logged.
5. Show the model this sufficient code context, captured call, and marked
   observation point. Name the intermediate variable, but never give its label,
   bug/fix status, expected return, candidate answers, or test assertions. Ask for
   the return after reasoning. Overlength prompts are excluded, never truncated.
6. Record each decoder block's output at the final chat-prompt token, before any
   generated reasoning. Fit one standardized L2 logistic probe per block with
   leave-one-project-out evaluation. The probe direction is transformed back to
   raw activation coordinates. Nearby-source and input-only text classifiers use
   the identical splits. All layers are reported, without test-set layer tuning.
7. Generate a clean answer, then generate again after replacing only the donor's
   scalar projection along that layer's held-out probe direction. Do this in both
   directions, at every block. Patch only the final prompt token, once, during
   cache prefill. Generate reasoning freely afterward. Compare against three
   norm-matched random orthogonal edits and unchanged-value swaps.
8. Parse `FINAL: <Python literal>` and measure the change in donor-answer match.
   Missing answers count as incorrect. Also report the subset where both clean
   answers are correct. Sampling uses Qwen's recommended temperature/top-p/top-k
   with the same RNG seed for each clean/intervened generation.

The first implementation uses **Boolean values only**, as agreed. It measures a
representation before reasoning starts, not the whole evolution of the reasoning
trace. There is no synthetic-data generation, SAE training, fine-tuning, or
additional benchmark suite.

## Outputs

In `results/bugstate/qwen3-4b/`:

| File | Content |
|---|---|
| `decodability.png` / `.pdf` | Boolean probe accuracy by layer, with input/source baselines |
| `intervention_effect.png` / `.pdf` | Donor-answer shift by layer: probe, random, unchanged controls |
| `report.md` | Method, coverage, observed results, exclusions and interpretation |
| `decodability.csv` | Probe/baseline scores, project-bootstrap intervals, counts |
| `intervention_effect.csv` | Effects for all pairs and clean-both-correct pairs |
| `cases.csv`, `projects.csv` | Included variables, returns, cases and project counts |
| `*_generations.jsonl`, `interventions.jsonl` | Full generated text and per-example scores |
| `included_pairs.jsonl` | Exact code, arguments, labels, test commands and commit provenance |

Plot summaries average directions/seeds within pairs, pairs within projects, and
then projects equally. Shading bootstraps projects, not correlated activations.
Small coverage and missing controls appear explicitly in the report.

To regenerate figures/report without a GPU:

```bash
bash jobs/bugstate.sh report
```

Interpret decodability as accessibility of information. Selective intervention
effects provide evidence of use, but real bug fixes can change other correlated
properties; a probe direction is not guaranteed to encode only the chosen variable.
A null result does not prove absence of semantic understanding. The strict subset
also limits generalization to large, stateful repository code.

## Validation

```bash
# All checks, including a tiny random-initialized Qwen3 (CPU, no downloads):
bash jobs/bugstate.sh test
# Instrumentation checks alone, with only Python's standard library:
python3 -m unittest discover -s tests/bugstate -p test_data.py -v
```

The tests check before-line tracing, line alignment, exception handling, rejection
of nondeterministic calls, prompt label leakage, probe geometry, norm controls,
zero-edit generation, hook removal, held-out fitting, and report/figure creation.
The tiny model and fixture programs are software tests, not scientific results.
