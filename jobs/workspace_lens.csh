#!/bin/csh
# E19 — the PUBLISHED J-lens and R-lens, on the code models.
#
# This is NOT the repository's earlier lens work. The estimator here is the
# reference implementation released with "Verbalizable Representations Form a
# Global Workspace in Language Models" (transformer-circuits.pub/2026/workspace),
# vendored unmodified at third_party/jacobian-lens and imported as `jlens`. The
# R-lens is that same estimator run under the published RelP backward rules
# (alignmentforum.org/posts/nv8oedrnLXKRzNEL9). The repository's older,
# differently-defined lenses now live under names that cannot be confused with
# these: src/models/cotangent_lens.py and src/models/cotangent_lrp.py.
#
#   200  corpus + probe suite     CPU        seconds   tokenizer only
#   201  fit J-lens AND R-lens    GPU        HOURS     the whole cost of E19
#   202  the seven-check gate     GPU        minutes   REQUIRED before 203-205
#   203  J vs R vs logit readout  GPU        minutes   one forward per item
#   204  causal ablation          GPU        minutes   ~20 forwards per item
#   205  tables, figures, report  CPU        seconds
#
# ** RUN `make lens-fit-dry` FIRST. ** Stage 201 costs about `2 * d_model`
# forward passes per prompt per lens, which is 1.4 PFLOP/prompt at 1.3B and
# 14 PFLOP/prompt at 6.7B. The dry run prints the backward-pass count, the PFLOP
# total, the resume-checkpoint size and the host-RAM hint without loading any
# weights, so a run is sized before it is launched rather than after.
#
# ** HOST RAM, NOT JUST VRAM. ** The per-layer Jacobians are accumulated on the
# CPU in float32: `(n_layers - 1) * d_model^2 * 4` bytes for the running sum,
# again for the per-prompt matrices, and again while the checkpoint is written.
# At 6.7B that is 2.0 GB each, so budget ~8 GB of host RAM. This is the one
# resource that is easy to under-request and it fails late.
#
# ** RUN THE MODELS ONE AT A TIME. ** `ModelLoader` loads with device_map="auto",
# so a second model resident on the same card silently offloads the first's tail
# to meta placeholders — and the tail is `model.norm` and `lm_head`, exactly what
# the readout needs.
#
# bfloat16, not float16: this is a BACKWARD pass through up to 30 blocks, and
# fp16 gradients underflow. bfloat16 is the checkpoints' native dtype and keeps
# float32's exponent range. Set DTYPE=float32 if stage 202's W4 or W5f looks
# numerically marginal; it doubles the VRAM.
#
# Stage 201 checkpoints every 10 prompts and resumes automatically, so a killed
# job is restarted by re-running this script.
#
# Run inside a screen session, sequentially, WITH A LOG. `screen -L` is worth the
# extra flag: this is a multi-hour job, and when a stage fails the useful part of
# the output is minutes of scrollback that a detached screen will not keep.
#
#   screen -L -Logfile lens-1.3b.log -dmS lens-1.3b \
#       env MODEL=deepseek-coder-1.3b HALVES=--halves jobs/workspace_lens.csh
#   # wait (screen -ls), then:
#   screen -L -Logfile lens-3b.log   -dmS lens-3b   env MODEL=starcoder2-3b       jobs/workspace_lens.csh
#   screen -L -Logfile lens-6.7b.log -dmS lens-6.7b env MODEL=deepseek-coder-6.7b jobs/workspace_lens.csh
#
# On a stage failure the script stops immediately rather than cascading, so the
# real error is the LAST thing in the log, not the first of five tracebacks.
if (! $?MODEL) setenv MODEL deepseek-coder-1.3b
source jobs/common.csh
if (! $?DTYPE)     setenv DTYPE bfloat16
if (! $?LENS_N)    setenv LENS_N 100
if (! $?DIM_BATCH) setenv DIM_BATCH 16
# --halves fits two extra lenses per kind on disjoint corpus halves, which is
# what gate W6 (build repeatability) reads. It triples stage 201, so it is on
# for the 1.3b run only; W6 is a property of the estimator, not of the model,
# and reporting it once on the cheapest model is the honest way to buy it.
if (! $?HALVES)    setenv HALVES "--no-halves"

set CORPUS = "data/lens_corpus/pile10k-n${LENS_N}.jsonl"
set SUITE  = "data/lens_eval/code-semantics-${MODEL}.jsonl"
set OUT    = "results/workspace_lens/${MODEL}"

echo "=== stage 200: fitting corpus + probe suite — CPU ==="
$PYTHON scripts/200_lens_corpus.py --model "$MODEL" --n-prompts "$LENS_N" --corpus pile

echo "=== stage 201 preflight: can this host run the fit? (no weights loaded) ==="
# Stage 200 above needs only a tokenizer, so it succeeds on a host where the fit
# cannot run at all — a missing `jlens` install is the usual cause. Fail here,
# loudly and in seconds, rather than after the queue wait.
$PYTHON scripts/201_lens_fit.py --model "$MODEL" --corpus "$CORPUS" --check-env
if ($status != 0) then
    echo "*** preflight FAILED — not starting the fit. Fix the rows marked FAIL."
    exit 1
endif

echo "=== stage 201 cost estimate (no weights loaded) ==="
$PYTHON scripts/201_lens_fit.py --model "$MODEL" --corpus "$CORPUS" \
    --dim-batch "$DIM_BATCH" --dry-run

echo "=== stage 201: fit the J-lens and the R-lens — GPU, HOURS ==="
nvidia-smi --query-gpu=memory.used,memory.total --format=csv 2>/dev/null
$PYTHON scripts/201_lens_fit.py --model "$MODEL" --corpus "$CORPUS" \
    --dim-batch "$DIM_BATCH" --dtype "$DTYPE" $HALVES
# A failed fit MUST abort. Everything downstream reads `lens.pt`, so without
# this guard a dead stage 201 produces four more tracebacks and its own — the
# only one that says anything — scrolls off the top of the log.
if ($status != 0) then
    echo ""
    echo "*** stage 201 FAILED. Stopping here: 202-205 all read lens.pt and"
    echo "*** would only produce FileNotFoundError on top of the real error."
    echo "*** The real error is immediately above this line."
    if (-e "${OUT}/j-lens/fit_checkpoint.pt") then
        echo "*** A fit checkpoint exists, so the fit STARTED and died partway."
        echo "*** Re-running this script resumes from it; no work is lost."
    endif
    exit 1
endif

echo "=== stage 202: the GATE — GPU ==="
# Deliberately NOT chained with && from here: if the gate fails we still want
# the readout tables on disk to diagnose WHY, clearly marked as ungated by the
# gate CSV that 205 embeds at the top of the report.
$PYTHON scripts/202_lens_validate.py --model "$MODEL" --corpus "$CORPUS" \
    --suite "$SUITE" --dtype "$DTYPE"
set GATE = $status

echo "=== stage 203: J-lens vs R-lens vs logit lens — GPU ==="
$PYTHON scripts/203_lens_readout.py --model "$MODEL" --suite "$SUITE" --dtype "$DTYPE"

echo "=== stage 204: causal ablation of the read directions — GPU ==="
$PYTHON scripts/204_lens_ablate.py --model "$MODEL" --suite "$SUITE" \
    --readout "${OUT}/readout/workspace_lens_rows.csv" --dtype "$DTYPE"
# Not fatal — the readout tables stand on their own — but it must be SAID.
# Unguarded, a dead stage 204 let stage 205 render a report with no ablation
# section, which looks exactly like a report that never asked for one.
set ABLATE = $status
if ($ABLATE != 0) echo "*** stage 204 FAILED (exit $ABLATE) — the report will have no ablation section."

echo "=== stage 205: tables, figures, report — CPU ==="
$PYTHON scripts/205_lens_report.py --model "$MODEL"

echo ""
if ($GATE != 0) then
    echo "*** GATE FAILED (stage 202 exit $GATE). ***"
    echo "The tables below EXIST but are NOT interpretable. Read"
    echo "${OUT}/validate/workspace_lens_gate.csv first and fix the failing"
    echo "check before quoting any number from this run."
else
    echo "Gate passed. Read ${OUT}/workspace_lens_report.md."
endif
echo "The gate table is reproduced at the top of that report, so a reader"
echo "never sees the pass@k tables without seeing what certified them."
