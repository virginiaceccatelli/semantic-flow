#!/bin/csh
# Run from the cluster checkout. Reuses an existing validated published J-lens.
# No model downloads/fits or real-test adaptation are performed by this job.
source jobs/common.csh
if (! $?MONITOR_BASES) setenv MONITOR_BASES 8
if (! $?MONITOR_RUN) setenv MONITOR_RUN v1
if (! $?MONITOR_LAYERS) setenv MONITOR_LAYERS 6,11,20
if (! $?MONITOR_MAX_TOKENS) setenv MONITOR_MAX_TOKENS 2048
set DATA = "data/security_monitor/dev-${MONITOR_RUN}.jsonl"
set OUT = "results/security_monitor/${MODEL}/${MONITOR_RUN}"
set LENS = "results/workspace_lens/${MODEL}"

$PYTHON -c 'import torch, sklearn, jlens; print("Monitor dependencies available")'
if ($status != 0) exit 1
if (! -e "${LENS}/j-lens/lens.pt") then
    echo "Missing published J-lens: ${LENS}/j-lens/lens.pt"
    echo "Read docs/SECURITY_MONITOR.md before fitting a new lens. Existing cluster artifacts can be reused."
    exit 1
endif
if (! -e "$DATA") then
    $PYTHON scripts/210_security_monitor.py generate --output "$DATA" --n-per-template "$MONITOR_BASES"
    if ($status != 0) exit 1
endif
$PYTHON scripts/210_security_monitor.py validate --dataset "$DATA"
if ($status != 0) exit 1
$PYTHON scripts/210_security_monitor.py extract --dataset "$DATA" --output "${OUT}/features" \
    --model "$MODEL" --lens-dir "$LENS" --layers "$MONITOR_LAYERS" --max-tokens "$MONITOR_MAX_TOKENS"
if ($status != 0) exit 1
if (! -e "${OUT}/frozen/monitor.pkl") then
    $PYTHON scripts/210_security_monitor.py fit --dataset "$DATA" --features "${OUT}/features" --output "${OUT}/frozen"
    if ($status != 0) exit 1
endif
if (! -e "${OUT}/synthetic_test/report.md") then
    $PYTHON scripts/210_security_monitor.py evaluate --dataset "$DATA" --features "${OUT}/features" \
        --bundle "${OUT}/frozen/monitor.pkl" --split synthetic_test --output "${OUT}/synthetic_test"
    if ($status != 0) exit 1
endif
echo "Development run complete: ${OUT}/synthetic_test/report.md"
echo "Frozen tools: ${OUT}/frozen/monitor.pkl. Final real-code results have NOT been run."
