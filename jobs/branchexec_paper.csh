#!/bin/csh
# BranchExec paper run: three models, sequentially, then the cross-model table.
#
#   jobs/branchexec_paper.csh
#   setenv MODELS "deepseek-coder-1.3b starcoder2-3b"; jobs/branchexec_paper.csh   # subset
#
# deepseek-coder-6.7b/full already exists: its passed stages are skipped and only
# stages 256 (link), 257 (repair) and the v2 report are added. The other models
# run the whole pipeline. Models run one at a time, never co-resident.
source jobs/common.csh
if (! $?MODELS) setenv MODELS "deepseek-coder-6.7b deepseek-coder-1.3b starcoder2-3b"

foreach M ($MODELS)
    setenv MODEL $M
    # StarCoder2 was trained in bfloat16; float16 risks overflow in its activations.
    if ("$M" =~ starcoder2*) then
        setenv DTYPE bfloat16
    else
        setenv DTYPE float16
    endif
    csh jobs/branchexec.csh || exit 1
end

set RUNS = ()
foreach M ($MODELS)
    set RUNS = ($RUNS results/branchexec/$M/full)
end
set STAMP = `date +%Y%m%d_%H%M`
echo "=== stage 258: representation vs utilisation across models — CPU ==="
$PYTHON scripts/258_branch_compare.py --runs $RUNS --output results/branchexec/compare_$STAMP || exit 1
cat results/branchexec/compare_$STAMP/compare.md
