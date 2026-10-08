# Synthetic versus real: saved stage-20 data

Separately trained within-dataset grouped CV; not synthetic-to-real transfer.

Maximum saved CV accuracy over transformer layers >= 0; first tied layer. Descriptive, not nested selection.

Raw CSV tables are authoritative for this export. Some older Markdown summaries and narrative numbers differ.

| Model | Dataset | Task | Stratum | Accuracy | Layer | Surface accuracy |
|---|---|---|---|---:|---:|---:|
| deepseek-coder-1.3b | core | binding | all | 0.9949 | 3 | 0.7979 |
| deepseek-coder-1.3b | core | defuse_edge | all | 0.9938 | 3 | 0.6821 |
| deepseek-coder-1.3b | core | taint_state | all | 1.0000 | 0 | — |
| deepseek-coder-1.3b | core | binding | context_matched | 0.9811 | 7 | 0.5000 |
| deepseek-coder-1.3b | core | binding | same_name_diff_binding | 0.9901 | 7 | 0.8827 |
| deepseek-coder-1.3b | csn_python_200 | binding | all | 0.9144 | 3 | 0.6850 |
| deepseek-coder-1.3b | csn_python_200 | defuse_edge | all | 0.9124 | 3 | 0.5993 |
| deepseek-coder-1.3b | csn_python_200 | taint_state | all | not measured | — | — |
| deepseek-coder-1.3b | csn_python_200 | binding | context_matched | not measured | — | — |
| deepseek-coder-1.3b | csn_python_200 | binding | same_name_diff_binding | 0.5164 | 19 | 0.7674 |
| deepseek-coder-6.7b | core | binding | all | 0.9886 | 11 | 0.7979 |
| deepseek-coder-6.7b | core | defuse_edge | all | 0.9899 | 3 | 0.6821 |
| deepseek-coder-6.7b | core | taint_state | all | 1.0000 | 0 | — |
| deepseek-coder-6.7b | core | binding | context_matched | 0.9906 | 11 | 0.5000 |
| deepseek-coder-6.7b | core | binding | same_name_diff_binding | 0.9827 | 11 | 0.8827 |
| deepseek-coder-6.7b | csn_python_200 | binding | all | 0.9024 | 7 | 0.6850 |
| deepseek-coder-6.7b | csn_python_200 | defuse_edge | all | 0.9108 | 3 | 0.5993 |
| deepseek-coder-6.7b | csn_python_200 | taint_state | all | not measured | — | — |
| deepseek-coder-6.7b | csn_python_200 | binding | context_matched | not measured | — | — |
| deepseek-coder-6.7b | csn_python_200 | binding | same_name_diff_binding | 0.4939 | 31 | 0.7674 |
| starcoder2-3b | core | binding | all | 0.9944 | 19 | 0.7719 |
| starcoder2-3b | core | defuse_edge | all | 0.9944 | 19 | 0.6584 |
| starcoder2-3b | core | taint_state | all | 1.0000 | 0 | — |
| starcoder2-3b | core | binding | context_matched | 0.9811 | 11 | 0.5000 |
| starcoder2-3b | core | binding | same_name_diff_binding | 0.9887 | 15 | 0.8660 |
| starcoder2-3b | csn_python_200 | binding | all | 0.9131 | 3 | 0.6870 |
| starcoder2-3b | csn_python_200 | defuse_edge | all | 0.9043 | 3 | 0.5857 |
| starcoder2-3b | csn_python_200 | taint_state | all | not measured | — | — |
| starcoder2-3b | csn_python_200 | binding | context_matched | not measured | — | — |
| starcoder2-3b | csn_python_200 | binding | same_name_diff_binding | 0.5512 | 29 | 0.8227 |

`same_name_diff_binding` contains negative examples only. Its accuracy is specificity on that subset; 0.5 is not an established chance floor. It is not a standalone balanced shadowing task.

Tagged rows inherit whole-task counts, not stratum counts. Aggregate accuracy is mean fold accuracy; tagged accuracy pools held-out hits.

No stratum denominators or row-level predictions in these tables; no confidence intervals computed.

Missing real context-matched and taint cells are missing evidence, not zero scores. Synthetic taint accuracy alone does not establish general taint-flow reasoning.

Reproduce: `python3 scripts/summarize_synthetic_real.py`. Full precision, controls, task counts, all layers and source hashes are in `data.json`.
