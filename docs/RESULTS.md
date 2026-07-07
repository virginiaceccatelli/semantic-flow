# Results index

Raw data of record: `results/tables/*.csv` (tidy, one row per measurement).
Rendered summaries: `results/tables/md/*.md`. Figures: `results/figures/`
(png for viewing, pdf for the paper). Regenerate everything with
`python scripts/90_make_paper_assets.py`.

Status legend: ☐ not run · ◐ dev model (1.3b) · ● main model (6.7b)

| Exp | Table(s) | Figure(s) | 1.3b | 6.7b |
|-----|----------|-----------|------|------|
| E1 token type | `static_probes_{model}_core.csv` | `layers_accuracy_*` | ☐ | ☐ |
| E2 binding + strata | same | `layers_selectivity_*`, `binding_strata_*` | ☐ | ☐ |
| E3 def-use + distance | same | `defuse_distance_*` | ☐ | ☐ |
| E4 control dep | same | `layers_selectivity_*` | ☐ | ☐ |
| E5 context degradation | `context_degradation_{model}.csv` | `context_binding_*`, `context_defuse_edge_*` | ☐ | ☐ |
| E6 lead time | `behavioral_leadtime{,_summary}_{model}.csv` | `leadtime_*` | ☐ | ☐ |
| E7 causal patching | `causal_patching{,_summary}_{model}.csv` | `patching_recovery_*` | ☐ | ☐ |
| E8 real code | `static_probes_{model}_csn_python_200.csv` | `layers_*_csn*` | ☐ | ☐ |

Update the status cells as runs land, and record headline numbers below.

## Headline findings

_(fill in as results arrive)_

- E2 hard-stratum (same-name-different-binding) peak accuracy: — at layer —
- E3 accuracy drop from nearest to farthest distance bucket: —
- E5 steepest-degrading filler type: —
- E6 fraction of examples with positive lead time: — (CI —)
- E7 best non-readout recovery: — at (layer —, position —)
