# DeepSeek-Coder execution and U-Space comparison

Analysis date: 2026-10-13. All four runs passed their completion gates and all registered artifact hashes verified. Model predictions were regraded against the saved expected literals and reproduced the saved correctness labels. Both models used identical prepared cases, expected answers, prompt text, anchor terms, implementation hashes, a 64-token generation budget, bfloat16, and block 6 at `prompt_end`. No cases were excluded for context length.

## Main result

6.7B improves exact execution-answer accuracy on both datasets. The U-Space cone score at this location does not provide useful error ranking in its prescribed direction. The entropy-weighted U-Lens adaptation behaves almost identically to answer entropy. More recognizable projected words in 6.7B do not translate into reliable error identification or source attribution.

## Execution accuracy

| Dataset | Calls / source groups | 1.3B macro accuracy [95% CI] | 6.7B macro accuracy [95% CI] | Paired improvement [95% CI] |
|---|---:|---:|---:|---:|
| CRUXEval | 726 / 726 | 28.10% [24.93, 31.27] | 33.75% [30.30, 37.05] | +5.65 pp [2.62, 8.68] |
| MBPP | 1,345 / 448 | 31.58% [28.27, 34.71] | 44.87% [41.18, 48.47] | +13.28 pp [10.16, 16.33] |

Intervals use 2,000 percentile bootstrap replicates, seed 42, resampling exact-source groups. MBPP contains 449 retained problem IDs and 448 exact-source groups. Macro accuracy averages calls within a source group first. MBPP call accuracy is 31.67% (426/1,345) and 44.98% (605/1,345). CRUXEval counts are 204/726 and 245/726.

6.7B fixes 84 CRUXEval answers while losing 43 previously correct ones. On MBPP it fixes 265 and loses 86. Thus scaling improves average reliability without producing a superset of the smaller model's successes.

For MBPP, source groups with every retained test correct increase from 51/448 (11.4%) to 104/448 (23.2%). Groups with all tests wrong decrease from 212 to 156. This is broader than a few extra isolated correct calls, but most functions still are not consistently solved.

## Format and execution are partly mixed in the score

| Run | Invalid literals | Parsed but incorrect |
|---|---:|---:|
| 1.3B CRUXEval | 56 (7.7%) | 466 |
| 6.7B CRUXEval | 55 (7.6%) | 426 |
| 1.3B MBPP | 147 (10.9%) | 772 |
| 6.7B MBPP | 228 (17.0%) | 512 |

There is a prompt/scorer mismatch: completing `assert call ==` permits a trailing assertion message, but the literal scorer interprets `True, "message"` as a tuple. Four 1.3B MBPP answers and five 6.7B MBPP answers contain the correct literal in this form.

A conservative post-hoc AST audit also finds 13 and 51 MBPP failures, respectively, that provide correct constant expressions, such as `(4*4)` for `16`. These are valid expressions but do not satisfy the requested evaluated-literal format. Fifteen MBPP failures in each model encode the correct value as a string; that is a real type mismatch. CRUXEval has two such string cases per model. These diagnostics have NOT replaced the primary scores. The audit never executes model code or calls model-supplied functions; it evaluates only a bounded whitelist of constant operators.

Allowing those three explicitly identified categories would change MBPP call accuracy from 31.67% to 34.05% and from 44.98% to 50.26%. This is a sensitivity calculation, not a new benchmark result or a complete estimate of format-attributable error. Correct expressions do not demonstrate that the model performed the remaining arithmetic. Hundreds of parsed incorrect answers remain.

Concrete shared failures in CRUXEval:

- `sample_12`: a loop strips `x` only when the string starts with `x`. The input ends in `Daniel` and does not start with it. Both models nevertheless remove that suffix; the correct output is unchanged.
- `sample_17`: `text.find(",")` on `"There are, no, commas, in this text"` returns 9. Both models answer -1.
- `sample_14`: stripping spaces and reversing `"   OOP   "` should produce `"POO"`; both models retain spaces or miss part of the transformation.

These demonstrate concrete behavioural misses. They suggest hypotheses about guard evaluation, exact character indexing, and operation composition; final answers alone do not establish internal causes or systematic concept deficits.

## Uncertainty

Error-detection AUROC on the common non-null-score subset; higher score means higher uncertainty, and 0.5 is chance ranking:

| Run | N | Cone alone | Answer entropy | Cone × entropy |
|---|---:|---:|---:|---:|
| 1.3B CRUXEval | 726 | .427 | .640 | .636 |
| 6.7B CRUXEval | 724 | .435 | .688 | .686 |
| 1.3B MBPP | 1,345 | .499 | .642 | .640 |
| 6.7B MBPP | 1,344 | .503 | .635 | .637 |

The cone is approximately chance on MBPP and reversed on CRUXEval. Its CRUXEval 95% CIs are [.380, .472] and [.393, .476]. This is an association at this location, not an argument for silently reversing the score after inspecting the test labels.

The composite-minus-entropy AUROC differences [95% paired group-bootstrap CI] are:

- 1.3B CRUXEval: -.00332 [-.00639, -.00018].
- 6.7B CRUXEval: -.00262 [-.00581, .00056].
- 1.3B MBPP: -.00242 [-.00600, .00106].
- 6.7B MBPP: +.00254 [.00019, .00499].

The small positive MBPP difference for 6.7B has a nominal interval above zero, but its magnitude is tiny and does not yield better retained-answer error at the 80% coverage target. Avoid claiming no statistical difference everywhere; the supported conclusion is no substantial, consistent added value here. These are exploratory intervals without multiple-comparison correction.

Composite-score versus entropy Spearman correlations are .9955–.9984. The cone varies far less than entropy, so the product is largely an entropy ranking. On 6.7B, keeping the 80% least-uncertain answers leaves about 61.8% CRUXEval errors and 53.9% MBPP errors. It is insufficient for a strong reliability filter.

The result survives two descriptive checks:

- On parsed answers only, cone AUROC remains .431/.440 on CRUXEval and .484/.503 on MBPP (1.3B/6.7B).
- Comparing correct/incorrect pairs only within the same expected-output type and generated-answer-length bin gives composite/entropy AUROC .751/.752, .805/.806, .645/.656, and .757/.759, respectively for 1.3B CRUXEval, 6.7B CRUXEval, 1.3B MBPP, 6.7B MBPP. Bins are exact lengths 0–4, then 5–8, 9–16, and 17+ tokens. The aggregate weights strata by their number of correct–incorrect pairs; single-class strata do not contribute. These are sensitivity diagnostics, not causal adjustment or replacement headline metrics.

Individual category directions show isolated associations but no stable interpretation across all four runs. For example, conflicting-evidence AUROC is .599 for 1.3B CRUXEval and .401 for 6.7B CRUXEval. These measurements do not validate specific uncertainty causes.

## What the surfaced words mean

Every requested read is on the same literal token, ` ==`. The ordinary J-lens is dominated by syntax or generic continuation tokens:

- 1.3B CRUXEval: leading-space backtick is top-1 in 433/726 cases.
- 6.7B CRUXEval: ` *` is top-1 in 554/726 cases.
- MBPP additionally has frequent end-of-sequence, newline, and bracket tokens.

The vocabulary decoding of the signed U-Space projection differs strongly:

- 1.3B: top-1 decodes to replacement-character fragments (`�` or ` �`) on every case. Other common entries are informal, multilingual, or unrelated fragments. A replacement character can result from decoding an isolated byte-level token; it is not by itself evidence of tokenizer corruption. Token IDs are retained in the raw records.
- 6.7B CRUXEval: ` mysterious` is top-1 in 463/726 cases, ` weird` in 179, and ` mystery` in 58.
- 6.7B MBPP: ` weird` is top-1 in 1,253/1,345 cases (93.2%).

These recognizable words are expected to be enriched when decoding a subspace constructed from uncertainty-related word anchors. Their prevalence across both successes and failures, together with poor cone error ranking, does not establish instance-specific uncertainty or semantic diagnosis. Also, this projected vocabulary decoding is an exploratory extension; it is distinct from the published U-Lens category readout and the ordinary full-state J-lens.

## Scope and next step

The strongest supported claim is: on these execution-verified published subsets, 6.7B has higher strict output accuracy, while this block-6, prompt-end U-Space adaptation offers little consistent error-detection benefit beyond predictive entropy.

Do not infer absence of execution representations, absence of uncertainty representations at every layer, or a causal source of mistakes. Both models are pretrained code-completion models; the U-Lens composite is adapted to answer entropy rather than the published reasoning-token entropy. Block 6 is post-block depth 7/24 for 1.3B and 7/32 for 6.7B, so the lens comparison is not at matched relative depth. Published benchmark exposure and shallow shortcuts are not ruled out by correct answers.

Before the next main experiment, align the answer prompt and scoring format and preserve these runs as the original baseline. For the planned Magistral experiment, use the exact supported checkpoint, its native chat/reasoning format, and its documented U-Space readout. Code-execution transfer will still need evaluation. A Magistral reasoning run changes model and inference protocol together, so describe it as a new setup rather than a controlled model-size comparison.

## Reproduction

From the repository root:

```bash
python results/execution_uncertainty/analysis/analyze.py
```

`comparison.json` contains bootstrap intervals, paired transitions, category scores, format sensitivity examples, subset counts, and hashes of the input configs/predictions/gates. The script verifies all registered input artifacts and regrades answers. Original run artifacts remain unchanged. This Markdown report interprets those outputs and is not regenerated by the script.
