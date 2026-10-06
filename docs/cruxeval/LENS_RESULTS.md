# CruxEval lens results

What the published J-lens, the matched R-lens and the ordinary logit lens
surface on real CruxEval code, and what obfuscation does to it. Stages 236 and
242–246. [MECHANISTIC.md](MECHANISTIC.md) has the commands; this file has the
numbers and what they do and do not support.

**Model:** deepseek-coder-6.7b-base, 32 blocks, vocabulary 32256.
**Population:** the 500-program `lens_targets_rebuilt` CruxEval set.
**Lenses:** the E19 stage-201 J/R pair, fitted on the independent Pile corpus.
Nothing here refits a lens. Fitted source layers are 0–30; layer 30 is the
identity anchor, where J = R = logit = the model's own head.

Depth is `(layer + 1) / 32`.

---

## Summary

Three findings and three nulls. All are observational: a lens rank is not
evidence of causal use.

**Findings**

1. **The R-lens returns a prompt-independent prior.** Its top-20 is the same
   function words and punctuation for every program at every depth (mean
   cross-program Jaccard 0.68–0.94), and its output-token pass@20 is flat at
   ~10% from layer 0 to layer 8. Its apparent early-layer advantage over J is
   not a fact about the model.
2. **The J-lens carries output-**type** information at `use` @ layer 16**
   (53% depth), at ~2.3× the ordinary logit lens on the same forward pass,
   selected on calibration groups and reported on disjoint test groups.
3. **That type information survives alpha-renaming.** Renaming every local
   identifier costs the lexical channel 57% and the type channel 8%. It is not
   keyed on identifier names.

**Nulls**

4. **The logit lens is null through 28% depth** — median output-token rank
   12k–19k, pass@20 ≤ 0.003. The output token is not in raw output-vocabulary
   coordinates early.
5. **The J-lens surfaces nothing about the operations a program performs.**
   Operational specificity 0.0081 against a logit control of 0.0076 — a 4%
   margin, i.e. nothing.
6. **The J-lens layer-5 rank spike is an output-format prior, not semantics.**

---

## 1. Full-vocabulary output-token ranks (stages 236, 242)

Combined pre-answer score over `use`, `post_use` and `call`, each program
weighted equally; `rr = 1/(rank+1)`. `answer` is a positive control and never
selects a layer. Full table in `lens_layer_discovery/alignment_by_layer.csv`.

| layer | depth | J MRR | J median rank | J pass@20 | R MRR | R median rank | R pass@20 | logit MRR | logit median rank |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 3.1% | 0.00016 | 29752 | 0.0007 | 0.0407 | 190 | 0.102 | 0.00016 | 19523 |
| 1 | 6.2% | 0.00083 | 12441 | 0.0026 | 0.0346 | 180 | 0.104 | 0.00019 | 16744 |
| 2 | 9.4% | 0.0038 | 2572 | 0.0107 | 0.0276 | 200 | 0.096 | 0.00030 | 14276 |
| 3 | 12.5% | 0.0041 | 1363 | 0.0151 | 0.0621 | 143 | 0.107 | 0.00038 | 14978 |
| 4 | 15.6% | 0.0043 | 1908 | 0.0147 | 0.0668 | 112 | 0.111 | 0.00051 | 12229 |
| **5** | **18.75%** | **0.0278** | **111** | **0.168** | 0.0670 | 114 | 0.109 | 0.00064 | 13209 |
| 6 | 21.9% | 0.0122 | 414 | 0.0563 | 0.0505 | 112 | 0.117 | 0.00077 | 12719 |
| 7 | 25.0% | 0.0141 | 396 | 0.0567 | 0.0468 | 140 | 0.116 | 0.00081 | 14497 |
| 8 | 28.1% | 0.0073 | 2006 | 0.0206 | 0.0328 | 225 | 0.106 | 0.0012 | 13130 |

**R is depth-independent.** Pass@20 moves from 0.102 to 0.106 while the model
does nine blocks of work. At layer 0 the context-matched binding probe reads
0.528 — chance — so there is no computed content there for R to be reading.

**J is depth-dependent but non-monotonic.** Median rank falls 270× from layer 0
to layer 5, pass@20 rises 11× between layers 4 and 5, then both regress by
layer 8. Stage 242 selected layer 5 as the J, R and consensus pre-answer layer.

**The instrument is sound.** At the layer-30 identity anchor all three lenses
coincide exactly, as they must, and the answer position reads MRR 0.6986,
pass@1 0.578, pass@10 0.946, median rank 0. The pre-answer null is about the
model, not the readout.

| answer control | MRR | pass@1 | pass@10 | median rank |
|---|---:|---:|---:|---:|
| J @ L5 | 0.0263 | 0.0012 | 0.0733 | 92 |
| R @ L5 | 0.0834 | 0.0571 | 0.0965 | 117 |
| logit @ L5 | 0.0263 | 0.0119 | 0.0467 | 1613 |
| all three @ L30 (anchor) | 0.6986 | 0.5783 | 0.9458 | 0 |

### Selection status

Stage 242's layer choice is **exploratory and same-dataset** — selected on the
same 500 programs it then reads. It is not held-out model selection. Stage 244
(below) is held out.

---

## 2. What the lenses literally surface (stage 243)

Unfiltered top-20 over the full vocabulary at four concrete token positions,
layers 0–8. Computed over the deterministic 20-program sample in
`lens_top20_clean/examples.md`; 2160 blocks, all complete.

### R-lens returns one list for everything

Mean pairwise Jaccard of the top-20 across the 20 programs:

| read | L0 | L3 | L5 | L8 |
|---|---:|---:|---:|---:|
| `use` | 0.68 | 0.82 | 0.77 | 0.50 |
| `call` | 0.73 | 0.89 | 0.90 | 0.49 |

Its most frequent tokens at `use`, at every layer: `' a'`, `','`, `' ('`,
`' '`, `' in'`, `'\n'`, `' to'`, `' and'`, `' the'`, `' at'` — each in 20 of 20
programs. A unigram frequency prior. This is the mechanism behind the flat
~10% pass@20 in §1: CruxEval outputs are punctuation-heavy, so a constant
bracket-and-comma list scores hits without reading anything.

### The logit lens returns noise

Jaccard ~0.02 — different junk per program. 80% of its slots are rare word
fragments and CJK tokens.

### The J-lens layer-5 spike is a format prior

J's lists are program-specific at most layers (Jaccard 0.03–0.19 at `use`) but
the content is not task vocabulary: `'variant'`, `'prior'`, `' maybe'`,
`' cant'`, `' stuff'`, `' things'`.

At layer 5 its character mix shifts sharply and its cross-program Jaccard
*rises* to 0.35 — the lists converge toward a shared set:

| layer | word share | digit | punctuation | whitespace |
|---:|---:|---:|---:|---:|
| 4 | 0.36 | 0.00 | 0.49 | 0.08 |
| **5** | **0.12** | **0.07** | 0.53 | **0.22** |
| 6 | 0.42 | 0.01 | 0.41 | 0.09 |

Its layer-5 shared tokens are `'['` (20/20), `'\n'` (20/20), `' '`, `'.'`,
`'...'`, `'1'` (15/20), `'2'` (14/20). Classifying every top-20 hit on the
program's own output as either from that shared set or program-specific:

| layer | shared-list hits | program-specific hits |
|---:|---:|---:|
| 3 | 0 | 7 |
| **5** | **25** | **5** |
| 7 | 8 | 7 |

**The layer-5 gain is "predict a bracket and a small digit".** Most CruxEval
answers are lists of small integers, so that scores without knowing the
program.

### No execution vocabulary at any pre-answer position

Scanning 60 Python/execution terms across 14,400 J-lens slots: `'result'` ×21,
`'list'` ×11, `'output'` ×7 — 0.15% of slots, concentrated at layers 0–3 where
J's MRR is ~0.0002. R's `' for'` ×589 is its constant list. The logit lens's
`'true'`/`'false'` at answer positions is a type prior for what follows `==`.

Share of top-20 tokens occurring anywhere in the program's own source, input or
output: 0.3–5.5% for J, 3.3–7.5% for R (its function words appear in source
text by chance), ≤1.3% for the logit lens. The only program-specific content
hit in the whole sample was `' txt'` for a program whose output is
`'hello.txt'` — and `.txt` is in that program's input, so it is copying.

---

## 3. J-lens semantic words (stage 244)

The word-masked view: the vocabulary is restricted to coherent ASCII words
(≥3 characters, alphabetic, leading space allowed; no specials, digits,
punctuation or CJK), and the J-lens is swept over layers 4–25 × four reads.

**The filter cannot be the finding.** Masking to words guarantees words come
out. The statistic is specificity — own-program word overlap minus mean
other-program overlap — so a lens returning one fixed plausible list scores 0
by construction. Each program's word sets are parsed from its own AST and
recorded output: `lexical` (its identifiers), `operational` (executed methods,
builtins, control constructs), `type` (words naming the output's type).

Selection on calibration source groups, reported on disjoint test groups.
`answer` excluded. Selected cell: **`use` @ layer 16 (53% depth)**.

| held-out test split | J-lens | logit control | ratio |
|---|---:|---:|---:|
| `specificity_semantic` (operational + type) | 0.0406 | 0.0233 | 1.74 |
| `specificity_type` | 0.0332 | 0.0162 | **2.05** |
| `specificity_operational` | 0.00740 | 0.00709 | **1.04** |
| `specificity_lexical` | 0.0136 | 0.0048 | 2.82 |
| `execution_lexicon_rate` | 0.123 | 0.088 | 1.40 |
| `control_lexicon_rate` | 0.000 | 0.0006 | — |
| `list_repeat_rate` | 0.018 | 0.013 | — |

Calibration 0.0350 → held-out 0.0406: the selection replicates out of sample.

**The operational channel is dead** — a 4% margin over the logit lens is
nothing. Whatever J surfaces at this site is not about what the program *does*.

**The type channel is the effect.** 2.05× the control on words naming the
output's type. `control_lexicon_rate` being exactly 0 while
`execution_lexicon_rate` is 0.123 says the surfaced words are code-execution
vocabulary rather than generic code vocabulary.

**The effect is small.** 0.033 × 20 ≈ 0.66 words per 20-word list of own-type
overlap above the cross-program baseline, of which ~0.34 words is J's margin
over the control.

`list_repeat_rate` 0.018 means the average surfaced word appears in under 2% of
programs' lists. The fixed-list failure mode that characterises the R-lens is
absent here.

---

## 4. Obfuscation (stages 245, 246)

Execution-verified variants of all 500 programs: each re-executes
`f(recorded input)` in an isolated subprocess and must return the recorded
output with matching **value and type**. Every use, call and answer anchor is
rebuilt from the variant's own source; nothing is inherited from the base.

Accepted variants: normalize 500, rename 499, opaque 498, encode 492,
flatten 489 (2478 total, every program contributing at least one).

Stage 246 applies the **frozen** `use` @ L16 cell from stage 244 — nothing is
re-selected per condition — and scores every condition against the **clean**
program's word sets.

### J-lens

| condition | n | semantic | lexical | operational | type | exec lexicon | repeat rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| clean | 500 | 0.0380 | 0.01341 | 0.00808 | 0.02995 | 0.120 | 0.0130 |
| L0 normalize | 500 | 0.0377 | 0.01334 | 0.00804 | 0.02964 | 0.120 | 0.0131 |
| L1 rename | 499 | 0.0360 | **0.00572** | 0.00837 | **0.02761** | 0.105 | 0.0134 |
| L2 opaque | 498 | 0.0339 | 0.00507 | 0.00828 | 0.02564 | 0.103 | 0.0129 |
| L3 encode | 492 | 0.0350 | 0.00479 | 0.00791 | 0.02711 | 0.102 | 0.0131 |
| L4 flatten | 489 | 0.0399 | 0.00328 | 0.00519 | 0.03472 | 0.139 | **0.0288** |

### Logit-lens control

| condition | n | semantic | lexical | operational | type | exec lexicon | repeat rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| clean | 500 | 0.0206 | 0.00446 | 0.00756 | 0.01309 | 0.088 | 0.0091 |
| L0 normalize | 500 | 0.0207 | 0.00445 | 0.00761 | 0.01312 | 0.088 | 0.0091 |
| L1 rename | 499 | 0.0208 | 0.00310 | 0.00783 | 0.01296 | 0.077 | 0.0098 |
| L2 opaque | 498 | 0.0174 | 0.00241 | 0.00694 | 0.01046 | 0.075 | 0.0098 |
| L3 encode | 492 | 0.0199 | 0.00255 | 0.00759 | 0.01232 | 0.077 | 0.0098 |
| L4 flatten | 489 | 0.0288 | 0.00304 | 0.00481 | 0.02395 | 0.068 | 0.0226 |

### The dissociation

Retention relative to clean, J-lens:

| level | lexical | type | J/control on type |
|---|---:|---:|---:|
| L0 normalize | 99.5% | 99.0% | 2.26 |
| **L1 rename** | **42.7%** | **92.2%** | 2.13 |
| L2 opaque | 37.8% | 85.6% | 2.45 |
| L3 encode | 35.7% | 90.5% | 2.20 |
| L4 flatten | 24.5% | 115.9% | 1.45 |

**Renaming costs the lexical channel 57% and the type channel 8%.** J's
advantage over the logit control on lexical words collapses from 3.0× to 1.85×
at rename; its type advantage holds at ~2.2× through rename, opaque predicates
and arithmetic encoding. The output-type information is not carried by
identifier names.

**L0 is an exact no-op control.** Every J number matches clean to three
decimals, although L0 goes through the same AST round-trip, re-tokenization and
anchor rebuild as every other level. That pins the L1–L4 differences as real
rather than artifacts of re-anchoring.

**L4 is not interpretable as a gain.** Flattening shows the highest type
specificity of any condition *and* doubles `list_repeat_rate` (0.0131 → 0.0288)
in both lenses. Flattened programs are structurally homogeneous, so the word
lists became markedly more shared; when a shared list drifts toward generic
collection words and most CruxEval outputs are lists, own-type overlap can rise
faster than the cross-program average while the readout becomes *less*
program-specific. The J/control ratio falling to 1.45 at exactly this level is
consistent with that. Treat L4 as confounded, not as robustness.

---

## What this does not establish

- **"Not lexical" is not "not surface."** The ladder removes *names*; it does
  not remove structural cues to output type. A program returning a list still
  contains `[`, `append` and `for` after rename, opaque and encode. Flattening
  is the only level that rewrites structure, and it is the confounded one. The
  supported claim is that the type signal is not keyed on identifier names.
- **Part of `specificity_type` is a type prior.** Type word sets are shared by
  all programs with the same output type, so some of the effect is "recognises
  the output is a list" rather than "reads this program". Stratifying the
  cross-program baseline by output type would separate these; it is a CPU
  re-scoring of the existing rows and has not been run.
- **Nothing here is causal.** These are lens ranks. Causal claims need the
  stage-237 erasure rows or the stage-240 interchange.
- **The operational channel was null before obfuscation entered.** Its flatness
  across levels is not robustness; there was no J-specific signal to lose.
- **Effect sizes are small throughout** — well under one word per 20-word list.
- **§2 is a 20-program sample** at layers 0–8. §1, §3 and §4 are the full 500.
- **Stage 242's layer selection is exploratory and same-dataset.** Stage 244's
  site selection is held out; stage 246 freezes it.

## Relation to existing results

§1 and §2 replicate [RESULTS.md](../RESULTS.md) E19 Question 2 on real code:
the concrete runtime value is essentially absent from the top vocabulary at
`use`, `post_use` and `call`, and readable at the answer position. Presence and
verbalizability remain different properties — the probes (R1/R2) and DAS (R10)
show binding is present and causally used at these depths.

The stage-244 site at 53% depth is consistent with E19's DeepSeek 6.7B panel,
where binding words first enter the top-10 at layers 11–14 and the strongest
controlled contrast is at layer 20.

§4's rename-survivable / flatten-confounded pattern is the same shape as R4's
atomic obfuscation result, though measured on a different instrument.

## Provenance and corrections

Every stage writes `gates.json`; all of the above come from passed gates.
Stage 236 reran the E19 required suite (corpus independence, matched
provenance, identity anchor, model-head equivalence, forward invariance, rule
binding, nontrivial J/R difference) before any readout, as did 243, 244 and 246.

One correction: stage 243's `examples.md` originally printed a single
"Output-token ranks" line per (read, layer) block above all three lenses.
Ranks are per lens, so that line reported only the J-lens's and mislabelled it
as a property of the position. Fixed; ranks now sit on each lens's own line,
with a regression test. **The CSV artifacts were never affected** —
`top20_long.csv.gz` and `top20_lists.csv.gz` always carried correct per-lens
`target_ranks` — and every number in §2 is computed from token lists, which the
defect did not touch.
