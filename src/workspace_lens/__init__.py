"""The published J-lens and R-lens, applied to code models.

This package is the *published* method, not an adaptation of it. The estimator
is the reference implementation released with "Verbalizable Representations Form
a Global Workspace in Language Models" (2026), vendored unmodified at
`third_party/jacobian-lens` and imported as `jlens`; the R-lens is the same
estimator run under the RelP backward rules of
`https://www.alignmentforum.org/posts/nv8oedrnLXKRzNEL9/`, implemented in
`relp.py`.

    corpus.py     the independent, pretraining-like fitting corpus
    adapter.py    this repository's models -> the released `LensModel` interface
    relp.py       the LN / identity / half rules that make an R-lens
    fitting.py    both lenses as a matched pair, with full provenance
    readout.py    full-vocabulary top-k, ranks, pass@k across layers
    evalsuite.py  the code-semantics probe suite
    ablation.py   causal edits along lens read directions
    validate.py   the seven-check pre-flight gate

The repository's earlier, differently-defined lenses live in
`src/models/cotangent_lens.py` and `src/models/cotangent_lrp.py` under names
that cannot be confused with these; see `docs/WORKSPACE_LENS.md`.
"""
