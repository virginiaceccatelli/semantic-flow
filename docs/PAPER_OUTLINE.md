# Paper drafting map: Tracing Semantic State in Code LLMs

## Evidence policy for drafting

- Data of record: `results/tables/*.csv`
- Generated views of the data: `results/tables/md/*.md`, `results/figures/*`
- Interpretation and experiment history: `docs/RESULTS.md`
- Intended designs and decision rules: `docs/EXPERIMENTS.md`, `docs/METHODS.md`, `docs/JLENS_PLAN.md`
- Conflict rule: CSV/table/figure result over prose claim; corrected E6 outputs over the original early-warning narrative
- Model roles:
  - DeepSeek-Coder 6.7B: main-results model; 32 blocks; layers −1, 0, 3, 7, 11, 15, 19, 23, 27, 31
  - DeepSeek-Coder 1.3B: cross-scale replication/development model; 24 blocks; layers −1, 0, 3, 7, 11, 15, 19, 23
  - StarCoder2-3B: planned optional architecture replication; no result artifact in repository; exclude from present empirical claims
- Canonical corpus sizes:
  - Synthetic core: 740 programs in current `core.jsonl`
  - Context variants: 1,200 programs; 40 bases × 5 filler types × 6 sizes
  - Obfuscation variants: 200 programs; 40 bases × 5 cumulative levels
  - Causal minimal pairs: 40 clean/corrupted pairs
  - Real code: 200 CodeSearchNet Python functions
- Run-status caution:
  - All E1–E10 result families available for both model sizes
  - Latest stage-61 manifest: 10-example, two-layer MPS smoke run; not replacement for full canonical J-lens tables
  - E6 1.3B: completed run but behavioral signal unusable; result undefined rather than negative
  - E9 status marker in `RESULTS.md` says development-only for 1.3B despite complete 1.3B table/figures; 6.7B remains headline
  - Working tree already contains an uncommitted `src/experiments/jlens_taint.py` modification and a smoke manifest; provenance note only, not paper evidence

## Recommended title and one-line thesis

- Working title: **Tracing Semantic State in Code Language Models: Decodability, Robustness, Causal Use, and Verbalizability**
- Shorter alternative: **Code Models Compute Semantics They Cannot Report**
- Central thesis:
  - variable binding and def–use structure absent from token-identical inputs, constructed in early transformer blocks, near-ceiling in middle layers, partially shed near output
  - representations robust to inert length and surface-preserving edits; vulnerable to scope interference and control-flow flattening
  - some semantic information causally used and routed across token positions
  - decodability distinct from verbalizability; control dependence near-perfectly probe-decodable yet chance under validated J-lens readout
  - no evidence for latent failure preceding behavioral failure under corrected E6 controls

## Abstract

- Opening problem:
  - code LLM competence compatible with lexical heuristics because identifier spelling, proximity, and indentation correlate with program relations
  - unresolved distinction: surface prediction versus internal computation of binding, data flow, control dependence, and taint
- Approach:
  - synthetic Python programs with construction-known semantic graphs and exact AST-to-token alignment
  - two frozen DeepSeek-Coder scales
  - grouped-CV linear probes plus shuffled-label selectivity and hard surface controls
  - frozen-probe stress tests under length, interference, and execution-verified obfuscation
  - activation patching for causal use
  - Jacobian lens for unsupervised verbalizability
  - CodeSearchNet transfer evaluation
- Primary result:
  - token-identical context-matched binding pairs: surface baseline 0.500; embedding layer 0.500; mid-layer peak ≈0.98 for both models
  - semantic relation built by contextual computation rather than recoverable from input token identity
- Stability result:
  - 6.7B binding at 500 filler tokens: comments 0.921 versus scope shadowing 0.570
  - scope shadowing at 1,000 tokens: binding 0.498 and def-use ≈0.59
  - renaming: early layers below chance, middle layers ≈0.85–0.90
  - flattening: best-layer binding ≈0.75; layer-average ≈0.59
- Causal/verbalizability result:
  - patching recovery moves from sink argument early (6.7B L0 0.985) to final position late (L31 1.000)
  - sanitizer-definition recovery 0.000 at every layer
  - J-lens validation: exact last-layer identity; next-token top-1 0.650 at 6.7B L27 versus random ≤0.050
  - temporally matched control-dependence J-lens at chance at every layer despite E4 AUC 0.999
- Corrective negative result:
  - usable E6 behavior only for 6.7B; balanced accuracy 0.836
  - trained probe mean early-warning excess reported as −0.010; no-model position floor +0.113
  - representation usually correct when behavior wrong; no anticipatory degradation
- Generalization:
  - real-code binding/def-use ≈0.90 accuracy and ≈0.98 AUC; same rise-then-decline layer shape
  - explicit limitation: real code transfer does not isolate semantic from lexical contribution
- Closing claim:
  - code models construct and sometimes use program-semantic state without consistently exposing it through output-aligned/verbalizable directions

## 1. Introduction

### 1.1 Motivation

- Code correctness dependent on relations not reducible to local token identity:
  - which definition a use resolves to
  - whether information flows from definition to use
  - which guard controls a statement
  - whether a value reaching a sink remains tainted
- Surface correlations enabling deceptive competence:
  - same spelling often same variable
  - nearby tokens often related
  - indentation often reveals control nesting
  - natural identifier names predictive of role
- Practical failure cases:
  - stale or shadowed bindings
  - sanitizer ignored at downstream sink
  - incorrect branch context
  - long-context distractor interference

### 1.2 Gap

- Behavioral accuracy alone:
  - no localization of internal semantic state
  - no distinction between semantic computation and lexical shortcut
- Standard probing alone:
  - information presence, not causal use
  - high accuracy potentially explained by class priors, token identity, distance, indentation, or within-program leakage
- Mechanistic intervention alone:
  - causal site without relation-specific decoding or robustness characterization
- Verbal report/output alignment:
  - stronger property than decodability
  - possible semantic state usable by computation but unavailable to model’s output head

### 1.3 Research questions

- RQ1 — representation:
  - linear decodability of lexical type, binding, def-use, control dependence, taint
  - depth at which relations appear, peak, and decline
- RQ2 — stability:
  - resistance to context length versus semantically relevant interference
- RQ3 — lexical versus semantic structure:
  - behavior under token-identical controls, renaming, opaque branches, expression encoding, flattening
  - contrast between relational semantics and locally reconstructable syntax
- RQ4 — temporal consequence:
  - internal readout failure before behavioral failure
  - corrected answer: no supported early warning
- RQ5 — causal use:
  - whether swapping relation-bearing activations changes model output
  - movement of causal information across positions and depth
- RQ6 — verbalizability:
  - whether probe-decodable relations align with output-disposing J-lens directions
  - decodable-versus-verbalizable dissociation

### 1.4 Contributions

- Controlled semantic probing benchmark:
  - exact graph ground truth by construction
  - verified source-span/token alignment
  - token-identical context-matched binding pairs
  - distance- and indentation-matched hard negatives
- Multi-axis evidence on same model family:
  - representation, robustness, real-code transfer, temporal behavior, causal intervention, output alignment
- Core empirical finding:
  - binding and def-use built after embeddings; ≈0.98–0.99 middle-layer decoding across both scales
- Structural boundary:
  - robustness to length/formatting; collapse under scope interference and flattened control structure
- Mechanistic finding:
  - causal signal routed from sink argument to final readout position; sanitizer definition causally inert
- J-lens contribution:
  - validated transfer of Jacobian-lens methodology to code models
  - control dependence decodable at AUC 0.999 but not verbalizable under token-level operationalization
- Methodological negative result:
  - early-warning metric rewards unreliable readouts without appropriate floors
  - corrected E6 rejects early-warning claim

### 1.5 Introduction ending / preview of findings

- Strongest evidence chain:
  - exact chance at surface and embeddings
  - rapid early-layer construction
  - mid-layer robustness to renaming and inert context
  - causal influence on output
  - real-code transfer with matching layer signature
- Nuanced endpoint:
  - semantic state neither universally robust nor necessarily reportable
  - “represented,” “used,” and “verbalizable” treated as separate claims

## 2. Related work

### 2.1 Structural probing of source-code models

- [Wan et al., *What Do They Capture?—A Structural Analysis of Pre-Trained Language Models for Source Code* (ICSE 2022)](https://arxiv.org/abs/2202.06840)
  - CodeBERT and GraphCodeBERT analyzed through attention, embedding probes, and syntax-tree induction
  - attention strongly aligned with code syntax
  - syntax preserved in intermediate transformer representations
  - pretrained models capable of inducing code syntax trees
  - main relevance:
    - early evidence for layerwise structural organization in code encoders
    - precedent for AST/tree structure as interpretability target
  - distinction from present work:
    - syntax rather than binding/def-use/control/taint semantics
    - attention alignment and syntax recovery rather than counterfactual surface controls or causal use
- [Hernández López et al., *AST-Probe: Recovering Abstract Syntax Trees from Hidden Representations of Pre-Trained Language Models* (2022)](https://arxiv.org/abs/2206.11719)
  - probe designed to recover an entire AST rather than isolated node labels
  - syntactic subspace found across five pretrained code models
  - middle layers carry most AST information
  - estimated syntactic subspace substantially lower-dimensional than full representation
  - direct connection to present layer results:
    - middle-layer structural peak consistent with E2–E4 rise/plateau
    - both reject final-layer-only accounts of structural knowledge
  - present extension:
    - relations defined by program analysis rather than grammar alone
    - exact-chance token-identical controls determining whether information must be contextually constructed
    - stability and intervention tests beyond recoverability
- Synthesis for paragraph:
  - prior syntax-probing literature establishes that code models preserve grammatical trees, commonly most strongly in middle layers
  - present question shifts from “can syntax be reconstructed?” to “are execution-relevant relations built beyond surface form, robustly maintained, causally used, and output-aligned?”

### 2.2 Broad probing suites for code syntax and semantics

- [Troshin and Chirkova, *Probing Pretrained Models of Source Code* (2022)](https://arxiv.org/abs/2202.08975)
  - diagnostic tasks covering syntactic structure/correctness, identifiers, data flow, namespaces, and natural-language naming
  - comparisons across code-specific pretraining objectives, model sizes, and fine-tuning
  - evidence that pretrained code models expose multiple code properties to probes
  - nearest overlap:
    - identifier, namespace, and data-flow information
    - model-size and layerwise comparisons
  - present differentiation:
    - hard negative strata targeted at exact shortcuts
    - grouped source-level cross-validation
    - within-program shuffled-label selectivity
    - frozen readouts tested under controlled distribution shifts
- [Ma et al., *Unveiling Code Pre-Trained Models: Investigating Syntax and Semantics Capacities* (2024 version)](https://arxiv.org/abs/2212.10017)
  - seven code models: CodeBERT, GraphCodeBERT, CodeT5, UniXcoder, StarCoder, CodeLlama, CodeT5+
  - probing targets spanning AST, CFG, control-dependence graph, and data-dependence graph
  - additional attention analysis for semantic structures and long-range token dependencies
  - broad finding:
    - syntax consistently captured
    - semantic encoding more variable across models and relations
  - closest prior work to E3/E4:
    - DDG and CDG reconstruction
    - syntax-versus-semantics comparison
  - present differentiation:
    - binding identity and taint added to DDG/CDG family
    - token-identical binding counterfactuals pin model-free and embedding baselines to chance
    - indentation-matched sibling guards expose high surface decodability of control dependence
    - accuracy supplemented by AUC, selectivity, strata, real-code transfer, robustness, and causal tests
- Recommended comparative sentence:
  - these probing suites show that code-model states correlate with structural labels; present experiments target whether those correlations survive when surface cues are explicitly held constant or adversarially transformed

### 2.3 Formal program semantics in next-token models

- [Jin and Rinard, *Emergent Representations of Program Semantics in Language Models Trained on Programs* (ICML 2024)](https://arxiv.org/abs/2305.11169)
  - transformer trained from scratch on synthetic grid-world programs plus partial input/output specifications
  - hidden-state probes recover unobserved intermediate execution states
  - semantic representations become increasingly accurate over training
  - interventional baseline separates information already represented by LM from computation learned by probe
  - strongest conceptual predecessor:
    - formal semantics emerging under next-token training without explicit semantic supervision
    - synthetic environment enabling exact latent-state labels
  - important differences:
    - custom-trained domain-specific language versus off-the-shelf pretrained Python code LMs
    - dynamic grid-world execution state versus binding, reaching definitions, control dependence, and taint
    - training-time emergence versus layerwise construction within a forward pass
    - present work adds natural-code transfer, semantics-preserving obfuscation, activation patching, and verbalizability
- [Guo et al., *GraphCodeBERT: Pre-training Code Representations with Data Flow* (ICLR 2021)](https://arxiv.org/abs/2009.08366)
  - explicit data-flow graph supplied during pretraining
  - graph-guided attention plus edge-prediction and node-alignment objectives
  - data flow defined as “where the value comes from” between variables
  - improvements on code search, clone detection, translation, and refinement
  - relevance:
    - demonstrates usefulness of directly injecting data-flow structure
    - motivates def-use as a compact semantic relation distinct from full AST hierarchy
  - contrast:
    - present models receive only token sequence at inference and were not instrumented with explicit program graphs
    - question is spontaneous internal recovery, not benefit from graph-supervised architecture/pretraining
- Positioning claim:
  - present results bridge pretrained code-model probing and exact formal-state studies
  - realistic pretrained models and Python syntax, but synthetic counterfactuals retain causal control over semantic ground truth

### 2.4 Probe validity, control tasks, and shortcut removal

- [Hewitt and Liang, *Designing and Interpreting Probes with Control Tasks* (2019)](https://arxiv.org/abs/1909.03368)
  - high probing accuracy insufficient when probe can memorize task independently of representation
  - selectivity = linguistic-task accuracy minus control-task accuracy
  - control tasks assign random labels learnable only through probe capacity/type identity
  - regularization and low capacity preferable to assuming linear/MLP accuracy is self-interpreting
  - direct methodological inheritance:
    - linear probes
    - shuffled-label control accuracy
    - selectivity reported alongside task accuracy
- Additional risks specific to source code:
  - token identity approximating variable identity
  - token distance approximating data flow
  - indentation approximating control dependence
  - program-specific rows leaking through random splits
- Present controls as code-specific extension of probe-validity literature:
  - context-matched binding pairs:
    - same anchor tokens/context except binding-flipping intervention
    - exact 0.500 model-free and embedding floors
  - same-name/different-binding negatives:
    - lexical identity predicts wrong label
  - distance-matched def-use negatives:
    - adjacency neutralized
  - indent-matched sibling guards:
    - nesting-depth cue neutralized
  - grouped CV by source program:
    - no shared program across folds
  - frozen-probe robustness evaluation:
    - prevents per-condition retraining from converting robustness question into learnability question
- Core positioning language:
  - stronger probe accuracy not treated as stronger semantic evidence unless a corresponding shortcut floor is unavailable or held at chance
  - headline hierarchy based on identification strength: context-matched binding above uncontrolled aggregate accuracy

### 2.5 Long context and “knows but does not tell” failures

- [Lu et al., *Insights into LLM Long-Context Failures: When Transformers Know but Don’t Tell* (2024)](https://arxiv.org/abs/2406.14673)
  - long-context positional bias studied through hidden-state probing
  - target position/information encoded internally even when final answer incorrect
  - retrieval/utilization disconnect framed as “know but don’t tell”
  - relationship examined between extraction time and final answer accuracy
  - direct connection:
    - E6 observation that taint probe can remain correct on every prefix where model answer fails
    - distinction between internal availability and output use
  - present extension:
    - semantic program state rather than document retrieval position
    - filler types separate raw length from lexical and scope interference
    - causal patching tests use rather than inferring utilization solely from probe/output disagreement
  - present corrective result:
    - no support for “latent failure precedes behavioral failure”
    - early-warning statistic shown to reward unreliable readouts
    - internal-state correctness during output error compatible with representational/behavioral dissociation, but not with anticipatory degradation
- Suggested framing:
  - replicate the broad “available internally, absent from answer” phenomenon
  - diverge on temporal interpretation: correct internal decoding at failure, not an earlier detectable breakdown

### 2.6 Robustness under semantic-preserving transformations

- Existing code probing usually evaluates held-out examples from same representation regime
- Present robustness question:
  - whether a fixed decoder continues to read the same relation after controlled changes
- Two orthogonal stress families:
  - inserted context:
    - inert comments/dead code
    - lexical decoys
    - competing updates
    - scope shadowing
  - whole-program transformation:
    - normalization
    - consistent renaming
    - opaque predicates
    - mixed Boolean-arithmetic encoding
    - control-flow flattening
- Relation to code-obfuscation literature:
  - obfuscation used as controlled distribution shift rather than malware detection or attack objective
  - execution verification preserves observable behavior
  - cumulative ladder identifies first transformation causing decoder failure
- Specific novelty relative to cited probing papers:
  - layerwise frozen-readout transfer under semantics-preserving transformations
  - comparison of lexical collapse in early layers with structural resilience in middle layers
  - separation of context length from semantic interference

### 2.7 Causal interpretability and activation patching

- [Zhang and Nanda, *Towards Best Practices of Activation Patching in Language Models: Metrics and Methods* (2023)](https://arxiv.org/abs/2309.16042)
  - activation patching/causal tracing outcomes sensitive to corruption design and effect metric
  - methodological choices can produce substantially different localization conclusions
  - recommendations for explicit metric and corruption justification
- Present implementation:
  - token-aligned clean/corrupted programs
  - one semantic difference at sink argument
  - normalized logit-difference recovery
  - layer × position sweep
  - last-token effects quarantined as potentially direct
- Why E7 strengthens probing evidence:
  - probe: information readable
  - patch: intervention at information-bearing location changes output
  - layerwise migration from sink token to output position consistent with information routing
- Remaining causal limitation:
  - whole-vector patching transports all differing information, not an isolated semantic subspace
  - recovery localizes a sufficient mediator under this corruption; not complete circuit or unique mechanism

### 2.8 Logit lens, Jacobian lens, and verbalizable representations

- Logit-lens family:
  - intermediate residual stream projected through fixed output unembedding
  - assumes shared coordinates between intermediate and final layers
- Tuned-lens family:
  - learned layer-specific mappings correct representational drift
  - supervised calibration introduces learned decoder
- [Gurnee et al., *Verbalizable Representations Form a Global Workspace in Language Models* (2026)](https://transformer-circuits.pub/2026/workspace/index.html)
  - Jacobian lens uses corpus-averaged derivative from intermediate residual state to final state
  - layer-specific output-token directions identify states disposed to affect future verbal report
  - positioned as mechanistically grounded refinement of logit lens without task-specific probe supervision
  - reports verbal report, directed modulation, internal reasoning, and flexible-generalization properties of J-space
  - acknowledged single-token limitation:
    - concepts without one-token vocabulary names may be distributed or missed
- Present adaptation:
  - first validate on code next-token prediction
  - exact final-layer J-lens/logit-lens identity check
  - random-direction and plain-logit controls
  - small, tokenizer-verified candidate identifier sets
  - same-anchor test of control-dependence target naming
- Present contribution:
  - code-domain replication of J-lens advantage over logit lens on next-token content
  - relational dissociation: supervised probe decodes control dependence; unsupervised output-aligned lens does not
  - operational claim restricted to token-level verbalizability, not general accessibility or consciousness

### 2.9 Related-work synthesis and novelty paragraph

- What prior code-probing work establishes:
  - syntax and several semantic graphs recoverable from pretrained representations
  - middle layers frequently most structurally informative
  - data-flow supervision useful when injected explicitly
- What formal-semantic probing adds:
  - next-token training can induce latent execution state
  - interventions required to distinguish LM representation from probe learning
- What long-context work adds:
  - internal availability can diverge from output utilization
- What probe-control and causal-method work requires:
  - accuracy contextualized by memorization/shortcut floors
  - causal results contextualized by corruption and metric
- Present paper’s combined gap:
  - no cited work jointly tests exact surface-controlled semantic decoding, layerwise robustness, natural-code transfer, behavioral timing, causal routing, and output-aligned verbalizability on the same code-model pipeline
- Recommended final novelty statement:
  - not the first demonstration that code models contain syntax or semantic information
  - contribution = evidential separation of surface availability, contextual semantic construction, robustness, causal use, and verbalizability

## 3. Experimental setup

### 3.1 Models

- DeepSeek-Coder 1.3B:
  - 24 transformer blocks
  - replication/development scale
- DeepSeek-Coder 6.7B:
  - 32 transformer blocks
  - main-results scale
- Frozen pretrained weights throughout
- Layer sampling:
  - embedding state denoted −1
  - block 0 and approximately every four blocks
  - final layer included
- Cross-scale comparisons by relative depth rather than identical absolute layer
- Not yet included:
  - StarCoder2-3B optional cross-architecture replication

### 3.2 Datasets and program families

- Synthetic core, 740 current records:
  - binding programs
  - taint programs
  - shadowing programs
  - matched-pair/control-dependence additions
- Binding examples:
  - definitions, uses, branches, reassignment
  - binding ID for every identifier occurrence
- Taint examples:
  - source → propagation → optional sanitizer → sink
  - per-prefix live-value taint state
- Shadow examples:
  - same spelling attached to distinct bindings
  - hard failure case for lexical identity
- Context dataset, 1,200 variants:
  - 40 bases
  - filler sizes 0, 50, 100, 200, 500, 1,000 real tokenizer tokens
  - comment prose, dead code, lexical decoy, competing update, scope shadow
- Obfuscation dataset, 200 variants:
  - 40 bases × five cumulative levels
  - all levels for a base retained or dropped together
  - observational equivalence checked by execution
- Minimal-pair dataset, 40 pairs:
  - clean program sinks sanitized variable
  - corrupted program sinks raw variable
  - token-for-token matching except sink argument
- Real dataset:
  - 200 fixed-seed, AST-parseable CodeSearchNet Python functions

### 3.3 Ground truth and alignment

- Python AST-derived spans and relations
- Binding IDs:
  - lexical scope and reassignment aware
- Def-use edges:
  - reaching-definition relationship
- Control dependence:
  - AST nesting with join-point-exact negatives
- Taint:
  - construction-known live state per line/prefix
- Token alignment:
  - AST character spans mapped to tokenizer offsets
  - exact source round-trip assertion
- Independent ground-truth cross-check:
  - differential testing of def-use graph against Beniget
  - previously exposed real mislabeling in `b = b + a`
- Tokenizer integrity:
  - refusal of tokenizer configuration failing exact code round trip
  - protection against Transformers 5.x DeepSeek-Coder mis-tokenization

### 3.4 Activation extraction

- Single frozen-model forward pass per example
- Stored values:
  - float16 hidden states at registered layers
  - input IDs
  - verified offsets
- Maximum sequence length:
  - 1,024 standard
  - 2,048 context variants
- One compressed activation artifact per example
- Separation of GPU extraction from CPU probe fitting

## 4. Probing methodology and controls

### 4.1 Probe tasks

- E1 lexical token type:
  - multiclass single-position readout
  - machinery sanity check, not semantic headline
- E2 binding:
  - pairwise same-definition classification
- E3 def-use:
  - pairwise edge classification
- E4 control dependence:
  - pairwise guard-to-statement classification
- Taint state:
  - single-position binary readout
  - supporting instrument for E6/E7; not standalone semantic claim

### 4.2 Probe architecture and training

- Linear logistic probe only
- Pair representation based on two anchor hidden states as implemented
- Regularization C = 0.1
- Maximum 2,000 iterations; tolerance 1e−3
- Maximum 20,000 samples per task/layer
- Five-fold grouped cross-validation
- Groups defined by source program
- No rows from same program divided across train/test
- Frozen final probe checkpoints for E5/E6/E9 use

### 4.3 Metrics

- Accuracy:
  - intuitive but threshold-dependent
- ROC AUC:
  - preferred for aggregate binary separation and real-code transfer
- Selectivity:
  - true-label accuracy minus within-source shuffled-label control accuracy
- Per-stratum recall/accuracy:
  - diagnostic; not equivalent to within-stratum AUC
- Distance buckets:
  - 0–10, 10–50, 50–200, 200+ tokens where available

### 4.4 Hard negative and baseline design

- Model-free surface baseline:
  - token IDs/local token windows/distance only
  - identical across model scales by construction
- Context-matched binding:
  - paired programs token-identical apart from one binding-flipping character/token
  - label flips while anchor surfaces remain fixed
  - surface and embedding floors exactly 0.500
  - primary E2 headline
- Same-name/different-binding:
  - identical identifier spelling, distinct semantic bindings
  - diagnostic transition from lexical failure to contextual success
- Distance-matched def-use negatives:
  - same token-gap distribution without true edge
  - removal of proximity shortcut
- Indent-matched control-dependence negatives:
  - statement under sibling guard at same nesting depth
  - removal of indentation shortcut
- Shuffled-label selectivity:
  - labels shuffled within source example
  - guards against priors and program-specific regularities

## 5. Results I — semantic relations emerge with depth

### 5.1 E1 lexical control

- Token type:
  - 1.000 accuracy at/near embedding layer in both scales
  - selectivity ≈0.86–0.90
- Interpretation:
  - lexical information already present before contextual computation
  - extraction/alignment sanity passed
- Reporting caution:
  - logged multiclass AUC 0.000 a reporting artifact
  - use accuracy/selectivity only
- Contrast prepared for E2/E3:
  - semantic relation at exact 0.500 embedding floor, unlike token type

### 5.2 E2 binding — primary result

- Main controlled comparison, context-matched pairs:
  - surface: 0.500 for both models
  - embedding −1: 0.500 for both models
  - 1.3B block 0 ≈0.557; layer 3 0.962; peak 0.981 at L7 in generated table; final L23 0.915
  - 6.7B block 0 0.528; layer 3 0.906; peak 0.991 at L11; final L31 0.934
- Interpretive sequence:
  - no binding information in isolated token identities
  - sharp construction during first few transformer blocks
  - high middle-layer plateau
  - partial late-layer decline
- Same-name/different-binding diagnostic:
  - embedding recall ≈0.001–0.002
  - layer 3 ≈0.967–0.987
  - identical spelling actively misleads input-level probe; context reverses error
- Aggregate probe results:
  - 1.3B binding AUC 0.997 at reported summary optimum
  - 6.7B binding AUC 0.998
  - selectivity ≈0.39–0.40
  - 423 source groups
- Cross-scale pattern:
  - same qualitative curve
  - peak at similar relative depth
  - 6.7B retains plateau deeper
- Claim licensed:
  - linearly accessible binding information computed from context beyond surface cues
- Claim not licensed:
  - symbolic variable table or exact algorithm identical to static analysis

### 5.3 E3 def-use — distance-robust data flow

- Aggregate:
  - 1.3B peak accuracy 0.989; AUC 0.999; selectivity 0.422
  - 6.7B peak accuracy 0.990; AUC 0.999; selectivity 0.409
  - 607 source groups
- Distance-controlled results:
  - 6.7B layer 3: 0–10 0.995; 10–50 0.989; 50–200 0.985
  - 6.7B final L31: 0.984; 0.982; 0.962
  - 1.3B layer 3: 0.996; 0.993; 0.993
  - 1.3B final L23: 0.983; 0.982; 0.974
- Interpretation:
  - mild long-distance decay rather than adjacency dependence
  - near-ceiling middle-layer edge decoding through 50–200-token bucket
  - same early construction / late decline as binding
- Surface caution:
  - easy diff-name/distance strata partly predictable without model
  - controlled distance curve supportive; context-matched binding remains cleanest semantic isolation

### 5.4 E4 control dependence — positive but surface-heavy

- Original invalid version:
  - surface baseline 1.000 from indentation shortcut
  - replaced by sibling-guard/indent-matched corpus
- Corrected surface baseline:
  - overall accuracy 0.927; AUC 0.990
  - positive recall 0.959
  - indent-matched negative recall 0.676
- Hidden probe best comparison:
  - 1.3B L11: positive recall 0.981; hard-negative recall 0.873; AUC 0.997
  - 6.7B L15: positive recall 0.995; hard-negative recall 0.923; AUC ≈0.999
  - hard-negative gain for 6.7B ≈+0.247 over surface
- Layer behavior:
  - embedding aggregate AUC ≈0.743
  - rise toward near-perfect separation in middle layers
  - similar relative-depth peak across scales
- Interpretation:
  - genuine hidden-state contribution beyond local surface features
  - relation nevertheless largely recoverable from syntax
  - weaker semantic-isolation claim than E2
- Reporting caveat:
  - layer −1 indent-matched recall of 1.000 = single-class threshold artifact
  - aggregate AUC required for threshold-proof interpretation
  - anchors on trailing literals rather than semantically central variable tokens

### 5.5 Synthesis figure and subsection close

- Suggested combined layer plot:
  - E1 lexical accuracy
  - E2 context-matched binding accuracy
  - E3 def-use AUC/accuracy
  - E4 aggregate AUC
- Visual message:
  - lexical property available immediately
  - semantic relations emerge with contextual processing
  - binding/def-use strongest isolation
  - control dependence computed but already heavily surface-predictable

## 6. Results II — robustness reveals what the representation depends on

### 6.1 E5 context degradation

- Frozen-probe design:
  - probes trained on core only
  - no adaptation to context variants
  - ground truth rebuilt per variant
- Main 6.7B binding result at 500 tokens:
  - comment prose 0.921
  - dead code 0.794
  - lexical decoy 0.795
  - competing update 0.859
  - scope shadow 0.570
- At 1,000 tokens:
  - scope-shadow binding 0.498, chance
  - scope-shadow def-use ≈0.59
  - all other filler categories above ≈0.70
- Per-layer behavior:
  - block 0 relatively stable around 0.75 under scope shadowing
  - middle layers—normally strongest semantic layers—collapse most
- Cross-scale result:
  - same ordering for 1.3B and 6.7B
- Interpretation:
  - raw distance not principal failure source
  - code-shaped filler costs more than inert comments
  - reuse of tracked names causes strongest degradation
  - vulnerability targets contextual binding computation rather than lexical lookup
- Alternative explanation to acknowledge:
  - frozen-probe distribution shift versus destruction of information
  - supported wording: accessible representation under original linear readout degrades

### 6.2 E9 semantics-preserving obfuscation

- Evaluation design:
  - same core-trained frozen binding/def-use probes
  - cumulative transformations
  - execution-verified observational equivalence
- 6.7B binding layer-average by level from generated table:
  - normalize 0.974
  - rename 0.726
  - opaque 0.722
  - encode 0.738
  - flatten 0.590
- 6.7B best-layer binding values emphasized in interpretation:
  - normalize ≈1.000
  - rename ≈0.897 at L11
  - opaque ≈0.857
  - encoding ≈0.846
  - flatten ≈0.750
- Layer-specific rename response:
  - embeddings/block 0 ≈0.29–0.33, below chance
  - middle layers ≈0.85–0.90
  - early lexical features actively fooled; middle representation more structural
- Transformation interpretation:
  - normalization negligible
  - alpha-renaming first major average cliff
  - opaque predicates and MBA encoding little additional damage
  - flattening second and deepest break
- Cross-scale replication:
  - 1.3B averages: binding 0.966 → 0.729 → 0.707 → 0.718 → 0.582
  - closely matched ladder shape
- Bound on claim:
  - representation not invariant to all semantics-preserving rewrites
  - dependence on recognizable control-flow scaffold

### 6.3 Unified stability interpretation

- Conditions largely survived:
  - inert comment length
  - normalization
  - opaque dead branches
  - arithmetic expression encoding after rename effect accounted for
- Conditions causing major break:
  - scope shadowing / tracked-name interference
  - global control-flow flattening
- Unified account:
  - model maintains relations across distance
  - accessible semantic state tied to scope and structured control context
  - middle layers less lexical than early layers, but not fully program-transform invariant

## 7. Results III — causal use and information routing

### 7.1 E7 design

- Length-matched clean/corrupted pairs
- Only sink argument differs:
  - sanitized variable versus raw tainted variable
- Patch clean residual vector into corrupted run
- Positions:
  - sink argument
  - sanitizer definition
  - last token
- Layers:
  - same registered model-specific sweep
- Outcome:
  - normalized logit-difference recovery
  - 0 = no recovery; 1 = complete clean/corrupted gap recovery
- Interpretive safeguard:
  - late last-token effects reported separately as direct readout forcing

### 7.2 6.7B routing result

- Embedding layer:
  - all positions 0.000
- Sink argument:
  - L0 0.985
  - L3 0.912
  - L7 0.708
  - L11 0.500
  - L15 0.235
  - ≈0 by final layer
- Last token:
  - L0 −0.007
  - L3 0.014
  - L7 0.074
  - L11 0.145
  - L15 0.308
  - L19 0.650
  - L23 0.76-range
  - L31 1.000
- Sanitizer definition:
  - 0.000 at every layer
- Interpretation:
  - causal signal initially localized at the token naming the sunk value
  - progressive transfer toward final readout position
  - crossover around middle depth
  - sanitization site not on causal path measured by this intervention

### 7.3 Cross-scale replication and nuance

- 1.3B:
  - sink argument L0 1.087; L3 0.872
  - last token rises to 0.899 at L19 and 1.000 at L23
  - sanitizer definition 0.000 throughout
- Shared qualitative path:
  - early sink-argument dominance
  - late last-token dominance
  - sanitizer-definition null
- Scale-specific timing:
  - 1.3B transfer occurs earlier in absolute layers
  - similar progression by relative depth
- Causal-class caveat:
  - encoded/used class labels incorporate probe-detection criterion
  - mean recovery is cleaner headline than class counts at earliest layers
  - early high recovery paired with `not_encoded` class in tables indicates classification-definition mismatch, not absence of causal effect

### 7.4 Claim boundary

- Supported:
  - activation at sink argument causally determines taint-answer logit difference early
  - causal locus moves toward final token with depth
- Not supported:
  - complete circuit identification
  - sanitizer never represented anywhere
  - semantic reasoning generally causal across all tasks

## 8. Results IV — behavioral failure does not show early internal warning

### 8.1 Original E6 failure mode

- Bare prompt constant responders:
  - 1.3B always “no”; raw accuracy 0.220; balanced accuracy 0.500
  - 6.7B always “yes”; raw accuracy 0.780; balanced accuracy 0.500
- Apparent scale split generated mechanically:
  - taint source always at first evaluable position
  - always-no model wrong immediately; positive lead impossible
  - always-yes model wrong near sanitizer; unreliable probe errors can appear earlier
- Required drafting treatment:
  - superseded result described only as motivation for corrected controls
  - no positive early-warning claim

### 8.2 Corrected behavioral signal

- Prompt diagnosis:
  - few-shot examples plus explicit current-variable name required
  - either component alone insufficient
- 6.7B canonical sanity table:
  - 342 prefixes
  - accuracy 0.939
  - balanced accuracy 0.836
  - tainted-class recall 1.000
  - clean-class recall 0.672
  - usable = true
- 1.3B:
  - accuracy 0.629
  - balanced accuracy 0.471
  - usable = false
  - lead time undefined, not zero

### 8.3 Floors and corrected result

- Readouts:
  - trained taint probe
  - norm-matched random direction
  - no-model position heuristic: tainted iff early step
- Analytic null:
  - expected early-error probability from per-prefix error rate
  - early-warning excess = observed rate − analytic null
- 6.7B headline:
  - position baseline excess +0.113
  - random directions range roughly +0.005 to +0.067 in concise summary, with layer variation in canonical table
  - trained probe mean excess reported −0.010 across valid interpretation
  - isolated positive cells do not survive corrected multiple-testing interpretation
- Metric pathology:
  - unreliable readout more likely to make an early mistake
  - J-lens experiment correlation: never-wrong fraction versus early-warning rate Pearson r = −0.905, p = 1.1×10⁻15
  - random readout mean apparent warning 0.634 versus J-lens 0.481, logit 0.373, probe 0.354 in full E10-2 diagnostic
- Strong negative conclusion:
  - no evidence probe failure precedes behavioral failure
  - position-only heuristic outperforming trained probe on “warning” metric demonstrates confound

### 8.4 Positive information retained from E6

- Probe per-prefix error at useful layers ≈0.005–0.027
- Position floor error 0.233
- Probe therefore reads taint rather than merely program depth
- At many layers:
  - probe correct on every prefix for all model-wrong examples
  - 6.7B often 19/19 `readout_never_wrong`
- Interpretation:
  - internal taint representation correct while forced-choice output wrong
  - failure of routing/reporting rather than detectable precursor degradation
- Alignment with E7:
  - sanitizer-definition patch does not alter output
  - representation accuracy and behavioral use separable

## 9. Results V — real-code transfer

### 9.1 E8 design

- Same activation/probe pipeline on 200 CodeSearchNet functions
- No synthetic template labels
- Grouped by function
- Binding and def-use only
- Purpose:
  - reject pure generator-artifact explanation
  - compare depth signature under natural Python

### 9.2 Main 6.7B results

- Binding:
  - surface AUC 0.673
  - embedding AUC 0.962
  - peak AUC 0.978 at L7
  - peak accuracy 0.902; selectivity 0.308
  - final AUC 0.913
- Def-use:
  - surface AUC 0.590
  - embedding AUC 0.958
  - peak AUC 0.979 at L3
  - peak accuracy 0.911; selectivity 0.308
  - final AUC 0.907
- Improvements over model-free surface:
  - binding ≈+0.31 AUC
  - def-use ≈+0.39 AUC

### 9.3 Cross-scale replication

- 1.3B binding:
  - surface 0.673
  - embedding 0.962
  - peak 0.980 at L3
  - final 0.907
- 1.3B def-use:
  - surface 0.590
  - embedding 0.959
  - peak 0.975 at L3
  - final 0.908
- Shared signature:
  - rise from embedding to early-middle peak
  - ≈0.07 decline toward output
  - nearly identical performance across scales

### 9.4 Interpretation and limitation

- Natural identifier identity already predictive:
  - embedding AUC ≈0.96
  - unlike exact 0.500 synthetic context-matched floor
- Supported:
  - learned decoder transfers to natural Python relations
  - depth signature not generator-specific
- Not supported:
  - isolated semantic component transfers independently of spelling
- Same-name/different-binding result:
  - low negative-class recall, e.g. 6.7B 0.095 at embeddings to 0.494 final
  - threshold-dependent per-stratum recall; no per-stratum AUC
  - stratum size absent from table
  - report as limitation/diagnostic, not headline failure

## 10. Results VI — decodable does not imply verbalizable

### 10.1 E10 validation gate

- Purpose:
  - establish J-lens implementation and applicability before interpreting task nulls
- Closed-form V1:
  - J-lens equals logit lens at final layer
  - measured 1.0000 in both models
- Next-token V2:
  - chance ≈0.038
  - 1.3B J-lens top-1 0.633 at L19
  - 6.7B J-lens top-1 0.650 at L27
  - random floor ≤0.133 in 1.3B and ≤0.050 in 6.7B
- J-lens over logit-lens advantage before final layer:
  - 1.3B maximum reported +0.150 at embedding
  - 6.7B maximum reported +0.183 at L19
- Interpretation:
  - Jacobian correction retrieves output-relevant content unavailable to plain logit lens
  - task nulls not caused by universally dead readout
- Validation caveat:
  - V3 taint disposition n=10; binary 0/1 cells; not load-bearing

### 10.2 E10-2 taint verbalizability

- Four readouts through same prefix stepping:
  - J-lens
  - logit lens
  - trained probe
  - norm-matched random lens
- 6.7B mean apparent early-warning rates:
  - J-lens 0.481
  - logit 0.373
  - probe 0.354
  - random 0.634
- Interpretation:
  - no evidence of verbalizable taint state
  - random floor exceeds semantic readouts
  - main contribution to paper = diagnostic exposing E6 metric pathology
- 1.3B:
  - model wrong at first evaluated prefix in original shared setup
  - all readouts forced to zero positive lead
  - not representational evidence

### 10.3 E10-3 control-dependence dissociation

- Same guard anchors; paired identifier-ranking task; chance 0.500
- Readout-alive control on 6.7B:
  - guard-variable ranking 0.507 at embedding
  - rises to 0.813 at L31
- Raw pooled control-dependence ranking:
  - near chance through L23
  - below chance late, 0.398 at L31
- Identified temporal confound:
  - positive target always after guard
  - negative target sometimes already visible before guard
  - causal attention/recency favors negative in unmatched rows
- Temporally matched correction:
  - negative constrained to appear after guard
  - n = 417 per 6.7B cell
  - layer range 0.453–0.537
  - L31 0.506
  - chance within every cluster-bootstrap 95% CI
  - Bonferroni-corrected tests null at every layer
- 1.3B isolated L7:
  - point 0.576
  - binomial p = 0.0023
  - cluster-bootstrap CI [0.495, 0.644]
  - not significant under clustered inference; no 6.7B replication
- Decodability contrast:
  - E4 hidden probe AUC up to 0.999
  - J-lens control-dependence ranking at chance
  - guard identity readable at same anchor
- Claim:
  - under token-level disposition-to-name-target operationalization, control dependence decodable but not verbalizable
- Limitation:
  - guard-variable control establishes identifier sensitivity, not ability to express arbitrary relational facts
  - no relational positive control at same anchor

### 10.4 Unified representation/use/reportability model

- Binding and def-use:
  - strongly decodable
  - robustness evidence for partial structural abstraction
- Taint:
  - highly probe-decodable
  - selectively causal through sink argument
  - no supported early-warning or verbalizable signal
- Control dependence:
  - highly probe-decodable
  - mostly surface-predictable
  - no J-lens verbalizable signature
- Three axes kept separate:
  - encoded
  - causally used
  - aligned with output/verbal report

## 11. Discussion

### 11.1 What code models appear to compute

- Binding relation constructed rapidly after token embedding
- Def-use relation maintained across substantial token distance
- Control dependence refined beyond local syntax despite strong surface baseline
- Shared relative-depth progression across 1.3B and 6.7B
- Early lexical / middle structural / late output-oriented organization

### 11.2 Robustness as evidence about representation type

- Inert distance tolerated:
  - comment padding high accuracy at 500–1,000 tokens
- Semantic interference harmful:
  - scope-shadow name reuse drives chance-level binding
- Identifier renaming:
  - early layers strongly name-specific
  - middle layers retain substantial structural signal
- Flattening:
  - remaining dependence on conventional control scaffold
- Best characterization:
  - neither shallow lexical lookup nor fully invariant symbolic semantics
  - context-sensitive structural representation

### 11.3 Information routing across depth

- Probe curve:
  - construction in early blocks
  - middle plateau
  - late decline
- Patching curve:
  - early local source at sink argument
  - later aggregation at final token
- Real-code curve:
  - same rise-and-shed pattern
- Joint hypothesis:
  - early blocks compute relations locally
  - middle blocks maintain/integrate them
  - late blocks transform selected information into next-token decision state
- Avoid overstatement:
  - layer correspondence correlational across different tasks
  - no direct circuit trace linking E2 to E7

### 11.4 Correct state, wrong answer

- E6 probe usually right where model forced-choice answer wrong
- E7 sanitizer-definition intervention causally inert
- E10 no verbalizable taint/control-dependence signal
- Possible account:
  - semantic state present but omitted from task-specific output computation
  - representation availability insufficient for elicitation
- Alternative accounts:
  - probe exploits information not normally consumed
  - forced-choice prompt measures instruction following/calibration more than code understanding
  - J-lens candidate-token operationalization too narrow for relational knowledge

### 11.5 Decodability versus verbalizability

- Key empirical dissociation:
  - E4 AUC 0.999
  - matched J-lens ≈0.50 across layers
  - guard-variable control up to 0.813
- Broader implication:
  - successful supervised readout does not imply model can name or report relation
  - interpretability tools answer different questions, not interchangeable confidence levels
- Cautious terminology:
  - “verbalizable” = output-aligned token disposition under implemented J-lens test
  - not conscious access, general reasoning availability, or natural-language explanation ability

### 11.6 Scale

- Strong replication across scales:
  - binding/def-use magnitude
  - relative layer shape
  - context and obfuscation ordering
  - causal routing
  - control-dependence J-lens null
- Differences:
  - absolute layer index/depth of plateau and routing
  - only 6.7B supports usable corrected taint forced-choice behavior
- No strong scaling law claim:
  - two related model sizes only
  - same architecture/training family

## 12. Limitations

- Model coverage:
  - two sizes from one architecture family
  - StarCoder2 replication unrun
- Synthetic-data scope:
  - small single-function Python programs
  - restricted semantic phenomena
  - no dynamic features, exceptions, aliasing-heavy objects, interprocedural flow, concurrency
- Probe interpretation:
  - linear decodability ≠ native model use
  - frozen-probe failure can reflect representational rotation/distribution shift
- Surface controls:
  - E2 context-matched control strong
  - E4 surface baseline already near ceiling
  - E8 no context-matched natural-code pairs
- Real-code ground truth:
  - AST-parseable CodeSearchNet subset
  - lexical and semantic contributions entangled
  - same-name/different-binding stratum size not recorded
- Causal patching:
  - single-token/single-position interventions
  - last-token late-layer effect partly direct
  - mean recovery may exceed 1 and does not by itself identify normal computation circuit
- E6:
  - only 6.7B behavioral signal usable
  - 19 model-wrong evaluation examples in corrected canonical summary
  - taint label correlated with prefix depth, r ≈−0.57
  - single calibration split/seed
  - metric demonstrated sensitivity to readout unreliability
- J-lens:
  - small candidate vocabularies
  - token-level proxy for relational verbalization
  - V3 validation n=10
  - guard identity control non-relational
- Anchor choice:
  - E4 guard/statement anchors on trailing literal tokens
- Reproducibility:
  - E6 layer sweep and stored probe checkpoints produced under different Python/scikit-learn environments
  - manifests record arguments and git SHA but not explicit success/status fields

## 13. Conclusion

- Direct answer:
  - yes, DeepSeek-Coder hidden states contain context-computed binding and def-use information beyond available surface cues
- Strongest quantitative recap:
  - context-matched binding 0.500 surface/embedding → ≈0.98–0.99 middle layers
  - real-code relation decoding ≈0.98 AUC
- Structural recap:
  - robust to inert length and several surface-preserving transforms
  - fragile to scope shadowing and control-flow flattening
- Mechanistic recap:
  - causal influence routed from sink argument to final token
  - sanitizer-definition site inert under patching
- Conceptual recap:
  - encoded, used, and verbalizable semantic state separable
  - control dependence strongest clean dissociation: probe AUC 0.999, matched J-lens chance
- Negative-result recap:
  - corrected E6 provides no early-warning evidence
  - rigorous floors reverse prior interpretation
- Final implication:
  - reliable code-model analysis requires controls that break surface-semantic correlations and triangulation across decoding, intervention, robustness, and output alignment

## 14. Reproducibility and artifact statement

- Pipeline stages:
  - 00 data generation
  - 10 activation extraction
  - 20 probes
  - 30 context degradation
  - 31 obfuscation
  - 40 lead time
  - 41 retrospective lead-time floors
  - 50 causal patching
  - 60 J-lens validation gate
  - 61 J-lens taint
  - 62 J-lens control dependence
  - 63 temporal correction
  - 90 paper assets
- Fixed seed 42 for canonical configuration
- All paper tables/figures regenerated from CSV alone
- Per-run manifests:
  - git SHA
  - CLI arguments
  - provenance timestamps through filenames
- Frozen activation store and probe checkpoints enable CPU-only re-analysis
- Tests:
  - data generation
  - graph extraction
  - token alignment
  - probes
  - lens calculations
  - ground-truth cross-check

## Appendix plan

### Appendix A — Dataset generation

- Templates for binding, taint, shadow, sibling guards
- Context filler-generation details
- Matched-pair construction constraints
- Obfuscation transforms and execution tests
- Dataset counts and filtering attrition

### Appendix B — Static analysis and alignment

- AST span extraction
- Binding and reaching-definition rules
- Control-dependence construction
- Beniget differential validation
- tokenizer round-trip invariant

### Appendix C — Probe details

- Feature construction
- hyperparameters
- fold grouping
- shuffle-control procedure
- surface baseline feature set
- per-task class balance and sample counts

### Appendix D — Full E1–E4 tables

- every layer for both models
- accuracy, AUC, selectivity, control accuracy
- binding strata
- def-use distance buckets
- control-dependence hard-negative recalls
- convergence indicators

### Appendix E — Full robustness tables

- E5 task × layer × filler × size
- E9 task × layer × obfuscation level
- 1.3B and 6.7B side-by-side

### Appendix F — Causal patching

- exact logit-difference definition
- clean/corrupted gap normalization
- per-pair distributions and not only means
- causal-class definition
- quarantined last-token results

### Appendix G — E6 correction audit

- original constant-responder diagnosis
- prompt sweep
- behavioral sanity tables
- analytic null derivation
- position and random floors
- multiple-testing correction
- reason 1.3B lead time undefined

### Appendix H — J-lens method and checks

- Jacobian construction and freezing split
- candidate vocabularies
- fp16 gradient scaling and retry logic
- V1/V2/V3 checks
- raw versus temporally matched control-dependence tables
- cluster bootstrap and multiple-comparison procedure

### Appendix I — Real-code analysis

- CodeSearchNet selection/filtering
- per-layer binding/def-use results
- natural identifier lexical advantage
- same-name/different-binding diagnostic
- proposed context-matched mutation follow-up

## Recommended main-paper figure and table order

- Figure 1 — Experimental pipeline and claim ladder:
  - exact programs/graphs → hidden-state decoding → robustness → behavioral/causal/output-aligned tests
- Figure 2 — Core representation result:
  - context-matched binding across layers, both scales
  - surface and embedding 0.500 floors marked
- Figure 3 — Semantic relation comparison:
  - binding strata plus def-use distance
  - optional E1 lexical curve inset
- Figure 4 — Stability:
  - E5 context curves emphasizing comments versus scope shadow
  - E9 per-layer rename/flatten heatmap or paired panels
- Figure 5 — Causal routing:
  - sink argument, sanitizer definition, last token recovery by layer
- Figure 6 — Decodable/not-verbalizable dissociation:
  - E4 probe AUC against temporally matched control-dependence J-lens
  - guard-variable live-readout control
- Figure 7 or appendix — Real-code layer transfer:
  - AUC curves for binding/def-use across scales
- E6 figure:
  - main paper only if methodological correction central to venue/story
  - show observed early-warning excess against analytic null, position, random, probe
  - otherwise detailed appendix with concise negative result in main text
- Table 1 — datasets, models, tasks, controls, sample sizes
- Table 2 — main headline results per experiment/model
- Table 3 — E4 threshold-proof positive/hard-negative comparison
- Table 4 — J-lens validation and corrected control-dependence inference

## Result-strength ranking for drafting emphasis

### Tier 1 — load-bearing claims

- E2 context-matched binding:
  - strongest surface-proof result
  - exact 0.500 surface and embedding floors
  - ≈0.98–0.99 mid-layer performance
  - replication across model scale
- E7 causal routing:
  - strong intervention result
  - clear positional migration across depth
  - exact sanitizer-definition null across layers/models
- E10-3 probe/J-lens dissociation:
  - same relation and anchor family
  - validated readout with live guard-variable control
  - temporal confound explicitly corrected
  - cluster-aware inference

### Tier 2 — strong supporting claims

- E3 def-use across distance:
  - ≈0.99 performance with mild distance decay
  - distance-matched controls
- E5 length versus interference:
  - large, interpretable contrast between comments and scope shadowing
  - replicated ordering across scales
- E9 layer-specific obfuscation:
  - mid-layer renaming resilience versus early-layer collapse
  - flattening as meaningful boundary
  - execution-verified semantics
- E8 real-code transfer:
  - ≈0.98 AUC and replicated layer signature
  - strong anti-generator-artifact evidence
  - weaker isolation of semantics specifically
- J-lens validation:
  - exact V1 plus strong V2
  - method credibility and code-domain replication

### Tier 3 — nuanced/contextual claims

- E4 control-dependence encoding:
  - hidden probe clearly beats hard-negative surface recall
  - surface baseline already AUC 0.990
  - best used to motivate syntactic-versus-semantic continuum and E10 dissociation
- E1 token type:
  - sanity control only
- Taint-state probe near ceiling:
  - instrumentation evidence, not standalone semantic finding

### Tier 4 — negative/corrective result

- E6 early warning:
  - no positive mechanistic claim
  - useful methodological lesson about behavioral sanity, analytic nulls, and unreliable-readout floors
  - 1.3B unmeasurable
  - 6.7B clear negative under corrected controls
- E10-2 taint J-lens:
  - null for verbalizable taint
  - strongest value as independent diagnosis of E6 metric failure

## What is complete, planned, optional, or missing

### Complete and suitable for current draft

- E1–E4 synthetic static probes, both models
- E5 context degradation, both models
- E6 corrected lead-time pipeline and floors, both models; interpretable only for 6.7B
- E7 causal patching, both models
- E8 CodeSearchNet transfer, both models
- E9 obfuscation, both model tables present; 6.7B headline
- E10 validation, taint diagnostic, and control-dependence test, both models
- Paper tables and figures regenerated through stage 90

### Highest-value follow-ups before submission

- Real-code context-matched pairs:
  - upgrade E8 from general decoder transfer to surface-controlled semantic replication
  - 150 candidate mutation sites across 61/200 functions before tokenizer constraint
- E8 hard-stratum counts:
  - record n for same-name/different-binding before discussing low recall
- E4 anchor polish:
  - rerun with guard variable and statement target rather than trailing literal
- E6 robustness:
  - multiple calibration seeds
  - balanced taint-depth corpus with randomized initial taint/re-taint transitions
- Environment consistency:
  - rerun E6/E7 in one software environment if exact reproducibility required

### Optional extensions

- StarCoder2-3B architecture replication
- E10-4 sink-argument J-lens generality
- Binding context-matched J-lens validation
- Coordinate/subspace patching rather than whole residual-vector patching

### Do not claim in current paper

- Early warning of model failure
- Semantic invariance under arbitrary obfuscation
- Symbolic execution or explicit compiler-like data structures
- Natural-code semantic isolation from E8 alone
- General scaling law from two same-family sizes
- General verbal incapacity from one token-level J-lens operationalization
- Absence of sanitizer representation solely from zero sanitizer-site patching

## Drafting order

- First:
  - Methods §§3–4 from stable repository design
  - Results §5.2 E2 and §7 E7
- Second:
  - Results §6 robustness and §10 J-lens dissociation
- Third:
  - E8 transfer and E4 nuance
- Fourth:
  - E6 corrective negative section
- Then:
  - Introduction shaped around final evidence hierarchy
  - Discussion and limitations
  - Abstract written last
