# What Code Models Learn About Program Structure

## A concise overview of the completed findings

This project asks a simple question: **do code language models represent the
meaningful relationships that make a program work, or do they mainly rely on
surface patterns such as names, nearby tokens, and familiar formatting?**

The experiments study three program relationships. **Variable binding** is the
link between a variable use and the definition it refers to. **Def–use flow** is
the path from a definition to a later use of its value. **Source-to-sink flow**
asks whether data from an untrusted source reaches a security-sensitive
operation. These relationships have exact answers derived from the program's
structure, so the evaluation does not depend on human judgement.

The evidence is organised into four parts. Each part asks a stronger question
than the previous one: whether the information is present, what it depends on,
what form it takes, and whether the model actually uses it. Only experiments
that produced interpretable results under their stated controls are discussed
below.

# Part I — The models construct program relationships

The first experiments test whether binding and def–use relationships can be
recovered from the model's hidden states. A small linear classifier is trained
to decide whether two code locations participate in the same relationship. The
critical examples come in matched pairs whose local tokens and token distances
are identical while the correct binding changes. A classifier using only local
tokens and distance therefore scores exactly **50%**, as does the model's input
embedding. Any improvement above that floor must arise after the transformer has
processed the surrounding program.

Binding accuracy rises from **50% at the input** to approximately **98.4% in
middle layers** in both DeepSeek-Coder 1.3B and 6.7B. The information appears
within the first few transformer blocks, remains strong through the middle of
the network, and declines slightly near the output. This depth profile is
important: binding is not already encoded in the variable token. The model
constructs it by combining the token with its context.

Def–use relationships show the same pattern. They remain approximately
**96–99% decodable** even when the definition and use are 50–200 tokens apart,
with negatives matched for distance. The model is therefore not succeeding
merely because related locations tend to be close together.

The clean conclusion from Part I is that these models build internal
representations of binding and data flow that cannot be explained by the tested
local surface features. This is evidence that program structure is recoverable
from the hidden states. It is not yet evidence that the model uses that
information when producing an answer; that stronger question is addressed in
Part IV.

# Part II — The representations depend on program structure

The second set of experiments asks what makes these representations stable or
fragile. A classifier trained on ordinary programs is frozen and applied to
meaning-preserving rewrites. Because the classifier is not retrained, changes in
accuracy reveal whether the original representation remains readable in the
same form.

Longer context alone is not the main problem. In DeepSeek-Coder 6.7B, binding
accuracy remains **92.1% after inserting 500 tokens of harmless prose**. In
contrast, inserting a nested scope that reuses the tracked variable names lowers
accuracy to **57.0% at 500 tokens** and approximately chance at 1,000 tokens.
The model is much more affected by genuine reference ambiguity than by distance.
This is the behaviour expected from a system attempting to resolve program
relationships rather than relying only on proximity.

The clearest robustness boundary comes from testing individual obfuscations on
all three models. Renaming identifiers causes a modest loss, while opaque
predicates and equivalent arithmetic rewrites cause little or no additional
damage. **Control-flow flattening is the only transformation that causes a
large, reproducible collapse.** For binding, it lowers best-layer accuracy to
approximately **55.5% on DeepSeek 1.3B, 61.5% on DeepSeek 6.7B, and 52.7% on
StarCoder2 3B**. Def–use measurements show the same boundary.

The security experiment produces an especially clear replication. A frozen
classifier determines whether the argument to operations such as `os.system`,
`cursor.execute`, or `eval` is derived from an untrusted source. It reaches
**100% accuracy on held-out clean programs** while two independently measured
surface baselines remain at chance. When the transformations are applied one at
a time, control-flow flattening again accounts for almost the entire loss. What
remains after flattening is largely each model's tendency to favour one class,
not a reliable reading of the data flow.

Together, these results identify a specific boundary. The representations are
comparatively robust to distance, spelling changes, irrelevant branches, and
equivalent expression syntax. They are much less robust when competing scopes
make reference resolution harder or when the program's visible control
structure is replaced. The supported claim is about **frozen linear readouts**:
flattening prevents the original readout from transferring. It does not prove
that every possible representation of the relationship has disappeared.

# Part III — The distinction is distributed rather than written as a word

The lens experiments ask whether the safe/unsafe difference is aligned with the
model's own output vocabulary. The clean result comes from the simplest method,
the **logit lens**, which applies the model's ordinary output head to an
intermediate state. The more elaborate J-lens and R-lens do not improve this
result and are not needed for the conclusion below.

For every matched pair, the experiment records the change across the model's
entire vocabulary of roughly 32,000 tokens. Training pairs define an average
safe-to-unsafe direction, which is then frozen and evaluated on unseen pairs.
The measurement is taken at a position containing the same token in both
programs, making the input-level difference exactly zero. Same-label pairs
provide an additional control for ordinary variation between programs.

The result replicates across all three models: **72 out of 72 held-out pairs
point in the predicted direction in each model**. Their mean cosine similarity
with the training direction is **0.383, 0.380, and 0.390** for DeepSeek 1.3B,
DeepSeek 6.7B, and StarCoder2 3B respectively. The direction emerges around one
quarter of the way through the network and weakens sharply under control-flow
flattening, matching the independent probe results.

However, this direction is not concentrated in words such as `unsafe`,
`tainted`, or `vulnerable`. Its strongest coordinates are unrelated token
fragments, and its weight is spread across thousands of vocabulary dimensions.
It also fails the preregistered test that asked whether the label direction is
the largest source of variation between programs. The finding is therefore
precise but limited: **the safe/unsafe difference is reliably aligned with the
output space, but it is distributed and does not resemble an explicit semantic
label.**

A positive-control experiment confirms that this is not simply a blind
measurement. When the models are explicitly asked a yes/no taint question, the
same readout detects the models' graded answer margins with internal sign
consistency of approximately **0.85–0.94**. The models still have strong answer
biases—their final yes/no accuracy is only 50%—and one model reverses under a
different prompt. The positive control therefore validates the readout without
showing that the models solve the security task reliably.

No further run is needed to establish the replicated full-vocabulary finding.
There is currently **no clean semantic result unique to the J-lens or R-lens**.
The R-lens produced a small routing pattern on DeepSeek 1.3B, but it failed its
preregistered mean-based permutation control and has not been replicated on
6.7B. It should not be presented as a finding. A future R-lens claim would
require replication on 6.7B and a result that passes the declared control;
StarCoder2 cannot be included without extending the method to its architecture.

# Part IV — A binding representation is causally used

The first three parts observe information in hidden states. Observation alone
cannot show that the model uses that information. Part IV therefore performs an
intervention: it changes a hidden state along a learned one-dimensional subspace
and measures whether the model's output changes as predicted.

The experiment uses programs in which binding structure and assigned values are
varied independently. This design is crucial. A subspace representing “which
definition is in scope” predicts the same intervention effect in both
experimental arms. A subspace representing a particular token or the final
answer predicts the opposite effect in the second arm. The direction is learned
on one arm and evaluated on the other, so the alternatives can be directly
falsified.

On DeepSeek-Coder 6.7B, a rank-one intervention at layer 8 makes the model emit
the value selected by the installed binding on **100% of held-out examples in
both arms**. A matched random direction succeeds on only about **2%**. An
explicit answer direction falls from **27.9% in the training-style arm to 4.3%
in the held-out arm**, exactly the failure predicted for an answer-based
explanation. A simple difference-of-means direction transfers to approximately
**76%**, but the learned direction reaches 100% while changing the hidden state
by substantially less.

This is the project's strongest result. At this model, layer, and code location,
the downstream computation reads a low-dimensional representation of **which
definition is in scope**. The result is causal because changing that
representation changes the emitted value, and the factorial design rules out a
fixed token or answer direction as the explanation.

The scope should remain explicit. This intervention has been completed on one
model, one layer, one site, and one synthetic construction. Replication on a
second model and site is needed before claiming that the same causal mechanism
is general. Before a paper release, stages 106–107 should also be rerun so the
saved gate report reflects the already-adopted full-vocabulary decision metric;
the underlying intervention rows and reported outcomes do not change.

# Overall conclusion

Across four increasingly demanding experiments, the results form a coherent
picture. Code models construct binding and data-flow relationships that are not
available from the tested local surface cues. Those representations are robust
to many changes in presentation but fragile when scope interference increases
or visible control structure is flattened. A security-relevant flow distinction
appears in the models' output coordinates, but as a distributed pattern rather
than a human-readable security word. Finally, a controlled intervention shows
that DeepSeek-Coder 6.7B causally uses a low-dimensional binding representation
at the tested site.

The appropriate claim is not that these models “understand code” in a general
human sense. The evidence supports a narrower and more useful conclusion: **the
tested models compute specific program-structural relationships, expose them in
measurable internal representations, and—in one rigorously isolated binding
case—use such a representation to determine their output.**
