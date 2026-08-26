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
result and are not needed for the conclusion immediately below; the R-lens earns
its place later in this part, on a question no vocabulary projection can ask.

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
The **J-lens still adds no semantic result** of its own.

The **R-lens now does**. It divides the model's answer score among the syntactic
roles of the program and asks whether relevance moves between the two data-flow
chains when a different chain is connected to the sink — on text that is
identical at every role except the sink argument itself. The chain feeding the
sink loses relevance share and the other gains, on **65 of 72 pairs in DeepSeek
1.3B and 64 of 72 in 6.7B**. On 6.7B the effect also passes the preregistered
permutation test of the mean, which 1.3B fails: a few large pairs reverse its
mean while most pairwise signs agree, so the 1.3B result rests on the sign test
alone.

Two limits keep this modest. The magnitude is small — the median pair moves
**1–2%** of the answer score. And the two models do not route at the same depth.
The pattern is a paired one, the tainted chain losing share while the trusted
chain gains, and DeepSeek 1.3B shows it in its first few layers while 6.7B shows
nothing there and produces it between roughly a quarter and a half of the way
through the network. Why the same routing happens at a different stage in the
larger model is unexplained. StarCoder2 cannot be included without extending the
method to its architecture, so this is one model family measured twice.

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

The answer-based alternative was also implemented as a concrete control. In the
first arm, installing the other binding requires the answer to change from `a`
to `b`. The control uses the model's J-lens output directions to construct an
explicit `a`-to-`b` push at the intervention layer. It is given the same edit
norm as DAS, so it cannot fail merely because it is weaker. In the crossed arm,
the binding change instead requires `b` to become `a`. The fixed answer push
should therefore fail or reverse, whereas an abstract binding direction should
continue to work. This is what “the answer direction did not transfer” means.

On DeepSeek-Coder 6.7B, a rank-one intervention at layer 8 makes the model emit
the value selected by the installed binding on **100% of held-out examples in
both arms**. A matched random direction succeeds on only about **2%**. An
explicit answer direction falls from **27.9% in the training-style arm to 4.3%
in the held-out arm**, exactly the failure predicted for an answer-based
explanation. A simple difference-of-means direction transfers to approximately
**76%**, but the learned direction reaches 100% while changing the hidden state
by substantially less.

**The same experiment has now been completed on StarCoder2-3B, and it agrees.** A
rank-one intervention at layer 11 also reaches **100% in both arms**, at
essentially the same edit size (0.478 of the hidden state's norm, against 6.7B's
0.479). All six gates pass. The falsification is sharper in this model: the
explicit answer direction does not merely weaken across the arms, it **reverses**,
which is the strongest form of the failure the design predicts for an
answer-based explanation.

This matters more than an ordinary replication because the two models are
different architectures, not two sizes of the same one. They use different
normalisation and different feed-forward layers. Finding the same one-dimensional
causal handle in both is evidence that it reflects how the task is solved rather
than one network's idiosyncrasy.

This is the project's strongest result. At these models, layers, and code
locations, the downstream computation reads a low-dimensional representation of
**which definition is in scope**. The result is causal because changing that
representation changes the emitted value, and the factorial design rules out a
fixed token or answer direction as the explanation.

## The same programs, read without intervening

A companion experiment reads the identical programs with an attribution method
instead of an intervention. It asks where the model's own answer score comes from,
and whether that moves when the binding changes. The two programs being compared
differ at exactly one token out of twenty-one — the inner definition's name — so
everything the measurement reads is textually identical.

On DeepSeek-Coder 6.7B it moves, cleanly. The definition that has just come into
scope gains about **5%** of the answer score and the one that has just left it
loses about **8%**, in the same direction on **all 280** held-out programs and at
every depth measured, peaking at roughly **22%** of the score in the middle of the
network. The single token that does differ between the two programs accounts for
about **1.5%** of that movement, so the effect is carried by text that did not
change. Reversing the value assignment does not reverse the effect, and holding
the scored output token literally fixed does not remove it.

Two limits are worth stating plainly. First, the effect is a property of the
*contrast* between the two program shapes rather than of any individual program:
pairing programs from different examples gives the same number, because every
example here is the same template with different names. Second, the measurement
does not work at all on the 1.3B model, where the score being decomposed is
non-positive on about 8% of readings — and those are precisely the shadowed
programs the smaller model already fails to answer reliably.

Most importantly, this is **not** additional causal evidence. Describing where an
answer is attributed and showing that the model uses a representation are
different claims, and only the intervention above supports the second. The two
results are reported together because the contrast between them is instructive,
not because the attribution reinforces the intervention.

The scope should remain explicit. The intervention has been completed on two
models but still at one layer, one site, and one synthetic construction, so the
model count is no longer the limitation — the location and the construction are.

One finding is genuinely unresolved and should not be smoothed over. The
rank-one edit works *better* than replacing the entire hidden state with the
donor's: 100% against 86% on 6.7B and 100% against 69% on StarCoder2. The
explanation on offer — that replacing everything also installs components that
work against the change — is plausible but has not been tested. Until it is, the
comparison against a "whole-state ceiling" is not measuring what its name
suggests. Separately, before a paper release stages 106–107 should be rerun for
6.7B so its saved gate report reflects the already-adopted decision metric; the
underlying rows and reported outcomes do not change, and StarCoder2's report
already uses the current rule.

# Overall conclusion

Across four increasingly demanding experiments, the results form a coherent
picture. Code models construct binding and data-flow relationships that are not
available from the tested local surface cues. Those representations are robust
to many changes in presentation but fragile when scope interference increases
or visible control structure is flattened. A security-relevant flow distinction
appears in the models' output coordinates, but as a distributed pattern rather
than a human-readable security word. Finally, a controlled intervention shows
that two different model architectures causally use a low-dimensional binding
representation at the tested site.

The appropriate claim is not that these models “understand code” in a general
human sense. The evidence supports a narrower and more useful conclusion: **the
tested models compute specific program-structural relationships, expose them in
measurable internal representations, and—in one rigorously isolated binding
case—use such a representation to determine their output.**
