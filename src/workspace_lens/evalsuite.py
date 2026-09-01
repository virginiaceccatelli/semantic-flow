"""The code-semantics probe suite: what do the lenses surface while reading code?

The question this suite asks is the paper's question, put to a code model:
**at a given layer and token position, which vocabulary tokens is the residual
stream poised to be verbalised as, and do those tokens name the program-semantic
intermediate the model needs there?**

That is deliberately not the question the archived experiments in this
repository asked. Those fixed a two-word candidate vocabulary in advance and
compared margins inside it. The J-lens ranks over the *whole* vocabulary, so the
right question is what comes out unprompted — and the honest metric is a rank
and a pass@k against a target the program's semantics determines, with a
distractor the program's surface would produce instead.

## The families

Each family isolates one program-semantic intermediate and pairs it with a
distractor that a surface reader would emit, so a high pass@k cannot be earned
by copying a nearby token.

    binding      x = 3 outside, x = 7 shadowing inside; read at the use of `x`.
                 target = the value actually in scope, distractor = the other
                 one. The crossed design is the repository's own: both members
                 are token-identical at the queried position and differ only in
                 which definition is live, so a bounded surface reader is at
                 chance by construction.
    defuse       a definition many lines above its single use; read at the use.
    alias        b = a; read at `b`, target = a's value.
    call         a function whose body returns a constant; read at the call.
    typeof       x = "..." vs x = [...]; read at `x.`, target = a method that
                 only that type has (`upper` vs `append`).
    arith        c = a + b with a, b bound to literals; target = the sum, which
                 appears nowhere in the prompt.
    loopvar      for ch in "...": read at `ch`, target = string-ness.
    scopeword    the same shadowing programs, scored against scope vocabulary
                 (`local`, `global`, `inner`, `outer`) rather than values — the
                 "concept, not answer" arm.

`arith`, `typeof` and `scopeword` are the load-bearing ones for the paper's own
claim: their targets are tokens that **never appear in the prompt**, so a hit
cannot be attention copying an input token forward.

## Two read positions, because the first run showed one is not enough

The first 1.3B run read every value-carrying family at the *use* token — the
`x` of `return x` — and returned an exact null. The top-k explained it: at that
position the model is poised to say `' + '` or `' == '`, because the natural
continuation of `return x` is an operator. The value was never what that
position was about to verbalize, so "the lens does not surface it" answered a
question nobody asked.

Every value-carrying family therefore now carries **two** items per program,
distinguished by `read`:

    read="use"      the variable's use token — is the bound value verbalizable
                    at the moment the variable is read?
    read="answer"   a validated `assert f() == ` suffix, where the value IS the
                    next token the model must emit — how early does it appear?

The suffix comes from `src.data.counterfactual_pairs.choose_answer_suffix`,
which this repository already validates per tokenizer: it checks that writing
the answer appends *exactly one* token and that the answerless encoding is a
strict prefix. Scoring a space token instead of an answer is a silent failure,
and it is not one this suite invents a second time.

The contrast is the point. `answer` establishes that the model has the value and
says when; `use` then asks whether it was verbalizable earlier, at the position
where binding is resolved. A null at `use` beside a hit at `answer` is a
finding. A null at both is a null about the instrument's reach.

## Positions

Positions are named by an anchor substring rather than an index, and resolved
against the real tokenization at read time (`resolve_position`). An index would
silently drift the moment a tokenizer split a literal differently, and every
number in the readout is a statement about one specific position.

## Targets

A target is a *set* of token ids — the same concept spelled with and without a
leading space, and any single-token synonym — and an item's score is the best
rank over that set. Multi-token concepts are excluded at construction time
rather than scored partially, because a rank over the vocabulary is only defined
for single tokens; `build_suite` reports how many candidates it dropped for that
reason.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional, Sequence

logger = logging.getLogger(__name__)

FAMILIES = ("binding", "defuse", "alias", "call", "typeof", "arith",
            "loopvar", "scopeword")


@dataclass
class ProbeItem:
    """One (prompt, position, target, distractor) evaluation item."""

    item_id: str
    family: str
    prompt: str
    anchor: str                 # substring whose LAST token is the read position
    target_words: list[str]     # concept spellings; best rank over them scores
    distractor_words: list[str] # what a surface reader would emit instead
    arm: str = ""               # which member of a crossed pair this is
    pair_id: str = ""           # the two arms of a crossed pair share this
    read: str = "use"           # "use" | "answer" — WHERE the lens is read
    target_in_prompt: bool = True
    notes: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class Suite:
    name: str
    items: list[ProbeItem] = field(default_factory=list)
    dropped_multitoken: dict = field(default_factory=dict)
    answer_reads: str = "unknown"   # "built" | why they could not be

    def prompts(self) -> list[str]:
        return [item.prompt for item in self.items]

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(json.dumps({"_meta": {"name": self.name,
                                          "n_items": len(self.items),
                                          "answer_reads": self.answer_reads,
                                          "dropped_multitoken":
                                              self.dropped_multitoken}}) + "\n")
            for item in self.items:
                f.write(json.dumps(item.as_dict()) + "\n")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Suite":
        items, meta = [], {}
        with open(path) as f:
            for line in f:
                obj = json.loads(line)
                if "_meta" in obj:
                    meta = obj["_meta"]
                else:
                    items.append(ProbeItem(**obj))
        return cls(name=meta.get("name", Path(path).stem), items=items,
                   dropped_multitoken=meta.get("dropped_multitoken", {}),
                   answer_reads=meta.get("answer_reads", "unknown"))


# ── program templates ────────────────────────────────────────────────────────

_HEADER = "# utility helpers\nimport os\nimport sys\n\n"


def _answer_suffix(tokenizer, fname: str) -> str:
    """`\nassert <fname>() == `, in the spelling this tokenizer validates.

    Delegates the hard part to the repository's existing
    `choose_answer_suffix`, which verifies per tokenizer that the answer is
    exactly one appended token; only the entry-point name is substituted.
    """
    from src.data.counterfactual_pairs import choose_answer_suffix

    return choose_answer_suffix(tokenizer).replace("f()", f"{fname}()")


def _positioned_items(source: ProbeItem, suffix: str) -> list[ProbeItem]:
    """The same program read at every valid position between use and answer.

    One read position cannot carry a null. "The bound value is not verbalizable
    at the use token" is a claim about the model; "…at the one position we
    happened to pick" is a claim about the experiment, and only replication
    across positions tells them apart.

    Four reads on one prompt, all unambiguous substrings of it, ordered by how
    far the computation has progressed:

        use       the variable's use token — binding is resolved, nothing emitted
        post_use  the first token after the function body closes
        call      the call site's closing paren — the value is being produced
        answer    the position where the value IS the next token

    Every one is *valid*: the binding is determined at all four, and none is
    after the answer. If the value is verbalizable anywhere before it must be
    emitted, one of these should show it — and if none does, the null is about
    the model rather than about a position choice.
    """
    full = source.prompt + suffix
    call_anchor = suffix.rstrip()[:-3].rstrip() if suffix.rstrip().endswith("==") \
        else suffix.split("==")[0].rstrip()

    def item(tag: str, anchor: str) -> Optional[ProbeItem]:
        if full.count(anchor) != 1:
            return None                      # ambiguous here; drop rather than guess
        return ProbeItem(
            item_id=f"{source.item_id}-{tag}", family=source.family,
            pair_id=source.pair_id, arm=source.arm, read=tag,
            prompt=full, anchor=anchor,
            target_words=list(source.target_words),
            distractor_words=list(source.distractor_words),
            target_in_prompt=source.target_in_prompt,
            notes=f"{source.notes}; read at the {tag} position",
        )

    candidates = [
        ("use", source.anchor),
        ("post_use", source.anchor + "\nassert"),
        ("call", call_anchor),
        ("answer", suffix),
    ]
    return [i for i in (item(tag, a) for tag, a in candidates) if i is not None]


def _binding_pair(pair: int, outer: int, inner: int, fname: str, other: str):
    """The repository's shadowing construction, both arms.

    Token-identical at `return x`; only which definition is live differs. Arm
    `outer` has an unrelated local, arm `inner` shadows. The value in scope is
    the target and the other value is the distractor, so a reader that copies
    the nearest literal scores at chance across the pair.
    """
    common = f"{_HEADER}x = {outer}\n\n\ndef {fname}():\n"
    return [
        ProbeItem(
            item_id=f"binding-{pair}-outer", family="binding", pair_id=f"binding-{pair}",
            arm="outer",
            prompt=common + f"    {other} = {inner}\n    return x",
            anchor="    return x",
            target_words=[str(outer)], distractor_words=[str(inner)],
            notes="global x is in scope; the inner assignment binds another name",
        ),
        ProbeItem(
            item_id=f"binding-{pair}-inner", family="binding", pair_id=f"binding-{pair}",
            arm="inner",
            prompt=common + f"    x = {inner}\n    return x",
            anchor="    return x",
            target_words=[str(inner)], distractor_words=[str(outer)],
            notes="the inner assignment shadows the global",
        ),
    ]


def _scopeword_pair(pair: int, outer: int, inner: int, fname: str, other: str):
    """The same two programs, scored against scope vocabulary instead of values.

    The concept arm: nothing in either prompt contains the words `local`,
    `global`, `inner` or `outer`, so any rank they achieve is the lens naming
    a relation the model computed rather than echoing the input.
    """
    common = f"{_HEADER}x = {outer}\n\n\ndef {fname}():\n"
    return [
        ProbeItem(
            item_id=f"scopeword-{pair}-outer", family="scopeword",
            pair_id=f"scopeword-{pair}", arm="outer",
            prompt=common + f"    {other} = {inner}\n    return x",
            anchor="    return x",
            target_words=["global", "outer", "module"],
            distractor_words=["local", "inner"],
            target_in_prompt=False,
            notes="scope vocabulary, unprompted",
        ),
        ProbeItem(
            item_id=f"scopeword-{pair}-inner", family="scopeword",
            pair_id=f"scopeword-{pair}", arm="inner",
            prompt=common + f"    x = {inner}\n    return x",
            anchor="    return x",
            target_words=["local", "inner"],
            distractor_words=["global", "outer", "module"],
            target_in_prompt=False,
            notes="scope vocabulary, unprompted",
        ),
    ]


def _defuse(i: int, value: int, name: str, filler: int):
    """A definition separated from its single use by `filler` statements.

    The distractor is the last filler statement's value — the nearest competing
    literal — nudged if it would coincide with the target: a distractor that IS
    the target makes an item silently unscorable rather than hard.
    """
    while filler * 2 == value:
        filler += 1
    body = "".join(f"    step_{k} = {k} * 2\n" for k in range(filler))
    return ProbeItem(
        item_id=f"defuse-{i}", family="defuse",
        prompt=f"{_HEADER}def compute():\n    {name} = {value}\n{body}    return {name}",
        anchor=f"    return {name}",
        target_words=[str(value)], distractor_words=[str(filler * 2)],
        notes=f"{filler} intervening statements between definition and use",
    )


def _alias(i: int, value: int, a: str, b: str):
    return ProbeItem(
        item_id=f"alias-{i}", family="alias",
        prompt=f"{_HEADER}def run():\n    {a} = {value}\n    {b} = {a}\n    return {b}",
        anchor=f"    return {b}",
        target_words=[str(value)], distractor_words=[a],
        notes="one level of aliasing between the literal and the use",
    )


def _call(i: int, value: int, fname: str):
    return ProbeItem(
        item_id=f"call-{i}", family="call",
        prompt=(f"{_HEADER}def {fname}():\n    return {value}\n\n\n"
                f"def main():\n    total = {fname}()\n    return total"),
        anchor="    return total",
        target_words=[str(value)], distractor_words=[fname],
        notes="the value must come through a call boundary",
    )


_TYPES = [
    ("\"hello world\"", ["upper", "lower", "strip"], ["append", "keys"], "str"),
    ("[1, 2, 3]",       ["append", "extend"],        ["upper", "keys"],  "list"),
    ("{\"a\": 1}",      ["keys", "items", "values"], ["append", "upper"], "dict"),
]


def _typeof(i: int, literal: str, targets: list[str], distractors: list[str], tname: str):
    return ProbeItem(
        item_id=f"typeof-{i}-{tname}", family="typeof",
        prompt=f"{_HEADER}def handle():\n    value = {literal}\n    return value.",
        anchor="    return value.",
        target_words=targets, distractor_words=distractors,
        target_in_prompt=False,
        notes=f"the method must follow from the inferred type ({tname})",
    )


def _arith(i: int, a: int, b: int):
    return ProbeItem(
        item_id=f"arith-{i}", family="arith",
        prompt=(f"{_HEADER}def total():\n    first = {a}\n    second = {b}\n"
                f"    result = first + second\n    return result"),
        anchor="    return result",
        target_words=[str(a + b)], distractor_words=[str(a), str(b)],
        target_in_prompt=False,
        notes="the sum appears nowhere in the prompt",
    )


def _loopvar(i: int, text: str):
    return ProbeItem(
        item_id=f"loopvar-{i}", family="loopvar",
        prompt=(f"{_HEADER}def scan():\n    out = []\n    for ch in \"{text}\":\n"
                f"        out.append(ch."),
        anchor="        out.append(ch.",
        target_words=["upper", "isdigit", "lower"], distractor_words=["append", "keys"],
        target_in_prompt=False,
        notes="the loop variable's type follows from the iterable",
    )


# ── construction ─────────────────────────────────────────────────────────────

_NAMES = ["helper", "process", "collect", "reduce_all", "gather", "resolve",
          "compute_all", "select", "combine", "expand"]
_OTHER = ["y", "z", "acc", "tmp", "buf", "idx", "cur", "prev", "res", "val"]

#: Literals are chosen from this pool *per tokenizer* rather than fixed in
#: advance, because which integers survive as single tokens is a property of the
#: BPE merge table.
_VALUE_POOL = tuple(range(2, 1000))


def _single_token_values(tokenizer) -> list[int]:
    """The integers this tokenizer keeps whole.

    Measured, not assumed, and the measurement is a constraint on the whole
    design: DeepSeek-Coder and StarCoder2 both segment every multi-digit number
    digit by digit, so the usable pool is exactly 2-9. A vocabulary-rank readout
    can therefore only ask a code model about single-digit values, which is why
    `_arith` uses operands whose sum stays single-digit and why literals are
    reused across items rather than being unique to one.
    """
    return [v for v in _VALUE_POOL if _is_single_token(tokenizer, str(v))]


def _value_pairs(values: list[int], n: int, *, seed: int = 0,
                 require_sum: bool = False, min_gap: int = 3
                 ) -> list[tuple[int, int]]:
    """`n` distinct (outer, inner) literal pairs drawn from `values`.

    `min_gap` keeps the two members well separated, so a lens that has merely
    localised "a small integer" cannot score as though it had resolved which
    one. `require_sum` additionally demands that `outer + inner` is itself a
    single token, which only the `arith` family needs.

    Literals are reused *across* pairs (never within one), because they have to
    be: see `_single_token_values` — every model here tokenizes multi-digit
    numbers digit by digit, so the entire usable pool is 2-9 and a no-reuse rule
    would cap the suite at three items per family. Items sharing an answer token
    is harmless, since each is a separate program scored on its own rank; items
    silently disappearing would not be.

    Deterministic given `values`, `n` and `seed`; constraints are relaxed in a
    fixed order rather than returning fewer pairs than asked for.
    """
    import random

    usable = set(values)
    rng = random.Random(seed)
    candidates = [(a, b) for a in values for b in values
                  if a != b and (not require_sum or (a + b) in usable)]
    for gap in (min_gap, max(min_gap - 1, 1), 1):
        pool = [p for p in candidates if abs(p[0] - p[1]) >= gap]
        if len(pool) >= n:
            rng.shuffle(pool)
            return pool[:n]
    rng.shuffle(candidates)
    return candidates[:n]


def build_suite(tokenizer, n_per_family: int = 10, name: str = "code-semantics") -> Suite:
    """Generate the suite against THIS tokenizer, keeping only scorable items.

    Two filters, in this order:

    * **Literals are chosen for the tokenizer.** A rank over the vocabulary is
      defined for single tokens only, so the integers a program is built from
      are drawn from what this tokenizer keeps whole, rather than fixed in
      advance and discarded afterwards.

    * **Crossed pairs are dropped together.** The binding and scopeword families
      earn their interpretation from having two arms that are token-identical at
      the read position; keeping one arm of a pair would silently turn a
      controlled contrast into an uncontrolled single condition. If either arm
      is unscorable, both go, and the count is reported.

    Whatever is dropped is recorded on the suite and printed by stage 200, so a
    thin family is visible rather than inferred from a small `n`.
    """
    values = _single_token_values(tokenizer)
    if len(values) < 4:
        raise RuntimeError(
            f"only {len(values)} single-token integer literals for this "
            "tokenizer; the value-carrying families cannot be built. Inspect "
            "the tokenizer before continuing — this is the signature of the "
            "mis-resolved code tokenizer that load_tokenizer() guards against."
        )
    try:
        _answer_suffix(tokenizer, "f")
        answer_reads = "built"
    except Exception as exc:                                    # noqa: BLE001
        answer_reads = f"unavailable: {type(exc).__name__}: {str(exc)[:120]}"
        logger.warning("no answer-position reads for this tokenizer (%s); the "
                       "use-position families are unaffected", answer_reads)

    pairs = _value_pairs(values, n_per_family, seed=0)
    # The arith family needs its operands' SUM to be a single token too,
    # which is a strictly tighter constraint, so it draws its own pairs
    # rather than shrinking every other family to fit.
    arith_pairs = _value_pairs(values, n_per_family, seed=1,
                               require_sum=True, min_gap=1)

    items: list[ProbeItem] = []
    for i, (outer, inner) in enumerate(pairs):
        fname, other = _NAMES[i % len(_NAMES)], _OTHER[i % len(_OTHER)]
        binding = _binding_pair(i, outer, inner, fname, other)
        defuse = _defuse(i, outer, other, filler=3 + i)
        alias = _alias(i, inner, other, _OTHER[(i + 3) % len(_OTHER)])
        call = _call(i, outer, fname)
        a, b = arith_pairs[i % len(arith_pairs)]
        arith = _arith(i, a, b)

        value_items = [*binding, defuse, alias, call, arith]
        items += _scopeword_pair(i, outer, inner, fname, other)

        # The same value-carrying programs, read where the value must actually
        # be emitted. Entry points differ per family, so the suffix is built per
        # item rather than once.
        #
        # A tokenizer that cannot make the answer a single appended token gets
        # the use-position families and no answer reads, rather than no suite at
        # all: the answer position is an addition to the design, and losing it
        # should cost that addition, not everything else. Why it was lost is
        # recorded on the suite and printed by stage 200.
        # One prompt, four read positions — not one prompt per position. The
        # four reads are then differences in *where* the lens looks, with the
        # program held exactly fixed; emitting a separate un-suffixed prompt for
        # the use read would confound position with prompt.
        if answer_reads == "built":
            for source, entry in ((binding[0], fname), (binding[1], fname),
                                  (defuse, "compute"), (alias, "run"),
                                  (call, "main"), (arith, "total")):
                items += _positioned_items(source, _answer_suffix(tokenizer, entry))
        else:
            items += value_items

        literal, targets, distractors, tname = _TYPES[i % len(_TYPES)]
        items.append(_typeof(i, literal, targets, distractors, tname))
        items.append(_loopvar(i, ["hello", "world", "alpha", "gamma"][i % 4]))

    scorable, dropped = {}, {}
    for item in items:
        good_targets = [w for w in item.target_words if _is_single_token(tokenizer, w)]
        good_distractors = [w for w in item.distractor_words
                            if _is_single_token(tokenizer, w)]
        scorable[item.item_id] = bool(good_targets)
        item.target_words = good_targets
        item.distractor_words = good_distractors

    # A pair is scorable only if both of its arms are.
    by_pair: dict[str, list[ProbeItem]] = {}
    for item in items:
        if item.pair_id:
            by_pair.setdefault(item.pair_id, []).append(item)
    for pair_id, arms in by_pair.items():
        if not all(scorable[a.item_id] for a in arms):
            for a in arms:
                scorable[a.item_id] = False

    kept = []
    for item in items:
        if scorable[item.item_id]:
            kept.append(item)
        else:
            dropped[item.family] = dropped.get(item.family, 0) + 1

    if dropped:
        logger.warning("dropped %d unscorable items (multi-token target, or the "
                       "other arm of a crossed pair): %s",
                       sum(dropped.values()), dropped)
    return Suite(name=name, items=kept, dropped_multitoken=dropped,
                 answer_reads=answer_reads)


def _is_single_token(tokenizer, word: str) -> bool:
    """One token with a leading space, or one token bare. Either counts."""
    for form in (" " + word, word):
        if len(tokenizer(form, add_special_tokens=False)["input_ids"]) == 1:
            return True
    return False


def target_token_ids(tokenizer, words: Sequence[str]) -> list[int]:
    """Every single-token spelling of every word — space-prefixed and bare.

    Both forms are kept because which one a model prefers is a fact about its
    tokenizer, not about whether it represents the concept, and scoring only
    one form would penalise a model for a segmentation convention.
    """
    ids: list[int] = []
    for word in words:
        for form in (" " + word, word):
            enc = tokenizer(form, add_special_tokens=False)["input_ids"]
            if len(enc) == 1 and enc[0] not in ids:
                ids.append(int(enc[0]))
    return ids


def resolve_position(tokenizer, prompt: str, anchor: str,
                     input_ids: Optional[Sequence[int]] = None) -> int:
    """Index of the LAST token of `anchor` inside the tokenized prompt.

    Resolved against the real tokenization every time rather than stored, so an
    item survives a tokenizer that segments a literal differently. Raises if the
    anchor is absent or ambiguous: a silently wrong read position would make
    every number downstream a statement about the wrong token.
    """
    count = prompt.count(anchor)
    if count == 0:
        raise ValueError(f"anchor {anchor!r} does not occur in the prompt")
    if count > 1:
        raise ValueError(f"anchor {anchor!r} occurs {count} times; make it unique")
    char_end = prompt.index(anchor) + len(anchor)

    if input_ids is None:
        input_ids = tokenizer(prompt)["input_ids"]
    # Walk the decoded prefix forward: robust to BOS, to byte-level merges and
    # to tokenizers with no offset mapping.
    for i in range(len(input_ids)):
        decoded = tokenizer.decode(input_ids[: i + 1], skip_special_tokens=True)
        if len(decoded) >= char_end:
            return i
    raise ValueError(f"anchor {anchor!r} not reached within {len(input_ids)} tokens")
