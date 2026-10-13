"""Stage 280 steer: what each domain's binding direction surfaces, and the J-lens at its layer.

For every direction from `evaluate` (synthetic, cruxeval, real: the difference of
class means of the use-token state, survives minus overwritten, at the layer its
probe chose) and for test snippets from every domain:

  code section   the program up to the end of the line holding the use, so its last
                 token is where the decoder has read everything that decides the label
  J-lens         top-10 tokens of the published Jacobian lens (stage 201's fitted
                 `results/workspace_lens/<model>/j-lens`) at the last token, at the
                 direction's layer, unsteered and with the steering vector added
  steering       the direction, scaled by c x (distance between the class means), added
                 to the residual at that layer at the last token and every generated
                 token; records the next-token top-10, a greedy continuation, and the
                 tokens whose logits it raises and lowers on average
  direction      the J-lens and logit-lens reading of the bare direction, both signs

A layer the lens was not fitted at (the last two blocks) is read with the logit lens
and labelled as such.
"""
from __future__ import annotations

import logging
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from src.cruxeval.artifacts import checked_gate, read_json, read_jsonl, register_files, sha256, stage_run, write_jsonl
from src.defsurvival.pipeline import DOMAINS, tokenizer_sha

log = logging.getLogger(__name__)


def code_section(program, row):
    """Token ids up to the last token that starts before the end of the use's line."""
    lines = program["source"].splitlines(keepends=True)
    end = sum(len(x) for x in lines[:row["u_line"]]) - lines[row["u_line"] - 1].endswith("\n")
    last = max(i for i, (a, b) in enumerate(program["offsets"]) if a < end and b > a)
    assert last >= row["u_anchor"], "Code section must contain the use"
    return program["input_ids"][:last + 1]


class Residual:
    """Forward hook on one block: optionally adds `vector` at the last position, and keeps
    that position's (pre-steering) output."""

    def __init__(self, block):
        self.vector, self.state = None, None
        self.handle = block.register_forward_hook(self.hook)

    def hook(self, module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        self.state = hidden[0, -1].detach().float().clone()
        if self.vector is not None:
            hidden[:, -1, :] += self.vector.to(device=hidden.device, dtype=hidden.dtype)
        return output

    def remove(self):
        self.handle.remove()


def greedy(model, ids, n):
    """First-step logits and `n` greedy tokens, with the KV cache."""
    import torch
    out = model(input_ids=ids, use_cache=True)
    first = out.logits[0, -1].float()
    token, past, tokens = first.argmax(), out.past_key_values, []
    for _ in range(n):
        tokens.append(int(token))
        out = model(input_ids=token.view(1, 1), past_key_values=past, use_cache=True)
        past, token = out.past_key_values, out.logits[0, -1].argmax()
    return first, torch.tensor(tokens)


def steer(prepared, evaluation, output, lens_dir=None, n_prompts=40, coefficients=(-8.0, -4.0, 4.0, 8.0),
          max_new_tokens=24, device="cuda", dtype="float16", seed=42):
    import jlens
    import torch

    from src.models.hooks import HookManager
    from src.models.loader import ModelConfig, ModelLoader
    from src.workspace_lens.fitting import load_lens
    from src.workspace_lens.readout import place_lens_jacobians, top_tokens
    prepared, evaluation, output = Path(prepared), Path(evaluation), Path(output)
    checked_gate(prepared, "280_prepare")
    checked_gate(evaluation, "280_evaluate")
    meta = read_json(prepared / "meta.json")
    lens_dir = Path(lens_dir or Path("results/workspace_lens") / meta["model"] / "j-lens")
    info = read_json(evaluation / "directions.json")
    units = np.load(evaluation / "directions.npz")
    programs = {p["program_id"]: p for p in read_jsonl(prepared / "programs.jsonl")}
    cands = pd.read_csv(prepared / "candidates.csv")
    args = dict(prepared_sha256=sha256(prepared / "gates.json"), evaluation_sha256=sha256(evaluation / "gates.json"),
                lens=str(lens_dir), n_prompts=n_prompts, coefficients=list(coefficients),
                max_new_tokens=max_new_tokens, device=device, dtype=dtype, seed=seed)

    with stage_run(output, "280_steer", args) as gate:
        cfg = ModelConfig.from_registry(meta["model"], device=device, dtype=getattr(torch, dtype))
        loader = ModelLoader(cfg)
        tokenizer, model = loader.tokenizer, loader.model
        model.eval()
        assert tokenizer_sha(tokenizer) == meta["tokenizer_sha256"], "Tokenizer changed"
        lens, provenance = load_lens(lens_dir)
        assert lens.d_model == cfg.d_model, "Lens was fitted on a different model width"
        fitted_on = provenance.get("model", {}).get("hf_id")
        assert fitted_on in (None, cfg.hf_id), f"Lens was fitted on {fitted_on}, not {cfg.hf_id}"
        lens_model = jlens.from_hf(model, tokenizer)  # For its final-norm + unembedding only.
        model_device = model.get_input_embeddings().weight.device
        place_lens_jacobians({"j-lens": lens}, sorted({d["layer"] for d in info.values()}), model_device)
        blocks = HookManager(model)._get_decoder_layers()
        decode = lambda ids: tokenizer.decode([int(i) for i in ids], skip_special_tokens=True)  # noqa: E731

        def read(state, layer):
            """(lens name, top-10) for a residual at `layer`."""
            if layer in lens.jacobians:
                return "j-lens", top_tokens(lens_model.unembed(lens.transport(state[None], layer))[0].float().cpu(), tokenizer)
            return "logit-lens", top_tokens(lens_model.unembed(state[None])[0].float().cpu(), tokenizer)

        directions = {}
        for name in DOMAINS:
            unit = torch.tensor(units[f"{name}_unit"], device=model_device, dtype=torch.float32)
            directions[name] = dict(info[name], unit=unit)

        direction_rows = []
        for name, d in directions.items():
            for sign in (1, -1):
                vector = sign * d["delta_norm"] * d["unit"]
                lens_name, top = read(vector, d["layer"])
                direction_rows.append(dict(direction=name, layer=d["layer"], sign="+survives" if sign > 0 else "+overwritten",
                                           lens=lens_name, top10=" | ".join(t for t, _, _ in top)))

        # Test snippets: one candidate per program, n_prompts programs per domain.
        rng = random.Random(seed)
        snippets = []
        for domain in DOMAINS:
            part = cands[(cands.domain == domain) & (cands.split == "test")]
            chosen = rng.sample(sorted(part.program_id.unique()), min(n_prompts, part.program_id.nunique()))
            for program_id in chosen:
                row = part[part.program_id == program_id].sample(1, random_state=seed).iloc[0].to_dict()
                snippets.append((domain, row, code_section(programs[program_id], row)))

        lens_rows, steering_rows = [], []
        changes = {}
        hooks = {d["layer"]: Residual(blocks[d["layer"]]) for d in directions.values()}
        try:
            for n, (domain, row, ids) in enumerate(snippets):
                tensor = torch.tensor([ids], device=model_device)
                with torch.no_grad():
                    model(input_ids=tensor, use_cache=False)  # Hooks keep the code section's last-token states.
                    states = {layer: hook.state.clone() for layer, hook in hooks.items()}
                    base_logits, base_tokens = greedy(model, tensor, max_new_tokens)
                common = dict(test_domain=domain, program_id=row["program_id"], label=int(row["label"]),
                              category=row["category"], code_tail=decode(ids[-24:]))
                for name, d in directions.items():
                    lens_name, top = read(states[d["layer"]], d["layer"])
                    lens_rows.append(dict(common, direction=name, layer=d["layer"], lens=lens_name,
                                          top10=" | ".join(t for t, _, _ in top)))
                    for c in coefficients:
                        vector = c * d["delta_norm"] * d["unit"]
                        hooks[d["layer"]].vector = vector
                        try:
                            with torch.no_grad():
                                logits, tokens = greedy(model, tensor, max_new_tokens)
                        finally:
                            hooks[d["layer"]].vector = None
                        _, steered_top = read(states[d["layer"]] + vector.to(states[d["layer"]].device), d["layer"])
                        delta = (torch.log_softmax(logits, -1) - torch.log_softmax(base_logits, -1)).cpu()
                        key = (name, domain, c)
                        changes[key] = changes.get(key, 0) + delta
                        steering_rows.append(dict(
                            common, direction=name, layer=d["layer"], coefficient=c,
                            relative_norm=float(c * d["delta_norm"] / states[d["layer"]].norm()),
                            next_top10=[t for t, _, _ in top_tokens(logits.cpu(), tokenizer)],
                            base_next_top10=[t for t, _, _ in top_tokens(base_logits.cpu(), tokenizer)],
                            continuation=decode(tokens), base_continuation=decode(base_tokens),
                            steered_lens_top10=[t for t, _, _ in steered_top]))
                if (n + 1) % 10 == 0:
                    log.info("Steered %d/%d snippets", n + 1, len(snippets))
        finally:
            for hook in hooks.values():
                hook.remove()

        counts = Counter((domain for domain, _, _ in snippets))
        promoted = []
        for (name, domain, c), total in changes.items():
            mean = total / counts[domain]
            for rank, (up, down) in enumerate(zip(torch.topk(mean, 10).indices, torch.topk(-mean, 10).indices), 1):
                promoted.append(dict(direction=name, test_domain=domain, coefficient=c, rank=rank,
                                     raised=decode([up]), raised_change=float(mean[up]),
                                     lowered=decode([down]), lowered_change=float(mean[down])))

        lens_rows, promoted, direction_rows = pd.DataFrame(lens_rows), pd.DataFrame(promoted), pd.DataFrame(direction_rows)
        lens_rows.to_csv(output / "jlens_top10.csv", index=False)
        promoted.to_csv(output / "promoted.csv", index=False)
        direction_rows.to_csv(output / "direction_lens.csv", index=False)
        write_jsonl(output / "steering.jsonl", steering_rows)
        (output / "report.md").write_text(steer_report(lens_rows, promoted, direction_rows, steering_rows, coefficients))
        register_files(gate, output, ["jlens_top10.csv", "promoted.csv", "direction_lens.csv", "steering.jsonl", "report.md"])
    return output


def steer_report(lens_rows, promoted, direction_rows, steering_rows, coefficients):
    lines = ["# Binding in effect: steering and J-lens", "",
             "## What each bare direction reads as", "",
             direction_rows.to_markdown(index=False), "",
             "## J-lens at the last token of the code section, at each direction's layer (unsteered)", "",
             "Tokens most often in the top 10, across the test snippets of each domain.", ""]
    for (name, domain), part in lens_rows.groupby(["direction", "test_domain"]):
        counts = Counter(t for row in part.top10 for t in row.split(" | "))
        common = ", ".join(f"`{t!r}` {c}" for t, c in counts.most_common(12))
        lines.append(f"- **{name} layer, {domain} code** ({len(part)} snippets, {part.lens.iloc[0]}): {common}")
    strongest = max(coefficients, key=abs)
    lines += ["", f"## Tokens each direction raises and lowers (coefficient {strongest:+g}, mean log-prob change)", ""]
    for (name, domain), part in promoted[promoted.coefficient == strongest].groupby(["direction", "test_domain"]):
        up = ", ".join(f"`{t!r}`" for t in part.sort_values("rank").raised)
        down = ", ".join(f"`{t!r}`" for t in part.sort_values("rank").lowered)
        lines += [f"- **{name} direction on {domain} code** — raised: {up}; lowered: {down}"]
    lines += ["", f"## Example continuations (coefficient {strongest:+g} vs unsteered)", ""]
    shown = Counter()
    for r in steering_rows:
        if r["coefficient"] != strongest or shown[r["direction"], r["test_domain"]] >= 2:
            continue
        shown[r["direction"], r["test_domain"]] += 1
        lines += [f"- {r['direction']} direction, {r['test_domain']} ({r['category']}), …`{r['code_tail']!r}`",
                  f"  - unsteered: `{r['base_continuation']!r}`", f"  - steered: `{r['continuation']!r}`"]
    lines += ["", "Every snippet, coefficient and continuation: `steering.jsonl`. Per-snippet J-lens: `jlens_top10.csv`.", ""]
    return "\n".join(lines)
