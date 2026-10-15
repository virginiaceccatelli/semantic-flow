"""One-model activation extraction and prompt-position causal interventions."""
from __future__ import annotations
import re

import numpy as np
import torch
from .data import canonical_literal


def parse_answer(text):
    # Qwen's thinking text is not an answer; require the explicit final marker.
    if '<think>' in text and '</think>' not in text:
        return None
    text = text.split('</think>')[-1]
    matches = re.findall(r'FINAL:\s*([^\n]+)', text)
    if not matches:
        return None
    try:
        return canonical_literal(matches[-1].strip().strip('`'))
    except (ValueError, SyntaxError):
        return None


def swap_delta(host, donor, direction):
    direction = np.asarray(direction, dtype=np.float32)
    direction = direction / max(float(np.linalg.norm(direction)), 1e-12)
    return np.dot(donor - host, direction) * direction


def random_delta(delta, direction, seed):
    rng = np.random.default_rng(seed)
    r = rng.normal(size=len(delta))
    q = direction / max(float(np.linalg.norm(direction)), 1e-12)
    r -= np.dot(r, q) * q
    r /= max(float(np.linalg.norm(r)), 1e-12)
    return (r * np.linalg.norm(delta)).astype(np.float32)


class Engine:
    def __init__(self, model, tokenizer, max_input_tokens=8192, max_new_tokens=4096, seed=42):
        self.model, self.tokenizer = model, tokenizer
        self.layers = model.model.layers
        self.max_input_tokens, self.max_new_tokens = max_input_tokens, max_new_tokens
        self.device = model.get_input_embeddings().weight.device
        self.seed = seed

    def tokens(self, text):
        rendered = self.tokenizer.apply_chat_template(
            [{'role': 'user', 'content': text}], tokenize=False,
            add_generation_prompt=True, enable_thinking=True)
        ids = self.tokenizer(rendered, return_tensors='pt', add_special_tokens=False)['input_ids']
        if ids.shape[1] > self.max_input_tokens:
            raise ValueError(f'prompt has {ids.shape[1]} tokens (limit {self.max_input_tokens}); not truncated')
        return ids.to(self.device)

    @torch.inference_mode()
    def extract(self, ids):
        states = []
        handles = []
        def capture(module, inputs, output):
            x = output[0] if isinstance(output, tuple) else output
            states.append(x[0, -1].detach().float().cpu().numpy())
        try:
            for block in self.layers:
                handles.append(block.register_forward_hook(capture))
            self.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False,
                       logits_to_keep=1)
        finally:
            for h in handles:
                h.remove()
        return np.stack(states)

    @torch.inference_mode()
    def generate(self, ids, layer=None, delta=None):
        # Qwen recommends sampling in thinking mode. Reset the same seed for
        # clean and intervened runs, so an unmodified run is reproducible.
        torch.manual_seed(self.seed)
        handle = None
        calls = 0
        position = ids.shape[1] - 1
        def edit(module, inputs, output):
            nonlocal calls
            x = output[0] if isinstance(output, tuple) else output
            # Patch exactly once, on the final prompt token during cache prefill.
            if calls == 0:
                if x.shape[1] != ids.shape[1]:
                    raise RuntimeError('unexpected generation prefill shape')
                y = x.clone()
                y[0, position] += torch.as_tensor(delta, device=x.device, dtype=x.dtype)
                calls += 1
                return (y,) + output[1:] if isinstance(output, tuple) else y
            return output
        try:
            if layer is not None:
                handle = self.layers[layer].register_forward_hook(edit)
            output = self.model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                max_new_tokens=self.max_new_tokens, do_sample=True, temperature=0.6,
                top_p=0.95, top_k=20, use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id)
        finally:
            if handle is not None:
                handle.remove()
        if layer is not None and calls != 1:
            raise RuntimeError('intervention did not run exactly once')
        suffix = output[0, ids.shape[1]:]
        text = self.tokenizer.decode(suffix, skip_special_tokens=False)
        text = text.replace(self.tokenizer.eos_token or '', '')
        return dict(text=text, answer=parse_answer(text), new_tokens=len(suffix),
                    hit_token_limit=len(suffix) >= self.max_new_tokens)


def load_engine(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map='auto', attn_implementation='eager').eval()
    model.requires_grad_(False)
    return Engine(model, tokenizer, args.max_input_tokens, args.max_new_tokens, args.seed)
