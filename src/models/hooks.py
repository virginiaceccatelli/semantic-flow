"""PyTorch forward hooks for extracting hidden states from transformer layers."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Generator, Optional

import torch
import torch.nn as nn


class ActivationCache:
    """Stores hidden states captured by forward hooks, indexed by layer."""

    def __init__(self):
        self._cache: dict[int, torch.Tensor] = {}

    def store(self, layer_idx: int, hidden: torch.Tensor):
        # Detach, move to CPU, and drop the batch dim (we always run one example at a time).
        self._cache[layer_idx] = hidden.detach().cpu().squeeze(0)

    def get(self, layer_idx: int) -> torch.Tensor:
        return self._cache[layer_idx]

    def layers(self) -> list[int]:
        return sorted(self._cache.keys())

    def all_hidden_states(self) -> torch.Tensor:
        """Return tensor of shape (n_layers, seq_len, d_model) sorted by layer index."""
        layers = self.layers()
        return torch.stack([self._cache[l] for l in layers], dim=0)

    def clear(self):
        self._cache.clear()

    def __len__(self) -> int:
        return len(self._cache)

    def __repr__(self) -> str:
        if not self._cache:
            return "ActivationCache(empty)"
        first = next(iter(self._cache.values()))
        return f"ActivationCache(layers={self.layers()}, shape={tuple(first.shape)})"


class HookManager:
    """Register and remove forward hooks on transformer decoder layers."""

    def __init__(self, model: nn.Module, layer_indices: Optional[list[int]] = None):
        self.model = model
        self.layer_indices = layer_indices
        self._handles: list[torch.utils.hooks.RemovableHook] = []
        self.cache = ActivationCache()

    def _get_decoder_layers(self) -> list[tuple[int, nn.Module]]:
        """Return (index, module) pairs for transformer decoder layers.

        Tries common attribute names used across model families.
        """
        for attr in ("layers", "h", "blocks", "decoder_layers"):
            layers = getattr(self.model, attr, None)
            if layers is None:
                # Try nested (e.g., model.model.layers for LlamaForCausalLM)
                inner = getattr(self.model, "model", None)
                if inner is not None:
                    layers = getattr(inner, attr, None)
            if layers is not None:
                return [(i, l) for i, l in enumerate(layers)]
        raise RuntimeError(
            "Could not locate decoder layers. Inspect model.named_modules() "
            "and pass layer_indices explicitly."
        )

    def _make_hook(self, layer_idx: int) -> Callable:
        cache = self.cache

        def hook(module, input, output):
            # output is typically (hidden_states, ...) or just hidden_states
            hidden = output[0] if isinstance(output, tuple) else output
            cache.store(layer_idx, hidden)

        return hook

    def register(self):
        """Attach hooks to the requested layers."""
        all_layers = self._get_decoder_layers()
        indices = self.layer_indices if self.layer_indices is not None else [i for i, _ in all_layers]
        idx_set = set(indices)
        for i, layer in all_layers:
            if i in idx_set:
                handle = layer.register_forward_hook(self._make_hook(i))
                self._handles.append(handle)

    def remove(self):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    @contextmanager
    def active(self) -> Generator["HookManager", None, None]:
        """Context manager that registers hooks, yields, then removes them."""
        self.cache.clear()
        self.register()
        try:
            yield self
        finally:
            self.remove()


@torch.no_grad()
def extract_hidden_states(
    model: nn.Module,
    input_ids: torch.Tensor,
    layer_indices: Optional[list[int]] = None,
    attention_mask: Optional[torch.Tensor] = None,
) -> ActivationCache:
    """Run a forward pass and return hidden states for the specified layers."""
    manager = HookManager(model, layer_indices=layer_indices)
    with manager.active():
        model(input_ids=input_ids, attention_mask=attention_mask)
    return manager.cache


def patch_activations(
    model: nn.Module,
    input_ids: torch.Tensor,
    patches: dict[int, torch.Tensor],
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run a forward pass with activation patches at specified layers.

    patches: {layer_idx: replacement_hidden_states (seq_len, d_model)}

    Returns logits tensor. Used for Phase 5 causal intervention experiments.
    """
    handles: list[torch.utils.hooks.RemovableHook] = []

    def _get_layers():
        for attr in ("layers", "h", "blocks", "decoder_layers"):
            layers = getattr(model, attr, None)
            if layers is None:
                inner = getattr(model, "model", None)
                if inner is not None:
                    layers = getattr(inner, attr, None)
            if layers is not None:
                return list(enumerate(layers))
        raise RuntimeError("Could not locate decoder layers for patching.")

    for layer_idx, layer in _get_layers():
        if layer_idx not in patches:
            continue
        patch = patches[layer_idx]

        def make_patch_hook(p):
            def hook(module, input, output):
                hidden = output[0] if isinstance(output, tuple) else output
                # Replace with patch, keeping rest of output tuple intact
                if isinstance(output, tuple):
                    return (p.to(hidden.device),) + output[1:]
                return p.to(hidden.device)
            return hook

        handles.append(layer.register_forward_hook(make_patch_hook(patch)))

    try:
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask)
        return out.logits
    finally:
        for h in handles:
            h.remove()
