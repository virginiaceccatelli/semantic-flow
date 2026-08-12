"""PyTorch forward hooks for extracting hidden states from transformer layers."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Generator, Optional, Sequence

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
        """Attach hooks to the requested layers.

        Index -1 hooks the input-embedding module: truly context-free token
        features (decoder-layer index 0 has already mixed context once)."""
        all_layers = self._get_decoder_layers()
        indices = self.layer_indices if self.layer_indices is not None else [i for i, _ in all_layers]
        idx_set = set(indices)
        if -1 in idx_set:
            emb = self.model.get_input_embeddings()
            self._handles.append(emb.register_forward_hook(self._make_hook(-1)))
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


@torch.no_grad()
def extract_hidden_states_and_logits(
    model: nn.Module,
    input_ids: torch.Tensor,
    layer_indices: Optional[list[int]] = None,
    attention_mask: Optional[torch.Tensor] = None,
) -> tuple[ActivationCache, torch.Tensor]:
    """`extract_hidden_states`, but the forward pass's logits are kept too.

    Experiments that need both (read a state *and* score the answer it leads
    to) would otherwise run the same forward pass twice.
    """
    manager = HookManager(model, layer_indices=layer_indices)
    with manager.active():
        out = model(input_ids=input_ids, attention_mask=attention_mask)
    return manager.cache, out.logits


@torch.no_grad()
def transform_positions(
    model: nn.Module,
    input_ids: torch.Tensor,
    transforms: dict[int, dict[int, Callable[[torch.Tensor], torch.Tensor]]],
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Like `patch_positions`, but each entry EDITS the live vector.

    transforms: {layer_idx: {position: fn(vec: (d,)) -> (d,)}}

    The distinction matters for interventions defined relative to the state
    they act on (`h + V (swap(c) - c)`): a replacement patch would need the
    patched value computed ahead of time, which is impossible when an earlier
    layer's edit has already changed what arrives here. Listing several layers
    therefore applies the edits in forward order, each seeing the previous
    one's effect — which is what a layer *band* means.

    Index -1 edits the input-embedding output, matching `HookManager`'s
    convention, so the context-free layer can be intervened on like any other.

    Returns the logits tensor.
    """
    manager = HookManager(model)  # reuse its decoder-layer discovery
    all_layers = manager._get_decoder_layers()
    handles: list = []

    def make_transform_hook(pos_transforms: dict[int, Callable]):
        def hook(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            for pos, fn in pos_transforms.items():
                edited = fn(hidden[0, pos, :])
                hidden[0, pos, :] = edited.to(device=hidden.device, dtype=hidden.dtype)
            if isinstance(output, tuple):
                return (hidden,) + output[1:]
            return hidden
        return hook

    if -1 in transforms:
        embeddings = model.get_input_embeddings()
        handles.append(embeddings.register_forward_hook(make_transform_hook(transforms[-1])))
    for layer_idx, layer in all_layers:
        if layer_idx in transforms:
            handles.append(layer.register_forward_hook(make_transform_hook(transforms[layer_idx])))

    try:
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        return out.logits
    finally:
        for h in handles:
            h.remove()


@torch.no_grad()
def transform_positions_batched(
    model: nn.Module,
    input_ids: torch.Tensor,
    layer: int,
    positions: Sequence[int],
    fns: Sequence[Callable[[torch.Tensor], torch.Tensor]],
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """One edit per ROW of a batch, at that row's own position, in one pass.

    `transform_positions` hardcodes batch index 0 and therefore forces one
    forward pass per intervention. At E13's prompt length (21 tokens, uniform
    across every generated cell) a 6.7B forward is almost entirely per-call
    overhead — hook registration, launch latency, host-device transfers — so
    batch size 1 wastes most of the GPU. Batching amortises all of it and the
    arithmetic is unchanged: row `i` gets exactly the edit it would have got
    alone.

    Requires equal-length sequences, which E13 guarantees by construction and
    `tests/test_binding.py` asserts. Returns the logits, shape (B, T, V).
    """
    if not (input_ids.shape[0] == len(positions) == len(fns)):
        raise ValueError(
            f"batch mismatch: {input_ids.shape[0]} sequences, "
            f"{len(positions)} positions, {len(fns)} edit functions")

    manager = HookManager(model)
    all_layers = manager._get_decoder_layers()
    handles: list = []

    def hook(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        for row, (pos, fn) in enumerate(zip(positions, fns)):
            edited = fn(hidden[row, int(pos), :])
            hidden[row, int(pos), :] = edited.to(
                device=hidden.device, dtype=hidden.dtype)
        return (hidden,) + output[1:] if isinstance(output, tuple) else hidden

    if int(layer) == -1:
        handles.append(model.get_input_embeddings().register_forward_hook(hook))
    for layer_idx, module in all_layers:
        if layer_idx == int(layer):
            handles.append(module.register_forward_hook(hook))

    try:
        return model(input_ids=input_ids, attention_mask=attention_mask).logits
    finally:
        for h in handles:
            h.remove()


@torch.no_grad()
def transform_and_capture(
    model: nn.Module,
    input_ids: torch.Tensor,
    transforms: dict[int, dict[int, Callable[[torch.Tensor], torch.Tensor]]],
    layer_indices: Optional[list[int]] = None,
    attention_mask: Optional[torch.Tensor] = None,
) -> tuple[ActivationCache, torch.Tensor]:
    """Edit some positions and read the resulting states, in ONE forward pass.

    E12's primary endpoint is internal: after installing another run's value at
    the injection site, what does a frozen decoder read at the *next*
    statement's anchor? Answering that needs the edit and the capture in the
    same pass, and needs the capture to see the edited stream.

    Hook order is what guarantees that. A forward hook returning a value
    replaces the module's output for every hook registered after it, so the
    transform hooks are registered first and the capturing hooks second; at the
    intervened layer the cache therefore holds the edited state, and at later
    layers it holds the states the edit produced.

    Returns (cache, logits).
    """
    manager = HookManager(model, layer_indices=layer_indices)
    all_layers = manager._get_decoder_layers()
    handles: list = []

    def make_transform_hook(pos_transforms: dict[int, Callable]):
        def hook(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            for pos, fn in pos_transforms.items():
                hidden[0, pos, :] = fn(hidden[0, pos, :]).to(
                    device=hidden.device, dtype=hidden.dtype)
            if isinstance(output, tuple):
                return (hidden,) + output[1:]
            return hidden
        return hook

    if -1 in transforms:
        embeddings = model.get_input_embeddings()
        handles.append(embeddings.register_forward_hook(make_transform_hook(transforms[-1])))
    for layer_idx, layer in all_layers:
        if layer_idx in transforms:
            handles.append(layer.register_forward_hook(make_transform_hook(transforms[layer_idx])))

    try:
        with manager.active():                    # registered second: sees the edit
            out = model(input_ids=input_ids, attention_mask=attention_mask)
        return manager.cache, out.logits
    finally:
        for h in handles:
            h.remove()


def transform_positions_with_grad(
    model: nn.Module,
    input_ids: torch.Tensor,
    transforms: dict[int, dict[int, Callable[[torch.Tensor], torch.Tensor]]],
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """`transform_positions` with autograd left ON — a sibling, not a variant.

    Every other stage in this pipeline runs forward-only, so `transform_positions`
    is decorated `@torch.no_grad()` and writes the edit in place. Learning an
    intervention subspace (`src/models/das.py`) needs the opposite on both
    counts: gradients must reach the parameters the edit is built from, and an
    in-place write into the hooked output would break the graph.

    The edit is therefore applied to a **clone**. A clone is a non-leaf tensor,
    so an index assignment into it is recorded by autograd with a well-defined
    backward, while the module's own output buffer is never mutated.

    The model's parameters are expected to be frozen by the caller; nothing
    here changes them.
    """
    manager = HookManager(model)  # reuse its decoder-layer discovery
    all_layers = manager._get_decoder_layers()
    handles: list = []

    def make_transform_hook(pos_transforms: dict[int, Callable]):
        def hook(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            edited = hidden.clone()
            for pos, fn in pos_transforms.items():
                new_vec = fn(hidden[0, pos, :])
                edited[0, pos, :] = new_vec.to(device=hidden.device, dtype=hidden.dtype)
            if isinstance(output, tuple):
                return (edited,) + output[1:]
            return edited
        return hook

    if -1 in transforms:
        embeddings = model.get_input_embeddings()
        handles.append(embeddings.register_forward_hook(make_transform_hook(transforms[-1])))
    for layer_idx, layer in all_layers:
        if layer_idx in transforms:
            handles.append(layer.register_forward_hook(make_transform_hook(transforms[layer_idx])))

    try:
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        return out.logits
    finally:
        for h in handles:
            h.remove()


@torch.no_grad()
def patch_positions(
    model: nn.Module,
    input_ids: torch.Tensor,
    patches: dict[int, dict[int, torch.Tensor]],
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run a forward pass with specific residual-stream POSITIONS replaced.

    patches: {layer_idx: {position: replacement_vector (d_model,)}}

    Only the listed positions are overwritten at each layer's output; all other
    positions flow through untouched. Returns the logits tensor.
    """
    manager = HookManager(model)  # reuse its decoder-layer discovery
    all_layers = manager._get_decoder_layers()
    handles: list = []

    def make_patch_hook(pos_patches: dict[int, torch.Tensor]):
        def hook(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            for pos, vec in pos_patches.items():
                hidden[:, pos, :] = vec.to(device=hidden.device, dtype=hidden.dtype)
            if isinstance(output, tuple):
                return (hidden,) + output[1:]
            return hidden
        return hook

    for layer_idx, layer in all_layers:
        if layer_idx in patches:
            handles.append(layer.register_forward_hook(make_patch_hook(patches[layer_idx])))

    try:
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        return out.logits
    finally:
        for h in handles:
            h.remove()
