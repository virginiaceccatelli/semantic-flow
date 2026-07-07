"""Load code LLMs from HuggingFace and manage tokenization."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

MODEL_REGISTRY: dict[str, dict] = {
    "deepseek-coder-1.3b": {
        "hf_id": "deepseek-ai/deepseek-coder-1.3b-base",
        "n_layers": 24,
        "d_model": 2048,
    },
    "deepseek-coder-6.7b": {
        "hf_id": "deepseek-ai/deepseek-coder-6.7b-base",
        "n_layers": 32,
        "d_model": 4096,
    },
    "starcoder2-3b": {
        "hf_id": "bigcode/starcoder2-3b",
        "n_layers": 30,
        "d_model": 3072,
    },
    "starcoder2-7b": {
        "hf_id": "bigcode/starcoder2-7b",
        "n_layers": 32,
        "d_model": 4608,
    },
    "codellama-7b": {
        "hf_id": "codellama/CodeLlama-7b-hf",
        "n_layers": 32,
        "d_model": 4096,
    },
}


@dataclass
class ModelConfig:
    name: str
    hf_id: str
    n_layers: int
    d_model: int
    dtype: torch.dtype = torch.float16
    device: str = "cuda"
    probe_layers: list[int] = field(default_factory=list)

    def __post_init__(self):
        if not self.probe_layers:
            # Default: every 4th layer plus first and last
            self.probe_layers = (
                [0]
                + list(range(3, self.n_layers - 1, 4))
                + [self.n_layers - 1]
            )

    @classmethod
    def from_registry(cls, name: str, **kwargs) -> "ModelConfig":
        if name not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY.keys())}")
        info = MODEL_REGISTRY[name]
        return cls(name=name, **info, **kwargs)


_TOKENIZER_PROBE = "def func():\n    a = 17\n    return a"


def load_tokenizer(hf_id: str) -> PreTrainedTokenizerBase:
    """Load a tokenizer and VERIFY it round-trips code exactly.

    On transformers 5.x, AutoTokenizer resolves deepseek-coder to the slow
    sentencepiece LlamaTokenizer, which silently mis-tokenizes code
    ('def func' → ['de','ff','unc'], whitespace lost). PreTrainedTokenizerFast
    loads the repo's tokenizer.json directly and is correct, so try it first.
    Any tokenizer that fails the round-trip check is rejected rather than
    silently corrupting every downstream label and activation.
    """
    from transformers import PreTrainedTokenizerFast

    last_error: Optional[Exception] = None
    for loader_fn in (
        lambda: PreTrainedTokenizerFast.from_pretrained(hf_id),
        lambda: AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True),
    ):
        try:
            tok = loader_fn()
        except Exception as e:  # try the next loading strategy
            last_error = e
            continue
        ids = tok(_TOKENIZER_PROBE)["input_ids"]
        if tok.decode(ids, skip_special_tokens=True) == _TOKENIZER_PROBE:
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            return tok
    raise RuntimeError(
        f"No tokenizer for {hf_id} round-trips code exactly "
        f"(last load error: {last_error}). Refusing to continue with a "
        "tokenizer that would corrupt inputs and labels."
    )


class ModelLoader:
    """Load a pretrained code LLM and its tokenizer."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self._model: Optional[PreTrainedModel] = None
        self._tokenizer: Optional[PreTrainedTokenizerBase] = None

    @property
    def model(self) -> PreTrainedModel:
        if self._model is None:
            self._model = self._load_model()
        return self._model

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        if self._tokenizer is None:
            self._tokenizer = load_tokenizer(self.config.hf_id)
        return self._tokenizer

    def _load_model(self) -> PreTrainedModel:
        device_map = "auto" if self.config.device == "cuda" else self.config.device
        model = AutoModelForCausalLM.from_pretrained(
            self.config.hf_id,
            dtype=self.config.dtype,
            device_map=device_map,
            trust_remote_code=True,
        )
        model.eval()
        return model

    def tokenize(self, code: str, max_length: int = 2048) -> dict[str, torch.Tensor]:
        return self.tokenizer(
            code,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding=False,
        )

    def token_strings(self, input_ids: torch.Tensor) -> list[str]:
        """Convert token ids to human-readable token strings."""
        return [self.tokenizer.decode([t]) for t in input_ids.squeeze().tolist()]

    def unload(self):
        """Free GPU memory."""
        del self._model
        self._model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
