"""Shared helpers for the model-comparison tests.

Model loading / cache allocation utilities formerly lived in
scripts/compare_logits_with_hf.py and scripts/compare_logits_with_cache.py.
"""

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from core.cache import Cache
from core.model import Transformer

MODEL_ID = "HuggingFaceTB/SmolLM2-135M"


def load_our_model(config, device):
    """Instantiate core.model.Transformer and load HF weights with strict=True."""
    model = Transformer(config)

    weights_path = hf_hub_download(MODEL_ID, "model.safetensors")
    state_dict = load_file(weights_path)

    # SmolLM2 ties input/output embeddings, so the checkpoint may omit
    # lm_head.weight. Our model ties them too; supply the key so that
    # strict=True loading sees exactly the same set of parameters.
    if config.tie_word_embeddings and "lm_head.weight" not in state_dict:
        state_dict["lm_head.weight"] = state_dict["model.embed_tokens.weight"]

    incompatible = model.load_state_dict(state_dict, strict=True)
    assert not incompatible.missing_keys and not incompatible.unexpected_keys

    return model.to(device=device, dtype=torch.float32).eval()


def make_cache(config, batch_size: int, max_len: int, device, dtype=torch.float32) -> Cache:
    """Allocate a KV cache sized for `config` (GQA: num_key_value_heads)."""
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    return Cache(
        n_layers=config.num_hidden_layers,
        batch_size=batch_size,
        n_heads=config.num_key_value_heads,
        max_len=max_len,
        head_dim=head_dim,
        device=device,
        dtype=dtype,
    )


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()
