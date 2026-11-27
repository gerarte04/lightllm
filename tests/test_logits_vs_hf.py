"""Logits of core.model.Transformer must match HuggingFace Qwen3-0.6B (float32).

Replaces scripts/compare_logits_with_hf.py: our model is loaded from the same
HF checkpoint with strict=True, and its logits are compared against
AutoModelForCausalLM (eager attention) — absolute/relative closeness, cosine
similarity, argmax agreement and top-k overlap at the last position.

Run:
    pytest tests/test_logits_vs_hf.py -v
"""

import pytest
import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM

from tests.helpers import MODEL_ID

PROMPTS = [
    "The capital of France is",
    "Once upon a time, in a distant land, there lived a little dragon.",
]
TOP_K = 10


@pytest.fixture(scope="module")
def hf_model(env):
    _, _, _, device = env
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float32, attn_implementation="eager"
    )
    return model.to(device).eval()


def test_parameter_count_matches_hf(env, hf_model):
    """Same architecture: identical parameter count (both tie embeddings)."""
    _, _, model, _ = env
    ours = sum(p.numel() for p in model.parameters())
    hf = sum(p.numel() for p in hf_model.parameters())
    assert ours == hf


def test_embeddings_are_tied(env):
    config, _, model, _ = env
    assert config.tie_word_embeddings
    assert model.lm_head.weight is model.model.embed_tokens.weight


@pytest.mark.parametrize("prompt", PROMPTS)
def test_logits_match_hf(env, hf_model, prompt):
    _, tokenizer, model, device = env
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    with torch.no_grad():
        ours = model(input_ids)
        hf = hf_model(input_ids).logits

    assert ours.shape == hf.shape, f"shape mismatch: {ours.shape} vs {hf.shape}"
    diff = (ours - hf).abs()
    max_diff = diff.max().item()
    assert max_diff < 1e-3, f"max abs diff {max_diff:.3e}"
    assert torch.allclose(ours, hf, atol=1e-4, rtol=1e-4), \
        f"max abs diff {max_diff:.3e}, mean {diff.mean().item():.3e}"

    cos = F.cosine_similarity(
        ours.flatten().cpu().double(), hf.flatten().cpu().double(), dim=0).item()
    assert cos > 0.9999, f"cosine similarity {cos:.8f}"

    # both models must pick the same token at every position
    assert torch.equal(ours.argmax(-1), hf.argmax(-1)), "argmax mismatch"

    # and the same top-k candidates for the next token
    ours_top = set(ours[0, -1].topk(TOP_K).indices.tolist())
    hf_top = set(hf[0, -1].topk(TOP_K).indices.tolist())
    assert ours_top == hf_top, f"top-{TOP_K} overlap {len(ours_top & hf_top)}/{TOP_K}"
