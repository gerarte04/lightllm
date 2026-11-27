"""Logits with the KV cache must match clean full-sequence forward passes.

Replaces scripts/compare_logits_with_cache.py. Because attention is causal,
logits[:, t] of a full clean pass (cache=None) must equal the logits produced
for token t during cached decoding: full prefill in one pass, prefill of a
prefix followed by chunks (decoding), and token-by-token decoding. Also checks
clean-pass determinism, that every layer writes to the cache, and that greedy
generation produces identical tokens with and without the cache.

Run:
    pytest tests/test_logits_with_cache.py -v
"""

import pytest
import torch
from torch.nn import functional as F

from tests.helpers import make_cache

PROMPT = "The capital of France is Paris, and the capital of Germany is"
GEN_TOKENS = 16
CLEAN_PASSES = 3


@pytest.fixture(scope="module")
def input_ids(env):
    _, tokenizer, _, device = env
    return tokenizer(PROMPT, return_tensors="pt").input_ids.to(device)


@pytest.fixture(scope="module")
def ref_logits(env, input_ids):
    """Reference: one clean forward pass over the full sequence (cache=None)."""
    _, _, model, _ = env
    with torch.no_grad():
        return model(input_ids)


@torch.no_grad()
def cached_forward(model, config, input_ids, prefill, chunk_size, device):
    """Prefill `prefill` tokens, then consume the rest in chunks of `chunk_size`."""
    B, T = input_ids.shape
    cache = make_cache(config, B, T, device)

    chunks = []
    pos = 0
    while pos < T:
        size = prefill if pos == 0 else min(chunk_size, T - pos)
        size = min(size, T - pos)
        chunk = input_ids[:, pos:pos + size]
        logits = model(chunk, cache=cache, start_pos=pos)
        chunks.append(logits)
        pos += size

    assert pos == T
    return torch.cat(chunks, dim=1), cache


def assert_logits_match(ref, cached):
    assert ref.shape == cached.shape, f"shape mismatch: {ref.shape} vs {cached.shape}"
    diff = (ref - cached).abs()
    max_diff = diff.max().item()
    # pass criterion of the original script: max|diff| < 1e-3 and argmax agreement
    assert max_diff < 1e-3, f"max abs diff {max_diff:.3e}, mean {diff.mean().item():.3e}"

    cos = F.cosine_similarity(
        ref.flatten().cpu().double(), cached.flatten().cpu().double(), dim=0).item()
    assert cos > 0.9999, f"cosine similarity {cos:.8f}"

    assert torch.equal(ref.argmax(-1), cached.argmax(-1)), "argmax mismatch"


def test_clean_forward_is_deterministic(env, input_ids, ref_logits):
    """Repeated clean passes without cache must give identical logits."""
    _, _, model, _ = env
    with torch.no_grad():
        for _ in range(CLEAN_PASSES - 1):
            other = model(input_ids)
            assert torch.equal(ref_logits, other), \
                f"non-deterministic: max diff {(ref_logits - other).abs().max().item():.3e}"


@pytest.mark.parametrize("chunk_size", [1, 4], ids=["chunk1", "chunk4"])
def test_full_prefill_matches_clean(env, input_ids, ref_logits, chunk_size):
    """Full prefill in one cached pass (start_pos=0, T tokens)."""
    config, _, model, device = env
    cached, _ = cached_forward(model, config, input_ids,
                               prefill=input_ids.size(1), chunk_size=chunk_size, device=device)
    assert_logits_match(ref_logits, cached)


@pytest.mark.parametrize("chunk_size", [1, 4], ids=["chunk1", "chunk4"])
def test_partial_prefill_matches_clean(env, input_ids, ref_logits, chunk_size):
    """Prefill half the prompt, then decode the rest in chunks."""
    config, _, model, device = env
    prefill = max(1, input_ids.size(1) // 2)
    cached, cache = cached_forward(model, config, input_ids,
                                   prefill=prefill, chunk_size=chunk_size, device=device)
    assert_logits_match(ref_logits, cached)

    # every layer must have written to the cache (layer_idx bug guard)
    filled = cache.k.abs().sum(dim=(1, 2, 3, 4))
    assert (filled > 0).all(), "some layers never wrote to the cache"


def test_token_by_token_decoding_matches_clean(env, input_ids, ref_logits):
    """Single-token prefill, then 1 token per cached pass."""
    config, _, model, device = env
    cached, _ = cached_forward(model, config, input_ids,
                               prefill=1, chunk_size=1, device=device)
    assert_logits_match(ref_logits, cached)


def test_greedy_generate_with_cache_matches(env, input_ids):
    """Greedy generation must produce identical tokens with and without cache."""
    config, _, model, device = env
    plain = model.naive_generate(input_ids, max_length=GEN_TOKENS, temperature=0.0)
    cache = make_cache(config, input_ids.size(0), input_ids.size(1) + GEN_TOKENS, device)
    cached = model.naive_generate(input_ids, max_length=GEN_TOKENS, cache=cache, temperature=0.0)
    assert torch.equal(plain, cached), "greedy tokens differ with/without cache"
