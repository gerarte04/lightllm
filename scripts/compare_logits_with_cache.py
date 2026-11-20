#!/usr/bin/env python3
"""Compare logits of core.model.Transformer with and without the KV cache.

Reference: several clean forward passes over the full sequence (cache=None).
Cached runs: prefill a prefix, then feed the remaining tokens in chunks
(1 token per step = decoding, or --chunk-size > 1), passing start_pos so that
RoPE positions and the causal mask stay aligned.

Because attention is causal, logits[:, t] of a full clean pass must equal the
logits produced for token t during cached decoding.

Usage:
    python scripts/compare_logits_with_cache.py
    python scripts/compare_logits_with_cache.py --prompt "..." --chunk-size 4 --device mps
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.nn import functional as F
from transformers import AutoConfig, AutoTokenizer

from core.cache import Cache
from scripts.compare_logits_with_hf import MODEL_ID, load_our_model


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


@torch.no_grad()
def clean_forward(model, input_ids, n_passes: int, device):
    """Several independent full-sequence forward passes without any cache."""
    runs = []
    for i in range(n_passes):
        logits = model(input_ids)
        sync(device)
        runs.append(logits.clone())
        print(f"[clean pass {i + 1}/{n_passes}] shape={tuple(logits.shape)}, "
              f"last-pos argmax={logits[0, -1].argmax().item()}")

    ref = runs[0]
    for i, other in enumerate(runs[1:], start=2):
        d = (ref - other).abs().max().item()
        print(f"[clean pass {i}] max abs diff vs pass 1: {d:.3e} "
              f"({'deterministic' if d == 0 else 'NON-DETERMINISTIC'})")
    return ref


@torch.no_grad()
def cached_forward(model, config, input_ids, prefill: int, chunk_size: int, device):
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
        sync(device)
        chunks.append(logits.clone())
        pos += size

    assert pos == T
    return torch.cat(chunks, dim=1), cache


def report(name, ref, cached, input_ids, tokenizer):
    assert ref.shape == cached.shape, f"shape mismatch: {ref.shape} vs {cached.shape}"
    diff = (ref - cached).abs()
    per_pos = diff.amax(dim=(0, 2))  # max abs diff per sequence position
    argmax_match = (ref.argmax(-1) == cached.argmax(-1)).float().mean().item()

    print(f"\n=== {name} ===")
    print(f"max abs diff:            {diff.max().item():.3e}")
    print(f"mean abs diff:           {diff.mean().item():.3e}")
    print(f"RMSE:                    {diff.pow(2).mean().sqrt().item():.3e}")
    print(f"max diff / max |logit|:  {(diff.max() / ref.abs().max()).item():.3e}")
    cos = F.cosine_similarity(ref.flatten().cpu().double(), cached.flatten().cpu().double(), dim=0).item()
    print(f"cosine similarity:       {cos:.8f}")
    print(f"allclose(atol=1e-4):     {torch.allclose(ref, cached, atol=1e-4, rtol=1e-4)}")
    print(f"allclose(atol=1e-3):     {torch.allclose(ref, cached, atol=1e-3, rtol=1e-3)}")
    print(f"argmax agreement:        {argmax_match * 100:.2f}% of all positions")
    worst = per_pos.argmax().item()
    print(f"worst position:          t={worst} (max abs diff {per_pos[worst].item():.3e}), "
          f"token={tokenizer.decode([ref[0, worst].argmax().item()])!r}")

    print(f"\n{'pos':>4} {'token':<14} {'max|diff|':>12} {'argmax ref':>11} {'argmax cache':>13}")
    for t in range(ref.shape[1]):
        tok = repr(tokenizer.decode([input_ids[0, t].item()]))
        print(f"{t:>4} {tok:<14} {per_pos[t].item():>12.3e} "
              f"{ref[0, t].argmax().item():>11} {cached[0, t].argmax().item():>13}")

    return diff.max().item(), argmax_match


@torch.no_grad()
def check_greedy_generate(model, config, input_ids, n_tokens, device):
    """Greedy generation must produce identical tokens with and without cache."""
    plain = model.generate(input_ids, max_length=n_tokens, temperature=0.0)
    cache = make_cache(config, input_ids.size(0), input_ids.size(1) + n_tokens, device)
    cached = model.generate(input_ids, max_length=n_tokens, cache=cache, temperature=0.0)
    same = torch.equal(plain, cached)
    print(f"\n=== greedy generate ({n_tokens} new tokens) ===")
    print(f"identical token ids:     {same}")
    if not same:
        first = (plain != cached).nonzero()[0].tolist()
        print(f"first mismatch at {first}: plain={plain[tuple(first)].item()}, "
              f"cached={cached[tuple(first)].item()}")
    return same


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompt", default="The capital of France is Paris, and the capital of Germany is",
                        help="input text for the forward passes")
    parser.add_argument("--device", default="cpu", help="cpu | cuda | mps")
    parser.add_argument("--clean-passes", type=int, default=3,
                        help="number of clean forward passes without cache (default: 3)")
    parser.add_argument("--prefill", type=int, default=0,
                        help="tokens to prefill in the first cached pass, 0 = half the prompt (default: 0)")
    parser.add_argument("--chunk-size", type=int, default=1,
                        help="tokens per cached forward pass after prefill (default: 1)")
    parser.add_argument("--gen-tokens", type=int, default=16,
                        help="tokens for the greedy generate cross-check, 0 = skip (default: 16)")
    args = parser.parse_args()

    device = torch.device(args.device)
    config = AutoConfig.from_pretrained(MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids.to(device)
    T = input_ids.shape[1]
    prefill = args.prefill if args.prefill > 0 else max(1, T // 2)
    prefill = min(prefill, T)

    print(f"model: {MODEL_ID}\ndevice: {device}, dtype: float32")
    print(f"prompt: {args.prompt!r}")
    print(f"input_ids ({T} tokens): {input_ids[0].tolist()}")
    print(f"prefill={prefill}, chunk_size={args.chunk_size}, clean_passes={args.clean_passes}\n")

    model = load_our_model(config, device)

    ref = clean_forward(model, input_ids, args.clean_passes, device)

    results = []
    # 1. full prefill in one cached pass (start_pos=0, T tokens)
    full, _ = cached_forward(model, config, input_ids, prefill=T, chunk_size=args.chunk_size, device=device)
    name = "cache: full prefill (1 pass)"
    results.append((name, *report(name, ref, full, input_ids, tokenizer)))

    # 2. prefill a prefix, then decode the rest
    part, cache = cached_forward(model, config, input_ids, prefill=prefill,
                                 chunk_size=args.chunk_size, device=device)
    name = f"cache: prefill {prefill} + chunks of {args.chunk_size}"
    results.append((name, *report(name, ref, part, input_ids, tokenizer)))
    filled = cache.k.abs().sum(dim=(1, 2, 3, 4))
    print(f"\ncache filled per layer (sum |k|): "
          f"{['%.1f' % v for v in filled.tolist()]}")
    assert (filled > 0).all(), "some layers never wrote to the cache (layer_idx bug?)"

    # 3. token-by-token decoding from a single-token prefill
    single, _ = cached_forward(model, config, input_ids, prefill=1, chunk_size=1, device=device)
    name = "cache: prefill 1 + 1 token per pass"
    results.append((name, *report(name, ref, single, input_ids, tokenizer)))

    greedy_ok = True
    if args.gen_tokens > 0:
        greedy_ok = check_greedy_generate(model, config, input_ids, args.gen_tokens, device)

    print("\n=== Summary ===")
    for name, max_diff, agreement in results:
        status = "OK" if max_diff < 1e-3 and agreement == 1.0 else "FAIL"
        print(f"{status:<4} {name:<40} max|diff|={max_diff:.3e}, argmax={agreement * 100:.2f}%")
    if args.gen_tokens > 0:
        print(f"{'OK' if greedy_ok else 'FAIL':<4} greedy generate with/without cache")

    ok = greedy_ok and all(d < 1e-3 and a == 1.0 for _, d, a in results)
    print(f"\n{'PASS' if ok else 'FAIL'}: KV cache {'matches' if ok else 'DOES NOT match'} clean forward passes")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
