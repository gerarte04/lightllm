#!/usr/bin/env python3
"""Benchmark Transformer.generate() with and without the KV cache (tokens/s).

Generates --max-tokens (default 500) new tokens twice: once recomputing the
whole sequence every step (cache=None) and once with a preallocated Cache.
Greedy decoding by default, so the two runs must produce identical tokens.

Usage:
    python scripts/compare_tokens_per_sec.py
    python scripts/compare_tokens_per_sec.py --max-tokens 500 --device mps --repeats 3
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from transformers import AutoConfig, AutoTokenizer

from tests.helpers import MODEL_ID, load_our_model, make_cache, sync


@torch.no_grad()
def timed_generate(model, config, input_ids, n_tokens, device, use_cache, sampling, seed):
    torch.manual_seed(seed)
    cache = make_cache(config, input_ids.size(0), input_ids.size(1) + n_tokens, device) if use_cache else None

    sync(device)
    start = time.perf_counter()
    output = model.generate(input_ids, max_length=n_tokens, cache=cache, **sampling)
    sync(device)
    elapsed = time.perf_counter() - start

    new_tokens = output.size(1) - input_ids.size(1)
    cache_bytes = 0 if cache is None else 2 * cache.k.numel() * cache.k.element_size()
    return {
        "elapsed": elapsed,
        "tokens": new_tokens,
        "tps": new_tokens / max(elapsed, 1e-9),
        "output": output,
        "cache_bytes": cache_bytes,
    }


@torch.no_grad()
def timed_prefill(model, config, input_ids, device):
    """Cost of the first (prompt) forward pass, measured on a fresh cache."""
    cache = make_cache(config, input_ids.size(0), input_ids.size(1), device)
    sync(device)
    start = time.perf_counter()
    model(input_ids, cache=cache)
    sync(device)
    return time.perf_counter() - start


def summarize(runs):
    tps = [r["tps"] for r in runs]
    return {
        "best_tps": max(tps),
        "mean_tps": sum(tps) / len(tps),
        "mean_elapsed": sum(r["elapsed"] for r in runs) / len(runs),
        "tokens": runs[0]["tokens"],
        "cache_bytes": runs[0]["cache_bytes"],
        "output": runs[0]["output"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompt", default="The capital of France is",
                        help="input text (prompt)")
    parser.add_argument("--max-tokens", type=int, default=500,
                        help="number of new tokens to generate (default: 500)")
    parser.add_argument("--device", default="cpu", help="cpu | cuda | mps")
    parser.add_argument("--repeats", type=int, default=1,
                        help="timed runs per configuration (default: 1)")
    parser.add_argument("--warmup", type=int, default=8,
                        help="untimed warmup tokens per configuration (default: 8)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="0 = greedy, makes both runs comparable (default: 0)")
    parser.add_argument("--top-k", type=int, default=0, help="top-k filtering, 0 = off (default: 0)")
    parser.add_argument("--top-p", type=float, default=1.0, help="nucleus filtering, 1.0 = off (default: 1.0)")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for sampling (default: 0)")
    args = parser.parse_args()

    device = torch.device(args.device)
    config = AutoConfig.from_pretrained(MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = load_our_model(config, device)

    input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids.to(device)
    prompt_len = input_ids.size(1)
    sampling = dict(temperature=args.temperature, top_k=args.top_k, top_p=args.top_p)

    print(f"model: {MODEL_ID}\ndevice: {device}, dtype: float32, threads: {torch.get_num_threads()}")
    print(f"prompt: {args.prompt!r} ({prompt_len} tokens)")
    print(f"generating {args.max_tokens} new tokens, sampling: {sampling}, repeats: {args.repeats}\n")

    results = {}
    for use_cache in (False, True):
        label = "with cache" if use_cache else "no cache"
        if args.warmup > 0:
            timed_generate(model, config, input_ids, args.warmup, device, use_cache, sampling, args.seed)
        runs = []
        for i in range(args.repeats):
            r = timed_generate(model, config, input_ids, args.max_tokens, device,
                               use_cache, sampling, args.seed)
            runs.append(r)
            print(f"[{label:<10} run {i + 1}/{args.repeats}] "
                  f"{r['tokens']} tokens in {r['elapsed']:.2f}s -> {r['tps']:.1f} tok/s")
        results[use_cache] = summarize(runs)

    prefill = timed_prefill(model, config, input_ids, device)

    plain, cached = results[False], results[True]
    identical = torch.equal(plain["output"], cached["output"])
    speedup = cached["best_tps"] / max(plain["best_tps"], 1e-9)

    print("\n=== tokens/s (generate, {} new tokens) ===".format(args.max_tokens))
    print(f"{'':<12} {'tok/s (best)':>13} {'tok/s (mean)':>13} {'time (mean)':>13} {'ms/token':>10}")
    for label, r in (("no cache", plain), ("with cache", cached)):
        print(f"{label:<12} {r['best_tps']:>13.1f} {r['mean_tps']:>13.1f} "
              f"{r['mean_elapsed']:>12.2f}s {1000 / r['best_tps']:>10.2f}")

    print(f"\nspeedup (best tok/s):    {speedup:.2f}x")
    print(f"prompt prefill:          {prefill * 1000:.1f} ms for {prompt_len} tokens")
    decode_time = max(cached["mean_elapsed"] - prefill, 1e-9)
    print(f"decode only (cached):    {(cached['tokens'] - 1) / decode_time:.1f} tok/s "
          f"(excluding prefill, approx)")
    print(f"KV cache memory:         {cached['cache_bytes'] / 2**20:.1f} MiB "
          f"(max_len={prompt_len + args.max_tokens}, {config.num_hidden_layers} layers, "
          f"{config.num_key_value_heads} kv heads)")
    print(f"identical output tokens: {identical}")
    if not identical:
        a, b = plain["output"], cached["output"]
        mismatches = (a != b).nonzero()
        print(f"  {mismatches.shape[0]} differing positions, first at {mismatches[0].tolist()}")

    print(f"\nno cache text:   {tokenizer.decode(plain['output'][0, prompt_len:])[:300]!r}")
    print(f"with cache text: {tokenizer.decode(cached['output'][0, prompt_len:])[:300]!r}")

    if args.temperature == 0.0 and not identical:
        print("\nFAIL: greedy decoding must give identical tokens with and without cache")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
