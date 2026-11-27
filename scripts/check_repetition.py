#!/usr/bin/env python3
"""Generate with repetition_penalty / no_repeat_ngram_size DISABLED and check
whether the output gets stuck in a loop.

This runs the model exactly as if the anti-repetition features did not exist
(repetition_penalty=1.0, no_repeat_ngram_size=0). Greedy decoding is the most
loop-prone, so it is the default; a sampled run can be added with --temperature.

Usage:
    python scripts/check_repetition.py
    python scripts/check_repetition.py --max-tokens 300 --temperature 0.8
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from transformers import AutoConfig, AutoTokenizer

from tests.helpers import MODEL_ID, load_our_model, make_cache

PROMPTS = [
    "The capital of France is",
    "Once upon a time, in a distant land, there lived a little dragon.",
    "def quicksort(arr):",
    "The meaning of life is",
]


def detect_loop(ids, max_period=160):
    """Find the strongest (approximate) periodic stretch: return (period, run_len)
    where run_len is the longest run of positions with ids[i] == ids[i-period].
    Catches phrase-level loops even when the tail is truncated mid-cycle."""
    n = len(ids)
    best = None
    for p in range(1, min(max_period, n // 2) + 1):
        run = best_run = 0
        for i in range(p, n):
            if ids[i] == ids[i - p]:
                run += 1
                if run > best_run:
                    best_run = run
            else:
                run = 0
        if best_run >= 2 * p and best_run >= 30 and (best is None or best_run > best[1]):
            best = (p, best_run)
    return best


def repeated_ngram_fraction(ids, n=4):
    """Fraction of n-gram occurrences that are duplicates (0 = no repetition)."""
    if len(ids) < n:
        return 0.0
    grams = [tuple(ids[i:i + n]) for i in range(len(ids) - n + 1)]
    seen, dup = set(), 0
    for g in grams:
        if g in seen:
            dup += 1
        seen.add(g)
    return dup / len(grams)


@torch.no_grad()
def run(model, config, tokenizer, device, prompt, max_tokens, temperature):
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    cache = make_cache(config, 1, ids.size(1) + max_tokens, device)
    out = model.naive_generate(
        ids, max_length=max_tokens, cache=cache, temperature=temperature,
        repetition_penalty=1.0, no_repeat_ngram_size=0,   # both DISABLED
        eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)[0]
    new = out[ids.size(1):].tolist()
    return new


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0, help="0 = greedy")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--prompt", action="append", help="override prompts (repeatable)")
    args = ap.parse_args()

    device = torch.device(args.device)
    config = AutoConfig.from_pretrained(MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = load_our_model(config, device)

    prompts = args.prompt or PROMPTS
    mode = "greedy" if args.temperature == 0 else f"sampled T={args.temperature}"
    print(f"model: {MODEL_ID} | {mode} | repetition_penalty=1.0 no_repeat_ngram_size=0 "
          f"(disabled) | {args.max_tokens} new tokens\n")

    any_loop = False
    for prompt in prompts:
        new = run(model, config, tokenizer, device, prompt, args.max_tokens, args.temperature)
        text = tokenizer.decode(new, skip_special_tokens=True)
        cyc = detect_loop(new)
        frac = repeated_ngram_fraction(new, 4)
        looped = cyc is not None or frac > 0.45
        any_loop |= looped
        print("=" * 78)
        print(f"PROMPT: {prompt!r}   ({len(new)} tokens)")
        print(f"loop: {'YES' if looped else 'no'}   "
              f"periodic stretch: {cyc and f'period={cyc[0]} tok over {cyc[1]} tok'}   "
              f"repeated 4-gram fraction: {frac:.2%}")
        print("-" * 78)
        print(text)
        print()

    print("=" * 78)
    print("VERDICT:", "LOOPING detected in at least one prompt" if any_loop
          else "no looping detected")
    return 0


if __name__ == "__main__":
    sys.exit(main())
