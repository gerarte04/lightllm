#!/usr/bin/env python3
"""Stream text generation with core.model.Transformer (SmolLM2-135M).

Calls model.generate() one token at a time and prints each new token to
stdout as soon as it is produced.

Usage:
    python scripts/generate.py MAX_TOKENS "input text"
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from transformers import AutoConfig, AutoTokenizer

from scripts.compare_logits_with_hf import MODEL_ID, load_our_model

DIM = "\x1b[2m"
RESET = "\x1b[0m"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("max_tokens", type=int, help="maximum number of tokens to generate")
    parser.add_argument("text", help="input text (prompt)")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="sampling temperature, 0 = greedy (default: 0.7)")
    parser.add_argument("--top-k", type=int, default=50, help="top-k filtering, 0 = off (default: 50)")
    parser.add_argument("--top-p", type=float, default=0.9, help="nucleus filtering, 1.0 = off (default: 0.9)")
    parser.add_argument("--repetition-penalty", type=float, default=1.2,
                        help="penalty for already used tokens, 1.0 = off (default: 1.2)")
    parser.add_argument("--no-repeat-ngram", type=int, default=0,
                        help="ban repeating n-grams of this size, 0 = off (default: 0)")
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    config = AutoConfig.from_pretrained(MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = load_our_model(config, device)

    input_ids = tokenizer(args.text, return_tensors="pt").input_ids.to(device)

    # Decode the whole sequence each step and print only the new tail:
    # this stays correct even when a UTF-8 character spans several tokens.
    printed = tokenizer.decode(input_ids[0])
    print(f"\n{DIM}{printed}{RESET}", end="", flush=True)

    generated = input_ids
    n_tokens = 0
    start = time.perf_counter()
    for _ in range(args.max_tokens):
        generated = model.generate(generated, max_length=1, temperature=args.temperature,
                                   top_k=args.top_k, top_p=args.top_p,
                                   repetition_penalty=args.repetition_penalty,
                                   no_repeat_ngram_size=args.no_repeat_ngram)
        if generated[0, -1].item() == tokenizer.eos_token_id:
            break
        text = tokenizer.decode(generated[0])
        print(text[len(printed):], end="", flush=True)
        printed = text
        n_tokens += 1
    elapsed = time.perf_counter() - start

    print(f"\n\n{DIM}[{n_tokens} tokens in {elapsed:.2f}s "
          f"({n_tokens / max(elapsed, 1e-9):.1f} tok/s), device={device}]{RESET}")


if __name__ == "__main__":
    main()
