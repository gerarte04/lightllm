"""Continuous batching must reproduce naive_generate token-for-token.

~10 prompts of different lengths with different max_tokens, greedy. Reference:
each prompt generated alone (no batch) via naive_generate. Then all requests go
through the Engine with max_seqs < number of prompts, so they really queue,
get admitted and finish while slots are reused. Outputs must match exactly.

Run:
    pytest tests/test_continuous_batching.py -v
"""

import pytest
import torch

from core.engine import make_engine
from tests.helpers import make_cache

# (prompt, max_tokens) — varied lengths and budgets
REQUESTS = [
    ("Hello", 5),
    ("The capital of France is", 12),
    ("Once upon a time, in a distant land, there lived a little dragon "
     "who dreamed of flying beyond the mountains.", 8),
    ("def quicksort(arr):", 20),
    ("The meaning of life is", 6),
    ("In a hole in the ground there lived a hobbit. Not a nasty, dirty, wet hole, "
     "filled with the ends of worms and an oozy smell.", 15),
    ("import torch", 10),
    ("Yesterday, all my troubles seemed so far away. Now it looks as though "
     "they're here to stay.", 25),
    ("Q: What is 2 + 2? A:", 7),
    ("The quick brown fox jumps over the lazy dog.", 18),
]
MAX_SEQS = 4


def reference_tokens(model, config, tokenizer, device, prompt, max_tokens):
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    cache = make_cache(config, 1, ids.size(1) + max_tokens, device)
    out = model.naive_generate(ids, max_length=max_tokens, cache=cache, temperature=0.0,
                               eos_token_id=tokenizer.eos_token_id,
                               pad_token_id=tokenizer.pad_token_id)[0]
    return out[ids.size(1):].tolist()


@pytest.mark.parametrize("max_seqs", [1, MAX_SEQS], ids=["seqs1", "seqs4"])
def test_engine_matches_naive_generate(env, max_seqs):
    config, tokenizer, model, device = env
    assert max_seqs < len(REQUESTS), "requests must queue"

    refs = [reference_tokens(model, config, tokenizer, device, p, mt)
            for p, mt in REQUESTS]

    prompt_lens = [len(tokenizer(p).input_ids) for p, _ in REQUESTS]
    max_len = max(prompt_lens) + max(mt for _, mt in REQUESTS) + 1
    engine = make_engine(model, max_seqs=max_seqs, max_len=max_len,
                         eos_id=tokenizer.eos_token_id, pad_id=tokenizer.pad_token_id)

    seqs = [engine.add_request(tokenizer(p).input_ids, max_tokens=mt, temperature=0.0)
            for p, mt in REQUESTS]

    saw_queue = False
    peak_running = 0
    while engine.has_unfinished():
        if engine.scheduler.waiting and engine.scheduler.running:
            saw_queue = True
        peak_running = max(peak_running, len(engine.scheduler.running))
        engine.step()

    assert saw_queue, "no request ever waited while others ran"
    assert peak_running <= max_seqs

    for i, seq in enumerate(seqs):
        assert seq.output_ids == refs[i], (
            f"request {i} ({REQUESTS[i][0]!r}) differs from naive_generate:\n"
            f"  engine: {seq.output_ids}\n  ref:    {refs[i]}")
        assert seq.finish_reason in ("stop", "length")
        assert len(seq.output_ids) <= REQUESTS[i][1]
