"""Static batching tests for core.model.Transformer.

A left-padded batch of prompts with different lengths must produce the same
logits (and the same greedy tokens) as running each prompt individually.
Generation must stop per-row on EOS / config.max_position_embeddings
(excluding padding) and drop finished rows from the active batch.

Reference for model loading / cache allocation: tests/helpers.py.

Run:
    pytest tests/test_static_batching.py -v
"""

import pytest
import torch

from tests.helpers import make_cache

PROMPTS = [
    "Hello",
    "The capital of France is",
    "Once upon a time, in a distant land, there lived a little dragon "
    "who dreamed of flying beyond the mountains.",
]
N_NEW_TOKENS = 100  # generate at least 100 tokens per prompt


def left_pad(prompts, tokenizer, device):
    """Tokenize prompts and left-pad them to a common width with pad_token_id."""
    seqs = [tokenizer(p, return_tensors="pt").input_ids[0] for p in prompts]
    width = max(s.numel() for s in seqs)
    input_ids = torch.full((len(seqs), width), tokenizer.pad_token_id, dtype=torch.long)
    pad_size = torch.zeros(len(seqs), dtype=torch.long)
    for i, s in enumerate(seqs):
        input_ids[i, width - s.numel():] = s
        pad_size[i] = width - s.numel()
    assert (pad_size > 0).any(), "test needs prompts of different lengths"
    return input_ids.to(device), pad_size.to(device), seqs


def check_row(batched_row, pad_size_b, ref, pad_token_id):
    """Real tokens of a batched row must equal the individual run exactly,
    and everything after them must be padding."""
    p = int(pad_size_b)
    real = batched_row[p:p + ref.numel()]
    assert torch.equal(real, ref), "generated tokens differ from the individual run"
    tail = batched_row[p + ref.numel():]
    assert (tail == pad_token_id).all(), "finished row must be padded, not continued"


@pytest.mark.parametrize("use_cache", [True, False], ids=["with_cache", "no_cache"])
def test_batched_prefill_logits_match_individual(env, use_cache):
    config, tokenizer, model, device = env
    input_ids, pad_size, seqs = left_pad(PROMPTS, tokenizer, device)

    cache = make_cache(config, len(seqs), input_ids.size(1), device) if use_cache else None
    with torch.no_grad():
        batched = model(input_ids, cache=cache, pad_size=pad_size)

    for b, seq in enumerate(seqs):
        with torch.no_grad():
            single = model(seq[None].to(device))[0]
        ours = batched[b, int(pad_size[b]):]  # real (non-pad) positions only
        assert ours.shape == single.shape
        assert torch.allclose(ours, single, atol=1e-4, rtol=1e-4), \
            f"prompt {b}: max diff {(ours - single).abs().max().item():.3e}"
        assert torch.equal(ours.argmax(-1), single.argmax(-1)), f"prompt {b}: argmax mismatch"


def test_batched_generate_matches_individual(env):
    """Greedy generation of >=100 tokens: padded batch == per-prompt runs."""
    config, tokenizer, model, device = env
    input_ids, pad_size, seqs = left_pad(PROMPTS, tokenizer, device)
    eos, pad = tokenizer.eos_token_id, tokenizer.pad_token_id

    cache = make_cache(config, len(seqs), input_ids.size(1) + N_NEW_TOKENS, device)
    batched = model.generate(input_ids, max_length=N_NEW_TOKENS, cache=cache,
                             temperature=0.0, eos_token_id=eos, pad_token_id=pad)

    n_new_tokens, hit_eos = [], []
    for b, seq in enumerate(seqs):
        single_cache = make_cache(config, 1, seq.numel() + N_NEW_TOKENS, device)
        ref = model.generate(seq[None].to(device), max_length=N_NEW_TOKENS, cache=single_cache,
                             temperature=0.0, eos_token_id=eos, pad_token_id=pad)[0]
        # both runs use the same stopping rules, so lengths must agree exactly
        check_row(batched[b], pad_size[b], ref, pad)
        n_new = ref.numel() - seq.numel()
        assert n_new >= N_NEW_TOKENS or ref[-1].item() == eos, \
            f"prompt {b}: only {n_new} new tokens without hitting EOS"
        n_new_tokens.append(n_new)
        hit_eos.append(ref[-1].item() == eos)

    # the test must actually generate at least 100 new tokens
    assert max(n_new_tokens) >= N_NEW_TOKENS or all(hit_eos)


def test_generate_stops_at_eos_and_shrinks_batch(env, monkeypatch):
    """A row that emits EOS is padded afterwards and dropped from the batch."""
    config, tokenizer, model, device = env
    input_ids, pad_size, seqs = left_pad(PROMPTS, tokenizer, device)
    pad = tokenizer.pad_token_id

    # Find a token greedy decoding will actually emit for prompt 0, and use
    # it as a fake EOS so that row is guaranteed to finish early.
    probe_cache = make_cache(config, 1, seqs[0].numel() + N_NEW_TOKENS, device)
    probe = model.generate(seqs[0][None].to(device), max_length=N_NEW_TOKENS, cache=probe_cache,
                           temperature=0.0, eos_token_id=-1, pad_token_id=pad)[0]
    new_tokens = probe[seqs[0].numel():]
    eos = int(new_tokens[15].item())
    first_hit = int((new_tokens == eos).float().argmax().item())
    stop_len = seqs[0].numel() + first_hit + 1  # row length incl. the EOS token
    assert stop_len < probe.numel(), "fake EOS must occur before the 100-token budget"

    # Spy on forward passes to verify finished rows stop being computed.
    batch_sizes = []
    orig_forward = model.forward

    def spy(x, *args, **kwargs):
        batch_sizes.append(x.size(0))
        return orig_forward(x, *args, **kwargs)

    monkeypatch.setattr(model, "forward", spy)

    cache = make_cache(config, len(seqs), input_ids.size(1) + N_NEW_TOKENS, device)
    batched = model.generate(input_ids, max_length=N_NEW_TOKENS, cache=cache,
                             temperature=0.0, eos_token_id=eos, pad_token_id=pad)
    batched_call_sizes = list(batch_sizes)

    # row 0: exactly the tokens up to (and including) the fake EOS, then pads
    p0 = int(pad_size[0])
    row = batched[0, p0:]
    assert torch.equal(row[:stop_len], probe[:stop_len])
    assert (row[stop_len:] == pad).all(), "finished row must be padded, not continued"

    # every other row must match an individual run with the same fake EOS
    for b in range(1, len(seqs)):
        single_cache = make_cache(config, 1, seqs[b].numel() + N_NEW_TOKENS, device)
        ref = model.generate(seqs[b][None].to(device), max_length=N_NEW_TOKENS, cache=single_cache,
                             temperature=0.0, eos_token_id=eos, pad_token_id=pad)[0]
        check_row(batched[b], pad_size[b], ref, pad)

    # after row 0 finished, decode steps must run with a smaller batch
    assert min(batched_call_sizes) < len(seqs), "finished rows were not removed from the batch"


def test_generate_stops_at_config_max_length(env, monkeypatch):
    """Rows stop at max_position_embeddings non-pad tokens, at different steps."""
    config, tokenizer, model, device = env
    input_ids, pad_size, seqs = left_pad(PROMPTS, tokenizer, device)
    pad = tokenizer.pad_token_id

    real_lens = [s.numel() for s in seqs]
    limit = max(real_lens) + 7  # each row may generate a different number of tokens
    monkeypatch.setattr(model.config, "max_position_embeddings", limit)

    cache = make_cache(config, len(seqs), input_ids.size(1) + N_NEW_TOKENS, device)
    batched = model.generate(input_ids, max_length=N_NEW_TOKENS, cache=cache,
                             temperature=0.0, eos_token_id=-1, pad_token_id=pad)

    for b, seq in enumerate(seqs):
        single_cache = make_cache(config, 1, seq.numel() + N_NEW_TOKENS, device)
        ref = model.generate(seq[None].to(device), max_length=N_NEW_TOKENS, cache=single_cache,
                             temperature=0.0, eos_token_id=-1, pad_token_id=pad)[0]
        assert ref.numel() == limit, f"prompt {b}: individual run did not stop at the limit"
        assert limit - real_lens[b] < N_NEW_TOKENS
        check_row(batched[b], pad_size[b], ref, pad)

    # the whole loop must stop once the longest row reaches the limit
    assert batched.size(1) == input_ids.size(1) + (limit - min(real_lens))
