"""OpenAI-compatible API tests: non-streaming + SSE streaming, and concurrent
requests batched by the engine. Greedy API output must match naive_generate.

Run:
    pytest tests/test_api.py -v
"""

import json
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from core.api import create_app, build_chat_prompt_ids
from tests.helpers import make_cache

MAX_SEQS = 2
MAX_LEN = 256


@pytest.fixture(scope="module")
def client(env):
    config, tokenizer, model, device = env
    app = create_app(model, tokenizer, max_seqs=MAX_SEQS, max_len=MAX_LEN,
                     model_name="test-model")
    with TestClient(app) as c:
        yield c


def reference_text(model, config, tokenizer, device, prompt_ids, max_tokens):
    import torch
    ids = torch.tensor([prompt_ids], device=device)
    cache = make_cache(config, 1, ids.size(1) + max_tokens, device)
    out = model.naive_generate(ids, max_length=max_tokens, cache=cache, temperature=0.0,
                               eos_token_id=tokenizer.eos_token_id,
                               pad_token_id=tokenizer.pad_token_id)[0]
    return tokenizer.decode(out[ids.size(1):].tolist(), skip_special_tokens=True)


def sse_data(lines):
    for line in lines:
        if line.startswith("data:"):
            yield line[len("data:"):].strip()


def test_health_and_models(client):
    assert client.get("/health").json()["status"] == "ok"
    models = client.get("/v1/models").json()
    assert models["object"] == "list"
    assert models["data"][0]["id"] == "test-model"


def test_completions_matches_naive_generate(env, client):
    config, tokenizer, model, device = env
    prompt = "The capital of France is"
    max_tokens = 12
    r = client.post("/v1/completions", json={
        "prompt": prompt, "max_tokens": max_tokens, "temperature": 0.0})
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "text_completion"
    assert body["choices"][0]["finish_reason"] in ("stop", "length")
    assert body["usage"]["prompt_tokens"] == len(tokenizer(prompt).input_ids)

    ref = reference_text(model, config, tokenizer, device,
                         tokenizer(prompt).input_ids, max_tokens)
    assert body["choices"][0]["text"] == ref


def test_chat_completions_matches_naive_generate(env, client):
    config, tokenizer, model, device = env
    messages = [{"role": "user", "content": "Name a primary color."}]
    max_tokens = 10
    r = client.post("/v1/chat/completions", json={
        "messages": messages, "max_tokens": max_tokens, "temperature": 0.0})
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert body["choices"][0]["finish_reason"] in ("stop", "length")

    prompt_ids = build_chat_prompt_ids(tokenizer, messages)
    ref = reference_text(model, config, tokenizer, device, prompt_ids, max_tokens)
    assert body["choices"][0]["message"]["content"] == ref


def test_completions_streaming(env, client):
    config, tokenizer, model, device = env
    prompt = "Once upon a time"
    max_tokens = 15
    payload = {"prompt": prompt, "max_tokens": max_tokens, "temperature": 0.0}

    plain = client.post("/v1/completions", json=payload).json()["choices"][0]["text"]

    pieces, saw_done, finish = [], False, None
    with client.stream("POST", "/v1/completions", json={**payload, "stream": True}) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        for data in sse_data(resp.iter_lines()):
            if data == "[DONE]":
                saw_done = True
                break
            chunk = json.loads(data)
            assert chunk["object"] == "text_completion"
            choice = chunk["choices"][0]
            pieces.append(choice["text"])
            if choice["finish_reason"]:
                finish = choice["finish_reason"]

    assert saw_done
    assert finish in ("stop", "length")
    assert "".join(pieces) == plain
    ref = reference_text(model, config, tokenizer, device,
                         tokenizer(prompt).input_ids, max_tokens)
    assert "".join(pieces) == ref


def test_chat_streaming(client):
    messages = [{"role": "user", "content": "Say hello."}]
    payload = {"messages": messages, "max_tokens": 12, "temperature": 0.0}

    plain = client.post("/v1/chat/completions", json=payload).json()
    plain_text = plain["choices"][0]["message"]["content"]

    pieces, saw_done, finish, saw_role = [], False, None, False
    with client.stream("POST", "/v1/chat/completions", json={**payload, "stream": True}) as resp:
        assert resp.status_code == 200
        for data in sse_data(resp.iter_lines()):
            if data == "[DONE]":
                saw_done = True
                break
            chunk = json.loads(data)
            assert chunk["object"] == "chat.completion.chunk"
            delta = chunk["choices"][0]["delta"]
            if delta.get("role") == "assistant":
                saw_role = True
            pieces.append(delta.get("content", ""))
            if chunk["choices"][0]["finish_reason"]:
                finish = chunk["choices"][0]["finish_reason"]

    assert saw_done and saw_role
    assert finish in ("stop", "length")
    assert "".join(pieces) == plain_text


def test_concurrent_requests_are_batched(env, client):
    config, tokenizer, model, device = env
    prompts = ["Hello", "The capital of France is", "def add(a, b):", "import os"]
    max_tokens = [8, 10, 12, 6]
    refs = [reference_text(model, config, tokenizer, device,
                           tokenizer(p).input_ids, mt)
            for p, mt in zip(prompts, max_tokens)]

    def call(i):
        r = client.post("/v1/completions", json={
            "prompt": prompts[i], "max_tokens": max_tokens[i], "temperature": 0.0})
        assert r.status_code == 200
        return r.json()["choices"][0]["text"]

    with ThreadPoolExecutor(max_workers=len(prompts)) as pool:
        texts = list(pool.map(call, range(len(prompts))))

    assert texts == refs
