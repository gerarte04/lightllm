from __future__ import annotations

import asyncio
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from core.engine import Engine, make_engine
from core.sequence import SeqStatus

DEFAULT_MAX_TOKENS = 256


@dataclass
class RequestState:
    seq: object
    queue: asyncio.Queue = field(default_factory=asyncio.Queue)


class InferenceService:
    """Bridges the synchronous Engine and async HTTP handlers via per-request queues."""

    def __init__(self, engine: Engine, tokenizer):
        self.engine = engine
        self.tokenizer = tokenizer
        self.active: dict[int, RequestState] = {}
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._wake = asyncio.Event()
        self._task: asyncio.Task | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    async def start(self):
        self._loop = asyncio.get_running_loop()
        self._task = asyncio.create_task(self._run())

    async def stop(self):
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._executor.shutdown(wait=False)

    async def _run(self):
        while True:
            await self._wake.wait()
            self._wake.clear()
            while self.engine.has_unfinished():
                batch = await self._loop.run_in_executor(self._executor, self.engine.step)
                if batch is None:
                    continue
                for seq in batch.seqs:
                    st = self.active.get(seq.seq_id)
                    if st is None:
                        continue
                    st.queue.put_nowait(seq.output_ids[-1])
                    if seq.status is SeqStatus.FINISHED:
                        st.queue.put_nowait(None)

    def submit(self, prompt_ids, **params) -> RequestState:
        seq = self.engine.add_request(prompt_ids, **params)
        st = RequestState(seq=seq)
        self.active[seq.seq_id] = st
        if seq.status is SeqStatus.FINISHED:
            st.queue.put_nowait(None)
        else:
            self._wake.set()
        return st

    def release(self, seq_id: int):
        self.active.pop(seq_id, None)


class TextStream:
    """Incremental detokenization: decode cumulatively, emit the new suffix."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.ids: list[int] = []
        self.text = ""

    def push(self, token: int) -> str:
        self.ids.append(token)
        new = self.tokenizer.decode(self.ids, skip_special_tokens=True)
        delta = new[len(self.text):]
        self.text = new
        return delta


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    max_tokens: int | None = None
    temperature: float | None = 1.0
    top_p: float | None = 1.0
    top_k: int | None = 0
    repetition_penalty: float | None = 1.0
    no_repeat_ngram_size: int | None = 0
    stream: bool = False


class CompletionRequest(BaseModel):
    model: str | None = None
    prompt: str | list[str] = ""
    max_tokens: int | None = None
    temperature: float | None = 1.0
    top_p: float | None = 1.0
    top_k: int | None = 0
    repetition_penalty: float | None = 1.0
    no_repeat_ngram_size: int | None = 0
    stream: bool = False


def _sampling_params(req) -> dict:
    return dict(
        max_tokens=req.max_tokens if req.max_tokens is not None else DEFAULT_MAX_TOKENS,
        temperature=req.temperature if req.temperature is not None else 1.0,
        top_p=req.top_p if req.top_p is not None else 1.0,
        top_k=req.top_k or 0,
        repetition_penalty=req.repetition_penalty if req.repetition_penalty is not None else 1.0,
        no_repeat_ngram_size=req.no_repeat_ngram_size or 0,
    )


def _sse(payload: dict) -> str:
    import json
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def build_chat_prompt_ids(tokenizer, messages: list[dict]) -> list[int]:
    """Chat-template prompt ids, falling back to plain concatenation if the
    tokenizer has no chat template (e.g. base models)."""
    try:
        out = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        # transformers >= 5 returns a BatchEncoding; older versions a list of ids.
        if hasattr(out, "input_ids"):
            out = out["input_ids"]
        if hasattr(out, "tolist"):
            out = out.tolist()
        if out and isinstance(out[0], (list, tuple)):   # unwrap a single-conversation batch
            out = out[0]
        return [int(t) for t in out]
    except Exception:
        text = "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"
        return tokenizer(text).input_ids


def create_app(model, tokenizer, max_seqs: int = 8, max_len: int = None,
               model_name: str = "local-model") -> FastAPI:
    engine = make_engine(model, max_seqs=max_seqs, max_len=max_len,
                         eos_id=tokenizer.eos_token_id, pad_id=tokenizer.pad_token_id)
    service = InferenceService(engine, tokenizer)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await service.start()
        yield
        await service.stop()

    app = FastAPI(title="lightllm OpenAI-compatible API", lifespan=lifespan)
    app.state.service = service

    def usage(prompt_ids, seq):
        n_out = len(seq.output_ids)
        return {"prompt_tokens": len(prompt_ids), "completion_tokens": n_out,
                "total_tokens": len(prompt_ids) + n_out}

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/v1/models")
    async def list_models():
        return {"object": "list",
                "data": [{"id": model_name, "object": "model", "owned_by": "local"}]}

    async def collect(st: RequestState) -> list[int]:
        ids = []
        try:
            while True:
                item = await st.queue.get()
                if item is None:
                    break
                ids.append(item)
        finally:
            service.release(st.seq.seq_id)
        return ids

    @app.post("/v1/completions")
    async def completions(req: CompletionRequest):
        prompt = req.prompt if isinstance(req.prompt, str) else (req.prompt[0] if req.prompt else "")
        prompt_ids = tokenizer(prompt).input_ids
        if not prompt_ids:
            raise HTTPException(status_code=400, detail="prompt is empty")
        st = service.submit(prompt_ids, **_sampling_params(req))
        rid = f"cmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())

        if not req.stream:
            ids = await collect(st)
            text = tokenizer.decode(ids, skip_special_tokens=True)
            return JSONResponse({
                "id": rid, "object": "text_completion", "created": created, "model": model_name,
                "choices": [{"index": 0, "text": text, "logprobs": None,
                             "finish_reason": st.seq.finish_reason}],
                "usage": usage(prompt_ids, st.seq)})

        async def gen():
            ts = TextStream(tokenizer)
            base = {"id": rid, "object": "text_completion", "created": created, "model": model_name}
            try:
                while True:
                    item = await st.queue.get()
                    if item is None:
                        break
                    delta = ts.push(item)
                    if delta:
                        yield _sse({**base, "choices": [{"index": 0, "text": delta,
                                                          "logprobs": None, "finish_reason": None}]})
                yield _sse({**base, "choices": [{"index": 0, "text": "", "logprobs": None,
                                                "finish_reason": st.seq.finish_reason}]})
                yield "data: [DONE]\n\n"
            finally:
                service.release(st.seq.seq_id)

        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest):
        if not req.messages:
            raise HTTPException(status_code=400, detail="messages is empty")
        prompt_ids = build_chat_prompt_ids(
            tokenizer, [{"role": m.role, "content": m.content} for m in req.messages])
        st = service.submit(prompt_ids, **_sampling_params(req))
        rid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())

        if not req.stream:
            ids = await collect(st)
            text = tokenizer.decode(ids, skip_special_tokens=True)
            return JSONResponse({
                "id": rid, "object": "chat.completion", "created": created, "model": model_name,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                             "finish_reason": st.seq.finish_reason}],
                "usage": usage(prompt_ids, st.seq)})

        async def gen():
            ts = TextStream(tokenizer)
            base = {"id": rid, "object": "chat.completion.chunk", "created": created, "model": model_name}
            try:
                yield _sse({**base, "choices": [{"index": 0, "delta": {"role": "assistant"},
                                                "finish_reason": None}]})
                while True:
                    item = await st.queue.get()
                    if item is None:
                        break
                    delta = ts.push(item)
                    if delta:
                        yield _sse({**base, "choices": [{"index": 0, "delta": {"content": delta},
                                                          "finish_reason": None}]})
                yield _sse({**base, "choices": [{"index": 0, "delta": {},
                                                "finish_reason": st.seq.finish_reason}]})
                yield "data: [DONE]\n\n"
            finally:
                service.release(st.seq.seq_id)

        return StreamingResponse(gen(), media_type="text/event-stream")

    return app


def load_model(model_id: str, device: str = "cpu"):
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoTokenizer
    from core.model import Transformer

    config = AutoConfig.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = Transformer(config)
    state_dict = load_file(hf_hub_download(model_id, "model.safetensors"))
    if config.tie_word_embeddings and "lm_head.weight" not in state_dict:
        state_dict["lm_head.weight"] = state_dict["model.embed_tokens.weight"]
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device=device, dtype=torch.float32).eval()
    return model, tokenizer, config


def main():
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-seqs", type=int, default=8)
    parser.add_argument("--max-len", type=int, default=2048)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    model, tokenizer, _ = load_model(args.model_id, args.device)
    app = create_app(model, tokenizer, max_seqs=args.max_seqs, max_len=args.max_len,
                     model_name=args.model_id)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
