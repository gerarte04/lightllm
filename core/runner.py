import torch

from core.cache import Cache
from core.model import Transformer, SlotContext
from core.sampler import Sampler
from core.sequence import Batch, BatchPhase, Sequence


class ModelRunner:
    """
    PREFILL: left-padded prompt batch, real K/V written compactly into slots.
    DECODE:  (B,1) last tokens, per-row slot/position and a mask over the slot.
    """

    def __init__(self, model: Transformer, max_seqs: int, max_len: int = None,
                 device=None, dtype=None, pad_id: int = None):
        self.model = model.eval()
        cfg = model.config
        self.device = device or next(model.parameters()).device
        self.dtype = dtype or next(model.parameters()).dtype
        head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        ctx_len = getattr(cfg, "max_position_embeddings", None)
        self.max_len = min(max_len or ctx_len, ctx_len) if ctx_len else (max_len or 8192)
        self.pad_id = pad_id if pad_id is not None else (
            getattr(cfg, "pad_token_id", None) or getattr(cfg, "eos_token_id", None) or 0)
        self.cache = Cache(cfg.num_hidden_layers, max_seqs, cfg.num_key_value_heads,
                           self.max_len, head_dim, self.device, self.dtype)

    @torch.no_grad()
    def run(self, batch: Batch) -> list[int]:
        if not batch.seqs:
            return []
        if batch.phase is BatchPhase.PREFILL:
            return self._prefill(batch.seqs)
        return self._decode(batch.seqs)

    def _prefill(self, seqs: list[Sequence]) -> list[int]:
        B = len(seqs)
        T = max(len(s.prompt_ids) for s in seqs)
        input_ids = torch.full((B, T), self.pad_id, dtype=torch.long, device=self.device)
        pad_size = torch.zeros(B, dtype=torch.long, device=self.device)
        slots = torch.tensor([s.slot for s in seqs], dtype=torch.long, device=self.device)
        for b, s in enumerate(seqs):
            p = len(s.prompt_ids)
            input_ids[b, T - p:] = torch.tensor(s.prompt_ids, dtype=torch.long, device=self.device)
            pad_size[b] = T - p

        ctx = SlotContext(slots=slots)
        logits = self.model(input_ids, cache=self.cache, pad_size=pad_size, ctx=ctx)  # (B,T,V)
        history = input_ids if self._penalties(seqs) else None
        nxt = self._sampler(seqs)(logits[:, -1, :], history=history, pad_size=pad_size)
        return nxt[:, 0].tolist()

    def _decode(self, seqs: list[Sequence]) -> list[int]:
        B = len(seqs)
        input_ids = torch.tensor([[s.last_token] for s in seqs], dtype=torch.long, device=self.device)  # (B,1)
        slots = torch.tensor([s.slot for s in seqs], dtype=torch.long, device=self.device)
        positions = torch.tensor([s.num_tokens - 1 for s in seqs], dtype=torch.long, device=self.device)
        keys = torch.arange(self.max_len, device=self.device)
        attn_mask = (keys[None, :] > positions[:, None])[:, None, None, :]  # (B,1,1,max_len)

        ctx = SlotContext(slots=slots, positions=positions, attn_mask=attn_mask)
        logits = self.model(input_ids, cache=self.cache, ctx=ctx)           # (B,1,V)
        history, pad_size = self._history(seqs) if self._penalties(seqs) else (None, None)
        nxt = self._sampler(seqs)(logits[:, 0, :], history=history, pad_size=pad_size)
        return nxt[:, 0].tolist()

    @staticmethod
    def _sampler(seqs: list[Sequence]) -> Sampler:
        return Sampler(
            temperature=[s.temperature for s in seqs],
            top_k=[s.top_k for s in seqs],
            top_p=[s.top_p for s in seqs],
            repetition_penalty=[s.repetition_penalty for s in seqs],
            no_repeat_ngram_size=[s.no_repeat_ngram_size for s in seqs])

    @staticmethod
    def _penalties(seqs: list[Sequence]) -> bool:
        return any(s.repetition_penalty != 1.0 or s.no_repeat_ngram_size > 0 for s in seqs)

    def _history(self, seqs: list[Sequence]):
        rows = [s.prompt_ids + s.output_ids for s in seqs]
        W = max(len(r) for r in rows)
        hist = torch.full((len(seqs), W), self.pad_id, dtype=torch.long, device=self.device)
        pad_size = torch.zeros(len(seqs), dtype=torch.long, device=self.device)
        for b, r in enumerate(rows):
            hist[b, W - len(r):] = torch.tensor(r, dtype=torch.long, device=self.device)
            pad_size[b] = W - len(r)
        return hist, pad_size
