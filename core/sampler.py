"""Next-token sampling. Params may be scalar or per-row (B,) so a batch can mix
requests with different settings. naive_generate uses one Sampler with scalars."""

from __future__ import annotations

import torch
from torch.nn import functional as F


class Sampler:
    def __init__(
            self,
            temperature: float = 1.0,
            top_k: int = 0,
            top_p: float = 1.0,
            repetition_penalty: float = 1.0,
            no_repeat_ngram_size: int = 0,
    ):
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.no_repeat_ngram_size = no_repeat_ngram_size

    @staticmethod
    def _broadcast(value, batch_size: int, device, dtype) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            t = value.to(device=device, dtype=dtype).reshape(-1)
        elif isinstance(value, (list, tuple)):
            t = torch.tensor(value, device=device, dtype=dtype).reshape(-1)
        else:
            return torch.full((batch_size,), value, device=device, dtype=dtype)
        if t.numel() == 1:
            t = t.expand(batch_size)
        assert t.numel() == batch_size, "per-row sampling parameter has wrong length"
        return t

    def __call__(
            self,
            logits: torch.Tensor,
            history: torch.Tensor = None,
            pad_size: torch.Tensor = None,
    ) -> torch.Tensor:
        """logits (B,V) -> sampled ids (B,1). history (B,L) + pad_size (B,) feed
        repetition_penalty / no_repeat_ngram_size."""
        B, V = logits.shape
        device = logits.device
        temp = self._broadcast(self.temperature, B, device, torch.float32)
        top_k = self._broadcast(self.top_k, B, device, torch.long)
        top_p = self._broadcast(self.top_p, B, device, torch.float32)
        rep = self._broadcast(self.repetition_penalty, B, device, torch.float32)
        ngram = self._broadcast(self.no_repeat_ngram_size, B, device, torch.long)

        logits = logits.clone()

        if history is not None:
            if pad_size is None:
                pad_size = torch.zeros(B, dtype=torch.long, device=device)
            for b in range(B):
                prev = history[b, int(pad_size[b]):]
                if float(rep[b]) != 1.0:
                    toks = prev.unique()
                    scores = logits[b, toks]
                    logits[b, toks] = torch.where(
                        scores > 0, scores / rep[b], scores * rep[b])
                n = int(ngram[b])
                if n > 0:
                    seq = prev.tolist()
                    if len(seq) >= n:
                        ngrams = {}
                        for j in range(len(seq) - n + 1):
                            ng = tuple(seq[j:j + n])
                            ngrams.setdefault(ng[:-1], set()).add(ng[-1])
                        banned = ngrams.get(tuple(seq[-(n - 1):]), ())
                        if banned:
                            logits[b, list(banned)] = float('-inf')

        greedy = temp <= 0
        safe_temp = torch.where(greedy, torch.ones_like(temp), temp)
        logits = logits / safe_temp[:, None]

        # top-k then top-p on a descending sort
        sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
        k_eff = top_k.clamp(min=1, max=V)
        kth = sorted_logits.gather(1, (k_eff - 1)[:, None])          # (B,1)
        keep_k = (top_k[:, None] <= 0) | (sorted_logits >= kth)
        sorted_logits = sorted_logits.masked_fill(~keep_k, float('-inf'))

        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumulative = sorted_probs.cumsum(dim=-1)
        keep_p = (cumulative - sorted_probs) <= top_p[:, None]
        sorted_logits = sorted_logits.masked_fill(~keep_p, float('-inf'))

        logits = torch.zeros_like(sorted_logits).scatter(-1, sorted_idx, sorted_logits)

        argmaxed = logits.argmax(dim=-1, keepdim=True)               # (B,1)
        probs = F.softmax(logits, dim=-1)
        if greedy.any():
            probs = torch.where(
                greedy[:, None],
                F.one_hot(argmaxed[:, 0], num_classes=V).to(probs.dtype),
                probs)
        sampled = torch.multinomial(probs, num_samples=1)
        return torch.where(greedy[:, None], argmaxed, sampled)
