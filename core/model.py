import re

import torch
from torch import nn
from torch.nn import functional as F

from core.cache import Cache


def rope_cos_sin(T, d, base):
    inv_freq = 1.0 / (base ** (torch.arange(0, d, 2).float() / d))  # θ_i, (d/2,)
    pos = torch.arange(T).float()                                    # m
    freqs = torch.outer(pos, inv_freq)                               # m·θ_i, (T, d/2)
    emb = torch.cat([freqs, freqs], dim=-1)                          # (T, d)
    return emb.cos(), emb.sin()

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)        # first and second halfs
    return torch.cat([-x2, x1], dim=-1)

def apply_rope(x, cos, sin):           # x: (B, n_heads, T, d)
    return x * cos + rotate_half(x) * sin


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.to(torch.float32)
        x = x * torch.rsqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.weight * x.to(dtype)


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class GroupQueryAttention(nn.Module):
    def __init__(self, config, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.num_att_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.layer_idx = layer_idx

        self.q_proj = nn.Linear(config.hidden_size, self.num_att_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_att_heads * self.head_dim, config.hidden_size, bias=False)

    def forward(
            self,
            x: torch.Tensor,
            cos: torch.Tensor,
            sin: torch.Tensor,
            cache: Cache = None,
            start_pos: int = None,
            pad_size: torch.Tensor = None
            ) -> torch.Tensor:
        batch_size, num_tokens, _ = x.size()
        if start_pos is None:
            start_pos = 0
        if pad_size is None:
            pad_size = torch.zeros((batch_size,), dtype=torch.long, device=x.device)

        q = self.q_proj(x).view(batch_size, num_tokens, self.num_att_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, num_tokens, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, num_tokens, self.num_kv_heads, self.head_dim).transpose(1, 2)

        cos = cos.unsqueeze(1)  # (B, 1, T, head_dim)
        sin = sin.unsqueeze(1)
        k = apply_rope(k, cos, sin)
        q = apply_rope(q, cos, sin)

        if cache is not None:
            cache.update(self.layer_idx, k, v, start_pos)
            k, v = cache.get(self.layer_idx, 0, start_pos + num_tokens)

        k = k.repeat_interleave(self.num_att_heads // self.num_kv_heads, dim=1)
        v = v.repeat_interleave(self.num_att_heads // self.num_kv_heads, dim=1)

        attn_weights = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        total_len = k.size(2)
        q_pos = start_pos + torch.arange(num_tokens, device=x.device)             # (T,)
        k_pos = torch.arange(total_len, device=x.device)                          # (total_len,)
        pad_border = pad_size[:, None, None]                                      # (B, 1, 1)
        causal_mask = k_pos[None, None, :] > q_pos[None, :, None]
        pad_key = k_pos[None, None, :] < pad_border
        real_query = q_pos[None, :, None] >= pad_border
        causal_mask = causal_mask | (pad_key & real_query)                        # (B, T, total_len)
        attn_weights = attn_weights.masked_fill(causal_mask[:, None, :, :], float('-inf'))
        attn_weights = F.softmax(attn_weights, dim=-1)

        attn_output = attn_weights @ v
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, num_tokens, -1)
        return self.o_proj(attn_output)


class Block(nn.Module):
    def __init__(self, config, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = GroupQueryAttention(config, layer_idx=layer_idx)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = SwiGLU(config.hidden_size, config.intermediate_size)

    def forward(
            self,
            x: torch.Tensor,
            cos: torch.Tensor,
            sin: torch.Tensor,
            cache: Cache = None,
            start_pos: int = None,
            pad_size: torch.Tensor = None
            ) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, cache, start_pos, pad_size)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Block(config, layer_idx=i) for i in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        rope_theta = getattr(config, "rope_theta", None)
        if rope_theta is None:
            rope_theta = config.rope_parameters["rope_theta"]

        cos, sin = rope_cos_sin(config.max_position_embeddings, head_dim, base=rope_theta)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(
            self,
            x: torch.Tensor,
            cache: Cache = None,
            start_pos: int = None,
            pad_size: torch.Tensor = None
            ) -> torch.Tensor:
        x = self.embed_tokens(x)
        B, T, _ = x.shape
        if start_pos is None:
            start_pos = 0
        if pad_size is None:
            pad_size = torch.zeros((B,), dtype=torch.long, device=x.device)
        positions = start_pos - pad_size[:, None] + torch.arange(T, device=x.device)
        positions = positions.clamp(min=0)
        cos, sin = self.cos[positions], self.sin[positions]
        for layer in self.layers:
            x = layer(x, cos, sin, cache, start_pos, pad_size)
        return self.norm(x)


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if getattr(config, "tie_word_embeddings", False):
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
            self,
            x: torch.Tensor,
            cache: Cache = None,
            start_pos: int = None,
            pad_size: torch.Tensor = None
            ) -> torch.Tensor:
        return self.lm_head(self.model(x, cache, start_pos, pad_size))

    @staticmethod
    def _sample_next_token(
            logits: torch.Tensor,
            history: torch.Tensor,
            pad_size: torch.Tensor,
            temperature: float,
            top_k: int,
            top_p: float,
            repetition_penalty: float,
            no_repeat_ngram_size: int
            ) -> torch.Tensor:
        if repetition_penalty != 1.0:
            # penalize tokens that already occurred (ignoring left padding)
            for b in range(history.size(0)):
                prev_tokens = history[b, int(pad_size[b]):].unique()
                scores = logits[b, prev_tokens]
                logits[b, prev_tokens] = torch.where(
                    scores > 0, scores / repetition_penalty, scores * repetition_penalty)

        if no_repeat_ngram_size > 0:
            # ban any token that would complete an already seen n-gram
            n = no_repeat_ngram_size
            for b in range(history.size(0)):
                seq = history[b, int(pad_size[b]):].tolist()
                if len(seq) < n:
                    continue
                ngrams = {}
                for j in range(len(seq) - n + 1):
                    ngram = tuple(seq[j:j + n])
                    ngrams.setdefault(ngram[:-1], set()).add(ngram[-1])
                banned = ngrams.get(tuple(seq[-(n - 1):]), ())
                if banned:
                    logits[b, list(banned)] = float('-inf')

        if temperature <= 0:  # greedy decoding
            return logits.argmax(dim=-1, keepdim=True)
        logits = logits / temperature
        if top_k > 0:
            k = min(top_k, logits.size(-1))
            kth_value = logits.topk(k, dim=-1).values[..., -1, None]
            logits = logits.masked_fill(logits < kth_value, float('-inf'))
        if top_p < 1.0:
            sorted_logits, sorted_indices = logits.sort(dim=-1, descending=True)
            sorted_probs = F.softmax(sorted_logits, dim=-1)
            cumulative_probs = sorted_probs.cumsum(dim=-1)
            remove = (cumulative_probs - sorted_probs) > top_p
            sorted_logits = sorted_logits.masked_fill(remove, float('-inf'))
            logits = torch.zeros_like(sorted_logits).scatter(-1, sorted_indices, sorted_logits)
        return torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)

    @staticmethod
    def _assemble_rows(rows: list, fill_id: int, device, dtype) -> torch.Tensor:
        width = max(t.numel() for t in rows)
        out = torch.full((len(rows), width), fill_id, dtype=dtype, device=device)
        for i, t in enumerate(rows):
            out[i, :t.numel()] = t
        return out

    @torch.no_grad()
    def generate(
            self,
            input_ids: torch.Tensor,
            max_length: int,
            cache: Cache = None,
            temperature: float = 1.0,
            top_k: int = 0,
            top_p: float = 1.0,
            repetition_penalty: float = 1.0,
            no_repeat_ngram_size: int = 0,
            eos_token_id: int = None,
            pad_token_id: int = None
            ) -> torch.Tensor:
        self.eval()
        device = input_ids.device
        B = input_ids.size(0)
        if eos_token_id is None:
            eos_token_id = getattr(self.config, "eos_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(self.config, "pad_token_id", None)
        max_seq_len = getattr(self.config, "max_position_embeddings", None)
        fill_id = pad_token_id if pad_token_id is not None else (eos_token_id if eos_token_id is not None else 0)

        # number of leading pad tokens per row (left padding)
        if pad_token_id is not None:
            pad_size = (input_ids == pad_token_id).long().cumprod(dim=1).sum(dim=1)
        else:
            pad_size = torch.zeros(B, dtype=torch.long, device=device)

        cur, cur_pad = input_ids, pad_size      # active rows only, shrinks over time
        row_ids = torch.arange(B, device=device)  # their original batch indices
        outputs = [None] * B                    # finished rows, by original index

        def finish(done: torch.Tensor) -> bool:
            """Save finished rows and drop them from the batch/cache.
            Returns True when no active rows are left."""
            nonlocal cur, cur_pad, row_ids, cache
            if not done.any():
                return False
            for b in done.nonzero(as_tuple=True)[0]:
                outputs[row_ids[b]] = cur[b]
            keep = ~done
            if not keep.any():
                return True
            if cache is not None:
                cache = cache.subset(keep.nonzero(as_tuple=True)[0])
            cur, cur_pad, row_ids = cur[keep], cur_pad[keep], row_ids[keep]
            return False

        # rows already at the config length limit before generating anything
        if max_seq_len is not None and finish(cur.size(1) - cur_pad >= max_seq_len):
            return self._assemble_rows(outputs, fill_id, device, input_ids.dtype)

        for step in range(max_length):
            tokens = cur if step == 0 or cache is None else cur[:, -1:]
            logits = self(tokens, cache=cache, start_pos=cur.size(1) - tokens.size(1), pad_size=cur_pad)
            next_token = self._sample_next_token(
                logits[:, -1, :], cur, cur_pad, temperature, top_k, top_p,
                repetition_penalty, no_repeat_ngram_size)
            cur = torch.cat((cur, next_token), dim=1)

            done = torch.zeros(cur.size(0), dtype=torch.bool, device=device)
            if eos_token_id is not None:
                done |= next_token[:, 0] == eos_token_id
            if max_seq_len is not None:
                done |= cur.size(1) - cur_pad >= max_seq_len
            if finish(done):
                break

        for b, i in enumerate(row_ids):  # rows still active when the loop ended
            outputs[i] = cur[b]
        return self._assemble_rows(outputs, fill_id, device, input_ids.dtype)
