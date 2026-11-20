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

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, cache: Cache = None, start_pos: int = 0) -> torch.Tensor:
        batch_size, num_tokens, _ = x.size()
        if cache is None:
            start_pos = 0

        q = self.q_proj(x).view(batch_size, num_tokens, self.num_att_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, num_tokens, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, num_tokens, self.num_kv_heads, self.head_dim).transpose(1, 2)

        k = apply_rope(k, cos[start_pos:start_pos + num_tokens], sin[start_pos:start_pos + num_tokens])
        q = apply_rope(q, cos[start_pos:start_pos + num_tokens], sin[start_pos:start_pos + num_tokens])

        if cache is not None:
            cache.update(self.layer_idx, k, v, start_pos)
            k, v = cache.get(self.layer_idx, 0, start_pos + num_tokens)
    
        k = k.repeat_interleave(self.num_att_heads // self.num_kv_heads, dim=1)
        v = v.repeat_interleave(self.num_att_heads // self.num_kv_heads, dim=1)

        attn_weights = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if num_tokens > 1:
            total_len = k.size(2)
            q_pos = torch.arange(num_tokens, device=x.device) + start_pos
            k_pos = torch.arange(total_len, device=x.device)
            causal_mask = torch.zeros(num_tokens, total_len, device=x.device, dtype=attn_weights.dtype)
            causal_mask = causal_mask.masked_fill(k_pos[None, :] > q_pos[:, None], float('-inf'))
            attn_weights += causal_mask
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

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, cache: Cache = None, start_pos: int = 0) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, cache=cache, start_pos=start_pos)
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

    def forward(self, x: torch.Tensor, cache: Cache = None, start_pos: int = 0) -> torch.Tensor:
        x = self.embed_tokens(x)
        T = x.size(1)
        if cache is None:
            start_pos = 0
        # attention slices [start_pos : start_pos + T] itself, so keep the prefix
        cos, sin = self.cos[:start_pos + T, :], self.sin[:start_pos + T, :]
        for layer in self.layers:
            x = layer(x, cos, sin, cache=cache, start_pos=start_pos)
        return self.norm(x)


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if getattr(config, "tie_word_embeddings", False):
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, x: torch.Tensor, cache: Cache = None, start_pos: int = 0) -> torch.Tensor:
        return self.lm_head(self.model(x, cache=cache, start_pos=start_pos))

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
            no_repeat_ngram_size: int = 0
            ) -> torch.Tensor:
        self.eval()
        generated = input_ids
        start_pos = 0
        for step in range(max_length):
            if cache is None:
                # no cache: recompute the whole sequence on every step
                logits = self(generated)
            elif step == 0:
                logits = self(generated, cache=cache)  # prefill
                start_pos = generated.size(1)
            else:
                logits = self(generated[:, -1:], cache=cache, start_pos=start_pos)
                start_pos += 1
            next_token_logits = logits[:, -1, :]

            if repetition_penalty != 1.0:
                # penalize tokens that already occurred
                for b in range(generated.size(0)):
                    prev_tokens = generated[b].unique()
                    scores = next_token_logits[b, prev_tokens]
                    next_token_logits[b, prev_tokens] = torch.where(
                        scores > 0, scores / repetition_penalty, scores * repetition_penalty)

            if no_repeat_ngram_size > 0:
                # ban any token that would complete an already seen n-gram
                n = no_repeat_ngram_size
                for b in range(generated.size(0)):
                    seq = generated[b].tolist()
                    if len(seq) < n:
                        continue
                    ngrams = {}
                    for j in range(len(seq) - n + 1):
                        ngram = tuple(seq[j:j + n])
                        ngrams.setdefault(ngram[:-1], set()).add(ngram[-1])
                    banned = ngrams.get(tuple(seq[-(n - 1):]), ())
                    if banned:
                        next_token_logits[b, list(banned)] = float('-inf')

            if temperature <= 0:  # greedy decoding
                next_token = next_token_logits.argmax(dim=-1, keepdim=True)
            else:
                next_token_logits = next_token_logits / temperature
                if top_k > 0:
                    k = min(top_k, next_token_logits.size(-1))
                    kth_value = next_token_logits.topk(k, dim=-1).values[..., -1, None]
                    next_token_logits = next_token_logits.masked_fill(next_token_logits < kth_value, float('-inf'))
                if top_p < 1.0:  # nucleus sampling: keep the smallest set whose cumulative prob >= top_p
                    sorted_logits, sorted_indices = next_token_logits.sort(dim=-1, descending=True)
                    sorted_probs = F.softmax(sorted_logits, dim=-1)
                    cumulative_probs = sorted_probs.cumsum(dim=-1)
                    remove = (cumulative_probs - sorted_probs) > top_p
                    sorted_logits = sorted_logits.masked_fill(remove, float('-inf'))
                    next_token_logits = torch.zeros_like(sorted_logits).scatter(-1, sorted_indices, sorted_logits)
                next_token = torch.multinomial(F.softmax(next_token_logits, dim=-1), num_samples=1)
            generated = torch.cat((generated, next_token), dim=1)
        return generated
