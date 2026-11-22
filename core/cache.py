import torch

class Cache:
    def __init__(
            self,
            n_layers: int,
            batch_size: int,
            n_heads: int,
            max_len: int,
            head_dim: int,
            device: torch.device,
            dtype: torch.dtype = torch.float32
    ):
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.n_heads = n_heads
        self.max_len = max_len
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype

        self.k = torch.zeros(n_layers, batch_size, n_heads, max_len, head_dim, device=device, dtype=dtype)
        self.v = torch.zeros_like(self.k)

    def get(self, layer: int, start_pos: int, end_pos: int):
        return self.k[layer, :, :, start_pos:end_pos, :], self.v[layer, :, :, start_pos:end_pos, :]

    def update(self, layer: int, k: torch.Tensor, v: torch.Tensor, start_pos: int):
        T = k.size(2)
        self.k[layer, :, :, start_pos:start_pos + T, :] = k
        self.v[layer, :, :, start_pos:start_pos + T, :] = v

    def subset(self, rows: torch.Tensor) -> "Cache":
        new = Cache(self.n_layers, len(rows), self.n_heads, self.max_len,
                    self.head_dim, self.device, self.dtype)
        new.k = self.k[:, rows].clone()
        new.v = self.v[:, rows].clone()
        return new
