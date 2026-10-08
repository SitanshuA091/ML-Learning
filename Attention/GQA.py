import torch
import torch.nn as nn
import torch.nn.functional as F


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    # Repeat each KV head n_rep times.

    if n_rep == 1:
        return x

    batch, n_kv_heads, seq_len, head_dim = x.shape

    # [B, KV, S, D] -> [B, KV, 1, S, D]
    x = x[:, :, None, :, :]

    # [B, KV, 1, S, D] -> [B, KV, n_rep, S, D]
    x = x.expand(batch, n_kv_heads, n_rep, seq_len, head_dim)

    # Merge KV-head and repetition dimensions
    # [B, KV, n_rep, S, D] -> [B, KV * n_rep, S, D]
    return x.reshape(batch, n_kv_heads * n_rep, seq_len, head_dim)


class GroupedQueryAttention(nn.Module):
    # Each KV head is shared by:
    # n_heads // n_kv_heads - groups

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        bias: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()

        assert n_heads % n_kv_heads == 0, (
            "n_heads must be divisible by n_kv_heads"
        )

        assert d_model % n_heads == 0, (
            "d_model must be divisible by n_heads"
        )

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads

        # Number of query heads sharing each KV head
        self.n_rep = n_heads // n_kv_heads

        self.head_dim = d_model // n_heads

        # Q has n_heads, while K/V only have n_kv_heads.
        self.q_proj = nn.Linear(
            d_model,
            n_heads * self.head_dim,
            bias=bias,
        )

        self.k_proj = nn.Linear(
            d_model,
            n_kv_heads * self.head_dim,
            bias=bias,
        )

        self.v_proj = nn.Linear(
            d_model,
            n_kv_heads * self.head_dim,
            bias=bias,
        )

        self.out_proj = nn.Linear(
            d_model,
            d_model,
            bias=bias,
        )

        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        # Project input into Q, K and V.
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Split the embedding dimension into attention heads.
        #
        # Q: [B, S, H, D]
        # K: [B, S, KV, D]
        # V: [B, S, KV, D]
        q = q.view(batch, seq_len, self.n_heads, self.head_dim)
        k = k.view(batch, seq_len, self.n_kv_heads, self.head_dim)
        v = v.view(batch, seq_len, self.n_kv_heads, self.head_dim)

        # Move the head dimension before sequence dimension:
        #
        # [B, S, H, D] -> [B, H, S, D]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Expand K/V so that their number of heads matches Q.
        # Each KV head is repeated 4 times.
        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        # Now:
        # Q: [B, H,  S, D] n same foe  K & V
        attn_output = F.scaled_dot_product_attention(
            q,k,v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )

        # [B, H, S, D] -> [B, S, H, D]
        attn_output = attn_output.transpose(1, 2)

        # Merge attention heads:
        # [B, S, H, D] -> [B, S, H * D]
        attn_output = attn_output.contiguous().view(
            batch,
            seq_len,
            self.d_model,
        )

        return self.out_proj(attn_output)