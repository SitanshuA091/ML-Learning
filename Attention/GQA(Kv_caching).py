
import torch
import torch.nn as nn
import torch.nn.functional as F


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    batch, n_kv_heads, seq_len, head_dim = x.shape
    x = x[:, :, None, :, :]
    x = x.expand(batch, n_kv_heads, n_rep, seq_len, head_dim)

    return x.reshape(batch, n_kv_heads * n_rep, seq_len, head_dim)


class GroupedQueryAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()

        assert n_heads % n_kv_heads == 0
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout

        self.q_proj = nn.Linear(
            d_model, n_heads * self.head_dim, bias=bias
        )
        self.k_proj = nn.Linear(
            d_model, n_kv_heads * self.head_dim, bias=bias
        )
        self.v_proj = nn.Linear(
            d_model, n_kv_heads * self.head_dim, bias=bias
        )
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

    def forward(
        self,
        x: torch.Tensor,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ):

        ## x: [B, S, d_model]
        ## past_kv: Previously cached K and V, each shaped
        ## [B, n_kv_heads, past_seq_len, head_dim].

        batch, seq_len, _ = x.shape

        # 1. Project the CURRENT input tokens only.
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(batch, seq_len, self.n_heads, self.head_dim)
        k = k.view(batch, seq_len, self.n_kv_heads, self.head_dim)
        v = v.view(batch, seq_len, self.n_kv_heads, self.head_dim)

        # [B, S, heads, D] -> [B, heads, S, D]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        #  KV CACHE: append new K/V to previously cached K/V.
        #    Old tokens are NOT passed through k_proj/v_proj again.
        if past_kv is not None:
            past_k, past_v = past_kv

            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        # Save the updated cache BEFORE repeating KV heads.
        #    Cache only n_kv_heads, not n_heads.
        present_kv = (k, v) if use_cache else None

        # Repeat KV heads for attention computation.
        # The cache itself remains compact.
        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        # Causal attention.
        # With a multi-token prefill, causal masking is required.
        # With a single new token and a cache containing all past
        if past_kv is not None and seq_len == 1:
            is_causal = False
        else:
            is_causal = True

        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            is_causal=is_causal,
            dropout_p=self.dropout if self.training else 0.0,
        )

        output = output.transpose(1, 2).contiguous()
        output = output.view(batch, seq_len, self.d_model)
        output = self.out_proj(output)

        return output, present_kv
