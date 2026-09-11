import torch
import torch.nn as nn


class ROPE(nn.Module):

    @staticmethod
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        # Rotate each adjacent pair of hidden dimensions: (x1, x2) -> (-x2, x1)
        x = x.reshape(*x.shape[:-1], -1, 2)

        x1 = x[..., 0]
        x2 = x[..., 1]

        x_rotated = torch.stack((-x2, x1), dim=-1)

        return x_rotated.flatten(-2)


    def __init__(
        self,
        dim: int,
        max_seq_len: int,
        base: float = 10000.0
    ):
        super().__init__()

        assert dim % 2 == 0, "RoPE dimension must be even."

        self.dim = dim
        self.max_seq_len = max_seq_len

        # theta_i = 1 / base^(2i / dim)
        # One frequency is needed for every pair of hidden dimensions.
        inv_freq = 1.0 / (
            base ** (
                torch.arange(0, dim, 2, dtype=torch.float32) / dim
            )
        )

        # Position variable m = 0, 1, 2, ..., max_seq_len - 1
        positions = torch.arange(
            max_seq_len,
            dtype=torch.float32
        )

        # Compute m * theta_i for every position m and frequency theta_i.
        # Shape: [max_seq_len, dim // 2]
        freqs = torch.einsum(
            "m,d->md",
            positions,
            inv_freq
        )

        # Repeat each frequency twice because each theta_i rotates
        # one adjacent pair of hidden dimensions.
        #
        # [θ1, θ2, θ3] -> [θ1, θ1, θ2, θ2, θ3, θ3]
        emb = torch.repeat_interleave(
            freqs,
            repeats=2,
            dim=-1
        )

        # Precompute cos(mθ) and sin(mθ) for all supported positions.
        self.register_buffer(
            "cos",
            emb.cos(),
            persistent=False
        )

        self.register_buffer(
            "sin",
            emb.sin(),
            persistent=False
        )


    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:

        # x can typically have shape:
        # [batch, seq_len, dim]
        # or
        # [batch, heads, seq_len, head_dim]

        seq_len = x.shape[-2]

        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds "
                f"max_seq_len={self.max_seq_len}"
            )

        # Select positional rotations required for this sequence.
        # Shape: [seq_len, dim]
        cos = self.cos[:seq_len]
        sin = self.sin[:seq_len]

        # Add leading singleton dimensions so cos/sin broadcast over
        # batch and, if present, attention-head dimensions.
        shape = [1] * (x.ndim - 2) + [seq_len, self.dim]

        cos = cos.view(*shape)
        sin = sin.view(*shape)

        # R_theta(x) = x cos(theta) + rotate_half(x) sin(theta)
        x_rotated = (
            x * cos
            + self.rotate_half(x) * sin
        )

        return x_rotated