import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest

# 1. Self-Contained SwiGLU FeedForward Expert
class SimpleSwiGLUFeedForward(nn.Module):
    def __init__(self, d_model, hidden_dim, dropout=0.0):
        super().__init__()
        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, d_model, bias=False)
        self.w3 = nn.Linear(d_model, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w2(self.dropout(F.silu(self.w1(x)) * self.w3(x)))


# 2. Independent MoE Layer
class MoELayer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.d_model = config.d_model
        self.n_experts = config.n_experts
        self.n_active_experts = config.n_active_experts
        self.dropout = getattr(config, "dropout", 0.0)
        self.aux_loss_coef = getattr(config, "moe_aux_loss_coef", 0.01)

        hidden_dim = getattr(
            config,
            "moe_expert_hidden_dim",
            getattr(config, "d_ff", 4 * self.d_model)
        )

        self.router = nn.Linear(
            self.d_model,
            self.n_experts,
            bias=False
        )

        self.experts = nn.ModuleList([
            SimpleSwiGLUFeedForward(
                self.d_model,
                hidden_dim,
                dropout=self.dropout
            )
            for _ in range(self.n_experts)
        ])

    def forward(self, x):
        batch_size, seq_len, d_model = x.shape

        router_logits = self.router(x)
        router_probs = F.softmax(router_logits, dim=-1)

        topk_probs, topk_idx = torch.topk(
            router_probs,
            k=self.n_active_experts,
            dim=-1
        )

        # Normalize top-k probabilities
        topk_probs = topk_probs / topk_probs.sum(
            dim=-1,
            keepdim=True
        )

        importance = router_probs.mean(dim=(0, 1))

        with torch.no_grad():
            load = torch.zeros(
                self.n_experts,
                device=x.device,
                dtype=x.dtype
            )
            first_choice = topk_idx[..., 0].reshape(-1)
            load.scatter_add_(
                0,
                first_choice,
                torch.ones_like(
                    first_choice,
                    dtype=x.dtype
                )
            )
            load /= first_choice.numel()

        aux_loss = (
            self.aux_loss_coef
            * self.n_experts
            * torch.sum(importance * load)
        )

        flat_x = x.reshape(-1, d_model)
        flat_idx = topk_idx.reshape(-1, self.n_active_experts)
        flat_prob = topk_probs.reshape(-1, self.n_active_experts)

        output = torch.zeros_like(flat_x)

        for expert_id in range(self.n_experts):
            mask = flat_idx == expert_id
            if not mask.any():
                continue

            token_ids = mask.any(dim=-1).nonzero(as_tuple=True)[0]
            expert_input = flat_x[token_ids]
            expert_output = self.experts[expert_id](expert_input)

            weights = (
                flat_prob[token_ids]
                * mask[token_ids].float()
            ).sum(dim=-1)

            output[token_ids] += (
                expert_output
                * weights.unsqueeze(-1)
            )

        output = output.view(batch_size, seq_len, d_model)
        return output, aux_loss


# 3. PyTest Unit Tests
class MoEConfig:
    def __init__(self, d_model=64, n_experts=4, n_active_experts=2, dropout=0.1, moe_aux_loss_coef=0.01, d_ff=256):
        self.d_model = d_model
        self.n_experts = n_experts
        self.n_active_experts = n_active_experts
        self.dropout = dropout
        self.moe_aux_loss_coef = moe_aux_loss_coef
        self.d_ff = d_ff

@pytest.fixture
def config():
    return MoEConfig()

def test_moe_output_shape(config):
    layer = MoELayer(config)
    batch_size, seq_len = 2, 16
    x = torch.randn(batch_size, seq_len, config.d_model)

    output, aux_loss = layer(x)

    # Check output shape matches input shape
    assert output.shape == (batch_size, seq_len, config.d_model)
    # Check aux loss is a scalar tensor
    assert aux_loss.ndim == 0
    assert not torch.isnan(aux_loss)

def test_moe_gradient_flow(config):
    layer = MoELayer(config)
    x = torch.randn(2, 8, config.d_model, requires_grad=True)

    output, aux_loss = layer(x)
    total_loss = output.sum() + aux_loss
    total_loss.backward()

    # Check that gradients flow back to inputs and parameters
    assert x.grad is not None
    assert layer.router.weight.grad is not None
    assert layer.experts[0].w1.weight.grad is not None
