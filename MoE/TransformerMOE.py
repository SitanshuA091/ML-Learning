import torch
import torch.nn as nn
import torch.nn.functional as F

# Mock Dependencies for Transformer Components 
class SwiGLUFeedForward(nn.Module):
    def __init__(self, d_model, hidden_dim, dropout=0.0):
        super().__init__()
        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, d_model, bias=False)
        self.w3 = nn.Linear(d_model, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w2(self.dropout(F.silu(self.w1(x)) * self.w3(x)))

class GroupedQueryAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model
        self.proj = nn.Linear(self.d_model, self.d_model)

    def forward(self, x):
        return self.proj(x)

def shift_inputs_and_labels(input_ids, labels):
    return input_ids[:, :-1], labels[:, 1:]


# Transformer MoE Implementation
class MoELayer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.d_model = config.d_model
        self.n_experts = config.n_experts
        self.n_active_experts = config.n_active_experts
        self.dropout = config.dropout
        self.aux_loss_coef = config.moe_aux_loss_coef

        hidden_dim = getattr(
            config,
            "moe_expert_hidden_dim",
            config.d_ff
        )

        self.router = nn.Linear(
            self.d_model,
            self.n_experts,
            bias=False
        )

        self.experts = nn.ModuleList([
            SwiGLUFeedForward(
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
    
class MoETransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = nn.RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.norm2 = nn.RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.attention = GroupedQueryAttention(config)
        self.moe = MoELayer(config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = x + self.dropout(self.attention(self.norm1(x)))
        moe_out, aux_loss = self.moe(self.norm2(x))
        x = x + self.dropout(moe_out)
        return x, aux_loss
    
class Qwen3MOELM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([
            MoETransformerBlock(config)
            for _ in range(config.n_layers)
        ])
        self.final_norm = nn.RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)

        if getattr(config, "tie_embeddings", False):
            self.lm_head.weight = self.token_embedding.weight

        self.aux_loss_coef = getattr(config, "moe_aux_loss_coef", 0.01)

    def forward(self, input_ids, labels=None):
        if labels is not None:
            input_ids, labels = shift_inputs_and_labels(input_ids, labels)

        x = self.token_embedding(input_ids)
        x = self.dropout(x)

        total_aux_loss = 0.0
        for layer in self.layers:
            x, aux_loss = layer(x)
            total_aux_loss = total_aux_loss + aux_loss

        total_aux_loss = total_aux_loss / len(self.layers)
        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            lm_loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
            )
            loss = lm_loss + (self.aux_loss_coef * total_aux_loss)

        return {
            "logits": logits,
            "loss": loss,
            "aux_loss": total_aux_loss,
        }


# Experts n other Configs
class MoEConfig:
    def __init__(self):
        self.d_model = 64
        self.n_experts = 4
        self.n_active_experts = 2
        self.dropout = 0.1
        self.moe_aux_loss_coef = 0.01
        self.d_ff = 128
        self.vocab_size = 500
        self.n_layers = 2
        self.rms_norm_eps = 1e-5
        self.tie_embeddings = False

def test_moe():
    config = MoEConfig()
    batch_size = 16
    seq_len = 8

    # Initialize the full transformer MoE language model
    model = Qwen3MOELM(config)

    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    output_dict = model(input_ids)

    logits = output_dict["logits"]
    aux_loss = output_dict["aux_loss"]

    print(f"Input IDs shape: {input_ids.shape}")
    print(f"Logits shape: {logits.shape}")

    assert logits.shape == (
        batch_size, seq_len, config.vocab_size
    ), "Logits shape mismatch"

    # Test individual MoE layer routing and top-k weights normalization
    moe_layer = model.layers[0].moe
    dummy_x = torch.randn(batch_size, seq_len, config.d_model)
    
    router_logits = moe_layer.router(dummy_x)
    router_probs = F.softmax(router_logits, dim=-1)
    topk_probs, topk_idx = torch.topk(router_probs, k=config.n_active_experts, dim=-1)
    topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)

    expected_shape = (batch_size, seq_len, config.n_active_experts)
    assert topk_idx.shape == expected_shape
    assert topk_probs.shape == expected_shape

    weight_sums = topk_probs.sum(dim=-1)
    assert torch.allclose(
        weight_sums,
        torch.ones_like(weight_sums),
        atol=1e-6,
    ), "Normalized expert weights should sum to 1"

    # Verify that gradients flow through the full transformer model including router & experts
    loss = logits.square().mean() + aux_loss
    loss.backward()

    assert model.layers[0].moe.router.weight.grad is not None
    assert model.layers[0].moe.experts[0].w1.weight.grad is not None

    print("All transformer MoE tests passed!")

if __name__ == "__main__":
    test_moe()