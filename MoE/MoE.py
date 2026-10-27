
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ExpertModule(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x


class MOEGating(nn.Module):
    def __init__(self, input_size, num_experts, top_k=2):
        super().__init__()

        if not 1 <= top_k <= num_experts:
            raise ValueError("top_k must be between 1 and num_experts")

        self.input_size = input_size
        self.num_experts = num_experts
        self.top_k = top_k

        self.weight = nn.Parameter(
            torch.empty(num_experts, input_size)
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):
        # [batch, seq_len, input_size] -> [tokens, input_size]
        x_flat = x.reshape(-1, self.input_size)

        # One routing score per token per expert
        scores = F.linear(x_flat, self.weight)
        scores = F.softmax(scores, dim=-1)

        # Select the top-k experts for each token
        top_k_scores, top_k_indices = torch.topk(
            scores, k=self.top_k, dim=-1
        )

        # Normalize selected expert weights
        top_k_scores = top_k_scores / (
            top_k_scores.sum(dim=-1, keepdim=True) + 1e-10
        )

        return top_k_indices, top_k_scores


class MixtureOfExperts(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        num_experts,
        top_k=2,
    ):
        super().__init__()

        self.num_experts = num_experts
        self.top_k = top_k
        self.output_size = output_size

        self.experts = nn.ModuleList([
            ExpertModule(input_size, hidden_size, output_size)
            for _ in range(num_experts)
        ])

        self.gate = MOEGating(input_size, num_experts, top_k)

    def forward(self, x):
        batch_size, seq_len, input_size = x.shape

        # Routing decisions: [tokens, top_k]
        expert_indices, expert_weights = self.gate(x)

        x_flat = x.reshape(-1, input_size)
        num_tokens = x_flat.shape[0]

        # Each token's output is the weighted sum of its selected experts
        final_output = x_flat.new_zeros(
            (num_tokens, self.output_size)
        )

        # Process each selected routing slot
        for k in range(self.top_k):
            current_indices = expert_indices[:, k]
            current_weights = expert_weights[:, k]

            # Dispatch tokens to their selected experts
            for expert_id in range(self.num_experts):
                mask = current_indices == expert_id

                if mask.any():
                    expert_input = x_flat[mask]
                    expert_output = self.experts[expert_id](expert_input)

                    weighted_output = (
                        expert_output
                        * current_weights[mask].unsqueeze(-1)
                    )

                    # Accumulate outputs for the original token positions
                    final_output[mask] += weighted_output

        return final_output.reshape(
            batch_size, seq_len, self.output_size
        )


def test_moe():
    input_size = 64
    hidden_size = 128
    output_size = 32
    num_experts = 4
    top_k = 2
    batch_size = 16
    seq_len = 8

    model = MixtureOfExperts(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        num_experts=num_experts,
        top_k=top_k,
    )

    x = torch.randn(batch_size, seq_len, input_size)

    output = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    assert output.shape == (
        batch_size, seq_len, output_size
    ), "Output shape mismatch"

    expert_indices, expert_weights = model.gate(x)

    print(f"Expert indices shape: {expert_indices.shape}")
    print(f"Expert weights shape: {expert_weights.shape}")

    expected_shape = (batch_size * seq_len, top_k)

    assert expert_indices.shape == expected_shape
    assert expert_weights.shape == expected_shape

    weight_sums = expert_weights.sum(dim=-1)

    assert torch.allclose(
        weight_sums,
        torch.ones_like(weight_sums),
        atol=1e-6,
    ), "Expert weights should sum to 1"

    # Verify that gradients flow through the model
    loss = output.square().mean()
    loss.backward()

    assert model.gate.weight.grad is not None
    assert model.experts[0].fc1.weight.grad is not None

    print("All tests passed!")


if __name__ == "__main__":
    test_moe()
