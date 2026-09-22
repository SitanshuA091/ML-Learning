import torch
import torch
import torch.nn.functional as F


def greedy_decode(logits: torch.Tensor) -> int:
    # Convert logits into probabilities over the vocabulary.
    probs = F.softmax(logits, dim=-1)

    # Return the vocabulary index with the highest probability.
    token_id = torch.argmax(probs, dim=-1)

    return token_id.item()


def apply_temperature(
    logits: torch.Tensor,
    temperature: float
) -> torch.Tensor:

    if temperature <= 0:
        raise ValueError("Temperature must be greater than 0.")

    # Temperature is applied BEFORE softmax.
    scaled_logits = logits / temperature

    # Convert the scaled logits into probabilities.
    probs = F.softmax(scaled_logits, dim=-1)

    return probs


logits = torch.tensor([2.1, 0.8, 3.4, 1.2, -0.5])

print("Greedy:", greedy_decode(logits))

print("T = 0.5:", apply_temperature(logits, 0.5))
print("T = 1.0:", apply_temperature(logits, 1.0))
print("T = 2.0:", apply_temperature(logits, 2.0))