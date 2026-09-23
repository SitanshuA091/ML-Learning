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

    return logits / temperature

def top_k_sampling(logits: torch.Tensor, k: int) -> int:
    if k <= 0:
        raise ValueError("k must be greater than 0.")

    if k > logits.size(-1):
        raise ValueError("k cannot be larger than vocabulary size.")

    # Get the k largest logits and their original vocabulary indices.
    top_k_logits, top_k_indices = torch.topk(logits, k)

    # Convert only those top-k logits into a probability distribution.
    top_k_probs = F.softmax(top_k_logits, dim=-1)

    # Sample an index WITHIN the top-k candidates.
    sampled_position = torch.multinomial(
        top_k_probs,
        num_samples=1
    )

    # Convert that position back to the actual vocabulary token ID.
    token_id = top_k_indices[sampled_position]

    return token_id.item()

def top_p_sampling(logits: torch.Tensor, p: float) -> int:
    if not 0.0 < p <= 1.0:
        raise ValueError("p must be between 0 and 1.")

    probs = F.softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(
        probs,
        descending=True
    )

    # Calculate cumulative probability.
    cumulative_probs = torch.cumsum(
        sorted_probs,
        dim=-1
    )

    # Find how many tokens are required to reach p, Keep the token that actually crosses the threshold as well.
    keep_mask = cumulative_probs - sorted_probs < p

    # Remove probabilities outside the nucleus.
    filtered_probs = sorted_probs * keep_mask

    # Renormalize so remaining probabilities sum to 1.
    filtered_probs = filtered_probs / filtered_probs.sum()

    # Sample a position within the sorted candidates.
    sampled_position = torch.multinomial(
        filtered_probs,
        num_samples=1
    )

    # Map back to the original vocabulary token ID.
    token_id = sorted_indices[sampled_position]

    return token_id.item()

def sample_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None
) -> int:

    logits = apply_temperature(logits, temperature)

    if top_k is not None and top_p is None:
        return top_k_sampling(logits, top_k)

    if top_p is not None and top_k is None:
        return top_p_sampling(logits, top_p)

    if top_k is None and top_p is None:
        probs = torch.softmax(logits, dim=-1)
        token_id = torch.multinomial(probs, num_samples=1)
        return token_id.item()

    raise ValueError(
        "For this learning implementation, use either top_k or top_p, not both."
    )

# logits = torch.tensor([2.1, 0.8, 3.4, 1.2, -0.5])
# print("Greedy:", greedy_decode(logits))

# print("T = 0.5:", apply_temperature(logits, 0.5))
# print("T = 1.0:", apply_temperature(logits, 1.0))
# print("T = 2.0:", apply_temperature(logits, 2.0))

# logits = torch.tensor([2.1, 0.8, 3.4, 1.2, -0.5])
# token = top_k_sampling(logits, k=3)
# print(token)