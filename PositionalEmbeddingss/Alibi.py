import math
import torch


def get_slopes(n: int):
    def get_slopes_power_of_2(n):
        # First slope and common ratio of the geometric sequence.
        start = 2 ** (-8 / n)
        ratio = start

        return [
            start * (ratio ** i)
            for i in range(n)
        ]

    # Simple case: number of heads is a power of 2.
    if math.log2(n).is_integer():
        return get_slopes_power_of_2(n)

    # For a non-power-of-2 head count, first generate slopes for the
    # largest power of 2 smaller than n.
    closest_power_of_2 = 2 ** math.floor(math.log2(n))

    # Add additional slopes sampled from the next power-of-2 sequence.
    return (
        get_slopes_power_of_2(closest_power_of_2)
        + get_slopes(2 * closest_power_of_2)[0::2][
            : n - closest_power_of_2
        ]
    )


## Sample ALiBi attention-bias construction


num_heads = 8
seq_len = 6

slopes = torch.tensor(
    get_slopes(num_heads),
    dtype=torch.float32
)

# Shape:
# [num_heads, 1, 1]
#
# Each attention head receives its own slope.
slopes = slopes.view(num_heads, 1, 1)


# Token positions in the sequence:
#
# m = [0, 1, 2, ..., seq_len - 1]
positions = torch.arange(seq_len)


# query_position: [[0],[1],[2],...]
# key_position: [[0, 1, 2, ...]]
query_position = positions[:, None]
key_position = positions[None, :]


# Distance from query i to key j.
# For causal attention, query q_i is allowed to attend only to keys k_j from 1 to i
# Example:
# query position i = 4
# keys:       0  1  2  3  4
# distances:  4  3  2  1  0
relative_distance = query_position - key_position

# Shape after broadcasting:
# [num_heads, seq_len, seq_len]
alibi_bias = -slopes * relative_distance

## Causal mask

# Query q_i must not attend to any future key k_j where j > i.
# Upper triangular entries in that matrice therefore become -inf.
causal_mask = torch.triu(
    torch.full(
        (seq_len, seq_len),
        float("-inf")
    ),
    diagonal=1
)

## Example usage inside attention

# q shape: [batch, num_heads, seq_len, head_dim]
# k shape: [batch, num_heads, seq_len, head_dim]
# attention_scores = q @ k.transpose(-2, -1)
# attention_scores /= math.sqrt(head_dim)


# ALiBi is added directly to the QK attention logits.
# attention_scores shape:
#     [batch, heads, query_position, key_position]
# alibi_bias shape:
#     [heads, query_position, key_position]
# Broadcasting handles the batch dimension.
# attention_scores = attention_scores + alibi_bias


# Then causal masking prevents q_i from attending to keys after i.
#
# attention_scores = attention_scores + causal_mask

# Finally
# attention_weights = torch.softmax(attention_scores, dim=-1)