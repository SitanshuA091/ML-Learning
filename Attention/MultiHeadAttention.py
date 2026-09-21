import math
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        assert d_model % h == 0, "d_model must be divisible by h"
        
        self.d_model = d_model
        self.h = h
        self.d_k = d_model // h  # Integer division required for tensor shapes
        
        # FUSED WEIGHT MATRICES: Instead of creating 'h' separate weight matrices,(mentioned in README explaination)
        # we create one large (d_model x d_model) linear layer for Q, K, and V.
        self.w_q = nn.Linear(d_model, d_model) 
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        
        # Output projection layer to fuse concatenated head outputs back to d_model
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        

    def attention(query, key, value, mask=None, dropout=None):
        d_k = query.shape[-1]
        
        # 1. Calculate Raw Scores: (Q @ K_transposed) / sqrt(d_k)
        # Shape: (batch_size, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        
        # 2. Optional Causal/Padding Masking
        if mask is not None:
            # Replaces masked 0 positions with a tiny negative value 
            # so softmax turns them into 0% probability.
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
            
        # 3. Softmax along the last dimension (over Keys for each Query)
        attention_scores = attention_scores.softmax(dim=-1)
        
        if dropout is not None:
            attention_scores = dropout(attention_scores)
            
        # 4. Weighted sum over Values: Softmax(Scores) @ V
        # Shape: (batch_size, h, seq_len, d_k)
        x = attention_scores @ value
        
        return x, attention_scores
        
    def forward(self, q, k, v, mask=None):
        batch_size = q.shape[0]
        
        # Linear projections (d_model -> d_model)
        query = self.w_q(q) # Shape: (batch_size, seq_len, d_model)
        key = self.w_k(k)   # Shape: (batch_size, seq_len, d_model)
        value = self.w_v(v) # Shape: (batch_size, seq_len, d_model)

        # Split d_model into 'h' heads and transpose for parallel computation
        # .view() breaks d_model into (h, d_k)
        # .transpose(1, 2) aligns dimensions to: (batch_size, h, seq_len, d_k)
        query = query.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        key = key.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        value = value.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)

        # Run Scaled Dot-Product Attention concurrently on all 'h' heads
        x, self.attention_scores = MultiHeadAttention.attention(query, key, value, mask, self.dropout)
        
        # Concatenate head outputs side-by-side
        # Transpose back to: (batch_size, seq_len, h, d_k)
        # .view() flattens (h, d_k) back into d_model
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        # Final Linear projection W_o to blend multi-head representations
        return self.w_o(x)