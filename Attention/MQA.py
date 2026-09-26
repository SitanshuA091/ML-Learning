import math
import torch
import torch.nn as nn

class MultiQueryAttention(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        assert d_model % h == 0, "d_model must be divisible by h"
        
        self.d_model = d_model
        self.h = h
        self.d_k = d_model // h
        
        # Queries retain 'h' distinct heads -> output dime is d_model (h * d_k)
        self.w_q = nn.Linear(d_model, d_model)
        
        # Keys and Values use ONLY 1 head total -> output dim is d_k
        self.w_k = nn.Linear(d_model, self.d_k)
        self.w_v = nn.Linear(d_model, self.d_k)
        
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask=None, dropout=None):
        d_k = query.shape[-1]
        
        # query shape: (batch_size, h, seq_len, d_k)
        # key.transpose(-2, -1) shape: (batch_size, 1, d_k, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
            
        attention_scores = attention_scores.softmax(dim=-1)
        
        if dropout is not None:
            attention_scores = dropout(attention_scores)
            
        # attention_scores shape: (batch_size, h, seq_len, seq_len)
        # value shape: (batch_size, 1, seq_len, d_k)
        x = attention_scores @ value
        
        return x, attention_scores
        
    def forward(self, q, k, v, mask=None):
        batch_size = q.shape[0]
        
        # 1. Project Queries to (batch_size, seq_len, d_model)
        query = self.w_q(q)
        
        # Project Keys and Values to (batch_size, seq_len, d_k) -> 1 head only
        key = self.w_k(k)
        value = self.w_v(v)

        # Reshape Queries into h heads 
        query = query.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        
        # Reshaping now the Keys and Values into 1 head -> (batch_size, 1, seq_len, d_k)
        key = key.view(batch_size, -1, 1, self.d_k).transpose(1, 2)
        value = value.view(batch_size, -1, 1, self.d_k).transpose(1, 2)

        # Compute attention with broadcasting over the single K/V head
        x, self.attention_scores = MultiQueryAttention.attention(query, key, value, mask, self.dropout)
        
        # Concatenate query head outputs back to d_model
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.w_o(x)