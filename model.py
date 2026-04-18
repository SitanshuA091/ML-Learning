import torch
import torch.nn as nn
import math

class InputEmbedding(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
    
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype = torch.float).unsqueeze(1)
        denom_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(1000.0)/d_model))
        # for odd positions
        pe[:, 0::2] =torch.sin(position*denom_term)
        pe[:, 1::2] = torch.cos(position*denom_term)
        
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x+ (self.pe[:, :x.shape[1], :]).requires_grad(False)
        return self.dropout

class LayerNormalization(nn.Module):
    def __init__(self, epsilon : float) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.aplha = nn.Parameter(torch.zeros(1))
        self.bias = nn.Parameter(torch.zeros(1))
        