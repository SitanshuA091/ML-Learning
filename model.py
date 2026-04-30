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


class ResidualConnection(nn.Module):
    
        def __init__(self, features: int, dropout: float) -> None:
            super().__init__()
            self.dropout = nn.Dropout(dropout)
            self.norm = LayerNormalization(features)
    
        def forward(self, x, sublayer):
            return x + self.dropout(sublayer(self.norm(x)))

class LayerNormalization(nn.Module):
    def __init__(self, epsilon : float) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.aplha = nn.Parameter(torch.zeros(1))
        self.bias = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        mean = x.mean(dim = -1, keepdim = True)
        std = x.std(dim = -1, keepdim = True)
        return self.aplha * (x -mean) / (std + self.epsilon) + self.bias
        
class FeedForward(nn.Module):
    def __init__(self, d_model : int, d_ff: int, dropout)-> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) 
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
        
    def forward(self, x):
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
    

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float)-> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h ==0, "dimension should be divisble by h"
        self.d_k = d_model/h
        self.w_q = nn.Linear(d_model, d_model) 
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    @staticmethod
    def attention(query, key, value, mask, dropout):
        d_k = query.shape[-1]
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        
    def forward(self, q, k, v, mask):
        query = self.w_q(q)
        key = self.w_k(k) 
        value = self.w_v(v) 

        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        x, self.attention_scores = MultiHeadAttention.attention(query, key, value, mask, self.dropout)
        
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        return self.w_o(x)

class encoder(nn.Module):

    def __init__(self, features: int, self_attention_block: MultiHeadAttention, feed_forward_block: FeedForward, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x
    
class Encoder(nn.Module):

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)\
            
class decoder(nn.Module):

    def __init__(self, features: int, self_attention_block: MultiHeadAttention, cross_attention_block: MultiHeadAttention, feed_forward_block: FeedForward, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x
    
class Decoder(nn.Module):

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)

class ProjectionLayer(nn.Module):

    def __init__(self, d_model, vocab_size) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x) -> None:
        return self.proj(x)
    
##Transformer build final left
class Transformer(nn.Module):
    def __init__(self, encoder: Encoder,decoder: Decoder, source_emb : InputEmbedding, trgt_emb: InputEmbedding, source_pos: PositionalEncoding, trgt_pos: PositionalEncoding,projection_layer: ProjectionLayer)-> None:
        super.__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.source_emb = source_emb
        self.trgt_emb =  trgt_emb
        self.source_pos = source_pos
        self.trgt_pos = trgt_pos
        self.projection_layer = projection_layer
        
    def encode(self, source, source_mask):
        source = self.source_emb(source)
        source = self.source_pos(source)
        return self.encoder(source, source_mask)
    
    def decode(self, encoder_output, source_mask, trgt, trgt_mask):
        trgt = self.trgt_emb(trgt)
        trgt = self.trgt_pos(trgt)
        return self.decoder(trgt, encoder_output, source_mask, trgt_mask)
    
    def project(self, x):
        return self.projection_layer(x)
    
def Build_Model(source_vocab_size: int, trgt_vocab_size: int, source_seq_len, trgt_seq_len, d_model: int =512, N: int = 6, h: int =8, dropout : float =0.1, d_ff = 2048):
    source_emb = InputEmbedding(d_model, source_vocab_size)
    trgt_emb = InputEmbedding(d_model, trgt_vocab_size)
    source_pos = PositionalEncoding(d_model, source_seq_len, dropout)
    trgt_pos = PositionalEncoding(d_model, trgt_seq_len, dropout)
    
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttention(d_model, h, dropout)
        feed_forward_block = FeedForward(d_model, d_ff, dropout)
        encoder_block = encoder(d_model, encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttention(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttention(d_model, h, dropout)
        feed_forward_block = FeedForward(d_model, d_ff, dropout)
        decoder_block = decoder(d_model, decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)
    
    # Create the encoder and decoder
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))
    
    # Create the projection layer
    projection_layer = ProjectionLayer(d_model, trgt_vocab_size)
    
    # Create the transformer
    transformer = Transformer(encoder, decoder, source_emb, trgt_emb, source_pos, trgt_pos, projection_layer)
    
    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer
    