import math

import torch
from torch import nn

"""
https://www.youtube.com/watch?v=ISNdQcPhsts&t=372s&ab_channel=UmarJamil
"""


class InputEmbeddings(nn.Module):
    """
    Step 1 (Input Embeddings)
    """

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.model_sqrt = math.sqrt(self.d_model)
        self.vocab_size = vocab_size

        self.embeddings = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        embeddings = self.embeddings(x) * self.model_sqrt
        return embeddings


class PositionalEncoding(nn.Module):
    """
    Step 2 Positional Encoding
    Dropout for less overfit
    """

    def __init__(self, d_model: int, sequence_length: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.sequence_length = sequence_length
        self.dropout = nn.Dropout(dropout)

        # Create a matrix of shape (sequence_legth, d_model)
        # We need different (sequence_legth) number of (d_model) dimensional matrices
        pe = torch.zeros(sequence_length, d_model)

        # Create a vector of shape (sequence_length, 1) This holds position indexes
        position = torch.arange(0, sequence_length, dtype=torch.float).unsqueeze(1)

        # Denominator in the PE formula
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Apply the sin to even positions
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, sequence_length, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # x.shape[1] means take the positional encodings till sequence_length of X
        # It means, If X is a sentence that has 6 words it is (6, 512) matrix we add (6, 512) matrix of PE
        # PE has no learnable parameters
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad(False)
        return self.dropout(x)


class LayerNormalization(nn.Module):
    """
    Step 4 Layer Normalization
    """

    def __init__(self, eps: float = 10e-6):
        super().__init__()
        self.eps = eps

        # (X * alpha) + beta
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.ones(1))

    def forward(self, x):
        mean = torch.mean(x, dim=-1, keepdim=True)
        std = torch.std(x, dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class FeedForward(nn.Module):
    """
    Step 6 Feed Forward (Uses Encoder & Decoder)

    D_ff is a hyperparameter, it is 2048 in paper
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.layer_1 = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.layer_2 = nn.Sequential(
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        """
        x (batch_size, sequence_length, d_model)
            - layer_1 : (batch_size, sequence_length, d_model) -> (batch_size, sequence_length, d_ff)
            - layer_2 : (batch_size, sequence_length, d_ff) -> (batch_size, sequence_length, d_model)
        """
        return self.layer_2(self.layer_1(x))


class MultiHeadAttention(nn.Module):
    """
    In multi head attention, every head has access to all words in the sequence but small parts of its feature vectors
    For example: 6 words -> (6, 512) every words has 512 feature vector
    If we have 4 attention heads all the heads has (6, 128) feature vector
    After all the calculations single head results are concatenated into (6, 512) again
    This is used for different meanings and usages(verb, noun) of words
    """

    def __init__(self, d_model: int, num_of_heads: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.num_of_heads = num_of_heads

        assert self.d_model % self.num_of_heads == 0

        # While reading paper d_k & d_v are equal
        # d_v comes from last multiplication made with V matrix
        self.d_k = self.d_model // self.num_of_heads

        self.w_q = nn.Linear(d_model, d_model)  # (6, 512) * (512, 512) -> (6, 512) so weights are (512, 512)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.w_o = nn.Linear(d_model, d_model)  # Same matrix, it is used after concat of results

        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        """
        query (batch_size, num_of_heads, sequence_legth, d_k)
        key (batch_size, num_of_heads, sequence_legth, d_k)
        value (batch_size, num_of_heads, sequence_legth, d_k)
        mask (sequence_length, sequence_length)
        dropout:
        """
        d_k = query.shape[-1]
        # (batch_size, num_of_heads, sequence_legth, d_k) @ (batch_size, num_of_heads, d_k, sequence_length)
        # (batch_size, num_of_heads, sequence_length, sequence_length) attention scores
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask:
            attention_scores.masked_fill(mask == 0, -1e9)
        # (batch_size, num_of_heads, sequence_length, sequence_length)
        attention_scores = attention_scores.softmax(dim=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        # (batch_size, sequence_legth, d_model) -> (batch_size, sequence_legth, d_model)
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        # Splitting matrices into smaller matrices
        # Transpose made for each head should see all the words in sentence it should be
        # (batch_size,[head] sequence_length, d_k)
        query = query.view(query.shape[0], query.shape[1], self.num_of_heads, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.num_of_heads, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.num_of_heads, self.d_k).transpose(1, 2)

        x, self.attention_scores = MultiHeadAttention.attention(query=query,
                                                                key=key,
                                                                value=value,
                                                                mask=mask,
                                                                dropout=self.dropout)

        # (batch_size, num_of_heads, sequence_length, d_k) -> (batch_size, sequence_length, num_of_heads, d_k)
        x = x.transpose(1, 2)
        # (batch_size, sequence_length, num_of_heads, d_k) -> (batch_size, sequence_length, d_model)
        x.contiguous().view(x.shape[0], -1, self.num_of_heads * self.d_k)
        # multiply x with w_o
        return self.w_o(x)


class ResidualConnection(nn.Module):
    """
    Residual connections in Add + Norm layer in the diagram that's why LayerNorm is in here
    """

    def __init__(self, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        sublayer_output = self.dropout(sublayer(self.norm(x)))
        return x + sublayer_output


class EncoderBlock(nn.Module):
    """
    N encoder block needed for attention
    """

    def __init__(self, self_attention_block: MultiHeadAttention,
                 feed_forward_block: FeedForward,
                 dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.dropout = nn.Dropout()

        self.residual_connections = nn.ModuleList([
            ResidualConnection(dropout) for _ in range(2)
        ])

    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block(x))
        return x


class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttention,
                 cross_attention_block: MultiHeadAttention,
                 feed_forward_block: FeedForward,
                 dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.dropout = nn.Dropout(dropout)

        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))

        # query from decoder, key & value from encoder, mask from encoder
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output,
                                                                                 src_mask))

        x = self.residual_connections[2](x, self.feed_forward_block)
        return x


class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)


class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model

        self.layer = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # (batch_size, sequence_length, d_model) -> (batch_size, sequence_length, vocab_size)
        return torch.log_softmax(self.layer(x), dim=-1)


class Transformer(nn.Module):
    def __init__(self, encoder: Encoder,
                 decoder: Decoder,
                 src_embed: InputEmbeddings,
                 tgt_embed: InputEmbeddings,
                 src_pos: PositionalEncoding,
                 tgt_pos: PositionalEncoding,
                 projection_layer: ProjectionLayer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x):
        return self.projection_layer(x)


def build_transformer(src_vocab_size: int,
                      tgt_vocab_size: int,
                      src_seq_length: int,
                      tgt_seq_length: int,
                      d_model: int = 512,
                      num_of_layers: int = 6,
                      num_of_heads: int = 8,
                      dropout: float = .01,
                      d_ff: int = 2048):
    # Create embedding layers
    src_embed = InputEmbeddings(d_model=d_model, vocab_size=src_vocab_size)
    tgt_embed = InputEmbeddings(d_model=d_model, vocab_size=tgt_vocab_size)

    # Create positional encodings
    src_pos = PositionalEncoding(d_model=d_model, sequence_length=src_seq_length, dropout=dropout)
    tgt_pos = PositionalEncoding(d_model=d_model, sequence_length=tgt_seq_length, dropout=dropout)

    # Create encoder blocks
    encoder_layers = []
    for _ in range(num_of_layers):
        encoder_self_attention = MultiHeadAttention(d_model=d_model, num_of_heads=num_of_heads, dropout=dropout)
        encoder_feed_forward = FeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)
        encoder_layers.append(
            EncoderBlock(self_attention_block=encoder_self_attention,
                         feed_forward_block=encoder_feed_forward,
                         dropout=dropout)
        )
    # Create encoder
    encoder = Encoder(layers=nn.ModuleList(encoder_layers))

    # Create decoder blocks
    decoder_layers = []
    for _ in range(num_of_layers):
        decoder_self_attention = MultiHeadAttention(d_model=d_model, num_of_heads=num_of_heads, dropout=dropout)
        decoder_cross_attention = MultiHeadAttention(d_model=d_model, num_of_heads=num_of_heads, dropout=dropout)
        decoder_feed_forward = FeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)
        decoder_layers.append(
            DecoderBlock(self_attention_block=decoder_self_attention,
                         cross_attention_block=decoder_cross_attention,
                         feed_forward_block=decoder_feed_forward,
                         dropout=dropout)
        )

    # Create decoder
    decoder = Decoder(layers=nn.ModuleList(decoder_layers))

    # Create projection
    projection_layer = ProjectionLayer(d_model=d_model, vocab_size=tgt_vocab_size)

    transformer = Transformer(encoder=encoder,
                              decoder=decoder,
                              src_embed=src_embed,
                              tgt_embed=tgt_embed,
                              src_pos=src_pos,
                              tgt_pos=tgt_pos,
                              projection_layer=projection_layer)

    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)

    return transformer
