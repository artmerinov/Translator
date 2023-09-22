import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math


class InputEmbedding(nn.Module):
    """Convert input tokens into scaled embeddings."""

    def __init__(self, embed_size: int, vocab_size: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embed_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x) * math.sqrt(self.embed_size)


class PositionalEncoding(nn.Module):
    """Inject information about relative or absolute positions of tokens."""

    def __init__(self, embed_size: int, max_len: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # create a matrix of shape (max_len, embed_size)
        # so for each word we will calculate embed_size values
        pe = torch.zeros(max_len, embed_size)
        # create a vector "position" of shape (max_len, 1)
        # that will represent the position of word inside the sentence
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_size, 2) * (-math.log(10_000.0) / embed_size)
        )
        # apply sin to even positions and cos to odd positions
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # so far we have the shape (max_len, embed_size)
        # threfore we add batch dimention to the tensor to get (1, max_len, embed_size) shape
        # so we can apply it to the whole batch of sentences
        pe = pe.unsqueeze(0)
        # "pe" is not a learned parameter
        # but we want it to be saved in the file
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # add positioal encoding to every word inside the sentence
        # positional encoding is fixed
        # it is Variable (and not nn.Parameter) because we dont want
        # to add it into the modules parameters
        x = x + Variable(self.pe[:, : x.shape[1], :], requires_grad=False)
        x = self.dropout(x)
        return x


class LayerNormalisation(nn.Module):
    """Layer normalisation."""

    def __init__(self, embed_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        # prevent divition by zero
        self.eps = eps
        # introduce learnable parameteres alpha (multiplicative) and bias (additive)
        self.alpha = nn.Parameter(data=torch.ones(embed_size), requires_grad=True)
        self.bias = nn.Parameter(data=torch.zeros(embed_size), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # keep the dimension for broadcasting
        # because by default mean and std squeezes dimention
        mu = x.mean(dim=-1, keepdim=True)
        sd = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mu) / (sd + self.eps) + self.bias


class FeedForwardBlock(nn.Module):
    """Fully connected feed-forward network."""

    def __init__(self, embed_size: int, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(embed_size, hidden_size, bias=True)
        self.fc2 = nn.Linear(hidden_size, embed_size, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(self.relu(self.fc1(x))))


class MultiHeadAttentionBlock(nn.Module):
    """Multi-head attention."""

    def __init__(self, embed_size: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.embed_size = embed_size  # d_model
        self.heads = heads  # h
        self.head_dim = embed_size // heads  # d_k

        assert (
            embed_size % heads == 0
        ), "Embedding size needs to be divisible by number of heads"

        self.w_q = nn.Linear(embed_size, embed_size, bias=False)  # Wq
        self.w_k = nn.Linear(embed_size, embed_size, bias=False)  # Wk
        self.w_v = nn.Linear(embed_size, embed_size, bias=False)  # Wv
        self.w_o = nn.Linear(embed_size, embed_size, bias=False)  # Wo
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        q_new = self.w_q(q)  # Q' = Q * Wq
        k_new = self.w_k(k)  # K' = K * Wk
        v_new = self.w_v(v)  # V' = V * Wv

        # split matrices along embedding dimention into heads parts
        # (batch_size, max_len, embed_size) -->
        # (batch_size, max_len, heads, head_dim) -->
        # (batch_size, heads, max_len, head_dim)
        q_new = q_new.view(q_new.shape[0], q_new.shape[1], self.heads, self.head_dim).transpose(1, 2)
        k_new = k_new.view(k_new.shape[0], k_new.shape[1], self.heads, self.head_dim).transpose(1, 2)
        v_new = v_new.view(v_new.shape[0], v_new.shape[1], self.heads, self.head_dim).transpose(1, 2)

        # calculate token-to-token attention scores matrix
        # (batch_size, heads, max_len, head_dim) @
        # (batch_size, heads, head_dim, max_len) -->
        # (batch_size, heads, max_len, max_len)
        attention_scores = (q_new @ k_new.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            # replace all values where mask == 0 with very large negative number -1e9
            # so after applying sigmoid function values will be close to 0
            # we do not want words to watch future words
            # or padding values to interact with other values
            # (we need padding to make sentence to reach the max_len sequence length)
            attention_scores.masked_fill_(mask == 0, -1e9)

        # apply softmax function
        # to normalise attention scores for each token in a row
        attention_scores = F.softmax(attention_scores, dim=-1)
        attention_scores = self.dropout(attention_scores)

        # calculate heads matrix
        # (batch_size, heads, max_len, max_len) @
        # (batch_size, heads, max_len, head_dim) -->
        # (batch_size, heads, max_len, head_dim)
        x = attention_scores @ v_new

        # combine all the heads together
        # (batch_size, heads, max_len, head_dim) -->
        # (batch_size, max_len, heads, head_dim) -->
        # (batch_size, max_len, embed_size)
        x = (x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.heads * self.head_dim))

        # multiply by Wo
        # (batch_size, max_len, embed_size) -->
        # (batch_size, max_len, embed_size)
        x = self.w_o(x)

        return x


class ResidualConnection(nn.Module):
    """Skip-connection."""

    def __init__(self, embed_size: int, dropout: float) -> None:
        super().__init__()
        self.norm = LayerNormalisation(embed_size=embed_size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, prev_layer: nn.Module) -> torch.Tensor:
        # apply residual connection to sublayer
        return x + self.dropout(prev_layer(self.norm(x)))


class EncoderBlock(nn.Module):
    """Encoder block."""

    def __init__(
        self,
        embed_size: int,
        self_attention_block: MultiHeadAttentionBlock,
        feed_forward_block: FeedForwardBlock,
        dropout: float,
    ) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection1 = ResidualConnection(embed_size=embed_size, dropout=dropout)
        self.residual_connection2 = ResidualConnection(embed_size=embed_size, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        x = self.residual_connection1(
            x=x, prev_layer=lambda x: self.self_attention_block(q=x, k=x, v=x, mask=src_mask))
        x = self.residual_connection2(
            x=x, prev_layer=lambda x: self.feed_forward_block(x=x))
        x = self.dropout(x)
        return x


class Encoder(nn.Module):
    """Encoder may consist of N encoder blocks."""

    def __init__(self, embed_size: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalisation(embed_size=embed_size)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x=x, src_mask=mask)
        x = self.norm(x)
        return x


class DecoderBlock(nn.Module):
    """Decoder block."""

    def __init__(
        self,
        embed_size: int,
        self_attention_block: MultiHeadAttentionBlock,
        cross_attention_block: MultiHeadAttentionBlock,
        feed_forward_block: FeedForwardBlock,
        dropout: float,
    ) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection1 = ResidualConnection(embed_size=embed_size, dropout=dropout)
        self.residual_connection2 = ResidualConnection(embed_size=embed_size, dropout=dropout)
        self.residual_connection3 = ResidualConnection(embed_size=embed_size, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        x = self.residual_connection1(
            x=x, prev_layer=lambda x: self.self_attention_block(q=x, k=x, v=x, mask=tgt_mask))
        x = self.residual_connection2(
            x=x, prev_layer=lambda x: self.cross_attention_block(q=x, k=encoder_output, v=encoder_output, mask=src_mask))
        x = self.residual_connection3(
            x=x, prev_layer=lambda x: self.feed_forward_block(x=x))
        x = self.dropout(x)
        return x


class Decoder(nn.Module):
    """Decoder may consist of N decoder blocks."""

    def __init__(self, embed_size: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalisation(embed_size=embed_size)

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x=x, encoder_output=encoder_output, src_mask=src_mask, tgt_mask=tgt_mask)
        x = self.norm(x)
        return x


class ProjectionLayer(nn.Module):
    """Project embedding into vocabulary."""

    def __init__(self, embed_size, vocab_size) -> None:
        super().__init__()
        self.proj = nn.Linear(embed_size, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch, seq_len, embed_size) --> (batch, seq_len, vocab_size)
        x = self.proj(x)
        return x


class Transformer(nn.Module):
    """Put everything together."""

    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        src_embed: InputEmbedding,
        tgt_embed: InputEmbedding,
        src_pos: PositionalEncoding,
        tgt_pos: PositionalEncoding,
        projection_layer: ProjectionLayer,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        # (batch_size, max_len, embed_size)
        src = self.src_embed(src)
        src = self.src_pos(src)
        src = self.encoder(x=src, mask=src_mask)
        return src

    def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        # (batch_size, max_len, embed_size)
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        tgt = self.decoder(x=tgt, encoder_output=encoder_output, src_mask=src_mask, tgt_mask=tgt_mask)
        return tgt

    def project(self, x: torch.Tensor) -> torch.Tensor:
        # (batch_size, max_len, embed_size) -->
        # (batch_size, max_len, vocab_size)
        return self.projection_layer(x)


def build_transformer(
    src_vocab_size: int,
    tgt_vocab_size: int,
    src_seq_len: int,
    tgt_seq_len: int,
    embed_size: int,
    N: int = 6,
    heads: int = 8,
    dropout: float = 0.1,
    hidden_size: int = 2048,
) -> Transformer:
    """
    Given all parameters build a transformer.
    """
    # create the embedding layers
    src_embed = InputEmbedding(embed_size=embed_size, vocab_size=src_vocab_size)
    tgt_embed = InputEmbedding(embed_size=embed_size, vocab_size=tgt_vocab_size)

    # create the positional encoding layers
    src_pos = PositionalEncoding(embed_size=embed_size, max_len=src_seq_len, dropout=dropout)
    tgt_pos = PositionalEncoding(embed_size=embed_size, max_len=tgt_seq_len, dropout=dropout)

    # create the encoder blocks
    encoder_blocks = nn.ModuleList()
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(embed_size=embed_size, heads=heads, dropout=dropout)
        feed_forward_block = FeedForwardBlock(embed_size=embed_size, hidden_size=hidden_size, dropout=dropout)
        encoder_block = EncoderBlock(
            embed_size=embed_size, 
            self_attention_block=encoder_self_attention_block, 
            feed_forward_block=feed_forward_block, 
            dropout=dropout
        )
        encoder_blocks.append(encoder_block)

    # create the decoder blocks
    decoder_blocks = nn.ModuleList()
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(embed_size=embed_size, heads=heads, dropout=dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(embed_size=embed_size, heads=heads, dropout=dropout)
        feed_forward_block = FeedForwardBlock(embed_size=embed_size, hidden_size=hidden_size, dropout=dropout)
        decoder_block = DecoderBlock(
            embed_size=embed_size,
            self_attention_block=decoder_self_attention_block,
            cross_attention_block=decoder_cross_attention_block,
            feed_forward_block=feed_forward_block,
            dropout=dropout
        )
        decoder_blocks.append(decoder_block)

    # create the encoder and decoder
    encoder = Encoder(embed_size=embed_size, layers=encoder_blocks)
    decoder = Decoder(embed_size=embed_size, layers=decoder_blocks)

    # create the projection layer
    projection_layer = ProjectionLayer(embed_size=embed_size, vocab_size=tgt_vocab_size)

    # create the transformer
    transformer = Transformer(
        encoder=encoder,
        decoder=decoder,
        src_embed=src_embed,
        tgt_embed=tgt_embed,
        src_pos=src_pos,
        tgt_pos=tgt_pos,
        projection_layer=projection_layer,
    )

    # initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer
