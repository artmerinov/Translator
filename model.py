import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math


class InputEmbedding(nn.Module):
    """
    Convert input tokens into scaled embeddings.
    """
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
    """
    Inject information about relative or absolute positions of tokens.
    """
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
    """
    Layer normalisation.
    """
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
    """
    Fully connected feed-forward network.
    """
    def __init__(self, embed_size: int, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(embed_size, hidden_size, bias=True)
        self.fc2 = nn.Linear(hidden_size, embed_size, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(self.relu(self.fc1(x))))
    

class ProjectionLayer(nn.Module):
    """
    Projection layer to generate probabilities.
    """
    def __init__(self, embed_size: int, vocab_size: int) -> None:
        super().__init__()
        self.fc = nn.Linear(embed_size, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = F.log_softmax(x, dim=-1)
        return x


class MultiHeadAttentionBlock(nn.Module):
    """
    Multi-head attention.
    """
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


class EncoderBlock(nn.Module):
    """
    Encoder block.
    """
    def __init__(self, embed_size: int,  heads: int, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = MultiHeadAttentionBlock(
            embed_size=embed_size,
            heads=heads,
            dropout=dropout
        )
        self.norm = LayerNormalisation(embed_size=embed_size)
        self.feed_forward_block = FeedForwardBlock(
            embed_size=embed_size, 
            hidden_size=hidden_size, 
            dropout=dropout
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        
        x_norm1 = self.norm(x)
        self_att = self.self_attention_block(q=x_norm1, k=x_norm1, v=x_norm1, mask=src_mask)
        x = x + self.dropout(self_att)

        x_norm2 = self.norm(x)
        ff = self.feed_forward_block(x=x_norm2)
        x = x + self.dropout(ff)
        
        return x


class Encoder(nn.Module):
    """
    Encoder.
    """
    def __init__(self, embed_size: int, dropout: float, heads: int, hidden_size: int, N: int) -> None:
        super().__init__()
        self.encoder_block = EncoderBlock(
            embed_size=embed_size,
            heads=heads,
            hidden_size=hidden_size,
            dropout=dropout
        )
        self.layers = nn.ModuleList(modules=[self.encoder_block for _ in range(N)])
        self.norm = LayerNormalisation(embed_size=embed_size)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x=x, src_mask=mask)
        x = self.norm(x)
        return x


class DecoderBlock(nn.Module):
    """
    Decoder block.
    """
    def __init__(self, embed_size: int, heads: int, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.attention_block = MultiHeadAttentionBlock(
            embed_size=embed_size,
            heads=heads,
            dropout=dropout
        )
        self.norm = LayerNormalisation(embed_size=embed_size)
        self.feed_forward_block = FeedForwardBlock(
            embed_size=embed_size, 
            hidden_size=hidden_size, 
            dropout=dropout
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(
            self, 
            x: torch.Tensor, 
            encoder_output: torch.Tensor, 
            src_mask: torch.Tensor, 
            tgt_mask: torch.Tensor
    ) -> torch.Tensor:

        x_norm1 = self.norm(x)
        self_att = self.attention_block(q=x_norm1, k=x_norm1, v=x_norm1, mask=tgt_mask)
        x = x + self.dropout(self_att)

        x_norm2 = self.norm(x)
        cross_att = self.attention_block(q=x_norm2, k=encoder_output, v=encoder_output, mask=src_mask)
        x = x + self.dropout(cross_att)
    
        x_norm3 = self.norm(x)
        ff = self.feed_forward_block(x=x_norm3)
        x = x + self.dropout(ff)

        return x


class Decoder(nn.Module):
    """
    Decoder.
    """
    def __init__(self, embed_size: int, dropout: float, heads: int, hidden_size: int, N: int) -> None:
        super().__init__()
        self.decoder_block = DecoderBlock(
            embed_size=embed_size,
            heads=heads,
            hidden_size=hidden_size,
            dropout=dropout
        )
        self.layers = nn.ModuleList(modules=[self.decoder_block for _ in range(N)])
        self.norm = LayerNormalisation(embed_size=embed_size)

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x=x, encoder_output=encoder_output, src_mask=src_mask, tgt_mask=tgt_mask)
        x = self.norm(x)
        return x


class Transformer(nn.Module):
    """
    Put everything together.
    """
    def __init__(
            self,
            embed_size: int, 
            src_vocab_size: int,
            tgt_vocab_size: int,
            max_len: int,
            dropout: float,
            heads: int,
            hidden_size: int,
            N: int,
    ) -> None:
        super().__init__()

        self.src_input_embedding = InputEmbedding(
            embed_size=embed_size, 
            vocab_size=src_vocab_size
        )
        self.tgt_input_embedding = InputEmbedding(
            embed_size=embed_size, 
            vocab_size=tgt_vocab_size
        )
        self.pos_encoding = PositionalEncoding(
            embed_size=embed_size, 
            max_len=max_len,
            dropout=dropout
        )
        self.encoder = Encoder(
            embed_size=embed_size,
            dropout=dropout,
            heads=heads,
            hidden_size=hidden_size,
            N=N
        )
        self.decoder = Decoder(
            embed_size=embed_size,
            dropout=dropout,
            heads=heads,
            hidden_size=hidden_size,
            N=N
        )
        self.proj = ProjectionLayer(
            embed_size=embed_size, 
            vocab_size=tgt_vocab_size
        )

        self._reset_parameters()

    def _reset_parameters(self):
        """
        Initiate parameters in the transformer model.
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        x = src
        x = self.src_input_embedding(x=x)
        x = self.pos_encoding(x=x)
        x = self.encoder(x=x, mask=src_mask)
        return x
    
    def decode(self, tgt: torch.Tensor, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        x = tgt
        x = self.tgt_input_embedding(x=x)
        x = self.pos_encoding(x=x)
        x = self.decoder(x=x, encoder_output=encoder_output, src_mask=src_mask, tgt_mask=tgt_mask)
        x = self.proj(x=x)
        return x
    
    def forward(self, src: torch.Tensor, tgt: torch.Tensor, src_mask: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        enc_out = self.encode(src=src, src_mask=src_mask)
        dec_out = self.decode(tgt=tgt, encoder_output=enc_out, src_mask=src_mask, tgt_mask=tgt_mask)
        return dec_out
