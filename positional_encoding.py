import math
import torch
from torch import nn, Tensor

class SinusoidalPositionalEncoding(nn.Module):

    def __init__(self, d_model: int, scale: bool = True, dropout: float = 0.1, max_len: int = 2048):
        """
        module which adds a (non-trainable) sinusoidal positional encoding to the input tensor

        Parameters
        ----------
        d_model : int
            model dimension.
        scale : bool, optional
            whether to scale added positional encodings to account for scaling in dot product attention,
            by default True
        dropout : float, optional
            dropout rate, by default 0.1
        max_len : int, optional
            maximum length to consider, by default 2048
        """

        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.scale = math.sqrt(d_model) if scale else 1

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        pe = pe[:, 0, :]
        self.register_buffer('pe', pe)


    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """

        x = self.scale * x + self.pe[:x.size(1)]
        return self.dropout(x)

class LearnedPositionalEmbeddings(nn.Module):

    def __init__(self, d_model: int, scale: bool = True, dropout : float = 0.1, max_len: int = 2048):
        """module which adds a learnable positionall embedding to the input tensor.

        Parameters
        ----------
        d_model : int
            model dimension.
        scale : bool, optional
            whether to scale added positional encodings to account for scaling in dot product attention,
            by default True
        dropout : float, optional
            dropout rate, by default 0.1
        max_len : int, optional
            maximum length to consider, by default 2048
        """

        super().__init__()
        self.scale = math.sqrt(d_model) if scale else 1

        self.position_embeddings = nn.Embedding(max_len, d_model)
        self.max_len = max_len
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """

        positions = torch.arange(0, x.size(1), dtype=torch.long, device=x.device)
        positional_embedding = self.position_embeddings(positions)
        x = self.scale * x + positional_embedding
        return self.dropout(x)