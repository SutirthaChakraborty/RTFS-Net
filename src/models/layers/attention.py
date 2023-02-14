import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import DropPath
from .cnn_layers import ConvNormAct


class PositionalEncoding(nn.Module):
    def __init__(
        self,
        channels: int,
        max_len: int = 10000,
    ):
        super(PositionalEncoding, self).__init__()
        self.channels = channels
        self.max_len = max_len

        pe = torch.zeros(self.max_len, self.channels, requires_grad=False)
        position = torch.arange(0, self.max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, self.channels, 2).float() * -(math.log(10000.0) / self.channels))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return x


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        in_chan: int,
        n_head: int = 8,
        dropout: int = 0.1,
    ):
        super(MultiHeadSelfAttention, self).__init__()
        self.in_chan = in_chan
        self.n_head = n_head
        self.dropout = dropout

        assert self.in_chan % self.n_head == 0, "In channels: {} must be divisible by the number of heads: {}".format(
            self.in_chan, self.n_head
        )

        self.norm1 = nn.LayerNorm(self.in_chan)
        self.pos_enc = PositionalEncoding(self.in_chan)
        self.attention = nn.MultiheadAttention(self.in_chan, self.n_head, self.dropout)
        self.dropout_layer = nn.Dropout(self.dropout)
        self.norm2 = nn.LayerNorm(self.in_chan)

    def forward(self, x):
        x = x.transpose(1, 2)

        x = self.norm1(x)
        x = self.pos_enc(x)
        residual = x
        x = self.attention(x, x, x)[0]
        x = self.dropout_layer(x) + residual
        x = self.norm2(x)

        return x.transpose(2, 1)


class FeedForwardNetwork(nn.Module):
    def __init__(
        self,
        in_chan: int,
        hid_chan: int,
        kernel_size: int = 5,
        dropout: float = 0.1,
    ):
        super(FeedForwardNetwork, self).__init__()
        self.in_chan = in_chan
        self.hid_chan = hid_chan
        self.kernel_size = kernel_size
        self.dropout = dropout

        self.encoder = ConvNormAct(self.in_chan, self.hid_chan, 1, norm_type="gLN", bias=False)
        self.refiner = ConvNormAct(
            self.hid_chan, self.hid_chan, self.kernel_size, groups=self.hid_chan, padding=((self.kernel_size - 1) // 2), act_type="ReLU"
        )
        self.decoder = ConvNormAct(self.hid_chan, self.in_chan, 1, norm_type="gLN", bias=False)
        self.dropout_layer = nn.Dropout(self.dropout)

    def forward(self, x):
        x = self.encoder(x)
        x = self.refiner(x)
        x = self.dropout_layer(x)
        x = self.decoder(x)
        x = self.dropout_layer(x)
        return x


class GlobalAttention(nn.Module):
    def __init__(
        self,
        in_chan: int,
        n_head: int = 8,
        kernel_size: int = 5,
        dropout: float = 0.1,
        drop_path: float = 0.1,
    ):
        super(GlobalAttention, self).__init__()
        self.in_chan = in_chan
        self.n_head = n_head
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.drop_path = drop_path

        self.mhsa = MultiHeadSelfAttention(self.in_chan, self.n_head, self.dropout)
        self.ffn = FeedForwardNetwork(self.in_chan, self.in_chan * 2, self.kernel_size, self.dropout)
        self.drop_path_layer = DropPath(self.drop_path) if self.drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path_layer(self.mhsa(x))
        x = x + self.drop_path_layer(self.ffn(x))
        return x
