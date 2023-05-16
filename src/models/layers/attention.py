import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import DropPath
from . import cnn_layers


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
        positional_encoding: bool = True,
    ):
        super(MultiHeadSelfAttention, self).__init__()
        self.in_chan = in_chan
        self.n_head = n_head
        self.dropout = dropout
        self.positional_encoding = positional_encoding

        assert self.in_chan % self.n_head == 0, "In channels: {} must be divisible by the number of heads: {}".format(
            self.in_chan, self.n_head
        )

        self.norm1 = nn.LayerNorm(self.in_chan)
        self.pos_enc = PositionalEncoding(self.in_chan) if self.positional_encoding else nn.Identity()
        self.attention = nn.MultiheadAttention(self.in_chan, self.n_head, self.dropout, batch_first=True)
        self.dropout_layer = nn.Dropout(self.dropout)
        self.norm2 = nn.LayerNorm(self.in_chan)
        self.drop_path_layer = DropPath(self.dropout)

    def forward(self, x: torch.Tensor):
        res = x
        x = x.transpose(1, 2)  # B, C, T -> B, T, C

        x = self.norm1(x)
        x = self.pos_enc(x)
        residual = x
        x = self.attention(x, x, x)[0]
        x = self.dropout_layer(x) + residual
        x = self.norm2(x)

        x = x.transpose(2, 1)  # B, T, C -> B, C, T
        x = self.drop_path_layer(x) + res
        return x


class GlobalAttention(nn.Module):
    def __init__(
        self,
        in_chan: int,
        hid_chan: int = None,
        ffn_name: str = "FeedForwardNetwork",
        kernel_size: int = 5,
        n_head: int = 8,
        dropout: float = 0.1,
        pos_enc: bool = True,
        *args,
        **kwargs,
    ):
        super(GlobalAttention, self).__init__()
        self.in_chan = in_chan
        self.hid_chan = hid_chan if hid_chan is not None else 2 * self.in_chan
        self.ffn_name = ffn_name
        self.kernel_size = kernel_size
        self.n_head = n_head
        self.dropout = dropout
        self.pos_enc = pos_enc

        self.MHSA = MultiHeadSelfAttention(self.in_chan, self.n_head, self.dropout, self.pos_enc)
        self.FFN = cnn_layers.get(self.ffn_name)(self.in_chan, self.hid_chan, self.kernel_size, dropout=self.dropout)

    def forward(self, x: torch.Tensor):
        x = self.MHSA(x)
        x = self.FFN(x)
        return x


class GlobalAttention2D(nn.Module):
    def __init__(
        self,
        in_chan: int,
        hid_chan: int = None,
        ffn_name: str = "FeedForwardNetwork",
        kernel_size: int = 5,
        n_head: int = 8,
        dropout: float = 0.1,
        single_ffn: bool = True,
        group_ffn: bool = False,
        pos_enc: bool = True,
        *args,
        **kwargs,
    ):
        super(GlobalAttention2D, self).__init__()
        self.in_chan = in_chan
        self.hid_chan = hid_chan if hid_chan is not None else 2 * self.in_chan
        self.ffn_name = ffn_name
        self.kernel_size = kernel_size
        self.n_head = n_head
        self.dropout = dropout
        self.single_ffn = single_ffn
        self.group_ffn = group_ffn
        self.pos_enc = pos_enc

        self.time_MHSA = MultiHeadSelfAttention(self.in_chan, self.n_head, self.dropout, self.pos_enc)
        self.freq_MHSA = MultiHeadSelfAttention(self.in_chan, self.n_head, self.dropout, self.pos_enc)

        self.group_FFN = nn.Identity()
        self.time_FFN = nn.Identity()
        self.freq_FFN = nn.Identity()

        if self.single_ffn:
            self.time_FFN = cnn_layers.get(self.ffn_name)(self.in_chan, self.hid_chan, self.kernel_size, dropout=dropout)
            self.freq_FFN = cnn_layers.get(self.ffn_name)(self.in_chan, self.hid_chan, self.kernel_size, dropout=dropout)

        if self.group_ffn:
            self.group_FFN = cnn_layers.FeedForwardNetwork(self.in_chan, self.hid_chan, self.kernel_size, dropout=dropout, is2d=True)

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.size()

        x = x.permute(0, 3, 1, 2).contiguous().view(B * W, C, H)
        x = self.time_MHSA.forward(x)
        x = self.time_FFN.forward(x)
        x = x.view(B, W, C, H).permute(0, 2, 3, 1).contiguous()

        x = self.group_FFN(x)

        x = x.permute(0, 2, 1, 3).contiguous().view(B * H, C, W)
        x = self.freq_MHSA.forward(x)
        x = self.freq_FFN.forward(x)
        x = x.view(B, H, C, W).permute(0, 2, 1, 3).contiguous()

        x = self.group_FFN(x)

        return x


class GlobalGALR(nn.Module):
    def __init__(
        self,
        in_chan: int,
        hid_chan: int = None,
        ffn_name: str = "FeedForwardNetwork",
        kernel_size: int = 5,
        n_head: int = 8,
        dropout: float = 0.1,
        group_ffn: bool = False,
        pos_enc: bool = True,
        *args,
        **kwargs,
    ):
        super(GlobalGALR, self).__init__()
        self.in_chan = in_chan
        self.hid_chan = hid_chan if hid_chan is not None else 2 * self.in_chan
        self.ffn_name = ffn_name
        self.kernel_size = kernel_size
        self.n_head = n_head
        self.dropout = dropout
        self.group_ffn = group_ffn
        self.pos_enc = pos_enc

        self.time_RNN = cnn_layers.RNNProjection(self.in_chan, self.in_chan, dropout=self.dropout)
        self.freq_MHSA = MultiHeadSelfAttention(self.in_chan, self.n_head, self.dropout, self.pos_enc)
        self.freq_FFN = cnn_layers.get(ffn_name)(self.in_chan, self.hid_chan, self.kernel_size, dropout=dropout)

        self.group_FFN = nn.Identity()
        if self.group_ffn:
            self.group_FFN = cnn_layers.FeedForwardNetwork(self.in_chan, self.hid_chan, self.kernel_size, dropout=dropout, is2d=True)

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.size()

        x = x.permute(0, 3, 1, 2).contiguous().view(B * W, C, H)
        x = self.time_RNN.forward(x)
        x = x.view(B, W, C, H).permute(0, 2, 3, 1).contiguous()

        x = x.permute(0, 2, 1, 3).contiguous().view(B * H, C, W)
        x = self.freq_MHSA.forward(x)
        x = self.freq_FFN.forward(x)
        x = x.view(B, H, C, W).permute(0, 2, 1, 3).contiguous()

        x = self.group_FFN(x)

        return x
