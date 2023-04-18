import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import DropPath
from .cnn_layers import ConvNormAct, FeedForwardNetwork
from .rnn_layers import RNNProjection


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

        self.pos_enc = PositionalEncoding(self.in_chan)
        self.attention = nn.MultiheadAttention(self.in_chan, self.n_head, self.dropout, batch_first=True)
        self.norm = nn.LayerNorm(self.in_chan)
        self.drop_path_layer = DropPath(self.dropout)

    def forward(self, x: torch.Tensor):
        x = x.transpose(1, 2)  # B, C, T -> B, T, C
        res = x

        x = self.pos_enc(x)
        x = self.attention(x, x, x)[0]
        x = self.norm(self.drop_path_layer(x) + res)

        x = x.transpose(2, 1)  # B, T, C -> B, C, T
        return x


class ConvMod(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.norm = nn.GroupNorm(1, dim, eps=1e-6)
        self.a = nn.Sequential(nn.Conv2d(dim, dim, 1), nn.GELU(), nn.Conv2d(dim, dim, 11, padding=5, groups=dim))

        self.v = nn.Conv2d(dim, dim, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape
        x = self.norm(x)
        a = self.a(x)  # A=QK^T
        x = a * self.v(x)
        x = self.proj(x)

        return x


class GlobalAttention(nn.Module):
    def __init__(
        self,
        in_chan: int,
        hid_chan: int = None,
        kernel_size: int = 5,
        n_head: int = 8,
        dropout: float = 0.1,
        verbose: bool = False,
        *args,
        **kwargs,
    ):
        super(GlobalAttention, self).__init__()
        self.in_chan = in_chan
        self.hid_chan = hid_chan if hid_chan is not None else 2 * self.in_chan
        self.kernel_size = kernel_size
        self.n_head = n_head
        self.dropout = dropout
        self.verbose = verbose

        self.mhsa = MultiHeadSelfAttention(self.in_chan, self.n_head, self.dropout)
        self.ffn = FeedForwardNetwork(self.in_chan, self.hid_chan, self.kernel_size, dropout=self.dropout)

        if self.verbose:
            self.mhsa_params = sum(p.numel() for p in self.mhsa.parameters() if p.requires_grad) / 1000
            self.ffn_params = sum(p.numel() for p in self.ffn.parameters() if p.requires_grad) / 1000

            s = f"MHSA Params: {self.mhsa_params}\n" f"FFN Params: {self.ffn_params}\n"

            print(s)

    def forward(self, x: torch.Tensor):
        x = self.mhsa(x)
        x = self.ffn(x)
        return x


class GlobalAttention2D(nn.Module):
    def __init__(
        self,
        in_chan: int,
        hid_chan: int = None,
        kernel_size: int = 5,
        n_head: int = 8,
        dropout: float = 0.1,
        verbose: bool = False,
        *args,
        **kwargs,
    ):
        super(GlobalAttention2D, self).__init__()
        self.in_chan = in_chan
        self.hid_chan = hid_chan if hid_chan is not None else 2 * self.in_chan
        self.kernel_size = kernel_size
        self.n_head = n_head
        self.dropout = dropout
        self.verbose = verbose

        self.mhsa_height = MultiHeadSelfAttention(self.in_chan, self.n_head, self.dropout)
        self.mhsa_width = MultiHeadSelfAttention(self.in_chan, self.n_head, self.dropout)

        if self.kernel_size > 0:
            self.ffn_height = FeedForwardNetwork(self.in_chan, self.hid_chan, self.kernel_size, dropout=dropout)
            self.ffn_width = FeedForwardNetwork(self.in_chan, self.hid_chan, self.kernel_size, dropout=dropout)
        else:
            self.ffn_height = RNNProjection(self.in_chan, self.hid_chan, dropout=dropout, bidirectional=True)
            self.ffn_width = RNNProjection(self.in_chan, self.hid_chan, dropout=dropout, bidirectional=True)

        if self.verbose:
            self.mhsa_height_params = sum(p.numel() for p in self.mhsa_height.parameters() if p.requires_grad) / 1000
            self.mhsa_width_params = sum(p.numel() for p in self.mhsa_width.parameters() if p.requires_grad) / 1000
            self.ffn_height_params = sum(p.numel() for p in self.ffn_height.parameters() if p.requires_grad) / 1000
            self.ffn_width_params = sum(p.numel() for p in self.ffn_width.parameters() if p.requires_grad) / 1000

            s = (
                f"MHSA Height: {self.mhsa_height_params}\n"
                f"MHSA Width: {self.mhsa_width_params}\n"
                f"FFN Height: {self.ffn_height_params}\n"
                f"FFN Width: {self.ffn_width_params}\n"
            )

            print(s)

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.size()

        h_input = x.permute(0, 3, 1, 2).contiguous().view(B * W, C, H)
        h_output = self.mhsa_height.forward(h_input)
        h_ffn = self.ffn_height.forward(h_output)
        x = h_ffn.view(B, W, C, H).permute(0, 2, 3, 1).contiguous()

        w_input = x.permute(0, 2, 1, 3).contiguous().view(B * H, C, W)
        w_output = self.mhsa_width.forward(w_input)
        w_ffn = self.ffn_width.forward(w_output)
        x = w_ffn.view(B, H, C, W).permute(0, 2, 1, 3).contiguous()

        return x
