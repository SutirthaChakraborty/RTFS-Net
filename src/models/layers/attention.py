import math
import torch
import torch.nn as nn
import torch.nn.functional as functional

from . import cnn_layers
from timm.models.layers import DropPath


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


class MultiHeadSelfAttention2D(nn.Module):
    def __init__(
        self,
        in_chan: int,
        n_freqs: int,
        n_head: int = 4,
        hid_chan: int = 4,
        act_type: str = "PReLU",
        norm_type: str = "LayerNormalization4D",
    ):
        super(MultiHeadSelfAttention2D, self).__init__()
        self.in_chan = in_chan
        self.n_freqs = n_freqs
        self.n_head = n_head
        self.hid_chan = hid_chan
        self.act_type = act_type
        self.norm_type = norm_type

        assert self.in_chan % self.n_head == 0

        self.Queries = nn.ModuleList()
        self.Keys = nn.ModuleList()
        self.Values = nn.ModuleList()

        for _ in range(self.n_head):
            self.Queries.append(
                cnn_layers.ConvActNorm(
                    in_chan=self.in_chan,
                    out_chan=self.hid_chan,
                    kernel_size=1,
                    act_type=self.act_type,
                    norm_type=self.norm_type,
                    n_freqs=self.n_freqs,
                    is2d=True,
                )
            )
            self.Keys.append(
                cnn_layers.ConvActNorm(
                    in_chan=self.in_chan,
                    out_chan=self.hid_chan,
                    kernel_size=1,
                    act_type=self.act_type,
                    norm_type=self.norm_type,
                    n_freqs=self.n_freqs,
                    is2d=True,
                )
            )
            self.Values.append(
                cnn_layers.ConvActNorm(
                    in_chan=self.in_chan,
                    out_chan=self.in_chan // self.n_head,
                    kernel_size=1,
                    act_type=self.act_type,
                    norm_type=self.norm_type,
                    n_freqs=self.n_freqs,
                    is2d=True,
                )
            )

        self.attn_concat_proj = cnn_layers.ConvActNorm(
            in_chan=self.in_chan,
            out_chan=self.in_chan,
            kernel_size=1,
            act_type=self.act_type,
            norm_type=self.norm_type,
            n_freqs=self.n_freqs,
            is2d=True,
        )

    def forward(self, x: torch.Tensor):
        batch_size, _, time, freq = x.size()
        residual = x

        all_Q, all_K, all_V = [], [], []
        for ii in range(self.n_head):
            all_Q.append(self.Queries[ii](x))  # [B, E, T, F]
            all_K.append(self.Keys[ii](x))  # [B, E, T, F]
            all_V.append(self.Values[ii](x))  # [B, C/n_head, T, F]

        Q = torch.cat(all_Q, dim=0)  # [B', E, T, F]    B' = B*n_head
        K = torch.cat(all_K, dim=0)  # [B', E, T, F]
        V = torch.cat(all_V, dim=0)  # [B', C/n_head, T, F]

        Q = Q.transpose(1, 2).flatten(start_dim=2)  # [B', T, E*F]
        K = K.transpose(1, 2).flatten(start_dim=2)  # [B', T, E*F]
        V = V.transpose(1, 2)  # [B', T, C/n_head, F]
        old_shape = V.shape
        V = V.flatten(start_dim=2)  # [B', T, C*F/n_head]
        emb_dim = Q.shape[-1]  # C*F/n_head

        attn_mat = torch.matmul(Q, K.transpose(1, 2)) / (emb_dim**0.5)  # [B', T, T]
        attn_mat = functional.softmax(attn_mat, dim=2)  # [B', T, T]
        V = torch.matmul(attn_mat, V)  # [B', T, C*F/n_head]

        V = V.reshape(old_shape)  # [B', T, C/n_head, F]
        V = V.transpose(1, 2)  # [B', C/n_head, T, F]
        emb_dim = V.shape[1]  # C/n_head

        x = V.view([self.n_head, batch_size, emb_dim, time, freq])  # [n_head, B, C/n_head, T, F])
        x = x.transpose(0, 1).contiguous()  # [B, n_head, C/n_head, T, F])
        x = x.view([batch_size, self.n_head * emb_dim, time, freq])  # [B, C, T, F])
        x = self.attn_concat_proj(x)  # [B, C, T, F])
        
        x = x + residual

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


class GlobalAttentionRNN(nn.Module):
    def __init__(
        self,
        in_chan: int,
        hid_chan: int = None,
        dropout: float = 0.1,
        rnn_type: str = "LSTM",
        bidirectional: bool = True,
        *args,
        **kwargs,
    ):
        super(GlobalAttentionRNN, self).__init__()
        self.in_chan = in_chan
        self.hid_chan = hid_chan if hid_chan is not None else self.in_chan
        self.dropout = dropout
        self.rnn_type = rnn_type
        self.bidirectional = bidirectional

        self.RNN = cnn_layers.RNNProjection(self.in_chan, self.hid_chan, self.rnn_type, self.dropout, self.bidirectional)

    def forward(self, x: torch.Tensor):
        x = self.RNN(x)
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
        rnn_type: str = "LSTM",
        bidirectional: bool = True,
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
        self.rnn_type = rnn_type
        self.bidirectional = bidirectional

        self.time_RNN = cnn_layers.RNNProjection(self.in_chan, self.in_chan, self.rnn_type, self.dropout, self.bidirectional)
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
