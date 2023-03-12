import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import normalizations
from .cnn_layers import ConvNormAct


class GridNetBlock(nn.Module):
    def __init__(
        self,
        in_chan: int,
        hid_chan: int,
        kernel_size: int,
        stride: int,
        n_freqs: int,
        norm_type: str = "LN4d",
        act_type: str = "PReLU",
        n_head: int = 4,
        approx_qk_dim: int = 512,
        eps=1e-5,
    ):
        super(GridNetBlock, self).__init__()

        in_channels = in_chan * kernel_size

        self.intra_norm = normalizations.get(norm_type)((in_chan, 1), eps=eps)
        self.intra_rnn = nn.LSTM(in_channels, hid_chan, 1, batch_first=True, bidirectional=True)
        self.intra_linear = nn.ConvTranspose1d(hid_chan * 2, in_chan, kernel_size, stride=stride)

        self.inter_norm = normalizations.get(norm_type)((in_chan, 1), eps=eps)
        self.inter_rnn = nn.LSTM(in_channels, hid_chan, 1, batch_first=True, bidirectional=True)
        self.inter_linear = nn.ConvTranspose1d(hid_chan * 2, in_chan, kernel_size, stride=stride)

        E = math.ceil(approx_qk_dim * 1.0 / n_freqs)  # approx_qk_dim is only approximate
        assert in_chan % n_head == 0

        for ii in range(n_head):
            self.add_module(
                "attn_conv_Q_%d" % ii,
                ConvNormAct(in_chan, E, 1, act_type=act_type, norm_type=norm_type, eps=eps, n_freqs=n_freqs, is2d=True),
            )
            self.add_module(
                "attn_conv_K_%d" % ii,
                ConvNormAct(in_chan, E, 1, act_type=act_type, norm_type=norm_type, eps=eps, n_freqs=n_freqs, is2d=True),
            )
            self.add_module(
                "attn_conv_V_%d" % ii,
                ConvNormAct(in_chan, in_chan // n_head, 1, act_type=act_type, norm_type=norm_type, eps=eps, n_freqs=n_freqs, is2d=True),
            )

        self.attn_concat_proj = ConvNormAct(
            in_chan, in_chan, 1, act_type=act_type, norm_type=norm_type, eps=eps, n_freqs=n_freqs, is2d=True
        )

        self.emb_dim = in_chan
        self.emb_ks = kernel_size
        self.emb_hs = stride
        self.n_head = n_head

    def forward(self, x: torch.Tensor):
        B, C, old_T, old_Q = x.shape
        T = math.ceil((old_T - self.emb_ks) / self.emb_hs) * self.emb_hs + self.emb_ks
        Q = math.ceil((old_Q - self.emb_ks) / self.emb_hs) * self.emb_hs + self.emb_ks
        x = F.pad(x, (0, Q - old_Q, 0, T - old_T))

        # intra RNN
        residual = x
        intra_rnn = self.intra_norm(residual)  # [B, C, T, Q]
        intra_rnn = intra_rnn.transpose(1, 2).contiguous().view(B * T, C, Q)  # [BT, C, Q]
        intra_rnn = F.unfold(intra_rnn[..., None], (self.emb_ks, 1), stride=(self.emb_hs, 1))  # [BT, C*emb_ks, -1]
        intra_rnn = intra_rnn.transpose(1, 2)  # [BT, -1, C*emb_ks]
        intra_rnn = self.intra_rnn(intra_rnn)[0]  # [BT, -1, H]
        intra_rnn = intra_rnn.transpose(1, 2)  # [BT, H, -1]
        intra_rnn = self.intra_linear(intra_rnn)  # [BT, C, Q]
        intra_rnn = intra_rnn.view([B, T, C, Q])
        intra_rnn = intra_rnn.transpose(1, 2).contiguous()  # [B, C, T, Q]
        intra_rnn = intra_rnn + residual  # [B, C, T, Q]

        # inter RNN
        residual = intra_rnn
        inter_rnn = self.inter_norm(residual)  # [B, C, T, F]
        inter_rnn = inter_rnn.permute(0, 3, 1, 2).contiguous().view(B * Q, C, T)  # [BF, C, T]
        inter_rnn = F.unfold(inter_rnn[..., None], (self.emb_ks, 1), stride=(self.emb_hs, 1))  # [BF, C*emb_ks, -1]
        inter_rnn = inter_rnn.transpose(1, 2)  # [BF, -1, C*emb_ks]
        inter_rnn = self.inter_rnn(inter_rnn)[0]  # [BF, -1, H]
        inter_rnn = inter_rnn.transpose(1, 2)  # [BF, H, -1]
        inter_rnn = self.inter_linear(inter_rnn)  # [BF, C, T]
        inter_rnn = inter_rnn.view([B, Q, C, T])
        inter_rnn = inter_rnn.permute(0, 2, 3, 1).contiguous()  # [B, C, T, Q]
        inter_rnn = inter_rnn + residual  # [B, C, T, Q]

        # attention
        inter_rnn = inter_rnn[..., :old_T, :old_Q]
        batch = inter_rnn

        all_Q, all_K, all_V = [], [], []
        for ii in range(self.n_head):
            all_Q.append(self["attn_conv_Q_%d" % ii](batch))  # [B, C, T, Q]
            all_K.append(self["attn_conv_K_%d" % ii](batch))  # [B, C, T, Q]
            all_V.append(self["attn_conv_V_%d" % ii](batch))  # [B, C, T, Q]

        Q = torch.cat(all_Q, dim=0)  # [B', C, T, Q]
        K = torch.cat(all_K, dim=0)  # [B', C, T, Q]
        V = torch.cat(all_V, dim=0)  # [B', C, T, Q]

        Q = Q.transpose(1, 2)
        Q = Q.flatten(start_dim=2)  # [B', T, C*Q]
        K = K.transpose(1, 2)
        K = K.flatten(start_dim=2)  # [B', T, C*Q]
        V = V.transpose(1, 2)  # [B', T, C, Q]
        old_shape = V.shape
        V = V.flatten(start_dim=2)  # [B', T, C*Q]
        emb_dim = Q.shape[-1]

        attn_mat = torch.matmul(Q, K.transpose(1, 2)) / (emb_dim**0.5)  # [B', T, T]
        attn_mat = F.softmax(attn_mat, dim=2)  # [B', T, T]
        V = torch.matmul(attn_mat, V)  # [B', T, C*Q]

        V = V.reshape(old_shape)  # [B', T, C, Q]
        V = V.transpose(1, 2)  # [B', C, T, Q]
        emb_dim = V.shape[1]

        batch = V.view([self.n_head, B, emb_dim, old_T, -1])  # [n_head, B, C, T, Q])
        batch = batch.transpose(0, 1)  # [B, n_head, C, T, Q])
        batch = batch.contiguous().view([B, self.n_head * emb_dim, old_T, -1])  # [B, C, T, Q])
        batch = self.attn_concat_proj(batch)  # [B, C, T, Q])

        out = batch + inter_rnn
        return out
