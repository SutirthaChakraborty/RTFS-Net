import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import activations


class GridNetBlock(nn.Module):
    def __init__(
        self,
        in_chan: int,
        kernel_size: int,
        stride: int,
        n_freqs: int,
        hid_chan: int,
        n_head: int = 4,
        approx_qk_dim: int = 512,
        act_type: str = "PReLU",
        eps=1e-5,
    ):
        super().__init__()

        in_channels = in_chan * kernel_size

        self.intra_norm = LayerNormalization4D(in_chan, eps=eps)
        self.intra_rnn = nn.LSTM(in_channels, hid_chan, 1, batch_first=True, bidirectional=True)
        self.intra_linear = nn.ConvTranspose1d(hid_chan * 2, in_chan, kernel_size, stride=stride)

        self.inter_norm = LayerNormalization4D(in_chan, eps=eps)
        self.inter_rnn = nn.LSTM(in_channels, hid_chan, 1, batch_first=True, bidirectional=True)
        self.inter_linear = nn.ConvTranspose1d(hid_chan * 2, in_chan, kernel_size, stride=stride)

        E = math.ceil(approx_qk_dim * 1.0 / n_freqs)  # approx_qk_dim is only approximate
        assert in_chan % n_head == 0
        for ii in range(n_head):
            self.add_module(
                "attn_conv_Q_%d" % ii,
                nn.Sequential(
                    nn.Conv2d(in_chan, E, 1),
                    activations.get(act_type)(),
                    LayerNormalization4DCF((E, n_freqs), eps=eps),
                ),
            )
            self.add_module(
                "attn_conv_K_%d" % ii,
                nn.Sequential(
                    nn.Conv2d(in_chan, E, 1),
                    activations.get(act_type)(),
                    LayerNormalization4DCF((E, n_freqs), eps=eps),
                ),
            )
            self.add_module(
                "attn_conv_V_%d" % ii,
                nn.Sequential(
                    nn.Conv2d(in_chan, in_chan // n_head, 1),
                    activations.get(act_type)(),
                    LayerNormalization4DCF((in_chan // n_head, n_freqs), eps=eps),
                ),
            )
        self.add_module(
            "attn_concat_proj",
            nn.Sequential(
                nn.Conv2d(in_chan, in_chan, 1),
                activations.get(act_type)(),
                LayerNormalization4DCF((in_chan, n_freqs), eps=eps),
            ),
        )

        self.emb_dim = in_chan
        self.emb_ks = kernel_size
        self.emb_hs = stride
        self.n_head = n_head

    def forward(self, x: torch.Tensor):
        batch_size, n_chan, old_time, old_freq = x.shape
        time_dim = math.ceil((old_time - self.emb_ks) / self.emb_hs) * self.emb_hs + self.emb_ks
        freq_dim = math.ceil((old_freq - self.emb_ks) / self.emb_hs) * self.emb_hs + self.emb_ks
        x = F.pad(x, (0, freq_dim - old_freq, 0, time_dim - old_time))

        # intra RNN
        residual = x
        intra_rnn = self.intra_norm(residual)  # [B, C, T, Q]
        intra_rnn = intra_rnn.transpose(1, 2).contiguous().view(batch_size * time_dim, n_chan, freq_dim)  # [BT, C, Q]
        intra_rnn = F.unfold(intra_rnn[..., None], (self.emb_ks, 1), stride=(self.emb_hs, 1))  # [BT, C*emb_ks, -1]
        intra_rnn = intra_rnn.transpose(1, 2)  # [BT, -1, C*emb_ks]
        intra_rnn = self.intra_rnn(intra_rnn)[0]  # [BT, -1, H]
        intra_rnn = intra_rnn.transpose(1, 2)  # [BT, H, -1]
        intra_rnn = self.intra_linear(intra_rnn)  # [BT, C, Q]
        intra_rnn = intra_rnn.view([batch_size, time_dim, n_chan, freq_dim])
        intra_rnn = intra_rnn.transpose(1, 2).contiguous()  # [B, C, T, Q]
        intra_rnn = intra_rnn + residual  # [B, C, T, Q]

        # inter RNN
        residual = intra_rnn
        inter_rnn = self.inter_norm(residual)  # [B, C, T, Q]
        inter_rnn = inter_rnn.permute(0, 3, 1, 2).contiguous().view(batch_size * freq_dim, n_chan, time_dim)  # [BQ, C, T]
        inter_rnn = F.unfold(inter_rnn[..., None], (self.emb_ks, 1), stride=(self.emb_hs, 1))  # [BQ, C*emb_ks, -1]
        inter_rnn = inter_rnn.transpose(1, 2)  # [BQ, -1, C*emb_ks]
        inter_rnn = self.inter_rnn(inter_rnn)[0]  # [BQ, -1, H]
        inter_rnn = inter_rnn.transpose(1, 2)  # [BQ, H, -1]
        inter_rnn = self.inter_linear(inter_rnn)  # [BQ, C, T]
        inter_rnn = inter_rnn.view([batch_size, freq_dim, n_chan, time_dim])
        inter_rnn = inter_rnn.permute(0, 2, 3, 1).contiguous()  # [B, C, T, Q]
        inter_rnn = inter_rnn + residual  # [B, C, T, Q]

        # attention
        inter_rnn = inter_rnn[..., :old_time, :old_freq]
        batch = inter_rnn

        all_Q, all_K, all_V = [], [], []
        for ii in range(self.n_head):
            all_Q.append(self["attn_conv_Q_%d" % ii](batch))  # [B, C, T, Q]
            all_K.append(self["attn_conv_K_%d" % ii](batch))  # [B, C, T, Q]
            all_V.append(self["attn_conv_V_%d" % ii](batch))  # [B, C, T, Q]

        freq_dim = torch.cat(all_Q, dim=0)  # [B', C, T, Q]
        K = torch.cat(all_K, dim=0)  # [B', C, T, Q]
        V = torch.cat(all_V, dim=0)  # [B', C, T, Q]

        freq_dim = freq_dim.transpose(1, 2)
        freq_dim = freq_dim.flatten(start_dim=2)  # [B', T, C*Q]
        K = K.transpose(1, 2)
        K = K.flatten(start_dim=2)  # [B', T, C*Q]
        V = V.transpose(1, 2)  # [B', T, C, Q]
        old_shape = V.shape
        V = V.flatten(start_dim=2)  # [B', T, C*Q]
        emb_dim = freq_dim.shape[-1]

        attn_mat = torch.matmul(freq_dim, K.transpose(1, 2)) / (emb_dim**0.5)  # [B', T, T]
        attn_mat = F.softmax(attn_mat, dim=2)  # [B', T, T]
        V = torch.matmul(attn_mat, V)  # [B', T, C*Q]

        V = V.reshape(old_shape)  # [B', T, C, Q]
        V = V.transpose(1, 2)  # [B', C, T, Q]
        emb_dim = V.shape[1]

        batch = V.view([self.n_head, batch_size, emb_dim, old_time, -1])  # [n_head, B, C, T, Q])
        batch = batch.transpose(0, 1)  # [B, n_head, C, T, Q])
        batch = batch.contiguous().view([batch_size, self.n_head * emb_dim, old_time, -1])  # [B, C, T, Q])
        batch = self["attn_concat_proj"](batch)  # [B, C, T, Q])

        out = batch + inter_rnn
        return out
