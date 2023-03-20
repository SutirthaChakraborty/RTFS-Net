import torch
import torch.nn as nn
from ..utils import split_feature, merge_feature
from ..layers import TAC, GlobalAttention, ConvolutionalRNN, RNNProjection


class GC_RNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, gc3_params: dict):
        super(GC_RNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn_type = gc3_params.get("rnn_type", "LSTM")
        self.group_size = gc3_params.get("group_size", 1)
        self.num_layers = gc3_params.get("num_layers", 2)
        self.tac_multiplier = gc3_params.get("tac_multiplier", 2)

        self.TAC = nn.ModuleList([])
        self.rnn = nn.ModuleList([])
        self.LN = nn.ModuleList([])

        for _ in range(self.num_layers):
            self.TAC.append(
                TAC(
                    input_size=self.input_size // self.group_size,
                    hidden_size=self.hidden_size * self.tac_multiplier // self.group_size,
                )
            )
            if self.rnn_type == "GlobalAttention":
                self.rnn.append(
                    GlobalAttention(
                        in_chan=self.input_size // self.group_size,
                        hid_chan=self.hidden_size // self.group_size,
                        **gc3_params,
                    )
                )
            elif self.rnn_type == "ConvolutionalRNN":
                self.rnn.append(
                    ConvolutionalRNN(
                        in_chan=self.input_size // self.group_size,
                        hid_chan=self.hidden_size // self.group_size,
                        **gc3_params,
                    )
                )
            else:
                self.rnn.append(
                    RNNProjection(
                        input_size=self.input_size // self.group_size,
                        hidden_size=self.hidden_size // self.group_size,
                        **gc3_params,
                    )
                )
            self.LN.append(nn.GroupNorm(num_groups=1, num_channels=self.input_size // self.group_size))

    def forward(self, x: torch.Tensor):
        batch_size, dim, seq_len = x.shape
        x = x.view(batch_size, self.group_size, -1, seq_len)

        for i in range(self.num_layers):
            x = self.TAC[i](x)
            res = x
            x = self.rnn[i](x)
            x = self.LN[i](x.view(batch_size * self.group_size, -1, seq_len)).view(batch_size, self.group_size, -1, seq_len)
            x = res + x

        x = x.view(batch_size, dim, seq_len)

        return x


class ContextEncoder(nn.Module):
    def __init__(
        self,
        in_chan: int,
        hid_chan: int,
        gc3_params: dict = dict(),
    ):
        super(ContextEncoder, self).__init__()

        self.in_chan = in_chan
        self.hid_chan = hid_chan
        self.gc3_params = gc3_params

        self.context_size = gc3_params.get("context_size", 1)

        if self.context_size > 1:
            self.group_context_rnn = GC_RNN(input_size=self.in_chan, hidden_size=self.hid_chan, gc3_params=self.gc3_params)

    def forward(self, x: torch.Tensor):
        if self.context_size > 1:
            squeeze_block, squeeze_rest = split_feature(x, self.context_size)  # B, N, context, L
            batch_size, bn_dim, _, seq_len = squeeze_block.shape
            squeeze_input = squeeze_block.permute(0, 3, 1, 2).contiguous().view(batch_size * seq_len, bn_dim, self.context_size)
            squeeze_output = self.group_context_rnn(squeeze_input)  # B*L, N, context
            squeeze_mean = squeeze_output.mean(2).view(batch_size, seq_len, bn_dim).transpose(1, 2).contiguous()  # B, N, L

            return squeeze_mean, squeeze_block, squeeze_rest
        else:
            return x, 0, 0


class ContextDecoder(nn.Module):
    def __init__(
        self,
        in_chan: int,
        hid_chan: int,
        gc3_params: dict = dict(),
    ):
        super().__init__()

        self.in_chan = in_chan
        self.hid_chan = hid_chan
        self.gc3_params = gc3_params

        self.context_size = gc3_params.get("context_size", 1)

        if self.context_size > 1:
            self.context_dec = GC_RNN(input_size=self.in_chan, hidden_size=self.hid_chan, gc3_params=self.gc3_params)

    def forward(self, x: torch.Tensor, res: torch.Tensor, squeeze_rest: torch.Tensor):
        if self.context_size > 1:
            x = x.unsqueeze(2) + res  # B, N, context, L
            batch_size, bn_dim, _, seq_len = x.shape
            x = x.permute(0, 3, 1, 2).contiguous().view(batch_size * seq_len, bn_dim, self.context_size)  # B*L, N, context
            unsqueeze_output = self.context_dec(x).view(batch_size, seq_len, bn_dim, self.context_size)  # B, L, N, context
            unsqueeze_output = unsqueeze_output.permute(0, 2, 3, 1).contiguous()  # B, N, context, L
            unsqueeze_output = merge_feature(unsqueeze_output, squeeze_rest)  # B, N, T

            return unsqueeze_output
        else:
            return x
