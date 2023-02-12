import torch.nn as nn
from ..utils import split_feature, merge_feature
from ..layers import GC_RNN


class ContextEncoder(nn.Module):
    def __init__(
        self,
        in_chan: int,
        hid_chan: int,
        rnn_type: str = "LSTM",
        num_group: int = 2,
        context_size: int = 2,
        dropout: float = 0,
        num_layers: int = 1,
        bidirectional: bool = True,
    ):
        super(ContextEncoder, self).__init__()

        self.in_chan = in_chan
        self.hid_chan = hid_chan
        self.rnn_type = rnn_type
        self.num_group = num_group
        self.context_size = context_size
        self.dropout = dropout
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.group_context_rnn = GC_RNN(
            input_size=self.in_chan,
            hidden_size=self.hid_chan,
            rnn_type=self.rnn_type,
            num_group=self.num_group,
            dropout=self.dropout,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional,
        )

    def forward(self, x):
        squeeze_block, squeeze_rest = split_feature(x, self.context_size)  # B, N, context, L
        batch_size, bn_dim, _, seq_len = squeeze_block.shape
        squeeze_input = squeeze_block.permute(0, 3, 1, 2).contiguous().view(batch_size * seq_len, bn_dim, self.context_size)
        squeeze_output = self.group_context_rnn(squeeze_input)  # B*L, N, context
        squeeze_mean = squeeze_output.mean(2).view(batch_size, seq_len, bn_dim).transpose(1, 2).contiguous()  # B, N, L

        return squeeze_mean, squeeze_block, squeeze_rest


class ContextDecoder(nn.Module):
    def __init__(
        self,
        in_chan: int,
        hid_chan: int,
        rnn_type: str = "LSTM",
        num_group: int = 2,
        context_size: int = 2,
        dropout: float = 0,
        num_layers: int = 1,
        bidirectional: bool = True,
    ):
        super().__init__()

        self.in_chan = in_chan
        self.hid_chan = hid_chan
        self.rnn_type = rnn_type
        self.num_group = num_group
        self.context_size = context_size
        self.dropout = dropout
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.context_dec = GC_RNN(
            input_size=self.in_chan,
            hidden_size=self.hid_chan,
            rnn_type=self.rnn_type,
            num_group=self.num_group,
            dropout=self.dropout,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional,
        )

    def forward(self, x, res, squeeze_rest):
        x = x.unsqueeze(2) + res  # B, N, context, L
        batch_size, bn_dim, _, seq_len = x.shape
        x = x.permute(0, 3, 1, 2).contiguous().view(batch_size * seq_len, bn_dim, self.context_size)  # B*L, N, context
        unsqueeze_output = self.context_dec(x).view(batch_size, seq_len, bn_dim, self.context_size)  # B, L, N, context
        unsqueeze_output = unsqueeze_output.permute(0, 2, 3, 1).contiguous()  # B, N, context, L
        unsqueeze_output = merge_feature(unsqueeze_output, squeeze_rest)  # B, N, T

        return unsqueeze_output
