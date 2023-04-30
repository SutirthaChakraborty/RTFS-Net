import torch
import torch.nn as nn


class TAC(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super(TAC, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.TAC_input = nn.Sequential(nn.Linear(self.input_size, self.hidden_size), nn.PReLU())
        self.TAC_mean = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size), nn.PReLU())
        self.TAC_output = nn.Sequential(nn.Linear(self.hidden_size * 2, self.input_size), nn.PReLU())
        self.TAC_norm = nn.GroupNorm(1, self.input_size)

    def forward(self, x: torch.Tensor):
        shape = x.shape
        x = x.view(*shape[:3], -1)
        # input shape: batch, group, N, seq_length, (freq)

        batch_size, groups, n_chan, seq_len = x.shape
        residual = x

        # transform
        group_input = x.permute(0, 3, 1, 2).contiguous().view(-1, n_chan)  # B*T*G, N
        group_output = self.TAC_input(group_input).view(batch_size, seq_len, groups, -1)  # B, T, G, H

        # mean pooling
        group_mean = group_output.mean(2).view(batch_size * seq_len, -1)  # B*T, H

        # concate
        group_output = group_output.view(batch_size * seq_len, groups, -1)  # B*T, G, H
        group_mean = self.TAC_mean(group_mean).unsqueeze(1).expand_as(group_output).contiguous()  # B*T, G, H
        group_output = torch.cat([group_output, group_mean], 2)  # B*T, G, 2H
        group_output = self.TAC_output(group_output.view(-1, group_output.shape[-1]))  # B*T*G, N
        group_output = group_output.view(batch_size, seq_len, groups, -1).permute(0, 2, 3, 1).contiguous()  # B, G, N, T
        group_output = self.TAC_norm(group_output.view(batch_size * groups, n_chan, seq_len))  # B*G, N, T
        output = residual + group_output.view(x.shape)

        output = output.view(*shape)

        return output


class RNNProjection(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        rnn_type: str = "LSTM",
        dropout: float = 0,
        bidirectional: bool = True,
        *args,
        **kwargs,
    ):
        super(RNNProjection, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn_type = rnn_type
        self.dropout = dropout
        self.num_direction = int(bidirectional) + 1

        self.norm1 = nn.LayerNorm(self.input_size)
        self.rnn = getattr(nn, rnn_type)(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.proj = nn.Sequential(
            nn.PReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size * self.num_direction, self.input_size),
            nn.Dropout(self.dropout),
        )
        self.norm2 = nn.LayerNorm(self.input_size)

    def forward(self, x: torch.Tensor):
        res = x
        x = x.transpose(1, 2).contiguous()

        x = self.norm1(x)
        residual = x
        x = self.rnn(x)[0].contiguous()  # B, L, num_direction * H
        x = self.proj(x)
        x = self.norm2(x + residual)  # B, L, N

        x = x.transpose(1, 2).contiguous()
        x = x + res
        return x
