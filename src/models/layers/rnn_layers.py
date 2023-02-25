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

    def forward(self, x):
        # input shape: batch, group, N, seq_length

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

        return output


class RNNProjection(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        rnn_type: str = "LSTM",
        dropout: float = 0,
        bidirectional: bool = False,
    ):
        super(RNNProjection, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn_type = rnn_type
        self.dropout = dropout
        self.num_direction = int(bidirectional) + 1

        self.rnn = getattr(nn, rnn_type)(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=1,
            dropout=self.dropout,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.proj = nn.Linear(self.hidden_size * self.num_direction, self.input_size)

    def forward(self, x):
        batch_size, num_group, _, seq_len = x.shape  # B, G, N, L

        x = x.transpose(2, 3).contiguous().view(batch_size * num_group, seq_len, -1)  # B*G, L, N
        rnn_output = self.rnn(x)[0]  # B*G, L, num_direction * H
        rnn_output = rnn_output.contiguous().view(-1, self.num_direction * self.hidden_size)  # B*G*L, num_direction * H
        proj_output = self.proj(rnn_output)  # B*G*L, N
        proj_output = proj_output.view(batch_size, num_group, seq_len, -1).transpose(2, 3).contiguous()  # B, G, N, L

        return proj_output


class GC_RNN(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        rnn_type: str = "LSTM",
        num_group: int = 2,
        dropout: float = 0,
        num_layers: int = 1,
        bidirectional: bool = True,
    ):
        super(GC_RNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn_type = rnn_type
        self.num_group = num_group
        self.dropout = dropout
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.TAC = nn.ModuleList([])
        self.rnn = nn.ModuleList([])
        self.LN = nn.ModuleList([])

        for _ in range(num_layers):
            self.TAC.append(
                TAC(
                    input_size=self.input_size // self.num_group,
                    hidden_size=self.hidden_size * 3 // self.num_group,
                )
            )
            self.rnn.append(
                RNNProjection(
                    input_size=self.input_size // self.num_group,
                    hidden_size=self.hidden_size // self.num_group,
                    rnn_type=self.rnn_type,
                    dropout=self.dropout,
                    bidirectional=self.bidirectional,
                )
            )
            self.LN.append(nn.GroupNorm(num_groups=1, num_channels=self.input_size // self.num_group))

    def forward(self, x):
        batch_size, dim, seq_len = x.shape
        x = x.view(batch_size, self.num_group, -1, seq_len)

        for i in range(self.num_layers):
            x = self.TAC[i](x)
            res = x
            x = self.rnn[i](x)
            x = self.LN[i](x.view(batch_size * self.num_group, -1, seq_len)).view(batch_size, self.num_group, -1, seq_len)
            x = res + x

        x = x.view(batch_size, dim, seq_len)

        return x
