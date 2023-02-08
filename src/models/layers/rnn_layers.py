import torch
import torch.nn as nn


class TAC(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(TAC, self).__init__()

        self.TAC_input = nn.Sequential(nn.Linear(input_size, hidden_size), nn.PReLU())
        self.TAC_mean = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.PReLU())
        self.TAC_output = nn.Sequential(nn.Linear(hidden_size * 2, input_size), nn.PReLU())
        self.TAC_norm = nn.GroupNorm(1, input_size)

    def forward(self, input):
        # input shape: batch, group, N, seq_length

        batch_size, G, N, T = input.shape
        output = input

        # transform
        group_input = output  # B, G, N, T
        group_input = output.permute(0, 3, 1, 2).contiguous().view(-1, N)  # B*T*G, N
        group_output = self.TAC_input(group_input).view(batch_size, T, G, -1)  # B, T, G, H

        # mean pooling
        group_mean = group_output.mean(2).view(batch_size * T, -1)  # B*T, H

        # concate
        group_output = group_output.view(batch_size * T, G, -1)  # B*T, G, H
        group_mean = self.TAC_mean(group_mean).unsqueeze(1).expand_as(group_output).contiguous()  # B*T, G, H
        group_output = torch.cat([group_output, group_mean], 2)  # B*T, G, 2H
        group_output = self.TAC_output(group_output.view(-1, group_output.shape[-1]))  # B*T*G, N
        group_output = group_output.view(batch_size, T, G, -1).permute(0, 2, 3, 1).contiguous()  # B, G, N, T
        group_output = self.TAC_norm(group_output.view(batch_size * G, N, T))  # B*G, N, T
        output = output + group_output.view(input.shape)

        return output


class ProjRNN(nn.Module):
    def __init__(self, input_size, hidden_size, rnn_type="LSTM", dropout=0, bidirectional=False):
        super(ProjRNN, self).__init__()

        self.input_size = input_size
        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.num_direction = int(bidirectional) + 1

        self.rnn = getattr(nn, rnn_type)(input_size, hidden_size, 1, dropout=dropout, batch_first=True, bidirectional=bidirectional)
        self.proj = nn.Linear(hidden_size * self.num_direction, input_size)

    def forward(self, input):
        # input shape: batch, seq, dim
        output = input
        rnn_output, _ = self.rnn(output)
        proj_output = self.proj(rnn_output.contiguous().view(-1, rnn_output.shape[2])).view(output.shape)
        return proj_output


class GC_RNN(nn.Module):
    def __init__(self, input_size, hidden_size, rnn_type="LSTM", num_group=2, dropout=0, num_layers=1, bidirectional=False):
        super(GC_RNN, self).__init__()

        self.TAC = nn.ModuleList([])
        self.rnn = nn.ModuleList([])
        self.LN = nn.ModuleList([])

        self.num_layers = num_layers
        self.num_group = num_group

        for i in range(num_layers):
            self.TAC.append(TAC(input_size // num_group, hidden_size * 3 // num_group))
            self.rnn.append(ProjRNN(input_size // num_group, hidden_size // num_group, rnn_type, dropout, bidirectional))
            self.LN.append(nn.GroupNorm(1, input_size // num_group))

    def forward(self, input):
        batch_size, dim, seq_len = input.shape

        output = input.view(batch_size, self.num_group, -1, seq_len)
        for i in range(self.num_layers):
            output = self.TAC[i](output).transpose(2, 3).contiguous()
            output = output.view(batch_size * self.num_group, seq_len, -1)
            rnn_output = self.rnn[i](output)
            norm_output = self.LN[i](rnn_output.transpose(1, 2))
            output = output + norm_output.transpose(1, 2)
            output = output.view(batch_size, self.num_group, seq_len, -1).transpose(2, 3).contiguous()

        output = output.view(batch_size, dim, seq_len)

        return output
