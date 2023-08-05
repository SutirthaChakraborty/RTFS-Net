import math
import torch
import inspect
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import DropPath
from .. import normalizations, activations


class ConvNormAct(nn.Module):
    def __init__(
        self,
        in_chan: int = 1,
        out_chan: int = 1,
        kernel_size: int = -1,
        stride: int = 1,
        groups: int = 1,
        dilation: int = 1,
        padding: int = None,
        norm_type: str = None,
        act_type: str = None,
        xavier_init: bool = False,
        bias: bool = True,
        is2d: bool = False,
        *args,
        **kwargs,
    ):
        super(ConvNormAct, self).__init__()
        self.in_chan = in_chan
        self.out_chan = out_chan
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.dilation = dilation
        self.padding = padding
        self.norm_type = norm_type
        self.act_type = act_type
        self.xavier_init = xavier_init
        self.bias = bias

        if self.padding is None:
            self.padding = dilation * (kernel_size - 1) // 2 if self.stride > 1 else "same"

        if kernel_size > 0:
            conv = nn.Conv2d if is2d else nn.Conv1d

            self.conv = conv(
                in_channels=self.in_chan,
                out_channels=self.out_chan,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
                bias=self.bias,
            )
            if self.xavier_init:
                nn.init.xavier_uniform_(self.conv.weight)
        else:
            self.conv = nn.Identity()

        self.norm = normalizations.get(self.norm_type)(self.out_chan)
        self.act = activations.get(self.act_type)()

    def forward(self, x: torch.Tensor):
        output = self.conv(x)
        output = self.norm(output)
        output = self.act(output)
        return output

    def get_config(self):
        encoder_args = {}

        for k, v in (self.__dict__).items():
            if not k.startswith("_") and k != "training":
                if not inspect.ismethod(v):
                    encoder_args[k] = v

        return encoder_args


class ConvActNorm(nn.Module):
    def __init__(
        self,
        in_chan: int = 1,
        out_chan: int = 1,
        kernel_size: int = -1,
        stride: int = 1,
        groups: int = 1,
        dilation: int = 1,
        padding: int = None,
        norm_type: str = None,
        act_type: str = None,
        n_freqs: int = -1,
        xavier_init: bool = False,
        bias: bool = True,
        is2d: bool = False,
        *args,
        **kwargs,
    ):
        super(ConvActNorm, self).__init__()
        self.in_chan = in_chan
        self.out_chan = out_chan
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.dilation = dilation
        self.padding = padding
        self.norm_type = norm_type
        self.act_type = act_type
        self.n_freqs = n_freqs
        self.xavier_init = xavier_init
        self.bias = bias

        if self.padding is None:
            self.padding = 0 if self.stride > 1 else "same"

        if kernel_size > 0:
            conv = nn.Conv2d if is2d else nn.Conv1d

            self.conv = conv(
                in_channels=self.in_chan,
                out_channels=self.out_chan,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
                bias=self.bias,
            )
            if self.xavier_init:
                nn.init.xavier_uniform_(self.conv.weight)
        else:
            self.conv = nn.Identity()

        self.act = activations.get(self.act_type)()
        self.norm = normalizations.get(self.norm_type)(
            (self.out_chan, self.n_freqs) if self.norm_type == "LayerNormalization4D" else self.out_chan
        )

    def forward(self, x: torch.Tensor):
        output = self.conv(x)
        output = self.act(output)
        output = self.norm(output)
        return output

    def get_config(self):
        encoder_args = {}

        for k, v in (self.__dict__).items():
            if not k.startswith("_") and k != "training":
                if not inspect.ismethod(v):
                    encoder_args[k] = v

        return encoder_args


class FeedForwardNetwork(nn.Module):
    def __init__(
        self,
        in_chan: int,
        hid_chan: int,
        kernel_size: int = 5,
        norm_type: str = "gLN",
        act_type: str = "ReLU",
        dropout: float = 0,
        is2d: bool = False,
        *args,
        **kwargs,
    ):
        super(FeedForwardNetwork, self).__init__()
        self.in_chan = in_chan
        self.hid_chan = hid_chan
        self.kernel_size = kernel_size
        self.norm_type = norm_type
        self.act_type = act_type
        self.dropout = dropout
        self.is2d = is2d

        self.encoder = ConvNormAct(self.in_chan, self.hid_chan, 1, norm_type=self.norm_type, bias=False, is2d=self.is2d)  # FC 1
        self.refiner = ConvNormAct(
            self.hid_chan,
            self.hid_chan,
            self.kernel_size,
            groups=self.hid_chan,
            act_type=self.act_type,
            is2d=self.is2d,
        )  # DW seperable conv
        self.decoder = ConvNormAct(self.hid_chan, self.in_chan, 1, norm_type=self.norm_type, bias=False, is2d=self.is2d)  # FC 2
        self.dropout_layer = DropPath(self.dropout)

    def forward(self, x: torch.Tensor):
        res = x
        x = self.encoder(x)
        x = self.refiner(x)
        x = self.dropout_layer(x)
        x = self.decoder(x)
        x = self.dropout_layer(x) + res
        return x


class ConvolutionalRNN(nn.Module):
    def __init__(
        self,
        in_chan: int,
        hid_chan: int,
        kernel_size: int = 5,
        norm_type: str = "gLN",
        act_type: str = "ReLU",
        dropout: float = 0,
        is2d: bool = False,
        *args,
        **kwargs,
    ):
        super(ConvolutionalRNN, self).__init__()
        self.in_chan = in_chan
        self.hid_chan = hid_chan
        self.kernel_size = kernel_size
        self.norm_type = norm_type
        self.act_type = act_type
        self.dropout = dropout
        self.is2d = is2d

        self.encoder = ConvNormAct(self.in_chan, self.hid_chan, 1, norm_type=self.norm_type, bias=False, is2d=self.is2d)  # FC 1
        self.forward_pass = ConvNormAct(
            self.hid_chan,
            self.hid_chan,
            self.kernel_size,
            groups=self.hid_chan,
            act_type=self.act_type,
            is2d=self.is2d,
        )  # DW seperable conv
        self.backward_pass = ConvNormAct(
            self.hid_chan,
            self.hid_chan,
            self.kernel_size,
            groups=self.hid_chan,
            act_type=self.act_type,
            is2d=self.is2d,
        )  # DW seperable conv
        self.decoder = ConvNormAct(self.hid_chan * 2, self.in_chan, 1, norm_type=self.norm_type, bias=False, is2d=self.is2d)  # FC 2
        self.dropout_layer = DropPath(self.dropout)

    def forward(self, x: torch.Tensor):
        res = x
        x = self.encoder(x)
        forward_features = self.forward_pass(x)
        if self.is2d:
            backward_features = self.backward_pass(x.flip([2, 3]))
        else:
            backward_features = self.backward_pass(x.flip(2))
        x = torch.cat([forward_features, backward_features], dim=1)
        x = self.dropout_layer(x)
        x = self.decoder(x)
        x = self.dropout_layer(x) + res
        return x


class InjectionMultiSum(nn.Module):
    def __init__(
        self,
        in_chan: int,
        hid_chan: int,
        kernel_size: int,
        norm_type: str = "gLN",
        is2d: bool = False,
    ):
        super(InjectionMultiSum, self).__init__()
        self.in_chan = in_chan
        self.hid_chan = hid_chan
        self.kernel_size = kernel_size
        self.norm_type = norm_type
        self.is2d = is2d

        self.groups = in_chan if in_chan == hid_chan else 1

        self.local_embedding = ConvNormAct(
            in_chan=self.in_chan,
            out_chan=self.hid_chan,
            kernel_size=self.kernel_size,
            groups=self.groups,
            norm_type=self.norm_type,
            bias=False,
            is2d=self.is2d,
        )
        self.global_embedding = ConvNormAct(
            in_chan=self.in_chan,
            out_chan=self.hid_chan,
            kernel_size=self.kernel_size,
            groups=self.groups,
            norm_type=self.norm_type,
            bias=False,
            is2d=self.is2d,
        )
        self.global_gate = ConvNormAct(
            in_chan=self.in_chan,
            out_chan=self.hid_chan,
            kernel_size=self.kernel_size,
            groups=self.groups,
            norm_type=self.norm_type,
            act_type="Sigmoid",
            bias=False,
            is2d=self.is2d,
        )

    def forward(self, local_features: torch.Tensor, global_features: torch.Tensor):
        old_shape = global_features.shape[-(len(local_features.shape) // 2) :]
        new_shape = local_features.shape[-(len(local_features.shape) // 2) :]

        local_emb = self.local_embedding(local_features)
        if torch.prod(torch.tensor(new_shape)) > torch.prod(torch.tensor(old_shape)):
            global_emb = F.interpolate(self.global_embedding(global_features), size=new_shape, mode="nearest")
            gate = F.interpolate(self.global_gate(global_features), size=new_shape, mode="nearest")
        else:
            g_interp = F.interpolate(global_features, size=new_shape, mode="nearest")
            global_emb = self.global_embedding(g_interp)
            gate = self.global_gate(g_interp)

        injection_sum = local_emb * gate + global_emb

        return injection_sum


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
        self.rnn.flatten_parameters()
        x = self.rnn(x)[0]  # B, L, num_direction * H
        x = self.proj(x)
        x = self.norm2(x + residual)  # B, L, N

        x = x.transpose(1, 2).contiguous()
        x = x + res
        return x


class DualPathRNN(nn.Module):
    def __init__(
        self,
        in_chan: int,
        hid_chan: int,
        dim: int,
        kernel_size: int = 8,
        stride: int = 1,
        rnn_type: str = "LSTM",
        norm_type: str = "LayerNormalization4D",
        bidirectional: bool = True,
        *args,
        **kwargs,
    ):
        super(DualPathRNN, self).__init__()
        self.in_chan = in_chan
        self.hid_chan = hid_chan
        self.dim = dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.rnn_type = rnn_type
        self.norm_type = norm_type
        self.bidirectional = bidirectional
        self.num_direction = int(bidirectional) + 1

        self.unfolded_chan = self.in_chan * self.kernel_size

        self.norm = normalizations.get(self.norm_type)((self.in_chan, 1) if self.norm_type == "LayerNormalization4D" else self.in_chan)
        self.unfold = nn.Unfold((self.kernel_size, 1), stride=(self.stride, 1))
        self.rnn = getattr(nn, self.rnn_type)(
            input_size=self.unfolded_chan,
            hidden_size=self.hid_chan,
            num_layers=1,
            batch_first=True,
            bidirectional=self.bidirectional,
        )
        self.linear = nn.ConvTranspose1d(self.hid_chan * self.num_direction, self.in_chan, self.kernel_size, stride=self.stride)

    def forward(self, x: torch.Tensor):
        if self.dim == 4:
            x = x.transpose(-2, -1).contiguous()

        B, C, old_T, old_F = x.shape
        new_T = math.ceil((old_T - self.kernel_size) / self.stride) * self.stride + self.kernel_size
        new_F = math.ceil((old_F - self.kernel_size) / self.stride) * self.stride + self.kernel_size
        x = F.pad(x, (0, new_F - old_F, 0, new_T - old_T))

        residual = x
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2).contiguous().view(B * new_F, C, new_T, 1)
        x = self.unfold(x)
        x = x.transpose(1, 2)
        x = self.rnn(x)[0]
        x = x.transpose(1, 2)
        x = self.linear(x)
        x = x.view([B, new_F, C, new_T])
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x + residual
        x = x[..., :old_T, :old_F]

        if self.dim == 4:
            x = x.transpose(-2, -1).contiguous()

        return x


class ConvLSTMCell(nn.Module):
    def __init__(self, in_chan: int, hid_chan: int, kernel_size: int = 1, num_directions: int = 1):
        super(ConvLSTMCell, self).__init__()
        self.in_chan = in_chan
        self.hid_chan = hid_chan
        self.kernel_size = kernel_size
        self.num_directions = num_directions

        self.linear_ih = nn.Sequential(
            ConvActNorm(self.in_chan, self.in_chan, self.kernel_size, groups=self.in_chan),
            ConvActNorm(self.in_chan, 4 * self.hid_chan, 1),
        )
        self.linear_hh = ConvActNorm(self.hid_chan, 4 * self.hid_chan, 1)

        if self.num_directions > 1:
            self.linear_ih_b = nn.Sequential(
                ConvActNorm(self.in_chan, self.in_chan, self.kernel_size, groups=self.in_chan),
                ConvActNorm(self.in_chan, 4 * self.hid_chan, 1),
            )
            self.linear_hh_b = ConvActNorm(self.hid_chan, 4 * self.hid_chan, 1)

    def forward(self, input: torch.Tensor, hidden_t: torch.Tensor, cell_t: torch.Tensor):
        # x has size: (B, C, L)
        batch_size = input.shape[0]

        if self.num_directions > 1:
            input_f, input_b = input.chunk(2, 1)
            hidden_t_f, hidden_t_b = hidden_t.chunk(2, 1)
            gates_f = self.linear_ih(input_f) + self.linear_hh(hidden_t_f)[:batch_size]
            gates_b = self.linear_ih_b(input_b) + self.linear_hh_b(hidden_t_b)[:batch_size]
            gates = torch.cat((gates_f, gates_b), dim=1)
        else:
            gates = self.linear_ih(input) + self.linear_hh(hidden_t)[:batch_size]

        i_t, f_t, g_t, o_t = gates.chunk(4, 1)

        i_t = torch.sigmoid(i_t)
        f_t = torch.sigmoid(f_t)
        g_t = torch.tanh(g_t)
        o_t = torch.sigmoid(o_t)

        c_next = (f_t * cell_t[:batch_size]) + (i_t * g_t)
        h_next = o_t * torch.tanh(c_next)

        return h_next, c_next


class BiLSTM2D(nn.Module):
    def __init__(
        self,
        in_chan: int,
        hid_chan: int,
        dim: int = 3,
        kernel_size: int = 5,
        window: int = 8,
        stride: int = 1,
        act_type: str = "PReLU",
        norm_type: str = "gLN",
        bidirectional: bool = True,
    ):
        super(BiLSTM2D, self).__init__()
        self.in_chan = in_chan
        self.hid_chan = hid_chan
        self.dim = dim
        self.kernel_size = kernel_size
        self.window = window
        self.stride = stride
        self.act_type = act_type
        self.norm_type = norm_type
        self.bidirectional = bidirectional

        self.num_dir = int(bidirectional) + 1

        self.norm = normalizations.get(self.norm_type)(self.in_chan)
        self.lstm_cell = ConvLSTMCell(self.in_chan * self.window, self.hid_chan, self.kernel_size, self.num_dir)
        ks = (self.window, 1)
        s = (self.stride, 1)
        self.unfold = nn.Unfold(kernel_size=ks, stride=s)
        self.projection = nn.Sequential(
            nn.ConvTranspose2d(
                self.hid_chan * self.num_dir,
                self.hid_chan * self.num_dir,
                ks,
                groups=self.hid_chan * self.num_dir,
                stride=s,
            ),
            activations.get(self.act_type)(),
            normalizations.get(self.norm_type)(self.hid_chan * self.num_dir),
            ConvActNorm(self.hid_chan * self.num_dir, self.in_chan, 1, is2d=True),
        )
        # make this depth wise seperable

    def pad(self, x: torch.Tensor):
        old_w, old_h = x.shape[-2:]
        new_w = math.ceil((old_w - self.window) / self.stride) * self.stride + self.window
        new_h = math.ceil((old_h - self.window) / self.stride) * self.stride + self.window
        x = F.pad(x, (0, new_h - old_h, 0, new_w - old_w))

        iterations = math.ceil(new_h / self.window)

        return x, old_w, old_h, iterations

    def init_states(self, x: torch.Tensor):
        hidden_t = torch.zeros((1, self.hid_chan * self.num_dir, 1), device=x.device)
        cell_t = torch.zeros((1, self.hid_chan * self.num_dir, 1), device=x.device)

        return hidden_t, cell_t

    def forward(self, x: torch.Tensor):
        # x has size: (B, C, T, F)
        bs = x.shape[0]

        residual = x
        x = self.norm(x)

        x = torch.cat((x, x.flip(self.dim - 1)), dim=1) if self.bidirectional else x
        x = x.transpose(-1, -2).contiguous() if self.dim == 3 else x

        x, old_w, old_h, iterations = self.pad(x)
        hidden_t, cell_t = self.init_states(x)

        outputs = [None] * iterations
        for i in range(iterations):
            x_slice = x[..., i * self.window : (i + 1) * self.window]
            w, h = x_slice.shape[-2:]
            x_slice = x_slice.permute(0, 3, 1, 2).contiguous().view(bs * h, self.in_chan * self.num_dir, w, 1)
            x_slice = self.unfold(x_slice)
            hidden_t, cell_t = self.lstm_cell(x_slice, hidden_t, cell_t)
            outputs[i] = hidden_t.view(bs, h, self.hid_chan * self.num_dir, -1).permute(0, 2, 3, 1).contiguous()

        x = self.projection(torch.cat(outputs, dim=-1))[..., :old_w, :old_h]
        x = x.transpose(-1, -2).contiguous() if self.dim == 3 else x
        x = x + residual

        return x


class ConvLSTMFusionCell(nn.Module):
    def __init__(self, in_chan_a: int, in_chan_b, kernel_size: int = 1, bidirectional: bool = False):
        super(ConvLSTMFusionCell, self).__init__()
        self.in_chan_a = in_chan_a
        self.in_chan_b = in_chan_b
        self.kernel_size = kernel_size
        self.bidirectional = bidirectional
        self.num_dir = int(bidirectional) + 1

        self.conv_a = ConvActNorm(
            self.in_chan_a * self.num_dir,
            self.in_chan_a * 4,
            self.kernel_size,
            is2d=True,
            groups=self.in_chan_a * self.num_dir,
            norm_type="gLN",
        )
        self.conv_b = ConvActNorm(
            self.in_chan_b * self.num_dir,
            self.in_chan_a * 4,
            self.kernel_size,
            is2d=True,
            groups=self.in_chan_a * self.num_dir,
            norm_type="gLN",
        )

    def forward(self, tensor_a: torch.Tensor, tensor_b: torch.Tensor):
        if self.bidirectional:
            tensor_a = torch.cat((tensor_a, tensor_a.flip(-1).flip(-2)), dim=1)
            tensor_b = torch.cat((tensor_b, tensor_b.flip(-1).flip(-2)), dim=1)

        old_shape = tensor_b.shape[-(len(tensor_a.shape) // 2) :]
        new_shape = tensor_a.shape[-(len(tensor_a.shape) // 2) :]

        if torch.prod(torch.tensor(new_shape)) > torch.prod(torch.tensor(old_shape)):
            gates = self.conv_a(tensor_a) + F.interpolate(self.conv_b(tensor_b), size=new_shape, mode="nearest")
        else:
            gates = self.conv_a(tensor_a) + self.conv_b(F.interpolate(tensor_b, size=new_shape, mode="nearest"))

        i_t, f_t, g_t, o_t = gates.chunk(4, 1)

        i_t = torch.sigmoid(i_t)
        f_t = torch.sigmoid(f_t)
        g_t = torch.tanh(g_t)
        o_t = torch.sigmoid(o_t)

        c_next = f_t + (i_t * g_t)
        h_next = o_t * torch.tanh(c_next)

        return h_next


def get(identifier):
    if identifier is None:
        return nn.Identity
    elif callable(identifier):
        return identifier
    elif isinstance(identifier, str):
        cls = globals().get(identifier)

        if cls is None:
            raise ValueError("Could not interpret normalization identifier: " + str(identifier))
        return cls
    else:
        raise ValueError("Could not interpret normalization identifier: " + str(identifier))
