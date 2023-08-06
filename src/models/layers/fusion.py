import torch
import torch.nn as nn
import torch.nn.functional as F


from .conv_layers import ConvNormAct


class InjectionMultiSum(nn.Module):
    def __init__(
        self,
        in_chan: int,
        kernel_size: int,
        norm_type: str = "gLN",
        is2d: bool = False,
    ):
        super(InjectionMultiSum, self).__init__()
        self.in_chan = in_chan
        self.kernel_size = kernel_size
        self.norm_type = norm_type
        self.is2d = is2d

        self.local_embedding = ConvNormAct(
            in_chan=self.in_chan,
            out_chan=self.in_chan,
            kernel_size=self.kernel_size,
            groups=self.in_chan,
            norm_type=self.norm_type,
            bias=False,
            is2d=self.is2d,
        )
        self.global_embedding = ConvNormAct(
            in_chan=self.in_chan,
            out_chan=self.in_chan,
            kernel_size=self.kernel_size,
            groups=self.in_chan,
            norm_type=self.norm_type,
            bias=False,
            is2d=self.is2d,
        )
        self.global_gate = ConvNormAct(
            in_chan=self.in_chan,
            out_chan=self.in_chan,
            kernel_size=self.kernel_size,
            groups=self.in_chan,
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


class ConvLSTMFusionCell(nn.Module):
    def __init__(self, in_chan_a: int, in_chan_b, kernel_size: int = 1, bidirectional: bool = False):
        super(ConvLSTMFusionCell, self).__init__()
        self.in_chan_a = in_chan_a
        self.in_chan_b = in_chan_b
        self.kernel_size = kernel_size
        self.bidirectional = bidirectional
        self.num_dir = int(bidirectional) + 1

        self.conv_a = ConvNormAct(
            self.in_chan_a * self.num_dir,
            self.in_chan_a * 4,
            self.kernel_size,
            is2d=True,
            groups=self.in_chan_a // 4,
            norm_type="gLN",
        )
        self.conv_b = ConvNormAct(
            self.in_chan_b * self.num_dir,
            self.in_chan_a * 4,
            self.kernel_size,
            is2d=True,
            groups=self.in_chan_a // 4,
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
