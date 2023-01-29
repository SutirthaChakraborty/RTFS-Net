import torch
import torch.nn as nn
import torch.nn.functional as F

from . import normalizations, activations


class ConvNormAct(nn.Module):
    def __init__(
        self,
        in_chan: int,
        out_chan: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
        dilation: int = 1,
        padding: int = 0,
        norm_type="gLN",
        act_type="PReLU",
        xavier_init=False,
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

        self.conv = nn.Conv1d(
            in_channels=self.in_chan,
            out_channels=self.out_chan,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            bias=True,
        )
        if self.xavier_init:
            nn.init.xavier_uniform_(self.conv.weight)

        self.norm = normalizations.get(self.norm_type)(self.out_chan)
        self.act = activations.get(self.act_type)()

    def forward(self, x):
        output = self.conv(x)
        output = self.norm(output)
        output = self.act(output)
        return output


class FRCNNBlock(nn.Module):
    def __init__(
        self,
        in_chan=128,
        out_chan=512,
        kernel_size=5,
        dilation=1,
        norm_type="BatchNorm1d",
        act_type="PReLU",
        depth=4,
        dropout=-1,
    ):
        super(FRCNNBlock, self).__init__()
        self.in_chan = in_chan
        self.out_chan = out_chan
        self.depth = depth
        self.norm_type = norm_type
        self.act_type = act_type
        self.dilation = dilation
        self.kernel_size = kernel_size

        self.proj = ConvNormAct(in_chan, out_chan, kernel_size=1, norm_type=norm_type, act_type=act_type)
        self.spp_dw = self._build_downsample_layers()
        self.fuse_layers = self._build_fusion_layers()
        self.concat_layer = self._build_concat_layers()
        self.last = nn.Sequential(
            ConvNormAct(out_chan * depth, out_chan, kernel_size=1, norm_type=norm_type, act_type=act_type), nn.Conv1d(out_chan, in_chan, 1)
        )
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else None

    def _build_downsample_layers(self):
        out = nn.ModuleList()
        out.append(
            ConvNormAct(
                self.out_chan,
                self.out_chan,
                kernel_size=self.kernel_size,
                groups=self.out_chan,
                padding=((self.kernel_size - 1) // 2) * 1,
                norm_type=self.norm_type,
                act_type=None,
            )
        )
        # ----------Down Sample Layer----------
        for _ in range(1, self.depth):
            out.append(
                ConvNormAct(
                    self.out_chan,
                    self.out_chan,
                    kernel_size=self.kernel_size,
                    stride=2,
                    groups=self.out_chan,
                    padding=((self.kernel_size - 1) // 2) * 1,
                    norm_type=self.norm_type,
                    act_type=None,
                )
            )
        return out

    def _build_fusion_layers(self):
        out = nn.ModuleList()
        for i in range(self.depth):
            fuse_layer = nn.ModuleList()
            for j in range(self.depth):
                if i == j or (j - i == 1):
                    fuse_layer.append(None)
                elif i - j == 1:
                    fuse_layer.append(
                        ConvNormAct(
                            self.out_chan,
                            self.out_chan,
                            kernel_size=self.kernel_size,
                            stride=2,
                            groups=self.out_chan,
                            padding=((self.kernel_size - 1) // 2) * 1,
                            norm_type=self.norm_type,
                            act_type=None,
                        )
                    )
            out.append(fuse_layer)
        return out

    def _build_concat_layers(self):
        out = nn.ModuleList()
        for i in range(self.depth):
            if i == 0 or i == self.depth - 1:
                out.append(ConvNormAct(self.out_chan * 2, self.out_chan, kernel_size=1, norm_type=self.norm_type, act_type=self.act_type))
            else:
                out.append(ConvNormAct(self.out_chan * 3, self.out_chan, kernel_size=1, norm_type=self.norm_type, act_type=self.act_type))
        return out

    def forward(self, x):
        # x: shape (B, C, T)
        res = x
        x = self.proj(x)

        # bottom-up
        output = [self.spp_dw[0](x)]
        for k in range(1, self.depth):
            out_k = self.spp_dw[k](output[-1])
            output.append(out_k)

        # lateral connection
        x_fuse = []
        for i in range(self.depth):
            T = output[i].shape[-1]
            y = torch.cat(
                (
                    self.fuse_layers[i][0](output[i - 1]) if i - 1 >= 0 else torch.Tensor().to(x.device),
                    output[i],
                    F.interpolate(output[i + 1], size=T, mode="nearest") if i + 1 < self.depth else torch.Tensor().to(x.device),
                ),
                dim=1,
            )
            x_fuse.append(self.concat_layer[i](y))

        # resize to T
        T = output[0].shape[-1]
        for i in range(1, len(x_fuse)):
            x_fuse[i] = F.interpolate(x_fuse[i], size=T, mode="nearest")

        # concat and shortcut
        x = self.last(torch.cat(x_fuse, dim=1))
        # dropout
        if self.dropout_layer:
            x = self.dropout_layer(x)

        return res + x
