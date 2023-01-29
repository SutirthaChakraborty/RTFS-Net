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
        norm_type: str = None,
        act_type: str = None,
        xavier_init: bool = False,
        bias: bool = True,
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

        self.conv = nn.Conv1d(
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
        in_chan: int,
        out_chan: int,
        kernel_size: int = 5,
        norm_type: str = "BatchNorm1d",
        act_type: str = "PReLU",
        upsampling_depth: int = 4,
        dropout: int = -1,
    ):
        super(FRCNNBlock, self).__init__()
        self.in_chan = in_chan
        self.out_chan = out_chan
        self.kernel_size = kernel_size
        self.norm_type = norm_type
        self.act_type = act_type
        self.upsampling_depth = upsampling_depth
        self.dropout = dropout

        self.proj = ConvNormAct(self.in_chan, self.out_chan, kernel_size=1, norm_type=self.norm_type, act_type=self.act_type)
        self.spp_dw = self._build_downsample_layers()
        self.fuse_layers = self._build_fusion_layers()
        self.concat_layer = self._build_concat_layers()
        self.last = nn.Sequential(
            ConvNormAct(
                self.out_chan * self.upsampling_depth, self.out_chan, kernel_size=1, norm_type=self.norm_type, act_type=self.act_type
            ),
            nn.Conv1d(self.out_chan, self.in_chan, 1),
        )
        self.dropout_layer = nn.Dropout(self.dropout) if self.dropout > 0 else nn.Identity()

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
        for _ in range(1, self.upsampling_depth):
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
        for i in range(self.upsampling_depth):
            fuse_layer = nn.ModuleList()
            for j in range(self.upsampling_depth):
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
        for i in range(self.upsampling_depth):
            if i == 0 or i == self.upsampling_depth - 1:
                out.append(
                    ConvNormAct(
                        self.out_chan * 2,
                        self.out_chan,
                        kernel_size=1,
                        norm_type=self.norm_type,
                        act_type=self.act_type,
                    )
                )
            else:
                out.append(
                    ConvNormAct(
                        self.out_chan * 3,
                        self.out_chan,
                        kernel_size=1,
                        norm_type=self.norm_type,
                        act_type=self.act_type,
                    )
                )
        return out

    def forward(self, x):
        # x: shape (B, C, T)
        res = x
        x = self.proj(x)

        # bottom-up
        output = [self.spp_dw[0](x)]
        for k in range(1, self.upsampling_depth):
            out_k = self.spp_dw[k](output[-1])
            output.append(out_k)

        # lateral connection
        x_fuse = []
        for i in range(self.upsampling_depth):
            T = output[i].shape[-1]
            y = torch.cat(
                (
                    self.fuse_layers[i][0](output[i - 1]) if i - 1 >= 0 else torch.Tensor().to(x.device),
                    output[i],
                    F.interpolate(output[i + 1], size=T, mode="nearest") if i + 1 < self.upsampling_depth else torch.Tensor().to(x.device),
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
        x = self.dropout_layer(x)

        return res + x


class FRCNN(nn.Module):
    def __init__(
        self,
        in_chan: int,
        out_chan: int,
        upsampling_depth: int = 4,
        repeats: int = 4,
        shared: bool = False,
        dropout: float = -1,
        norm_type: str = "BatchNorm1d",
        act_type: str = "PReLU",
        kernel_size: int = 5,
        *args,
        **kwargs
    ):
        super(FRCNN, self).__init__()
        self.in_chan = in_chan
        self.out_chan = out_chan
        self.upsampling_depth = upsampling_depth
        self.repeats = repeats
        self.shared = shared
        self.dropout = dropout
        self.norm_type = norm_type
        self.act_type = act_type
        self.kernel_size = kernel_size

        self.frcnn = self._build_frcnn()
        self.concat_block = self._build_concat_block()

    def _build_frcnn(self):
        if self.shared:
            return FRCNNBlock(
                in_chan=self.in_chan,
                out_chan=self.out_chan,
                kernel_size=self.kernel_size,
                norm_type=self.norm_type,
                act_type=self.act_type,
                upsampling_depth=self.upsampling_depth,
                dropout=self.dropout,
            )
        else:
            out = nn.ModuleList()
            for _ in range(self.repeats):
                out.append(
                    FRCNNBlock(
                        in_chan=self.in_chan,
                        out_chan=self.out_chan,
                        kernel_size=self.kernel_size,
                        norm_type=self.norm_type,
                        act_type=self.act_type,
                        upsampling_depth=self.upsampling_depth,
                        dropout=self.dropout,
                    )
                )
            return out

    def _build_concat_block(self):
        if self.shared:
            return ConvNormAct(in_chan=self.in_chan, out_chan=self.in_chan, kernel_size=1, groups=self.in_chan, act_type="PReLU")
        else:
            out = nn.ModuleList([None])
            for _ in range(self.repeats - 1):
                out.append(ConvNormAct(in_chan=self.in_chan, out_chan=self.in_chan, kernel_size=1, groups=self.in_chan, act_type="PReLU"))
            return out

    def get_frcnn_block(self, i):
        if self.shared:
            return self.frcnn
        else:
            return self.frcnn[i]

    def get_concat_block(self, i):
        if self.shared:
            return self.concat_block
        else:
            return self.concat_block[i]

    def forward(self, x):
        # x: shape (B, C, T)
        res = x
        for i in range(self.repeats):
            frcnn = self.get_frcnn_block(i)
            concat_block = self.get_concat_block(i)
            if i == 0:
                x = frcnn(x)
            else:
                x = frcnn(concat_block(res + x))
        return x
