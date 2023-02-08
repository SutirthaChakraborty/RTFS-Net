import torch
import torch.nn as nn
import torch.nn.functional as F

from .cnn_layers import ConvNormAct


class FRCNNBlock(nn.Module):
    def __init__(
        self,
        in_chan: int,
        hid_chan: int,
        kernel_size: int = 5,
        stride: int = 2,
        norm_type: str = "BatchNorm1d",
        act_type: str = "PReLU",
        upsampling_depth: int = 4,
        dropout: int = -1,
    ):
        super(FRCNNBlock, self).__init__()
        self.in_chan = in_chan
        self.hid_chan = hid_chan
        self.kernel_size = kernel_size
        self.stride = stride
        self.norm_type = norm_type
        self.act_type = act_type
        self.upsampling_depth = upsampling_depth
        self.dropout = dropout

        self.projection = ConvNormAct(self.in_chan, self.hid_chan, 1, norm_type=self.norm_type, act_type=self.act_type)
        self.downsample_layers = self.__build_downsample_layers()
        self.fusion_layers = self.__build_fusion_layers()
        self.concat_layers = self.__build_concat_layers()
        self.residual_conv = nn.Sequential(
            ConvNormAct(self.hid_chan * self.upsampling_depth, self.hid_chan, 1, norm_type=self.norm_type, act_type=self.act_type),
            nn.Conv1d(self.hid_chan, self.in_chan, 1),
        )
        self.dropout_layer = nn.Dropout(self.dropout) if self.dropout > 0 else nn.Identity()

    def __build_downsample_layers(self):
        out = nn.ModuleList()
        for i in range(self.upsampling_depth):
            stride = 1 if i == 0 else self.stride
            out.append(
                ConvNormAct(
                    self.hid_chan,
                    self.hid_chan,
                    kernel_size=self.kernel_size,
                    stride=stride,
                    groups=self.hid_chan,
                    padding=(self.kernel_size - 1) // 2,
                    norm_type=self.norm_type,
                    act_type=None,
                )
            )
        return out

    def __build_fusion_layers(self):
        out = nn.ModuleList()
        for i in range(self.upsampling_depth):
            fuse_layer = nn.ModuleList()
            for j in range(self.upsampling_depth):
                if i == j or (j - i == 1):
                    fuse_layer.append(None)
                elif i - j == 1:
                    fuse_layer.append(
                        ConvNormAct(
                            self.hid_chan,
                            self.hid_chan,
                            kernel_size=self.kernel_size,
                            stride=self.stride,
                            groups=self.hid_chan,
                            padding=(self.kernel_size - 1) // 2,
                            norm_type=self.norm_type,
                            act_type=None,
                        )
                    )
            out.append(fuse_layer)
        return out

    def __build_concat_layers(self):
        out = nn.ModuleList()
        for i in range(self.upsampling_depth):
            if i == 0 or i == self.upsampling_depth - 1:
                out.append(
                    ConvNormAct(
                        self.hid_chan * 2,
                        self.hid_chan,
                        kernel_size=1,
                        norm_type=self.norm_type,
                        act_type=self.act_type,
                    )
                )
            else:
                out.append(
                    ConvNormAct(
                        self.hid_chan * 3,
                        self.hid_chan,
                        kernel_size=1,
                        norm_type=self.norm_type,
                        act_type=self.act_type,
                    )
                )
        return out

    def forward(self, x):
        # x: shape (B, C, T)
        res = x
        x = self.projection(x)

        # bottom-up
        output = [self.downsample_layers[0](x)]
        for k in range(1, self.upsampling_depth):
            out_k = self.downsample_layers[k](output[-1])
            output.append(out_k)

        # lateral connection
        x_fuse = []
        for i in range(self.upsampling_depth):
            T = output[i].shape[-1]
            y = torch.cat(
                (
                    self.fusion_layers[i][0](output[i - 1]) if i - 1 >= 0 else torch.Tensor().to(x.device),
                    output[i],
                    F.interpolate(output[i + 1], size=T, mode="nearest") if i + 1 < self.upsampling_depth else torch.Tensor().to(x.device),
                ),
                dim=1,
            )
            x_fuse.append(self.concat_layers[i](y))

        # resize to T
        T = output[0].shape[-1]
        for i in range(1, len(x_fuse)):
            x_fuse[i] = F.interpolate(x_fuse[i], size=T, mode="nearest")

        # concat and shortcut
        x = self.residual_conv(torch.cat(x_fuse, dim=1))
        # dropout
        x = self.dropout_layer(x)

        return res + x


class FRCNN(nn.Module):
    def __init__(
        self,
        in_chan: int,
        hid_chan: int,
        kernel_size: int = 5,
        stride: int = 2,
        norm_type: str = "BatchNorm1d",
        act_type: str = "PReLU",
        upsampling_depth: int = 4,
        dropout: float = -1,
        repeats: int = 4,
        shared: bool = False,
        *args,
        **kwargs,
    ):
        super(FRCNN, self).__init__()
        self.in_chan = in_chan
        self.hid_chan = hid_chan
        self.upsampling_depth = upsampling_depth
        self.repeats = repeats
        self.shared = shared
        self.dropout = dropout
        self.norm_type = norm_type
        self.act_type = act_type
        self.kernel_size = kernel_size
        self.stride = stride

        self.frcnn = self.__build_frcnn()
        self.concat_block = self.__build_concat_block()

    def __build_frcnn(self):
        if self.shared:
            out = FRCNNBlock(
                in_chan=self.in_chan,
                hid_chan=self.hid_chan,
                kernel_size=self.kernel_size,
                stride=self.stride,
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
                        hid_chan=self.hid_chan,
                        kernel_size=self.kernel_size,
                        stride=self.stride,
                        norm_type=self.norm_type,
                        act_type=self.act_type,
                        upsampling_depth=self.upsampling_depth,
                        dropout=self.dropout,
                    )
                )

        return out

    def __build_concat_block(self):
        if self.shared:
            out = ConvNormAct(in_chan=self.in_chan, out_chan=self.in_chan, kernel_size=1, groups=self.in_chan, act_type="PReLU")
        else:
            out = nn.ModuleList([None])
            for _ in range(self.repeats - 1):
                out.append(ConvNormAct(in_chan=self.in_chan, out_chan=self.in_chan, kernel_size=1, groups=self.in_chan, act_type="PReLU"))

        return out

    def get_block(self, i):
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
            frcnn = self.get_block(i)
            concat_block = self.get_concat_block(i)
            if i == 0:
                x = frcnn(x)
            else:
                x = frcnn(concat_block(res + x))
        return x
