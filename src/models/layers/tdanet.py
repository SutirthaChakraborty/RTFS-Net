import torch
import torch.nn as nn
import torch.nn.functional as F

from .cnn_layers import ConvNormAct
from .attention import GlobalAttention


class InjectionMultiSum(nn.Module):
    def __init__(
        self,
        in_chan: int,
        hid_chan: int,
        kernel_size: int,
    ):
        super(InjectionMultiSum, self).__init__()
        self.in_chan = in_chan
        self.hid_chan = hid_chan
        self.kernel_size = kernel_size

        self.groups = in_chan if in_chan == hid_chan else 1

        self.local_embedding = ConvNormAct(
            in_chan=self.in_chan,
            out_chan=self.hid_chan,
            kernel_size=self.kernel_size,
            groups=self.groups,
            padding=((self.kernel_size - 1) // 2),
            norm_type="gLN",
            bias=False,
        )
        self.global_embedding = ConvNormAct(
            in_chan=self.in_chan,
            out_chan=self.hid_chan,
            kernel_size=self.kernel_size,
            groups=self.groups,
            padding=((self.kernel_size - 1) // 2),
            norm_type="gLN",
            bias=False,
        )
        self.global_gate = ConvNormAct(
            in_chan=self.in_chan,
            out_chan=self.hid_chan,
            kernel_size=self.kernel_size,
            groups=self.groups,
            padding=((self.kernel_size - 1) // 2),
            norm_type="gLN",
            act_type="Sigmoid",
            bias=False,
        )

    def forward(self, local_features, global_features):
        length = local_features.shape[-1]

        local_emb = self.local_embedding(local_features)
        global_emb = F.interpolate(self.global_embedding(global_features), size=length, mode="nearest")
        gate = F.interpolate(self.global_gate(global_features), size=length, mode="nearest")

        injection_sum = local_emb * gate + global_emb

        return injection_sum


class TDANetBlock(nn.Module):
    """
    This class defines the block which performs successive downsampling and
    upsampling in order to be able to analyze the input features in multiple
    resolutions.
    """

    def __init__(
        self,
        in_chan: int,
        hid_chan: int,
        kernel_size: int = 5,
        stride: int = 2,
        upsampling_depth: int = 4,
        n_head: int = 8,
        dropout: int = 0.1,
        drop_path: int = 0.1,
    ):
        super(TDANetBlock, self).__init__()
        self.in_chan = in_chan
        self.hid_chan = hid_chan
        self.kernel_size = kernel_size
        self.stride = stride
        self.upsampling_depth = upsampling_depth
        self.n_head = n_head
        self.dropout = dropout
        self.drop_path = drop_path

        self.projection = ConvNormAct(self.in_chan, self.hid_chan, 1, norm_type="gLN", act_type="PReLU")
        self.downsample_layers = self.__build_downsample_layers()
        self.fusion_layers = self.__build_fusion_layers()
        self.concat_layers = self.__build_concat_layers()
        self.residual_conv = nn.Conv1d(hid_chan, in_chan, 1)

        self.globalatt = GlobalAttention(
            in_chan=self.hid_chan,
            n_head=self.n_head,
            kernel_size=self.kernel_size,
            dropout=self.dropout,
            drop_path=self.drop_path,
        )

    def __build_downsample_layers(self):
        out = nn.ModuleList()
        for i in range(self.upsampling_depth):
            stride = 1 if i == 0 else self.stride
            out.append(
                ConvNormAct(
                    in_chan=self.hid_chan,
                    out_chan=self.hid_chan,
                    kernel_size=self.kernel_size,
                    stride=stride,
                    groups=self.hid_chan,
                    padding=(self.kernel_size - 1) // 2,
                    norm_type="gLN",
                )
            )

        return out

    def __build_fusion_layers(self):
        out = nn.ModuleList([])
        for _ in range(self.upsampling_depth):
            out.append(InjectionMultiSum(self.hid_chan, self.hid_chan, 1))

        return out

    def __build_concat_layers(self):
        out = nn.ModuleList([])
        for _ in range(self.upsampling_depth - 1):
            out.append(InjectionMultiSum(self.hid_chan, self.hid_chan, self.kernel_size))

        return out

    def forward(self, x):
        residual = x
        x_enc = self.projection(x)

        # Do the downsampling process from the previous level
        downsampled_outputs = [self.downsample_layers[0](x_enc)]
        for i in range(1, self.upsampling_depth):
            out_i = self.downsample_layers[i](downsampled_outputs[-1])
            downsampled_outputs.append(out_i)

        # global features
        shape = downsampled_outputs[-1].shape
        global_features = torch.zeros(shape, requires_grad=True, device=x_enc.device)
        for features in downsampled_outputs:
            global_features += F.adaptive_avg_pool1d(features, output_size=shape[-1])
        global_features = self.globalatt(global_features)  # [B, N, T]

        x_fused = []
        # Gather them now in reverse order
        for i in range(self.upsampling_depth):
            local_features = downsampled_outputs[i]
            x_fused.append(self.fusion_layers[i](local_features, global_features))

        expanded = self.concat_layers[i](x_fused[-2], x_fused[-1])
        for i in range(self.upsampling_depth - 3, -1, -1):
            expanded = self.concat_layers[i](x_fused[i], expanded)

        out = self.residual_conv(expanded) + residual

        return out


class TDANet(nn.Module):
    def __init__(
        self,
        in_chan: int,
        hid_chan: int,
        kernel_size: int = 5,
        stride: int = 2,
        upsampling_depth: int = 4,
        n_head: int = 8,
        dropout: float = 0.1,
        drop_path: float = 0.1,
        repeats: int = 4,
        shared: bool = False,
        *args,
        **kwargs,
    ):
        super(TDANet, self).__init__()
        print("Unused args: ", args)
        print("Unused kwargs: ", kwargs)
        self.in_chan = in_chan
        self.hid_chan = hid_chan
        self.kernel_size = kernel_size
        self.stride = stride
        self.upsampling_depth = upsampling_depth
        self.n_head = n_head
        self.dropout = dropout
        self.drop_path = drop_path
        self.repeats = repeats
        self.shared = shared

        self.tdanet = self.__build_tdanet()
        self.concat_block = self.__build_concat_block()

    def __build_tdanet(self):
        if self.shared:
            out = TDANetBlock(
                in_chan=self.in_chan,
                hid_chan=self.hid_chan,
                kernel_size=self.kernel_size,
                stride=self.stride,
                upsampling_depth=self.upsampling_depth,
                n_head=self.n_head,
                dropout=self.dropout,
                drop_path=self.drop_path,
            )
        else:
            out = nn.ModuleList()
            for _ in range(self.repeats):
                out.append(
                    TDANetBlock(
                        in_chan=self.in_chan,
                        hid_chan=self.hid_chan,
                        kernel_size=self.kernel_size,
                        stride=self.stride,
                        upsampling_depth=self.upsampling_depth,
                        n_head=self.n_head,
                        dropout=self.dropout,
                        drop_path=self.drop_path,
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

    def get_frcnn_block(self, i):
        if self.shared:
            return self.tdanet
        else:
            return self.tdanet[i]

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
