import torch
import torch.nn as nn
import torch.nn.functional as F

from .rnn_layers import TAC
from .cnn_layers import ConvNormAct
from .attention import GlobalAttention, GlobalAttention2D


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
            padding=((self.kernel_size - 1) // 2),
            norm_type=self.norm_type,
            bias=False,
            is2d=self.is2d,
        )
        self.global_embedding = ConvNormAct(
            in_chan=self.in_chan,
            out_chan=self.hid_chan,
            kernel_size=self.kernel_size,
            groups=self.groups,
            padding=((self.kernel_size - 1) // 2),
            norm_type=self.norm_type,
            bias=False,
            is2d=self.is2d,
        )
        self.global_gate = ConvNormAct(
            in_chan=self.in_chan,
            out_chan=self.hid_chan,
            kernel_size=self.kernel_size,
            groups=self.groups,
            padding=((self.kernel_size - 1) // 2),
            norm_type=self.norm_type,
            act_type="Sigmoid",
            bias=False,
            is2d=self.is2d,
        )

    def forward(self, local_features: torch.Tensor, global_features: torch.Tensor):
        length = local_features.shape[-(len(local_features.shape) // 2) :]

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
        norm_type: str = "gLN",
        act_type: str = "PReLU",
        upsampling_depth: int = 4,
        n_head: int = 8,
        dropout: int = 0.1,
        drop_path: int = 0.1,
        group_size: int = 1,
        is2d: bool = False,
    ):
        super(TDANetBlock, self).__init__()
        self.in_chan = in_chan
        self.hid_chan = hid_chan
        self.kernel_size = kernel_size
        self.stride = stride
        self.norm_type = norm_type
        self.act_type = act_type
        self.upsampling_depth = upsampling_depth
        self.n_head = n_head
        self.dropout = dropout
        self.drop_path = drop_path
        self.group_size = group_size
        self.is2d = is2d

        self.att = GlobalAttention2D if self.is2d else GlobalAttention
        self.pool = F.adaptive_avg_pool2d if self.is2d else F.adaptive_avg_pool1d

        self.projection = ConvNormAct(
            in_chan=self.in_chan // self.group_size,
            out_chan=self.hid_chan // self.group_size,
            kernel_size=1,
            norm_type=self.norm_type,
            act_type=self.act_type,
            is2d=self.is2d,
        )
        self.downsample_layers = self.__build_downsample_layers()
        self.fusion_layers = self.__build_fusion_layers()
        self.concat_layers = self.__build_concat_layers()
        self.residual_conv = ConvNormAct(self.hid_chan // self.group_size, self.in_chan // self.group_size, 1, is2d=self.is2d)

        self.globalatt = self.att(
            in_chan=self.hid_chan // self.group_size,
            kernel_size=self.kernel_size,
            n_head=self.n_head,
            dropout=self.dropout,
            drop_path=self.drop_path,
        )

    def __build_downsample_layers(self):
        out = nn.ModuleList()
        for i in range(self.upsampling_depth):
            stride = 1 if i == 0 else self.stride
            out.append(
                ConvNormAct(
                    in_chan=self.hid_chan // self.group_size,
                    out_chan=self.hid_chan // self.group_size,
                    kernel_size=self.kernel_size,
                    stride=stride,
                    groups=self.hid_chan // self.group_size,
                    padding=(self.kernel_size - 1) // 2,
                    norm_type=self.norm_type,
                    is2d=self.is2d,
                )
            )

        return out

    def __build_fusion_layers(self):
        out = nn.ModuleList([])
        for _ in range(self.upsampling_depth):
            out.append(
                InjectionMultiSum(
                    in_chan=self.hid_chan // self.group_size,
                    hid_chan=self.hid_chan // self.group_size,
                    kernel_size=1,
                    norm_type=self.norm_type,
                    is2d=self.is2d,
                )
            )

        return out

    def __build_concat_layers(self):
        out = nn.ModuleList([])
        for _ in range(self.upsampling_depth - 1):
            out.append(
                InjectionMultiSum(
                    in_chan=self.hid_chan // self.group_size,
                    hid_chan=self.hid_chan // self.group_size,
                    kernel_size=self.kernel_size,
                    norm_type=self.norm_type,
                    is2d=self.is2d,
                )
            )

        return out

    def forward(self, x):
        # x: B, C, T, (F)
        residual = x
        x_enc = self.projection(x)

        # bottom-up
        downsampled_outputs = [self.downsample_layers[0](x_enc)]
        for i in range(1, self.upsampling_depth):
            out_i = self.downsample_layers[i](downsampled_outputs[-1])
            downsampled_outputs.append(out_i)

        # global features
        shape = downsampled_outputs[-1].shape
        global_features = torch.zeros(shape, requires_grad=True, device=x_enc.device)
        for features in downsampled_outputs:
            global_features = global_features + self.pool(features, output_size=shape[-(len(shape) // 2) :])
        global_features = self.globalatt(global_features)  # B, N, T, (F)

        x_fused = []
        # Gather them now in reverse order
        for i in range(self.upsampling_depth):
            local_features = downsampled_outputs[i]
            x_fused.append(self.fusion_layers[i](local_features, global_features))

        expanded = self.concat_layers[-1](x_fused[-2], x_fused[-1])
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
        norm_type: str = "gLN",
        act_type: str = "PReLU",
        upsampling_depth: int = 4,
        repeats: int = 4,
        shared: bool = False,
        n_head: int = 8,
        dropout: float = 0.1,
        drop_path: float = 0.1,
        group_size: int = 1,
        tac_multiplier: int = 2,
        is2d: bool = False,
        *args,
        **kwargs,
    ):
        super(TDANet, self).__init__()
        self.in_chan = in_chan
        self.hid_chan = hid_chan
        self.kernel_size = kernel_size
        self.stride = stride
        self.norm_type = norm_type
        self.act_type = act_type
        self.upsampling_depth = upsampling_depth
        self.repeats = repeats
        self.shared = shared
        self.n_head = n_head
        self.dropout = dropout
        self.drop_path = drop_path
        self.group_size = group_size
        self.tac_multiplier = tac_multiplier
        self.is2d = is2d

        self.tac = self.__build_tac()
        self.blocks = self.__build_blocks()
        self.concat_block = self.__build_concat_block()

    def __build_tac(self):
        if self.shared:
            out = (
                TAC(self.in_chan // self.group_size, self.hid_chan * self.tac_multiplier // self.group_size)
                if self.group_size > 1
                else nn.Identity()
            )
        else:
            out = nn.ModuleList()
            for _ in range(self.repeats):
                out.append(
                    TAC(self.in_chan // self.group_size, self.hid_chan * self.tac_multiplier // self.group_size)
                    if self.group_size > 1
                    else nn.Identity()
                )

        return out

    def __build_blocks(self):
        if self.shared:
            out = TDANetBlock(
                in_chan=self.in_chan,
                hid_chan=self.hid_chan,
                kernel_size=self.kernel_size,
                stride=self.stride,
                norm_type=self.norm_type,
                act_type=self.act_type,
                upsampling_depth=self.upsampling_depth,
                n_head=self.n_head,
                dropout=self.dropout,
                drop_path=self.drop_path,
                group_size=self.group_size,
                is2d=self.is2d,
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
                        norm_type=self.norm_type,
                        act_type=self.act_type,
                        upsampling_depth=self.upsampling_depth,
                        n_head=self.n_head,
                        dropout=self.dropout,
                        drop_path=self.drop_path,
                        group_size=self.group_size,
                        is2d=self.is2d,
                    )
                )

        return out

    def __build_concat_block(self):
        if self.shared:
            out = ConvNormAct(
                in_chan=self.in_chan // self.group_size,
                out_chan=self.in_chan // self.group_size,
                kernel_size=1,
                groups=self.in_chan // self.group_size,
                act_type=self.act_type,
                is2d=self.is2d,
            )
        else:
            out = nn.ModuleList([None])
            for _ in range(self.repeats - 1):
                out.append(
                    ConvNormAct(
                        in_chan=self.in_chan // self.group_size,
                        out_chan=self.in_chan // self.group_size,
                        kernel_size=1,
                        groups=self.in_chan // self.group_size,
                        act_type=self.act_type,
                        is2d=self.is2d,
                    )
                )

        return out

    def get_tac(self, i: int):
        if self.shared:
            return self.tac
        else:
            return self.tac[i]

    def get_block(self, i: int):
        if self.shared:
            return self.blocks
        else:
            return self.blocks[i]

    def get_concat_block(self, i: int):
        if self.shared:
            return self.concat_block
        else:
            return self.concat_block[i]

    def forward(self, x: torch.Tensor):
        # x: shape (B, C, T)
        batch_size = x.shape[0]
        T = x.shape[-(len(x.shape) // 2) :]

        res = x.view(batch_size * self.group_size, -1, *T)

        for i in range(self.repeats):
            x = self.get_tac(i)(x.view(batch_size, self.group_size, -1, *T)).view(batch_size * self.group_size, -1, *T)
            frcnn = self.get_block(i)
            concat_block = self.get_concat_block(i)
            if i == 0:
                x = frcnn(x)
            else:
                x = frcnn(concat_block(res + x))

        return x.view(batch_size, -1, *T)
