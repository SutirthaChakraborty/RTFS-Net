import torch
import torch.nn as nn
import torch.nn.functional as F
from ...layers.normalizations import get as get_norm
from ...layers import (
    normalizations,
    activations,
    Conv1DBlock,
    ConvNormAct,
    ConvNorm,
    Video1DConv,
    Concat,
    FRCNNBlock,
    make_enc_dec,
    MultiHeadedSelfAttentionModule,
)


def cal_padding(input_size, kernel_size=1, stride=1, dilation=1):
    return (kernel_size - input_size + (kernel_size - 1) * (dilation - 1) + stride * (input_size - 1)) // 2


class CTCBlock(nn.Module):
    def __init__(
        self,
        in_chan=128,
        depth=4,
        norm_type="BatchNorm1d",
        act_type="PReLU",
        kernel_size=5,
        num_idx=1,
    ):
        super().__init__()
        if num_idx != depth - 1:
            self.current_down = ConvNorm(
                in_chan,
                in_chan,
                kernel_size=kernel_size,
                stride=2,
                groups=in_chan,
                dilation=1,
                padding=((kernel_size - 1) // 2) * 1,
                norm_type=norm_type,
            )

        self.fusion_down = ConvNorm(
            in_chan,
            in_chan,
            kernel_size=kernel_size,
            stride=2 ** (num_idx + 1),
            groups=in_chan,
            dilation=1,
            padding=((kernel_size - 1) // 2) * 1,
            norm_type=norm_type,
        )
        self.current_fusion = ConvNormAct(in_chan * 2, in_chan, 1, 1, norm_type=norm_type, act_type=act_type)
        self.lastout = ConvNormAct(in_chan, in_chan, 1, 1, norm_type=norm_type, act_type=act_type)

        if num_idx != 0:
            self.fusion = ConvNormAct(in_chan * 2, in_chan, 1, 1, norm_type=norm_type, act_type=act_type)

        if num_idx == 0:
            self.current = ConvNorm(
                in_chan,
                in_chan,
                kernel_size=kernel_size,
                stride=1,
                groups=in_chan,
                dilation=1,
                padding=((kernel_size - 1) // 2) * 1,
                norm_type=norm_type,
            )
        self.num_idx = num_idx
        self.depth = depth

    def forward(self, curr, fus=None):
        if self.num_idx == 0:
            curr_fus = self.current(curr)
            fus = curr_fus
            fus_down = self.fusion_down(fus)
        if self.num_idx != 0:
            curr_fus = F.interpolate(curr, size=fus.shape[-1], mode="nearest")
            fus = self.fusion(torch.cat((curr_fus, fus), dim=1))
            fus_down = self.fusion_down(fus)
        if self.num_idx != self.depth - 1:
            curr_down = self.current_down(curr)
            curr = self.current_fusion(torch.cat((curr_down, fus_down), dim=1))
        if self.num_idx == self.depth - 1:
            curr = self.lastout(fus)

        return curr, fus


class FRCNNBlock(nn.Module):
    def __init__(
        self,
        in_chan=128,
        out_chan=512,
        upsampling_depth=4,
        norm_type="BatchNorm1d",
        act_type="PReLU",
        kernel_size=5,
    ):
        super().__init__()
        self.proj_1x1 = ConvNormAct(
            in_chan,
            out_chan,
            kernel_size=1,
            stride=1,
            groups=1,
            dilation=1,
            padding=0,
            norm_type=norm_type,
            act_type=act_type,
        )
        self.depth = upsampling_depth
        self.spp_dw = nn.ModuleList(
            [CTCBlock(out_chan, upsampling_depth, norm_type, act_type, kernel_size, i) for i in range(upsampling_depth)]
        )

        self.res_conv = nn.Conv1d(out_chan, in_chan, 1)
        # ----------parameters-------------
        self.depth = upsampling_depth

    def forward(self, x):
        """
        :param x: input feature map
        :return: transformed feature map
        """
        residual = x.clone()
        # Reduce --> project high-dimensional feature maps to low-dimensional space
        output1 = self.proj_1x1(x)
        fus = None
        for idx in range(len(self.spp_dw)):
            output1, fus = self.spp_dw[idx](output1, fus)
        expanded = self.res_conv(output1)
        return expanded + residual


class FRCNNCTC(nn.Module):
    """[summary]

    x --> FRCNNBlock --> Conv1d|PReLU -> FRCNNBlock -> Conv1d|PReLU -> FRCNNBlock -> ...
     \               /                             /                             /
      \-------------/-----------------------------/-----------------------------/

    Args:
        in_chan (int, optional): [description]. Defaults to 128.
        out_chan (int, optional): [description]. Defaults to 512.
        iter (int, optional): [description]. Defaults to 4.
        shared (bool, optional): [description]. Defaults to False.
    """

    def __init__(
        self,
        in_chan=128,
        out_chan=512,
        depth=4,
        iter=4,
        dilations=None,
        shared=False,
        dropout=-1,
        norm_type="BatchNorm1d",
        act_type="prelu",
        kernel_size=5,
    ):
        super(FRCNNCTC, self).__init__()
        self.in_chan = in_chan
        self.out_chan = out_chan
        self.depth = depth
        self.iter = iter
        self.dilations = (
            [
                1,
            ]
            * iter
            if dilations is None
            else dilations
        )
        self.shared = shared
        self.dropout = dropout
        self.norm_type = norm_type
        self.act_type = act_type
        self.kernel_size = kernel_size

        self.frcnn = self._build_frcnn()
        self.concat_block = self._build_concat_block()

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

    def _build_frcnn(self):
        if self.shared:
            return self._build_frcnnblock(
                self.in_chan,
                self.out_chan,
                self.depth,
                self.norm_type,
                self.act_type,
                kernel_size=self.kernel_size,
            )
        else:
            out = nn.ModuleList()
            for i in range(self.iter):
                out.append(
                    self._build_frcnnblock(
                        self.in_chan,
                        self.out_chan,
                        self.depth,
                        self.norm_type,
                        self.act_type,
                        kernel_size=self.kernel_size,
                    )
                )
            return out

    def _build_frcnnblock(self, *args, dilation=1, **kwargs):
        if dilation == 1:
            return FRCNNBlock(*args, **kwargs)

    def _build_concat_block(self):
        if self.shared:
            return nn.Sequential(
                nn.Conv1d(self.in_chan, self.in_chan, 1, 1, groups=self.in_chan),
                nn.PReLU(),
            )
        else:
            out = nn.ModuleList([None])
            for _ in range(self.iter - 1):
                out.append(
                    nn.Sequential(
                        nn.Conv1d(self.in_chan, self.in_chan, 1, 1, groups=self.in_chan),
                        nn.PReLU(),
                    )
                )
            return out

    def forward(self, x):
        # x: shape (B, C, T)
        res = x
        for i in range(self.iter):
            frcnn = self.get_frcnn_block(i)
            concat_block = self.get_concat_block(i)
            if i == 0:
                x = frcnn(x)
            else:
                x = frcnn(concat_block(res + x))
        return x
