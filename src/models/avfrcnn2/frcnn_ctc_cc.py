###
# Author: Kai Li
# Date: 2022-02-17 11:32:01
# Email: lk21@mails.tsinghua.edu.cn
# LastEditTime: 2022-02-17 11:48:10
###
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...layers.normalizations import get as get_norm
from ...layers import (
    normalizations,
    activations,
    Conv1DBlock,
    ConvNorm,
    Video1DConv,
    Concat,
    FRCNNBlock,
    make_enc_dec,
    MultiHeadedSelfAttentionModule
)

def cal_padding(input_size, kernel_size=1, stride=1, dilation=1):
    return (kernel_size - input_size + (kernel_size-1)*(dilation-1) \
        + stride*(input_size-1)) // 2

class ConvNormAct(nn.Module):
    def __init__(
            self,
            in_chan,
            out_chan,
            kernel_size,
            stride=1,
            groups=1,
            dilation=1,
            padding=0,
            norm_type="BatchNorm1d",
            act_type=None,
        ):
        super(ConvNormAct, self).__init__()
        self.conv = nn.Conv1d(
            in_chan,
            out_chan,
            kernel_size,
            stride=stride,
            dilation=dilation,
            padding=padding,
            bias=True,
            groups=groups,
        )
        # self.norm = getattr(nn, norm_type)(out_chan)
        if hasattr(nn, norm_type):
            self.norm = getattr(nn, norm_type)(out_chan)
        else:
            self.norm = get_norm(norm_type)(out_chan)
        self.act = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        if self.act:
            return self.act(x)
        else:
            return x


class CTCBlock(nn.Module):
    def __init__(self, 
                 in_chan=128,
                 depth=4,
                 norm_type="BatchNorm1d",
                 act_type="PReLU",
                 kernel_size=5,
                 num_idx=1):
        super().__init__()
        self.fusion_down = ConvNorm(
                                    in_chan,
                                    in_chan,
                                    kernel_size=kernel_size,
                                    stride=2**(num_idx+1),
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

    def forward(self, curr, curr_down, fus=None):
        if self.num_idx == 0:
            curr_fus = curr
            fus = curr_fus
            fus_down = self.fusion_down(fus)
        if self.num_idx != 0:
            curr_fus = F.interpolate(curr, size=fus.shape[-1], mode="nearest")
            fus = self.fusion(torch.cat((curr_fus, fus), dim=1))
            fus_down = self.fusion_down(fus)
        if self.num_idx != self.depth - 1:
            curr = self.current_fusion(torch.cat((curr_down, fus_down), dim=1))
        if self.num_idx == self.depth - 1:
            curr = self.lastout(fus)
        
        return curr, fus

class FRCNNBlock(nn.Module):
    """[summary]

                   [spp_dw_3] --------\        ...-->\ 
                        /              \             \ 
                   [spp_dw_2] --------> [c] -PC.N.A->\ 
                        /              /             \ 
                   [spp_dw_1] -DC.N.--/        ...-->\ 
                        /                            \ 
    x -> [proj] -> [spp_dw_0]                  ...--> [c] -PC.N.A.PC--[(dropout)]--> 
     \                                                                           / 
      \-------------------------------------------------------------------------/

    Args:
        in_chan (int, optional): [description]. Defaults to 128.
        out_chan (int, optional): [description]. Defaults to 512.
        depth (int, optional): [description]. Defaults to 4.
        norm_type (str, optional): [description]. Defaults to "BatchNorm1d".
        act_type (str, optional): [description]. Defaults to "PReLU".
    """
    def __init__(
            self, 
            in_chan=128, 
            out_chan=512, 
            depth=4, 
            dropout=-1,
            norm_type="BatchNorm1d", 
            act_type="PReLU",
            dilation=1,
            kernel_size=5
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
            ConvNormAct(out_chan*depth, out_chan, kernel_size=1, norm_type=norm_type, act_type=act_type),
            nn.Conv1d(out_chan, in_chan, 1)
        )
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else None


    def _build_downsample_layers(self):
        out = nn.ModuleList()
        out.append(
            ConvNormAct(self.out_chan, self.out_chan, kernel_size=self.kernel_size, 
                groups=self.out_chan, padding=((self.kernel_size - 1) // 2) * 1, 
                norm_type=self.norm_type, act_type=None
            )
        )
        # ----------Down Sample Layer----------
        for _ in range(1, self.depth):
            out.append(
                ConvNormAct(self.out_chan, self.out_chan, kernel_size=self.kernel_size, stride=2,
                    groups=self.out_chan, padding=((self.kernel_size - 1) // 2) * 1, 
                    norm_type=self.norm_type, act_type=None
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
                        ConvNormAct(self.out_chan, self.out_chan, kernel_size=self.kernel_size, stride=2,
                            groups=self.out_chan, padding=((self.kernel_size - 1) // 2) * 1,
                            norm_type=self.norm_type, act_type=None
                        )
                    )
            out.append(fuse_layer)
        return out


    def _build_concat_layers(self):
        out = nn.ModuleList()
        for i in range(self.depth):
            if i == 0 or i == self.depth - 1:
                out.append(
                    ConvNormAct(self.out_chan*2, self.out_chan, kernel_size=1, 
                        norm_type=self.norm_type, act_type=self.act_type
                    )
                )
            else:
                out.append(
                    ConvNormAct(self.out_chan*3, self.out_chan, kernel_size=1,
                        norm_type=self.norm_type, act_type=self.act_type
                    )
                )
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
                    self.fuse_layers[i][0](output[i - 1]) if i - 1 >= 0 \
                        else torch.Tensor().to(x.device),
                    output[i],
                    F.interpolate(output[i + 1], size=T, mode="nearest") if i + 1 < self.depth
                        else torch.Tensor().to(x.device),
                ),
                dim=1,
            )
            x_fuse.append(self.concat_layer[i](y))

        return x_fuse

class FRCNNBlockCTC(nn.Module):
    def __init__(
        self, in_chan=128, out_chan=512, upsampling_depth=4, norm_type="BatchNorm1d", act_type="PReLU", kernel_size=5
    ):
        super().__init__()
        self.depth = upsampling_depth
        
        self.spp_dw = FRCNNBlock(in_chan=out_chan, out_chan=out_chan, depth=upsampling_depth, norm_type=norm_type, act_type=act_type, kernel_size=kernel_size)

        self.ctc = nn.ModuleList([CTCBlock(out_chan, upsampling_depth, norm_type, act_type, kernel_size, i) for i in range(upsampling_depth)])
        
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
        x_fuse = self.spp_dw(x)
        # import pdb; pdb.set_trace()
        fus = None
        for idx in range(self.depth):
            if idx != self.depth - 1:
                x_fuse[idx+1], fus = self.ctc[idx](x_fuse[idx], x_fuse[idx+1], fus)
            if idx == self.depth - 1:
                output1, fus = self.ctc[idx](x_fuse[idx], None, fus)
        expanded = self.res_conv(output1)
        return expanded + residual


class FRCNNCTCCC(nn.Module):
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

    def __init__(self,
                 in_chan=128, 
                 out_chan=512, 
                 depth=4, 
                 iter=4, 
                 dilations=None,
                 shared=False, 
                 dropout=-1,
                 norm_type="BatchNorm1d",
                 act_type="prelu",
                 kernel_size=5):
        super(FRCNNCTCCC, self).__init__()
        self.in_chan = in_chan
        self.out_chan = out_chan
        self.depth = depth
        self.iter = iter
        self.dilations = [1,]*iter if dilations is None else dilations
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
            return self._build_frcnnblock(self.in_chan, self.out_chan, self.depth,
                        self.norm_type, self.act_type, kernel_size=self.kernel_size)
        else:
            out = nn.ModuleList()
            for i in range(self.iter):
                out.append(
                    self._build_frcnnblock(self.in_chan, self.out_chan, self.depth,
                        self.norm_type, self.act_type, kernel_size=self.kernel_size))
            return out


    def _build_frcnnblock(self, *args, dilation=1, **kwargs):
        if dilation == 1:
            return FRCNNBlockCTC(*args, **kwargs)


    def _build_concat_block(self):
        if self.shared:
            return nn.Sequential(
                nn.Conv1d(self.in_chan, self.in_chan,
                          1, 1, groups=self.in_chan),
                nn.PReLU())
        else:
            out = nn.ModuleList([None])
            for _ in range(self.iter-1):
                out.append(nn.Sequential(
                    nn.Conv1d(self.in_chan, self.in_chan,
                              1, 1, groups=self.in_chan),
                    nn.PReLU()))
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