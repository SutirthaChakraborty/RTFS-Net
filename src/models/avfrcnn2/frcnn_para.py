import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
from ...layers.normalizations import get as get_norm
from ...layers.activations import get as get_act

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
        self.last = nn.ModuleList([nn.Conv1d(out_chan, in_chan, 1) for i in range(self.depth)])
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

        # concat and shortcut
        x = [self.last[i](x_fuse[i]) for i in range(self.depth)]
        # dropout
        if self.dropout_layer:
            x = self.dropout_layer(x)

        return [output[i] + x[i]  for i in range(self.depth)]

class FRCNNBParalock(nn.Module):
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
            kernel_size=5,
        ):
        super(FRCNNBParalock, self).__init__()
        self.in_chan = in_chan
        self.out_chan = out_chan
        self.depth = depth
        self.norm_type = norm_type
        self.act_type = act_type
        self.dilation = dilation
        self.kernel_size = kernel_size

        self.proj = nn.ModuleList([
            ConvNormAct(in_chan, out_chan, kernel_size=1, norm_type=norm_type, act_type=act_type)
            for i in range(self.depth)]
        )
        self.spp_dw = self._build_downsample_layers()
        self.fuse_layers = self._build_fusion_layers()
        self.concat_layer = self._build_concat_layers()
        
        self.last = nn.ModuleList([nn.Sequential(get_norm("BatchNorm1d")(out_chan), nn.PReLU(), nn.Conv1d(out_chan, in_chan, 1)) for i in range(self.depth)])
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
        residual = x
        output = [self.proj[i](x[i]) for i in range(self.depth)]
        # lateral connection
        x_fuse = []
        for i in range(self.depth):
            T = output[i].shape[-1]
            y = torch.cat(
                (
                    self.fuse_layers[i][0](output[i - 1]) if i - 1 >= 0 \
                        else torch.Tensor().to(x[0].device),
                    output[i],
                    F.interpolate(output[i + 1], size=T, mode="nearest") if i + 1 < self.depth
                        else torch.Tensor().to(x[0].device),
                ),
                dim=1,
            )
            x_fuse.append(self.concat_layer[i](y))


        # concat and shortcut
        concate = [self.last[i](x_fuse[i]) for i in range(self.depth)]
        # dropout
        if self.dropout_layer:
            concate = self.dropout_layer(concate)
        return [residual[i] + concate[i]  for i in range(self.depth)]

class FRCNNBlockDilation(FRCNNBlock):
    """FRCNN with dilation instead of stride

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
    def __init__(self, *args, dilation=2, **kwargs):
        super(FRCNNBlockDilation, self).__init__(*args, **kwargs)
        self.dilation = dilation

    def _build_downsample_layers(self):
        out = nn.ModuleList()
        out.append(
            ConvNormAct(self.out_chan, self.out_chan, kernel_size=self.kernel_size, 
                groups=self.out_chan, padding=cal_padding(0, self.kernel_size, 1, 1), 
                norm_type=self.norm_type, act_type=None
            )
        )
        # ----------Down Sample Layer----------
        for _ in range(1, self.depth):
            out.append(
                ConvNormAct(self.out_chan, self.out_chan, kernel_size=self.kernel_size, stride=1,
                    dilation=self.dilation,
                    groups=self.out_chan, padding=cal_padding(0, self.kernel_size, 1, self.dilation), 
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
                        ConvNormAct(self.out_chan, self.out_chan, kernel_size=self.kernel_size, stride=1,
                            dilation=1,
                            groups=self.out_chan, padding=cal_padding(0, self.kernel_size, 1, 1),
                            norm_type=self.norm_type, act_type=None
                        )
                    )
            out.append(fuse_layer)
        return out





class FRCNNPARA(nn.Module):
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
                 act_type="PReLU",
                 kernel_size=5):
        super(FRCNNPARA, self).__init__()
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
            return self._build_frcnnblock(self.in_chan, self.out_chan, self.depth, self.dropout,
                        self.norm_type, self.act_type, dilation=self.dilations,
                        kernel_size=self.kernel_size)
        else:
            out = nn.ModuleList()
            for i in range(self.iter):
                if i == 0:
                    out.append(
                        self._build_frcnnblock(self.in_chan, self.out_chan, self.depth, self.dropout,
                            self.norm_type, self.act_type, dilation=self.dilations[i],
                            kernel_size=self.kernel_size, first=True))
                else:
                    out.append(
                        self._build_frcnnblock(self.in_chan, self.out_chan, self.depth, self.dropout,
                            self.norm_type, self.act_type, dilation=self.dilations[i],
                            kernel_size=self.kernel_size))
            return out


    def _build_frcnnblock(self, *args, dilation=1, first=False, **kwargs):
        if first == True:
            return FRCNNBlock(*args, **kwargs)
        else:
            return FRCNNBParalock(*args, **kwargs)


    def _build_concat_block(self):
        if self.shared:
            return nn.Sequential(
                nn.Conv1d(self.in_chan, self.in_chan,
                          1, 1, groups=self.in_chan),
                nn.PReLU())
        else:
            out = nn.ModuleList([None])
            for _ in range(self.iter-1):
                out.append(nn.ModuleList([nn.Sequential(
                    nn.Conv1d(self.in_chan, self.in_chan,
                              1, 1, groups=self.in_chan),
                    nn.PReLU()) for i in range(self.depth)]))
            return out

    def forward(self, x):
        # x: shape (B, C, T)
        res = None
        for i in range(self.iter):
            frcnn = self.get_frcnn_block(i)
            concat_block = self.get_concat_block(i)
            if i == 0:
                x = frcnn(x)
                res = x
            else:
                x = frcnn([concat_block[i](res[i] + x[i]) for i in range(self.depth) ])
        return x