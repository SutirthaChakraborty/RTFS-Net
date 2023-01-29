import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers import ConvNormAct, FRCNNBlock


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
