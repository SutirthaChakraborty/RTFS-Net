import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import layers
from ..layers import ConvNormAct


class DPTNetBlock(nn.Module):
    def __init__(
        self,
        in_chan: int,
        hid_chan: int,
        kernel_size: int = 5,
        norm_type: str = "gLN",
        act_type: str = "PReLU",
        attention_type: str = None,
        n_head: int = 8,
        attention_ks: int = None,
        dropout: int = 0.1,
        is2d: bool = False,
    ):
        super(DPTNetBlock, self).__init__()
        self.in_chan = in_chan
        self.hid_chan = hid_chan
        self.kernel_size = kernel_size
        self.norm_type = norm_type
        self.act_type = act_type
        self.attention_type = attention_type
        self.n_head = n_head
        self.dropout = dropout
        self.is2d = is2d

        self.attention_ks = kernel_size if attention_ks is None else attention_ks
        if attention_type is None:
            self.attention_type = "GlobalAttention2D" if self.is2d else "GlobalAttention"

        self.att = layers.get(self.attention_type)

        self.projection = ConvNormAct(
            in_chan=self.in_chan,
            out_chan=self.hid_chan,
            kernel_size=1,
            norm_type=self.norm_type,
            act_type=self.act_type,
            is2d=self.is2d,
        )
        self.globalatt = self.att(
            in_chan=self.hid_chan,
            kernel_size=self.attention_ks,
            n_head=self.n_head,
            dropout=self.dropout,
        )
        self.residual_conv = ConvNormAct(
            in_chan=self.hid_chan,
            out_chan=self.in_chan,
            kernel_size=1,
            is2d=self.is2d,
        )

    def forward(self, x):
        # x: B, C, T, (F)
        residual = x
        x_enc = self.projection(x)

        expanded = self.globalatt(x_enc)

        out = self.residual_conv(expanded) + residual

        return out


class DPTNet(nn.Module):
    def __init__(
        self,
        in_chan: int = -1,
        hid_chan: int = -1,
        kernel_size: int = 5,
        norm_type: str = "gLN",
        act_type: str = "PReLU",
        repeats: int = 4,
        shared: bool = False,
        attention_type: str = None,
        n_head: int = 8,
        attention_ks: int = None,
        dropout: float = 0.1,
        is2d: bool = False,
        *args,
        **kwargs,
    ):
        super(DPTNet, self).__init__()
        self.in_chan = in_chan
        self.hid_chan = hid_chan
        self.kernel_size = kernel_size
        self.norm_type = norm_type
        self.act_type = act_type
        self.repeats = repeats
        self.shared = shared
        self.attention_type = attention_type
        self.n_head = n_head
        self.attention_ks = attention_ks
        self.dropout = dropout
        self.is2d = is2d

        self.blocks = self.__build_blocks()
        self.concat_block = self.__build_concat_block()

    def __build_blocks(self):
        clss = DPTNetBlock if self.in_chan > 0 else nn.Identity
        if self.shared:
            out = clss(
                in_chan=self.in_chan,
                hid_chan=self.hid_chan,
                kernel_size=self.kernel_size,
                norm_type=self.norm_type,
                act_type=self.act_type,
                attention_type=self.attention_type,
                n_head=self.n_head,
                attention_ks=self.attention_ks,
                dropout=self.dropout,
                is2d=self.is2d,
            )
        else:
            out = nn.ModuleList()
            for _ in range(self.repeats):
                out.append(
                    clss(
                        in_chan=self.in_chan,
                        hid_chan=self.hid_chan,
                        kernel_size=self.kernel_size,
                        norm_type=self.norm_type,
                        act_type=self.act_type,
                        attention_type=self.attention_type,
                        n_head=self.n_head,
                        attention_ks=self.attention_ks,
                        dropout=self.dropout,
                        is2d=self.is2d,
                    )
                )

        return out

    def __build_concat_block(self):
        clss = ConvNormAct if (self.in_chan > 0) and (self.repeats > 1 or self.is2d) else nn.Identity
        if self.shared:
            out = clss(
                in_chan=self.in_chan,
                out_chan=self.in_chan,
                kernel_size=1,
                groups=self.in_chan,
                act_type=self.act_type,
                norm_type=self.norm_type,
                is2d=self.is2d,
            )
        else:
            out = nn.ModuleList([None])
            for _ in range(self.repeats - 1):
                out.append(
                    clss(
                        in_chan=self.in_chan,
                        out_chan=self.in_chan,
                        kernel_size=1,
                        groups=self.in_chan,
                        act_type=self.act_type,
                        norm_type=self.norm_type,
                        is2d=self.is2d,
                    )
                )

        return out

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
        residual = x
        for i in range(self.repeats):
            x = self.get_concat_block(i)(x + residual) if i > 0 else x
            x = self.get_block(i)(x)
        return x
