import math
import torch
import torch.nn as nn

from ..layers import DualPathRNN, MultiHeadSelfAttention2D, ConvNormAct


class GridNetBlock(nn.Module):
    def __getitem__(self, key):
        return getattr(self, key)

    def __init__(
        self,
        in_chan: int,
        rnn_1_conf: dict,
        rnn_2_conf: dict,
        attention_conf: dict,
        concat_block: bool = False,
    ):
        super(GridNetBlock, self).__init__()
        self.in_chan = in_chan
        self.rnn_1_conf = rnn_1_conf
        self.rnn_2_conf = rnn_2_conf
        self.attention_conf = attention_conf
        self.concat_block = concat_block

        if self.concat_block:
            self.gateway = ConvNormAct(
                in_chan=self.in_chan,
                out_chan=self.in_chan,
                kernel_size=1,
                groups=self.in_chan,
                act_type=self.attention_conf.get("act_type", "PReLU"),
                is2d=True,
            )
        self.first_rnn = DualPathRNN(in_chan=self.in_chan, **self.rnn_1_conf)
        self.second_rnn = DualPathRNN(in_chan=self.in_chan, **self.rnn_2_conf)
        self.attention = MultiHeadSelfAttention2D(in_chan=self.in_chan, **self.attention_conf)

    def forward(self, x: torch.Tensor):
        if self.concat_block:
            x = self.gateway(x)
        x = self.first_rnn(x)
        x = self.second_rnn(x)
        x = self.attention(x)
        return x


class GridNet(nn.Module):
    def __init__(
        self,
        in_chan: int = -1,
        rnn_1_conf: dict = dict(),
        rnn_2_conf: dict = dict(),
        attention_conf: dict = dict(),
        repeats: int = 4,
        shared: bool = False,
        *args,
        **kwargs,
    ):
        super(GridNet, self).__init__()
        self.in_chan = in_chan
        self.rnn_1_conf = rnn_1_conf
        self.rnn_2_conf = rnn_2_conf
        self.attention_conf = attention_conf
        self.repeats = repeats
        self.shared = shared

        self.blocks = self.__build_blocks()

    def __build_blocks(self):
        clss = GridNetBlock if self.in_chan > 0 else nn.Identity
        if self.shared:
            out = clss(
                in_chan=self.in_chan,
                rnn_1_conf=self.rnn_1_conf,
                rnn_2_conf=self.rnn_2_conf,
                attention_conf=self.attention_conf,
            )
        else:
            out = nn.ModuleList()
            for _ in range(self.repeats):
                out.append(
                    clss(
                        in_chan=self.in_chan,
                        rnn_1_conf=self.rnn_1_conf,
                        rnn_2_conf=self.rnn_2_conf,
                        attention_conf=self.attention_conf,
                    )
                )

        return out

    def get_block(self, i: int):
        if self.shared:
            return self.blocks
        else:
            return self.blocks[i]

    def forward(self, x: torch.Tensor):
        residual = x
        for i in range(self.repeats):
            x = self.get_block(i)((x + residual) if i > 0 else x)
        return x
