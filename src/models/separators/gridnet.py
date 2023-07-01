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
    ):
        super(GridNetBlock, self).__init__()
        self.in_chan = in_chan
        self.rnn_1_conf = rnn_1_conf
        self.rnn_2_conf = rnn_2_conf
        self.attention_conf = attention_conf

        self.first_rnn = DualPathRNN(in_chan=self.in_chan, **self.rnn_1_conf)
        self.second_rnn = DualPathRNN(in_chan=self.in_chan, **self.rnn_2_conf)
        self.attention = MultiHeadSelfAttention2D(in_chan=self.in_chan, **self.attention_conf)

    def forward(self, x: torch.Tensor):
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
        concat_first: bool = False,
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
        self.concat_first = concat_first

        self.blocks = self.__build_blocks()
        self.concat_block = self.__build_concat_block()

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

    def __build_concat_block(self):
        clss = ConvNormAct if (self.in_chan > 0) and ((self.repeats > 1) or self.concat_first) else nn.Identity
        if self.shared:
            out = clss(
                in_chan=self.in_chan,
                out_chan=self.in_chan,
                kernel_size=1,
                groups=self.in_chan,
                act_type="PReLU",
                is2d=True,
            )
        else:
            out = nn.ModuleList() if self.concat_first else nn.ModuleList([None])
            for _ in range(self.repeats) if self.concat_first else range(self.repeats - 1):
                out.append(
                    clss(
                        in_chan=self.in_chan,
                        out_chan=self.in_chan,
                        kernel_size=1,
                        groups=self.in_chan,
                        act_type="PReLU",
                        is2d=True,
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
