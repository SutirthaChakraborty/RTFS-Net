import torch
import torch.nn as nn
from ..layers import DualPathRNN, MultiHeadSelfAttention2D, ConvNormAct, BiLSTM2D, get as get_layers


class GridNetBlock(nn.Module):
    def __init__(self, in_chan: int, rnn_1_conf: dict, rnn_2_conf: dict, attention_conf: dict, *args, **kwargs):
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


class LSTM2DBlock(nn.Module):
    def __init__(self, in_chan: int, rnn_1_conf: dict, rnn_2_conf: dict, attention_conf: dict, *args, **kwargs):
        super(LSTM2DBlock, self).__init__()
        self.in_chan = in_chan
        self.rnn_1_conf = rnn_1_conf
        self.rnn_2_conf = rnn_2_conf
        self.attention_conf = attention_conf

        self.first_rnn = BiLSTM2D(in_chan=self.in_chan, **self.rnn_1_conf)
        self.second_rnn = BiLSTM2D(in_chan=self.in_chan, **self.rnn_2_conf)
        self.attention = MultiHeadSelfAttention2D(in_chan=self.in_chan, **self.attention_conf)

    def forward(self, x: torch.Tensor):
        x = self.first_rnn(x)
        x = self.second_rnn(x)
        x = self.attention(x)
        return x


class GridNetTransformerBlock(nn.Module):
    def __init__(self, in_chan: int, rnn_1_conf: dict, attention_conf: dict, *args, **kwargs):
        super(GridNetTransformerBlock, self).__init__()
        self.in_chan = in_chan
        self.rnn_1_conf = rnn_1_conf
        self.attention_conf = attention_conf

        self.rnn = DualPathRNN(in_chan=self.in_chan, **self.rnn_1_conf)
        self.mhsa = MultiHeadSelfAttention2D(in_chan=self.in_chan, **self.attention_conf)

    def forward(self, x: torch.Tensor):
        x = self.rnn(x)
        x = self.mhsa(x)
        return x


class AttentionBlock2D(nn.Module):
    def __init__(self, in_chan: int, attention_conf: dict, *args, **kwargs):
        super(AttentionBlock2D, self).__init__()
        self.in_chan = in_chan
        self.attention_conf = attention_conf
        self.att_type = attention_conf.get("att_type", "MultiHeadSelfAttention2D")

        self.attention2d = get_layers(self.att_type)(in_chan=self.in_chan, **self.attention_conf)

    def forward(self, x: torch.Tensor):
        return self.attention2d(x)


class TFGridNet(nn.Module):
    def __init__(
        self,
        in_chan: int,
        hid_chan: int,
        block_type: str,
        rnn_1_conf: dict,
        rnn_2_conf: dict,
        attention_conf: dict,
    ):
        super(TFGridNet, self).__init__()
        self.in_chan = in_chan
        self.hid_chan = hid_chan
        self.block_type = block_type
        self.rnn_1_conf = rnn_1_conf
        self.rnn_2_conf = rnn_2_conf
        self.attention_conf = attention_conf

        self.gateway = ConvNormAct(
            in_chan=self.in_chan,
            out_chan=self.in_chan,
            kernel_size=1,
            groups=self.in_chan,
            act_type=self.attention_conf.get("act_type", "PReLU"),
            is2d=True,
        )
        self.projection = ConvNormAct(
            in_chan=self.in_chan,
            out_chan=self.hid_chan,
            kernel_size=1,
            is2d=True,
        )
        self.globalatt = get(self.block_type)(
            in_chan=self.hid_chan,
            rnn_1_conf=self.rnn_1_conf,
            rnn_2_conf=self.rnn_2_conf,
            attention_conf=self.attention_conf,
        )
        self.residual_conv = ConvNormAct(
            in_chan=self.hid_chan,
            out_chan=self.in_chan,
            kernel_size=1,
            is2d=True,
        )

    def forward(self, x: torch.Tensor):
        residual = self.gateway(x)
        x = self.projection(residual)
        x = self.globalatt(x)
        x = self.residual_conv(x) + residual
        return x


class GridNet(nn.Module):
    def __init__(
        self,
        in_chan: int = -1,
        hid_chan: int = -1,
        block_type: str = "GridNetBlock",
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
        self.hid_chan = hid_chan
        self.block_type = block_type
        self.rnn_1_conf = rnn_1_conf
        self.rnn_2_conf = rnn_2_conf
        self.attention_conf = attention_conf
        self.repeats = repeats
        self.shared = shared

        self.blocks = self.__build_blocks()

    def __build_blocks(self):
        clss = TFGridNet if self.in_chan > 0 else nn.Identity
        if self.shared:
            out = clss(
                in_chan=self.in_chan,
                hid_chan=self.hid_chan,
                block_type=self.block_type,
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
                        hid_chan=self.hid_chan,
                        block_type=self.block_type,
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


def get(identifier):
    if identifier is None:
        return nn.Identity
    elif callable(identifier):
        return identifier
    elif isinstance(identifier, str):
        cls = globals().get(identifier)

        if cls is None:
            raise ValueError("Could not interpret GridNet identifier: " + str(identifier))
        return cls
    else:
        raise ValueError("Could not interpret GridNet identifier: " + str(identifier))
