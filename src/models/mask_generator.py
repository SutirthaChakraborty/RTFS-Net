import torch
import inspect
import torch.nn as nn

from .layers import ConvNormAct, ConvNormAct2D


class MaskGenerator(nn.Module):
    def __init__(self, n_src: int, audio_emb_dim: int, bottleneck_chan: int, mask_act: str, is2d: bool = False):
        super(MaskGenerator, self).__init__()
        self.n_src = n_src
        self.in_chan = audio_emb_dim
        self.bottleneck_chan = bottleneck_chan
        self.mask_act = mask_act
        self.is2d = is2d

        conv = ConvNormAct2D if self.is2d else ConvNormAct

        self.mask_generator = nn.Sequential(
            nn.PReLU(),
            conv(self.bottleneck_chan, self.n_src * self.in_chan, 1, act_type=self.mask_act),
        )

    def forward(self, x: torch.Tensor):
        shape = x.shape

        out = self.mask_generator(x)
        out = out.view(shape[0], self.n_src, self.in_chan, *shape[-(len(shape) // 2) :])

        return out

    def get_config(self):
        encoder_args = {}

        for k, v in (self.__dict__).items():
            if not k.startswith("_") and k != "training":
                if not inspect.ismethod(v):
                    encoder_args[k] = v

        return encoder_args
