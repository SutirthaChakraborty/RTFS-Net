import inspect
import torch.nn as nn

from .layers import ConvActNorm


class MaskGenerator(nn.Module):
    def __init__(self, n_src: int, audio_emb_dim: int, bottleneck_chan: int, mask_act: str):
        super(MaskGenerator, self).__init__()
        self.n_src = n_src
        self.in_chan = audio_emb_dim
        self.bottleneck_chan = bottleneck_chan
        self.mask_act = mask_act

        self.mask_generator = nn.Sequential(
            nn.PReLU(),
            ConvActNorm(self.bottleneck_chan, self.n_src * self.in_chan, 1, act_type=self.mask_act),
        )

    def forward(self, x):
        batch_size = x.shape[0]
        return self.mask_generator(x).view(batch_size, self.n_src, self.in_chan, -1)

    def get_config(self):
        encoder_args = {}

        for k, v in (self.__dict__).items():
            if not k.startswith("_") and k != "training":
                if not inspect.ismethod(v):
                    encoder_args[k] = v

        return encoder_args
