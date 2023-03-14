import torch
import inspect
import torch.nn as nn

from .layers import ConvNormAct


class MaskGenerator(nn.Module):
    def __init__(
        self,
        n_src: int,
        audio_emb_dim: int,
        bottleneck_chan: int,
        kernel_size: int = 1,
        mask_act: str = "ReLU",
        is2d: bool = False,
    ):
        super(MaskGenerator, self).__init__()
        self.n_src = n_src
        self.in_chan = audio_emb_dim
        self.bottleneck_chan = bottleneck_chan
        self.kernel_size = kernel_size
        self.mask_act = mask_act
        self.is2d = is2d

        self.mask_generator = (
            nn.Sequential(
                nn.PReLU(),
                ConvNormAct(
                    self.bottleneck_chan,
                    self.n_src * self.in_chan,
                    self.kernel_size,
                    act_type=self.mask_act,
                    is2d=self.is2d,
                ),
            )
            if self.kernel_size > 0
            else nn.Identity()
        )

    def __apply_masks(self, masks: torch.Tensor, audio_mixture_embedding: torch.Tensor):
        shape = audio_mixture_embedding.shape
        separated_audio_embedding = masks * audio_mixture_embedding.unsqueeze(1)
        separated_audio_embedding = separated_audio_embedding.view(shape[0], self.n_src * self.in_chan, *shape[-(len(shape) // 2) :])
        return separated_audio_embedding

    def forward(self, refined_features: torch.Tensor, audio_mixture_embedding: torch.Tensor):
        shape = refined_features.shape

        if self.kernel_size > 0:
            masks = self.mask_generator(refined_features)
            masks = masks.view(shape[0], self.n_src, self.in_chan, *shape[-(len(shape) // 2) :])
            separated_audio_embedding = self.__apply_masks(masks, audio_mixture_embedding)
        else:
            separated_audio_embedding = refined_features

        return separated_audio_embedding

    def get_config(self):
        encoder_args = {}

        for k, v in (self.__dict__).items():
            if not k.startswith("_") and k != "training":
                if not inspect.ismethod(v):
                    encoder_args[k] = v

        return encoder_args
