import torch
import inspect
import torch.nn as nn
import torch.nn.functional as F

from .layers import ConvNormAct


class BaseMaskGenerator(nn.Module):
    def get_config(self):
        encoder_args = {}

        for k, v in (self.__dict__).items():
            if not k.startswith("_") and k != "training":
                if not inspect.ismethod(v):
                    encoder_args[k] = v

        return encoder_args


class MaskGenerator(BaseMaskGenerator):
    def __init__(
        self,
        n_src: int,
        audio_emb_dim: int,
        bottleneck_chan: int,
        kernel_size: int = 1,
        mask_act: str = "ReLU",
        is2d: bool = False,
        output_gate=False,
        oneshot_gate=False,
        *args,
        **kwargs,
    ):
        super(MaskGenerator, self).__init__()
        self.n_src = n_src
        self.in_chan = audio_emb_dim
        self.bottleneck_chan = bottleneck_chan
        self.kernel_size = kernel_size
        self.mask_act = mask_act
        self.is2d = is2d
        self.output_gate = output_gate
        self.oneshot_gate = oneshot_gate

        assert not (self.output_gate and self.oneshot_gate)

        mask_output_chan = self.n_src * self.in_chan * 2 if self.oneshot_gate else self.n_src * self.in_chan

        self.mask_generator = nn.Sequential(
            nn.PReLU(),
            ConvNormAct(
                self.bottleneck_chan,
                mask_output_chan,
                self.kernel_size,
                act_type=self.mask_act,
                is2d=self.is2d,
            ),
        )
        if self.output_gate:
            self.output = ConvNormAct(mask_output_chan, mask_output_chan, 1, act_type="Tanh", is2d=self.is2d)
            self.gate = ConvNormAct(mask_output_chan, mask_output_chan, 1, act_type="Sigmoid", is2d=self.is2d)

    def __apply_masks(self, masks: torch.Tensor, audio_mixture_embedding: torch.Tensor):
        separated_audio_embedding = masks * audio_mixture_embedding.unsqueeze(1)

        return separated_audio_embedding

    def forward(self, refined_features: torch.Tensor, audio_mixture_embedding: torch.Tensor):
        batch_size = refined_features.size(0)
        dims = refined_features.shape[-(len(refined_features.shape) // 2) :]

        if not self.oneshot_gate:
            masks = self.mask_generator(refined_features).view(batch_size * self.n_src, self.in_chan, *dims)
            if self.output_gate:
                masks = self.output(masks) * self.gate(masks)
        else:
            masks = self.mask_generator(refined_features).view(batch_size * self.n_src, 2, self.in_chan, *dims)
            masks = F.tanh(masks[:, 0]) * F.sigmoid(masks[:, 1])

        masks = masks.view(batch_size, self.n_src, self.in_chan, *dims)
        separated_audio_embedding = self.__apply_masks(masks, audio_mixture_embedding)

        return separated_audio_embedding


class DW_MaskGenerator(BaseMaskGenerator):
    def __init__(
        self,
        n_src: int,
        audio_emb_dim: int,
        bottleneck_chan: int,
        kernel_size: int = 1,
        mask_act: str = "ReLU",
        is2d: bool = False,
        output_gate=False,
        oneshot_gate=False,
        *args,
        **kwargs,
    ):
        super(DW_MaskGenerator, self).__init__()
        self.n_src = n_src
        self.in_chan = audio_emb_dim
        self.bottleneck_chan = bottleneck_chan
        self.kernel_size = kernel_size
        self.mask_act = mask_act
        self.is2d = is2d
        self.output_gate = output_gate
        self.oneshot_gate = oneshot_gate

        assert not (self.output_gate and self.oneshot_gate)

        mask_output_chan = self.n_src * self.in_chan * 2 if self.oneshot_gate else self.n_src * self.in_chan

        self.mask_generator = nn.Sequential(
            nn.PReLU(),
            ConvNormAct(
                self.bottleneck_chan,
                mask_output_chan,
                self.kernel_size,
                act_type=self.mask_act,
                is2d=self.is2d,
            ),
        )
        if self.output_gate:
            self.output = ConvNormAct(mask_output_chan, mask_output_chan, 1, groups=mask_output_chan, act_type="Tanh", is2d=self.is2d)
            self.gate = ConvNormAct(mask_output_chan, mask_output_chan, 1, groups=mask_output_chan, act_type="Sigmoid", is2d=self.is2d)

    def __apply_masks(self, masks: torch.Tensor, audio_mixture_embedding: torch.Tensor):
        separated_audio_embedding = masks * audio_mixture_embedding.unsqueeze(1)

        return separated_audio_embedding

    def forward(self, refined_features: torch.Tensor, audio_mixture_embedding: torch.Tensor):
        batch_size = refined_features.size(0)
        dims = refined_features.shape[-(len(refined_features.shape) // 2) :]

        if not self.oneshot_gate:
            masks = self.mask_generator(refined_features).view(batch_size * self.n_src, self.in_chan, *dims)
            if self.output_gate:
                masks = self.output(masks) * self.gate(masks)
        else:
            masks = self.mask_generator(refined_features).view(batch_size * self.n_src, 2, self.in_chan, *dims)
            masks = F.tanh(masks[:, 0]) * F.sigmoid(masks[:, 1])

        masks = masks.view(batch_size, self.n_src, self.in_chan, *dims)
        separated_audio_embedding = self.__apply_masks(masks, audio_mixture_embedding)

        return separated_audio_embedding


class RI_MaskGenerator(BaseMaskGenerator):
    def __init__(
        self,
        n_src: int,
        audio_emb_dim: int,
        bottleneck_chan: int,
        kernel_size: int = 1,
        mask_act: str = "ReLU",
        is2d: bool = False,
        output_gate=False,
        direct=False,
        *args,
        **kwargs,
    ):
        super(RI_MaskGenerator, self).__init__()
        self.n_src = n_src
        self.in_chan = audio_emb_dim
        self.bottleneck_chan = bottleneck_chan
        self.kernel_size = kernel_size
        self.mask_act = mask_act
        self.is2d = is2d
        self.output_gate = output_gate
        self.direct = direct

        if not self.direct:
            mask_output_chan = self.n_src * self.in_chan

            self.mask_generator = nn.Sequential(
                nn.PReLU(),
                ConvNormAct(
                    self.bottleneck_chan,
                    mask_output_chan,
                    self.kernel_size,
                    act_type=self.mask_act,
                    is2d=self.is2d,
                ),
            )

            if self.output_gate:
                self.output = ConvNormAct(mask_output_chan, mask_output_chan, 1, act_type="Tanh", is2d=self.is2d)
                self.gate = ConvNormAct(mask_output_chan, mask_output_chan, 1, act_type="Sigmoid", is2d=self.is2d)

    def __apply_masks(self, masks: torch.Tensor, audio_mixture_embedding: torch.Tensor):
        mask_real = masks[:, :, 0]  # B, n_src, C/2, T, (F)
        mask_imag = masks[:, :, 1]  # B, n_src, C/2, T, (F)
        emb_real = audio_mixture_embedding[:, 0].unsqueeze(1)  # B, 1, C/2, T, (F)
        emb_imag = audio_mixture_embedding[:, 1].unsqueeze(1)  # B, 1, C/2, T, (F)

        est_spec_real = emb_real * mask_real - emb_imag * mask_imag  # B, n_src, C/2, T, (F)
        est_spec_imag = emb_real * mask_imag + emb_imag * mask_real  # B, n_src, C/2, T, (F)

        separated_audio_embedding = torch.cat([est_spec_real, est_spec_imag], 2)  # B, n_src, C, T, (F)

        return separated_audio_embedding

    def forward(self, refined_features: torch.Tensor, audio_mixture_embedding: torch.Tensor):
        if self.direct:
            return refined_features
        else:
            batch_size = refined_features.size(0)
            dims = refined_features.shape[-(len(refined_features.shape) // 2) :]

            masks = self.mask_generator(refined_features).view(batch_size * self.n_src, self.in_chan, *dims)
            if self.output_gate:
                masks = self.output(masks) * self.gate(masks)

            masks = masks.view(batch_size, self.n_src, 2, self.in_chan // 2, *dims)
            audio_mixture_embedding = audio_mixture_embedding.view(batch_size, 2, self.in_chan // 2, *dims)
            separated_audio_embedding = self.__apply_masks(masks, audio_mixture_embedding)

            return separated_audio_embedding


def get(identifier):
    if identifier is None:
        return nn.Identity
    elif callable(identifier):
        return identifier
    elif isinstance(identifier, str):
        cls = globals().get(identifier)
        if cls is None:
            raise ValueError("Could not interpret normalization identifier: " + str(identifier))
        return cls
    else:
        raise ValueError("Could not interpret normalization identifier: " + str(identifier))
