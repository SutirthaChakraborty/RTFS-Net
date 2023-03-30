import torch
import inspect
import torch.nn as nn

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
    ):
        super(MaskGenerator, self).__init__()
        self.n_src = n_src
        self.in_chan = audio_emb_dim
        self.bottleneck_chan = bottleneck_chan
        self.kernel_size = kernel_size
        self.mask_act = mask_act
        self.is2d = is2d
        self.output_gate = output_gate

        if self.kernel_size > 0:
            self.mask_generator = nn.Sequential(
                nn.PReLU(),
                ConvNormAct(
                    self.bottleneck_chan,
                    self.n_src * self.in_chan,
                    self.kernel_size,
                    act_type=self.mask_act,
                    is2d=self.is2d,
                ),
            )
            if self.output_gate:
                self.output = ConvNormAct(self.in_chan, self.in_chan, 1, act_type="Tanh", is2d=self.is2d)
                self.gate = ConvNormAct(self.in_chan, self.in_chan, 1, act_type="Sigmoid", is2d=self.is2d)
        else:
            self.mask_generator = nn.Identity()

    def __apply_masks(self, masks: torch.Tensor, audio_mixture_embedding: torch.Tensor):
        separated_audio_embedding = masks * audio_mixture_embedding.unsqueeze(1)

        return separated_audio_embedding

    def forward(self, refined_features: torch.Tensor, audio_mixture_embedding: torch.Tensor, context: torch.Tensor):
        shape = refined_features.shape

        if self.kernel_size > 0:
            masks = self.mask_generator(refined_features).view(shape[0] * self.n_src, self.in_chan, *shape[-(len(shape) // 2) :])
            if self.output_gate:
                masks = self.output(masks) * self.gate(masks)

            masks = masks.view(shape[0], self.n_src, self.in_chan, *shape[-(len(shape) // 2) :])
            separated_audio_embedding = self.__apply_masks(masks, audio_mixture_embedding)
        else:
            separated_audio_embedding = refined_features.view(shape[0], self.n_src, self.in_chan, *shape[-(len(shape) // 2) :])

        return separated_audio_embedding


class BSRNNMaskGenerator(BaseMaskGenerator):
    def __init__(
        self,
        nband: int,
        ratio: int,
        band_width: list,
        audio_emb_dim: int,
        bottleneck_chan: int,
        n_src: int,
        *args,
        **kwargs,
    ):
        super(BSRNNMaskGenerator, self).__init__()
        self.nband = nband
        self.ratio = ratio
        self.band_width = band_width
        self.audio_emb_dim = audio_emb_dim
        self.bottleneck_chan = bottleneck_chan
        self.n_src = n_src

        self.mask = nn.ModuleList([])
        for i in range(self.nband):
            self.mask.append(
                nn.Sequential(
                    nn.GroupNorm(1, self.bottleneck_chan, torch.finfo(torch.float32).eps),
                    nn.Conv1d(self.bottleneck_chan, self.bottleneck_chan * 4, 1),
                    nn.Tanh(),
                    nn.Conv1d(self.bottleneck_chan * 4, self.audio_emb_dim * 4, 1),
                    nn.Tanh(),
                    nn.Conv1d(self.audio_emb_dim * 4, self.band_width[i] * self.ratio * 4 * self.n_src, 1),
                )
            )

    def forward(self, sep_output: torch.Tensor, audio_mixture_embedding: torch.Tensor, context: torch.Tensor):
        # context: B, Context, BW, T
        batch_size = sep_output.shape[0]

        sep_output = sep_output.permute(0, 3, 1, 2).contiguous()  # B, nband, N, T

        sep_subband_spec = []
        for i in range(len(self.band_width)):
            this_output = self.mask[i](sep_output[:, i]).view(batch_size, 2, 2, self.n_src, self.ratio, self.band_width[i], -1)
            this_mask = this_output[:, 0] * torch.sigmoid(this_output[:, 1])  # B*nch, 2, n_src, K, BW, T
            mask_real = this_mask[:, 0]  # B*nch, n_src, K, BW, T
            mask_imag = this_mask[:, 1]  # B*nch, n_src, K, BW, T
            est_spec_real = (context[i].real.unsqueeze(1) * mask_real).mean(1) - (context[i].imag.unsqueeze(1) * mask_imag).mean(1)
            est_spec_imag = (context[i].real.unsqueeze(1) * mask_imag).mean(1) + (context[i].imag.unsqueeze(1) * mask_real).mean(1)
            sep_subband_spec.append(torch.complex(est_spec_real, est_spec_imag))
        est_spec = torch.cat(sep_subband_spec, 1)  # B*nch, n_src, F, T

        return est_spec


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
