import math
import torch
import inspect
import torch.nn as nn
import torch.nn.functional as F

from . import normalizations
from .layers import ConvNormAct
from .utils import get_bandwidths


class BaseEncoder(nn.Module):
    def __init__(self, out_chan: int, kernel_size: int, upsampling_depth: int):
        super(BaseEncoder, self).__init__()
        self.out_chan = out_chan
        self.kernel_size = kernel_size
        self.upsampling_depth = upsampling_depth

        # Appropriate padding is needed for arbitrary lengths
        self.lcm_1 = abs(self.out_chan // 2 * 2**self.upsampling_depth) // math.gcd(self.kernel_size // 2, 2**self.upsampling_depth)
        self.lcm_2 = abs(self.kernel_size // 2 * 2**self.upsampling_depth) // math.gcd(self.kernel_size // 2, 2**self.upsampling_depth)

    def unsqueeze_to_3D(self, x: torch.Tensor):
        if x.ndim == 1:
            return x.reshape(1, 1, -1)
        elif x.ndim == 2:
            return x.unsqueeze(1)
        else:
            return x

    def unsqueeze_to_2D(self, x: torch.Tensor):
        if x.ndim == 1:
            return x.reshape(1, -1)
        elif len(s := x.shape) == 3:
            assert s[1] == 1
            return x.reshape(s[0], -1)
        else:
            return x

    def pad(self, x: torch.Tensor, lcm: int):
        values_to_pad = int(x.shape[-1]) % lcm
        if values_to_pad:
            appropriate_shape = x.shape
            padding = torch.zeros(
                list(appropriate_shape[:-1]) + [lcm - values_to_pad],
                dtype=x.dtype,
                device=x.device,
            )
            padded_x = torch.cat([x, padding], dim=-1)
            return padded_x
        else:
            return x

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def get_config(self):
        encoder_args = {}

        for k, v in (self.__dict__).items():
            if not k.startswith("_") and k != "training":
                if not inspect.ismethod(v):
                    encoder_args[k] = v

        return encoder_args


class ConvolutionalEncoder(BaseEncoder):
    def __init__(
        self,
        in_chan: int,
        out_chan: int,
        kernel_size: int,
        stride: int,
        act_type: str = None,
        norm_type: str = "gLN",
        bias: bool = False,
        upsampling_depth: int = 4,
        layers: int = 1,
        *args,
        **kwargs,
    ):
        super(ConvolutionalEncoder, self).__init__(out_chan, kernel_size, upsampling_depth)

        self.in_chan = in_chan
        self.out_chan = out_chan
        self.kernel_size = kernel_size
        self.stride = stride
        self.bias = bias
        self.act_type = act_type
        self.norm_type = norm_type
        self.layers = layers

        self.encoder = nn.ModuleList()

        for i in range(layers):
            dilation = i + 1
            kernel_size = self.kernel_size * dilation
            self.encoder.append(
                ConvNormAct(
                    in_chan=self.in_chan,
                    out_chan=self.out_chan,
                    kernel_size=kernel_size,
                    stride=self.stride,
                    dilation=dilation,
                    padding=dilation * (kernel_size - 1) // 2,
                    norm_type=self.norm_type,
                    act_type=self.act_type,
                    xavier_init=True,
                    bias=self.bias,
                )
            )

    def forward(self, x: torch.Tensor):
        x = self.unsqueeze_to_3D(x)

        padded_x = self.pad(x, self.lcm_1)
        padded_x = self.pad(padded_x, self.lcm_2)
        feature_maps = []
        for i in range(self.layers):
            feature_maps.append(self.encoder[i](padded_x))

        feature_map = torch.stack(feature_maps).sum(dim=0)

        return feature_map, False


class STFTEncoder(BaseEncoder):
    def __init__(
        self,
        win: int,
        hop_length: int,
        out_chan: int,
        kernel_size: int = 3,
        stride: int = 1,
        act_type: str = None,
        norm_type: str = "gLN",
        bias: bool = False,
        *args,
        **kwargs,
    ):
        super(STFTEncoder, self).__init__(out_chan, 0, 0)

        self.win = win
        self.hop_length = hop_length
        self.out_chan = out_chan
        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1) // 2
        self.stride = stride
        self.bias = bias
        self.act_type = act_type
        self.norm_type = norm_type

        self.conv = ConvNormAct(
            in_chan=2,
            out_chan=self.out_chan,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            act_type=self.act_type,
            norm_type=self.norm_type,
            xavier_init=True,
            bias=self.bias,
            is2d=True,
        )

        self.register_buffer("window", torch.hann_window(self.win), False)

    def forward(self, x: torch.Tensor):
        x = self.unsqueeze_to_2D(x)

        spec = torch.stft(
            x,
            n_fft=self.win,
            hop_length=self.hop_length,
            window=self.window,
            return_complex=True,
        )

        spec = torch.stack([spec.real, spec.imag], 1).transpose(2, 3).contiguous()  # B, 2, T, F
        spec_feature_map = self.conv(spec)  # B, C, T, F

        return spec_feature_map, False


class BSRNNEncoder(BaseEncoder):
    def __init__(
        self,
        win: int,
        hop_length: int,
        out_chan: int,
        norm_type: str = "gLN",
        sample_rate: int = 16000,
        *args,
        **kwargs,
    ):
        super(BSRNNEncoder, self).__init__(out_chan, 0, 0)

        self.win = win
        self.hop_length = hop_length
        self.out_chan = out_chan
        self.norm_type = norm_type
        self.sample_rate = sample_rate
        self.eps = torch.finfo(torch.float32).eps

        self.register_buffer("window", torch.hann_window(self.win), False)

        self.band_width = get_bandwidths(self.win, self.sample_rate)

        print(self.band_width)

        self.BN = nn.ModuleList([])
        for i in range(len(self.band_width)):
            in_chan = self.band_width[i] * 2
            self.BN.append(
                nn.Sequential(normalizations.get(self.norm_type)(in_chan), ConvNormAct(in_chan, self.out_chan, 1, xavier_init=True))
            )

    def forward(self, x: torch.Tensor):
        x = self.unsqueeze_to_2D(x)
        batch_size = x.shape[0]

        spec = torch.stft(
            x,
            n_fft=self.win,
            hop_length=self.hop_length,
            window=self.window,
            return_complex=True,
        )

        # concat real and imag, split to subbands
        spec_RI = torch.stack([spec.real, spec.imag], 1)  # B, 2, F, T
        subband_spec = []
        spec_by_band_width = []
        band_idx = 0
        for i in range(len(self.band_width)):
            subband_spec.append(spec_RI[:, :, band_idx : band_idx + self.band_width[i]].contiguous())
            spec_by_band_width.append(spec[:, band_idx : band_idx + self.band_width[i]])  # B, BW, T
            band_idx += self.band_width[i]

        # normalization and bottleneck
        spec_feature_map = []
        for i in range(len(self.band_width)):
            spec_feature_map.append(self.BN[i](subband_spec[i].view(batch_size, self.band_width[i] * 2, -1)))
        spec_feature_map = torch.stack(spec_feature_map, 1)  # B, nband, N, T
        spec_feature_map = spec_feature_map.permute(0, 2, 3, 1).contiguous()

        return spec_feature_map, spec_by_band_width


class RI_STFTEncoder(BaseEncoder):
    def __init__(
        self,
        win: int,
        hop_length: int,
        out_chan: int,
        kernel_size: int = 3,
        stride: int = 1,
        act_type: str = None,
        norm_type: str = "gLN",
        bias: bool = False,
        *args,
        **kwargs,
    ):
        super(RI_STFTEncoder, self).__init__(out_chan, 0, 0)

        self.win = win
        self.hop_length = hop_length
        self.out_chan = out_chan * 2
        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1) // 2
        self.stride = stride
        self.bias = bias
        self.act_type = act_type
        self.norm_type = norm_type

        self.conv = ConvNormAct(
            in_chan=2,
            out_chan=self.out_chan,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            act_type=self.act_type,
            norm_type=self.norm_type,
            xavier_init=True,
            bias=self.bias,
            is2d=True,
        )

        self.register_buffer("window", torch.hann_window(self.win), False)

    def forward(self, x: torch.Tensor):
        x = self.unsqueeze_to_2D(x)

        spec = torch.stft(
            x,
            n_fft=self.win,
            hop_length=self.hop_length,
            window=self.window,
            return_complex=True,
        )

        spec = torch.stack([spec.real, spec.imag], 1).transpose(2, 3).contiguous()  # B, 2, T, F
        spec_feature_map = self.conv(spec)  # B, C, T, F

        return spec_feature_map, False


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
