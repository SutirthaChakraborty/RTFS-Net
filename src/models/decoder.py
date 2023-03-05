import torch
import inspect
import torch.nn as nn


class BaseDecoder(nn.Module):
    def pad_to_input_length(self, separated_audio, input_frames):
        output_frames = separated_audio.shape[-1]
        return nn.functional.pad(separated_audio, [0, input_frames - output_frames])

    def reconstruct_to_original_dimensions(self, separated_audio, input_shape):
        return separated_audio.squeeze(0) if len(input_shape) == 1 else separated_audio

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def get_config(self):
        raise NotImplementedError


class ConvolutionalDecoder(BaseDecoder):
    def __init__(
        self,
        in_chan: int,
        n_src: int,
        kernel_size: int,
        stride: int,
        bias=False,
        *args,
        **kwargs,
    ):
        super(ConvolutionalDecoder, self).__init__()

        self.in_chan = in_chan
        self.n_src = n_src
        self.kernel_size = kernel_size
        self.stride = stride
        self.bias = bias

        self.padding = (self.kernel_size - 1) // 2
        self.output_padding = ((self.kernel_size - 1) // 2) - 1

        self.decoder = nn.ConvTranspose1d(
            in_channels=self.in_chan,
            out_channels=self.n_src,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            bias=self.bias,
        )

        torch.nn.init.xavier_uniform_(self.decoder.weight)

    def forward(self, separated_audio_embedding, input_shape):
        separated_audio = self.decoder(separated_audio_embedding)
        separated_audio = self.pad_to_input_length(separated_audio, input_shape[-1])
        separated_audio = self.reconstruct_to_original_dimensions(separated_audio, input_shape)

        return separated_audio

    def get_config(self):
        decoder_args = {}

        for k, v in (self.__dict__).items():
            if not k.startswith("_") and k != "training":
                if not inspect.ismethod(v):
                    decoder_args[k] = v

        return decoder_args


class STFTDecoder(BaseDecoder):
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
    ):
        super(STFTDecoder, self).__init__()

        self.win = win
        self.hop_length = hop_length
        self.out_chan = out_chan
        self.kernel_size = kernel_size
        self.stride = stride
        self.bias = bias
        self.act_type = act_type
        self.norm_type = norm_type

        self.conv = nn.ConvTranspose2d(
            in_channels=out_chan,
            out_channels=2 * self.n_src,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=(self.kernel_size - 1) // 2,
            act_type=self.act_type,
            norm_type=self.norm_type,
            xavier_init=True,
            bias=self.bias,
        )
        self.window = torch.hann_window(self.win)

    def forward(self, x: torch.Tensor):

        output = torch.istft(
            x.view(batch_size * nch * self.num_spks, self.out_chan, -1),
            n_fft=self.win,
            hop_length=self.stride,
            window=torch.hann_window(self.win).to(x.device).type(x.type()),
            length=nsample,
        )

        spec = torch.stack([spec.real, spec.imag], 1).transpose(2, 3).contiguous()  # B, 2, T, F
        spec_feature_map = self.conv(spec)  # B, C, T, F

        return spec_feature_map

    def get_config(self):
        encoder_args = {}

        for k, v in (self.__dict__).items():
            if not k.startswith("_") and k != "training":
                if not inspect.ismethod(v):
                    encoder_args[k] = v

        return encoder_args


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
