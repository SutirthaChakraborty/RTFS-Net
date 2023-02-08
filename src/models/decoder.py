import torch
import inspect
import torch.nn as nn


class BaseDecoder(nn.Module):
    def __pad_to_input_length(self, separated_audio, input_frames):
        output_frames = separated_audio.shape[-1]
        return nn.functional.pad(separated_audio, [0, input_frames - output_frames])

    def __reconstruct_to_original_dimensions(self, separated_audio, input_shape):
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

        self.padding = self.kernel_size // 2
        self.output_padding = (self.kernel_size // 2) - 1

        self.decoder = nn.ConvTranspose1d(
            in_channels=self.in_chan,
            out_channels=self.n_src,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
        )

        torch.nn.init.xavier_uniform_(self.decoder.weight)

    def forward(self, separated_audio_embedding, input_shape):
        separated_audio = self.decoder(separated_audio_embedding)
        separated_audio = self.__pad_to_input_length(separated_audio, input_shape[-1])
        separated_audio = self.__reconstruct_to_original_dimensions(separated_audio, input_shape)

        return separated_audio

    def get_config(self):
        decoder_args = {}

        for k, v in (self.__dict__).items():
            if not k.startswith("_") and k != "training":
                if not inspect.ismethod(v):
                    decoder_args[k] = v

        return decoder_args
