import torch
import inspect
import torch.nn as nn


class Convolutional_Decoder(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        output_padding,
        groups,
        bias=False,
    ):
        super(Convolutional_Decoder, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.bias = bias

        self.decoder = nn.ConvTranspose1d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            groups=self.groups,
        )

        torch.nn.init.xavier_uniform_(self.decoder.weight)

    def pad_to_input_length(self, separated_audio, input_frames):
        output_frames = separated_audio.shape[-1]
        return nn.functional.pad(separated_audio, [0, input_frames - output_frames])

    def reconstruct_to_original_dimensions(self, separated_audio, input_shape):
        return separated_audio.squeeze(0) if len(input_shape) == 1 else separated_audio

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
