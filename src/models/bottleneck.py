import inspect
import torch.nn as nn

from .layers import normalizations


class AudioBottleneck(nn.Module):
    def __init__(self, in_chan: int, audio_bn_chan: int):
        super(AudioBottleneck, self).__init__()
        self.in_chan = in_chan
        self.audio_bn_chan = audio_bn_chan

        self.audio_bottleneck = nn.Conv1d(self.in_chan, self.audio_bn_chan, 1, 1)

    def forward(self, x):
        return self.audio_bottleneck(x)

    def get_config(self):
        encoder_args = {}

        for k, v in (self.__dict__).items():
            if not k.startswith("_") and k != "training":
                if not inspect.ismethod(v):
                    encoder_args[k] = v

        return encoder_args


class VideoBottleneck(nn.Module):
    def __init__(self, in_chan: int, video_bn_chan: int):
        super(VideoBottleneck, self).__init__()
        self.in_chan = in_chan
        self.video_bn_chan = video_bn_chan

        self.video_bottleneck = nn.Conv1d(self.in_chan, self.video_bn_chan, kernel_size=3, padding=1)

    def forward(self, x):
        return self.video_bottleneck(x)

    def get_config(self):
        encoder_args = {}

        for k, v in (self.__dict__).items():
            if not k.startswith("_") and k != "training":
                if not inspect.ismethod(v):
                    encoder_args[k] = v

        return encoder_args
