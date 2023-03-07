import torch
import torch.nn as nn
import torch.nn.functional as F


from ..layers import ConvNormAct, ConvNormAct2D


class FusionBasemodule(nn.Module):
    def __init__(self, ain_chan: int, vin_chan: int, is2d: bool):
        super(FusionBasemodule, self).__init__()
        self.ain_chan = ain_chan
        self.vin_chan = vin_chan
        self.is2d = is2d

    def forward(self, audio, video):
        raise NotImplementedError


class ConcatFusion(FusionBasemodule):
    def __init__(self, ain_chan: int, vin_chan: int, is2d: bool = False):
        super(ConcatFusion, self).__init__(ain_chan, vin_chan, is2d)
        conv = ConvNormAct2D if self.is2d else ConvNormAct

        self.audio_conv = conv(self.ain_chan + self.vin_chan, self.ain_chan, 1, norm_type="gLN")
        self.video_conv = conv(self.ain_chan + self.vin_chan, self.vin_chan, 1, norm_type="gLN")

    def forward(self, audio: torch.Tensor, video: torch.Tensor):

        video_interp = F.interpolate(video, size=audio.shape[-(len(audio.shape) // 2) :], mode="nearest")
        audio_video_concat = torch.cat([audio, video_interp], dim=1)
        audio_fused = self.audio_conv(audio_video_concat)

        audio_interp = F.interpolate(audio, size=video.shape[-(len(video.shape) // 2) :], mode="nearest")
        video_audio_concat = torch.cat([audio_interp, video], dim=1)
        video_fused = self.video_conv(video_audio_concat)

        return audio_fused, video_fused


class SumFusion(FusionBasemodule):
    def __init__(self, ain_chan: int, vin_chan: int, is2d: bool = False):
        super(SumFusion, self).__init__(ain_chan, vin_chan, is2d)
        conv = ConvNormAct2D if self.is2d else ConvNormAct

        self.video_conv = conv(self.vin_chan, self.ain_chan, 1, norm_type="gLN")
        self.audio_conv = conv(self.ain_chan, self.vin_chan, 1, norm_type="gLN")

    def forward(self, audio: torch.Tensor, video: torch.Tensor):

        audio_interp = F.interpolate(audio, size=video.shape[-(len(video.shape) // 2) :], mode="nearest")
        video_fused = self.audio_conv(audio_interp) + video

        video_interp = F.interpolate(video, size=audio.shape[-(len(audio.shape) // 2) :], mode="nearest")
        audio_fused = self.video_conv(video_interp) + audio

        return audio_fused, video_fused


class MultiModalFusion(nn.Module):
    def __init__(
        self,
        audio_bn_chan: int,
        video_bn_chan: int,
        fusion_repeats: int = 3,
        audio_repeats: int = 3,
        fusion_type: str = "ConcatFusion",
        fusion_shared: bool = False,
        is2d: bool = False,
    ):
        super(MultiModalFusion, self).__init__()
        self.audio_bn_chan = audio_bn_chan
        self.video_bn_chan = video_bn_chan
        self.fusion_repeats = fusion_repeats
        self.audio_repeats = audio_repeats
        self.fusion_type = fusion_type
        self.fusion_shared = fusion_shared
        self.is2d = is2d

        self.fusion_module = self.__build_fusion_module()

    def __build_fusion_module(self):
        fusion_class = globals().get(self.fusion_type)
        if self.fusion_shared:
            out = fusion_class(self.audio_bn_chan, self.video_bn_chan, self.is2d)
        else:
            out = nn.ModuleList()
            for _ in range(self.fusion_repeats):
                out.append(fusion_class(self.audio_bn_chan, self.video_bn_chan, self.is2d))

        return out

    def get_fusion_block(self, i: int):
        if self.fusion_shared:
            return self.fusion_module
        else:
            return self.fusion_module[i]

    def forward(self, audio: torch.Tensor, video: torch.Tensor):
        audio_residual = audio
        video_residual = video

        for i in range(self.fusion_repeats):
            if i == 0:
                audio_fused, video_fused = self.get_fusion_block(i)(audio, video)
            else:
                audio_fused, video_fused = self.get_fusion_block(i)(audio_fused + audio_residual, video_fused + video_residual)

        return audio_fused
