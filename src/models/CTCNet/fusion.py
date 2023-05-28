import torch
import torch.nn as nn
import torch.nn.functional as F


from ..layers import ConvNormAct


class FusionBasemodule(nn.Module):
    def __init__(self, ain_chan: int, vin_chan: int, is2d: bool, video_fusion: bool, nstack: bool):
        super(FusionBasemodule, self).__init__()
        self.ain_chan = ain_chan
        self.vin_chan = vin_chan
        self.is2d = is2d
        self.video_fusion = video_fusion
        self.nstack = nstack

    def forward(self, audio, video):
        raise NotImplementedError

    def wrangle_dims(self, audio: torch.Tensor, video: torch.Tensor):
        T1 = audio.shape[-(len(audio.shape) // 2) :]
        T2 = video.shape[-(len(video.shape) // 2) :]

        self.x = len(T1) > len(T2)
        self.y = len(T2) > len(T1)

        if self.x:
            video = torch.stack([video] * T1[-1], -1) if self.nstack else video.unsqueeze(-1)
        if self.y:
            audio = torch.stack([audio] * T2[-1], -1) if self.nstack else audio.unsqueeze(-1)

        return audio, video

    def unwrangle_dims(self, audio: torch.Tensor, video: torch.Tensor):
        video = video.squeeze(-1) if self.x else video
        audio = audio.squeeze(-1) if self.y else audio

        return audio, video


class ConcatFusion(FusionBasemodule):
    def __init__(self, ain_chan: int, vin_chan: int, is2d: bool = False, video_fusion: bool = True, nstack: bool = False):
        super(ConcatFusion, self).__init__(ain_chan, vin_chan, is2d, video_fusion, nstack)

        self.audio_conv = ConvNormAct(self.ain_chan + self.vin_chan, self.ain_chan, 1, norm_type="gLN", is2d=self.is2d)
        if video_fusion:
            self.video_conv = ConvNormAct(self.ain_chan + self.vin_chan, self.vin_chan, 1, norm_type="gLN", is2d=self.is2d)

    def forward(self, audio: torch.Tensor, video: torch.Tensor):
        audio, video = self.wrangle_dims(audio, video)

        video_interp = F.interpolate(video, size=audio.shape[-(len(audio.shape) // 2) :], mode="nearest")
        audio_video_concat = torch.cat([audio, video_interp], dim=1)
        audio_fused = self.audio_conv(audio_video_concat)

        if self.video_fusion:
            audio_interp = F.interpolate(audio, size=video.shape[-(len(video.shape) // 2) :], mode="nearest")
            video_audio_concat = torch.cat([audio_interp, video], dim=1)
            video_fused = self.video_conv(video_audio_concat)
        else:
            video_fused = video

        audio_fused, video_fused = self.unwrangle_dims(audio_fused, video_fused)

        return audio_fused, video_fused


class SumFusion(FusionBasemodule):
    def __init__(self, ain_chan: int, vin_chan: int, is2d: bool = False, video_fusion: bool = True, nstack: bool = False):
        super(SumFusion, self).__init__(ain_chan, vin_chan, is2d, video_fusion, nstack)

        if video_fusion:
            self.audio_conv = ConvNormAct(self.ain_chan, self.vin_chan, 1, norm_type="gLN", is2d=self.is2d)
        self.video_conv = ConvNormAct(self.vin_chan, self.ain_chan, 1, norm_type="gLN", is2d=self.is2d)

    def forward(self, audio: torch.Tensor, video: torch.Tensor):
        audio, video = self.wrangle_dims(audio, video)

        if self.video_fusion:
            audio_interp = F.interpolate(audio, size=video.shape[-(len(video.shape) // 2) :], mode="nearest")
            video_fused = self.audio_conv(audio_interp) + video
        else:
            video_fused = video

        video_interp = F.interpolate(video, size=audio.shape[-(len(audio.shape) // 2) :], mode="nearest")
        audio_fused = self.video_conv(video_interp) + audio

        audio_fused, video_fused = self.unwrangle_dims(audio_fused, video_fused)

        return audio_fused, video_fused


class MultiModalFusion(nn.Module):
    def __init__(
        self,
        audio_bn_chan: int,
        video_bn_chan: int,
        fusion_repeats: int = 3,
        fusion_type: str = "ConcatFusion",
        fusion_shared: bool = False,
        is2d: bool = False,
        nstack: bool = False,
    ):
        super(MultiModalFusion, self).__init__()
        self.audio_bn_chan = audio_bn_chan
        self.video_bn_chan = video_bn_chan
        self.fusion_repeats = fusion_repeats
        self.fusion_type = fusion_type
        self.fusion_shared = fusion_shared
        self.is2d = is2d
        self.nstack = nstack

        self.fusion_module = self.__build_fusion_module()

    def __build_fusion_module(self):
        fusion_class = globals().get(self.fusion_type) if self.fusion_repeats > 0 else nn.Identity
        if self.fusion_shared:
            out = fusion_class(self.audio_bn_chan, self.video_bn_chan, self.is2d, self.fusion_repeats > 1, self.nstack)
        else:
            out = nn.ModuleList()
            for i in range(self.fusion_repeats):
                out.append(fusion_class(self.audio_bn_chan, self.video_bn_chan, self.is2d, i != self.fusion_repeats - 1, self.nstack))

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
