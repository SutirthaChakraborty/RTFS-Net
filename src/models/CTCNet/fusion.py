import torch
import torch.nn as nn
import torch.nn.functional as F


from ..layers import ConvNormAct, FRCNN


class FusionBasemodule(nn.Module):
    def __init__(
        self,
        ain_chan: int = 128,
        vin_chan: int = 128,
    ):
        super(FusionBasemodule, self).__init__()
        self.ain_chan = ain_chan
        self.vin_chan = vin_chan

    def forward(self, audio, video):
        raise NotImplementedError


class ConcatFusion(FusionBasemodule):
    def __init__(
        self,
        ain_chan: int = 128,
        vin_chan: int = 128,
    ):
        super(ConcatFusion, self).__init__(ain_chan, vin_chan)
        self.audio_conv = ConvNormAct(ain_chan + vin_chan, ain_chan, 1, 1, norm_type="gLN")
        self.video_conv = ConvNormAct(ain_chan + vin_chan, vin_chan, 1, 1, norm_type="gLN")

    def forward(self, audio, video):
        video_interp = F.interpolate(video, size=audio.shape[-1], mode="nearest")
        audio_video_concat = torch.cat([audio, video_interp], dim=1)
        audio_fused = self.audio_conv(audio_video_concat)

        audio_interp = F.interpolate(audio, size=video.shape[-1], mode="nearest")
        video_audio_concat = torch.cat([audio_interp, video], dim=1)
        video_fused = self.video_conv(video_audio_concat)

        return audio_fused, video_fused


class SumFusion(FusionBasemodule):
    def __init__(self, ain_chan: int = 128, vin_chan: int = 128):
        super(SumFusion, self).__init__(ain_chan, vin_chan)
        self.audio_conv = ConvNormAct(vin_chan, ain_chan, 1, 1, norm_type="gLN")
        self.video_conv = ConvNormAct(ain_chan, vin_chan, 1, 1, norm_type="gLN")

    def forward(self, audio, video):
        audio_interp = F.interpolate(audio, size=video.shape[-1], mode="nearest")
        video_fused = self.video_conv(audio_interp) + video

        video_interp = F.interpolate(video, size=audio.shape[-1], mode="nearest")
        audio_fused = self.audio_conv(video_interp) + audio

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
    ):
        super(MultiModalFusion, self).__init__()
        self.audio_bn_chan = audio_bn_chan
        self.video_bn_chan = video_bn_chan
        self.fusion_repeats = fusion_repeats
        self.audio_repeats = audio_repeats
        self.fusion_type = fusion_type
        self.fusion_shared = fusion_shared

        fusion_class = globals().get(self.fusion_type)

        if self.fusion_shared:
            self.fusion_module = fusion_class(self.audio_bn_chan, self.video_bn_chan)
        else:
            self.fusion_module = nn.ModuleList([fusion_class(self.audio_bn_chan, self.video_bn_chan) for _ in range(self.fusion_repeats)])

    def __get_crossmodal_fusion(self, i):
        if self.fusion_shared:
            return self.fusion_module
        else:
            return self.fusion_module[i]

    def forward(self, audio, video, audio_frcnn: FRCNN, video_frcnn: FRCNN):
        audio_residual = audio
        video_residual = video

        for i in range(self.fusion_repeats):
            if i == 0:
                audio = audio_frcnn.get_frcnn_block(i)(audio)
                video = video_frcnn.get_frcnn_block(i)(video)
                audio_fused, video_fused = self.__get_crossmodal_fusion(i)(audio, video)
            else:
                audio_fused = audio_frcnn.get_frcnn_block(i)(audio_frcnn.get_concat_block(i)(audio_fused + audio_residual))
                video_fused = video_frcnn.get_frcnn_block(i)(video_frcnn.get_concat_block(i)(video_fused + video_residual))
                audio_fused, video_fused = self.__get_crossmodal_fusion(i)(audio_fused, video_fused)

        for i in range(self.audio_repeats):
            j = i + self.fusion_repeats
            audio_fused = audio_frcnn.get_frcnn_block(j)(audio_frcnn.get_concat_block(j)(audio_fused + audio_residual))

        return audio_fused
